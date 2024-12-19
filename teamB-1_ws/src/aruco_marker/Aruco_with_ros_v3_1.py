"""
ArUco Marker Detector with Pose Estimation (OpenCV 4.10.0+)
This program detects ArUco markers and calculates their pose (distance and orientation).
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

"""
1. 터틀봇의 아루코마커 x,y 좌표 방향 세팅
2. 카메라 화소 1280*720 추가
"""

# Select the desired ArUco dictionary 
desired_aruco_dictionary = "DICT_5X5_100"

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
}

# Load calibration data
camera_matrix = np.load('/home/july/rokey_prj/week5/AMR-3/teamB-1_ws/img_capture/calibration_matrix_4.npy')
dist_coeffs = np.load('/home/july/rokey_prj/week5/AMR-3/teamB-1_ws/img_capture/distortion_coefficients_4.npy')

marker_length = 0.107  # Marker size in meters

# Marker IDs
ORIGIN_MARKER_ID = 0
TURTLEBOT_MARKER_ID = 5
CONTAINER_MARKER_ID = 4

# 터틀봇 좌표계를 기준으로 변환된 속도 계산 (XY 평면에서 Y축 기준 각속도)
def calculate_velocity_commands(target_tvec, turtlebot_tvec, turtlebot_rvec, linear_gain=0.5, angular_gain=1.0):
    """
    Calculate linear and angular velocity commands based on target's position relative to the robot.
    :param target_tvec: Target marker's 3D position (in camera frame)
    :param turtlebot_tvec: TurtleBot's marker 3D position (in camera frame)
    :param turtlebot_rvec: TurtleBot's marker rotation vector
    :param linear_gain: Gain factor for linear velocity
    :param angular_gain: Gain factor for angular velocity
    :return: Linear velocity (m/s), Angular velocity (rad/s)
    """
    # Convert rotation vector to rotation matrix
    rotation_matrix, _ = cv2.Rodrigues(turtlebot_rvec)

    # Relative position in global frame
    relative_position_global = target_tvec - turtlebot_tvec

    # Transform target position to TurtleBot's coordinate frame
    relative_position_robot = rotation_matrix.T @ relative_position_global

    # Extract components in TurtleBot's coordinate frame
    x_robot, y_robot, z_robot = relative_position_robot

    # Distance to target in 3D
    distance = np.linalg.norm(relative_position_robot)

    # Ensure that linear velocity is only positive (move forward)
    linear_velocity = linear_gain * y_robot  # Forward motion along Y-axis

    # Calculate angular velocity based on Y-axis in XY plane
    angular_velocity = angular_gain * np.arctan2(y_robot, x_robot)  # Rotation toward target

    # Debugging information for relative position
    print(f"Relative Position (Robot Frame): x={x_robot:.2f}, y={y_robot:.2f}, z={z_robot:.2f}")
    print(f"Distance to Target: {distance:.2f}m")
    print(f"Linear Velocity: {linear_velocity:.2f}, Angular Velocity: {angular_velocity:.2f}")

    return linear_velocity, angular_velocity


class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_reached = False
        self.goal_threshold = 0.1  # Threshold for goal distance
        self.cap = cv2.VideoCapture(4)

        if not self.cap.isOpened():
            self.get_logger().error("Unable to access the webcam.")
            rclpy.shutdown()

        # Set camera resolution to 1280x720
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Verify the actual resolution
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if actual_width != 1280 or actual_height != 720:
            self.get_logger().warning(f"Requested resolution 1280x720 not supported. Using {actual_width}x{actual_height}.")

        self.timer = self.create_timer(0.1, self.detect_and_publish)

    def detect_and_publish(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to grab frame.")
            return

        # Detect markers
        this_aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[desired_aruco_dictionary])
        this_aruco_parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(this_aruco_dictionary, this_aruco_parameters)

        corners, ids, _ = detector.detectMarkers(frame)
        if ids is None:
            self.get_logger().info("No markers detected.")
            return

        ids = ids.flatten()
        origin_tvec = None
        turtlebot_tvec = None
        target_tvec = None
        distance_to_target = None  # Ensure the variable is defined

        for i, marker_id in enumerate(ids):
            marker_corners = corners[i].reshape((4, 2))

            # SolvePnP to estimate pose
            success, rvec, tvec = cv2.solvePnP(
                np.array([
                    [-marker_length / 2, marker_length / 2, 0],
                    [marker_length / 2, marker_length / 2, 0],
                    [marker_length / 2, -marker_length / 2, 0],
                    [-marker_length / 2, -marker_length / 2, 0],
                ]),
                marker_corners,
                camera_matrix,
                dist_coeffs
            )

            if success:
                tvec = tvec.ravel()  # Flatten tvec

                if marker_id == ORIGIN_MARKER_ID:
                    origin_tvec = tvec
                elif marker_id == TURTLEBOT_MARKER_ID:
                    turtlebot_tvec = tvec
                elif marker_id == CONTAINER_MARKER_ID:
                    target_tvec = tvec

                # Draw axes
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

        # Calculate and publish velocities if all markers are detected
        if origin_tvec is not None and turtlebot_tvec is not None and target_tvec is not None:
            distance_to_target = np.linalg.norm(target_tvec - turtlebot_tvec)  # Calculate distance
            if not self.goal_reached:
                if distance_to_target > 0.45:
                    linear_vel, angular_vel = calculate_velocity_commands(
                        target_tvec, turtlebot_tvec, rvec
                    )
                    self.publish_cmd_vel(linear_vel, angular_vel)
                else:
                    self.goal_reached = True
                    self.publish_cmd_vel(0.0, 0.0)

        # Display distance to target on frame if calculated
        if distance_to_target is not None:
            distance_text = f"Distance to Target: {distance_to_target:.2f}m"
            cv2.putText(frame, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Show frame
        cv2.imshow("ArUco Marker Detector", frame)
        cv2.waitKey(1)

    def publish_cmd_vel(self, linear_vel, angular_vel):
        twist = Twist()
        twist.linear.x = linear_vel
        twist.angular.z = angular_vel
        self.cmd_vel_publisher.publish(twist)
        self.get_logger().info(f"Published: Linear Vel = {linear_vel:.2f}, Angular Vel = {angular_vel:.2f}")

def main():
    rclpy.init()
    node = ArucoDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
