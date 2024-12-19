#!/usr/bin/env python
"""
ArUco Marker Detector with Pose Estimation (OpenCV 4.10.0+)
This program detects ArUco markers and calculates their pose (distance and orientation).
"""

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

# Select the desired ArUco dictionary
desired_aruco_dictionary = "DICT_5X5_100"

ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
}

# Load calibration data
camera_matrix = np.load('/home/july/rokey_prj/week5/AMR-3/teamB-1_ws/imgs/calibration_matrix_3.npy')
dist_coeffs = np.load('/home/july/rokey_prj/week5/AMR-3/teamB-1_ws/imgs/distortion_coefficients_3.npy')

marker_length = 0.107  # Marker size in meters

# Marker IDs
ORIGIN_MARKER_ID = 0
TURTLEBOT_MARKER_ID = 5
CONTAINER_MARKER_ID = 4

# Velocity calculation
def calculate_velocity_commands(target_tvec, robot_tvec, linear_gain=0.5, angular_gain=1.0):
    relative_position = target_tvec[:2] - robot_tvec[:2]
    distance = np.linalg.norm(relative_position)
    angle_to_target = np.arctan2(relative_position[1], relative_position[0])
    linear_velocity = linear_gain * distance
    angular_velocity = angular_gain * angle_to_target
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
            distance_to_target = np.linalg.norm(target_tvec[:2] - turtlebot_tvec[:2])
            if not self.goal_reached:
                if distance_to_target > self.goal_threshold:
                    linear_vel, angular_vel = calculate_velocity_commands(target_tvec, turtlebot_tvec)
                    self.publish_cmd_vel(linear_vel, angular_vel)
                else:
                    self.goal_reached = True
                    self.publish_cmd_vel(0.0, 0.0)

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
