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
2. 카메라 화소 1280*720추가
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

# 새로운 제어 상수 추가
MAX_LINEAR_VEL = 0.5
MIN_LINEAR_VEL = 0.1
MAX_ANGULAR_VEL = 1.0
DISTANCE_THRESHOLD = 0.5

# cal_vel v.3.7 방향 반대로 가는거 개선
def calculate_velocity_commands(target_tvec, turtlebot_tvec, turtlebot_rvec, target_rvec, linear_gain=0.1, angular_gain=0.2):
    # 회전 행렬 계산
    turtlebot_rotation, _ = cv2.Rodrigues(turtlebot_rvec)
    target_rotation, _ = cv2.Rodrigues(target_rvec)
    
    # 방향 벡터 추출
    turtlebot_forward = turtlebot_rotation[:, 1]  # y축
    target_normal = -target_rotation[:, 2]  # -z축
    
    # 두 벡터의 외적으로 회전 방향 결정
    cross_product = np.cross(turtlebot_forward, target_normal)
    rotation_direction = -np.sign(cross_product[2])  # 부호 반대로 변경
    
    # 각도 계산
    dot_product = np.dot(turtlebot_forward, target_normal)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # 회전 방향 적용
    angle = rotation_direction * angle # -값: 왼쪽회전, +값: 오른쪽회전
    
    # 거리 계산
    distance = np.linalg.norm(target_tvec - turtlebot_tvec)
    
    # 속도 계산
    linear_velocity = linear_gain * distance
    angular_velocity = angular_gain * angle
    
    if abs(angle) < np.radians(5):  # 5도 이내면 거의 정렬된 것으로 간주
        angular_velocity = 0.0
        
    return linear_velocity, angular_velocity

# 마커간 거리계산
def calculate_relative_position(marker1_tvec, marker2_tvec):
    """
    Calculate the relative position (distance and direction) between markers.
    :param marker_tvec: Target marker's tvec
    :param robot_tvec: Robot marker's tvec
    :return: Relative distance, angle in radians
    """
    # x, y만 사용하여 2D 방향 계산
    relative_position = marker1_tvec[:2] - marker2_tvec[:2]  # x, y만 고려
    distance = np.linalg.norm(relative_position)  # 평면 거리 계산
    angle = np.arctan2(relative_position[1], relative_position[0])  # 방향 각도 계산
    print(f"Relative Position: {relative_position}, Distance: {distance}, Angle: {np.degrees(angle):.2f}")
    return distance, angle


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
            # corners 순서 출력
            print(f"\n코드 V3_2 Marker {marker_id} corners:")
            print("Corner 0 (좌상단 예상):", marker_corners[0])
            print("Corner 1 (우상단 예상):", marker_corners[1])
            print("Corner 2 (우하단 예상):", marker_corners[2])
            print("Corner 3 (좌하단 예상):", marker_corners[3])
            
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
                rotation_matrix, _ = cv2.Rodrigues(rvec)
                print(f"\nMarker {marker_id} axes:")
                print("X axis:", rotation_matrix[:, 0])
                print("Y axis:", rotation_matrix[:, 1])
                print("Z axis:", rotation_matrix[:, 2])

                if marker_id == ORIGIN_MARKER_ID:
                    origin_tvec = tvec
                elif marker_id == TURTLEBOT_MARKER_ID:
                    turtlebot_tvec = tvec
                    turtlebot_rvec = rvec
                elif marker_id == CONTAINER_MARKER_ID:
                    target_tvec = tvec
                    target_rvec = rvec

                # Draw axes
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

        # Calculate and publish velocities if all markers are detected
        if origin_tvec is not None and turtlebot_tvec is not None and target_tvec is not None:
            # 변경
            distance_to_target = np.linalg.norm(target_tvec - turtlebot_tvec)  # Calculate distance
            if not self.goal_reached:
                if distance_to_target > 0.43:
                    linear_vel, angular_vel = calculate_velocity_commands(
                        target_tvec, turtlebot_tvec, turtlebot_rvec, target_rvec
                    )
                    self.publish_cmd_vel(linear_vel, angular_vel)
                else:
                    self.goal_reached = True
                    self.publish_cmd_vel(0.0, 0.0)

        # Display distance to target on frame if calculated
        if distance_to_target is not None:
            distance_text = f"Distance to Target: {distance_to_target:.2f}m"
            cv2.putText(frame, distance_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Display each marker's distance (if available)
        if turtlebot_tvec is not None:
            turtlebot_distance = np.linalg.norm(turtlebot_tvec)  # Camera to TurtleBot
            text = f"ID: {TURTLEBOT_MARKER_ID} | Dist: {turtlebot_distance:.2f}m"
            cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        if target_tvec is not None:
            target_distance = np.linalg.norm(target_tvec)  # Camera to Target
            text = f"ID: {CONTAINER_MARKER_ID} | Dist: {target_distance:.2f}m"
            cv2.putText(frame, text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

        # Show frame
        cv2.imshow("ArUco Marker Detector", frame)
        cv2.waitKey(1)


    def publish_cmd_vel(self, linear_vel, angular_vel):
        twist = Twist()
        twist.linear.x = -linear_vel
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
