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
import time

# Select the desired ArUco dictionary
desired_aruco_dictionary = "DICT_5X5_100"

# The different ArUco dictionaries built into the OpenCV library
ARUCO_DICT = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
}

# Load calibration data (캘리브레이션 해상도: 1280x720)
camera_matrix = np.load('/home/july/rokey_prj/week5/AMR-3/teamB-1_ws/imgs/calibration_matrix_3.npy')
dist_coeffs = np.load('/home/july/rokey_prj/week5/AMR-3/teamB-1_ws/imgs/distortion_coefficients_3.npy')


# Marker size in meters
marker_length = 0.107  # 10.7 cm

# 로봇 마커 ID (기준 마커)
ORIGIN_MARKER_ID = 0  # 기준 마커 ID (좌표계 원점)
TURTLEBOT_MARKER_ID = 5 # 터틀봇 마커 ID
CONTAINER_MARKER_ID = 4  # 타겟 마커 ID


# ORIGIN_MARKER 기준으로 터틀봇이 얼마나 움직여야할지 계산.
def calculate_velocity_commands(target_tvec, robot_tvec, linear_gain=0.5, angular_gain=1.0):
    """
    Calculate linear and angular velocity commands to navigate to the target position.
    :param target_tvec: Target marker's position in the coordinate frame (x, y, z)
    :param robot_tvec: Robot marker's position in the coordinate frame (x, y, z)
    :param linear_gain: Proportional gain for linear velocity
    :param angular_gain: Proportional gain for angular velocity
    :return: Linear velocity (m/s), Angular velocity (rad/s)
    """
    # Relative position in 2D (ignoring z for simplicity)
    relative_position = target_tvec[:2] - robot_tvec[:2]
    distance = np.linalg.norm(relative_position)  # Distance to target
    angle_to_target = np.arctan2(relative_position[1], relative_position[0])  # Angle to target

    # Proportional control
    linear_velocity = linear_gain * distance
    angular_velocity = angular_gain * angle_to_target

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

# 카메라 화소 잡는 코드
def adjust_camera_matrix(camera_matrix, original_width, original_height, new_width, new_height):
    """
    Adjust the camera matrix for a new resolution.
    """
    adjusted_matrix = camera_matrix.copy()
    adjusted_matrix[0, 0] *= new_width / original_width  # fx
    adjusted_matrix[1, 1] *= new_height / original_height  # fy
    adjusted_matrix[0, 2] *= new_width / original_width  # cx
    adjusted_matrix[1, 2] *= new_height / original_height  # cy
    return adjusted_matrix



def main():
    """
    Main method of the program.
    """
    # Validate the selected ArUco dictionary
    if ARUCO_DICT.get(desired_aruco_dictionary, None) is None:
        print(f"[ERROR] ArUco dictionary '{desired_aruco_dictionary}' is not supported.")
        return
    
    # 상태 변수 선언 
    goal_reached = False  # 목표 마커에 도달했는지 여부
    goal_threshold = 0.1  # 목표 거리 임계값 (단위: m)

    # Load the ArUco dictionary and parameters
    print(f"[INFO] Detecting '{desired_aruco_dictionary}' markers...")
    this_aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[desired_aruco_dictionary])
    this_aruco_parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(this_aruco_dictionary, this_aruco_parameters)

    # Start the video stream
    cap = cv2.VideoCapture(4)
    ret, frame = cap.read()

    if not cap.isOpened():
        print("[ERROR] Unable to access the webcam.")
        return

    # 명시적으로 카메라 해상도 설정 (1280x720)
    original_width, original_height = 1280, 720  # 캘리브레이션 해상도
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_height)

    # 실제 카메라가 제공하는 해상도 가져오기
    new_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    new_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 카메라 매트릭스 조정 (해상도가 변경된 경우)
    if new_width != original_width or new_height != original_height:
        print(f"[INFO] Adjusting camera matrix from {original_width}x{original_height} to {new_width}x{new_height}")
        camera_matrix_adjusted = adjust_camera_matrix(camera_matrix, original_width, original_height, new_width, new_height)
    else:
        camera_matrix_adjusted = camera_matrix

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        print(frame.shape)

        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        # Detect ArUco markers in the frame
        corners, ids, rejected = detector.detectMarkers(frame)
        if ids is not None and len(ids) > 0:
            ids = ids.flatten()  # Flatten the IDs list
            origin_rvec = None
            origin_tvec = None
            turtlebot_tvec = None  # 기준 마커의 tvec
            turtlebot_rvec = None  # 기준 마커의 rvec
            target_tvec = None
            markers_info = {}  # 다른 마커들의 정보 {ID: (rvec, tvec)}

            # Detected marker loop
            for i, marker_id in enumerate(ids):
                marker_corners = corners[i].reshape((4, 2))
                top_left, top_right, bottom_right, bottom_left = marker_corners

                # Draw bounding box
                cv2.line(frame, tuple(map(int, top_left)), tuple(map(int, top_right)), (0, 255, 0), 2)
                cv2.line(frame, tuple(map(int, top_right)), tuple(map(int, bottom_right)), (0, 255, 0), 2)
                cv2.line(frame, tuple(map(int, bottom_right)), tuple(map(int, bottom_left)), (0, 255, 0), 2)
                cv2.line(frame, tuple(map(int, bottom_left)), tuple(map(int, top_left)), (0, 255, 0), 2)

                # Pose estimation
                success, rvec, tvec = cv2.solvePnP(
                    np.array([
                        [-marker_length / 2, marker_length / 2, 0],
                        [marker_length / 2, marker_length / 2, 0],
                        [marker_length / 2, -marker_length / 2, 0],
                        [-marker_length / 2, -marker_length / 2, 0],
                    ]),
                    marker_corners,
                    camera_matrix_adjusted, dist_coeffs,
                )

                if success:
                    tvec = tvec.ravel()  # Flatten tvec
                    markers_info[marker_id] = (rvec, tvec)

                    # 기준 마커 저장
                    if marker_id == ORIGIN_MARKER_ID:
                        origin_tvec = tvec
                        origin_rvec = rvec
                    elif marker_id == TURTLEBOT_MARKER_ID:
                        turtlebot_tvec = tvec
                        turtlebot_rvec = rvec
                    elif marker_id == CONTAINER_MARKER_ID:
                        target_tvec = tvec
                        turtlebot_rvec = rvec

                    # Draw axes
                    cv2.drawFrameAxes(frame, camera_matrix_adjusted, dist_coeffs, rvec, tvec, 0.1)

            # 원점, 터틀봇, 타겟이 검출됐을때
            if origin_tvec is not None and turtlebot_tvec is not None  and target_tvec is not None:
                for marker_id, (rvec, tvec) in markers_info.items():
                    if marker_id == ORIGIN_MARKER_ID: # 원점마커가 검출됐을때만 아래 구문 실행
                        distance_to_target = np.linalg.norm(target_tvec[:2] - turtlebot_tvec[:2])
                        if not goal_reached:
                            if distance_to_target > goal_threshold:
                                # 속도 명령 계산
                                linear_vel, angular_vel = calculate_velocity_commands(target_tvec, turtlebot_tvec)
                                print(f"Linear Velocity = {linear_vel:.2f} m/s, Angular Velocity = {angular_vel:.2f} rad/s")
                                # 퍼블리시 코드
                                twist = Twist()
                                twist.linear.x = linear_vel
                                twist.angular.z = angular_vel
                                cmd_vel_publisher.publish(twist)                                
                            else:
                                # 목표 도달 시
                                goal_reached = True
                                print("Goal reached. Stopping TurtleBot.")
                                linear_vel, angular_vel = 0.0, 0.0
                                twist = Twist()
                                twist.linear.x = linear_vel
                                twist.angular.z = angular_vel
                                cmd_vel_publisher.publish(twist)
                                goal_reached = True
                        print(f"Target Marker {CONTAINER_MARKER_ID}: Linear Velocity = {linear_vel:.2f} m/s, Angular Velocity = {angular_vel:.2f} rad/s")
                        # 기준 마커의 각도 계산
                        rotation_matrix, _ = cv2.Rodrigues(origin_rvec)
                        angle_x = np.arctan2(rotation_matrix[2][1], rotation_matrix[2][2]) * (180.0 / np.pi)
                        angle_y = np.arctan2(-rotation_matrix[2][0],
                                            np.sqrt(rotation_matrix[2][1]**2 + rotation_matrix[2][2]**2)) * (180.0 / np.pi)
                        angle_z = np.arctan2(rotation_matrix[1][0], rotation_matrix[0][0]) * (180.0 / np.pi)

                        info_text = (f"ID: {marker_id} | Dist_from_CAM: {np.linalg.norm(tvec):.2f}m | "
                                    f"Angles: ({angle_x:.2f}, {angle_y:.2f}, {angle_z:.2f})")
                        cv2.putText(frame, info_text, (10, 30),  # 고정된 위치
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    else:
                        # 기준 마커와 다른 마커 간 상대 거리 계산
                        rel_distance, angle = calculate_relative_position(tvec, origin_tvec)

                        info_text = (f"ID: {marker_id} | Rel Dist: {rel_distance:.2f}m | "
                                    f"Rel_Angle: {np.degrees(angle):.2f}degree")
                        y_position = 50 + 20 * (marker_id % 10)  # 출력 위치 조정
                        cv2.putText(frame, info_text, (10, y_position),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Display the resulting frame
        cv2.imshow('ArUco Marker Detector', frame)

        # Break the loop if "q" is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print(__doc__)
    main()
