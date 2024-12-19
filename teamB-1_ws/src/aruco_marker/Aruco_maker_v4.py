#!/usr/bin/env python
"""
ArUco Marker Detector with Pose Estimation (OpenCV 4.10.0+)
This program detects ArUco markers and calculates their pose (distance and orientation).
"""

import cv2
import numpy as np

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

# camera_matrix = np.array([
#     [1.94937475e+03, 0.00000000e+00, 6.47921190e+02],
#     [0.00000000e+00, 1.48732242e+03, 3.88935994e+02],
#     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
# ])
# dist_coeffs = np.array([[1.24601716e+00, -8.68182411e+01, 1.91644766e-03, 4.79896522e-03, 1.76238268e+03]])


# Marker size in meters
marker_length = 0.107  # 10.7 cm

# 로봇 마커 ID (기준 마커)
ROBOT_MARKER_ID = 0  # 로봇 마커 ID를 지정 (예: 0)

# def calculate_relative_position(marker_tvec, robot_tvec):
#     """
#     Calculate the relative position (distance and direction) between markers.
#     :param marker_tvec: Target marker's tvec
#     :param robot_tvec: Robot marker's tvec
#     :return: Relative distance, angle in radians
#     """
#     relative_position = marker_tvec - robot_tvec
#     distance = np.linalg.norm(relative_position)  # 거리 계산
#     angle = np.arctan2(relative_position[1], relative_position[0])  # 방향 계산
#     return distance, angle

def calculate_relative_position(marker_tvec, robot_tvec):
    """
    Calculate the relative position (distance and direction) between markers.
    :param marker_tvec: Target marker's tvec
    :param robot_tvec: Robot marker's tvec
    :return: Relative distance, angle in radians
    """
    # x, y만 사용하여 2D 방향 계산
    relative_position = marker_tvec[:2] - robot_tvec[:2]  # x, y만 고려
    distance = np.linalg.norm(relative_position)  # 평면 거리 계산
    angle = np.arctan2(relative_position[1], relative_position[0])  # 방향 각도 계산
    print(f"Relative Position: {relative_position}, Distance: {distance}, Angle: {np.degrees(angle):.2f}")
    return distance, angle


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

    # Load the ArUco dictionary and parameters
    print(f"[INFO] Detecting '{desired_aruco_dictionary}' markers...")
    this_aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[desired_aruco_dictionary])
    this_aruco_parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(this_aruco_dictionary, this_aruco_parameters)

    # Start the video stream
    cap = cv2.VideoCapture(4)
    ret, frame = cap.read()
    print(frame.shape)

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
            robot_tvec = None  # 기준 마커의 tvec
            robot_rvec = None  # 기준 마커의 rvec
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
                    if marker_id == ROBOT_MARKER_ID:
                        robot_tvec = tvec
                        robot_rvec = rvec

                    # Draw axes
                    cv2.drawFrameAxes(frame, camera_matrix_adjusted, dist_coeffs, rvec, tvec, 0.1)

            # 기준 마커 정보 표시
            if robot_tvec is not None:
                for marker_id, (rvec, tvec) in markers_info.items():
                    if marker_id == ROBOT_MARKER_ID:
                        # 기준 마커의 각도 계산
                        rotation_matrix, _ = cv2.Rodrigues(robot_rvec)
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
                        rel_distance, angle = calculate_relative_position(tvec, robot_tvec)

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
