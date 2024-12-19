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
camera_matrix = np.load('/home/july/rokey_prj/week5/AMR-3/teamB-1_ws/calibration_matrix_3.npy')
dist_coeffs = np.load('/home/july/rokey_prj/week5/AMR-3/teamB-1_ws/distortion_coefficients_3.npy')

# Marker size in meters
marker_length = 0.107  # 10.7 cm


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

        if not ret:
            print("[ERROR] Failed to grab frame.")
            break

        # Detect ArUco markers in the frame
        corners, ids, rejected = detector.detectMarkers(frame)

        # Check if at least one ArUco marker was detected
        if ids is not None and len(ids) > 0:
            ids = ids.flatten()  # Flatten the IDs list

            # Loop over the detected ArUco markers
            for i, marker_id in enumerate(ids):
                # Extract corners for the marker
                marker_corners = corners[i].reshape((4, 2))
                (top_left, top_right, bottom_right, bottom_left) = marker_corners
                

                # Convert corner points to integers
                top_left = tuple(map(int, top_left))
                top_right = tuple(map(int, top_right))
                bottom_right = tuple(map(int, bottom_right))
                bottom_left = tuple(map(int, bottom_left))

                # Draw bounding box
                cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
                cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
                cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
                cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

                # Estimate pose of the marker
                success, rvec, tvec = cv2.solvePnP(
                    np.array([
                        [-marker_length / 2, marker_length / 2, 0],  # Top-left corner
                        [marker_length / 2, marker_length / 2, 0],   # Top-right corner
                        [marker_length / 2, -marker_length / 2, 0],  # Bottom-right corner
                        [-marker_length / 2, -marker_length / 2, 0],  # Bottom-left corner
                    ]),
                    marker_corners,
                    camera_matrix_adjusted,  # 조정된 카메라 매트릭스 사용
                    dist_coeffs,
                )

                if success:
                    
                    # 기존 거리 계산
                    distance = np.linalg.norm(tvec)  # 계산된 거리 (Measured Distance)
                    print(f"Marker {marker_id} | tvec: {tvec.ravel()} | Distance: {distance:.2f} m")
                    # 보정된 거리 계산
                    #corrected_distance = a * distance + b

                    # Angle calculation
                    rotation_matrix, _ = cv2.Rodrigues(rvec)
                    angle_x = np.arctan2(rotation_matrix[2][1], rotation_matrix[2][2]) * (180.0 / np.pi)
                    angle_y = np.arctan2(-rotation_matrix[2][0], np.sqrt(rotation_matrix[2][1]**2 + rotation_matrix[2][2]**2)) * (180.0 / np.pi)
                    angle_z = np.arctan2(rotation_matrix[1][0], rotation_matrix[0][0]) * (180.0 / np.pi)

                    # Overlay distance and angles on the video frame
                    info_text = f"ID: {marker_id} | Dist: {distance:.2f}m | Angles: ({angle_x:.2f}, {angle_y:.2f}, {angle_z:.2f})"
                    cv2.putText(frame, info_text, (top_left[0], top_left[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                    # Draw the pose axes on the marker
                    cv2.drawFrameAxes(frame, camera_matrix_adjusted, dist_coeffs, rvec, tvec, 0.1)

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
