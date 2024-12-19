import numpy as np
import cv2
from cv2 import aruco

# 1. 캘리브레이션 데이터 로드
camera_matrix = np.load('/home/july/rokey_prj/week5/AMR-3/teamB-1_ws/calibration_matrix_fisheye.npy')
dist_coeffs = np.load('/home/july/rokey_prj/week5/AMR-3/teamB-1_ws/distortion_coefficients_fisheye.npy')

# 2. ArUco 마커 사전 정의 (5x5 마커)
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
detector = cv2.aruco.ArucoDetector(aruco_dict)

# 3. 마커 크기 설정 (10.5cm -> 미터로 변환)
marker_length = 0.105  # 10.5cm 크기의 마커

# 4. 웹캠으로 이미지 스트림 읽기
cap = cv2.VideoCapture(0)  # 0번 웹캠
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break

    # 5. ArUco 마커 탐지
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)

    # 6. 마커가 검출되었는지 확인
    if ids is not None:
        # 마커마다 Pose 추정
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, dist_coeffs)

        # Pose 결과 출력
        for i, id_ in enumerate(ids):
            # 거리 계산 (tvecs는 [x, y, z] 형태로 반환됨)
            tvec = tvecs[i][0]
            distance = np.sqrt(tvec[0]**2 + tvec[1]**2 + tvec[2]**2)

            # 각도 계산 (rvecs를 회전 행렬로 변환 후 각도 추출)
            rvec = rvecs[i][0]
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            angle_x = np.arctan2(rotation_matrix[2][1], rotation_matrix[2][2]) * (180.0 / np.pi)
            angle_y = np.arctan2(-rotation_matrix[2][0], np.sqrt(rotation_matrix[2][1]**2 + rotation_matrix[2][2]**2)) * (180.0 / np.pi)
            angle_z = np.arctan2(rotation_matrix[1][0], rotation_matrix[0][0]) * (180.0 / np.pi)

            # 결과 출력
            print(f"Marker ID: {id_[0]}")
            print(f"Distance: {distance:.2f} meters")
            print(f"Angles (X, Y, Z): ({angle_x:.2f}, {angle_y:.2f}, {angle_z:.2f}) degrees")

            # 시각화 (마커와 축 그리기)
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], 0.1)

    # 결과 이미지 출력
    cv2.imshow('Pose Estimation', frame)

    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 웹캠 및 OpenCV 창 닫기
cap.release()
cv2.destroyAllWindows()