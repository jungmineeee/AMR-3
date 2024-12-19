"""
ArUco Marker Detector with Pose Estimation (OpenCV 4.10.0+)
This program detects ArUco markers and calculates their pose (distance and orientation).
"""

import cv2
import numpy as np
import threading
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

"""
1. 키보드 입력으로 goal 받음
2. goal 간다음 home으로 뒤로 이동
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
ORIGIN_MARKER_ID = 54
GOAL1_ID =  1 
GOAL2_ID = 2 
GOAL3_ID = 3
CONTAINER_MARKER_ID = 4
TURTLEBOT_MARKER_ID = 5


# 새로운 제어 상수 추가
MAX_LINEAR_VEL = 0.5
MIN_LINEAR_VEL = 0.1
MAX_ANGULAR_VEL = 1.0
DISTANCE_THRESHOLD = 0.5

class MarkerTracker:
    def __init__(self, max_history=5, z_threshold=0.5, x_threshold=0.1, y_threshold=0.1):
        """
        마커별 최근 tvec 및 rvec를 저장하고 평균을 계산하는 클래스
        :param max_history: 저장할 최근 데이터의 개수 (이동 평균 계산에 사용)
        :param z_threshold: z축 값의 이상치를 무시할 임계값
        :param x_threshold: x축 값의 이상치를 무시할 임계값
        :param y_threshold: y축 값의 이상치를 무시할 임계값
        """
        self.marker_positions = {}  # 마커별 tvec 저장소
        self.marker_rotations = {}  # 마커별 rvec 저장소
        self.max_history = max_history
        self.z_threshold = z_threshold  # 임계값 설정
        self.x_threshold = x_threshold
        self.y_threshold = y_threshold

    def update_marker(self, marker_id, tvec, rvec):
        """
        새로운 마커 tvec 및 rvec를 업데이트
        :param marker_id: 마커 ID
        :param tvec: 변환 벡터 (np.array 형태)
        :param rvec: 회전 벡터 (np.array 형태)
        """
        # tvec 저장
        if marker_id not in self.marker_positions:
            self.marker_positions[marker_id] = deque(maxlen=self.max_history)
        self.marker_positions[marker_id].append(tvec)

        # rvec 저장
        if marker_id not in self.marker_rotations:
            self.marker_rotations[marker_id] = deque(maxlen=self.max_history)
        self.marker_rotations[marker_id].append(rvec)

    def get_average_tvec(self, marker_id):
        """
        특정 마커의 평균 변환 벡터(tvec)를 반환하며, 특정 조건에서는 이상치를 무시합니다.
        :param marker_id: 마커 ID
        :return: 평균 tvec (np.array) 또는 None (저장된 데이터가 없을 경우)
        """
        if marker_id in self.marker_positions and len(self.marker_positions[marker_id]) > 0:
            positions = np.array(self.marker_positions[marker_id])
            
            # 특정 조건에서 이상치 제거 (예: |z| 값이 너무 큰 경우)
            valid_positions = positions[np.all(np.abs(positions[:, 2]) < self.z_threshold, axis=1)]
            
            if len(valid_positions) > 0:
                return np.mean(valid_positions, axis=0)
        return np.mean(valid_positions, axis=0)

    def get_average_rvec(self, marker_id):
        """
        특정 마커의 평균 회전 벡터(rvec)를 반환하며, 특정 조건에서는 이상치를 무시합니다.
        :param marker_id: 마커 ID
        :return: 평균 rvec (np.array) 또는 None (저장된 데이터가 없을 경우)
        """
        if marker_id in self.marker_rotations and len(self.marker_rotations[marker_id]) > 0:
            rotations = np.array(self.marker_rotations[marker_id])
            
            #print("rotations: ", rotations)
            # 특정 조건에서 이상치 제거 (예: |x|, |y|, |z| 값이 너무 큰 경우)
            # 특정 조건에서 이상치 제거 (예: |x|, |y|, |z| 값이 너무 큰 경우)
            valid_rotations = rotations[
                (np.abs(rotations[:, 0, 0]) < self.x_threshold) &  # x축 필터링
                (np.abs(rotations[:, 1, 0]) < self.y_threshold) &  # y축 필터링
                (np.abs(rotations[:, 2, 0]) < self.z_threshold)    # z축 필터링
            ]
            
            if len(valid_rotations) > 0:
                return np.mean(valid_rotations, axis=0)
        return np.mean(valid_rotations, axis=0)

    
class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_reached = False
        self.goal_threshold = 0.1  # Threshold for goal distance
        self.current_target_id = None  # 초기값은 None
        self.cap = cv2.VideoCapture(4)

        # MarkerTracker 객체 생성
        self.tracker = MarkerTracker(max_history=10)

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

        # 키보드 입력을 처리하는 스레드 시작
        self.keyboard_thread = threading.Thread(target=self.keyboard_input)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()

        # 추가된 부분: move_to_home 초기화
        self.move_to_home = False  # 이동 플래그 초기화
        self.move_started = False  # 이동 시작 플래그 초기화

    def keyboard_input(self):
        while True:
            try:
                key = input()
                if key in ['1', '2', '3', '4', '5', '51','55', '54' ]:
                    marker_id = int(key)
                    self.current_target_id = marker_id
                    self.goal_reached = False  # 새로운 타겟 선택시 리셋
                    self.get_logger().info(f'Target changed to marker {marker_id}')
                elif key == 's':
                    self.goal_reached = True
                    self.publish_cmd_vel(0.0, 0.0)
                    self.get_logger().info('Robot stopped')
            except Exception as e:
                self.get_logger().error(f'Error reading keyboard input: {e}')

    def set_new_goal(self, target_id):
        """
        새로운 목표를 설정.
        """
        if self.current_target_id != target_id:
            self.current_target_id = target_id
            self.goal_reached = False  # 목표 도달 상태 초기화
            self.get_logger().info(f"New goal set to Marker ID: {target_id}")

    def detect_and_publish(self):
        """
        ArUco 마커를 검출하고 이동 제어를 수행.
        """
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to grab frame.")
            return

        # ArUco 마커 검출
        this_aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[desired_aruco_dictionary])
        this_aruco_parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(this_aruco_dictionary, this_aruco_parameters)

        corners, ids, _ = detector.detectMarkers(frame)
        if ids is None:
            cv2.imshow("ArUco Marker Detector", frame)
            cv2.waitKey(1)
            return

        ids = ids.flatten()
        markers = {}  # 마커 정보 저장
        for i, marker_id in enumerate(ids):
            marker_corners = corners[i].reshape((4, 2))
            
            # corners 순서 출력
            # print(f"\n코드 V3_3 Marker {marker_id} corners:")
            # print("Corner 0 (좌상단 예상):", marker_corners[0])
            # print("Corner 1 (우상단 예상):", marker_corners[1])
            # print("Corner 2 (우하단 예상):", marker_corners[2])
            # print("Corner 3 (좌하단 예상):", marker_corners[3])
            
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
                markers[marker_id] = {
                    'tvec': tvec.ravel(),
                    'rvec': rvec,
                    'corners': marker_corners
                }
                # 검출된 id들의 tvec, rvec 저장
                self.tracker.update_marker(marker_id, tvec, rvec)

                # if marker_id == 2:
                #     # 회전 행렬로 변환해서 각 축 방향 확인
                #     rotation_matrix, _ = cv2.Rodrigues(rvec)
                #     print(f"\nMarker {marker_id} axes:")
                #     # print("X axis:", rotation_matrix[:, 0])
                #     # print("Y axis:", rotation_matrix[:, 1])
                #     print("Z axis:", rotation_matrix[:, 2])
                #     print(f"tvec: {tvec}")


        # 시각화: 바운딩 박스, 텍스트 추가
        for marker_id, marker_data in markers.items():
            
            tvec = marker_data['tvec']
            rvec = marker_data['rvec']
            corners = marker_data['corners']

            # 바운딩 박스 그리기
            top_left = tuple(map(int, corners[0]))
            top_right = tuple(map(int, corners[1]))
            bottom_right = tuple(map(int, corners[2]))
            bottom_left = tuple(map(int, corners[3]))

            cv2.line(frame, top_left, top_right, (0, 255, 0), 2)
            cv2.line(frame, top_right, bottom_right, (0, 255, 0), 2)
            cv2.line(frame, bottom_right, bottom_left, (0, 255, 0), 2)
            cv2.line(frame, bottom_left, top_left, (0, 255, 0), 2)

            # 중심점 표시
            center_x = int(corners[:, 0].mean())
            center_y = int(corners[:, 1].mean())
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

            # 텍스트 추가
            text = f"ID: {marker_id} | x: {tvec[0]:.2f}, y: {tvec[1]:.2f}, z: {tvec[2]:.2f}"
            cv2.putText(frame, text, (center_x + 10, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            # 축 그리기
            cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

        # 터틀봇과 ORIGIN 확인
        if TURTLEBOT_MARKER_ID not in markers or ORIGIN_MARKER_ID not in markers:
            cv2.imshow("ArUco Marker Detector", frame)
            cv2.waitKey(1)
            return

        turtlebot_marker = markers[TURTLEBOT_MARKER_ID]
        origin_marker = markers[ORIGIN_MARKER_ID]
        distance_to_origin = np.linalg.norm(turtlebot_marker['tvec'] - origin_marker['tvec'])
        #print("distance_to_origin",distance_to_origin)
        # # 초기 상태: 목표가 없고 HOME 위치가 아니면 STOP
        # if self.current_target_id is None:
        #     if distance_to_origin > 0.31:  # HOME에서 30cm 이상 떨어짐
        #         self.publish_cmd_vel(0.0, 0.0)
        #         self.get_logger().info("No target set. Robot is not at HOME. Stopping.")
        #     cv2.imshow("ArUco Marker Detector", frame)
        #     cv2.waitKey(1)
        #     return

        # 목표 위치 처리
        
        if self.current_target_id in markers:
            goal_marker = markers[self.current_target_id]
            #goal_tvec_avg = self.tracker.get_average_tvec(self.current_target_id)
            #goal_rvec_avg = self.tracker.get_average_rvec(self.current_target_id)
            #distance_to_goal = np.linalg.norm(turtlebot_marker['tvec'] - goal_tvec_avg)
            distance_to_goal = np.linalg.norm(turtlebot_marker['tvec'] - goal_marker['tvec'])

            # HOME 복귀 로직
            if self.move_to_home:
                if distance_to_origin > 0.32:  # HOME에서 32cm 이상 떨어진 경우
                    linear_vel, angular_vel = calculate_velocity_commands(
                        origin_marker['tvec'], turtlebot_marker['tvec'],
                        turtlebot_marker['rvec'], origin_marker['rvec']
                    )
                    self.publish_cmd_vel(linear_vel, angular_vel)
                    self.get_logger().info(f"cmd_vel: linear {linear_vel}, angular {angular_vel}")
                    self.get_logger().info("Returning to HOME.")
                else:  # HOME 도달
                    self.publish_cmd_vel(0.0, 0.0)
                    self.move_to_home = False  # HOME 복귀 완료
                    self.goal_reached = False  # 목표 상태 초기화
                    self.get_logger().info("Reached HOME. Waiting for next goal.")

            # 새로운 목표로 이동
            elif not self.move_to_home and self.current_target_id and not self.goal_reached:
                if distance_to_goal > 0.5:  # 목표로 이동
                    linear_vel, angular_vel = calculate_velocity_commands(
                        goal_marker['tvec'], turtlebot_marker['tvec'],
                        turtlebot_marker['rvec'], goal_marker['rvec']
                    )
                    self.publish_cmd_vel(linear_vel, angular_vel)
                    self.get_logger().info(f"cmd_vel:linear {linear_vel}, angular {angular_vel}")
                    self.get_logger().info(f"Moving to goal {self.current_target_id}")
                else:  # 목표 도달
                    self.publish_cmd_vel(0.0, 0.0)
                    self.get_logger().info(f"cmd_vel:linear 0.0, angular 0.0")
                    self.goal_reached = True
                    self.move_to_home = True  # 목표 도달 후 HOME으로 복귀
                    self.move_started = False  # 다음 이동을 위해 초기화
                    self.get_logger().info(f"Reached goal {self.current_target_id}. Preparing to return to HOME.")


        # 프레임 표시
        cv2.imshow("ArUco Marker Detector", frame)
        cv2.waitKey(1)


    def publish_cmd_vel(self, linear_vel, angular_vel):
        twist = Twist()
        twist.linear.x = -linear_vel
        twist.angular.z = angular_vel
        self.cmd_vel_publisher.publish(twist)
        self.get_logger().info(f"Published: Linear Vel = {linear_vel:.2f}, Angular Vel = {angular_vel:.2f}")



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
    
    # 후진 가능 여부: 목표 위치가 뒤에 있다면 후진 방향으로 이동
    direction_to_target = target_tvec - turtlebot_tvec
    if np.dot(direction_to_target, turtlebot_forward) < 0:  # 목표가 뒤에 있을 경우
        angle = -angle  # 회전 방향 반대로
        linear_gain = -linear_gain  # 후진 속도 적용
    
    # 회전 방향 적용
    angle = rotation_direction * angle  # -값: 왼쪽 회전, +값: 오른쪽 회전
    
    # 거리 계산
    distance = np.linalg.norm(target_tvec - turtlebot_tvec)
    
    # 속도 계산
    linear_velocity = linear_gain * distance
    angular_velocity = angular_gain * angle
    
    if abs(angle) < np.radians(20):  # 20도 이내면 거의 정렬된 것으로 간주
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
