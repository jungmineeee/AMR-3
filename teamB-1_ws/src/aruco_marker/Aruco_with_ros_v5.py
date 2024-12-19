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
1. 아루코마커 3.3버전 cal_vel 함수 사용
2. goal 위치주면 home -> goal 이동하고 다음 goal 들어올때까지 대기하게
3. 곡선이동 대신 직선이동 로직 필요
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

class TurtleBotController(Node):
    def __init__(self, cmd_vel_publisher):
        super().__init__('turtlebot_controller')
        self.cmd_vel_publisher = cmd_vel_publisher
        self.goal_marker_id = None  # goal 마커 ID (1, 2, 3)
        self.move_state = "idle"  # 이동 상태: "to_marker4", "to_home", "to_goal", "align_goal", "to_goal_final"
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        self.initial_y_axis = None  # 초기 y축 방향 저장

    def set_initial_direction(self, rvec):
        """
        Calculate and store the initial y-axis direction based on the initial rotation vector.
        """
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        self.initial_y_axis = rotation_matrix[:, 1]  # y축 방향 벡터 저장
        self.get_logger().info(f"Initial y-axis direction set: {self.initial_y_axis}")

    def publish_cmd_vel(self):
        """
        Publish the current velocity commands to /cmd_vel topic.
        """
        twist = Twist()
        twist.linear.x = -self.linear_vel
        twist.angular.z = -self.angular_vel
        self.cmd_vel_publisher.publish(twist)
        self.get_logger().info(f"Published cmd_vel: linear_vel={self.linear_vel:.2f}, angular_vel={self.angular_vel:.2f}")

    def set_goal(self, marker_id):
        """
        Set the goal marker ID and start the sequence.
        """
        self.goal_marker_id = marker_id
        self.move_state = "to_marker4"
        self.get_logger().info(f"Goal set to Marker ID: {marker_id}. Starting sequence.")

    def control_loop(self, turtlebot_marker, origin_marker, marker4, goal_marker):
        """
        Main control logic for TurtleBot navigation.
        """
        if self.move_state == "to_marker4":
            # Move to marker4
            print("컨테이너 집으러 가는중 ")
            self.linear_vel = 0.07
            self.angular_vel = 0.0
            distance_to_marker4 = np.linalg.norm(turtlebot_marker['tvec'] - marker4['tvec'])
            if distance_to_marker4 < 0.6:  # 도착
                self.move_state = "to_home"
                self.get_logger().info("Reached Marker 4. Moving back to Home.")

        elif self.move_state == "to_home":
            # Move back to home (reverse)
            self.linear_vel = -0.07
            self.angular_vel = 0.0
            distance_to_home = np.linalg.norm(turtlebot_marker['tvec'] - origin_marker['tvec'])
            if distance_to_home < 0.5:  # 도착
                self.move_state = "to_goal"
                self.get_logger().info("Reached Home. Moving to Goal intersection.")

        elif self.move_state == "to_goal":
            # Move to intersection point
            intersection = self.calculate_intersection(
                turtlebot_marker['tvec'], turtlebot_marker['rvec'],
                goal_marker['tvec'], goal_marker['rvec']
            )
            distance_to_intersection = np.linalg.norm(turtlebot_marker['tvec'][:2] - intersection[:2])
            if distance_to_intersection > 0.05:  # 이동 중
                self.linear_vel = 0.07
                self.angular_vel = 0.0
            else:  # 교점 도착
                self.move_state = "align_goal"
                self.get_logger().info("Reached Goal intersection. Aligning with Goal.")

        elif self.move_state == "align_goal":
            # Align with goal's z-axis
            turtlebot_y_axis = cv2.Rodrigues(turtlebot_marker['rvec'])[0][:, 1]  # 터틀봇의 y축 방향 벡터
            initial_y_axis = self.initial_y_axis  # 초기 y축 방향

            # 현재 y축과 초기 y축 사이의 각도 계산
            dot_product = np.dot(turtlebot_y_axis[:2], initial_y_axis[:2])  # 2D 평면에서의 내적
            angle_to_goal = np.arccos(np.clip(dot_product, -1.0, 1.0))  # 각도 계산
            angle_to_goal_degrees = np.degrees(angle_to_goal)

            # 방향성 계산 (외적)
            cross_product = np.cross(
                np.append(initial_y_axis[:2], 0),  # z축 추가
                np.append(turtlebot_y_axis[:2], 0)
            )
            rotation_direction = np.sign(cross_product[2])  # z축 방향(양수: 왼쪽, 음수: 오른쪽)

            # 왼쪽으로만 회전 고정
            self.linear_vel = 0.0
            self.angular_vel = -MAX_ANGULAR_VEL #if rotation_direction > 0 else MAX_ANGULAR_VEL
            #print(angle_to_goal_degrees)

            # y축이 -90도(왼쪽으로 회전)인 상태인지 확인
            if 85 <= angle_to_goal_degrees <= 95:  # ±10도 오차 허용
                self.linear_vel = 0.0
                self.angular_vel = 0.0
                self.move_state = "to_goal_final"
                self.get_logger().info("Aligned with Goal. Moving to Goal.")

            self.publish_cmd_vel()


        elif self.move_state == "to_goal_final":
            # Final move to goal
            distance_to_goal = np.linalg.norm(turtlebot_marker['tvec'] - goal_marker['tvec'])
            if distance_to_goal > 0.45: # 목표지점과 40cm 넘게 떨어져있을때
                self.linear_vel = 0.07
                self.angular_vel = 0.0
            else:  # 도착
                self.move_state = "idle"
                self.linear_vel = 0.0
                self.angular_vel = 0.0
                self.get_logger().info("Reached Goal. Sequence completed.")

        else:  # idle 상태
            # Stop movement in idle state
            self.linear_vel = 0.0
            self.angular_vel = 0.0

        # cmd_vel은 idle 상태가 아닐 때만 발행
        if self.move_state != "idle":
            self.publish_cmd_vel()


    def calculate_intersection(self, tvec1, rvec1, tvec2, rvec2):
        """
        Calculate the intersection point of two lines defined by their translation and rotation vectors.
        """
        direction1 = cv2.Rodrigues(rvec1)[0][:, 1]  # TurtleBot y축 방향
        direction2 = -cv2.Rodrigues(rvec2)[0][:, 2]  # Goal z축 방향

        # 직선의 방정식을 이용하여 교점 계산
        A = np.array([direction1[:2], -direction2[:2]]).T
        b = tvec2[:2] - tvec1[:2]
        t = np.linalg.solve(A, b)

        intersection = tvec1 + t[0] * direction1  # 교점 계산
        return intersection


class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')

        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        # TurtleBotController 초기화
        self.turtlebot_controller = TurtleBotController(self.cmd_vel_publisher)
        self.goal_reached = False
        self.goal_threshold = 0.1  # Threshold for goal distance
        self.current_target_id = None  # 초기값은 None
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

        # 키보드 입력을 처리하는 스레드 시작
        self.keyboard_thread = threading.Thread(target=self.keyboard_input)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()

        # 추가된 부분: move_to_home 초기화
        # self.move_to_home = False  # 이동 플래그 초기화
        # self.move_started = False  # 이동 시작 플래그 초기화

    def keyboard_input(self):
        while True:
            try:
                key = input()
                if key in ['1', '2', '3' ]:
                    marker_id = int(key)
                    self.current_target_id = marker_id
                    self.turtlebot_controller.set_goal(marker_id)
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

        if TURTLEBOT_MARKER_ID in markers and ORIGIN_MARKER_ID in markers:
            turtlebot_marker = markers[TURTLEBOT_MARKER_ID]
            origin_marker = markers[ORIGIN_MARKER_ID]
            container_marker = markers.get(CONTAINER_MARKER_ID, None)
            goal_marker = markers.get(self.current_target_id, None)  # 목표 마커가 없을 수 있으므로 기본값 None
            #self.get_logger().info("터틀봇 주행 시작")
            # 초기 y축 방향 설정
            if self.turtlebot_controller.initial_y_axis is None:
                self.turtlebot_controller.set_initial_direction(turtlebot_marker['rvec'])

            # 추가 - 터틀봇 제어 로직 실행
            self.turtlebot_controller.control_loop(
                turtlebot_marker,
                origin_marker,
                container_marker,
                goal_marker  # 키보드 입력
            )
        else:
            self.get_logger().info("TurtleBot or Origin marker not detected.")


        # turtlebot_marker = markers[TURTLEBOT_MARKER_ID]
        # origin_marker = markers[ORIGIN_MARKER_ID]
        # distance_to_origin = np.linalg.norm(turtlebot_marker['tvec'] - origin_marker['tvec'])

        # # 목표 위치 처리        
        # if self.current_target_id in markers:
        #     goal_marker = markers[self.current_target_id]
        #     #goal_tvec_avg = self.tracker.get_average_tvec(self.current_target_id)
        #     #goal_rvec_avg = self.tracker.get_average_rvec(self.current_target_id)
        #     #distance_to_goal = np.linalg.norm(turtlebot_marker['tvec'] - goal_tvec_avg)
        #     distance_to_goal = np.linalg.norm(turtlebot_marker['tvec'] - goal_marker['tvec'])

        #     # HOME 복귀 로직
        #     if self.move_to_home:
        #         if distance_to_origin > 0.32:  # HOME에서 32cm 이상 떨어진 경우
        #             linear_vel, angular_vel = calculate_velocity_commands(
        #                 origin_marker['tvec'], turtlebot_marker['tvec'],
        #                 turtlebot_marker['rvec'], origin_marker['rvec']
        #             )
        #             self.publish_cmd_vel(linear_vel, angular_vel)
        #             self.get_logger().info(f"cmd_vel: linear {linear_vel}, angular {angular_vel}")
        #             self.get_logger().info("Returning to HOME.")
        #         else:  # HOME 도달
        #             self.publish_cmd_vel(0.0, 0.0)
        #             self.move_to_home = False  # HOME 복귀 완료
        #             self.goal_reached = False  # 목표 상태 초기화
        #             self.get_logger().info("Reached HOME. Waiting for next goal.")

        #     # 새로운 목표로 이동
        #     elif not self.move_to_home and self.current_target_id and not self.goal_reached:
        #         if distance_to_goal > 0.5:  # 목표로 이동
        #             linear_vel, angular_vel = calculate_velocity_commands(
        #                 goal_marker['tvec'], turtlebot_marker['tvec'],
        #                 turtlebot_marker['rvec'], goal_marker['rvec']
        #             )
        #             self.publish_cmd_vel(linear_vel, angular_vel)
        #             self.get_logger().info(f"cmd_vel:linear {linear_vel}, angular {angular_vel}")
        #             self.get_logger().info(f"Moving to goal {self.current_target_id}")
        #         else:  # 목표 도달
        #             self.publish_cmd_vel(0.0, 0.0)
        #             self.get_logger().info(f"cmd_vel:linear 0.0, angular 0.0")
        #             self.goal_reached = True
        #             self.move_to_home = True  # 목표 도달 후 HOME으로 복귀
        #             self.move_started = False  # 다음 이동을 위해 초기화
        #             self.get_logger().info(f"Reached goal {self.current_target_id}. Preparing to return to HOME.")


        # 프레임 표시
        cv2.imshow("ArUco Marker Detector", frame)
        cv2.waitKey(1)


    def publish_cmd_vel(self, linear_vel, angular_vel):
        twist = Twist()
        twist.linear.x = -linear_vel
        twist.angular.z = angular_vel
        self.cmd_vel_publisher.publish(twist)
        self.get_logger().info(f"Published: Linear Vel = {linear_vel:.2f}, Angular Vel = {angular_vel:.2f}")



def calculate_velocity_commands(
    target_tvec, turtlebot_tvec, turtlebot_rvec, target_rvec, 
    linear_gain=0.1, angular_gain=0.2, reverse=False, align=False
):
    """
    Velocity calculation for moving towards a target with linear or angular adjustments.
    
    Parameters:
    - target_tvec: Target translation vector
    - turtlebot_tvec: TurtleBot translation vector
    - turtlebot_rvec: TurtleBot rotation vector
    - target_rvec: Target rotation vector
    - reverse: If True, move backwards
    - align: If True, align direction without moving forward
    
    Returns:
    - linear_velocity, angular_velocity
    """
    # 회전 행렬 계산
    turtlebot_rotation, _ = cv2.Rodrigues(turtlebot_rvec)
    target_rotation, _ = cv2.Rodrigues(target_rvec)
    
    # 방향 벡터 추출
    turtlebot_forward = turtlebot_rotation[:, 1]  # y축
    target_direction = target_tvec - turtlebot_tvec  # 목표 방향 벡터
    target_distance = np.linalg.norm(target_direction)  # 목표와의 거리

    # 목표가 정면에 있을 경우 직선 이동
    dot_product = np.dot(turtlebot_forward, target_direction / target_distance)
    angle_error = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_error_degrees = np.degrees(angle_error)

    # 후진 이동
    if reverse:
        linear_velocity = -linear_gain * target_distance
    else:
        linear_velocity = linear_gain * target_distance

    # 각도 기반으로 angular_velocity 고정
    if align:
        # 정렬 모드: 목표 z축에 정렬
        if angle_error_degrees > 5:  # ±5도 이상 차이가 나면 회전
            angular_velocity = MAX_ANGULAR_VEL if dot_product > 0 else -MAX_ANGULAR_VEL
        else:
            angular_velocity = 0.0
        linear_velocity = 0.0  # 정렬 중에는 직진하지 않음
    else:
        # 직선 이동 모드: ±5도 이하만 직진
        if abs(angle_error_degrees) <= 5:
            angular_velocity = 0.0
        else:
            angular_velocity = MAX_ANGULAR_VEL if dot_product > 0 else -MAX_ANGULAR_VEL

    # 목표에 매우 가까운 경우 정지
    if target_distance < DISTANCE_THRESHOLD:
        linear_velocity = 0.0
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
