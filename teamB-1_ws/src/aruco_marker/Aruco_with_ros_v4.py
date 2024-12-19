import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import threading

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
GOAL1_ID =  1 #1
GOAL2_ID = 2 #2
GOAL3_ID = 3
CONTAINER_MARKER_ID = 4#4
TURTLEBOT_MARKER_ID = 5#5


# 새로운 제어 상수 추가
MAX_LINEAR_VEL = 0.5
MIN_LINEAR_VEL = 0.1
MAX_ANGULAR_VEL = 0.5
DISTANCE_THRESHOLD = 0.5

def calculate_projected_position(target_pos, turtlebot_pos, container_rvec):
    """
    4번(container) 마커의 z-x 평면에 목표 위치와 터틀봇 위치를 투영
    """
    # 4번 마커의 z-x 평면 정의 (y축이 법선벡터)
    container_rotation, _ = cv2.Rodrigues(container_rvec)
    # 벡터 투영 공식: v_proj = v - (v·n)n
    # 여기서 v는 위치 벡터, n은 단위 법선 벡터
    plane_normal = container_rotation[:, 1]  # y축
    
    # 투영 계산
    target_projection = target_pos - np.dot(target_pos, plane_normal) * plane_normal
    turtlebot_projection = turtlebot_pos - np.dot(turtlebot_pos, plane_normal) * plane_normal
    
    return target_projection, turtlebot_projection

def calculate_goal_position(target_proj, container_rvec, distance=0.5):
    """
    프로젝션된 타겟 마커 위치로부터 일정 거리 떨어진 목표 위치 계산
        target_proj: 평면에 투영된 타겟 위치
        container_rvec: 4번 마커의 회전 벡터
        distance: 타겟으로부터 떨어질 거리 (기본값 0.5m)
    Returns:
        goal_position: 실제 도달할 목표 위치
    """
    # 4번 마커의 좌표계에서 목표 위치 계산
    container_rotation, _ = cv2.Rodrigues(container_rvec)
    # z축 방향 벡터 추출 (마커 평면에서의 전진 방향)
    forward_direction = -container_rotation[:, 2]  # z축 방향
    forward_direction = forward_direction / np.linalg.norm(forward_direction)  # 정규화

    # 타겟 마커 앞에 목표 지점 설정
    # 마커로부터 forward_direction 방향으로 distance만큼 떨어진 지점
    goal_position = target_proj + distance * forward_direction
    
    return goal_position

def calculate_target_position(target_marker_pos, target_marker_rvec, desired_distance):
    """
    아루코 마커로부터 일정 거리 떨어진 목표 위치 계산
    """
    # 마커의 방향 벡터 계산
    target_rotation, _ = cv2.Rodrigues(target_marker_rvec)
    marker_direction = -target_rotation[:, 2]  # 마커가 바라보는 방향(-z축)
    
    # 마커로부터 일정 거리 떨어진 위치 계산
    target_position = target_marker_pos + desired_distance * marker_direction
    
    return target_position


def calculate_velocity_commands(target_tvec, turtlebot_tvec, turtlebot_rvec, target_rvec, linear_gain=0.1, angular_gain=0.2):
    # 회전 행렬 계산
    turtlebot_rotation, _ = cv2.Rodrigues(turtlebot_rvec)
    target_rotation, _ = cv2.Rodrigues(target_rvec)
    
    # 방향 벡터 추출
    turtlebot_forward = turtlebot_rotation[:, 1]  # y축
    target_normal = -target_rotation[:, 2]  # -z축
    
    # 두 벡터의 외적으로 회전 방향 결정
    cross_product = np.cross(turtlebot_forward, target_normal)
    rotation_direction = -np.sign(cross_product[2])
    
    # 각도 계산
    dot_product = np.dot(turtlebot_forward, target_normal)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle = rotation_direction * angle
    
    # 거리 계산
    distance = np.linalg.norm(target_tvec - turtlebot_tvec)
    
    # 속도 계산
    linear_velocity = linear_gain * distance
    angular_velocity = angular_gain * angle
    
    if abs(angle) < np.radians(5):
        angular_velocity = 0.0
        
    return linear_velocity, angular_velocity


class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.cmd_vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.goal_reached = False
        self.goal_threshold = 0.1  # Threshold for goal distance
        self.cap = cv2.VideoCapture(4)

        # 현재 목표 마커 ID 저장 변수 추가
        self.current_target_id = None  # 초기값은 None

        if not self.cap.isOpened():
            self.get_logger().error("Unable to access the webcam.")
            rclpy.shutdown()

        # 카메라 해상도 변경 1280x720
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # detect_and_pub 1초에 한번 호출 -> 추후에는 detect 랑 pub을 따로 떼서 
        # detect만 1초에 한번 부르고 pub는 필요할때만 호출해도 될듯.(수정해야함)
        self.timer = self.create_timer(0.1, self.detect_and_publish)

        # 키보드 입력을 처리하는 스레드 시작
        self.keyboard_thread = threading.Thread(target=self.keyboard_input)
        self.keyboard_thread.daemon = True
        self.keyboard_thread.start()

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

    def detect_and_publish(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().error("Failed to grab frame.")
            return

        # 마커 검출
        this_aruco_dictionary = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[desired_aruco_dictionary])
        this_aruco_parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(this_aruco_dictionary, this_aruco_parameters)

        corners, ids, _ = detector.detectMarkers(frame)
        if ids is None:
            #self.get_logger().info("No markers detected.")
            cv2.imshow("ArUco Marker Detector", frame)  # 감지 실패 시에도 프레임 출력
            cv2.waitKey(1)
            return
            

        ids = ids.flatten()
        origin_tvec = None
        turtlebot_tvec = None
        target_tvec = None
        distance_to_target = None  # Ensure the variable is defined
        markers = {} # 마커 정보 저장을 위한 딕셔너리

        for i, marker_id in enumerate(ids):
            marker_corners = corners[i].reshape((4, 2))
            top_left, top_right, bottom_right, bottom_left = marker_corners
            # BB 그리기
            cv2.line(frame, tuple(map(int, top_left)), tuple(map(int, top_right)), (0, 255, 0), 2)
            cv2.line(frame, tuple(map(int, top_right)), tuple(map(int, bottom_right)), (0, 255, 0), 2)
            cv2.line(frame, tuple(map(int, bottom_right)), tuple(map(int, bottom_left)), (0, 255, 0), 2)
            cv2.line(frame, tuple(map(int, bottom_left)), tuple(map(int, top_left)), (0, 255, 0), 2)

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
                # Display tvec values (x, y, z) near the marker center
                center_x = int((top_left[0] + top_right[0] + bottom_right[0] + bottom_left[0]) / 4)
                center_y = int((top_left[1] + top_right[1] + bottom_right[1] + bottom_left[1]) / 4)
                center = (center_x, center_y)

                # 각 마커 중심점 그리기
                cv2.circle(frame, center, 5, (0, 0, 255), -1)  # Red dot at center

                # 각 마커 중심 좌표 써주기
                text = f"ID: {marker_id} | x: {tvec[0]:.2f}m, y: {tvec[1]:.2f}m, z: {tvec[2]:.2f}m"
                cv2.putText(frame, text, (center_x + 10, center_y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # 마커 정보 저장
                markers[marker_id] = {
                    'tvec': tvec,
                    'rvec': rvec
                }

                # Draw axes
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

        # 마커 검출 후 제어 로직
        if TURTLEBOT_MARKER_ID in markers and self.current_target_id is not None:
            #print(f"Turtlebot marker detected: {TURTLEBOT_MARKER_ID}")
            if CONTAINER_MARKER_ID in markers and self.current_target_id in markers:  # 4번 마커와 목표 마커가 모두 보일 때
                #print(f"Container marker {CONTAINER_MARKER_ID} and target marker {self.current_target_id} detected")
                # 4번 마커의 평면에 투영
                target_proj, turtlebot_proj = calculate_projected_position(
                    markers[self.current_target_id]['tvec'],
                    markers[TURTLEBOT_MARKER_ID]['tvec'],
                    markers[CONTAINER_MARKER_ID]['rvec']
                )
                
                # 목표 마커 앞의 도달 목표 지점 계산
                goal_position = calculate_goal_position(target_proj, markers[CONTAINER_MARKER_ID]['rvec'])
                
                # 터틀봇과 목표 지점 간의 거리 계산
                distance_to_goal = np.linalg.norm(goal_position - turtlebot_proj)
                
                if not self.goal_reached:
                    if distance_to_goal > 0.4:  # 40cm 이내로 접근하면 정지
                        linear_vel, angular_vel = calculate_velocity_commands(
                            goal_position,  # 목표 위치
                            turtlebot_proj,  # 투영된 터틀봇 위치
                            markers[TURTLEBOT_MARKER_ID]['rvec'],
                            markers[self.current_target_id]['rvec']
                        )
                        self.publish_cmd_vel(linear_vel, angular_vel)
                    else:
                        self.goal_reached = True
                        self.publish_cmd_vel(0.0, 0.0)
                        self.get_logger().info(f'Reached goal position for marker {self.current_target_id}')
        
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
