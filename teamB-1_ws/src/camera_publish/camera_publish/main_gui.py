import sys
import rclpy
from rclpy.node import Node
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import numpy as np

from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

####################################################
class MainNode(Node):
    def __init__(self):
        super().__init__('main_node')
        self.callback_group = ReentrantCallbackGroup()
        self.subscription_webcam = None
        self.subscription_robotcam = None
        
        # 퍼블리셔 생성
        self.publisher = self.create_publisher(
            String,
            'gui_topic',
            10, # qos
            callback_group=self.callback_group # 콜백 그룹
        )
        
        # 여러가지 상태 시그널을 받는 서브스크라이버 하나 필요할듯 ?

    def webcam_subscription(self, topic_name, callback):
        """Initialize the subscription to the given topic."""
        # 웹캠 이미지를 subscribe 하는 서브스크라이버
        self.subscription_webcam = self.create_subscription(
            CompressedImage,
            topic_name, # 서브스크라이버 생성시 인자
            callback,   # 서브스크라이버 생성시 인자
            10,
            callback_group=self.callback_group
        )
        self.get_logger().info(f"Subscribed to topic: {topic_name}")
        
    def robotcam_subscription(self, topic_name, callback):
        """Initialize the subscription to the given topic."""
        # 로봇캠 이미지를 subscribe 하는 서브스크라이버
        self.subscription_robotcam = self.create_subscription(
            CompressedImage,
            topic_name, # 서브스크라이버 생성시 인자
            callback,   # 서브스크라이버 생성시 인자
            10,
            callback_group=self.callback_group
        )
        self.get_logger().info(f"Subscribed to topic: {topic_name}")

    def publish_message(self, message):
        """Publish a message to the button_click_topic."""
        msg = String()
        msg.data = message
        self.publisher.publish(msg)
        self.get_logger().info(f"퍼블리시 (메세지 / 신호): {message}")


#####################################################
class Ros2Worker(QThread):
    webcam_image_received = pyqtSignal(np.ndarray)
    robotcam_image_received = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        self.node = None
        self.running = True
        self.topic_name_cam = '/webcam_image'
        self.topic_name_robot = '/robotcam_image'

    def run(self):
        """Start ROS2 spinning in this thread."""
        rclpy.init()
        self.node = MainNode()
        
        self.node.webcam_subscription(self.topic_name_cam, self.handle_webcam_image)
        self.node.robotcam_subscription(self.topic_name_robot, self.handle_robotcam_image)
        
        # MultiThreadedExecutor 설정
        self.executor = MultiThreadedExecutor(num_threads=3)
        self.executor.add_node(self.node)

        while self.running:
            self.executor.spin_once(timeout_sec=0.1)

        # Clean up
        self.executor.shutdown()
        self.node.destroy_node()
        rclpy.shutdown()

    def handle_webcam_image(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.webcam_image_received.emit(cv_image)
        
    def handle_robotcam_image(self, msg):
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        self.robotcam_image_received.emit(cv_image)
    
    def publish_message(self, message):
        if self.node:
            if isinstance(message, list) :
                self.node.publish_message(message[0])
                self.node.publish_message(message[1])
                # 작업목록 json까지 두번 퍼블리시 로직
                
    #             # 숫자 추출 및 JSON 생성
    #         job_data = self.extract_numbers(job_text)
    #         job_json = json.dumps(job_data)
            # 리스트 위젯에서 선택한 값으로 추출
    # def extract_numbers(self, text):
    #     """
    #     텍스트에서 숫자 값을 추출하고 JSON 생성
    #     """
    #     # 정규표현식으로 red, blue, goto 숫자 추출
    #     red_match = re.search(r'red\s*x(\d+)', text)
    #     blue_match = re.search(r'blue\s*x(\d+)', text)
    #     goto_match = re.search(r'goto\s+goal\s+(\d+)', text)

    #     # 숫자 값 추출 (없을 경우 0으로 설정)
    #     red_count = int(red_match.group(1)) if red_match else 0
    #     blue_count = int(blue_match.group(1)) if blue_match else 0
    #     goto_count = int(goto_match.group(1)) if goto_match else 0

    #     # JSON 형식의 데이터 생성
    #     return {
    #         "red": red_count,
    #         "blue": blue_count,
    #         "to": goto_count
    #     }
            else :
                self.node.publish_message(message)

    def stop(self):
        """Stop the ROS2 spinning loop."""
        self.running = False
        self.quit()
        self.wait()


#####################################################
class MainGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("System Main GUI")
        self.resize(800, 630)  # 기본 창 크기를 800x630으로 설정

        # Initialize ROS2 worker
        self.ros_worker = Ros2Worker()  # Set your topic name here
        self.ros_worker.webcam_image_received.connect(self.update_webcam_image)
        # self.ros_worker.robotcam_image_received.connect(self.update_robotcam_image)
        self.ros_worker.start()
        
        # 타이머 및 상태 변수
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_timer)
        self.elapsed_time = 0
        self.is_paused = False

        # Setup UI
        self.setup_ui()
        
        # 작업목록 세팅
        file_path = "job_list.txt"
        try:
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()  # Remove leading/trailing whitespace
                    if line:
                        formatted_line = line.replace("\\n", "\n")
                        self.job_list.addItem(formatted_line)
                        
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")

        """ 이벤트 연결  """
        self.play_btn.clicked.connect(self.click_play_btn)
        self.stop_btn.clicked.connect(self.click_stop_btn)
        self.pause_btn.clicked.connect(self.click_pause_btn)
        self.resume_btn.clicked.connect(self.click_resume_btn)
        self.reset_btn.clicked.connect(self.click_reset_btn)
        self.conveyor_start_btn.clicked.connect(self.click_on_btn)
        self.conveyor_stop_btn.clicked.connect(self.click_off_btn)
        self.study_btn.clicked.connect(self.click_study_btn)
        

    """ 슬롯 함수 """
    def click_play_btn(self) :
        self.ros_worker.publish_message("PLAY")
        
        """ GUI 상태 트리거 """
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        
        """타이머 시작"""
        self.timer.start(1000)  # 1초마다 timeout 이벤트 발생
        self.is_paused = False
        
    def click_stop_btn(self) :
        self.ros_worker.publish_message("STOP")
        
        """ GUI 상태 트리거 """
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.reset_btn.setEnabled(True)
        
        """타이머 정지"""
        self.timer.stop()
        self.is_paused = False
        
    def click_pause_btn(self) :
        self.ros_worker.publish_message("PAUSE")
        
        """ GUI 상태 트리거 """
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(True)
        self.reset_btn.setEnabled(False)
        
        """타이머 일시 정지"""
        if self.timer.isActive():
            self.timer.stop()
            self.is_paused = True
        
    def click_resume_btn(self) :
        self.ros_worker.publish_message("RESUME")
        
        """ GUI 상태 트리거 """
        self.play_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.pause_btn.setEnabled(True)
        self.resume_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        
        """타이머 재개"""
        if self.is_paused:
            self.timer.start(1000)
            self.is_paused = False
        
    def click_reset_btn(self) :
        self.ros_worker.publish_message("RESET")
        
        """ GUI 상태 트리거 """
        self.play_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.pause_btn.setEnabled(False)
        self.resume_btn.setEnabled(False)
        self.reset_btn.setEnabled(False)
        
        """타이머 초기화"""
        self.timer.stop()
        self.elapsed_time = 0
        self.is_paused = False
        self.update_timer_label()
        
    def click_on_btn(self) :
        self.ros_worker.publish_message("ON")
        
        """ GUI 상태 트리거 """
        self.conveyor_start_btn.setEnabled(False)
        self.conveyor_stop_btn.setEnabled(True)
        
    def click_off_btn(self) :
        self.ros_worker.publish_message("OFF")
        
        """ GUI 상태 트리거 """
        self.conveyor_start_btn.setEnabled(True)
        self.conveyor_stop_btn.setEnabled(False)

    def click_study_btn(self) :
        self.ros_worker.publish_message("TRAIN")
            
    def update_timer(self):
        """타이머 업데이트"""
        self.elapsed_time += 1
        self.update_timer_label()

    def update_timer_label(self):
        """타이머 표시 업데이트"""
        hours = self.elapsed_time // 3600
        minutes = (self.elapsed_time % 3600) // 60
        seconds = self.elapsed_time % 60
        self.time.setText(f"{hours:02} : {minutes:02} : {seconds:02}")


    # UI 세팅
    def setup_ui(self):
        # Central widget
        self.central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(self.central_widget)

        # Camera image label
        self.cam_image = QLabel(self.central_widget)
        self.cam_image.setGeometry(QtCore.QRect(30, 20, 360, 300))
        self.cam_image.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                     "border-radius: 8px;\n")
        self.cam_image.setAlignment(QtCore.Qt.AlignCenter)
        self.cam_image.setText("Cam Image here!")

        # Robot camera label
        self.robot_cam = QLabel(self.central_widget)
        self.robot_cam.setGeometry(QtCore.QRect(410, 20, 360, 300))
        self.robot_cam.setStyleSheet("background-color: rgb(255, 255, 255);\n"
                                     "border-radius: 8px;\n")
        self.robot_cam.setAlignment(QtCore.Qt.AlignCenter)
        self.robot_cam.setText("Manipulator cam here!")

         # Job List
        self.job_list_label = QLabel(self.central_widget)
        self.job_list_label.setObjectName("job_list_label")
        self.job_list_label.setGeometry(QRect(30, 420, 261, 31))
        self.job_list_label.setStyleSheet(
            "background-color: rgb(98, 160, 234);\nborder-top-left-radius: 8px;\nborder-top-right-radius: 8px;\nfont-weight: bold;"
        )
        self.job_list_label.setAlignment(Qt.AlignCenter)
        self.job_list_label.setText("작업 목록")

        self.job_list = QListWidget(self.central_widget)
        self.job_list.setObjectName("job_list")
        self.job_list.setGeometry(QRect(30, 450, 261, 161))

        # Control Buttons
        self.robot_op_label = QLabel(self.central_widget)
        self.robot_op_label.setObjectName("robot_op_label")
        self.robot_op_label.setGeometry(QRect(330, 330, 67, 17))
        self.robot_op_label.setStyleSheet("font-weight: bold;")
        self.robot_op_label.setText("로봇 작동")

        self.play_btn = QPushButton(self.central_widget)
        self.play_btn.setObjectName("play_btn")
        self.play_btn.setGeometry(QRect(330, 360, 81, 41))
        self.play_btn.setStyleSheet("background-color: rgb(53, 132, 228);")
        self.play_btn.setText("Play")

        self.stop_btn = QPushButton(self.central_widget)
        self.stop_btn.setObjectName("stop_btn")
        self.stop_btn.setGeometry(QRect(420, 360, 81, 41))
        self.stop_btn.setStyleSheet("background-color: rgb(237, 51, 59);")
        self.stop_btn.setText("Stop")
        self.stop_btn.setDisabled(True)

        self.pause_btn = QPushButton(self.central_widget)
        self.pause_btn.setObjectName("pause_btn")
        self.pause_btn.setGeometry(QRect(510, 360, 81, 41))
        self.pause_btn.setStyleSheet("background-color: rgb(143, 240, 164);")
        self.pause_btn.setText("Pause")
        self.pause_btn.setDisabled(True)

        self.resume_btn = QPushButton(self.central_widget)
        self.resume_btn.setObjectName("resume_btn")
        self.resume_btn.setGeometry(QRect(600, 360, 81, 41))
        self.resume_btn.setStyleSheet("background-color: rgb(143, 240, 164);")
        self.resume_btn.setText("Resume")
        self.resume_btn.setDisabled(True)

        self.reset_btn = QPushButton(self.central_widget)
        self.reset_btn.setObjectName("reset_btn")
        self.reset_btn.setGeometry(QRect(690, 360, 81, 41))
        self.reset_btn.setText("Reset")
        self.reset_btn.setDisabled(True)

        # Conveyor Control
        self.conveyor_label = QLabel(self.central_widget)
        self.conveyor_label.setObjectName("conveyor_label")
        self.conveyor_label.setGeometry(QRect(330, 420, 101, 17))
        self.conveyor_label.setStyleSheet("font-weight: bold;")
        self.conveyor_label.setText("컨베이어 조작")

        self.conveyor_start_btn = QPushButton(self.central_widget)
        self.conveyor_start_btn.setObjectName("conveyor_start_btn")
        self.conveyor_start_btn.setGeometry(QRect(330, 450, 211, 41))
        self.conveyor_start_btn.setText("START")

        self.conveyor_stop_btn = QPushButton(self.central_widget)
        self.conveyor_stop_btn.setObjectName("conveyor_stop_btn")
        self.conveyor_stop_btn.setGeometry(QRect(560, 450, 211, 41))
        self.conveyor_stop_btn.setText("STOP")
        self.conveyor_stop_btn.setDisabled(True)

        # Manipulator Control
        self.manipulator_label = QLabel(self.central_widget)
        self.manipulator_label.setObjectName("manipulator_label")
        self.manipulator_label.setGeometry(QRect(330, 510, 101, 17))
        self.manipulator_label.setStyleSheet("font-weight: bold;")
        self.manipulator_label.setText("Manipulator")

        self.manipulator_run = QPushButton(self.central_widget)
        self.manipulator_run.setObjectName("manipulator_run")
        self.manipulator_run.setGeometry(QRect(330, 580, 211, 31))
        self.manipulator_run.setText("이동")
        
        self.study_btn = QPushButton(self.central_widget)
        self.study_btn.setObjectName("manipulator_run")
        self.study_btn.setGeometry(QRect(560, 580, 211, 31))
        self.study_btn.setText("학습 시작")

        # Position Inputs
        self.x_label = QLabel(self.central_widget)
        self.x_label.setObjectName("x_label")
        self.x_label.setGeometry(QRect(330, 540, 21, 21))
        self.x_label.setText("X:")

        self.x_input = QLineEdit(self.central_widget)
        self.x_input.setObjectName("x_input")
        self.x_input.setGeometry(QRect(350, 540, 101, 21))

        self.y_label = QLabel(self.central_widget)
        self.y_label.setObjectName("y_label")
        self.y_label.setGeometry(QRect(490, 540, 21, 21))
        self.y_label.setText("Y:")

        self.y_input = QLineEdit(self.central_widget)
        self.y_input.setObjectName("y_input")
        self.y_input.setGeometry(QRect(510, 540, 101, 21))

        self.z_label = QLabel(self.central_widget)
        self.z_label.setObjectName("z_label")
        self.z_label.setGeometry(QRect(650, 540, 21, 21))
        self.z_label.setText("Z:")

        self.z_input = QLineEdit(self.central_widget)
        self.z_input.setObjectName("z_input")
        self.z_input.setGeometry(QRect(670, 540, 101, 21))

        # Time and Status Labels
        self.time_label = QLabel(self.central_widget)
        self.time_label.setObjectName("time_label")
        self.time_label.setGeometry(QRect(50, 370, 101, 29))
        font = QFont()
        font.setPointSize(14)
        font.setBold(True)
        font.setWeight(75)
        self.time_label.setFont(font)
        self.time_label.setStyleSheet("font-weight: bold;")
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setText("작업 시간  :")

        self.time = QLabel(self.central_widget)
        self.time.setObjectName("time")
        self.time.setGeometry(QRect(160, 370, 111, 29))
        self.time.setFont(font)
        self.time.setAlignment(Qt.AlignCenter)
        self.time.setText("00 : 00 : 00")

        self.robot_status_label = QLabel(self.central_widget)
        self.robot_status_label.setObjectName("robot_status_label")
        self.robot_status_label.setGeometry(QRect(50, 330, 101, 31))
        self.robot_status_label.setFont(font)
        self.robot_status_label.setAlignment(Qt.AlignCenter)
        self.robot_status_label.setText("로봇 상태  :")

        self.robot_status = QLabel(self.central_widget)
        self.robot_status.setObjectName("robot_status")
        self.robot_status.setGeometry(QRect(170, 330, 84, 31))
        self.robot_status.setFont(font)
        self.robot_status.setAlignment(Qt.AlignCenter)
        self.robot_status.setText("Stop")
        
        self.line = QFrame(self.central_widget)
        self.line.setObjectName("line")
        self.line.setGeometry(QRect(330, 400, 441, 20))
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        self.line_2 = QFrame(self.central_widget)
        self.line_2.setObjectName("line_2")
        self.line_2.setGeometry(QRect(330, 490, 441, 20))
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

    # 웹캠 이미지 업데이트
    def update_webcam_image(self, cv_image):
        """Update the QLabel with the new image."""
        height, width, channels = cv_image.shape
        bytes_per_line = channels * width
        qt_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)
        self.cam_image.setPixmap(pixmap)
        
    # 로봇캠 이미지 업데이트
    def update_robotcam_image(self, cv_image):
        """Update the QLabel with the new image."""
        height, width, channels = cv_image.shape
        bytes_per_line = channels * width
        qt_image = QImage(cv_image.data, width, height, bytes_per_line, QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qt_image)
        self.robot_cam.setPixmap(pixmap)

    # QThred 종료
    def closeEvent(self, event):
        """Stop the ROS2 worker when the window is closed."""
        self.ros_worker.stop()
        event.accept()


#####################################################
# 로그인 페이지

class LoginWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Login Page")
        self.resize(665, 566)
        self.setStyleSheet("background-color: rgb(246, 245, 244);")
        
        # Setup UI
        self.setup_ui()
        
        """ 이벤트 연결 """
        self.enter_btn.clicked.connect(self.validate_login)
        self.exit_btn.clicked.connect(self.confirm_exit)

    def setup_ui(self) :
        # 중앙 위젯
        self.centralwidget = QWidget(self)
        self.centralwidget.setObjectName("centralwidget")
        self.setCentralWidget(self.centralwidget)

        # 제목 라벨
        self.name_label = QLabel(self.centralwidget)
        self.name_label.setGeometry(120, 90, 450, 81)
        self.name_label.setFont(QFont("Arial", 40, QFont.Bold))
        self.name_label.setAlignment(Qt.AlignCenter)
        self.name_label.setText("Rokey Fulfillment")

        # 버전 라벨
        self.version_label = QLabel(self.centralwidget)
        self.version_label.setGeometry(20, 20, 111, 17)
        self.version_label.setFont(QFont("Arial", 15))
        self.version_label.setText("Version 1.0")

        # 안내 문구
        self.label_5 = QLabel(self.centralwidget)
        self.label_5.setGeometry(140, 180, 401, 31)
        self.label_5.setFont(QFont("Arial", 16))
        self.label_5.setAlignment(Qt.AlignCenter)
        self.label_5.setText("로그인을 해주세요!")

        # 상단 구분선
        self.line = QFrame(self.centralwidget)
        self.line.setGeometry(100, 220, 481, 16)
        self.line.setFrameShape(QFrame.HLine)
        self.line.setFrameShadow(QFrame.Sunken)

        # User ID 입력
        self.id_label = QLabel(self.centralwidget)
        self.id_label.setGeometry(140, 270, 91, 31)
        self.id_label.setFont(QFont("Arial", 18))
        self.id_label.setAlignment(Qt.AlignCenter)
        self.id_label.setText("User ID")

        self.id = QLineEdit(self.centralwidget)
        self.id.setGeometry(270, 270, 271, 41)
        self.id.setAlignment(Qt.AlignCenter)
        self.id.setStyleSheet("""
            background-color: white;
            border-style: outset;
            border-width: 2px;
            border-radius: 6px;
            border-color: black;
            padding: 2px;
        """)

        # Password 입력
        self.pw_label = QLabel(self.centralwidget)
        self.pw_label.setGeometry(130, 330, 121, 41)
        self.pw_label.setFont(QFont("Arial", 18))
        self.pw_label.setAlignment(Qt.AlignCenter)
        self.pw_label.setText("Password")

        self.password = QLineEdit(self.centralwidget)
        self.password.setGeometry(270, 330, 271, 41)
        self.password.setAlignment(Qt.AlignCenter)
        self.password.setEchoMode(QLineEdit.Password)
        self.password.setStyleSheet("""
            background-color: white;
            border-style: outset;
            border-width: 2px;
            border-radius: 6px;
            border-color: black;
            padding: 2px;
        """)

        # 하단 구분선
        self.line_2 = QFrame(self.centralwidget)
        self.line_2.setGeometry(110, 400, 481, 16)
        self.line_2.setFrameShape(QFrame.HLine)
        self.line_2.setFrameShadow(QFrame.Sunken)

        # 로그인 버튼
        self.enter_btn = QPushButton(self.centralwidget)
        self.enter_btn.setGeometry(160, 440, 171, 91)
        self.enter_btn.setFont(QFont("Arial", 16))
        self.enter_btn.setText("일하러 가기")
        self.enter_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 16px;
                border-radius: 10px;
                border: 2px solid #357a38;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #2e7031;
            }
        """)

        # 종료 버튼
        self.exit_btn = QPushButton(self.centralwidget)
        self.exit_btn.setGeometry(350, 440, 171, 91)
        self.exit_btn.setFont(QFont("Arial", 16))
        self.exit_btn.setText("그냥 집에 가기")
        self.exit_btn.setStyleSheet("""
            QPushButton {
                background-color: #FAD4D4;
                color: black;
                font-size: 16px;
                border-radius: 10px;
                border: 2px solid #F5A6A6;
            }
            QPushButton:hover {
                background-color: #F9BDBD;
            }
            QPushButton:pressed {
                background-color: #F2A0A0;
            }
        """)
        
    def validate_login(self):
        """Validate the user credentials and open MainWindow if successful."""
        username = self.id.text()
        password = self.password.text()

        # Example: Hardcoded credentials
        if username == "admin" and password == "123456":
            self.open_main_window()
        else:
            QMessageBox.warning(self, "로그인 실패", "정확히 입력해주세요.")

    def confirm_exit(self):
        """Confirm before exiting the application."""
        reply = QMessageBox.question(
            self,
            "집에 가기",
            "정말로 집에 가시겠습니까?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if reply == QMessageBox.Yes:
            QApplication.instance().quit()

    def open_main_window(self):
        """Open the main application window and close the login window."""
        self.main_window = MainGUI()
        self.main_window.show()
        
        self.close()  # Close the login window


#####################################################
def main():
    app = QApplication(sys.argv)

    # Login Window
    login_window = LoginWindow()
    login_window.show()
    
    sys.exit(app.exec_())


#####################################################
if __name__ == '__main__':
    main()
