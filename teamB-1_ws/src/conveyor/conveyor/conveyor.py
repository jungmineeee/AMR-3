import subprocess
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from system_interface.srv import ReportError

import serial
import serial.tools.list_ports

class ControlConveyor(Node):
    def __init__(self):
        super().__init__('control_conveyor')
        self.error_sent = False    # 에러 상황 감지시에 처리를 한번만 하게 하는 플래그
        
        self.get_logger().info("ControlConveyor 노드 시작")
        
        # 컨베이어 상태 퍼블리셔 생성
        self.error_call = self.create_client(
            ReportError,
            'report_error'
        )
        self.error_call
        
        # 시그널 서브스크라이버 생성
        self.sig_subscription = self.create_subscription(
            String, 
            '/gui_topic', 
            self.listener_callback, 
            10
        )
        self.sig_subscription
        
        # 시리얼 연결 초기화
        self.serial_conn = None
        self.init_serial_connection()

        # 시리얼 연결 상태 체크 타이머
        self.create_timer(1.0, self.check_serial_connection)
        

    def listener_callback(self, msg):
        """메시지를 수신하면 호출되는 콜백 함수"""
        received_data = msg.data
        if received_data == "ON" or received_data == "OFF" :
            self.get_logger().info(f'Subscribe한 시그널 : {received_data}')
            self.process_message(received_data) 

    def send_serial_command(self, command):
        """아두이노로 시리얼 명령 전송"""
        # 명령 전송
        self.serial_conn.write(command.encode() + b'\n')
        self.get_logger().info(f"Sent command to Arduino: {command}")

    def process_message(self, data):
        if data == "ON":
            self.send_serial_command("START")  # START 명령을 아두이노로 전송
            self.get_logger().info(f"Turning conveyor ON ")
            
        elif data == "OFF":            
            self.send_serial_command("STOP")  # STOP 명령을 아두이노로 전송
            self.get_logger().info("Turning conveyor OFF")
            
        else:
            self.get_logger().warning(f"Unknown command: {data}")

    def set_usb_permissions(self, device_path):
        """USB 포트에 대한 권한을 자동으로 설정"""
        try:
            # 권한 변경 명령 실행
            subprocess.run(['sudo', 'chmod', '666', device_path], check=True)
            self.get_logger().info(f"Permissions updated for {device_path}")
            
        except subprocess.CalledProcessError as e:
            self.get_logger().error(f"Failed to update permissions for {device_path}: {str(e)}")
            
    def init_serial_connection(self):
        try:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
                
            # 아두이노 포트 자동 감지
            arduino_ports = [
                p.device for p in serial.tools.list_ports.comports()
                if 'Arduino' in p.description
            ]
            
            if not arduino_ports: # 감지된 아두이노 없음
                raise serial.SerialException("Arduino not found")
            
            self.set_usb_permissions(arduino_ports[0])
            self.serial_conn = serial.Serial(
                port=arduino_ports[0],
                baudrate=115200,
                timeout=1.0
            )
            self.get_logger().info(f"Connected to Arduino on {arduino_ports[0]}")
            
        except serial.SerialException as e:
            error_message = f"Failed to initialize serial connection: {str(e)}"
            self.get_logger().error(error_message)
            
            if not self.error_sent:  # 에러가 아직 보고되지 않은 경우에만 처리
                self.handle_error(error_message)
            self.serial_conn = None

    def check_serial_connection(self):
        """아두이노로부터 주기적으로 받는 데이터를 기반으로 연결 상태 체크"""
        try:
            if not self.serial_conn or not self.serial_conn.is_open:
                raise serial.SerialException("아두이노가 연결되지 않았습니다.")

            try: 
                # 데이터 확인 및 처리
                if self.serial_conn.in_waiting > 0 :  # 수신 대기 중인 데이터 확인
                    response = self.serial_conn.read(self.serial_conn.in_waiting).decode().strip()

                    if response.startswith("."):
                        self.get_logger().info("Conveyor is idle (connection OK).")
                        self.error_sent = False  # 정상 동작 시에만 에러 상태 초기화
                    elif response.startswith("_"):
                        self.get_logger().info("Conveyor is running (connection OK).")
                        self.error_sent = False
                    else:
                        self.get_logger().warning(f"Unexpected response from Arduino: {response}")
                        
            except OSError as e:
                    raise serial.SerialException("아두이노 연결이 끊어졌습니다.")

        except serial.SerialException as e:
            if not self.error_sent:  # 에러가 아직 보고되지 않은 경우에만 처리
                self.get_logger().error(f"Serial check error: {str(e)}")
                self.handle_error(str(e))
                
            self.serial_conn = None  # 연결 끊김 상태로 설정
            self.init_serial_connection()  # 재연결 시도
            
    def handle_error(self, error_msg):
        """에러 처리 및 서비스 호출"""
        # 에러 메시지 전송 상태를 먼저 설정
        self.error_sent = True
        
        while not self.error_call.wait_for_service(timeout_sec=0.2) :
            self.get_logger().warning("에러처리 서비스 노드가 OFF 상태입니다.")

        request = ReportError.Request()
        request.error_msg = error_msg
        
        self.get_logger().info(f"Error service request sent for: {error_msg}")
        
        future = self.error_call.call_async(request)
        future.add_done_callback(self.error_service_response)
            
    def error_service_response(self, future):
        """서비스 응답 처리"""
        response = future.result()
        
        if response.success :
            self.get_logger().info("이메일이 정상적으로 송신되었습니다.")
        else:
            self.get_logger().error("이메일이 송신에 실패하였습니다. 작동을 중단하고 시스템을 점검하세요.")

                
    def __del__(self):
        """소멸자에서 시리얼 연결 종료"""
        if hasattr(self, 'serial_conn') and self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
            
            
##############################################################
def main(args=None):
    rclpy.init(args=args)
    node = ControlConveyor()
    
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()


##############################################################
if __name__ == '__main__':
    main()
            