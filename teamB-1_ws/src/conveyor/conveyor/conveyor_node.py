import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import serial
import serial.tools.list_ports

class ConveyorNode(Node):
    def __init__(self):
        super().__init__('conveyor_node')
        # 이전 상태 저장 변수 추가
        self.last_status = None

        # 상태 퍼블리셔 생성 -> 에러노드에 보내주는 용
        self.status_publisher = self.create_publisher(
            String,
            'conveyor/status',
            10
        )

        # Topic 구독
        self.subscription = self.create_subscription(
            String,
            'gui_topic',  # 퍼블리싱되는 토픽 이름
            self.topic_callback,
            10
        )

        # 시리얼 연결 초기화
        self.serial_conn = None
        self.init_serial_connection()

        # 시리얼 연결 상태 체크 타이머
        self.create_timer(1.0, self.check_serial_connection)

    def init_serial_connection(self):
        """시리얼 연결 초기화"""
        try:
            # 아두이노 포트 자동 감지
            arduino_ports = [
                p.device for p in serial.tools.list_ports.comports()
                if 'Arduino' in p.description
            ]
            
            if not arduino_ports:
                raise serial.SerialException("Arduino not found")
                
            self.serial_conn = serial.Serial(
                port=arduino_ports[0],
                baudrate=115200,
                timeout=1.0
            )
            self.get_logger().info(f"Connected to Arduino on {arduino_ports[0]}")
            
        except serial.SerialException as e:
            self.handle_error(f"Serial connection error: {str(e)}")
        except Exception as e:
            self.handle_error(f"Unexpected error: {str(e)}")

    def topic_callback(self, msg):
        """토픽 메시지 처리 콜백"""
        command = msg.data
        self.get_logger().info(f"Received Command: {command}")

        if command == "ON":
            # ON 명령을 처리하고, 아두이노로 스텝 값을 전송
            step_count = 1000  # 원하는 스텝 수
            self.send_serial_command(f"{step_count}")  # 아두이노로 스텝 값 전송
            self.get_logger().info(f"Turning conveyor ON with {step_count} steps")
        elif command == "OFF":
            # OFF 명령을 처리
            self.send_serial_command("STOP")  # STOP 명령을 아두이노로 전송
            self.get_logger().info("Turning conveyor OFF")
        else:
            # 알 수 없는 명령 처리
            self.get_logger().warning(f"Unknown command received: {command}")

    def send_serial_command(self, command):
        """아두이노로 시리얼 명령 전송"""
        if not self.serial_conn or not self.serial_conn.is_open:
            raise serial.SerialException("Serial connection is not open")

        # 명령 전송
        self.serial_conn.write(command.encode() + b'\n')
        self.get_logger().info(f"Sent command to Arduino: {command}")

    def check_serial_connection(self):
        """아두이노로부터 주기적으로 받는 데이터를 기반으로 연결 상태 체크"""
        try:
            if not self.serial_conn or not self.serial_conn.is_open:
                raise serial.SerialException("Serial connection lost")

            # 아두이노로부터 데이터 읽기
            if self.serial_conn.in_waiting > 0:  # 수신 대기 중인 데이터 확인
                response = self.serial_conn.read(self.serial_conn.in_waiting).decode().strip()

                # 상태가 변경되었을 때만 로그 찍도록.
                if response != self.last_status:
                    self.last_status = response  # 상태 업데이트

                    # 상태 확인
                    if response.strip(".") == "" or response.strip("_") == "":
                        # `.` 또는 `_`만 포함된 응답 처리
                        if response.startswith("."):
                            self.get_logger().info("Conveyor is idle (connection OK).")
                        elif response.startswith("_"):
                            self.get_logger().info("Conveyor is running (connection OK).")
                    else:
                        # 예상되지 않은 응답에 대해 경고
                        self.get_logger().warning(f"Unexpected response from Arduino: {response}")

                    # 정상 상태 발행
                    msg = String()
                    msg.data = "Conveyor OK"
                    self.status_publisher.publish(msg)

                else:
                    self.get_logger().warning("No data received from Arduino (possible disconnection).")

        except serial.SerialException as e:
            self.handle_error(f"Serial check error: {str(e)}")
        except Exception as e:
            self.handle_error(f"Unexpected error during check: {str(e)}")


    def handle_error(self, error_msg):
        """에러 처리 및 로깅"""
        # 에러 로그 출력
        self.get_logger().error(error_msg)
        
        # 에러 상태 발행
        msg = String()
        msg.data = f"Conveyor Error: {error_msg}"
        self.status_publisher.publish(msg)

    def __del__(self):
        """소멸자에서 시리얼 연결 종료"""
        if hasattr(self, 'serial_conn') and self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()


def main():
    rclpy.init()
    node = ConveyorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
