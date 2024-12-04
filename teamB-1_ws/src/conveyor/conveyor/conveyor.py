import serial
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class ControlConveyor(Node):
    def __init__(self):
        super().__init__('control_conveyor')
        self.get_logger().info("ControlConveyor 노드 시작")
        
        # 시그널 서브스크라이버 생성
        self.subscription = self.create_subscription(
            String, 
            '/gui_topic', 
            self.listener_callback, 
            10
        )
        self.subscription
        
    def listener_callback(self, msg):
        """메시지를 수신하면 호출되는 콜백 함수"""
        received_data = msg.data
        self.get_logger().info(f'Subscribe한 시그널 : {received_data}')
        
        # 값 수신 후 함수 실행
        self.process_message(received_data) 

    def process_message(self, data):
        """받은 데이터를 활용한 추가 처리"""
        if data == "ON":
            self.get_logger().info("Starting process!")
            # 원하는 동작 추가
            
        elif data == "OFF":
            self.get_logger().info("Stopping process!")
            # 원하는 동작 추가
        else:
            self.get_logger().info(f"Unknown command: {data}")

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
            