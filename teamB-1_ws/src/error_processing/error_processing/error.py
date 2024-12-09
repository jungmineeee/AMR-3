import rclpy
from rclpy.node import Node
from system_interface.srv import ReportError
import smtplib
from email.mime.text import MIMEText

class ErrorHandling(Node):
    def __init__(self):
        super().__init__('error_handling')
        
        # 서비스 서버 생성
        self.email_service = self.create_service(
            ReportError,
            'report_error',
            self.handle_service_request
        )
        
        # ROS2 파라미터 초기화 / 사용 환경에 따라 변경
        self.declare_parameter('email_address', "u0landuu@gmail.com")
        self.declare_parameter('smtp_server', 'smtp.gmail.com')
        self.declare_parameter('smtp_port', 587)
        self.declare_parameter('email_user', 'aq3480@gmail.com')
        self.declare_parameter('email_password', 'bwxvnntmpazxbtzn')
    
        self.get_logger().info('Error Handling Service is ready.')


    def handle_service_request(self, request, response):
        """서비스 요청 처리"""
        try:
            # 시스템 에러 종류
            error = request.error_msg
            
            # ROS2 파라미터 가져오기
            email_address = self.get_parameter('email_address').get_parameter_value().string_value
            smtp_server = self.get_parameter('smtp_server').get_parameter_value().string_value
            smtp_port = self.get_parameter('smtp_port').get_parameter_value().integer_value
            email_user = self.get_parameter('email_user').get_parameter_value().string_value
            email_password = self.get_parameter('email_password').get_parameter_value().string_value

            # 이메일 전송
            self.send_email(smtp_server, smtp_port, email_user, email_password, email_address, error)

            response.success = True
            response.message = f'Error email sent to {email_address}'
            
        except Exception as e:
            self.get_logger().error(f"Failed to send email: {e}")
            response.success = False
            response.message = f'Failed to send email: {str(e)}'
            
        return response

    def send_email(self, smtp_server, smtp_port, email_user, email_password, email_address, error):
        """이메일 전송 함수"""
        self.get_logger().info(f'Sending email to {email_address}')
        subject = 'Error Notification'
        body = f'[ An error has occurred in your ROS2 system. ]\n\nError Type : \n    - {error}'

        # 이메일 메시지 생성
        msg = MIMEText(body)
        msg['Subject'] = subject
        msg['From'] = email_user
        msg['To'] = email_address

        # SMTP 서버 연결 및 이메일 전송
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()  # TLS 활성화
            server.login(email_user, email_password)  # 로그인
            server.sendmail(email_user, email_address, msg.as_string())  # 이메일 전송
            self.get_logger().info(f'Email sent successfully to {email_address}')

######################################################################
def main(args=None):
    rclpy.init(args=args)
    node = ErrorHandling()
    
    try:
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        node.get_logger().info('Node stopped by user.')
        
    finally:
        node.destroy_node()
        rclpy.shutdown()

######################################################################
if __name__ == '__main__':
    main()
