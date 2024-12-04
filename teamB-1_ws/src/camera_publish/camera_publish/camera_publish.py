import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2

class WebcamPublisher(Node):
    def __init__(self):
        super().__init__('webcam_publisher')
        
        # 이미지 퍼블리셔 생성(토픽 타입 CompressedImage)
        self.publisher_ = self.create_publisher(CompressedImage, '/webcam_image', 10)
        
        self.timer = self.create_timer(0.1, self.publish_webcam_image)
        
        # OpenCV VideoCapture for webcam (0: default camera)
        self.cap = cv2.VideoCapture(2)
        
        if not self.cap.isOpened():
            self.get_logger().error("Could not open webcam.")
            raise RuntimeError("Webcam not accessible.")
        
        self.get_logger().info("WebcamPublisher 노드 시작")
    
    def publish_webcam_image(self):
        ret, frame = self.cap.read()
        if not ret:
            self.get_logger().warn("Failed to capture image from webcam.")
            return
        
        # Compress the frame as JPEG
        success, compressed_image = cv2.imencode('.jpg', frame)
        if not success:
            self.get_logger().error("Failed to compress webcam frame.")
            return
        
        # Create CompressedImage message
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.format = 'jpeg'
        msg.data = compressed_image.tobytes()
        
        # Publish the compressed image
        self.publisher_.publish(msg)
        self.get_logger().info("Published webcam compressed image.")
    
    def destroy_node(self):
        # Release the webcam resource
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    
    node = WebcamPublisher()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node stopped cleanly.")
    except BaseException as e:
        node.get_logger().fatal(f"Exception in node: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
