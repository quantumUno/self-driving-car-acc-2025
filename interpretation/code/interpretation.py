 import rclpy
     from rclpy.node import Node
     from sensor_msgs.msg import Image
     from cv_bridge import CvBridge
     import cv2
     import numpy as np
     import os

     class InterpretationNode(Node):
         def __init__(self):
             super().__init__('interpretation_node')
             self.bridge = CvBridge()
             self.image_sub = self.create_subscription(
                 Image, '/qcar/rgb_camera', self.image_callback, 10)
             self.output_dir = 'interpretation_output'
             os.makedirs(self.output_dir, exist_ok=True)
             self.count = 0

         def image_callback(self, msg):
             cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
             # Lane detection
             gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
             edges = cv2.Canny(gray, 100, 200)
             lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
             if lines is not None:
                 for line in lines:
                     x1, y1, x2, y2 = line[0]
                     cv2.line(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
             # Traffic light detection (basic color-based)
             hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
             red_mask = cv2.inRange(hsv, np.array([0, 120, 70]), np.array([10, 255, 255]))
             if cv2.countNonZero(red_mask) > 1000:
                 cv2.putText(cv_image, 'Red Light', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
             # Save output
             cv2.imwrite(f'{self.output_dir}/lane_detection_{self.count:03d}.png', cv_image)
             self.get_logger().info(f'Processed image {self.count}')
             self.count += 1

     def main():
         rclpy.init()
         node = InterpretationNode()
         rclpy.spin(node)
         rclpy.shutdown()

     if __name__ == '__main__':
         main()
