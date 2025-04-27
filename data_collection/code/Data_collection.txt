  import rclpy
     from rclpy.node import Node
     from sensor_msgs.msg import Image, PointCloud2
     import message_filters
     import cv2
     from cv_bridge import CvBridge
     import os

     class DataCollectionNode(Node):
         def __init__(self):
             super().__init__('data_collection_node')
             self.bridge = CvBridge()
             self.image_sub = message_filters.Subscriber(self, Image, '/qcar/rgb_camera')
             self.lidar_sub = message_filters.Subscriber(self, PointCloud2, '/qcar/lidar')
             self.ts = message_filters.ApproximateTimeSynchronizer(
                 [self.image_sub, self.lidar_sub], queue_size=10, slop=0.1)
             self.ts.registerCallback(self.callback)
             self.count = 0
             self.output_dir = 'data_output'
             os.makedirs(self.output_dir, exist_ok=True)

         def callback(self, image_msg, lidar_msg):
             self.get_logger().info('Received synchronized image and LiDAR data')
             # Save image
             cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
             cv2.imwrite(f'{self.output_dir}/rgb_image_{self.count:03d}.png', cv_image)
             # Save LiDAR (placeholder; requires PCL processing)
             self.get_logger().info(f'Saved image {self.count}')
             self.count += 1

     def main():
         rclpy.init()
         node = DataCollectionNode()
         rclpy.spin(node)
         rclpy.shutdown()

     if __name__ == '__main__':
         main()
