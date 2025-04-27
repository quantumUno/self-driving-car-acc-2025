  # Data Collection
 This section outlines our approach to collecting sensor data from the QCar 2 digital twin in Quanser Interactive Labs (QLabs).

  ## Why This Matters
  Sensor data enables the QCar to perceive its environment, critical for autonomous driving.

  ## Setup in Self-Driving Car Studio
  - Peripherals: Configured camera (RGB, 30 fps), LiDAR (point clouds), IMU (acceleration, angular velocity), and encoders (wheel speed).
  - Map: Used Detailed Scenario’s road network.
  - Verification: Subscribed to ROS 2 topics (e.g., `/qcar/rgb_camera`).

  ## Prerequisites
  - ROS 2 Humble, Python 3.8+, NumPy, OpenCV.
  - QLabs with QCar 2 digital twin.
  - ROS packages: `sensor_msgs`, `geometry_msgs`.

  ## Installation
  1. Install dependencies (see main README).
  2. Download this repository.

  ## Code Walkthrough
  `code/data_collection_node.py` subscribes to sensor topics and saves data.

  ### Example Code
  ```python
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
          cv_image = self.bridge.imgmsg_to_cv2(image_msg, 'bgr8')
          cv2.imwrite(f'{self.output_dir}/rgb_image_{self.count:03d}.png', cv_image)
          self.count += 1

  def main():
      rclpy.init()
      node = DataCollectionNode()
      rclpy.spin(node)
      rclpy.shutdown()

  if __name__ == '__main__':
      main()
  ```

  ## Running the Code
  1. Source ROS 2 workspace.
  2. Run: `ros2 run qcar_control data_collection_node`.
  3. Outputs saved to `examples/`.

  ## Results
  - `examples/rgb_image_001.png`: Sample RGB frame.
  - `examples/map_screenshot.png`: Detailed Scenario map.

  ## Notes
  - Optimized for Detailed Scenario.
  - High-frequency data capture for dynamic environments.

