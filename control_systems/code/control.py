
 import rclpy
     from rclpy.node import Node
     from geometry_msgs.msg import Twist
     import numpy as np

     class ControlNode(Node):
         def __init__(self):
             super().__init__('control_node')
             self.cmd_pub = self.create_publisher(Twist, '/qcar/cmd_vel', 10)
             self.target_speed = 0.5  # m/s
             self.kp = 1.0  # PID gain
             self.red_light_detected = False  # Placeholder for traffic light state
             self.timer = self.create_timer(0.1, self.control_callback)

         def control_callback(self):
             cmd = Twist()
             if self.red_light_detected:
                 cmd.linear.x = 0.0  # Stop at red light
                 cmd.angular.z = 0.0
             else:
                 # Stanley controller (simplified)
                 cross_track_error = 0.1  # Placeholder: Get from interpretation
                 heading_error = 0.05  # Placeholder
                 steering = np.arctan2(self.kp * cross_track_error, self.target_speed)
                 cmd.linear.x = self.target_speed
                 cmd.angular.z = steering
             self.cmd_pub.publish(cmd)
             self.get_logger().info('Published control command')

     def main():
         rclpy.init()
         node = ControlNode()
         rclpy.spin(node)
         rclpy.shutdown()

     if __name__ == '__main__':
         main()
