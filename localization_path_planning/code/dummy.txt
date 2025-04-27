import rclpy
     from rclpy.node import Node
     from nav2_msgs.msg import Path
     from geometry_msgs.msg import PoseStamped
     import numpy as np

     class PathPlanningNode(Node):
         def __init__(self):
             super().__init__('path_planning_node')
             self.path_pub = self.create_publisher(Path, '/qcar/path', 10)
             self.timer = self.create_timer(1.0, self.plan_path)

         def plan_path(self):
             # Simplified A* (placeholder)
             path = Path()
             path.header.frame_id = 'map'
             for i in range(5):
                 pose = PoseStamped()
                 pose.header.frame_id = 'map'
                 pose.pose.position.x = float(i)
                 pose.pose.position.y = 0.0
                 path.poses.append(pose)
             self.path_pub.publish(path)
             self.get_logger().info('Published path')

     def main():
         rclpy.init()
         node = PathPlanningNode()
         rclpy.spin(node)
         rclpy.shutdown()

     if __name__ == '__main__':
         main()
