import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan

from .motion import MotionController
from .command_parser import CommandParser
from .lidar import LidarSafety


class QwenRobot(Node):
    def __init__(self):
        super().__init__('qwen_robot')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.motion = MotionController(self.cmd_pub)
        self.parser = CommandParser()
        self.lidar = LidarSafety()

        self.command_sub = self.create_subscription(
            String,
            '/qwen_robot/command',
            self.command_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.get_logger().info('Qwen Robot command node with LiDAR safety started.')
        self.get_logger().info('Listening for commands on /qwen_robot/command')
        self.get_logger().info('Listening for LiDAR on /scan')

    def scan_callback(self, msg):
        self.lidar.update(msg)

    def command_callback(self, msg):
        text = msg.data
        command = self.parser.parse(text)

        self.get_logger().info(f'Received: {text}')
        self.get_logger().info(f'Parsed: {command}')
        self.get_logger().info(f'Front distance: {self.lidar.front_distance}')

        if command == 'FORWARD':
            if self.lidar.blocked(0.35):
                self.get_logger().warn(
                    f'Blocked. Front obstacle at {self.lidar.front_distance:.2f} m.'
                )
                self.motion.stop()
            else:
                self.get_logger().info('Path clear. Moving forward.')
                self.motion.move(linear_x=0.05, angular_z=0.0, duration=1.0)

        elif command == 'BACKWARD':
            self.get_logger().info('Moving backward.')
            self.motion.move(linear_x=-0.05, angular_z=0.0, duration=1.0)

        elif command == 'LEFT':
            self.get_logger().info('Turning left.')
            self.motion.move(linear_x=0.0, angular_z=0.4, duration=1.0)

        elif command == 'RIGHT':
            self.get_logger().info('Turning right.')
            self.motion.move(linear_x=0.0, angular_z=-0.4, duration=1.0)

        elif command == 'STOP':
            self.get_logger().info('Stopping.')
            self.motion.stop()

        else:
            self.get_logger().warn('Unknown command.')


def main(args=None):
    rclpy.init(args=args)
    node = QwenRobot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
