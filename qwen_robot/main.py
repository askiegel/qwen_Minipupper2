import json

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge

from .motion import MotionController
from .command_parser import CommandParser
from .lidar import LidarSafety
from .camera import CameraManager


class QwenRobot(Node):
    def __init__(self):
        super().__init__('qwen_robot')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/qwen_robot/status', 10)

        self.motion = MotionController(self.cmd_pub)
        self.parser = CommandParser()
        self.lidar = LidarSafety()
        self.camera = CameraManager(CvBridge())

        self.last_command = 'none'
        self.motion_state = 'stopped'
        self.camera_ready = False

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

        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )

        self.status_timer = self.create_timer(1.0, self.publish_status)

        self.get_logger().info('Qwen Robot started.')
        self.get_logger().info('Command topic: /qwen_robot/command')
        self.get_logger().info('Status topic: /qwen_robot/status')
        self.get_logger().info('LiDAR topic: /scan')
        self.get_logger().info('Camera topic: /image_raw')

    def scan_callback(self, msg):
        self.lidar.update(msg)

    def image_callback(self, msg):
        try:
            self.camera.update(msg)
            self.camera_ready = True
        except Exception as e:
            self.camera_ready = False
            self.get_logger().warn(f'Camera error: {e}')

    def publish_status(self):
        status = {
            'motion_state': self.motion_state,
            'last_command': self.last_command,
            'front_distance': self.lidar.front_distance,
            'camera_ready': self.camera_ready,
        }

        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)

    def command_callback(self, msg):
        text = msg.data.lower().strip()
        command = self.parser.parse(text)

        self.last_command = text

        self.get_logger().info(f'Received: {text}')
        self.get_logger().info(f'Parsed: {command}')
        self.get_logger().info(f'Front distance: {self.lidar.front_distance}')

        if command == 'FORWARD':
            if self.lidar.blocked(0.35):
                self.motion_state = 'blocked'
                self.get_logger().warn(
                    f'Blocked. Front obstacle at {self.lidar.front_distance:.2f} m.'
                )
                self.motion.stop()
            else:
                self.motion_state = 'moving_forward'
                self.get_logger().info('Path clear. Moving forward.')
                self.motion.move(linear_x=0.05, angular_z=0.0, duration=1.0)
                self.motion_state = 'stopped'

        elif command == 'BACKWARD':
            self.motion_state = 'moving_backward'
            self.get_logger().info('Moving backward.')
            self.motion.move(linear_x=-0.05, angular_z=0.0, duration=1.0)
            self.motion_state = 'stopped'

        elif command == 'LEFT':
            self.motion_state = 'turning_left'
            self.get_logger().info('Turning left.')
            self.motion.move(linear_x=0.0, angular_z=0.4, duration=1.0)
            self.motion_state = 'stopped'

        elif command == 'RIGHT':
            self.motion_state = 'turning_right'
            self.get_logger().info('Turning right.')
            self.motion.move(linear_x=0.0, angular_z=-0.4, duration=1.0)
            self.motion_state = 'stopped'

        elif command == 'STOP':
            self.motion_state = 'stopped'
            self.get_logger().info('Stopping.')
            self.motion.stop()

        elif 'picture' in text or 'photo' in text or 'camera' in text or 'image' in text:
            filename = self.camera.save()
            if filename is None:
                self.get_logger().warn('No camera image received yet.')
            else:
                self.get_logger().info(f'Saved image to {filename}')

        elif 'status' in text or 'distance' in text or 'wall' in text:
            self.publish_status()
            if self.lidar.front_distance is None:
                self.get_logger().info('No LiDAR distance available yet.')
            else:
                self.get_logger().info(
                    f'Closest object in front: {self.lidar.front_distance:.2f} m'
                )

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
