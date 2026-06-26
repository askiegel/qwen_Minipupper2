import json

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from std_msgs.msg import String
from sensor_msgs.msg import LaserScan, Image
from cv_bridge import CvBridge

from .motion import MotionController
from .lidar import LidarSafety
from .camera import CameraManager
from .planner import Planner
from .vision_client import VisionClient
from .memory import RobotMemory
from .robot_controller import RobotController
from .config import VISION_SERVER_URL


class QwenRobot(Node):
    def __init__(self):
        super().__init__('qwen_robot')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.status_pub = self.create_publisher(String, '/qwen_robot/status', 10)

        self.motion = MotionController(self.cmd_pub)
        self.lidar = LidarSafety()
        self.camera = CameraManager(CvBridge())
        self.planner = Planner()
        self.vision = VisionClient(VISION_SERVER_URL)
        self.memory = RobotMemory()

        self.controller = RobotController(
            motion=self.motion,
            lidar=self.lidar,
            camera=self.camera,
            vision=self.vision,
            memory=self.memory,
            logger=self.get_logger()
        )

        self.last_command = 'none'
        self.last_action = 'none'
        self.last_plan = {}
        self.last_message = 'none'
        self.last_vision = {
            "objects": [],
            "description": "No vision result yet."
        }
        self.last_memory = {}
        self.last_compare = {}

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

        self.get_logger().info('Qwen Robot started with controller refactor.')
        self.get_logger().info('Command topic: /qwen_robot/command')
        self.get_logger().info('Status topic: /qwen_robot/status')
        self.get_logger().info('LiDAR topic: /scan')
        self.get_logger().info('Camera topic: /image_raw')
        self.get_logger().info(f'Vision server: {VISION_SERVER_URL}')

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
            'last_action': self.last_action,
            'last_plan': self.last_plan,
            'last_message': self.last_message,
            'last_vision': self.last_vision,
            'last_memory': self.last_memory,
            'last_compare': self.last_compare,
            'front_distance': self.lidar.front_distance,
            'camera_ready': self.camera_ready,
            'memory_count': self.memory.count(),
        }

        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)

    def command_callback(self, msg):
        text = msg.data.lower().strip()
        plan = self.planner.plan(text)

        self.last_command = text
        self.last_plan = plan
        self.last_action = plan.get('action', 'UNKNOWN')

        self.get_logger().info(f'Received: {text}')
        self.get_logger().info(f'Plan: {plan}')
        self.get_logger().info(f'Front distance: {self.lidar.front_distance}')

        action = plan.get('action', 'UNKNOWN')

        if action == 'MOVE':
            self.motion_state, self.last_message = self.controller.handle_move(plan)

        elif action == 'TURN':
            self.motion_state, self.last_message = self.controller.handle_turn(plan)

        elif action == 'STOP':
            self.motion_state, self.last_message = self.controller.handle_stop()

        elif action == 'PICTURE':
            self.motion_state, self.last_message = self.controller.handle_picture()

        elif action == 'STATUS':
            self.motion_state, self.last_message = self.controller.handle_status()

        elif action == 'VISION':
            self.motion_state, self.last_message, self.last_vision, observation = (
                self.controller.handle_vision()
            )
            self.last_memory = self.memory.summary()

        elif action == 'MEMORY':
            self.motion_state, self.last_memory = self.controller.handle_memory()
            self.last_message = 'Memory summary updated.'

        elif action == 'COMPARE':
            self.motion_state, self.last_compare = self.controller.handle_compare()
            self.last_message = self.last_compare.get('message', 'Comparison complete.')

        else:
            self.last_message = 'Unknown command.'
            self.get_logger().warn('Unknown command.')

        self.get_logger().info(f'Message: {self.last_message}')
        self.publish_status()


def main(args=None):
    rclpy.init(args=args)
    node = QwenRobot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
