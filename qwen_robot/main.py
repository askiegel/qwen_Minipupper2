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

        self.last_command = 'none'
        self.last_action = 'none'
        self.last_plan = {}
        self.last_vision = {
            "objects": [],
            "description": "No vision result yet."
        }

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

        self.get_logger().info('Qwen Robot started with vision client.')
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
            'last_vision': self.last_vision,
            'front_distance': self.lidar.front_distance,
            'camera_ready': self.camera_ready,
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
            self.handle_move(plan)

        elif action == 'TURN':
            self.handle_turn(plan)

        elif action == 'STOP':
            self.handle_stop()

        elif action == 'PICTURE':
            self.handle_picture()

        elif action == 'STATUS':
            self.handle_status()

        elif action == 'VISION':
            self.handle_vision()

        else:
            self.get_logger().warn('Unknown command.')

        self.publish_status()

    def handle_move(self, plan):
        direction = plan.get('direction', 'forward')
        speed = float(plan.get('speed', 0.05))
        duration = float(plan.get('duration', 1.0))

        speed = max(0.0, min(speed, 0.10))
        duration = max(0.1, min(duration, 3.0))

        if direction == 'forward':
            if self.lidar.blocked(0.35):
                self.motion_state = 'blocked'
                self.get_logger().warn(
                    f'Blocked. Front obstacle at {self.lidar.front_distance:.2f} m.'
                )
                self.motion.stop()
                return

            self.motion_state = 'moving_forward'
            self.get_logger().info(f'Moving forward: speed={speed}, duration={duration}')
            self.motion.move(linear_x=speed, angular_z=0.0, duration=duration)
            self.motion_state = 'stopped'

        elif direction == 'backward':
            self.motion_state = 'moving_backward'
            self.get_logger().info(f'Moving backward: speed={speed}, duration={duration}')
            self.motion.move(linear_x=-speed, angular_z=0.0, duration=duration)
            self.motion_state = 'stopped'

        else:
            self.get_logger().warn(f'Unknown move direction: {direction}')

    def handle_turn(self, plan):
        direction = plan.get('direction', 'left')
        speed = float(plan.get('speed', 0.4))
        duration = float(plan.get('duration', 1.0))

        speed = max(0.0, min(speed, 0.8))
        duration = max(0.1, min(duration, 3.0))

        if direction == 'left':
            self.motion_state = 'turning_left'
            self.get_logger().info(f'Turning left: speed={speed}, duration={duration}')
            self.motion.move(linear_x=0.0, angular_z=speed, duration=duration)
            self.motion_state = 'stopped'

        elif direction == 'right':
            self.motion_state = 'turning_right'
            self.get_logger().info(f'Turning right: speed={speed}, duration={duration}')
            self.motion.move(linear_x=0.0, angular_z=-speed, duration=duration)
            self.motion_state = 'stopped'

        else:
            self.get_logger().warn(f'Unknown turn direction: {direction}')

    def handle_stop(self):
        self.motion_state = 'stopped'
        self.get_logger().info('Stopping.')
        self.motion.stop()

    def handle_picture(self):
        filename = self.camera.save()

        if filename is None:
            self.get_logger().warn('No camera image received yet.')
        else:
            self.get_logger().info(f'Saved image to {filename}')

    def handle_status(self):
        self.publish_status()

        if self.lidar.front_distance is None:
            self.get_logger().info('No LiDAR distance available yet.')
        else:
            self.get_logger().info(
                f'Closest object in front: {self.lidar.front_distance:.2f} m'
            )

    def handle_vision(self):
        frame = self.camera.get_latest()
        self.get_logger().info('Sending frame to vision server...')

        result = self.vision.analyze_frame(frame)
        self.last_vision = result

        self.get_logger().info(
            f"Vision: {result.get('description', 'No description')}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = QwenRobot()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
