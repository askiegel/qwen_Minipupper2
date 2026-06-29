import os
import time
import math

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from cv_bridge import CvBridge

from .vision_client import VisionClient
from .object_tracker import ObjectTracker
from .follow_manager import FollowManager
from .behavior_manager import BehaviorManager


class QwenRobotNode(Node):
    def __init__(self):
        super().__init__("qwen_robot")

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_count = 0
        self.loop_count = 0

        self.front_distance = None

        self.image_sub = self.create_subscription(
            Image,
            "/image_raw",
            self.image_callback,
            10,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10,
        )

        vision_url = os.getenv(
            "VISION_SERVER_URL",
            "http://192.168.68.100:8000/detect",
        )

        target_object = os.getenv(
            "TARGET_OBJECT",
            "backpack",
        )

        self.vision = VisionClient(vision_url, timeout=0.7)
        self.tracker = ObjectTracker(target_object)
        self.follow = FollowManager()
        self.behavior = BehaviorManager()

        self.get_logger().info("Qwen Robot visual follower with LiDAR safety started")
        self.get_logger().info(f"Vision URL: {vision_url}")
        self.get_logger().info(f"Target object: {target_object}")
        self.get_logger().info("Camera topic: /image_raw")
        self.get_logger().info("LiDAR topic: /scan")
        self.get_logger().info("Front LiDAR safety sector: +/- 20 degrees")

        self.timer = self.create_timer(0.5, self.update)

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )
            self.frame_count += 1

        except Exception as e:
            self.get_logger().warn(f"Camera conversion failed: {e}")
            self.latest_frame = None

    def scan_callback(self, msg):
        front_ranges = []

        for i, distance in enumerate(msg.ranges):
            if math.isinf(distance) or math.isnan(distance):
                continue

            angle = msg.angle_min + i * msg.angle_increment

            if abs(angle) <= math.radians(20):
                if msg.range_min <= distance <= msg.range_max:
                    front_ranges.append(distance)

        if front_ranges:
            self.front_distance = min(front_ranges)
        else:
            self.front_distance = None

    def update(self):
        self.loop_count += 1

        if self.latest_frame is None:
            self.get_logger().info(
                f"LOOP {self.loop_count} | NO_CAMERA_FRAME | frames={self.frame_count}"
            )
            return

        start = time.time()

        detections = self.vision.get_detections_from_frame(self.latest_frame)

        elapsed = time.time() - start

        target = self.tracker.select_target(detections)

        cmd, motion_state = self.follow.compute_cmd(
            target,
            front_distance=self.front_distance,
        )

        state = self.behavior.update(
            target,
            self.front_distance,
            motion_state,
        )

        self.cmd_pub.publish(cmd)

        lidar_text = (
            f"{self.front_distance:.2f}m"
            if self.front_distance is not None
            else "None"
        )

        if target is None:
            labels = [d.get("label") for d in detections if d.get("label")]
            self.get_logger().info(
                f"LOOP {self.loop_count} | {state} | "
                f"frames={self.frame_count} | "
                f"seen={labels} | "
                f"front={lidar_text} | "
                f"vision_time={elapsed:.2f}s"
            )
            return

        self.get_logger().info(
            f"LOOP {self.loop_count} | "
            f"{state} | "
            f"{target['label']} | "
            f"conf={target['confidence']:.2f} | "
            f"cx={target['cx']:.1f} | "
            f"area={target['area']:.0f} | "
            f"front={lidar_text} | "
            f"vx={cmd.linear.x:.2f} | "
            f"wz={cmd.angular.z:.2f} | "
            f"vision_time={elapsed:.2f}s"
        )


def main(args=None):
    rclpy.init(args=args)

    node = QwenRobotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    stop = Twist()
    node.cmd_pub.publish(stop)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
