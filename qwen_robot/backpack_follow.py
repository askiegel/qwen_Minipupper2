#!/usr/bin/env python3

import os
import cv2
import requests
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge


VISION_URL = os.environ.get(
    "VISION_SERVER_URL",
    "http://192.168.68.100:8000/detect"
)

TARGET_LABEL = "backpack"

CENTER_DEADZONE = 70

MIN_AREA = 90000
MAX_AREA = 150000

FORWARD_SPEED = 0.06
BACKWARD_SPEED = -0.04
TURN_SPEED = 0.18

LIDAR_STOP_DISTANCE = 0.35


class BackpackFollow(Node):
    def __init__(self):
        super().__init__("backpack_follow")

        self.bridge = CvBridge()
        self.latest_frame = None
        self.front_distance = None

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        self.image_sub = self.create_subscription(
            Image,
            "/image_raw",
            self.image_callback,
            10
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/scan",
            self.scan_callback,
            10
        )

        self.timer = self.create_timer(0.35, self.loop)

        self.get_logger().info("Backpack follow started")
        self.get_logger().info(f"Vision URL: {VISION_URL}")

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().warn(f"Camera error: {e}")
            self.latest_frame = None

    def scan_callback(self, msg):
        ranges = list(msg.ranges)
        if not ranges:
            self.front_distance = None
            return

        n = len(ranges)
        center = n // 2
        window = max(5, n // 18)

        front = ranges[center - window:center + window]

        valid = [
            r for r in front
            if r is not None and r > 0.05 and r < 5.0
        ]

        self.front_distance = min(valid) if valid else None

    def stop(self):
        self.cmd_pub.publish(Twist())

    def detect(self):
        if self.latest_frame is None:
            return None

        ok, jpg = cv2.imencode(".jpg", self.latest_frame)

        if not ok:
            return None

        files = {
            "file": ("frame.jpg", jpg.tobytes(), "image/jpeg")
        }

        try:
            r = requests.post(VISION_URL, files=files, timeout=1.0)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            self.get_logger().warn(f"Vision request failed: {e}")
            return None

    def choose_target(self, detections):
        backpacks = []

        for d in detections:
            if d.get("label") != TARGET_LABEL:
                continue

            width = float(d.get("width", 0))
            height = float(d.get("height", 0))
            area = width * height

            backpacks.append((area, d))

        if not backpacks:
            return None

        backpacks.sort(reverse=True, key=lambda item: item[0])
        return backpacks[0][1]

    def loop(self):
        data = self.detect()

        if data is None:
            self.get_logger().info("Waiting for vision data")
            self.stop()
            return

        target = self.choose_target(data.get("detections", []))

        if target is None:
            self.get_logger().info("No backpack found")
            self.stop()
            return

        image_width = float(target.get("image_width", 640))
        center_x = float(target.get("center_x", image_width / 2.0))
        width = float(target.get("width", 0))
        height = float(target.get("height", 0))
        conf = float(target.get("confidence", 0))

        area = width * height
        error_x = center_x - image_width / 2.0

        cmd = Twist()

        if error_x > CENTER_DEADZONE:
            cmd.angular.z = -TURN_SPEED
        elif error_x < -CENTER_DEADZONE:
            cmd.angular.z = TURN_SPEED
        else:
            cmd.angular.z = 0.0

        if area < MIN_AREA:
            cmd.linear.x = FORWARD_SPEED
        elif area > MAX_AREA:
            cmd.linear.x = BACKWARD_SPEED
        else:
            cmd.linear.x = 0.0

        if (
            self.front_distance is not None
            and self.front_distance < LIDAR_STOP_DISTANCE
            and cmd.linear.x > 0
        ):
            self.get_logger().warn(
                f"Forward blocked by LiDAR: {self.front_distance:.2f} m"
            )
            self.stop()
            return

        self.get_logger().info(
            f"backpack conf={conf:.2f} area={area:.0f} "
            f"error_x={error_x:.0f} front={self.front_distance} "
            f"linear={cmd.linear.x:.2f} angular={cmd.angular.z:.2f}"
        )

        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = BackpackFollow()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
