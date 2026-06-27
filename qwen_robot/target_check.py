#!/usr/bin/env python3

import os
import cv2
import requests
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


VISION_URL = os.environ.get(
    "VISION_SERVER_URL",
    "http://192.168.68.100:8000/detect"
)


class VisionCheck(Node):
    def __init__(self):
        super().__init__("vision_check")

        self.bridge = CvBridge()
        self.latest_frame = None

        self.image_sub = self.create_subscription(
            Image,
            "/image_raw",
            self.image_callback,
            10
        )

        self.timer = self.create_timer(1.0, self.loop)

        self.get_logger().info("Vision check started")
        self.get_logger().info(f"Vision URL: {VISION_URL}")
        self.get_logger().info("Waiting for /image_raw...")

    def image_callback(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().warn(f"Camera conversion error: {e}")

    def loop(self):
        if self.latest_frame is None:
            self.get_logger().info("No camera frame yet")
            return

        ok, jpg = cv2.imencode(".jpg", self.latest_frame)

        if not ok:
            self.get_logger().warn("Could not encode frame")
            return

        files = {
            "file": ("frame.jpg", jpg.tobytes(), "image/jpeg")
        }

        try:
            r = requests.post(VISION_URL, files=files, timeout=1.0)
            r.raise_for_status()
            data = r.json()
        except Exception as e:
            self.get_logger().warn(f"Vision request failed: {e}")
            return

        detections = data.get("detections", [])

        if not detections:
            self.get_logger().info("No objects detected")
            return

        self.get_logger().info("Detections:")

        for d in detections:
            label = d.get("label", "unknown")
            conf = float(d.get("confidence", 0.0))
            cx = d.get("center_x", "?")
            cy = d.get("center_y", "?")
            w = d.get("width", "?")
            h = d.get("height", "?")

            self.get_logger().info(
                f"label={label} conf={conf:.2f} "
                f"center=({cx},{cy}) size=({w}x{h})"
            )


def main(args=None):
    rclpy.init(args=args)
    node = VisionCheck()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
