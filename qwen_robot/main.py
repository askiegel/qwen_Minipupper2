import os
import time
import math
import json

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import String
from cv_bridge import CvBridge

from .vision_client import VisionClient
from .object_tracker import ObjectTracker
from .follow_manager import FollowManager
from .behavior_manager import BehaviorManager
from .mission_manager import MissionManager
from .search_behavior import SearchBehavior
from .navigation_manager import NavigationManager


class QwenRobotNode(Node):
    def __init__(self):
        super().__init__("qwen_robot")

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self.status_pub = self.create_publisher(String, "/qwen_status", 10)

        self.mission_sub = self.create_subscription(
            String,
            "/qwen_mission",
            self.mission_callback,
            10,
        )

        self.bridge = CvBridge()

        self.latest_frame = None
        self.frame_count = 0
        self.loop_count = 0
        self.front_distance = None

        self.last_cmd = Twist()
        self.last_target_label = "none"
        self.last_state = "IDLE"
        self.last_vision_time = 0.0
        self.last_navigation = {}

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

        self.vision_url = os.getenv(
            "VISION_SERVER_URL",
            "http://192.168.68.100:8000/detect",
        )

        mission_name = os.getenv("MISSION", "FOLLOW_PERSON")
        self.mission = MissionManager(mission_name)

        target_object = self.mission.get_target_object() or "person"

        self.vision = VisionClient(self.vision_url, timeout=0.7)
        self.tracker = ObjectTracker(target_object)
        self.follow = FollowManager()
        self.behavior = BehaviorManager()
        self.search = SearchBehavior()
        self.navigation = NavigationManager()

        self.get_logger().info("Qwen Robot Dashboard Pro mission follower started")
        self.get_logger().info(f"Vision URL: {self.vision_url}")
        self.get_logger().info(f"Mission: {self.mission.status_text()}")
        self.get_logger().info(f"Target object: {target_object}")
        self.get_logger().info("Navigation Manager telemetry enabled")

        self.timer = self.create_timer(0.5, self.update)
        self.status_timer = self.create_timer(0.25, self.publish_status)

    def mission_callback(self, msg):
        mission = msg.data.strip().upper()

        self.mission.set_mission(mission)

        target = self.mission.get_target_object()
        if target is not None:
            self.tracker.set_target(target)

        self.search.reset()
        self.navigation.reset()

        stop = Twist()
        self.cmd_pub.publish(stop)
        self.last_cmd = stop

        self.get_logger().info(
            f"Mission changed to {self.mission.status_text()} target={target}"
        )

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

        self.front_distance = min(front_ranges) if front_ranges else None

    def estimate_bearing_deg(self, target):
        if target is None:
            return None

        try:
            cx = float(target.get("cx"))
            image_width = 640.0
            normalized = (cx - image_width / 2.0) / (image_width / 2.0)
            camera_half_fov_deg = 30.0
            return round(normalized * camera_half_fov_deg, 2)
        except Exception:
            return None

    def update_navigation(self, target):
        try:
            tracker_tel = self.tracker.telemetry()
        except Exception:
            tracker_tel = {}

        target_visible = target is not None
        target_id = tracker_tel.get("target_id")
        last_seen_age = tracker_tel.get("target_last_seen_age")

        target_range = self.front_distance if target_visible else None
        target_bearing = self.estimate_bearing_deg(target)

        confidence = 0.0
        if target is not None:
            confidence = float(target.get("confidence", target.get("conf", 0.0)))

        self.last_navigation = self.navigation.update(
            target_visible=target_visible,
            target_id=str(target_id) if target_id is not None else None,
            last_seen_age_s=last_seen_age,
            last_seen_range_m=target_range,
            last_seen_bearing_deg=target_bearing,
            confidence=confidence,
        )

        return self.last_navigation

    def update(self):
        self.loop_count += 1

        if not self.mission.should_move():
            stop = Twist()
            self.cmd_pub.publish(stop)

            self.last_cmd = stop
            self.last_state = "IDLE"
            self.last_target_label = "none"
            self.last_navigation = self.navigation.reset()
            return

        if self.latest_frame is None:
            self.last_state = "NO_CAMERA_FRAME"
            return

        start = time.time()
        detections = self.vision.get_detections_from_frame(self.latest_frame)
        self.last_vision_time = time.time() - start

        target = self.tracker.select_target(detections)
        navigation = self.update_navigation(target)

        cmd, motion_state = self.follow.compute_cmd(
            target,
            front_distance=self.front_distance,
        )

        if target is None and self.mission.status_text() in (
            "FIND_BACKPACK",
            "FOLLOW_PERSON",
            "FOLLOW",
        ):
            cmd, motion_state = self.search.update(
                cmd,
                front_distance=self.front_distance,
            )
        else:
            self.search.reset()

        state = self.behavior.update(
            self.mission.status_text(),
            target,
            self.front_distance,
            motion_state,
        )

        self.cmd_pub.publish(cmd)

        self.last_cmd = cmd
        self.last_state = state
        self.last_target_label = target["label"] if target else "none"

        lidar_text = (
            f"{self.front_distance:.2f}m"
            if self.front_distance is not None
            else "None"
        )

        nav_state = navigation.get("state", "UNKNOWN")
        nav_action = navigation.get("action", "UNKNOWN")

        if target is None:
            labels = [d.get("label") for d in detections if d.get("label")]
            self.get_logger().info(
                f"LOOP {self.loop_count} | "
                f"MISSION={self.mission.status_text()} | "
                f"STATE={state} | "
                f"NAV={nav_state}/{nav_action} | "
                f"seen={labels} | "
                f"front={lidar_text} | "
                f"vx={cmd.linear.x:.2f} | "
                f"wz={cmd.angular.z:.2f} | "
                f"vision_time={self.last_vision_time:.2f}s"
            )
            return

        self.get_logger().info(
            f"LOOP {self.loop_count} | "
            f"MISSION={self.mission.status_text()} | "
            f"STATE={state} | "
            f"NAV={nav_state}/{nav_action} | "
            f"{target['label']} | "
            f"conf={target['confidence']:.2f} | "
            f"cx={target['cx']:.1f} | "
            f"area={target['area']:.0f} | "
            f"front={lidar_text} | "
            f"vx={cmd.linear.x:.2f} | "
            f"wz={cmd.angular.z:.2f} | "
            f"vision_time={self.last_vision_time:.2f}s"
        )

    def publish_status(self):
        status = {
            "mission": self.mission.status_text(),
            "state": self.last_state,
            "target": self.last_target_label,
            "front_distance": self.front_distance,
            "linear_x": self.last_cmd.linear.x,
            "angular_z": self.last_cmd.angular.z,
            "frames": self.frame_count,
            "loops": self.loop_count,
            "vision_time": self.last_vision_time,
            "navigation": self.last_navigation,
        }

        try:
            status.update(self.tracker.telemetry())
        except Exception as e:
            status.update({
                "target_state": "UNKNOWN",
                "target_id": None,
                "target_label": self.last_target_label,
                "target_confidence": 0.0,
                "target_similarity": 0.0,
                "target_cx": None,
                "target_cy": None,
                "target_area": None,
                "target_lost_time": 0.0,
                "target_last_seen_age": 999.0,
                "target_telemetry_error": str(e),
            })

        msg = String()
        msg.data = json.dumps(status)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)

    node = QwenRobotNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    try:
        stop = Twist()
        node.cmd_pub.publish(stop)
    except Exception:
        pass

    node.destroy_node()

    try:
        rclpy.shutdown()
    except Exception:
        pass


if __name__ == "__main__":
    main()
