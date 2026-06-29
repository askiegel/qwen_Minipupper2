import os
import time
import json
import threading

import cv2
import rclpy
import requests

from flask import Flask, Response, request, jsonify
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge


app = Flask(__name__)

latest_jpeg = None
latest_status = {
    "mission": "UNKNOWN",
    "state": "UNKNOWN",
    "target": "none",
    "front_distance": None,
    "linear_x": 0.0,
    "angular_z": 0.0,
    "vision_time": 0.0,
}

frame_lock = threading.Lock()
status_lock = threading.Lock()
dashboard_node = None


class DashboardNode(Node):
    def __init__(self):
        super().__init__("qwen_dashboard")

        self.bridge = CvBridge()

        self.vision_url = os.getenv(
            "VISION_SERVER_URL",
            "http://192.168.68.100:8000/detect",
        )

        self.mission_pub = self.create_publisher(String, "/qwen_mission", 10)

        self.status_sub = self.create_subscription(
            String,
            "/qwen_status",
            self.status_callback,
            10,
        )

        self.image_sub = self.create_subscription(
            Image,
            "/image_raw",
            self.image_callback,
            10,
        )

        self.last_detect_time = 0.0
        self.detect_interval = 0.35
        self.latest_detections = []

        self.get_logger().info("Dashboard Pro started")
        self.get_logger().info(f"Vision URL: {self.vision_url}")

    def status_callback(self, msg):
        global latest_status

        try:
            data = json.loads(msg.data)
            with status_lock:
                latest_status = data
        except Exception:
            pass

    def publish_mission(self, mission):
        msg = String()
        msg.data = mission
        self.mission_pub.publish(msg)
        self.get_logger().info(f"Dashboard mission command: {mission}")

    def call_vision_server(self, frame):
        try:
            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                return []

            files = {
                "file": (
                    "frame.jpg",
                    encoded.tobytes(),
                    "image/jpeg",
                )
            }

            response = requests.post(
                self.vision_url,
                files=files,
                timeout=0.8,
            )

            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict):
                return data.get("detections", [])

            if isinstance(data, list):
                return data

            return []

        except Exception:
            return []

    def draw_boxes(self, frame, detections):
        with status_lock:
            status = dict(latest_status)

        active_target = status.get("target", "none")

        for det in detections:
            label = det.get("label", "object")
            confidence = float(det.get("confidence", 0.0))

            x1 = det.get("x1")
            y1 = det.get("y1")
            x2 = det.get("x2")
            y2 = det.get("y2")

            if None in (x1, y1, x2, y2):
                continue

            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            is_target = label == active_target

            if is_target:
                color = (0, 255, 255)
                text_color = (0, 0, 0)
                thickness = 3
                text = f"TARGET: {label} {confidence:.2f}"
            else:
                color = (0, 255, 0)
                text_color = (0, 0, 0)
                thickness = 2
                text = f"{label} {confidence:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                thickness,
            )

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            font_thickness = 2

            (text_w, text_h), baseline = cv2.getTextSize(
                text,
                font,
                font_scale,
                font_thickness,
            )

            label_x1 = x1
            label_y1 = max(0, y1 - text_h - baseline - 10)
            label_x2 = min(frame.shape[1], x1 + text_w + 12)
            label_y2 = min(frame.shape[0], label_y1 + text_h + baseline + 10)

            cv2.rectangle(
                frame,
                (label_x1, label_y1),
                (label_x2, label_y2),
                color,
                -1,
            )

            cv2.putText(
                frame,
                text,
                (label_x1 + 6, label_y2 - baseline - 5),
                font,
                font_scale,
                text_color,
                font_thickness,
            )

        h, w = frame.shape[:2]
        cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 0, 0), 1)
        cv2.line(frame, (0, h // 2), (w, h // 2), (255, 0, 0), 1)

        return frame

    def image_callback(self, msg):
        global latest_jpeg

        try:
            frame = self.bridge.imgmsg_to_cv2(
                msg,
                desired_encoding="bgr8",
            )

            now = time.time()
            if now - self.last_detect_time >= self.detect_interval:
                self.latest_detections = self.call_vision_server(frame)
                self.last_detect_time = now

            frame = self.draw_boxes(frame, self.latest_detections)

            ok, jpeg = cv2.imencode(".jpg", frame)

            if ok:
                with frame_lock:
                    latest_jpeg = jpeg.tobytes()

        except Exception as e:
            self.get_logger().warn(f"Dashboard frame error: {e}")


@app.route("/")
def index():
    return """
<!DOCTYPE html>
<html>
<head>
<title>Qwen Mini Pupper Dashboard Pro</title>
<style>
body {
    margin: 0;
    background: #101014;
    color: white;
    font-family: Arial, sans-serif;
}
.header {
    padding: 14px;
    background: #181820;
    font-size: 24px;
    font-weight: bold;
}
.container {
    display: flex;
    gap: 16px;
    padding: 16px;
}
.video {
    flex: 3;
}
.panel {
    flex: 1;
    background: #181820;
    padding: 16px;
    border-radius: 10px;
}
img {
    width: 100%;
    border: 3px solid #444;
    border-radius: 10px;
}
.card {
    background: #252532;
    padding: 12px;
    border-radius: 8px;
    margin-bottom: 10px;
}
.label {
    color: #aaa;
    font-size: 13px;
}
.value {
    font-size: 22px;
    font-weight: bold;
}
button {
    width: 100%;
    padding: 12px;
    margin-top: 8px;
    border: 0;
    border-radius: 8px;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
}
.idle { background: #777; color: white; }
.follow { background: #2d8cff; color: white; }
.backpack { background: #16a34a; color: white; }
.stop { background: #dc2626; color: white; }
</style>
</head>
<body>
<div class="header">Qwen Mini Pupper Dashboard Pro</div>

<div class="container">
    <div class="video">
        <img src="/video_feed">
    </div>

    <div class="panel">
        <div class="card"><div class="label">MISSION</div><div id="mission" class="value">---</div></div>
        <div class="card"><div class="label">STATE</div><div id="state" class="value">---</div></div>
        <div class="card"><div class="label">TARGET</div><div id="target" class="value">---</div></div>
        <div class="card"><div class="label">LIDAR FRONT</div><div id="lidar" class="value">---</div></div>
        <div class="card"><div class="label">LINEAR X</div><div id="vx" class="value">---</div></div>
        <div class="card"><div class="label">ANGULAR Z</div><div id="wz" class="value">---</div></div>
        <div class="card"><div class="label">VISION LATENCY</div><div id="latency" class="value">---</div></div>

        <button class="idle" onclick="sendMission('IDLE')">Idle / Stop</button>
        <button class="follow" onclick="sendMission('FOLLOW_PERSON')">Follow Person</button>
        <button class="backpack" onclick="sendMission('FIND_BACKPACK')">Find Backpack</button>
    </div>
</div>

<script>
async function refreshStatus() {
    const r = await fetch('/status');
    const s = await r.json();

    document.getElementById('mission').innerText = s.mission ?? '---';
    document.getElementById('state').innerText = s.state ?? '---';
    document.getElementById('target').innerText = s.target ?? '---';

    document.getElementById('lidar').innerText =
        s.front_distance == null ? 'None' : s.front_distance.toFixed(2) + ' m';

    document.getElementById('vx').innerText =
        s.linear_x == null ? '0.00' : s.linear_x.toFixed(2);

    document.getElementById('wz').innerText =
        s.angular_z == null ? '0.00' : s.angular_z.toFixed(2);

    document.getElementById('latency').innerText =
        s.vision_time == null ? '---' : Math.round(s.vision_time * 1000) + ' ms';
}

async function sendMission(mission) {
    await fetch('/mission', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({mission: mission})
    });
    refreshStatus();
}

setInterval(refreshStatus, 500);
refreshStatus();
</script>
</body>
</html>
"""


@app.route("/status")
def status():
    with status_lock:
        return jsonify(latest_status)


@app.route("/mission", methods=["POST"])
def mission():
    global dashboard_node

    data = request.get_json(force=True)
    mission_name = data.get("mission", "IDLE").upper()

    if dashboard_node is not None:
        dashboard_node.publish_mission(mission_name)

    return jsonify({"ok": True, "mission": mission_name})


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def generate_frames():
    while True:
        with frame_lock:
            frame = latest_jpeg

        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )

        time.sleep(0.05)


def ros_thread():
    global dashboard_node

    rclpy.init()
    dashboard_node = DashboardNode()

    try:
        rclpy.spin(dashboard_node)
    except KeyboardInterrupt:
        pass

    dashboard_node.destroy_node()

    try:
        rclpy.shutdown()
    except Exception:
        pass


def main():
    thread = threading.Thread(target=ros_thread, daemon=True)
    thread.start()

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
