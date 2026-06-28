import os
import time
import threading

import cv2
import rclpy
import requests
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from flask import Flask, Response


app = Flask(__name__)

latest_jpeg = None
latest_lock = threading.Lock()


class QwenDashboard(Node):
    def __init__(self):
        super().__init__("qwen_dashboard")

        self.bridge = CvBridge()

        self.vision_url = os.getenv(
            "VISION_SERVER_URL",
            "http://192.168.68.100:8000/detect",
        )

        self.last_detect_time = 0.0
        self.detect_interval = 0.30
        self.latest_detections = []

        self.image_sub = self.create_subscription(
            Image,
            "/image_raw",
            self.image_callback,
            10,
        )

        self.get_logger().info("Qwen dashboard started")
        self.get_logger().info(f"Vision URL: {self.vision_url}")
        self.get_logger().info("Drawing YOLO boxes on /image_raw")

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
        for det in detections:
            label = det.get("label", "object")
            confidence = float(det.get("confidence", 0.0))

            x1 = det.get("x1")
            y1 = det.get("y1")
            x2 = det.get("x2")
            y2 = det.get("y2")

            if None in (x1, y1, x2, y2):
                continue

            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)

            text = f"{label} {confidence:.2f}"

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.rectangle(
                frame,
                (x1, max(0, y1 - 25)),
                (x1 + 170, y1),
                (0, 255, 0),
                -1,
            )

            cv2.putText(
                frame,
                text,
                (x1 + 5, max(18, y1 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                2,
            )

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

            cv2.putText(
                frame,
                f"Objects: {len(self.latest_detections)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            ok, jpeg = cv2.imencode(".jpg", frame)

            if ok:
                with latest_lock:
                    latest_jpeg = jpeg.tobytes()

        except Exception as e:
            self.get_logger().warn(f"Dashboard frame error: {e}")


@app.route("/")
def index():
    return """
    <html>
      <head>
        <title>Qwen Robot Dashboard</title>
        <style>
          body {
            background: #111;
            color: white;
            font-family: Arial, sans-serif;
            text-align: center;
          }
          img {
            width: 90%;
            max-width: 900px;
            border: 3px solid #444;
            border-radius: 10px;
          }
        </style>
      </head>
      <body>
        <h1>Qwen Robot Dashboard</h1>
        <p>Live camera with YOLO object boxes</p>
        <img src="/video_feed">
      </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


def generate_frames():
    while True:
        with latest_lock:
            frame = latest_jpeg

        if frame is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" +
                frame +
                b"\r\n"
            )

        time.sleep(0.05)


def ros_thread():
    rclpy.init()
    node = QwenDashboard()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()


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
