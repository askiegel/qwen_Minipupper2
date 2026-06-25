import json
import threading
import time

import cv2
from flask import Flask, Response, render_template_string, request

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Qwen Robot Dashboard</title>
    <style>
        body {
            font-family: Arial;
            background: #111;
            color: white;
            text-align: center;
        }
        .card {
            background: #222;
            padding: 20px;
            margin: 20px auto;
            width: 85%;
            max-width: 800px;
            border-radius: 12px;
        }
        button {
            font-size: 22px;
            margin: 8px;
            padding: 18px;
            width: 170px;
            border-radius: 10px;
            border: none;
        }
        .stop {
            background: #b00020;
            color: white;
        }
        .status {
            text-align: left;
            display: inline-block;
            font-size: 20px;
        }
        img {
            max-width: 100%;
            border-radius: 10px;
            border: 2px solid #444;
        }
        code {
            color: #00ff99;
        }
    </style>
</head>
<body>
    <h1>Qwen Robot Dashboard</h1>

    <div class="card">
        <h2>Live Camera</h2>
        <img src="/video_feed" alt="Live camera stream">
    </div>

    <div class="card">
        <h2>Robot Status</h2>
        <div class="status">
            <p><b>Motion:</b> {{ status.motion_state }}</p>
            <p><b>Last Command:</b> {{ status.last_command }}</p>
            <p><b>Front Distance:</b> {{ status.front_distance }}</p>
            <p><b>Camera Ready:</b> {{ status.camera_ready }}</p>
        </div>
    </div>

    <div class="card">
        <h2>Manual Control</h2>
        <form method="post" action="/command">
            <button name="cmd" value="forward">Forward</button><br>
            <button name="cmd" value="left">Left</button>
            <button class="stop" name="cmd" value="stop">Stop</button>
            <button name="cmd" value="right">Right</button><br>
            <button name="cmd" value="backward">Backward</button><br>
            <button name="cmd" value="take picture">Take Picture</button>
            <button name="cmd" value="status">Status</button>
        </form>
    </div>

    <div class="card">
        <p>Publishes to <code>/qwen_robot/command</code></p>
        <p>Reads status from <code>/qwen_robot/status</code></p>
        <p>Reads camera from <code>/image_raw</code></p>
        <p><a href="/" style="color:#00ccff;">Refresh</a></p>
    </div>
</body>
</html>
"""


class DashboardNode(Node):
    def __init__(self):
        super().__init__('qwen_robot_dashboard')

        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_lock = threading.Lock()

        self.command_pub = self.create_publisher(String, '/qwen_robot/command', 10)

        self.status_sub = self.create_subscription(
            String,
            '/qwen_robot/status',
            self.status_callback,
            10
        )

        self.image_sub = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )

        self.status = {
            'motion_state': 'unknown',
            'last_command': 'none',
            'front_distance': 'unknown',
            'camera_ready': False,
        }

        self.get_logger().info('Qwen Robot dashboard ROS node started.')

    def publish_command(self, text):
        msg = String()
        msg.data = text
        self.command_pub.publish(msg)
        self.get_logger().info(f'Dashboard command: {text}')

    def status_callback(self, msg):
        try:
            data = json.loads(msg.data)

            front_distance = data.get('front_distance', None)
            if front_distance is None:
                front_distance_text = 'None'
            else:
                front_distance_text = f"{front_distance:.2f} m"

            self.status = {
                'motion_state': data.get('motion_state', 'unknown'),
                'last_command': data.get('last_command', 'none'),
                'front_distance': front_distance_text,
                'camera_ready': data.get('camera_ready', False),
            }

        except Exception as e:
            self.get_logger().warn(f'Status parse error: {e}')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.latest_frame = frame
        except Exception as e:
            self.get_logger().warn(f'Image stream error: {e}')

    def get_jpeg_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None

            frame = self.latest_frame.copy()

        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            return None

        return buffer.tobytes()


app = Flask(__name__)
ros_node = None


@app.route('/')
def index():
    return render_template_string(HTML, status=ros_node.status)


@app.route('/command', methods=['POST'])
def command():
    cmd = request.form.get('cmd', 'stop')

    if ros_node is not None:
        ros_node.publish_command(cmd)

    return render_template_string(HTML, status=ros_node.status)


def generate_video():
    while True:
        if ros_node is None:
            time.sleep(0.1)
            continue

        frame = ros_node.get_jpeg_frame()

        if frame is None:
            time.sleep(0.1)
            continue

        yield (
            b'--frame\r\n'
            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
        )

        time.sleep(0.05)


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_video(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


def ros_spin():
    rclpy.spin(ros_node)


def main(args=None):
    global ros_node

    rclpy.init(args=args)
    ros_node = DashboardNode()

    thread = threading.Thread(target=ros_spin)
    thread.daemon = True
    thread.start()

    ros_node.get_logger().info('Dashboard available at http://0.0.0.0:5000')
    app.run(host='0.0.0.0', port=5000, threaded=True)

    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
