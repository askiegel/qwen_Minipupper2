import json
import threading
import time
from collections import deque

import cv2
import psutil
from flask import Flask, Response, render_template_string, request, redirect

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
        body { font-family: Arial; background: #111; color: white; text-align: center; }
        .card { background: #222; padding: 20px; margin: 20px auto; width: 85%; max-width: 900px; border-radius: 12px; }
        button { font-size: 22px; margin: 8px; padding: 18px; width: 170px; border-radius: 10px; border: none; }
        .stop { background: #b00020; color: white; }
        .status { text-align: left; display: inline-block; font-size: 20px; }
        img { max-width: 100%; border-radius: 10px; border: 2px solid #444; }
        code { color: #00ff99; }
        .log { text-align: left; background: #000; padding: 15px; border-radius: 8px; font-family: monospace; max-height: 260px; overflow-y: auto; }
        .log-line { margin: 4px 0; color: #00ff99; }
        input[type=text] { font-size: 22px; padding: 14px; width: 70%; border-radius: 8px; border: none; }
        .send { width: 120px; background: #0078d7; color: white; }
    </style>
</head>
<body>
    <h1>Qwen Robot Dashboard</h1>

    <div class="card">
        <h2>Live Camera</h2>
        <img src="/video_feed" alt="Live camera stream">
    </div>

    <div class="card">
        <h2>Command Console</h2>
        <form method="post" action="/command">
            <input type="text" name="cmd" placeholder="Type command: forward, stop, status, take picture">
            <button class="send" type="submit">Send</button>
        </form>
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
        <h2>System Health</h2>
        <div class="status">
            <p><b>CPU:</b> {{ health.cpu }}%</p>
            <p><b>RAM:</b> {{ health.ram_percent }}% used</p>
            <p><b>RAM Used:</b> {{ health.ram_used }} / {{ health.ram_total }} GB</p>
            <p><b>Disk:</b> {{ health.disk }}% used</p>
            <p><b>Uptime:</b> {{ health.uptime }}</p>
            <p><b>Temperature:</b> {{ health.temperature }}</p>
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
        <h2>Robot Log</h2>
        <div class="log">
            {% for line in logs %}
                <div class="log-line">{{ line }}</div>
            {% endfor %}
        </div>
    </div>

    <div class="card">
        <p>Publishes to <code>/qwen_robot/command</code></p>
        <p>Reads status from <code>/qwen_robot/status</code></p>
        <p>Reads camera from <code>/image_raw</code></p>
        <p><a href="/" style="color:#00ccff;">Refresh Dashboard</a></p>
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
        self.logs = deque(maxlen=50)

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

        self.add_log('Dashboard started.')
        self.get_logger().info('Qwen Robot dashboard ROS node started.')

    def add_log(self, text):
        timestamp = time.strftime('%H:%M:%S')
        self.logs.appendleft(f'{timestamp}  {text}')

    def publish_command(self, text):
        msg = String()
        msg.data = text
        self.command_pub.publish(msg)
        self.add_log(f'Command sent: {text}')
        self.get_logger().info(f'Dashboard command: {text}')

    def status_callback(self, msg):
        try:
            data = json.loads(msg.data)

            front_distance = data.get('front_distance', None)
            if front_distance is None:
                front_distance_text = 'None'
            else:
                front_distance_text = f"{front_distance:.2f} m"

            old_motion = self.status.get('motion_state')
            old_command = self.status.get('last_command')

            self.status = {
                'motion_state': data.get('motion_state', 'unknown'),
                'last_command': data.get('last_command', 'none'),
                'front_distance': front_distance_text,
                'camera_ready': data.get('camera_ready', False),
            }

            if self.status['motion_state'] != old_motion:
                self.add_log(f"Motion state: {self.status['motion_state']}")

            if self.status['last_command'] != old_command:
                self.add_log(f"Last command: {self.status['last_command']}")

        except Exception as e:
            self.get_logger().warn(f'Status parse error: {e}')
            self.add_log(f'Status parse error: {e}')

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.latest_frame = frame
        except Exception as e:
            self.get_logger().warn(f'Image stream error: {e}')
            self.add_log(f'Image stream error: {e}')

    def get_jpeg_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            frame = self.latest_frame.copy()

        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            return None

        return buffer.tobytes()


def get_temperature():
    try:
        with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
            temp_c = int(f.read().strip()) / 1000.0
            return f'{temp_c:.1f} °C'
    except Exception:
        return 'Unavailable'


def get_health():
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    uptime_seconds = int(time.time() - psutil.boot_time())
    hours = uptime_seconds // 3600
    minutes = (uptime_seconds % 3600) // 60

    return {
        'cpu': psutil.cpu_percent(interval=0.1),
        'ram_percent': ram.percent,
        'ram_used': round(ram.used / (1024 ** 3), 2),
        'ram_total': round(ram.total / (1024 ** 3), 2),
        'disk': disk.percent,
        'uptime': f'{hours}h {minutes}m',
        'temperature': get_temperature(),
    }


app = Flask(__name__)
ros_node = None


@app.route('/')
def index():
    return render_template_string(
        HTML,
        status=ros_node.status,
        logs=list(ros_node.logs),
        health=get_health()
    )


@app.route('/command', methods=['POST'])
def command():
    cmd = request.form.get('cmd', 'stop').strip()

    if cmd and ros_node is not None:
        ros_node.publish_command(cmd)

    return redirect('/')


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
