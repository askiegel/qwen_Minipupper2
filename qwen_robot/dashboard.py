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
        .card { background: #222; padding: 20px; margin: 20px auto; width: 85%; max-width: 950px; border-radius: 12px; }
        button { font-size: 22px; margin: 8px; padding: 18px; width: 180px; border-radius: 10px; border: none; }
        .stop { background: #b00020; color: white; }
        .follow { background: #087a25; color: white; }
        .send { width: 120px; background: #0078d7; color: white; }
        .status, .vision { text-align: left; display: inline-block; font-size: 20px; max-width: 800px; }
        img { max-width: 100%; border-radius: 10px; border: 2px solid #444; }
        code { color: #00ff99; }
        .log { text-align: left; background: #000; padding: 15px; border-radius: 8px; font-family: monospace; max-height: 260px; overflow-y: auto; }
        .log-line { margin: 4px 0; color: #00ff99; }
        input[type=text] { font-size: 22px; padding: 14px; width: 70%; border-radius: 8px; border: none; }
    </style>
</head>
<body>
    <h1>Qwen Robot Dashboard</h1>

    <div class="card">
        <h2>Live Camera with YOLO Overlay</h2>
        <img src="/video_feed" alt="Live camera stream">
    </div>

    <div class="card">
        <h2>Command Console</h2>
        <form method="post" action="/command">
            <input type="text" name="cmd" placeholder="Try: follow me, what do you see, stop">
            <button class="send" type="submit">Send</button>
        </form>
    </div>

    <div class="card">
        <h2>Vision Result</h2>
        <div class="vision">
            <p><b>Description:</b> {{ vision.description }}</p>
            <p><b>Objects:</b> {{ vision.objects }}</p>
            <p><b>Detections:</b> {{ vision.detections }}</p>
        </div>
    </div>

    <div class="card">
        <h2>Robot Status</h2>
        <div class="status">
            <p><b>Motion:</b> {{ status.motion_state }}</p>
            <p><b>Follow Mode:</b> {{ status.follow_mode }}</p>
            <p><b>Last Command:</b> {{ status.last_command }}</p>
            <p><b>Last Action:</b> {{ status.last_action }}</p>
            <p><b>Last Message:</b> {{ status.last_message }}</p>
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
            <button class="follow" name="cmd" value="follow me">Follow Me</button>
            <button class="stop" name="cmd" value="stop">Stop</button>
            <button name="cmd" value="what do you see">What Do You See?</button>
            <button name="cmd" value="take picture">Take Picture</button><br>
            <button name="cmd" value="forward">Forward</button><br>
            <button name="cmd" value="left">Left</button>
            <button name="cmd" value="right">Right</button><br>
            <button name="cmd" value="backward">Backward</button>
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

        self.create_subscription(String, '/qwen_robot/status', self.status_callback, 10)
        self.create_subscription(Image, '/image_raw', self.image_callback, 10)

        self.status = {
            'motion_state': 'unknown',
            'follow_mode': False,
            'last_command': 'none',
            'last_action': 'none',
            'last_message': 'none',
            'front_distance': 'unknown',
            'camera_ready': False,
        }

        self.vision = {
            'objects': 'None',
            'description': 'No vision result yet.',
            'detections': 'None',
        }

        self.raw_detections = []
        self.follow_target = None
        self.find_target = None

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
            front_distance_text = 'None' if front_distance is None else f"{front_distance:.2f} m"

            old_command = self.status.get('last_command')
            old_message = self.status.get('last_message')
            old_vision_description = self.vision.get('description')

            self.status = {
                'motion_state': data.get('motion_state', 'unknown'),
                'follow_mode': data.get('follow_mode', False),
                'last_command': data.get('last_command', 'none'),
                'last_action': data.get('last_action', 'none'),
                'last_message': data.get('last_message', 'none'),
                'front_distance': front_distance_text,
                'camera_ready': data.get('camera_ready', False),
            }

            last_vision = data.get('last_vision', {})
            objects = last_vision.get('objects', [])
            detections = last_vision.get('detections', [])
            description = last_vision.get('description', 'No vision result yet.')

            objects_text = ', '.join(objects) if isinstance(objects, list) and objects else 'None'
            detections_text = str(len(detections)) if isinstance(detections, list) else 'None'

            self.vision = {
                'objects': objects_text,
                'description': description,
                'detections': detections_text,
            }

            self.raw_detections = detections if isinstance(detections, list) else []
            self.follow_target = data.get('follow_target', None)
            self.find_target = data.get('find_target', None)

            if self.status['last_command'] != old_command:
                self.add_log(f"Last command: {self.status['last_command']}")

            if self.status['last_message'] != old_message:
                self.add_log(f"Message: {self.status['last_message']}")

            if self.vision['description'] != old_vision_description:
                self.add_log(f"Vision: {self.vision['description']}")

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

    def draw_overlays(self, frame):
        h, w = frame.shape[:2]

        # Center crosshair
        cv2.line(frame, (w // 2, 0), (w // 2, h), (255, 180, 0), 1)
        cv2.line(frame, (0, h // 2), (w, h // 2), (255, 180, 0), 1)

        # Draw all detections
        for d in self.raw_detections:
            label = d.get('label', 'object')
            conf = d.get('confidence', 0.0)
            x1 = int(d.get('x1', 0))
            y1 = int(d.get('y1', 0))
            x2 = int(d.get('x2', 0))
            y2 = int(d.get('y2', 0))

            color = (0, 255, 0) if label == 'person' else (0, 200, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f'{label} {conf:.2f}',
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )

        # Follow target center marker and steering text
        target = self.follow_target if self.follow_target is not None else self.find_target
        if isinstance(target, dict):
            cx = int(target.get('center_x', w // 2))
            cy = int(target.get('center_y', h // 2))
            error = cx - (w // 2)

            cv2.drawMarker(
                frame,
                (cx, cy),
                (0, 255, 0),
                markerType=cv2.MARKER_CROSS,
                markerSize=30,
                thickness=3
            )

            cv2.putText(
                frame,
                f'error: {error}px',
                (20, h - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

            if error < -40:
                steer_text = 'steer left'
            elif error > 40:
                steer_text = 'steer right'
            else:
                steer_text = 'centered'

            cv2.putText(
                frame,
                steer_text,
                (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2
            )

        return frame

    def get_jpeg_frame(self):
        with self.frame_lock:
            if self.latest_frame is None:
                return None
            frame = self.latest_frame.copy()

        frame = self.draw_overlays(frame)

        success, buffer = cv2.imencode('.jpg', frame)
        return buffer.tobytes() if success else None


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
        vision=ros_node.vision,
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
