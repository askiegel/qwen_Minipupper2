import os
import time
import json
import threading
from pathlib import Path
from datetime import datetime

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
latest_target_jpeg = None
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

    def update_target_crop(self, frame, detections):
        global latest_target_jpeg

        with status_lock:
            status = dict(latest_status)

        target_label = status.get("target", "none")
        target_cx = status.get("target_cx", None)
        target_cy = status.get("target_cy", None)

        if target_label in (None, "none"):
            return

        best = None
        best_score = -1.0

        for det in detections:
            label = det.get("label", "object")
            if label != target_label:
                continue

            x1 = det.get("x1")
            y1 = det.get("y1")
            x2 = det.get("x2")
            y2 = det.get("y2")

            if None in (x1, y1, x2, y2):
                continue

            cx = (float(x1) + float(x2)) / 2.0
            cy = (float(y1) + float(y2)) / 2.0

            if target_cx is not None and target_cy is not None:
                dx = cx - float(target_cx)
                dy = cy - float(target_cy)
                score = -((dx * dx + dy * dy) ** 0.5)
            else:
                score = float(det.get("confidence", 0.0))

            if score > best_score:
                best = det
                best_score = score

        if best is None:
            return

        h, w = frame.shape[:2]

        x1 = int(max(0, min(w - 1, best.get("x1"))))
        y1 = int(max(0, min(h - 1, best.get("y1"))))
        x2 = int(max(0, min(w - 1, best.get("x2"))))
        y2 = int(max(0, min(h - 1, best.get("y2"))))

        if x2 <= x1 or y2 <= y1:
            return

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return

        crop = cv2.resize(crop, (240, 240), interpolation=cv2.INTER_AREA)

        ok, jpeg = cv2.imencode(".jpg", crop)

        if ok:
            with frame_lock:
                latest_target_jpeg = jpeg.tobytes()

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

            self.update_target_crop(frame, self.latest_detections)
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
.dev { background: #9333ea; color: white; }
.devpanel {
    display: none;
    margin-top: 12px;
    border: 1px solid #555;
}
.smallvalue {
    font-size: 16px;
    font-weight: bold;
}
.ok { color: #22c55e; }
.warn { color: #facc15; }
.bad { color: #ef4444; }
.crop {
    width: 100%;
    border: 2px solid #555;
    border-radius: 8px;
    margin-top: 8px;
}
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

        <div class="card"><div class="label">TARGET STATE</div><div id="target_state" class="value">---</div></div>
        <div class="card"><div class="label">TARGET ID</div><div id="target_id" class="value">---</div></div>
        <div class="card"><div class="label">SIMILARITY</div><div id="target_similarity" class="value">---</div></div>
        <div class="card"><div class="label">TARGET CONF</div><div id="target_confidence" class="value">---</div></div>
        <div class="card"><div class="label">APPEARANCE</div><div id="target_appearance_score" class="value">---</div></div>
        <div class="card"><div class="label">LOCATION</div><div id="target_location_score" class="value">---</div></div>
        <div class="card"><div class="label">SIZE</div><div id="target_size_score" class="value">---</div></div>
        <div class="card"><div class="label">HAS APPEARANCE</div><div id="target_has_appearance" class="value">---</div></div>
        <div class="card"><div class="label">LOST TIME</div><div id="target_lost_time" class="value">---</div></div>
        <div class="card"><div class="label">LAST SEEN</div><div id="target_last_seen_age" class="value">---</div></div>

        <button class="idle" onclick="sendMission('IDLE')">Idle / Stop</button>
        <button class="follow" onclick="sendMission('FOLLOW_PERSON')">Follow Person</button>
        <button class="backpack" onclick="sendMission('FIND_BACKPACK')">Find Backpack</button>
        <button class="dev" onclick="toggleDeveloperMode()">Developer Mode</button>
        <button class="dev" onclick="saveDiagnostics()">Save Diagnostics</button>

        <div id="developer_panel" class="devpanel">
            <div class="card">
                <div class="label">CURRENT TARGET CROP</div>
                <img class="crop" src="/target_crop">
            </div>
            <div class="card"><div class="label">BUILD</div><div id="dev_build" class="smallvalue">---</div></div>
            <div class="card"><div class="label">SYSTEM</div><div id="dev_system" class="smallvalue">---</div></div>
            <div class="card"><div class="label">ROS STATUS</div><div id="dev_ros" class="smallvalue">---</div></div>
            <div class="card"><div class="label">CAMERA FRAMES</div><div id="dev_frames" class="smallvalue">---</div></div>
            <div class="card"><div class="label">CONTROL LOOPS</div><div id="dev_loops" class="smallvalue">---</div></div>
            <div class="card"><div class="label">VISION FPS EST</div><div id="dev_fps" class="smallvalue">---</div></div>
            <div class="card"><div class="label">VISION HEALTH</div><div id="dev_vision" class="smallvalue">---</div></div>
            <div class="card"><div class="label">LIDAR HEALTH</div><div id="dev_lidar" class="smallvalue">---</div></div>
            <div class="card"><div class="label">TARGET CENTER</div><div id="dev_center" class="smallvalue">---</div></div>
            <div class="card"><div class="label">TARGET AREA</div><div id="dev_area" class="smallvalue">---</div></div>
            <div class="card"><div class="label">REID BREAKDOWN</div><div id="dev_reid" class="smallvalue">---</div></div>
            <div class="card">
                <div class="label">KNOWN TARGETS</div>
                <div id="dev_memory" class="smallvalue">---</div>
            </div>
        </div>
    </div>
</div>

<script>
let developerMode = false;
let lastFrames = 0;
let lastFrameTime = Date.now();

function toggleDeveloperMode() {
    developerMode = !developerMode;
    document.getElementById('developer_panel').style.display =
        developerMode ? 'block' : 'none';
}

function healthClass(value) {
    if (value === 'OK') return 'ok';
    if (value === 'WARN') return 'warn';
    return 'bad';
}

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

    document.getElementById('target_state').innerText =
        s.target_state ?? '---';

    document.getElementById('target_id').innerText =
        s.target_id == null ? '---' : s.target_id;

    document.getElementById('target_similarity').innerText =
        s.target_similarity == null ? '---' : s.target_similarity.toFixed(3);

    document.getElementById('target_confidence').innerText =
        s.target_confidence == null ? '---' : s.target_confidence.toFixed(3);

    document.getElementById('target_appearance_score').innerText =
        s.target_appearance_score == null ? '---' : s.target_appearance_score.toFixed(3);

    document.getElementById('target_location_score').innerText =
        s.target_location_score == null ? '---' : s.target_location_score.toFixed(3);

    document.getElementById('target_size_score').innerText =
        s.target_size_score == null ? '---' : s.target_size_score.toFixed(3);

    document.getElementById('target_has_appearance').innerText =
        s.target_has_appearance == null ? '---' : (s.target_has_appearance ? 'YES' : 'NO');

    document.getElementById('target_lost_time').innerText =
        s.target_lost_time == null ? '---' : s.target_lost_time.toFixed(2) + ' s';

    document.getElementById('target_last_seen_age').innerText =
        s.target_last_seen_age == null ? '---' : s.target_last_seen_age.toFixed(2) + ' s';

    const now = Date.now();
    const frames = s.frames ?? 0;
    const dt = (now - lastFrameTime) / 1000.0;
    const df = frames - lastFrames;
    const fps = dt > 0 ? df / dt : 0.0;

    if (dt >= 1.0) {
        lastFrames = frames;
        lastFrameTime = now;
    }

    const rosHealth = s.mission === 'UNKNOWN' ? 'BAD' : 'OK';
    const visionHealth = s.vision_time == null || s.vision_time <= 0.0
        ? 'BAD'
        : (s.vision_time < 0.35 ? 'OK' : 'WARN');

    const lidarHealth = s.front_distance == null
        ? 'WARN'
        : 'OK';

    document.getElementById('dev_build').innerText =
        'branch=' + (s.git_branch ?? 'unknown') +
        ' commit=' + (s.git_commit ?? 'unknown');

    document.getElementById('dev_system').innerText =
        'load=' + (s.load_avg_1m == null ? '---' : s.load_avg_1m.toFixed(2)) +
        ' mem=' + (s.mem_percent == null ? '---' : s.mem_percent.toFixed(1) + '%') +
        ' temp=' + (s.cpu_temp_c == null ? '---' : s.cpu_temp_c.toFixed(1) + 'C');

    document.getElementById('dev_ros').innerHTML =
        '<span class="' + healthClass(rosHealth) + '">' + rosHealth + '</span>';

    document.getElementById('dev_frames').innerText =
        s.frames == null ? '---' : s.frames;

    document.getElementById('dev_loops').innerText =
        s.loops == null ? '---' : s.loops;

    document.getElementById('dev_fps').innerText =
        fps.toFixed(1);

    document.getElementById('dev_vision').innerHTML =
        '<span class="' + healthClass(visionHealth) + '">' + visionHealth + '</span>'
        + ' / ' + (s.vision_time == null ? '---' : Math.round(s.vision_time * 1000) + ' ms');

    document.getElementById('dev_lidar').innerHTML =
        '<span class="' + healthClass(lidarHealth) + '">' + lidarHealth + '</span>'
        + ' / ' + (s.front_distance == null ? 'None' : s.front_distance.toFixed(2) + ' m');

    document.getElementById('dev_center').innerText =
        s.target_cx == null || s.target_cy == null
        ? '---'
        : '(' + s.target_cx.toFixed(1) + ', ' + s.target_cy.toFixed(1) + ')';

    document.getElementById('dev_area').innerText =
        s.target_area == null ? '---' : Math.round(s.target_area);

    document.getElementById('dev_reid').innerText =
        'sim=' + (s.target_similarity == null ? '---' : s.target_similarity.toFixed(3)) +
        ' app=' + (s.target_appearance_score == null ? '---' : s.target_appearance_score.toFixed(3)) +
        ' loc=' + (s.target_location_score == null ? '---' : s.target_location_score.toFixed(3)) +
        ' size=' + (s.target_size_score == null ? '---' : s.target_size_score.toFixed(3));

    const known = s.known_targets ?? [];
    if (known.length === 0) {
        document.getElementById('dev_memory').innerText = 'No known targets';
    } else {
        document.getElementById('dev_memory').innerHTML = known.map(t => {
            const age = t.last_seen_age == null ? '---' : t.last_seen_age.toFixed(1) + 's';
            const conf = t.confidence == null ? '---' : t.confidence.toFixed(2);
            return '#' + t.id + ' ' + t.label +
                   ' | seen ' + age +
                   ' | conf ' + conf +
                   ' | count ' + t.seen_count;
        }).join('<br>');
    }

    document.getElementById('dev_nav').innerHTML =
        'state=' + (s.nav_state ?? '---') + '<br>' +
        'last=' + (s.nav_last_target ?? '---') + '<br>' +
        'age=' + (s.nav_last_seen_age == null ? '---' : s.nav_last_seen_age.toFixed(2) + 's') + '<br>' +
        'direction=' + (s.nav_last_direction ?? '---') + '<br>' +
        'cx=' + (s.nav_last_cx == null ? '---' : s.nav_last_cx.toFixed(1));
}

async function sendMission(mission) {
    await fetch('/mission', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({mission: mission})
    });
    refreshStatus();
}

async function saveDiagnostics() {
    const r = await fetch('/save_diagnostics', {method: 'POST'});
    const result = await r.json();
    alert(result.ok ? ('Saved: ' + result.path) : ('Save failed: ' + result.error));
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


@app.route("/target_crop")
def target_crop():
    with frame_lock:
        frame = latest_target_jpeg

    if frame is None:
        return Response(status=204)

    return Response(
        frame,
        mimetype="image/jpeg",
    )


@app.route("/save_diagnostics", methods=["POST"])
def save_diagnostics():
    try:
        with status_lock:
            snapshot = dict(latest_status)

        out_dir = Path("/home/ubuntu/ros2_ws/src/qwen_robot/diagnostics")
        out_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = out_dir / f"diagnostics_{stamp}.json"

        with open(path, "w") as f:
            json.dump(snapshot, f, indent=2)

        return jsonify({"ok": True, "path": str(path)})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


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
