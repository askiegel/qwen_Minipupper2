import threading

from flask import Flask, Response, render_template_string, request

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Qwen Robot Dashboard</title>
    <style>
        body { font-family: Arial; text-align: center; background: #111; color: white; }
        button { font-size: 24px; margin: 10px; padding: 20px; width: 180px; }
        .stop { background: red; color: white; }
    </style>
</head>
<body>
    <h1>Qwen Robot Dashboard</h1>

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

    <p>Publishes to <b>/qwen_robot/command</b></p>
</body>
</html>
"""


class DashboardNode(Node):
    def __init__(self):
        super().__init__('qwen_robot_dashboard')
        self.command_pub = self.create_publisher(String, '/qwen_robot/command', 10)
        self.get_logger().info('Qwen Robot dashboard ROS node started.')

    def publish_command(self, text):
        msg = String()
        msg.data = text
        self.command_pub.publish(msg)
        self.get_logger().info(f'Dashboard command: {text}')


app = Flask(__name__)
ros_node = None


@app.route('/')
def index():
    return render_template_string(HTML)


@app.route('/command', methods=['POST'])
def command():
    cmd = request.form.get('cmd', 'stop')

    if ros_node is not None:
        ros_node.publish_command(cmd)

    return render_template_string(HTML)


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
    app.run(host='0.0.0.0', port=5000)

    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
