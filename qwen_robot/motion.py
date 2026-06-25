import time
from geometry_msgs.msg import Twist


class MotionController:
    def __init__(self, cmd_pub):
        self.cmd_pub = cmd_pub

    def stop(self):
        self.cmd_pub.publish(Twist())

    def move(self, linear_x=0.0, angular_z=0.0, duration=1.0):
        twist = Twist()
        twist.linear.x = linear_x
        twist.angular.z = angular_z

        start = time.time()

        while time.time() - start < duration:
            self.cmd_pub.publish(twist)
            time.sleep(0.1)

        self.stop()
