from geometry_msgs.msg import Twist


class FollowManager:
    def __init__(self):
        self.image_width = 640.0

        self.center_deadband = 60.0

        self.stop_area = 90000.0
        self.slow_area = 60000.0

        self.max_forward = 0.12
        self.slow_forward = 0.06

        self.turn_speed = 0.35

    def compute_cmd(self, target):
        cmd = Twist()

        if target is None:
            return cmd, "NO_TARGET"

        center = self.image_width / 2.0
        error = target["cx"] - center
        area = target["area"]

        if area >= self.stop_area:
            return cmd, "ARRIVED"

        if abs(error) > self.center_deadband:
            if error > 0:
                cmd.angular.z = -self.turn_speed
            else:
                cmd.angular.z = self.turn_speed

        if area >= self.slow_area:
            cmd.linear.x = self.slow_forward
        else:
            cmd.linear.x = self.max_forward

        return cmd, "FOLLOWING"
