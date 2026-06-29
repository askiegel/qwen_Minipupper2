from geometry_msgs.msg import Twist


class FollowManager:
    def __init__(self):
        self.image_width = 640.0

        # Distance behavior based on bounding-box area
        self.stop_area = 90000.0
        self.slow_area = 60000.0

        # Forward speed limits
        self.max_forward = 0.12
        self.min_forward = 0.03

        # PID steering gains
        self.kp = 0.0016
        self.ki = 0.0
        self.kd = 0.0008

        self.max_turn = 0.45

        self.prev_error = 0.0
        self.integral = 0.0

        self.last_turn_direction = 0.0
        self.lost_count = 0
        self.max_lost_count = 6

    def clamp(self, value, low, high):
        return max(low, min(high, value))

    def compute_cmd(self, target):
        cmd = Twist()

        if target is None:
            self.lost_count += 1

            if self.lost_count <= self.max_lost_count:
                cmd.angular.z = self.last_turn_direction * 0.18
                return cmd, "SEARCHING"

            self.prev_error = 0.0
            self.integral = 0.0
            return cmd, "NO_TARGET"

        self.lost_count = 0

        cx = target["cx"]
        area = target["area"]

        center = self.image_width / 2.0
        error = center - cx

        derivative = error - self.prev_error
        self.integral += error

        self.integral = self.clamp(self.integral, -10000.0, 10000.0)

        turn = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )

        turn = self.clamp(turn, -self.max_turn, self.max_turn)

        cmd.angular.z = turn

        if abs(turn) > 0.02:
            self.last_turn_direction = 1.0 if turn > 0 else -1.0

        self.prev_error = error

        if area >= self.stop_area:
            cmd.linear.x = 0.0
            cmd.angular.z = 0.0
            return cmd, "ARRIVED"

        if area >= self.slow_area:
            base_speed = self.min_forward
        else:
            base_speed = self.max_forward

        turn_slowdown = 1.0 - min(abs(turn) / self.max_turn, 1.0)
        speed = base_speed * (0.35 + 0.65 * turn_slowdown)

        cmd.linear.x = self.clamp(speed, self.min_forward, self.max_forward)

        return cmd, "FOLLOWING"
