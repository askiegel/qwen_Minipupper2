class SearchBehavior:
    def __init__(self):
        self.phase = "LEFT"
        self.counter = 0

        self.left_duration = 8
        self.right_duration = 16
        self.forward_duration = 6
        self.pause_duration = 4

        self.search_turn_speed = 0.18
        self.search_forward_speed = 0.04

    def reset(self):
        self.phase = "LEFT"
        self.counter = 0

    def update(self, cmd, front_distance=None):
        self.counter += 1

        obstacle_close = (
            front_distance is not None
            and front_distance < 0.30
        )

        cmd.linear.x = 0.0
        cmd.angular.z = 0.0

        if self.phase == "LEFT":
            cmd.angular.z = self.search_turn_speed

            if self.counter >= self.left_duration:
                self.phase = "RIGHT"
                self.counter = 0

        elif self.phase == "RIGHT":
            cmd.angular.z = -self.search_turn_speed

            if self.counter >= self.right_duration:
                self.phase = "FORWARD"
                self.counter = 0

        elif self.phase == "FORWARD":
            if not obstacle_close:
                cmd.linear.x = self.search_forward_speed

            if self.counter >= self.forward_duration:
                self.phase = "PAUSE"
                self.counter = 0

        elif self.phase == "PAUSE":
            if self.counter >= self.pause_duration:
                self.phase = "LEFT"
                self.counter = 0

        return cmd, f"SEARCH_{self.phase}"
