class BehaviorManager:
    def __init__(self):
        self.state = "IDLE"
        self.last_state = None
        self.state_counter = 0

    def update(self, target, front_distance, motion_state):
        self.last_state = self.state

        obstacle_close = (
            front_distance is not None
            and front_distance < 0.28
        )

        if obstacle_close:
            self.state = "AVOIDING_OBSTACLE"

        elif motion_state == "ARRIVED":
            self.state = "ARRIVED"

        elif target is not None:
            self.state = "FOLLOWING"

        elif motion_state == "SEARCHING":
            self.state = "SEARCHING"

        else:
            self.state = "IDLE"

        if self.state == self.last_state:
            self.state_counter += 1
        else:
            self.state_counter = 0

        return self.state

    def status_text(self):
        return f"{self.state}"
