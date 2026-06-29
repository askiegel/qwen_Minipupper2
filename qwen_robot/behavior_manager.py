class BehaviorManager:
    def __init__(self):
        self.state = "IDLE"
        self.last_state = None
        self.state_counter = 0

    def update(self, mission, target, front_distance, motion_state):
        self.last_state = self.state

        mission = mission.upper()

        obstacle_close = (
            front_distance is not None
            and front_distance < 0.28
        )

        if mission == "IDLE":
            self.state = "IDLE"

        elif obstacle_close:
            self.state = "AVOIDING_OBSTACLE"

        elif motion_state == "ARRIVED":
            self.state = "ARRIVED"

        elif target is not None:
            self.state = "FOLLOWING"

        elif mission in ("FIND_BACKPACK", "FOLLOW_PERSON", "FOLLOW"):
            self.state = "SEARCHING"

        else:
            self.state = "SEARCHING"

        if self.state == self.last_state:
            self.state_counter += 1
        else:
            self.state_counter = 0

        return self.state

    def status_text(self):
        return self.state
