class MissionManager:
    def __init__(self, default_mission="follow"):
        self.mission = default_mission.upper()
        self.target_object = "person"

        if self.mission == "FIND_BACKPACK":
            self.target_object = "backpack"
        elif self.mission == "FOLLOW_PERSON":
            self.target_object = "person"
        elif self.mission == "FOLLOW":
            self.target_object = "person"

    def set_mission(self, mission):
        self.mission = mission.upper()

        if self.mission == "FIND_BACKPACK":
            self.target_object = "backpack"
        elif self.mission == "FOLLOW_PERSON":
            self.target_object = "person"
        elif self.mission == "FOLLOW":
            self.target_object = "person"
        elif self.mission == "IDLE":
            self.target_object = None

    def should_move(self):
        return self.mission != "IDLE"

    def status_text(self):
        return self.mission

    def get_target_object(self):
        return self.target_object
