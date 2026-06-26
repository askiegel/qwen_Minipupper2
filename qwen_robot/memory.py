from collections import deque
from datetime import datetime


class RobotMemory:
    def __init__(self, max_observations=100):
        self.observations = deque(maxlen=max_observations)

    def add_vision_observation(self, vision_result):
        observation = {
            "time": datetime.now().isoformat(timespec="seconds"),
            "objects": vision_result.get("objects", []),
            "description": vision_result.get("description", "No description.")
        }

        self.observations.appendleft(observation)
        return observation

    def latest(self):
        if not self.observations:
            return None
        return self.observations[0]

    def count(self):
        return len(self.observations)

    def summary(self):
        if not self.observations:
            return {
                "count": 0,
                "latest": "No observations yet.",
                "recent_objects": []
            }

        recent_objects = []

        for obs in list(self.observations)[:10]:
            for obj in obs.get("objects", []):
                if obj not in recent_objects:
                    recent_objects.append(obj)

        return {
            "count": len(self.observations),
            "latest": self.observations[0],
            "recent_objects": recent_objects
        }

    def compare_latest_two(self):
        if len(self.observations) < 2:
            return {
                "changed": False,
                "message": "Not enough observations to compare."
            }

        newest = set(self.observations[0].get("objects", []))
        previous = set(self.observations[1].get("objects", []))

        added = sorted(list(newest - previous))
        removed = sorted(list(previous - newest))

        if not added and not removed:
            message = "I do not notice any object changes."
        else:
            parts = []
            if added:
                parts.append("New objects: " + ", ".join(added))
            if removed:
                parts.append("Missing objects: " + ", ".join(removed))
            message = ". ".join(parts) + "."

        return {
            "changed": bool(added or removed),
            "added": added,
            "removed": removed,
            "message": message
        }
