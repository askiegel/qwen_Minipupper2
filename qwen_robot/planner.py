from .llm import QwenPlannerLLM


class Planner:
    def __init__(self, use_qwen=True):
        self.use_qwen = use_qwen
        self.llm = QwenPlannerLLM()

    def plan(self, text):
        text = text.lower().strip()

        if any(word in text for word in ["stop", "halt", "freeze"]):
            return {"action": "STOP"}

        if any(word in text for word in ["backward", "back up", "reverse", "go back"]):
            return {
                "action": "MOVE",
                "direction": "backward",
                "speed": 0.05,
                "duration": 1.0,
            }

        if any(word in text for word in ["forward", "ahead", "go forward", "move forward"]):
            return {
                "action": "MOVE",
                "direction": "forward",
                "speed": 0.05,
                "duration": 1.0,
            }

        if any(word in text for word in ["turn left", "left"]):
            return {
                "action": "TURN",
                "direction": "left",
                "speed": 0.4,
                "duration": 1.0,
            }

        if any(word in text for word in ["turn right", "right"]):
            return {
                "action": "TURN",
                "direction": "right",
                "speed": 0.4,
                "duration": 1.0,
            }

        if any(word in text for word in ["picture", "photo", "camera", "image", "snapshot"]):
            return {"action": "PICTURE"}

        if any(word in text for word in ["status", "distance", "wall", "obstacle", "how close"]):
            return {"action": "STATUS"}

        if any(word in text for word in ["see", "look", "describe"]):
            return {"action": "VISION"}

        if self.use_qwen:
            return self.llm.classify(text)

        return {"action": "UNKNOWN"}
