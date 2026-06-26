from .llm import QwenPlannerLLM


class Planner:
    def __init__(self, use_qwen=True):
        self.use_qwen = use_qwen
        self.llm = QwenPlannerLLM()

    def plan(self, text):
        text = text.lower().strip()

        if any(w in text for w in ["stop", "halt", "freeze"]):
            return {"action": "STOP"}

        if any(w in text for w in ["memory", "remember", "what have you seen"]):
            return {"action": "MEMORY"}

        if any(w in text for w in ["what changed", "changed", "different", "compare"]):
            return {"action": "COMPARE"}

        if any(w in text for w in ["backward", "back up", "reverse", "go back"]):
            return {"action": "MOVE", "direction": "backward", "speed": 0.05, "duration": 1.0}

        if any(w in text for w in ["forward", "ahead", "go forward", "move forward"]):
            return {"action": "MOVE", "direction": "forward", "speed": 0.05, "duration": 1.0}

        if any(w in text for w in ["turn left", "left"]):
            return {"action": "TURN", "direction": "left", "speed": 0.4, "duration": 1.0}

        if any(w in text for w in ["turn right", "right"]):
            return {"action": "TURN", "direction": "right", "speed": 0.4, "duration": 1.0}

        if any(w in text for w in ["picture", "photo", "camera", "image", "snapshot"]):
            return {"action": "PICTURE"}

        if any(w in text for w in ["status", "distance", "wall", "obstacle", "how close"]):
            return {"action": "STATUS"}

        if any(w in text for w in ["see", "look", "describe"]):
            return {"action": "VISION"}

        if self.use_qwen:
            return self.llm.classify(text)

        return {"action": "UNKNOWN"}
