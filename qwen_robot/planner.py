from .llm import QwenPlannerLLM


class Planner:
    def __init__(self, use_qwen=True):
        self.use_qwen = use_qwen
        self.llm = QwenPlannerLLM()

    def plan(self, text):
        text = text.lower().strip()

        # ---------- Stop ----------
        if any(x in text for x in [
            "stop",
            "halt",
            "freeze",
            "stop following"
        ]):
            return {"action": "STOP"}

        # ---------- Follow Person ----------
        if any(x in text for x in [
            "follow me",
            "follow person",
            "follow the person",
            "track me",
            "track person"
        ]):
            return {"action": "FOLLOW_PERSON"}

        # ---------- Memory ----------
        if any(x in text for x in [
            "memory",
            "remember",
            "what have you seen"
        ]):
            return {"action": "MEMORY"}

        # ---------- Compare ----------
        if any(x in text for x in [
            "what changed",
            "changed",
            "different",
            "compare"
        ]):
            return {"action": "COMPARE"}

        # ---------- Move ----------
        if any(x in text for x in [
            "backward",
            "reverse",
            "go back"
        ]):
            return {
                "action": "MOVE",
                "direction": "backward",
                "speed": 0.05,
                "duration": 1.0,
            }

        if any(x in text for x in [
            "forward",
            "ahead",
            "go forward"
        ]):
            return {
                "action": "MOVE",
                "direction": "forward",
                "speed": 0.05,
                "duration": 1.0,
            }

        # ---------- Turn ----------
        if "left" in text:
            return {
                "action": "TURN",
                "direction": "left",
                "speed": 0.4,
                "duration": 1.0,
            }

        if "right" in text:
            return {
                "action": "TURN",
                "direction": "right",
                "speed": 0.4,
                "duration": 1.0,
            }

        # ---------- Camera ----------
        if any(x in text for x in [
            "picture",
            "photo",
            "camera",
            "snapshot"
        ]):
            return {"action": "PICTURE"}

        # ---------- Vision ----------
        if any(x in text for x in [
            "what do you see",
            "look",
            "describe"
        ]):
            return {"action": "VISION"}

        # ---------- Status ----------
        if any(x in text for x in [
            "status",
            "distance",
            "obstacle"
        ]):
            return {"action": "STATUS"}

        if self.use_qwen:
            return self.llm.classify(text)

        return {"action": "UNKNOWN"}
