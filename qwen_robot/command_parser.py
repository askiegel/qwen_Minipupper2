class CommandParser:
    def parse(self, text):
        text = text.lower().strip()

        if "stop" in text:
            return "STOP"
        if "backward" in text or "back" in text or "reverse" in text:
            return "BACKWARD"
        if "forward" in text or "ahead" in text:
            return "FORWARD"
        if "left" in text:
            return "LEFT"
        if "right" in text:
            return "RIGHT"

        return "UNKNOWN"
