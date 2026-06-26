import json
import ollama


class QwenPlannerLLM:
    def __init__(self, model='qwen2.5:0.5b'):
        self.model = model

    def classify(self, text):
        prompt = f"""
You control a Mini Pupper 2 robot.

Return ONLY valid JSON.

Allowed actions:
MOVE
TURN
STOP
PICTURE
STATUS
VISION
UNKNOWN

JSON formats:

{{"action":"MOVE","direction":"forward","speed":0.05,"duration":1.0}}
{{"action":"MOVE","direction":"backward","speed":0.05,"duration":1.0}}
{{"action":"TURN","direction":"left","speed":0.4,"duration":1.0}}
{{"action":"TURN","direction":"right","speed":0.4,"duration":1.0}}
{{"action":"STOP"}}
{{"action":"PICTURE"}}
{{"action":"STATUS"}}
{{"action":"VISION"}}
{{"action":"UNKNOWN"}}

User request:
{text}
"""

        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )

            raw = response["message"]["content"].strip()

            start = raw.find("{")
            end = raw.rfind("}") + 1

            if start == -1 or end == 0:
                return {"action": "UNKNOWN"}

            data = json.loads(raw[start:end])

            valid_actions = {
                "MOVE",
                "TURN",
                "STOP",
                "PICTURE",
                "STATUS",
                "VISION",
                "UNKNOWN",
            }

            if data.get("action") not in valid_actions:
                return {"action": "UNKNOWN"}

            return data

        except Exception:
            return {"action": "UNKNOWN"}
