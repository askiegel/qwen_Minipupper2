import cv2
import requests


class VisionClient:
    def __init__(self, server_url):
        self.server_url = server_url.rstrip("/")
        self.last_result = {
            "objects": [],
            "description": "No vision result yet."
        }

    def analyze_frame(self, frame):
        if frame is None:
            self.last_result = {
                "objects": [],
                "description": "No camera image available."
            }
            return self.last_result

        ok, buffer = cv2.imencode(".jpg", frame)

        if not ok:
            self.last_result = {
                "objects": [],
                "description": "Could not encode camera image."
            }
            return self.last_result

        files = {
            "file": ("frame.jpg", buffer.tobytes(), "image/jpeg")
        }

        try:
            response = requests.post(
                f"{self.server_url}/detect",
                files=files,
                timeout=10
            )

            response.raise_for_status()
            self.last_result = response.json()
            return self.last_result

        except Exception as e:
            self.last_result = {
                "objects": [],
                "description": f"Vision server error: {e}"
            }
            return self.last_result
