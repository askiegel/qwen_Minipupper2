import cv2
import requests


class VisionClient:
    def __init__(self, url, timeout=1.0):
        self.url = url
        self.timeout = timeout

    def get_detections_from_frame(self, frame):
        try:
            if frame is None:
                return []

            ok, encoded = cv2.imencode(".jpg", frame)

            if not ok:
                return []

            files = {
                "file": (
                    "frame.jpg",
                    encoded.tobytes(),
                    "image/jpeg",
                )
            }

            response = requests.post(
                self.url,
                files=files,
                timeout=self.timeout,
            )

            response.raise_for_status()
            data = response.json()

            if isinstance(data, dict):
                return data.get("detections", [])

            if isinstance(data, list):
                return data

            return []

        except Exception:
            return []
