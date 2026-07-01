from qwen_robot.target_manager import TargetManager


class ObjectTracker:
    def __init__(self, target_label="backpack"):
        self.target_label = target_label

        self.aliases = {
            "backpack": [
                "backpack",
                "suitcase",
                "handbag",
                "sports ball",
                "umbrella",
            ],
            "bag": [
                "backpack",
                "suitcase",
                "handbag",
                "umbrella",
            ],
            "person": ["person"],
            "chair": ["chair"],
            "bottle": ["bottle"],
            "laptop": ["laptop"],
        }

        self.min_confidence = {
            "backpack": 0.35,
            "bag": 0.35,
            "person": 0.45,
        }

        self.last_target = None
        self.locked = False
        self.missed_frames = 0
        self.max_missed_frames = 12
        self.target_id = 0

        # v0.7 lightweight Person ReID layer
        self.person_target_manager = TargetManager(
            target_label="person",
            lock_threshold=0.45,
            reacquire_threshold=0.35,
            lost_timeout=4.0,
            smoothing=0.25,
        )

    def reset(self):
        self.last_target = None
        self.locked = False
        self.missed_frames = 0
        self.person_target_manager.reset()

    def set_target(self, label):
        self.target_label = label
        self.reset()
        self.target_id += 1

    def label_matches(self, label):
        allowed = self.aliases.get(self.target_label, [self.target_label])
        return label in allowed

    def normalize_detection(self, det):
        label = det.get("label") or det.get("class") or det.get("name")

        if not self.label_matches(label):
            return None

        confidence = float(det.get("confidence", det.get("score", det.get("conf", 0.0))))

        required_conf = self.min_confidence.get(self.target_label, 0.40)

        if confidence < required_conf:
            return None

        x1 = det.get("x1")
        y1 = det.get("y1")
        x2 = det.get("x2")
        y2 = det.get("y2")

        if x1 is None and "bbox" in det:
            box = det["bbox"]
            x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

        if None in (x1, y1, x2, y2):
            return None

        width = float(x2) - float(x1)
        height = float(y2) - float(y1)

        if width <= 0 or height <= 0:
            return None

        area = width * height

        return {
            "id": self.target_id,
            "label": label,
            "confidence": confidence,
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "bbox": [float(x1), float(y1), float(x2), float(y2)],
            "cx": (float(x1) + float(x2)) / 2.0,
            "cy": (float(y1) + float(y2)) / 2.0,
            "width": width,
            "height": height,
            "area": area,
        }

    def score_candidate(self, candidate):
        if self.last_target is None:
            return candidate["confidence"] * 1000.0 + candidate["area"] * 0.002

        dx = candidate["cx"] - self.last_target["cx"]
        dy = candidate["cy"] - self.last_target["cy"]
        distance_penalty = (dx * dx + dy * dy) ** 0.5

        area_diff = abs(candidate["area"] - self.last_target["area"])
        area_penalty = area_diff / max(self.last_target["area"], 1.0)

        return (
            candidate["confidence"] * 1000.0
            + candidate["area"] * 0.001
            - distance_penalty * 1.5
            - area_penalty * 80.0
        )

    def select_target(self, detections, frame=None):
        candidates = []

        for det in detections:
            normalized = self.normalize_detection(det)
            if normalized is not None:
                candidates.append(normalized)

        # v0.7: person tracking now uses TargetManager identity lock/reacquisition
        if self.target_label == "person":
            selected = self.person_target_manager.update(candidates, frame=frame)

            if selected is None:
                self.missed_frames += 1
                return None

            self.last_target = selected
            self.locked = True
            self.missed_frames = 0
            return selected

        # Existing object behavior remains unchanged for backpack, bag, etc.
        if not candidates:
            self.missed_frames += 1

            if self.locked and self.last_target is not None:
                if self.missed_frames <= self.max_missed_frames:
                    return None

            self.locked = False
            self.last_target = None
            return None

        candidates.sort(
            key=self.score_candidate,
            reverse=True,
        )

        selected = candidates[0]
        selected["lost"] = False

        if not self.locked:
            self.target_id += 1
            selected["id"] = self.target_id

        self.last_target = selected
        self.locked = True
        self.missed_frames = 0

        return selected

    def telemetry(self):
        if self.target_label == "person":
            return self.person_target_manager.telemetry()

        return {
            "target_state": "LOCKED" if self.locked else "UNLOCKED",
            "target_id": self.target_id if self.locked else None,
            "target_label": self.target_label,
            "target_confidence": (
                round(float(self.last_target.get("confidence", 0.0)), 3)
                if self.last_target else 0.0
            ),
            "target_similarity": 1.0 if self.locked else 0.0,
            "target_cx": self.last_target.get("cx") if self.last_target else None,
            "target_cy": self.last_target.get("cy") if self.last_target else None,
            "target_area": self.last_target.get("area") if self.last_target else None,
            "target_lost_time": 0.0,
            "target_last_seen_age": 0.0,
        }
