class ObjectTracker:
    def __init__(self, target_label="backpack"):
        self.target_label = target_label

        self.aliases = {
            "backpack": ["backpack", "suitcase", "handbag"],
            "bag": ["backpack", "suitcase", "handbag"],
            "person": ["person"],
            "chair": ["chair"],
            "bottle": ["bottle"],
            "laptop": ["laptop"],
        }

        self.last_target = None
        self.locked = False
        self.missed_frames = 0
        self.max_missed_frames = 8

        self.target_id = 0

    def set_target(self, label):
        self.target_label = label
        self.last_target = None
        self.locked = False
        self.missed_frames = 0
        self.target_id += 1

    def label_matches(self, label):
        allowed = self.aliases.get(self.target_label, [self.target_label])
        return label in allowed

    def normalize_detection(self, det):
        label = det.get("label") or det.get("class") or det.get("name")

        if not self.label_matches(label):
            return None

        confidence = float(det.get("confidence", det.get("score", 0.0)))

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
        area = width * height

        if width <= 0 or height <= 0:
            return None

        return {
            "id": self.target_id,
            "label": label,
            "confidence": confidence,
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "cx": (float(x1) + float(x2)) / 2.0,
            "cy": (float(y1) + float(y2)) / 2.0,
            "width": width,
            "height": height,
            "area": area,
        }

    def score_candidate(self, candidate):
        if self.last_target is None:
            return candidate["confidence"] * 1000.0 + candidate["area"] * 0.001

        dx = candidate["cx"] - self.last_target["cx"]
        dy = candidate["cy"] - self.last_target["cy"]
        distance_penalty = (dx * dx + dy * dy) ** 0.5

        area_diff = abs(candidate["area"] - self.last_target["area"])
        area_penalty = area_diff / max(self.last_target["area"], 1.0)

        return (
            candidate["confidence"] * 1000.0
            - distance_penalty * 2.0
            - area_penalty * 100.0
        )

    def select_target(self, detections):
        candidates = []

        for det in detections:
            normalized = self.normalize_detection(det)
            if normalized is not None:
                candidates.append(normalized)

        if not candidates:
            self.missed_frames += 1

            if self.locked and self.last_target is not None:
                if self.missed_frames <= self.max_missed_frames:
                    ghost = dict(self.last_target)
                    ghost["confidence"] = 0.0
                    ghost["lost"] = True
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
