class ObjectTracker:
    def __init__(self, target_label="backpack"):
        self.target_label = target_label
        self.last_target = None

        self.aliases = {
            "backpack": ["backpack", "suitcase", "handbag"],
            "bag": ["backpack", "suitcase", "handbag"],
            "person": ["person"],
            "chair": ["chair"],
            "bottle": ["bottle"],
        }

    def set_target(self, label):
        self.target_label = label
        self.last_target = None

    def label_matches(self, label):
        allowed = self.aliases.get(self.target_label, [self.target_label])
        return label in allowed

    def select_target(self, detections):
        candidates = []

        for det in detections:
            label = det.get("label") or det.get("class") or det.get("name")

            if not self.label_matches(label):
                continue

            confidence = float(det.get("confidence", det.get("score", 0.0)))

            x1 = det.get("x1")
            y1 = det.get("y1")
            x2 = det.get("x2")
            y2 = det.get("y2")

            if x1 is None and "bbox" in det:
                box = det["bbox"]
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

            if None in (x1, y1, x2, y2):
                continue

            width = float(x2) - float(x1)
            height = float(y2) - float(y1)
            area = width * height

            candidates.append({
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
            })

        if not candidates:
            self.last_target = None
            return None

        candidates.sort(
            key=lambda obj: (obj["confidence"], obj["area"]),
            reverse=True,
        )

        self.last_target = candidates[0]
        return self.last_target
