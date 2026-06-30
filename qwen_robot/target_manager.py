import time
import math
from typing import Dict, List, Optional, Tuple


class TargetManager:
    """
    v0.7 Target Manager / lightweight Person ReID.

    Locks onto one person, ignores others, and attempts reacquisition
    after short occlusions.
    """

    def __init__(
        self,
        target_label: str = "person",
        lock_threshold: float = 0.45,
        reacquire_threshold: float = 0.35,
        lost_timeout: float = 4.0,
        smoothing: float = 0.25,
    ):
        self.target_label = target_label
        self.lock_threshold = lock_threshold
        self.reacquire_threshold = reacquire_threshold
        self.lost_timeout = lost_timeout
        self.smoothing = smoothing

        self.state = "UNLOCKED"
        self.target_id = None
        self.next_id = 1

        self.bbox = None
        self.cx = None
        self.cy = None
        self.area = None
        self.confidence = 0.0
        self.similarity = 0.0

        self.vx = 0.0
        self.vy = 0.0

        self.last_seen_time = 0.0
        self.lost_since = None

    def reset(self):
        self.state = "UNLOCKED"
        self.target_id = None
        self.bbox = None
        self.cx = None
        self.cy = None
        self.area = None
        self.confidence = 0.0
        self.similarity = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.last_seen_time = 0.0
        self.lost_since = None

    def update(self, detections: List[Dict]) -> Optional[Dict]:
        now = time.time()
        people = [d for d in detections if self._label(d) == self.target_label]

        if not people:
            self._mark_lost(now)
            return None

        if self.state == "UNLOCKED":
            chosen = self._choose_initial_target(people)
            self._lock(chosen, now)
            return self._selected_detection(chosen)

        chosen, score = self._best_match(people, now)

        threshold = (
            self.lock_threshold
            if self.state == "LOCKED"
            else self.reacquire_threshold
        )

        if chosen is not None and score >= threshold:
            self.similarity = score
            self._update_locked_target(chosen, now)
            return self._selected_detection(chosen)

        self._mark_lost(now)
        return None

    def telemetry(self) -> Dict:
        now = time.time()

        last_seen_age = 999.0
        if self.last_seen_time > 0:
            last_seen_age = now - self.last_seen_time

        lost_time = 0.0
        if self.lost_since is not None:
            lost_time = now - self.lost_since

        return {
            "target_state": self.state,
            "target_id": self.target_id,
            "target_label": self.target_label,
            "target_confidence": round(float(self.confidence), 3),
            "target_similarity": round(float(self.similarity), 3),
            "target_cx": self.cx,
            "target_cy": self.cy,
            "target_area": self.area,
            "target_lost_time": round(float(lost_time), 2),
            "target_last_seen_age": round(float(last_seen_age), 2),
        }

    def _label(self, d: Dict) -> str:
        return str(
            d.get("label", d.get("class", d.get("name", "")))
        ).lower()

    def _bbox(self, d: Dict) -> Tuple[float, float, float, float]:
        if "bbox" in d and d["bbox"] is not None:
            b = d["bbox"]
            return float(b[0]), float(b[1]), float(b[2]), float(b[3])

        x1 = d.get("x1", d.get("xmin", None))
        y1 = d.get("y1", d.get("ymin", None))
        x2 = d.get("x2", d.get("xmax", None))
        y2 = d.get("y2", d.get("ymax", None))

        if None not in (x1, y1, x2, y2):
            return float(x1), float(y1), float(x2), float(y2)

        cx = d.get("cx", None)
        cy = d.get("cy", None)
        area = d.get("area", None)

        if None not in (cx, cy, area):
            side = math.sqrt(max(float(area), 1.0))
            return (
                float(cx) - side / 2.0,
                float(cy) - side / 2.0,
                float(cx) + side / 2.0,
                float(cy) + side / 2.0,
            )

        return 0.0, 0.0, 1.0, 1.0

    def _features(self, d: Dict):
        x1, y1, x2, y2 = self._bbox(d)

        w = max(x2 - x1, 1.0)
        h = max(y2 - y1, 1.0)

        cx = float(d.get("cx", (x1 + x2) / 2.0))
        cy = float(d.get("cy", (y1 + y2) / 2.0))
        area = float(d.get("area", w * h))
        conf = float(d.get("confidence", d.get("conf", 0.0)))

        return (x1, y1, x2, y2), cx, cy, area, conf

    def _choose_initial_target(self, people: List[Dict]) -> Dict:
        return max(
            people,
            key=lambda d: (
                float(d.get("confidence", d.get("conf", 0.0))) * 0.35
                + float(d.get("area", self._area_from_bbox(d))) * 0.65
            ),
        )

    def _area_from_bbox(self, d: Dict) -> float:
        x1, y1, x2, y2 = self._bbox(d)
        return max((x2 - x1) * (y2 - y1), 1.0)

    def _lock(self, detection: Dict, now: float):
        bbox, cx, cy, area, conf = self._features(detection)

        self.state = "LOCKED"
        self.target_id = self.next_id
        self.next_id += 1

        self.bbox = bbox
        self.cx = cx
        self.cy = cy
        self.area = area
        self.confidence = conf
        self.similarity = 1.0

        self.vx = 0.0
        self.vy = 0.0

        self.last_seen_time = now
        self.lost_since = None

    def _best_match(self, people: List[Dict], now: float):
        best = None
        best_score = -1.0

        for d in people:
            score = self._score_detection(d, now)
            if score > best_score:
                best = d
                best_score = score

        return best, best_score

    def _score_detection(self, d: Dict, now: float) -> float:
        _, cx, cy, area, conf = self._features(d)

        dt = max(now - self.last_seen_time, 0.001)

        pred_cx = self.cx + self.vx * dt if self.cx is not None else cx
        pred_cy = self.cy + self.vy * dt if self.cy is not None else cy

        dist = math.sqrt((cx - pred_cx) ** 2 + (cy - pred_cy) ** 2)
        location_score = 1.0 / (1.0 + dist / 160.0)

        if self.area is None or self.area <= 0:
            size_score = 1.0
        else:
            size_score = min(area, self.area) / max(area, self.area)
            size_score = max(0.0, min(1.0, size_score))

        confidence_score = max(0.0, min(1.0, conf))

        score = (
            0.55 * location_score
            + 0.30 * size_score
            + 0.15 * confidence_score
        )

        return max(0.0, min(1.0, score))

    def _update_locked_target(self, detection: Dict, now: float):
        bbox, new_cx, new_cy, new_area, conf = self._features(detection)

        dt = max(now - self.last_seen_time, 0.001)

        if self.cx is not None:
            measured_vx = (new_cx - self.cx) / dt
            measured_vy = (new_cy - self.cy) / dt

            self.vx = 0.7 * self.vx + 0.3 * measured_vx
            self.vy = 0.7 * self.vy + 0.3 * measured_vy

        a = self.smoothing

        if self.cx is None:
            self.cx = new_cx
            self.cy = new_cy
            self.area = new_area
        else:
            self.cx = (1.0 - a) * self.cx + a * new_cx
            self.cy = (1.0 - a) * self.cy + a * new_cy
            self.area = (1.0 - a) * self.area + a * new_area

        self.bbox = bbox
        self.confidence = conf
        self.state = "LOCKED"
        self.last_seen_time = now
        self.lost_since = None

    def _mark_lost(self, now: float):
        if self.state == "UNLOCKED":
            return

        if self.lost_since is None:
            self.lost_since = now

        lost_time = now - self.lost_since

        if lost_time >= self.lost_timeout:
            self.reset()
        else:
            self.state = "SEARCHING"

    def _selected_detection(self, detection: Dict) -> Dict:
        selected = dict(detection)
        selected["target_id"] = self.target_id
        selected["target_state"] = self.state
        selected["target_similarity"] = self.similarity
        selected["cx"] = self.cx
        selected["cy"] = self.cy
        selected["area"] = self.area
        selected["bbox"] = self.bbox
        return selected
