import time
import math
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class TargetManager:
    """
    v0.8 feature branch: lightweight Person ReID with appearance matching.

    Tracks one person using:
      - location prediction
      - bounding-box size consistency
      - YOLO confidence
      - HSV upper-body appearance histogram

    The public interface remains simple:
      selected = update(detections, frame=None)
      telemetry = telemetry()
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

        self.location_score = 0.0
        self.size_score = 0.0
        self.confidence_score = 0.0
        self.appearance_score = 0.0

        self.vx = 0.0
        self.vy = 0.0

        self.last_seen_time = 0.0
        self.lost_since = None

        self.appearance_hist = None

    def reset(self):
        self.state = "UNLOCKED"
        self.target_id = None
        self.bbox = None
        self.cx = None
        self.cy = None
        self.area = None
        self.confidence = 0.0
        self.similarity = 0.0

        self.location_score = 0.0
        self.size_score = 0.0
        self.confidence_score = 0.0
        self.appearance_score = 0.0

        self.vx = 0.0
        self.vy = 0.0
        self.last_seen_time = 0.0
        self.lost_since = None
        self.appearance_hist = None

    def update(self, detections: List[Dict], frame=None) -> Optional[Dict]:
        now = time.time()
        people = [d for d in detections if self._label(d) == self.target_label]

        if not people:
            self._mark_lost(now)
            return None

        if self.state == "UNLOCKED":
            chosen = self._choose_initial_target(people)
            self._lock(chosen, now, frame)
            return self._selected_detection(chosen)

        chosen, score = self._best_match(people, now, frame)

        threshold = (
            self.lock_threshold
            if self.state == "LOCKED"
            else self.reacquire_threshold
        )

        if chosen is not None and score >= threshold:
            self.similarity = score
            self._update_locked_target(chosen, now, frame)
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
            "target_location_score": round(float(self.location_score), 3),
            "target_size_score": round(float(self.size_score), 3),
            "target_confidence_score": round(float(self.confidence_score), 3),
            "target_appearance_score": round(float(self.appearance_score), 3),
            "target_cx": self.cx,
            "target_cy": self.cy,
            "target_area": self.area,
            "target_lost_time": round(float(lost_time), 2),
            "target_last_seen_age": round(float(last_seen_age), 2),
            "target_has_appearance": self.appearance_hist is not None,
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
        conf = float(d.get("confidence", d.get("conf", d.get("score", 0.0))))

        return (x1, y1, x2, y2), cx, cy, area, conf

    def _choose_initial_target(self, people: List[Dict]) -> Dict:
        return max(
            people,
            key=lambda d: (
                float(d.get("confidence", d.get("conf", d.get("score", 0.0)))) * 0.35
                + float(d.get("area", self._area_from_bbox(d))) * 0.65
            ),
        )

    def _area_from_bbox(self, d: Dict) -> float:
        x1, y1, x2, y2 = self._bbox(d)
        return max((x2 - x1) * (y2 - y1), 1.0)

    def _lock(self, detection: Dict, now: float, frame=None):
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

        self.location_score = 1.0
        self.size_score = 1.0
        self.confidence_score = max(0.0, min(1.0, conf))
        self.appearance_score = 1.0

        self.vx = 0.0
        self.vy = 0.0

        self.last_seen_time = now
        self.lost_since = None

        self.appearance_hist = self._compute_appearance_hist(frame, bbox)

    def _best_match(self, people: List[Dict], now: float, frame=None):
        best = None
        best_score = -1.0
        best_parts = None

        for d in people:
            score, parts = self._score_detection(d, now, frame)
            if score > best_score:
                best = d
                best_score = score
                best_parts = parts

        if best_parts is not None:
            self.location_score = best_parts["location"]
            self.size_score = best_parts["size"]
            self.confidence_score = best_parts["confidence"]
            self.appearance_score = best_parts["appearance"]

        return best, best_score

    def _score_detection(self, d: Dict, now: float, frame=None):
        bbox, cx, cy, area, conf = self._features(d)

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

        candidate_hist = self._compute_appearance_hist(frame, bbox)
        appearance_score = self._compare_hist(candidate_hist)

        if self.appearance_hist is None or candidate_hist is None:
            score = (
                0.55 * location_score
                + 0.30 * size_score
                + 0.15 * confidence_score
            )
        else:
            score = (
                0.35 * location_score
                + 0.25 * size_score
                + 0.10 * confidence_score
                + 0.30 * appearance_score
            )

        parts = {
            "location": max(0.0, min(1.0, location_score)),
            "size": max(0.0, min(1.0, size_score)),
            "confidence": max(0.0, min(1.0, confidence_score)),
            "appearance": max(0.0, min(1.0, appearance_score)),
        }

        return max(0.0, min(1.0, score)), parts

    def _update_locked_target(self, detection: Dict, now: float, frame=None):
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

        new_hist = self._compute_appearance_hist(frame, bbox)
        if new_hist is not None:
            if self.appearance_hist is None:
                self.appearance_hist = new_hist
            else:
                self.appearance_hist = cv2.normalize(
                    0.90 * self.appearance_hist + 0.10 * new_hist,
                    None,
                    alpha=0,
                    beta=1,
                    norm_type=cv2.NORM_MINMAX,
                )

    def _compute_appearance_hist(self, frame, bbox):
        if frame is None or bbox is None:
            return None

        try:
            h_img, w_img = frame.shape[:2]
            x1, y1, x2, y2 = bbox

            x1 = int(max(0, min(w_img - 1, x1)))
            x2 = int(max(0, min(w_img - 1, x2)))
            y1 = int(max(0, min(h_img - 1, y1)))
            y2 = int(max(0, min(h_img - 1, y2)))

            if x2 <= x1 or y2 <= y1:
                return None

            box_w = x2 - x1
            box_h = y2 - y1

            # Upper-body crop: avoids pants/floor and focuses on shirt/jacket.
            ux1 = x1 + int(0.15 * box_w)
            ux2 = x2 - int(0.15 * box_w)
            uy1 = y1 + int(0.15 * box_h)
            uy2 = y1 + int(0.60 * box_h)

            if ux2 <= ux1 or uy2 <= uy1:
                return None

            crop = frame[uy1:uy2, ux1:ux2]
            if crop.size == 0:
                return None

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

            hist = cv2.calcHist(
                [hsv],
                [0, 1],
                None,
                [24, 16],
                [0, 180, 0, 256],
            )

            hist = cv2.normalize(
                hist,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
            )

            return hist.astype(np.float32)

        except Exception:
            return None

    def _compare_hist(self, candidate_hist):
        if self.appearance_hist is None or candidate_hist is None:
            return 0.0

        try:
            corr = cv2.compareHist(
                self.appearance_hist,
                candidate_hist,
                cv2.HISTCMP_CORREL,
            )

            # Correlation can be [-1, 1]. Convert to [0, 1].
            return max(0.0, min(1.0, (corr + 1.0) / 2.0))

        except Exception:
            return 0.0

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
        selected["target_location_score"] = self.location_score
        selected["target_size_score"] = self.size_score
        selected["target_confidence_score"] = self.confidence_score
        selected["target_appearance_score"] = self.appearance_score
        selected["cx"] = self.cx
        selected["cy"] = self.cy
        selected["area"] = self.area
        selected["bbox"] = self.bbox
        return selected
