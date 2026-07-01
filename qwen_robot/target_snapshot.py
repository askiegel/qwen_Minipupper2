#!/usr/bin/env python3
"""
Target Snapshot - v0.8 Alpha Sprint 2

Shared target data model for Target Manager, Memory Manager,
Navigation Manager, Dashboard, and future SLAM integration.

This file is intentionally standalone and safe:
- No ROS dependency
- No motion commands
- No changes to current behavior
"""

from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, Any, List
import time


@dataclass(frozen=True)
class TargetSnapshot:
    target_id: Optional[str] = None
    label: str = "none"
    visible: bool = False

    confidence: float = 0.0
    similarity: float = 0.0

    cx: Optional[float] = None
    cy: Optional[float] = None
    area: Optional[float] = None

    distance_m: Optional[float] = None
    bearing_deg: Optional[float] = None

    last_seen_time: Optional[float] = None
    last_seen_age_s: Optional[float] = None

    # Future SLAM/world-model fields.
    frame_id: str = "camera"
    world_x_m: Optional[float] = None
    world_y_m: Optional[float] = None
    world_theta_deg: Optional[float] = None

    seen_count: int = 0
    track_age_s: float = 0.0

    has_appearance: bool = False
    appearance_embedding: Optional[List[float]] = field(default=None, repr=False)

    status: str = "uninitialized"
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def empty(status: str = "no target") -> "TargetSnapshot":
        now = time.time()
        return TargetSnapshot(
            label="none",
            visible=False,
            confidence=0.0,
            similarity=0.0,
            last_seen_time=None,
            last_seen_age_s=999.0,
            status=status,
            updated_at=now,
        )

    @staticmethod
    def from_detection(
        detection: Optional[Dict[str, Any]],
        target_id: Optional[str] = None,
        similarity: float = 0.0,
        distance_m: Optional[float] = None,
        bearing_deg: Optional[float] = None,
        seen_count: int = 0,
        track_age_s: float = 0.0,
        has_appearance: bool = False,
        appearance_embedding: Optional[List[float]] = None,
    ) -> "TargetSnapshot":
        now = time.time()

        if detection is None:
            return TargetSnapshot.empty()

        label = detection.get("label", "unknown")
        confidence = float(detection.get("confidence", detection.get("conf", 0.0)))

        return TargetSnapshot(
            target_id=str(target_id) if target_id is not None else None,
            label=label,
            visible=True,
            confidence=confidence,
            similarity=float(similarity or 0.0),
            cx=_safe_float(detection.get("cx")),
            cy=_safe_float(detection.get("cy")),
            area=_safe_float(detection.get("area")),
            distance_m=distance_m,
            bearing_deg=bearing_deg,
            last_seen_time=now,
            last_seen_age_s=0.0,
            frame_id="camera",
            seen_count=int(seen_count or 0),
            track_age_s=float(track_age_s or 0.0),
            has_appearance=bool(has_appearance),
            appearance_embedding=appearance_embedding,
            status="visible",
            updated_at=now,
        )

    @staticmethod
    def from_status(status: Dict[str, Any]) -> "TargetSnapshot":
        now = time.time()

        label = status.get("target_label", status.get("target", "none"))
        target_id = status.get("target_id")
        confidence = float(status.get("target_confidence", 0.0) or 0.0)
        similarity = float(status.get("target_similarity", 0.0) or 0.0)

        visible = status.get("target_state") in ("LOCKED", "TRACKING", "VISIBLE")

        return TargetSnapshot(
            target_id=str(target_id) if target_id is not None else None,
            label=label or "none",
            visible=bool(visible),
            confidence=confidence,
            similarity=similarity,
            cx=_safe_float(status.get("target_cx")),
            cy=_safe_float(status.get("target_cy")),
            area=_safe_float(status.get("target_area")),
            distance_m=_safe_float(status.get("front_distance")),
            bearing_deg=_safe_float(status.get("target_bearing_deg")),
            last_seen_time=None,
            last_seen_age_s=_safe_float(status.get("target_last_seen_age", 999.0)),
            seen_count=int(status.get("seen_count", 0) or 0),
            track_age_s=float(status.get("track_age_s", 0.0) or 0.0),
            has_appearance=bool(status.get("target_has_appearance", False)),
            status=status.get("target_state", "UNKNOWN"),
            updated_at=now,
        )


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None
