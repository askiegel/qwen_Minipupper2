#!/usr/bin/env python3
"""
Navigation Manager - v0.8 Alpha Sprint 2

Purpose:
- Provide a navigation state machine.
- Prepare Go-To-Last-Seen behavior.
- Output telemetry for Dashboard Developer Mode.
- Avoid directly commanding motors for now.

States:
IDLE
TRACKING
GO_TO_LAST_SEEN
SEARCHING
ARRIVED
FAILED
"""

from dataclasses import dataclass, asdict
from enum import Enum
import time
from typing import Optional, Dict, Any


class NavigationState(str, Enum):
    IDLE = "IDLE"
    TRACKING = "TRACKING"
    GO_TO_LAST_SEEN = "GO_TO_LAST_SEEN"
    SEARCHING = "SEARCHING"
    ARRIVED = "ARRIVED"
    FAILED = "FAILED"


class NavigationAction(str, Enum):
    NONE = "NONE"
    KEEP_TRACKING = "KEEP_TRACKING"
    GO_TO_LAST_SEEN = "GO_TO_LAST_SEEN"
    SEARCH_ROTATE = "SEARCH_ROTATE"
    STOP = "STOP"
    FAILED = "FAILED"


@dataclass
class NavigationConfig:
    lost_timeout_s: float = 1.5
    last_seen_max_age_s: float = 20.0
    arrived_distance_m: float = 0.75
    search_duration_s: float = 12.0


@dataclass
class NavigationTelemetry:
    state: str = NavigationState.IDLE.value
    action: str = NavigationAction.NONE.value
    active_target_id: Optional[str] = None
    target_visible: bool = False
    last_seen_age_s: Optional[float] = None
    last_seen_range_m: Optional[float] = None
    last_seen_bearing_deg: Optional[float] = None
    confidence: float = 0.0
    status: str = "idle"
    updated_at: float = 0.0


class NavigationManager:
    def __init__(self, config: Optional[NavigationConfig] = None):
        self.config = config or NavigationConfig()
        self.state = NavigationState.IDLE
        self.active_target_id: Optional[str] = None

        self._last_visible_time: Optional[float] = None
        self._search_start_time: Optional[float] = None

        self.telemetry = NavigationTelemetry(updated_at=time.time())

    def reset(self) -> Dict[str, Any]:
        self.state = NavigationState.IDLE
        self.active_target_id = None
        self._last_visible_time = None
        self._search_start_time = None
        return self._set_telemetry(
            action=NavigationAction.NONE,
            status="navigation reset",
            confidence=0.0,
        )

    def update(
        self,
        target_visible: bool,
        target_id: Optional[str] = None,
        last_seen_age_s: Optional[float] = None,
        last_seen_range_m: Optional[float] = None,
        last_seen_bearing_deg: Optional[float] = None,
        confidence: float = 0.0,
    ) -> Dict[str, Any]:

        now = time.time()

        if target_visible:
            self.active_target_id = target_id or self.active_target_id
            self._last_visible_time = now
            self._search_start_time = None
            self.state = NavigationState.TRACKING

            return self._set_telemetry(
                action=NavigationAction.KEEP_TRACKING,
                target_visible=True,
                target_id=self.active_target_id,
                last_seen_age_s=0.0,
                last_seen_range_m=last_seen_range_m,
                last_seen_bearing_deg=last_seen_bearing_deg,
                confidence=confidence,
                status="target visible; continue tracking",
            )

        lost_for = None
        if self._last_visible_time is not None:
            lost_for = now - self._last_visible_time

        has_recent_memory = (
            last_seen_age_s is not None
            and last_seen_age_s <= self.config.last_seen_max_age_s
        )

        if self.state == NavigationState.IDLE:
            self.state = NavigationState.SEARCHING
            self._search_start_time = now

            return self._set_telemetry(
                action=NavigationAction.SEARCH_ROTATE,
                target_visible=False,
                target_id=self.active_target_id,
                last_seen_age_s=last_seen_age_s,
                last_seen_range_m=last_seen_range_m,
                last_seen_bearing_deg=last_seen_bearing_deg,
                confidence=0.2,
                status="active mission; searching for first target lock",
            )

        if has_recent_memory and last_seen_range_m is not None:
            if last_seen_range_m <= self.config.arrived_distance_m:
                self.state = NavigationState.ARRIVED
                self._search_start_time = now

                return self._set_telemetry(
                    action=NavigationAction.STOP,
                    target_visible=False,
                    target_id=self.active_target_id,
                    last_seen_age_s=last_seen_age_s,
                    last_seen_range_m=last_seen_range_m,
                    last_seen_bearing_deg=last_seen_bearing_deg,
                    confidence=0.7,
                    status="arrived at last seen location",
                )

            self.state = NavigationState.GO_TO_LAST_SEEN

            return self._set_telemetry(
                action=NavigationAction.GO_TO_LAST_SEEN,
                target_visible=False,
                target_id=self.active_target_id,
                last_seen_age_s=last_seen_age_s,
                last_seen_range_m=last_seen_range_m,
                last_seen_bearing_deg=last_seen_bearing_deg,
                confidence=0.6,
                status="target lost; navigate to last seen location",
            )

        if self.state == NavigationState.ARRIVED:
            self.state = NavigationState.SEARCHING
            self._search_start_time = now

        if self.state == NavigationState.SEARCHING:
            if self._search_start_time is None:
                self._search_start_time = now

            if now - self._search_start_time <= self.config.search_duration_s:
                return self._set_telemetry(
                    action=NavigationAction.SEARCH_ROTATE,
                    target_visible=False,
                    target_id=self.active_target_id,
                    last_seen_age_s=last_seen_age_s,
                    confidence=0.4,
                    status="searching near last seen location",
                )

        if lost_for is not None and lost_for < self.config.lost_timeout_s:
            return self._set_telemetry(
                action=NavigationAction.NONE,
                target_visible=False,
                target_id=self.active_target_id,
                last_seen_age_s=last_seen_age_s,
                confidence=0.2,
                status="brief target loss; waiting",
            )

        if self.active_target_id is None:
            self.state = NavigationState.SEARCHING
            return self._set_telemetry(
                action=NavigationAction.SEARCH_ROTATE,
                target_visible=False,
                target_id=self.active_target_id,
                last_seen_age_s=last_seen_age_s,
                last_seen_range_m=last_seen_range_m,
                last_seen_bearing_deg=last_seen_bearing_deg,
                confidence=0.2,
                status="active mission; still searching for first target lock",
            )

        self.state = NavigationState.FAILED
        return self._set_telemetry(
            action=NavigationAction.FAILED,
            target_visible=False,
            target_id=self.active_target_id,
            last_seen_age_s=last_seen_age_s,
            confidence=0.0,
            status="navigation failed; no recent usable target memory",
        )

    def _set_telemetry(
        self,
        action: NavigationAction,
        status: str,
        confidence: float,
        target_visible: bool = False,
        target_id: Optional[str] = None,
        last_seen_age_s: Optional[float] = None,
        last_seen_range_m: Optional[float] = None,
        last_seen_bearing_deg: Optional[float] = None,
    ) -> Dict[str, Any]:

        self.telemetry = NavigationTelemetry(
            state=self.state.value,
            action=action.value,
            active_target_id=target_id,
            target_visible=target_visible,
            last_seen_age_s=last_seen_age_s,
            last_seen_range_m=last_seen_range_m,
            last_seen_bearing_deg=last_seen_bearing_deg,
            confidence=round(confidence, 3),
            status=status,
            updated_at=time.time(),
        )
        return self.get_telemetry()

    def get_telemetry(self) -> Dict[str, Any]:
        return asdict(self.telemetry)
