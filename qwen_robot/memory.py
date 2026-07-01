#!/usr/bin/env python3

"""
Memory Manager
v0.8 Alpha

Stores the latest TargetSnapshot and a history of observations.
"""

import copy
import time

from .target_bus import target_bus
from .target_snapshot import TargetSnapshot


class MemoryManager:

    def __init__(self):

        self.current = TargetSnapshot.empty()

        self.history = []

        self.max_history = 500

        target_bus.subscribe(self.on_snapshot)

    def on_snapshot(self, snapshot: TargetSnapshot):

        self.current = snapshot

        self.history.append(copy.deepcopy(snapshot))

        if len(self.history) > self.max_history:
            self.history.pop(0)

    def latest(self):

        return self.current

    def has_recent_target(self, timeout=10.0):

        if not self.current.visible:

            if self.current.last_seen_age_s is None:
                return False

            return self.current.last_seen_age_s < timeout

        return True

    def last_seen(self):

        return self.current

    def history_count(self):

        return len(self.history)

    def clear(self):

        self.current = TargetSnapshot.empty()

        self.history.clear()

    def diagnostics(self):

        return {

            "history_size": len(self.history),

            "visible": self.current.visible,

            "label": self.current.label,

            "target_id": self.current.target_id,

            "seen_count": self.current.seen_count,

            "last_seen_age":

                self.current.last_seen_age_s,

            "updated":

                self.current.updated_at,

        }


memory_manager = MemoryManager()

