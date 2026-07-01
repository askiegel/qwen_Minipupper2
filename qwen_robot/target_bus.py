#!/usr/bin/env python3
"""
Target Bus

Lightweight in-process pub/sub for TargetSnapshot.

Every subsystem subscribes.
Vision publishes.
"""

from typing import Callable, List
from .target_snapshot import TargetSnapshot


class TargetBus:

    def __init__(self):
        self._snapshot = TargetSnapshot.empty()
        self._subscribers: List[Callable[[TargetSnapshot], None]] = []

    def publish(self, snapshot: TargetSnapshot):
        self._snapshot = snapshot

        for callback in list(self._subscribers):
            try:
                callback(snapshot)
            except Exception as e:
                print(f"TargetBus subscriber error: {e}")

    def subscribe(self, callback):
        self._subscribers.append(callback)

    def unsubscribe(self, callback):
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def latest(self):
        return self._snapshot


target_bus = TargetBus()
