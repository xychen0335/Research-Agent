"""Bounded async RL queue. Full actor-learner overlap stays off until GPU profiling exists."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AsyncRLConfig:
    max_queue: int = 8
    max_policy_lag: int = 2
    group_size: int = 4
    drop_stale: bool = True


@dataclass
class QueuedEpisode:
    episode_id: str
    group_id: str
    policy_version: str
    policy_step: int
    reward: float
    usable_for_rl: bool
    payload: dict[str, Any] = field(default_factory=dict)


class AsyncRolloutQueue:
    """Collect same-question groups without mixing policy versions."""

    def __init__(self, config: AsyncRLConfig | None = None):
        self.config = config or AsyncRLConfig()
        self._items: list[QueuedEpisode] = []
        self.dropped_stale = 0
        self.dropped_backpressure = 0
        self.dropped_mixed_version = 0
        self.groups_emitted = 0

    def enqueue(self, item: QueuedEpisode, *, trainer_step: int) -> str:
        lag = trainer_step - item.policy_step
        if self.config.drop_stale and abs(lag) > self.config.max_policy_lag:
            self.dropped_stale += 1
            return "stale"
        if len(self._items) >= self.config.max_queue:
            self.dropped_backpressure += 1
            return "backpressure"
        self._items.append(item)
        return "queued"

    def pop_ready_groups(self) -> list[list[QueuedEpisode]]:
        by_group: dict[str, list[QueuedEpisode]] = {}
        for item in self._items:
            by_group.setdefault(item.group_id, []).append(item)
        ready: list[list[QueuedEpisode]] = []
        remaining: list[QueuedEpisode] = []
        for members in by_group.values():
            versions = {item.policy_version for item in members}
            if len(versions) != 1:
                self.dropped_mixed_version += len(members)
                continue
            if len(members) >= self.config.group_size:
                ready.append(members[: self.config.group_size])
                remaining.extend(members[self.config.group_size :])
            else:
                remaining.extend(members)
        self._items = remaining
        self.groups_emitted += len(ready)
        return ready

    def report(self) -> dict[str, Any]:
        pending_groups = len({item.group_id for item in self._items})
        return {
            "queued": len(self._items),
            "pending_groups": pending_groups,
            "groups_emitted": self.groups_emitted,
            "dropped_stale": self.dropped_stale,
            "dropped_backpressure": self.dropped_backpressure,
            "dropped_mixed_version": self.dropped_mixed_version,
            "max_queue": self.config.max_queue,
            "max_policy_lag": self.config.max_policy_lag,
        }


def async_rl_gate(profiling_path: Path | None = None) -> dict[str, Any]:
    """PLAN §并发采样与异步 RL: full overlap is not a default deliverable."""
    path = profiling_path or Path("outputs/gpu-mini/profiling.json")
    if not path.exists():
        return {
            "enabled": False,
            "status": "blocked",
            "reason": "no GPU profiling report; full async RL waits on measured bottlenecks",
            "path": str(path),
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or not payload.get("allow_async_rl"):
        return {
            "enabled": False,
            "status": "blocked",
            "reason": "profiling report does not set allow_async_rl",
            "path": str(path),
            "profiling": payload if isinstance(payload, dict) else None,
        }
    return {
        "enabled": True,
        "status": "allowed",
        "path": str(path),
        "profiling": payload,
    }
