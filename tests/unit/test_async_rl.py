from research_agent.training.async_rl import (
    AsyncRLConfig,
    AsyncRolloutQueue,
    QueuedEpisode,
    async_rl_gate,
)

from tests.support import TempDirTestCase


def _episode(group_id: str, step: int, *, version: str = "p1", reward: float = 0.0) -> QueuedEpisode:
    return QueuedEpisode(
        episode_id=f"{group_id}-{step}",
        group_id=group_id,
        policy_version=version,
        policy_step=step,
        reward=reward,
        usable_for_rl=True,
    )


class TestAsyncRl(TempDirTestCase):
    def test_async_rl_gate_blocks_without_profiling(self):
        report = async_rl_gate(self.tmp_path / "missing.json")
        assert report["enabled"] is False
        assert report["status"] == "blocked"

    def test_async_rl_gate_requires_explicit_allow(self):
        path = self.tmp_path / "profiling.json"
        path.write_text('{"gpu_hours": 1}\n', encoding="utf-8")
        assert async_rl_gate(path)["enabled"] is False
        path.write_text('{"allow_async_rl": true, "gpu_hours": 1}\n', encoding="utf-8")
        assert async_rl_gate(path)["enabled"] is True

    def test_queue_drops_stale_and_emits_same_version_groups(self):
        queue = AsyncRolloutQueue(AsyncRLConfig(max_queue=8, max_policy_lag=1, group_size=2))
        assert queue.enqueue(_episode("g1", 0), trainer_step=0) == "queued"
        assert queue.enqueue(_episode("g1", 0), trainer_step=0) == "queued"
        groups = queue.pop_ready_groups()
        assert len(groups) == 1
        assert len(groups[0]) == 2
        assert queue.enqueue(_episode("g2", 0), trainer_step=4) == "stale"
        assert queue.report()["dropped_stale"] == 1

    def test_queue_rejects_mixed_policy_versions(self):
        queue = AsyncRolloutQueue(AsyncRLConfig(group_size=2, max_queue=8))
        queue.enqueue(_episode("g1", 0, version="a"), trainer_step=0)
        queue.enqueue(_episode("g1", 0, version="b"), trainer_step=0)
        assert queue.pop_ready_groups() == []
        assert queue.report()["dropped_mixed_version"] == 2
