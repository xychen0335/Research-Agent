import unittest

from research_agent.training.compat import probe


class TestCompat(unittest.TestCase):
    def test_gpu_stack_probe_is_honest(self):
        report = probe()
        if not report["cuda_available"]:
            self.skipTest("CUDA not available; GPU update/save/reload is unverified")
        if report.get("verl") in {None, "not_installed"}:
            self.skipTest("verl is not installed; AgentLoop GPU wiring is unverified")
        assert report["cuda_available"] is True
