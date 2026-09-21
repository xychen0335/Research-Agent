import json

from research_agent.contracts import TOOL_SEARCH, Budget, TaskInput
from research_agent.data.validate import FORBIDDEN_PUBLIC_KEYS
from research_agent.harness.loop import run_episode
from research_agent.models.scripted import ScriptedPolicy, tool_call

from tests.support import HarnessAsyncTestCase


class TestLabelIsolation(HarnessAsyncTestCase):
    def test_public_tasks_omit_labels(self):
        tasks_path = self.prepared.output_dir / "public" / "tasks.jsonl"
        for line in tasks_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            for key in FORBIDDEN_PUBLIC_KEYS:
                assert not row.get(key), (row["task_id"], key)

    def test_corpus_omits_label_keys(self):
        for line in (self.prepared.output_dir / "public" / "corpus.jsonl").read_text(encoding="utf-8").splitlines():
            assert "golden_answers" not in line
            assert "gold_evidence_ids" not in line

    async def test_observations_do_not_carry_gold(self):
        task = TaskInput(
            request_id="bio-001",
            task_id="bio-001",
            question="Which gene is mutated in childhood retinoblastoma?",
            environment_id="synthetic-dev",
            budget=Budget(),
        )
        model = ScriptedPolicy(
            {
                task.question: [
                    tool_call(TOOL_SEARCH, {"query": "retinoblastoma RB1"}),
                    tool_call("open", {"doc_id": "bio:rb1", "start": 0, "end": 1}),
                    tool_call("submit", {"answer": "RB1", "citations": ["bio:rb1:0"]}),
                ]
            }
        )
        record = await run_episode(task, model, self.tools)
        for event in record.events:
            blob = json.dumps(event.payload)
            assert "golden_answers" not in blob
            assert "gold_evidence_ids" not in blob
            assert "GradingSpec" not in blob
