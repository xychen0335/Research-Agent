import json
from unittest.mock import patch

from research_agent.data.prepare import prepare_papersearchqa_dev
from research_agent.data.papersearchqa import attach_abstracts, sample_indices
from research_agent.training.grpo import GRPOConfig, _collect_groups
from research_agent.models.scripted import ScriptedPolicy, tool_call

from tests.support import HarnessAsyncTestCase


class TestPapersearchqaDev(HarnessAsyncTestCase):
    def test_sample_indices_are_deterministic(self):
        a = sample_indices(5, seed=20260919, size=100)
        b = sample_indices(5, seed=20260919, size=100)
        c = sample_indices(5, seed=1, size=100)
        assert a == b
        assert a != c
        assert len(set(a)) == 5

    def test_attach_abstracts_drops_missing_pmid(self):
        kept, excluded = attach_abstracts(
            [{"pmid": "1", "question": "q", "answer": "a"}, {"pmid": "2", "question": "q2", "answer": "b"}],
            {"1": {"pmid": "1", "title": "t", "abstract": "body"}},
        )
        assert len(kept) == 1
        assert kept[0]["abstract"] == "body"
        assert excluded[0]["pmid"] == "2"

    def test_prepare_dev_subset_separates_test(self):
        def fake_sample(n, *, seed, size, exclude=None):
            return list(range(n))

        def fake_load(split, indices, raw_dir):
            rows = []
            for i in indices:
                pmid = f"{split}-{i}"
                rows.append(
                    {
                        "question": f"question {pmid}",
                        "answer": "RB1",
                        "golden_answers": ["RB1"],
                        "pmid": pmid,
                        "paper_title": f"paper {pmid}",
                        "cat": "genetics",
                    }
                )
            return rows, {"loader": "mock", "sha256": "x", "path": "mock"}

        def fake_fetch(pmids, getter=None, **kwargs):
            return {
                pmid: {"pmid": pmid, "title": f"t {pmid}", "abstract": f"RB1 is discussed in {pmid}.", "source": "mock"}
                for pmid in pmids
            }

        with (
            patch(
                "research_agent.data.papersearchqa.resolve_parquet",
                lambda split, raw_dir: raw_dir / f"{split}.parquet",
            ),
            patch("research_agent.data.papersearchqa.parquet_row_count", lambda path: 100),
            patch("research_agent.data.papersearchqa.sample_indices", fake_sample),
            patch("research_agent.data.papersearchqa.load_split_rows", fake_load),
            patch("research_agent.data.pubmed.fetch_abstracts", fake_fetch),
        ):
            prepared = prepare_papersearchqa_dev(
                self.tmp_path / "psqa",
                n_train=4,
                n_test=2,
                n_distractors=3,
                seed=1,
                fetch_fn=lambda pmid: {
                    "pmid": pmid,
                    "title": f"t {pmid}",
                    "abstract": f"RB1 is discussed in {pmid}.",
                    "source": "mock",
                },
            )
        assert prepared.n_tasks == 6
        assert prepared.report["n_train"] == 4
        assert prepared.report["n_test"] == 2
        splits = []
        for line in (prepared.output_dir / "public" / "tasks.jsonl").read_text(encoding="utf-8").splitlines():
            splits.append(json.loads(line)["split"])
        assert splits.count("train") == 4
        assert splits.count("test") == 2
        assert prepared.report["validation"]["ok"] is True

    async def test_grpo_skips_test_split(self):
        from dataclasses import replace

        from research_agent.grading.contracts import GradingSpec

        tasks, specs, tools = self.prepared_stack
        task = next(item for item in tasks if item.public_id() == "bio-001")
        test_task = replace(task, split="test")
        model = ScriptedPolicy(
            {task.question: [tool_call("submit", {"answer": "wrong", "citations": []})]},
            policy_version="grpo-skip-test",
        )
        groups = await _collect_groups(
            [test_task, task],
            {task.public_id(): specs[task.public_id()], test_task.public_id(): GradingSpec(task_id=test_task.public_id(), answer="nope")},
            model,
            tools,
            GRPOConfig(max_prompts=8, group_size=1),
        )
        assert [group["task_id"] for group in groups] == [task.public_id()]
