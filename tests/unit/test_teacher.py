from research_agent.data.teacher import TeacherFilter, generate_teacher_traces
from research_agent.evaluation.baselines import _oracle_scripts
from research_agent.models.scripted import ScriptedPolicy, tool_call

from tests.support import HarnessAsyncTestCase


class TestTeacher(HarnessAsyncTestCase):
    async def test_teacher_keeps_correct_legal_traces(self):
        tasks, specs, tools = self.prepared_stack
        subset = [task for task in tasks if task.public_id() == "bio-001"]
        model = ScriptedPolicy(_oracle_scripts(subset, specs), policy_version="oracle")
        samples = await generate_teacher_traces(subset, specs, model, tools)
        assert samples[0].kept
        assert samples[0].answer_score == 1.0

    async def test_teacher_drops_wrong_answer(self):
        tasks, specs, tools = self.prepared_stack
        task = next(item for item in tasks if item.public_id() == "bio-001")
        model = ScriptedPolicy(
            {task.question: [tool_call("submit", {"answer": "BRCA1", "citations": []})]},
            policy_version="wrong",
        )
        samples = await generate_teacher_traces([task], specs, model, tools, filters=TeacherFilter())
        assert not samples[0].kept
        assert samples[0].reason == "incorrect_answer"

    async def test_teacher_samples_per_task_and_skips_test(self):
        from dataclasses import replace

        tasks, specs, tools = self.prepared_stack
        task = next(item for item in tasks if item.public_id() == "bio-001")
        test_task = replace(task, split="test")
        model = ScriptedPolicy(
            {task.question: [tool_call("submit", {"answer": "BRCA1", "citations": []})]},
            policy_version="wrong",
        )
        seen = []
        samples = await generate_teacher_traces(
            [test_task, task],
            specs,
            model,
            tools,
            samples_per_task=2,
            on_sample=seen.append,
        )
        assert all(sample.record.task.split != "test" for sample in samples)
        assert len(samples) == 2
        assert len(seen) == 2

    def test_refilter_harvests_short_span_without_changing_eval(self):
        from research_agent.data.teacher import refilter_teacher_rows, rewrite_submit_answer
        from research_agent.grading.answers import score_answer
        from research_agent.grading.contracts import GradingSpec
        from research_agent.models.scripted import tool_call

        spec = GradingSpec(task_id="t1", answer="Round block technique", aliases=("RBT",))
        messages = [
            {"role": "assistant", "content": tool_call("open", {"doc_id": "pmid:1", "start": 0, "end": 0})},
            {
                "role": "assistant",
                "content": tool_call(
                    "submit",
                    {"answer": "The round block technique is an oncoplastic method.", "citations": ["pmid:1:0"]},
                ),
            },
        ]
        kept = refilter_teacher_rows(
            [{"task_id": "t1", "answer": "The round block technique is an oncoplastic method.", "citations": ["pmid:1:0"], "messages": messages}],
            {"t1": spec},
        )
        assert len(kept) == 1
        assert kept[0]["answer"] == "Round block technique"
        assert score_answer("The round block technique is an oncoplastic method.", spec)[0] == 0.0
        rewritten = rewrite_submit_answer(messages, "Round block technique")
        assert "Round block technique" in rewritten[-1]["content"]

    def test_tasks_without_kept_drops_successes(self):
        from research_agent.contracts import TaskInput
        from research_agent.data.teacher import tasks_without_kept

        tasks = [
            TaskInput(request_id="a", question="a", environment_id="e", task_id="a"),
            TaskInput(request_id="b", question="b", environment_id="e", task_id="b"),
        ]
        leftover = tasks_without_kept(tasks, [{"task_id": "a"}])
        assert [item.public_id() for item in leftover] == ["b"]
