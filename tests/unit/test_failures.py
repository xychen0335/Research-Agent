import unittest

from research_agent.contracts import Budget, Result, TaskInput, TokenTrace, Usage
from research_agent.evaluation.failures import summarize_failures
from research_agent.grading.contracts import Score
from research_agent.harness.trajectory import EpisodeRecord, make_event
from research_agent.models.scripted import tool_call


def _score(**kwargs):
    defaults = dict(task_id="t", answer_score=0.0, legal_citation_rate=0.0, evidence_support=None, reward=0.0, matched_alias=None)
    defaults.update(kwargs)
    return Score(**defaults)


def _task() -> TaskInput:
    return TaskInput(request_id="t", question="q", environment_id="e", budget=Budget(), task_id="t")


class TestFailures(unittest.TestCase):
    def test_summarize_failures_counts_missing_tool_calls(self):
        result = Result(
            answer="",
            claims=[],
            citations=[],
            conditions=[],
            unresolved_questions=[],
            status="budget_exhausted",
            trajectory_id="e1",
            policy_version="x",
            harness_version="0.1.0",
            environment_version="e",
            usage=Usage(invalid_actions=1, generation_turns=1, search_calls=0, open_calls=0, explore_calls=0),
            termination="budget_exhausted",
        )
        record = EpisodeRecord(
            episode_id="e1",
            task=_task(),
            events=[
                make_event("model_generation", "e1", {"text": "I guess RB1"}),
                make_event("observation", "e1", {"tool": "parser", "error": "no tool_call block", "error_kind": "invalid_action"}),
            ],
            result=result,
            token_trace=TokenTrace(),
        )
        summary = summarize_failures(
            [record],
            [_score()],
        )
        assert summary["reasons"]["no_tool_call"] == 1
        assert summary["termination"]["budget_exhausted"] == 1

    def test_summarize_failures_flags_wrong_answer(self):
        result = Result(
            answer="BRCA1",
            claims=[],
            citations=["d:0"],
            conditions=[],
            unresolved_questions=[],
            status="completed",
            trajectory_id="e2",
            policy_version="x",
            harness_version="0.1.0",
            environment_version="e",
            usage=Usage(generation_turns=1, submit_calls=1),
            termination="submitted",
        )
        record = EpisodeRecord(
            episode_id="e2",
            task=_task(),
            events=[
                make_event("model_generation", "e2", {"text": tool_call("submit", {"answer": "BRCA1", "citations": ["d:0"]})}),
                make_event("observation", "e2", {"tool": "submit", "error": None}),
            ],
            result=result,
            token_trace=TokenTrace(),
        )
        summary = summarize_failures(
            [record],
            [_score(legal_citation_rate=1.0, evidence_support=1.0)],
        )
        assert summary["reasons"]["wrong_answer"] == 1
        assert summary["submit_rate"] == 1.0
