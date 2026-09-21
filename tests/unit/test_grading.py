import unittest

from research_agent.contracts import Result, Usage
from research_agent.grading.answers import score_answer
from research_agent.grading.contracts import GradingSpec
from research_agent.grading.reward import combine_reward, score_episode


class TestGrading(unittest.TestCase):
    def test_normalize_and_alias_match(self):
        spec = GradingSpec(task_id="t", answer="RB1", aliases=("Rb1", "RB1 gene"))
        score, matched = score_answer(" rb1 ", spec)
        assert score == 1.0
        assert matched is not None

    def test_unanswerable_accepts_unknown(self):
        spec = GradingSpec(task_id="t", answer="unknown", answerable=False)
        score, _ = score_answer("Unknown", spec)
        assert score == 1.0
        score_wrong, _ = score_answer("RB1", spec)
        assert score_wrong == 0.0

    def test_lambda_zero_keeps_answer_score(self):
        assert combine_reward(1.0, cost=1.0, cost_lambda=0.0) == 1.0
        assert combine_reward(1.0, cost=1.0, cost_lambda=0.5) == 0.5

    def test_padded_exact_match_is_zero_but_alias_span_is_found(self):
        spec = GradingSpec(task_id="t", answer="Acute myeloid leukemia", aliases=("AML",))
        padded = "Acute myeloid leukemia (AML) and lymphoma are frequently associated."
        score, _ = score_answer(padded, spec)
        assert score == 0.0
        from research_agent.grading.answers import alias_span

        assert alias_span(padded, spec) == "AML"

    def test_unread_citations_are_legal_rate_zero(self):
        spec = GradingSpec(task_id="bio-001", answer="RB1", gold_evidence_ids=("bio:rb1:0",))
        result = Result(
            answer="RB1",
            claims=[],
            citations=["bio:rb1:0"],
            conditions=[],
            unresolved_questions=[],
            status="completed",
            trajectory_id="e",
            policy_version="p",
            harness_version="h",
            environment_version="v",
            usage=Usage(explore_calls=2),
            termination="submitted",
            unread_citations=["bio:rb1:0"],
        )
        score = score_episode(result, spec, max_explore=6)
        assert score.answer_score == 1.0
        assert score.legal_citation_rate == 0.0
        assert score.reward == 1.0
