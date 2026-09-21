import unittest

from research_agent.integrations.autotraining import parse_request, to_response, to_task


class TestAutotraining(unittest.TestCase):
    def test_request_strips_hidden_answers(self):
        request = parse_request(
            {
                "request_id": "at-1",
                "question": "Does BoostNet's gain need extra data?",
                "research_context": {"paper": "BoostNet", "answer": "should-not-leak"},
                "environment_id": "synthetic-dev",
                "budget": {"max_explore_calls": 6},
            }
        )
        assert "answer" not in request.research_context
        task = to_task(request)
        assert task.question.startswith("Does BoostNet")
        assert task.budget.max_explore_calls == 6
