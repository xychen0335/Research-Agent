from research_agent.contracts import (
    TOOL_SEARCH,
    Budget,
    Observation,
    TaskInput,
)
from research_agent.harness.loop import parse_tool_calls, run_episode
from research_agent.harness.state import EpisodeState
from research_agent.models.scripted import ScriptedPolicy, tool_call

from tests.support import HarnessAsyncTestCase


class TestBudget(HarnessAsyncTestCase):
    def test_parse_tool_call_json(self):
        text = 'prefix\n<tool_call>{"name": "search", "arguments": {"query": "RB1"}}</tool_call>'
        actions = parse_tool_calls(text)
        assert actions[0].name == "search"
        assert actions[0].arguments["query"] == "RB1"

    async def test_budget_stops_without_submit(self):
        budget = Budget(max_explore_calls=2, max_invalid_actions=8)
        task = TaskInput(
            request_id="budget",
            task_id="budget",
            question="Which gene is mutated in childhood retinoblastoma?",
            environment_id="test",
            budget=budget,
        )
        model = ScriptedPolicy(
            {
                task.question: [
                    tool_call(TOOL_SEARCH, {"query": "retinoblastoma"}),
                    tool_call(TOOL_SEARCH, {"query": "RB1 gene"}),
                    tool_call(TOOL_SEARCH, {"query": "should not run"}),
                ]
            },
            policy_version="scripted-budget",
        )
        record = await run_episode(task, model, self.tools)
        assert record.result.termination == "budget_exhausted"
        assert record.result.usage.explore_calls == 2
        assert record.result.usage.submit_calls == 0
        assert record.result.status == "budget_exhausted"

    async def test_invalid_action_limit(self):
        task = TaskInput(
            request_id="bad",
            question="Which gene is mutated in childhood retinoblastoma?",
            environment_id="test",
            budget=Budget(max_invalid_actions=2, max_explore_calls=6),
        )
        model = ScriptedPolicy({task.question: ["not a tool call", "still not", "nope"]}, policy_version="bad")
        record = await run_episode(task, model, self.tools)
        assert record.result.termination == "invalid_action_limit"
        assert record.result.usage.invalid_actions >= 2

    def test_submit_after_termination_state(self):
        state = EpisodeState(
            task=TaskInput(request_id="x", question="q", environment_id="e"),
            episode_id="e1",
            policy_version="p",
            harness_version="h",
            environment_version="v",
        )
        obs = Observation(tool="search", content={"hits": []})
        state.record_explore("search", obs)
        assert not state.terminated
