import argparse
import json
import tempfile
import unittest
from unittest.mock import patch

from sql_agent.demo import create_demo
from sql_agent.run import rollout


def response(name, arguments, call_id):
    return {"choices": [{"message": {"role": "assistant", "content": None,
        "tool_calls": [{"id": call_id, "type": "function", "function": {
            "name": name, "arguments": json.dumps(arguments)}}]}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5}}


class RolloutTests(unittest.TestCase):
    def test_native_tool_loop_and_hidden_gold(self):
        with tempfile.TemporaryDirectory() as directory:
            task = json.loads(create_demo(directory).read_text().splitlines()[0])
            # Equivalent SQL with different syntax makes gold leakage detectable.
            prediction = "SELECT SUM(amount) FROM orders WHERE status='paid' AND substr(created_at,1,4)='2025'"
            outputs = iter([
                response("get_schema", {}, "call1"),
                response("execute_sql", {"sql": "SELECT bad_column FROM orders"}, "call2"),
                response("execute_sql", {"sql": prediction}, "call3"),
                response("submit_sql", {"sql": prediction}, "call4"),
            ])
            payloads = []

            def mock_completion(base_url, api_key, payload):
                payloads.append(json.loads(json.dumps(payload)))
                return next(outputs)

            args = argparse.Namespace(base_url="http://unused/v1", model="Qwen/Qwen3.5-4B",
                max_calls=6, max_tokens=1024)
            with patch("sql_agent.run.completion", side_effect=mock_completion):
                result = rollout(task, args)
            self.assertTrue(result["score"]["correct"])
            self.assertEqual(result["tool_calls"], 3)
            self.assertEqual(len(result["usage"]), 4)
            for payload in payloads:
                self.assertNotIn(task["gold_sql"], json.dumps(payload))
                self.assertFalse(payload["chat_template_kwargs"]["enable_thinking"])
            self.assertIn("error", payloads[2]["messages"][-1]["content"])


if __name__ == "__main__":
    unittest.main()
