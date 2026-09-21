import unittest

from research_agent.training.sft import IGNORE_INDEX, assistant_chat_labels


class _ToyTokenizer:
    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        text = ""
        for item in messages:
            text += f"<{item['role']}>{item['content']}</{item['role']}>"
        if add_generation_prompt:
            text += "<assistant>"
        return list(range(len(text)))


class TestSftLabels(unittest.TestCase):
    def test_chat_template_labels_cover_only_assistant_span(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q"},
            {"role": "assistant", "content": "a"},
        ]
        ids, labels = assistant_chat_labels(messages, _ToyTokenizer(), max_seq_len=128)
        assert len(ids) == len(labels)
        assert IGNORE_INDEX in labels
        assert any(item != IGNORE_INDEX for item in labels)
        assistant = "<assistant>a</assistant>"
        start = len(ids) - len(assistant)
        assert labels[start:] == ids[start:]
        assert all(item == IGNORE_INDEX for item in labels[:start])
