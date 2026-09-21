"""Token-in token-out PolicyModel wrapping verl's LLMServerClient.generate.

This is not a trainer. Sampling logprobs come from the engine; retraining-time
recomputation is not a substitute. Chat-template tokenization uses the tokenizer
verl already attached to the Agent Loop. The hash codec is only a CPU stand-in.
"""

from __future__ import annotations

from typing import Any, Awaitable, Callable

from research_agent.contracts import Generation
from research_agent.harness.context import estimate_tokens
from research_agent.models.base import encode_text


GenerateFn = Callable[[list[int], dict[str, Any]], Awaitable[dict[str, Any]]]


def _as_id_list(value: Any) -> list[int]:
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        value = value.get("input_ids") or value.get("token_ids") or []
    if value and isinstance(value, list) and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


class VerlTokenPolicy:
    def __init__(
        self,
        generate_fn: GenerateFn | None = None,
        *,
        policy_version: str,
        sampling_params: dict[str, Any] | None = None,
        tokenizer: Any | None = None,
        chat_template_kwargs: dict[str, Any] | None = None,
    ):
        self.generate_fn = generate_fn
        self.policy_version = policy_version
        self.sampling_params = sampling_params or {"temperature": 1.0}
        self.tokenizer = tokenizer
        self.chat_template_kwargs = dict(chat_template_kwargs or {})

    async def generate(self, messages: list[dict[str, str]]) -> Generation:
        prompt_ids = self.encode_messages(messages)
        if self.generate_fn is None:
            raise RuntimeError("verl generate_fn is not attached; GPU rollout has not been measured")
        payload = await self.generate_fn(prompt_ids, self.sampling_params)
        token_ids = _as_id_list(payload.get("token_ids"))
        logprobs = payload.get("logprobs")
        text = str(payload.get("text") or "")
        if not text:
            text = self.decode_ids(token_ids)
        return Generation(
            text=text,
            token_ids=token_ids,
            logprobs=list(logprobs) if logprobs is not None else None,
            prompt_token_ids=prompt_ids,
            finish_reason=str(payload.get("finish_reason") or "stop"),
            prompt_tokens=len(prompt_ids),
            completion_tokens=len(token_ids),
            usable_for_rl=True,
        )

    def encode_messages(self, messages: list[dict[str, str]]) -> list[int]:
        tokenizer = self.tokenizer
        apply = getattr(tokenizer, "apply_chat_template", None) if tokenizer is not None else None
        if callable(apply):
            encoded = apply(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                **self.chat_template_kwargs,
            )
            ids = _as_id_list(encoded)
            if ids:
                return ids
        blob = "\n".join(item.get("content", "") for item in messages)
        return self.encode_text(blob)

    def encode_text(self, text: str) -> list[int]:
        tokenizer = self.tokenizer
        encode = getattr(tokenizer, "encode", None) if tokenizer is not None else None
        if callable(encode):
            return _as_id_list(encode(text, add_special_tokens=False))
        return encode_text(text)

    def decode_ids(self, token_ids: list[int]) -> str:
        tokenizer = self.tokenizer
        decode = getattr(tokenizer, "decode", None) if tokenizer is not None else None
        if callable(decode) and token_ids:
            return str(decode(token_ids, skip_special_tokens=True) or "")
        return ""

    def count_text(self, text: str) -> int:
        if self.tokenizer is not None:
            return max(1, len(self.encode_text(text)))
        return estimate_tokens(text)
