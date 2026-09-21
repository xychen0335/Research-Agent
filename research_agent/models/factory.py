"""Assemble a PolicyModel from a named backend. Harness stays backend-agnostic."""

from __future__ import annotations

import os
from typing import Any

from research_agent.models.openai_compatible import OpenAICompatiblePolicy
from research_agent.models.scripted import ScriptedPolicy

OLLAMA_DEFAULT_BASE = "http://127.0.0.1:11434/v1"
OLLAMA_DEFAULT_MODEL = "qwen3.5:4b"
HF_DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


def load_policy(
    kind: str,
    *,
    model_name: str | None = None,
    base_url: str | None = None,
    policy_version: str | None = None,
    scripts: dict[str, Any] | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    adapter: str | None = None,
    local_files_only: bool = False,
    trainable_adapter: bool = False,
) -> Any:
    name = (kind or "scripted").strip().lower().replace("-", "_")
    if name == "scripted":
        return ScriptedPolicy(scripts or {}, policy_version=policy_version or "scripted")
    if name in {"openai_compatible", "openai", "ollama"}:
        resolved_model = model_name or (
            OLLAMA_DEFAULT_MODEL if name == "ollama" else os.environ.get("RESEARCH_AGENT_MODEL") or HF_DEFAULT_MODEL
        )
        resolved_url = base_url or os.environ.get("RESEARCH_AGENT_BASE_URL")
        if name == "ollama":
            resolved_url = resolved_url or OLLAMA_DEFAULT_BASE
            if not model_name or model_name == HF_DEFAULT_MODEL:
                resolved_model = OLLAMA_DEFAULT_MODEL
        think = True
        return OpenAICompatiblePolicy(
            model=resolved_model,
            base_url=resolved_url,
            policy_version=policy_version,
            temperature=temperature,
            max_tokens=max_tokens,
            think=think,
        )
    if name in {"huggingface", "hf"}:
        from research_agent.models.huggingface import HuggingFacePolicy

        resolved_model = model_name or os.environ.get("RESEARCH_AGENT_HF_MODEL") or HF_DEFAULT_MODEL
        return HuggingFacePolicy(
            model_name=resolved_model,
            policy_version=policy_version or resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            adapter=adapter,
            local_files_only=local_files_only,
            trainable_adapter=trainable_adapter,
        )
    raise ValueError(f"unknown policy kind: {kind}")
