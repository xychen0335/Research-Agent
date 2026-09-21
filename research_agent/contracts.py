"""Public contracts for tasks, actions, observations, results, and events.

This module must not import training frameworks, graders, or gold labels.
Hidden answers live in `research_agent.grading.contracts`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any


TERMINATION_SUBMITTED = "submitted"
TERMINATION_BUDGET = "budget_exhausted"
TERMINATION_CONTEXT = "context_limit"
TERMINATION_INVALID = "invalid_action_limit"
TERMINATION_INFRA = "tool_infra_failure"

STATUS_COMPLETED = "completed"
STATUS_BUDGET = "budget_exhausted"
STATUS_TOOL_FAILURE = "tool_failure"
STATUS_INSUFFICIENT_EVIDENCE = "insufficient_evidence"

ERROR_INVALID_ACTION = "invalid_action"
ERROR_INVALID_PARAMS = "invalid_params"
ERROR_INFRA = "infra"
ERROR_EMPTY = "empty_result"

TOOL_SEARCH = "search"
TOOL_OPEN = "open"
TOOL_SUBMIT = "submit"


def to_plain(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return {k: to_plain(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    return value


@dataclass(frozen=True)
class Budget:
    max_explore_calls: int = 6
    max_submit_calls: int = 1
    max_observation_chars: int = 4000
    max_context_tokens: int = 8192
    max_generation_tokens: int = 1024
    max_open_paragraphs: int = 4
    search_topk: int = 3
    max_invalid_actions: int = 4


@dataclass
class TaskInput:
    request_id: str
    question: str
    environment_id: str
    budget: Budget = field(default_factory=Budget)
    research_context: dict[str, Any] = field(default_factory=dict)
    task_id: str | None = None
    split: str | None = None

    def public_id(self) -> str:
        return self.task_id or self.request_id


@dataclass
class Action:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    raw_text: str = ""


@dataclass
class Observation:
    tool: str
    content: dict[str, Any] = field(default_factory=dict)
    truncated: bool = False
    error: str | None = None
    error_kind: str | None = None
    infra_failure: bool = False

    def as_text(self) -> str:
        import json

        payload = {
            "tool": self.tool,
            "content": self.content,
            "truncated": self.truncated,
            "error": self.error,
            "error_kind": self.error_kind,
            "infra_failure": self.infra_failure,
        }
        return json.dumps(payload, ensure_ascii=False, sort_keys=True)


@dataclass
class Claim:
    text: str
    citation_ids: list[str] = field(default_factory=list)


@dataclass
class Usage:
    explore_calls: int = 0
    search_calls: int = 0
    open_calls: int = 0
    submit_calls: int = 0
    invalid_actions: int = 0
    generation_turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    observation_chars: int = 0
    latency_ms: float = 0.0
    context_truncated: bool = False


@dataclass
class Result:
    answer: str
    claims: list[Claim]
    citations: list[str]
    conditions: list[str]
    unresolved_questions: list[str]
    status: str
    trajectory_id: str
    policy_version: str
    harness_version: str
    environment_version: str
    usage: Usage
    termination: str
    unread_citations: list[str] = field(default_factory=list)


@dataclass
class Event:
    event_type: str
    timestamp: str
    episode_id: str
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class Generation:
    text: str
    token_ids: list[int] | None = None
    logprobs: list[float] | None = None
    prompt_token_ids: list[int] | None = None
    finish_reason: str = "stop"
    prompt_tokens: int = 0
    completion_tokens: int = 0
    usable_for_rl: bool = False


@dataclass
class TokenTrace:
    prompt_ids: list[int] = field(default_factory=list)
    response_ids: list[int] = field(default_factory=list)
    response_mask: list[int] = field(default_factory=list)
    response_logprobs: list[float] | None = None
    usable_for_rl: bool = True

    def append_generation(self, generation: Generation) -> None:
        if generation.prompt_token_ids and not self.prompt_ids:
            self.prompt_ids = list(generation.prompt_token_ids)
        if generation.token_ids is None:
            self.usable_for_rl = False
            return
        ids = list(generation.token_ids)
        self.response_ids.extend(ids)
        self.response_mask.extend([1] * len(ids))
        if generation.logprobs is None:
            if self.response_logprobs is not None:
                self.response_logprobs.extend([0.0] * len(ids))
        else:
            if self.response_logprobs is None:
                self.response_logprobs = [0.0] * (len(self.response_ids) - len(ids))
            self.response_logprobs.extend(list(generation.logprobs)[: len(ids)])
            if len(generation.logprobs) < len(ids):
                self.response_logprobs.extend([0.0] * (len(ids) - len(generation.logprobs)))

    def append_observation_tokens(self, token_ids: list[int]) -> None:
        self.response_ids.extend(token_ids)
        self.response_mask.extend([0] * len(token_ids))
        if self.response_logprobs is not None:
            self.response_logprobs.extend([0.0] * len(token_ids))
