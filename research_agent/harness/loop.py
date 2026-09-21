"""Single agent interaction loop for training, evaluation, and serving."""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from typing import Any, Protocol

from research_agent import HARNESS_VERSION
from research_agent.contracts import (
    TOOL_OPEN,
    TOOL_SEARCH,
    TOOL_SUBMIT,
    Action,
    Generation,
    Observation,
    TaskInput,
    TokenTrace,
)
from research_agent.harness.context import (
    TokenCounter,
    WhitespaceCounter,
    clip_observation,
    count_messages,
    initial_messages,
    load_system_prompt,
    observation_message,
    truncate_messages,
)
from research_agent.harness.state import EpisodeState
from research_agent.harness.trajectory import EpisodeRecord, build_result, make_event

TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
ALLOWED_TOOLS = {TOOL_SEARCH, TOOL_OPEN, TOOL_SUBMIT}


class PolicyModel(Protocol):
    policy_version: str

    async def generate(self, messages: list[dict[str, str]]) -> Generation: ...

    def encode_text(self, text: str) -> list[int]: ...

    def count_text(self, text: str) -> int: ...


class ToolDispatcher(Protocol):
    environment_version: str

    async def execute(self, action: Action, state: EpisodeState) -> Observation: ...


def parse_tool_calls(text: str) -> list[Action]:
    actions: list[Action] = []
    for match in TOOL_CALL_RE.finditer(text):
        raw = match.group(1).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        name = str(payload.get("name", "")).strip()
        arguments = payload.get("arguments", {})
        if not isinstance(arguments, dict):
            arguments = {}
        if name:
            actions.append(Action(name=name, arguments=arguments, raw_text=match.group(0)))
    return actions


async def run_episode(
    task: TaskInput,
    model: PolicyModel,
    tools: ToolDispatcher,
    *,
    episode_id: str | None = None,
    system_prompt: str | None = None,
    live: bool = True,
) -> EpisodeRecord:
    started = time.perf_counter()
    episode_id = episode_id or str(uuid.uuid4())
    prompt = system_prompt if system_prompt is not None else load_system_prompt()
    counter: TokenCounter = model if hasattr(model, "count_text") else WhitespaceCounter()
    state = EpisodeState(
        task=task,
        episode_id=episode_id,
        policy_version=getattr(model, "policy_version", "unknown"),
        harness_version=HARNESS_VERSION,
        environment_version=getattr(tools, "environment_version", "unknown"),
    )
    events = [
        make_event(
            "episode_start",
            episode_id,
            {
                "request_id": task.request_id,
                "task_id": task.public_id(),
                "question": task.question,
                "environment_id": task.environment_id,
                "budget": {
                    "max_explore_calls": task.budget.max_explore_calls,
                    "max_submit_calls": task.budget.max_submit_calls,
                    "max_context_tokens": task.budget.max_context_tokens,
                },
                "policy_version": state.policy_version,
                "harness_version": state.harness_version,
                "environment_version": state.environment_version,
                "live": live,
            },
        )
    ]
    messages = initial_messages(state, prompt)
    token_trace = TokenTrace()

    while not state.terminated:
        messages, truncated = truncate_messages(messages, counter, task.budget.max_context_tokens)
        if truncated:
            state.mark_context_truncated()
            events.append(
                make_event(
                    "context_truncated",
                    episode_id,
                    {"tokens": count_messages(messages, counter), "limit": task.budget.max_context_tokens},
                )
            )
        if count_messages(messages, counter) > task.budget.max_context_tokens:
            state.force_context_limit()
            break

        generation = await model.generate(messages)
        state.record_generation(generation.prompt_tokens, generation.completion_tokens)
        token_trace.append_generation(generation)
        events.append(
            make_event(
                "model_generation",
                episode_id,
                {
                    "text": generation.text,
                    "usable_for_rl": generation.usable_for_rl,
                    "token_id_count": len(generation.token_ids or []),
                    "finish_reason": generation.finish_reason,
                },
            )
        )
        messages.append({"role": "assistant", "content": generation.text})
        actions = parse_tool_calls(generation.text)
        if not actions:
            observation = Observation(
                tool="parser",
                content={},
                error="no tool_call block",
                error_kind="invalid_action",
            )
            state.record_invalid_action()
            _append_observation(state, messages, events, token_trace, model, observation)
            continue

        # One submit cannot mix with explore tools in the same turn.
        submit_actions = [item for item in actions if item.name == TOOL_SUBMIT]
        explore_actions = [item for item in actions if item.name != TOOL_SUBMIT]
        if submit_actions and explore_actions:
            observation = Observation(
                tool="parser",
                content={"names": [item.name for item in actions]},
                error="submit cannot mix with other tools in the same turn",
                error_kind="invalid_action",
            )
            state.record_invalid_action()
            _append_observation(state, messages, events, token_trace, model, observation)
            continue

        turn_actions = submit_actions or explore_actions
        for action in turn_actions:
            events.append(
                make_event(
                    "action",
                    episode_id,
                    {"name": action.name, "arguments": action.arguments},
                )
            )
            if action.name not in ALLOWED_TOOLS:
                observation = Observation(
                    tool=action.name,
                    content={},
                    error=f"unknown tool {action.name}",
                    error_kind="invalid_action",
                )
                state.record_invalid_action()
                _append_observation(state, messages, events, token_trace, model, observation)
                break
            if action.name == TOOL_SUBMIT:
                if state.submit_calls >= task.budget.max_submit_calls:
                    observation = Observation(
                        tool=TOOL_SUBMIT,
                        content={},
                        error="submit budget exhausted",
                        error_kind="invalid_params",
                    )
                    state.record_explore(TOOL_SUBMIT, observation)
                    _append_observation(state, messages, events, token_trace, model, observation)
                    break
                observation = await tools.execute(action, state)
                observation = clip_observation(observation, task.budget.max_observation_chars)
                if observation.error:
                    state.record_invalid_action()
                    _append_observation(state, messages, events, token_trace, model, observation)
                    break
                state.record_submit(action)
                _append_observation(state, messages, events, token_trace, model, observation)
                break

            observation = await tools.execute(action, state)
            observation = clip_observation(observation, task.budget.max_observation_chars)
            if action.name == TOOL_SEARCH:
                state.record_search_query(str(action.arguments.get("query", "")))
            if action.name == TOOL_OPEN:
                paragraphs = observation.content.get("paragraphs") or []
                ids = [str(item.get("paragraph_id")) for item in paragraphs if isinstance(item, dict)]
                state.record_open_paragraphs([pid for pid in ids if pid and pid != "None"])
            state.record_explore(action.name, observation)
            _append_observation(state, messages, events, token_trace, model, observation)
            if state.terminated:
                break

    latency_ms = (time.perf_counter() - started) * 1000
    result = build_result(state, latency_ms)
    events.append(
        make_event(
            "episode_end",
            episode_id,
            {
                "termination": state.termination,
                "result": {
                    "answer": result.answer,
                    "claims": [
                        {"text": claim.text, "citation_ids": claim.citation_ids} for claim in result.claims
                    ],
                    "citations": result.citations,
                    "conditions": result.conditions,
                    "unresolved_questions": result.unresolved_questions,
                    "status": result.status,
                    "trajectory_id": result.trajectory_id,
                    "policy_version": result.policy_version,
                    "harness_version": result.harness_version,
                    "environment_version": result.environment_version,
                    "usage": {
                        "explore_calls": result.usage.explore_calls,
                        "search_calls": result.usage.search_calls,
                        "open_calls": result.usage.open_calls,
                        "submit_calls": result.usage.submit_calls,
                        "invalid_actions": result.usage.invalid_actions,
                        "generation_turns": result.usage.generation_turns,
                        "prompt_tokens": result.usage.prompt_tokens,
                        "completion_tokens": result.usage.completion_tokens,
                        "latency_ms": result.usage.latency_ms,
                        "context_truncated": result.usage.context_truncated,
                    },
                    "termination": result.termination,
                    "unread_citations": result.unread_citations,
                },
                "token_trace_usable_for_rl": token_trace.usable_for_rl,
            },
        )
    )
    return EpisodeRecord(
        episode_id=episode_id,
        task=task,
        events=events,
        result=result,
        token_trace=token_trace,
        live=live,
        messages=messages,
    )


def _append_observation(
    state: EpisodeState,
    messages: list[dict[str, str]],
    events: list,
    token_trace: TokenTrace,
    model: PolicyModel,
    observation: Observation,
) -> None:
    text = observation.as_text()
    token_trace.append_observation_tokens(model.encode_text(text))
    messages.append(observation_message(observation))
    events.append(
        make_event(
            "observation",
            state.episode_id,
            {
                "tool": observation.tool,
                "content": observation.content,
                "truncated": observation.truncated,
                "error": observation.error,
                "error_kind": observation.error_kind,
                "infra_failure": observation.infra_failure,
            },
        )
    )


async def run_batch(
    tasks: list[TaskInput],
    model: PolicyModel,
    tools: ToolDispatcher,
    *,
    max_concurrency: int = 4,
    system_prompt: str | None = None,
    live: bool = True,
) -> list[EpisodeRecord]:
    semaphore = asyncio.Semaphore(max_concurrency)

    async def one(task: TaskInput) -> EpisodeRecord:
        async with semaphore:
            return await run_episode(
                task,
                model,
                tools,
                system_prompt=system_prompt,
                live=live,
            )

    return list(await asyncio.gather(*[one(task) for task in tasks]))
