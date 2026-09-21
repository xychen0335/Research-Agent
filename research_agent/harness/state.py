"""Episode state, budgets, and termination bookkeeping."""

from __future__ import annotations

from dataclasses import dataclass, field

from research_agent.contracts import (
    TERMINATION_BUDGET,
    TERMINATION_CONTEXT,
    TERMINATION_INFRA,
    TERMINATION_INVALID,
    TERMINATION_SUBMITTED,
    TOOL_OPEN,
    TOOL_SEARCH,
    TOOL_SUBMIT,
    Action,
    Budget,
    Claim,
    Observation,
    TaskInput,
    Usage,
)


@dataclass
class EpisodeState:
    task: TaskInput
    episode_id: str
    policy_version: str
    harness_version: str
    environment_version: str
    explore_calls: int = 0
    search_calls: int = 0
    open_calls: int = 0
    submit_calls: int = 0
    invalid_actions: int = 0
    generation_turns: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    observation_chars: int = 0
    opened_paragraph_ids: set[str] = field(default_factory=set)
    search_queries: list[str] = field(default_factory=list)
    context_truncated: bool = False
    truncation_events: int = 0
    termination: str | None = None
    infra_failure: bool = False
    answer: str = ""
    claims: list[Claim] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    conditions: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    unread_citations: list[str] = field(default_factory=list)

    @property
    def budget(self) -> Budget:
        return self.task.budget

    @property
    def terminated(self) -> bool:
        return self.termination is not None

    def remaining_explore(self) -> int:
        return max(0, self.budget.max_explore_calls - self.explore_calls)

    def record_generation(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.generation_turns += 1
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def record_invalid_action(self) -> None:
        self.invalid_actions += 1
        self.explore_calls += 1
        if self.invalid_actions >= self.budget.max_invalid_actions:
            self.termination = TERMINATION_INVALID
        elif self.explore_calls >= self.budget.max_explore_calls:
            self.termination = TERMINATION_BUDGET

    def record_explore(self, tool: str, observation: Observation) -> None:
        self.explore_calls += 1
        self.observation_chars += len(observation.as_text())
        if tool == TOOL_SEARCH:
            self.search_calls += 1
        elif tool == TOOL_OPEN:
            self.open_calls += 1
        if observation.infra_failure:
            self.infra_failure = True
            self.termination = TERMINATION_INFRA
        elif self.explore_calls >= self.budget.max_explore_calls and not self.terminated:
            self.termination = TERMINATION_BUDGET

    def record_open_paragraphs(self, paragraph_ids: list[str]) -> None:
        self.opened_paragraph_ids.update(paragraph_ids)

    def record_search_query(self, query: str) -> None:
        self.search_queries.append(query)

    def record_submit(self, action: Action) -> None:
        args = action.arguments
        self.submit_calls += 1
        self.answer = str(args.get("answer", "")).strip()
        self.citations = [str(x) for x in args.get("citations", []) if str(x).strip()]
        self.conditions = [str(x) for x in args.get("conditions", []) if str(x).strip()]
        self.unresolved_questions = [
            str(x) for x in args.get("unresolved_questions", []) if str(x).strip()
        ]
        raw_claims = args.get("claims", [])
        claims: list[Claim] = []
        if isinstance(raw_claims, list):
            for item in raw_claims:
                if isinstance(item, dict):
                    claims.append(
                        Claim(
                            text=str(item.get("text", "")).strip(),
                            citation_ids=[str(x) for x in item.get("citation_ids", [])],
                        )
                    )
                elif isinstance(item, str) and item.strip():
                    claims.append(Claim(text=item.strip(), citation_ids=list(self.citations)))
        if not claims and self.answer:
            claims.append(Claim(text=self.answer, citation_ids=list(self.citations)))
        self.claims = claims
        self.unread_citations = [
            cid for cid in self.citations if cid not in self.opened_paragraph_ids
        ]
        self.termination = TERMINATION_SUBMITTED

    def mark_context_truncated(self) -> None:
        self.context_truncated = True
        self.truncation_events += 1

    def force_context_limit(self) -> None:
        if not self.terminated:
            self.termination = TERMINATION_CONTEXT

    def usage(self, latency_ms: float) -> Usage:
        return Usage(
            explore_calls=self.explore_calls,
            search_calls=self.search_calls,
            open_calls=self.open_calls,
            submit_calls=self.submit_calls,
            invalid_actions=self.invalid_actions,
            generation_turns=self.generation_turns,
            prompt_tokens=self.prompt_tokens,
            completion_tokens=self.completion_tokens,
            observation_chars=self.observation_chars,
            latency_ms=latency_ms,
            context_truncated=self.context_truncated,
        )
