"""No-retrieval, fixed-retrieval RAG, and agent baselines sharing one environment."""

from __future__ import annotations

from enum import Enum

from research_agent.contracts import TOOL_OPEN, TOOL_SEARCH, TOOL_SUBMIT, TaskInput
from research_agent.data.sources.synthetic_dev import task_by_id
from research_agent.environment.tools import ToolEnvironment
from research_agent.grading.contracts import GradingSpec
from research_agent.models.scripted import ScriptedPolicy, tool_call
from research_agent.evaluation.runner import EvalResult, run_evaluation


class BaselineName(str, Enum):
    NO_RETRIEVAL = "no_retrieval"
    FIXED_RAG = "fixed_rag"
    AGENT = "agent"


def _no_retrieval_scripts(tasks: list[TaskInput]) -> dict[str, list[str]]:
    return {task.question: [tool_call(TOOL_SUBMIT, {"answer": "unknown", "citations": []})] for task in tasks}


def _rag_scripts(tasks: list[TaskInput], tools: ToolEnvironment) -> dict[str, list[str]]:
    scripts: dict[str, list[str]] = {}
    for task in tasks:
        hits = tools.index.search(task.question, topk=1)
        if not hits:
            scripts[task.question] = [tool_call(TOOL_SUBMIT, {"answer": "unknown", "citations": []})]
            continue
        doc_id = hits[0].doc_id
        scripts[task.question] = [
            tool_call(TOOL_SEARCH, {"query": task.question}),
            tool_call(TOOL_OPEN, {"doc_id": doc_id, "start": 0, "end": 0}),
            tool_call(TOOL_SUBMIT, {"answer": hits[0].title, "citations": [f"{doc_id}:0"]}),
        ]
    return scripts


def _oracle_scripts(tasks: list[TaskInput], specs: dict[str, GradingSpec]) -> dict[str, list[str]]:
    """Oracle uses grading search hints. Never used as a reported trained policy."""
    catalog = task_by_id()
    scripts: dict[str, list[str]] = {}
    for task in tasks:
        spec = specs[task.public_id()]
        hint = catalog[task.public_id()].search_hint if task.public_id() in catalog else task.question
        citations = list(spec.gold_evidence_ids)
        opens = [
            tool_call(TOOL_OPEN, {"doc_id": doc_id, "start": 0, "end": 4})
            for doc_id in spec.support_doc_ids
        ]
        scripts[task.question] = [
            tool_call(TOOL_SEARCH, {"query": hint}),
            *opens,
            tool_call(TOOL_SUBMIT, {"answer": spec.answer, "citations": citations}),
        ]
    return scripts


async def run_baseline(
    name: str,
    tasks: list[TaskInput],
    specs: dict[str, GradingSpec],
    tools: ToolEnvironment,
    *,
    run_id: str,
    output_dir=None,
    oracle: bool = False,
) -> EvalResult:
    if name == BaselineName.NO_RETRIEVAL.value:
        model = ScriptedPolicy(_no_retrieval_scripts(tasks), policy_version="scripted-no-retrieval")
    elif name == BaselineName.FIXED_RAG.value:
        model = ScriptedPolicy(_rag_scripts(tasks, tools), policy_version="scripted-fixed-rag")
    elif name == BaselineName.AGENT.value and oracle:
        model = ScriptedPolicy(_oracle_scripts(tasks, specs), policy_version="scripted-oracle")
    elif name == BaselineName.AGENT.value:
        model = ScriptedPolicy(_rag_scripts(tasks, tools), policy_version="scripted-heuristic-agent")
    else:
        raise ValueError(f"unknown baseline {name}")
    return await run_evaluation(
        tasks,
        specs,
        model,
        tools,
        run_id=run_id,
        output_dir=output_dir,
        live=True,
    )
