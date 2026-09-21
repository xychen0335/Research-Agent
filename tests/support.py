"""Shared unittest helpers. These are not pytest fixtures."""

from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from research_agent.contracts import Budget, TaskInput
from research_agent.data.prepare import prepare_synthetic_dev
from research_agent.data.sources.synthetic_dev import DOCUMENTS, TASKS, task_by_id
from research_agent.environment.corpus import CorpusSnapshot
from research_agent.environment.tools import ToolEnvironment
from research_agent.evaluation.runner import load_grading, load_tasks

assert len(TASKS) == 32


def make_corpus() -> CorpusSnapshot:
    return CorpusSnapshot.from_records(DOCUMENTS)


def make_tools(corpus: CorpusSnapshot | None = None) -> ToolEnvironment:
    return ToolEnvironment(corpus or make_corpus())


def make_prepared(tmp_path: Path):
    return prepare_synthetic_dev(tmp_path / "synthetic-dev")


def make_prepared_stack(prepared):
    tasks = load_tasks(prepared.output_dir / "public" / "tasks.jsonl", Budget())
    specs = load_grading(prepared.output_dir / "private" / "grading.jsonl")
    corpus = CorpusSnapshot.from_jsonl(prepared.output_dir / "public" / "corpus.jsonl")
    return tasks, specs, ToolEnvironment(corpus)


def task_input(task_id: str, budget: Budget | None = None) -> TaskInput:
    item = task_by_id()[task_id]
    return TaskInput(
        request_id=task_id,
        task_id=task_id,
        question=item.question,
        environment_id="synthetic-dev",
        budget=budget or Budget(),
        split="dev",
    )


class TempDirMixin:
    def setUp(self) -> None:
        super().setUp()
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()
        super().tearDown()


class HarnessMixin(TempDirMixin):
    def setUp(self) -> None:
        super().setUp()
        self.corpus = make_corpus()
        self.tools = make_tools(self.corpus)
        self._prepared = None
        self._prepared_stack = None

    @property
    def prepared(self):
        if self._prepared is None:
            self._prepared = make_prepared(self.tmp_path)
        return self._prepared

    @property
    def prepared_stack(self):
        if self._prepared_stack is None:
            self._prepared_stack = make_prepared_stack(self.prepared)
        return self._prepared_stack


def restore_asyncio_loop() -> None:
    """IsolatedAsyncioTestCase closes its loop; FastAPI TestClient needs a live one."""
    policy = asyncio.get_event_loop_policy()
    try:
        loop = policy.get_event_loop()
    except RuntimeError:
        loop = None
    if loop is None or loop.is_closed():
        policy.set_event_loop(asyncio.new_event_loop())


class TempDirTestCase(TempDirMixin, unittest.TestCase):
    def setUp(self) -> None:
        restore_asyncio_loop()
        super().setUp()


class HarnessTestCase(HarnessMixin, unittest.TestCase):
    pass


class HarnessAsyncTestCase(HarnessMixin, unittest.IsolatedAsyncioTestCase):
    pass
