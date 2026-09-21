"""Dataset source adapters. Official dumps are converted; they are not downloaded here."""

from research_agent.data.sources.papersearchqa import convert_papersearchqa_rows
from research_agent.data.sources.qasper import convert_qasper_rows
from research_agent.data.sources.synthetic_dev import DOCUMENTS, TASKS

__all__ = [
    "DOCUMENTS",
    "TASKS",
    "convert_papersearchqa_rows",
    "convert_qasper_rows",
]
