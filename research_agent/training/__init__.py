"""Training package. GRPO/SFT modules stay lazy so CPU imports do not pull the GPU stack."""

from __future__ import annotations

from typing import Any

__all__ = [
    "assistant_token_labels",
    "export_grpo_group",
    "export_sft_messages",
    "group_advantages",
    "sequence_grpo_loss",
]


def __getattr__(name: str) -> Any:
    if name in {"export_grpo_group", "export_sft_messages"}:
        from research_agent.training.export import export_grpo_group, export_sft_messages

        return export_grpo_group if name == "export_grpo_group" else export_sft_messages
    if name in {"group_advantages", "sequence_grpo_loss"}:
        from research_agent.training.grpo import group_advantages, sequence_grpo_loss

        return group_advantages if name == "group_advantages" else sequence_grpo_loss
    if name == "assistant_token_labels":
        from research_agent.training.sft import assistant_token_labels

        return assistant_token_labels
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
