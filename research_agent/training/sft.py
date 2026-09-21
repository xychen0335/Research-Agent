"""LoRA SFT over harness messages. Loss is computed only on assistant tokens."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import yaml

from research_agent.training.compat import cached_hf_model, probe, write_report

EncodeFn = Callable[[str], list[int]]
IGNORE_INDEX = -100


@dataclass
class SFTConfig:
    base: str = "Qwen/Qwen3.5-4B"
    teacher_jsonl: str = "outputs/teacher/sft.jsonl"
    output_dir: str = "outputs/sft"
    lora_rank: int = 16
    lora_alpha: int = 32
    learning_rate: float = 2.0e-4
    epochs: int = 1
    max_seq_len: int = 8192
    micro_batch_size: int = 1
    max_steps: int = 20
    allow_cpu: bool = False
    min_teacher_rows: int = 16


def load_sft_config(path: Path | None) -> SFTConfig:
    raw: dict[str, Any] = {}
    if path and path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if isinstance(loaded, dict):
            raw = loaded
    fields = {key: raw[key] for key in SFTConfig.__dataclass_fields__ if key in raw}
    return SFTConfig(**fields)


def assistant_token_labels(messages: Iterable[dict[str, str]], encode_fn: EncodeFn) -> tuple[list[int], list[int]]:
    """Concatenate role-tagged messages. Labels are -100 except assistant content."""
    input_ids: list[int] = []
    labels: list[int] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = str(message.get("content") or "")
        piece = encode_fn(f"{role}\n{content}")
        input_ids.extend(piece)
        if role == "assistant":
            labels.extend(piece)
        else:
            labels.extend([IGNORE_INDEX] * len(piece))
    if not any(item != IGNORE_INDEX for item in labels):
        raise ValueError("SFT row has no assistant tokens")
    return input_ids, labels


def _chat_ids(tokenizer: Any, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
    )
    if isinstance(encoded, dict):
        encoded = encoded["input_ids"]
    return list(encoded)


def assistant_chat_labels(
    messages: list[dict[str, str]],
    tokenizer: Any,
    *,
    max_seq_len: int,
) -> tuple[list[int], list[int]]:
    """Mask with the same chat template HuggingFacePolicy uses at inference."""
    full = _chat_ids(tokenizer, messages, add_generation_prompt=False)
    labels = [IGNORE_INDEX] * len(full)
    for index, message in enumerate(messages):
        if str(message.get("role") or "") != "assistant":
            continue
        before = _chat_ids(tokenizer, messages[:index], add_generation_prompt=False) if index else []
        until = _chat_ids(tokenizer, messages[: index + 1], add_generation_prompt=False)
        start, end = len(before), min(len(until), len(full))
        if start >= end:
            continue
        labels[start:end] = full[start:end]
    full = full[:max_seq_len]
    labels = labels[:max_seq_len]
    if not any(item != IGNORE_INDEX for item in labels):
        raise ValueError("SFT row has no assistant tokens under the chat template")
    return full, labels


def load_teacher_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("split") == "test":
                continue
            messages = row.get("messages") or []
            if not messages:
                continue
            rows.append(row)
    return rows


def _select_device(allow_cpu: bool) -> str:
    report = probe()
    if report.get("cuda_available"):
        return "cuda"
    try:
        import torch

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    if allow_cpu:
        return "cpu"
    return "unavailable"


def _torch_dtype(torch: Any, device: str):
    if device == "cuda" and getattr(torch.cuda, "is_bf16_supported", lambda: False)():
        return torch.bfloat16
    if device != "cpu":
        return torch.float16
    return torch.float32


def _lora_train(cfg: SFTConfig, rows: list[dict[str, Any]], device: str) -> dict[str, Any]:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from torch.optim import AdamW
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = _torch_dtype(torch, device)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=dtype,
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        ),
    )
    model.to(device)
    model.train()
    optimizer = AdamW((p for p in model.parameters() if p.requires_grad), lr=cfg.learning_rate)
    if hasattr(tokenizer, "apply_chat_template"):
        examples = [
            assistant_chat_labels(list(row["messages"]), tokenizer, max_seq_len=cfg.max_seq_len)
            for row in rows
        ]
        token_layout = "chat_template"
    else:
        def encode_fn(text: str) -> list[int]:
            return list(tokenizer.encode(text, add_special_tokens=False))

        examples = [assistant_token_labels(row["messages"], encode_fn) for row in rows]
        examples = [(ids[: cfg.max_seq_len], labs[: cfg.max_seq_len]) for ids, labs in examples]
        token_layout = "role_newline"
    steps = 0
    last_loss = None
    peak_mem = None
    for _epoch in range(cfg.epochs):
        for ids, labs in examples:
            if steps >= cfg.max_steps:
                break
            input_ids = torch.tensor([ids], device=device)
            labels = torch.tensor([labs], device=device)
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.item())
            steps += 1
            if device == "cuda":
                peak_mem = float(torch.cuda.max_memory_allocated() / (1024**2))
        if steps >= cfg.max_steps:
            break
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")
    tokenizer.save_pretrained(out / "adapter")
    return {
        "status": "ran",
        "steps": steps,
        "last_loss": last_loss,
        "n_rows": len(examples),
        "device": device,
        "peak_mem_mb": peak_mem,
        "adapter": str(out / "adapter"),
        "base": cfg.base,
        "loss": "assistant_tokens_only",
        "token_layout": token_layout,
        "dtype": str(dtype),
    }


def saved_adapter_manifest(path: Path) -> dict[str, Any]:
    """Check that a LoRA save can be reloaded. Missing files stay not_run, not a fake score."""
    dest = Path(path)
    names = {item.name for item in dest.iterdir()} if dest.exists() and dest.is_dir() else set()
    has_weights = any(name.startswith("adapter_model") for name in names)
    return {
        "path": str(dest),
        "exists": dest.exists(),
        "has_config": "adapter_config.json" in names,
        "has_weights": has_weights,
        "reloadable": "adapter_config.json" in names and has_weights,
        "files": sorted(names),
    }


def _persist(cfg: SFTConfig, result: dict[str, Any]) -> dict[str, Any]:
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sft_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def run_sft(cfg: SFTConfig) -> dict[str, Any]:
    report = write_report()
    rows = load_teacher_rows(Path(cfg.teacher_jsonl))
    result: dict[str, Any] = {
        "status": "not_run",
        "teacher_rows": len(rows),
        "teacher_jsonl": cfg.teacher_jsonl,
        "compat": report,
        "loss": "assistant_tokens_only",
        "base": cfg.base,
    }
    if len(rows) < cfg.min_teacher_rows:
        result["error"] = (
            f"only {len(rows)} teacher traces (need {cfg.min_teacher_rows}); "
            "4B self-sampling is not a cold start"
        )
        return _persist(cfg, result)
    device = _select_device(cfg.allow_cpu)
    if device == "unavailable":
        result["error"] = "no CUDA/MPS device; LoRA SFT was not started"
        return _persist(cfg, result)
    if not cached_hf_model(cfg.base):
        result["error"] = (
            f"{cfg.base} weights are not on disk; refusing Hub download from this machine. "
            f"disk_free_gb={report.get('disk_free_gb')}. Copy the HF snapshot onto the GPU node before SFT."
        )
        result["device"] = device
        return _persist(cfg, result)
    try:
        trained = _lora_train(cfg, rows, device)
        result.update(trained)
    except ImportError as exc:
        result["error"] = f"training stack missing: {exc}"
        result["status"] = "not_run"
    return _persist(cfg, result)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/training/sft.yaml")
    parser.add_argument("--teacher", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--allow-cpu", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_sft_config(Path(args.config))
    if args.teacher:
        cfg.teacher_jsonl = args.teacher
    if args.output:
        cfg.output_dir = args.output
    cfg.allow_cpu = bool(args.allow_cpu or cfg.allow_cpu)
    result = run_sft(cfg)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "sft_report.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "ran" else 2


if __name__ == "__main__":
    raise SystemExit(main())
