"""Workspace layout. Raw dumps and model snapshots are placed ahead of time."""

from __future__ import annotations

from pathlib import Path

DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
MODELS_DIR = Path("models")
DEFAULT_MODEL = MODELS_DIR / "Qwen3.5-4B"
HF_HUB_ID = "Qwen/Qwen3.5-4B"
PAPERSEARCHQA_RAW = DATA_RAW / "papersearchqa"
PUBMED_RAW = DATA_RAW / "pubmed_bioasq_2022"
PAPERSEARCHQA_PROCESSED = DATA_PROCESSED / "papersearchqa"


def resolve_model_path(name: str | None = None) -> Path:
    """Return the local snapshot directory. Does not download from the Hub."""
    if name:
        given = Path(name)
        if given.exists():
            return given
        local_name = MODELS_DIR / given.name
        if local_name.exists():
            return local_name
        if name in {HF_HUB_ID, "Qwen3.5-4B"} and DEFAULT_MODEL.exists():
            return DEFAULT_MODEL
        return given
    return DEFAULT_MODEL


def model_available(name: str | None = None) -> bool:
    path = resolve_model_path(name)
    if not path.exists():
        return False
    if path.is_file():
        return True
    return (path / "config.json").exists() or any(path.glob("*.safetensors")) or any(path.glob("*.bin"))
