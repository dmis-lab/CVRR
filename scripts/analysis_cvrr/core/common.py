"""Portable environment and provenance helpers for released analyses."""

from __future__ import annotations

import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def configure_huggingface(*, offline: bool) -> dict[str, str]:
    """Record an explicit offline policy without imposing a cluster cache path."""

    values = {"HF_HUB_OFFLINE": "1" if offline else "0"}
    os.environ.update(values)
    return values


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _git_value(*arguments: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def run_metadata(seed: int | None = None, **extra: Any) -> dict[str, Any]:
    """Return machine-readable provenance embedded in every analysis artifact."""

    commit = _git_value("rev-parse", "HEAD")
    status = _git_value("status", "--porcelain")
    metadata: dict[str, Any] = {
        "seed": seed,
        "git_commit": commit,
        "git_dirty": bool(status) if status is not None else None,
        "hostname": platform.node(),
        "python": sys.version.split()[0],
        "argv": sys.argv,
    }
    try:
        import torch
        import transformers

        metadata.update(
            torch=torch.__version__,
            transformers=transformers.__version__,
            cuda=torch.version.cuda,
            gpu=(
                torch.cuda.get_device_name(torch.cuda.current_device())
                if torch.cuda.is_available()
                else None
            ),
        )
    except ImportError:
        pass
    metadata.update(extra)
    return metadata


__all__ = ["configure_huggingface", "run_metadata", "set_seed"]
