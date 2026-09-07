"""Refuse to write train/eval/compress products into the Python venv."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def venv_roots() -> list[Path]:
    roots: list[Path] = []
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        roots.append(Path(venv))
    evie = os.environ.get("EVIE_ROOT")
    if evie:
        for name in ("env", "venv", ".venv", "train-env"):
            roots.append(Path(evie) / name)
    prefix = Path(sys.prefix)
    if (prefix / "pyvenv.cfg").is_file():
        roots.append(prefix)
    out: list[Path] = []
    seen: set[Path] = set()
    for raw in roots:
        try:
            resolved = raw.resolve()
        except OSError:
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def forbid_venv_path(path: str | Path, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    for root in venv_roots():
        try:
            path.relative_to(root)
        except ValueError:
            continue
        raise ValueError(
            f"{label} must not sit inside the Python env ({root}): {path}. "
            "Checkpoints -> $RUNS_DIR ($EVIE_ROOT/runs); "
            "HF caches -> $EVIE_ROOT/.cache; never $EVIE_ROOT/env."
        )
    return path
