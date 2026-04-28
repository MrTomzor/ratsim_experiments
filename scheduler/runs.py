"""Pure helpers for inspecting per-run stage progress.

Source of truth for "stage K is done" is the `.done` marker file written by
train.py / train_dreamerv3.py after a stage's checkpoint is fully saved.
The scheduler stores no progress state of its own — it scans these markers.
"""
from __future__ import annotations

from pathlib import Path


def stage_done(run_dir: Path, stage_idx: int) -> bool:
    return (run_dir / "checkpoints" / f"stage_{stage_idx}.done").exists()


def count_done_stages(run_dir: Path, n_stages: int) -> int:
    """Largest K such that stages 0..K-1 are all done. Stops at the first gap."""
    k = 0
    while k < n_stages and stage_done(run_dir, k):
        k += 1
    return k


def run_done(run_dir: Path, n_stages: int) -> bool:
    return count_done_stages(run_dir, n_stages) >= n_stages
