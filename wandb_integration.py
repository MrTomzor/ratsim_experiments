"""Weights & Biases integration for ratsim experiments.

W&B is an *additional* sink, not a replacement: both training frameworks fan a
single metric stream out to a list of outputs, so this module appends one more
output and changes nothing about what gets recorded. TensorBoard keeps writing
alongside (it costs 12 KB - 5 MB per run, noise next to a 225 MB replay dir).

Run identity is the actual problem this module solves.
-----------------------------------------------------
The scheduler dispatches one *process per stage* (`start_stage=K end_stage=K+1`,
see scheduler.py:build_command), so a 10-stage run is 10 separate invocations of
train.py. A plain `wandb.init()` would produce 10 disconnected W&B runs per seed,
on top of the 7-wide seed fan-out. We keep them together with a stable run id
plus `resume="allow"`.

That id is persisted to `<results_dir>/wandb_id.txt` rather than derived by
hashing the run path, on purpose: a path hash would silently *resume* a stale
W&B run if the results dir were ever deleted and the experiment re-run, quietly
appending fresh data onto old curves. A file makes new-vs-resume explicit.

Resuming works because both frameworks already carry a monotone global step
across stages -- SB3 runs with `reset_num_timesteps=False` and restores
`num_timesteps` from the checkpoint zip; DreamerV3 re-targets `run.steps` to the
cumulative end and restores its counter from `ckpt/latest`. W&B requires a
non-decreasing step, and that precondition is already met.

Enabling
--------
Off by default. Turn on per-run with `wandb=1`, or globally by exporting
`RATSIM_WANDB=1` (put it in your shell rc locally and in rci_env.sh on the
cluster, and every run reports without touching any def file).

    python train.py def=smoke_test method=ppo wandb=1
    python train.py def=smoke_test method=ppo wandb=1 wandb_project=ratsim-dev
    RATSIM_WANDB=1 python train.py def=smoke_test method=ppo

Credentials come from `WANDB_API_KEY` or `~/.netrc` -- never from a tracked
file. `rci_env.sh` is in git, so the key does not belong there.
"""

from __future__ import annotations

import numbers
import os
import uuid
from pathlib import Path

DEFAULT_PROJECT = "ratsim"


# -- enable / identity --------------------------------------------------------

def wandb_requested(overrides: dict) -> bool:
    """Pop `wandb=` off the CLI overrides; fall back to $RATSIM_WANDB.

    Mutates `overrides`, matching how train.py consumes every other CLI key.
    """
    cli = overrides.pop("wandb", None)
    if cli is not None:
        return str(cli).strip().lower() in ("1", "true", "yes", "on")
    return os.environ.get("RATSIM_WANDB", "").strip().lower() in ("1", "true", "yes", "on")


def _machine_tag() -> str:
    """Coarse origin tag, so laptop and cluster runs are filterable apart.

    Deliberately not the hostname on SLURM: compute nodes vary per dispatch
    (a07, g01, ...) and would shatter the tag into dozens of useless values.
    """
    if os.environ.get("RATSIM_MACHINE"):
        return os.environ["RATSIM_MACHINE"]
    if os.environ.get("SLURM_JOB_ID"):
        return "rci"
    import socket
    return socket.gethostname().split(".")[0]


def _stable_run_id(results_dir: Path) -> str:
    """Read-or-create the W&B run id for this results dir.

    Stages run sequentially within a run, and each run has its own results dir,
    so there is no race here even at 7-wide.
    """
    id_file = results_dir / "wandb_id.txt"
    if id_file.exists():
        existing = id_file.read_text().strip()
        if existing:
            return existing
    new_id = uuid.uuid4().hex[:16]
    id_file.write_text(new_id + "\n")
    return new_id


def init_run(results_dir, run_meta: dict, *, project: str | None = None,
             mode: str | None = None, extra_tags=()):
    """Start (or resume) the W&B run for this results dir. None if unavailable.

    `run_meta` is train.py's existing run_config.json payload -- exp_id,
    variation, method, seed, method_config -- which is already exactly the
    right W&B config, so nothing new has to be assembled.
    """
    try:
        import wandb
    except ImportError:
        print("[wandb] requested but not installed; skipping. "
              "`pip install wandb` in the venv to enable.")
        return None

    results_dir = Path(results_dir)
    run_id = _stable_run_id(results_dir)

    # run_name is a path for scheduler runs
    # ("experiments/<exp>/runs/<variation>__<method>__seed<n>"); the leaf is the
    # readable identity and `group` already carries the experiment.
    raw_name = str(run_meta.get("run_name") or results_dir.name)
    name = Path(raw_name).name

    tags = [t for t in (run_meta.get("method"), run_meta.get("variation"),
                        _machine_tag(), *extra_tags) if t]

    try:
        run = wandb.init(
            project=project or os.environ.get("WANDB_PROJECT") or DEFAULT_PROJECT,
            id=run_id,
            resume="allow",          # same id across the per-stage processes
            name=name,
            group=run_meta.get("exp_id"),   # collapses the seed fan-out
            job_type=run_meta.get("method"),
            tags=tags,
            config=run_meta,
            dir=str(results_dir),    # keep W&B state next to the run, NOT in
                                     # $TMPDIR -- rci_env.sh makes that per-job
                                     # and it is wiped when the job ends
            mode=mode,
            # The scheduler tees child stdout through a prefixing thread
            # (scheduler.py:spawn_job); wandb wrapping stdout fights with it.
            settings=wandb.Settings(console="off"),
        )
    except Exception as e:
        print(f"[wandb] init failed ({type(e).__name__}: {e}); continuing without it.")
        return None

    resumed = getattr(run, "resumed", False)
    print(f"[wandb] {'resumed' if resumed else 'started'} run '{name}' "
          f"id={run_id} group={run_meta.get('exp_id')} mode={run.settings.mode}")
    if run.url:
        print(f"[wandb] {run.url}")
    return run


def finish_run(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as e:
        print(f"[wandb] finish failed ({type(e).__name__}: {e}); ignoring.")


# -- SB3 hookup ---------------------------------------------------------------

def make_sb3_callback(run):
    """Callback mirroring SB3's logger stream into W&B. None if `run` is None."""
    if run is None:
        return None
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import KVWriter

    class _WandbOutputFormat(KVWriter):
        """One more output on SB3's existing logger fan-out.

        Every `self.logger.record(...)` site in train.py -- custom/*, timing/*,
        and SB3's own rollout/* and train/* -- arrives here untouched.
        """

        def __init__(self, wrun):
            self._run = wrun

        def write(self, key_values, key_excluded, step=0):
            payload = {}
            for key, value in key_values.items():
                if "wandb" in (key_excluded.get(key) or ()):
                    continue
                if isinstance(value, bool):
                    payload[key] = value
                elif isinstance(value, numbers.Number):
                    payload[key] = float(value)
                elif isinstance(value, str):
                    payload[key] = value
                # Video/Figure/Image/HParam are skipped: nothing in this repo
                # records them, and guessing at a mapping would be dead code.
            if payload:
                self._run.log(payload, step=int(step))

        def close(self):
            pass

    class WandbSb3Callback(BaseCallback):
        """Attaches the output format once training's real logger exists.

        Not attached at construction on purpose: SB3 builds the logger inside
        `learn()` (`_setup_learn` -> `configure_logger`), so anything attached
        beforehand is thrown away. The idempotence guard matters for the
        single-process multi-stage path, where `learn()` is called per stage
        but the logger is reused (`reset_num_timesteps=False`).
        """

        def __init__(self, wrun, verbose=0):
            super().__init__(verbose)
            self._run = wrun

        def _on_training_start(self) -> None:
            formats = self.model.logger.output_formats
            if not any(isinstance(f, _WandbOutputFormat) for f in formats):
                formats.append(_WandbOutputFormat(self._run))

        def _on_step(self) -> bool:
            return True

    return WandbSb3Callback(run)


# -- DreamerV3 / embodied hookup ----------------------------------------------

def make_elements_output(run, pattern: str = r".*"):
    """An `elements.logger` output writing to an already-initialised W&B run.

    Deliberately does NOT subclass `elements.logger.WandBOutput` and inherit its
    `__call__`, for two reasons:

    1. Its constructor calls `wandb.init()` itself, which would duplicate (and
       drift from) the run identity in `init_run`.
    2. Its image branch is broken upstream. It reads `value.shape[3]` on a 2/3-D
       array -- copy-pasted from the 4-D video branch below it -- so indices run
       out and *any* image summary raises IndexError. Verified against
       elements 3.x: both the 2-D and 3-D paths raise. Inheriting it would turn
       the first Dreamer report containing an image into a crashed training run.

    Shaping below is the same intent, with the index fixed and images left in
    the H×W×C layout `wandb.Image` documents.
    """
    if run is None:
        return None
    import re
    import numpy as np
    import wandb

    compiled = re.compile(pattern)

    class _RatsimWandBOutput:
        """Maps embodied's (step, name, value) triples onto W&B types."""

        def __call__(self, summaries):
            bystep = {}
            for step, name, value in summaries:
                if not compiled.search(name):
                    continue
                bucket = bystep.setdefault(step, {})
                if isinstance(value, str):
                    bucket[name] = value
                    continue
                value = np.asarray(value)
                ndim = len(value.shape)
                if ndim == 0:
                    bucket[name] = float(value)
                elif ndim == 1:
                    bucket[name] = wandb.Histogram(value)
                elif ndim in (2, 3):
                    img = value[..., None] if ndim == 2 else value
                    if img.shape[2] not in (1, 3, 4):
                        continue          # not an image; don't guess
                    if img.dtype != np.uint8:
                        img = (255 * np.clip(img, 0, 1)).astype(np.uint8)
                    bucket[name] = wandb.Image(img)   # H x W x C
                elif ndim == 4:
                    vid = value
                    if vid.shape[3] not in (1, 3, 4):
                        continue
                    if vid.dtype != np.uint8:
                        vid = (255 * np.clip(vid, 0, 1)).astype(np.uint8)
                    bucket[name] = wandb.Video(np.transpose(vid, [0, 3, 1, 2]))
            for step, metrics in bystep.items():
                run.log(metrics, step=int(step))

    return _RatsimWandBOutput()
