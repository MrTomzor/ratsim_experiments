"""Scheduler entry point.

Run / resume an experiment:
    python -m scheduler.scheduler run gps_ablation
    python -m scheduler.scheduler run defs/gps_ablation.yaml
    python -m scheduler.scheduler run gps_ablation --machine gpu_example
    python -m scheduler.scheduler run gps_ablation --step-multiplier 0.01

Read-only status:
    python -m scheduler.scheduler status gps_ablation

Folder layout:
    ratsim_experiments/results/experiments/<exp_id>/
        experiment.yaml                        — snapshot of the def at first dispatch
        state.json                             — pids of running children, failure log
        DONE                                   — touched when every run finishes
        runs/
            <variation>__<method>__seed<i>/    — same layout as ad-hoc runs
                checkpoints/, train_episodes.jsonl, tensorboard/, run_config.json
                scheduler_logs/stage_<i>_<ts>.log

The scheduler stores no progress state of its own. Stage completion is read
from `checkpoints/stage_<i>.done` markers written by the train scripts.
state.json only tracks transient things (running pids, failure log) used to
clean up on restart.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml

from . import config as cfg
from . import runs as runs_mod
from .ports import PortAllocator


SCHEDULER_DIR = Path(__file__).parent
EXPERIMENTS_DIR = SCHEDULER_DIR.parent
RESULTS_DIR = EXPERIMENTS_DIR / "results"
EXP_RESULTS_DIR = RESULTS_DIR / "experiments"
DEFS_DIR = EXPERIMENTS_DIR / "defs"
MACHINES_DIR = SCHEDULER_DIR / "machines"

POLL_INTERVAL = 2.0  # seconds between dispatch loop iterations

# Max consecutive failures of the same (run, stage) before the scheduler
# stops retrying and blocks it. Tracked in-memory only — restarting the
# scheduler resets the counter, which is usually what you want after fixing
# a typo / preset / venv issue.
MAX_CONSECUTIVE_FAILURES = 2

# When a stage fails, this many lines from the tail of its log are echoed to
# the scheduler's stdout so the user sees the error immediately without
# opening the log file. The full log stays at scheduler_logs/.
ERROR_TAIL_LINES = 25


# ---------------------------------------------------------------------------
# In-memory job tracking
# ---------------------------------------------------------------------------

@dataclass
class Run:
    run_id: str            # "<variation>__<method>__seed<i>"
    variation: cfg.VariationSpec
    method: cfg.MethodSpec
    seed: int
    run_dir: Path


@dataclass
class ActiveJob:
    run: Run
    stage_idx: int
    popen: subprocess.Popen
    port_base: int
    profile: cfg.MethodProfile
    started_at: datetime
    log_path: Path


class ResourceManager:
    """Tracks how much of each declared resource is currently reserved."""

    def __init__(self, capacities: dict[str, int]):
        self.capacities = dict(capacities)
        self.in_use = {k: 0 for k in capacities}

    def can_allocate(self, needs: dict[str, int]) -> bool:
        for k, v in needs.items():
            if k not in self.capacities:
                return False
            if self.in_use[k] + v > self.capacities[k]:
                return False
        return True

    def allocate(self, needs: dict[str, int]) -> None:
        for k, v in needs.items():
            self.in_use[k] += v

    def release(self, needs: dict[str, int]) -> None:
        for k, v in needs.items():
            self.in_use[k] = max(0, self.in_use[k] - v)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def resolve_exp_dir(arg: str) -> tuple[Path, Path]:
    """Resolve `<exp_id_or_path>` → (def_path, exp_results_dir).

    Accepted forms:
      * bare id like `gps_ablation` → defs/gps_ablation.yaml +
        results/experiments/gps_ablation/
      * relative or absolute path like `defs/foo.yaml` → that file +
        results/experiments/<stem>/
    """
    def_path = cfg.resolve_def_path(DEFS_DIR, arg)
    if not def_path.exists():
        raise FileNotFoundError(
            f"experiment def not found at {def_path}. "
            f"Available in {DEFS_DIR}: "
            f"{sorted(p.stem for p in DEFS_DIR.glob('*.yaml'))}")
    exp_results_dir = EXP_RESULTS_DIR / def_path.stem
    return def_path, exp_results_dir


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        with open(state_path) as f:
            return json.load(f)
    return {"running": [], "failed": [], "started_at": None, "last_event_at": None}


def save_state(state_path: Path, state: dict) -> None:
    state["last_event_at"] = datetime.now().isoformat(timespec="seconds")
    tmp = state_path.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(state, f, indent=2)
    tmp.replace(state_path)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return Path(f"/proc/{pid}").exists()
    return True


def _format_age(iso_ts: str | None) -> str:
    """Human-readable 'X ago' for an ISO timestamp string. Empty if unparseable."""
    if not iso_ts:
        return ""
    try:
        ts = datetime.fromisoformat(iso_ts)
    except ValueError:
        return ""
    secs = int((datetime.now() - ts).total_seconds())
    if secs < 0:
        return ""
    if secs < 60:
        return f"{secs}s ago"
    if secs < 3600:
        return f"{secs // 60}m {secs % 60:02d}s ago"
    if secs < 86400:
        h = secs // 3600
        return f"{h}h {(secs % 3600) // 60:02d}m ago"
    d = secs // 86400
    return f"{d}d {(secs % 86400) // 3600:02d}h ago"


def tail_log(log_path: Path, n_lines: int) -> str:
    """Return the last n_lines of log_path, or an empty string if unreadable."""
    try:
        with open(log_path, "rb") as f:
            try:
                f.seek(0, 2)
                size = f.tell()
                # Read backwards in chunks until we have enough newlines or hit start.
                chunk = 4096
                data = b""
                while size > 0 and data.count(b"\n") <= n_lines:
                    step = min(chunk, size)
                    size -= step
                    f.seek(size)
                    data = f.read(step) + data
            except OSError:
                f.seek(0)
                data = f.read()
        text = data.decode("utf-8", errors="replace")
        lines = text.splitlines()
        return "\n".join(lines[-n_lines:])
    except OSError:
        return ""


def kill_orphans(state: dict) -> None:
    """SIGTERM any pid recorded in state['running'] from a previous scheduler
    invocation. They might be alive (scheduler crashed but child kept going)
    or already dead. Either way we clear the list afterwards."""
    for entry in state.get("running", []):
        pid = entry.get("pid")
        if pid is None or not pid_alive(pid):
            continue
        try:
            os.killpg(pid, signal.SIGTERM)
            print(f"[scheduler] killed orphan pid {pid} "
                  f"({entry.get('run_id')} stage {entry.get('stage_idx')})")
        except (ProcessLookupError, PermissionError) as e:
            print(f"[scheduler] could not kill orphan pid {pid}: {e}")


def expand_runs(exp: cfg.ExperimentDef, exp_results_dir: Path) -> list[Run]:
    """One Run per (variation, method, seed) triple."""
    out = []
    for variation in exp.variations:
        for method in exp.methods:
            for seed in range(method.n_seeds):
                run_id = f"{variation.name}__{method.name}__seed{seed}"
                out.append(Run(
                    run_id=run_id,
                    variation=variation,
                    method=method,
                    seed=seed,
                    run_dir=exp_results_dir / "runs" / run_id,
                ))
    return out


# ---------------------------------------------------------------------------
# Dispatch ordering
# ---------------------------------------------------------------------------

def pick_candidates(runs: list[Run], n_stages: int, mode: str,
                    in_flight: set[tuple[str, int]],
                    blocked: set[tuple[str, int]]):
    """Yield (run, stage_idx) in dispatch-priority order. The caller picks
    candidates whose resources fit and skips the rest until the next iteration.

    Constraint shared by both modes: stage K of a run is only eligible once
    stages 0..K-1 are all `.done`. So a single run progresses sequentially,
    but different runs progress independently (two PPO seeds can both be on
    stage 1 simultaneously). `blocked` is the set of (run_id, stage_idx) that
    have hit MAX_CONSECUTIVE_FAILURES and won't be retried this invocation."""
    if mode == "dfs":
        # One run at a time, finish all stages before moving to next run.
        for run in runs:
            done = runs_mod.count_done_stages(run.run_dir, n_stages)
            if done >= n_stages:
                continue
            if any(rid == run.run_id for rid, _ in in_flight):
                continue
            if (run.run_id, done) in blocked:
                continue
            yield (run, done)
    elif mode == "bfs":
        # Outer loop on stage_idx → all runs make progress on early stages
        # before later stages, which gives "early signal across methods/
        # variations" without finishing any single run first.
        for stage_idx in range(n_stages):
            for run in runs:
                if runs_mod.stage_done(run.run_dir, stage_idx):
                    continue
                if (run.run_id, stage_idx) in in_flight:
                    continue
                if (run.run_id, stage_idx) in blocked:
                    continue
                done = runs_mod.count_done_stages(run.run_dir, n_stages)
                if done < stage_idx:
                    continue  # earlier stage of this run still pending
                yield (run, stage_idx)
    else:
        raise ValueError(f"unknown mode: {mode!r}")


# ---------------------------------------------------------------------------
# Spawn
# ---------------------------------------------------------------------------

def build_command(run: Run, stage_idx: int, profile: cfg.MethodProfile,
                  port_base: int, exp: cfg.ExperimentDef,
                  step_multiplier: float) -> list[str]:
    python = cfg.resolve_python(run.method, profile)
    train_script = cfg.resolve_train_script(run.method, profile)

    # run_folder is given relative to ratsim_experiments/results/.
    rel_run = run.run_dir.relative_to(RESULTS_DIR)

    cmd = [
        python, "-u", train_script,
        f"def={exp.source}",
        f"variation={run.variation.name}",
        f"run_folder={rel_run}",
        f"start_stage={stage_idx}",
        f"end_stage={stage_idx + 1}",
        f"step_multiplier={step_multiplier}",
        f"metaseed={run.seed}",
        f"base_port={port_base}",
    ]
    if train_script in cfg.SCRIPTS_NEEDING_METHOD_ARG:
        cmd.append(f"method={run.method.name}")

    # Merge: profile defaults < methodspec.args (def file) < common_args (def file).
    # variation.method_args is read directly by train.py from the def, not
    # passed here. Reserved keys ignored (scheduler controls them).
    merged: dict = {}
    merged.update(profile.args)
    merged.update(run.method.args)
    merged.update(exp.common_args)
    for k, v in merged.items():
        if k in cfg.RESERVED_ARGS:
            print(f"[scheduler] WARNING: ignoring reserved arg {k}={v} "
                  f"in profile/method/common_args")
            continue
        cmd.append(f"{k}={v}")
    return cmd


def spawn_job(run: Run, stage_idx: int, profile: cfg.MethodProfile,
              port_base: int, exp: cfg.ExperimentDef,
              step_multiplier: float) -> ActiveJob:
    cmd = build_command(run, stage_idx, profile, port_base, exp, step_multiplier)
    log_dir = run.run_dir / "scheduler_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"stage_{stage_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    print(f"[scheduler] dispatch {run.run_id} stage {stage_idx} port={port_base}")
    print(f"[scheduler]   log: {log_path}")
    print(f"[scheduler]   cmd: {' '.join(cmd)}")

    log_f = open(log_path, "w")
    popen = subprocess.Popen(
        cmd,
        cwd=EXPERIMENTS_DIR,
        stdout=log_f,
        stderr=subprocess.STDOUT,
        # Own process group so SIGINT to scheduler doesn't auto-propagate.
        # We forward it manually on graceful shutdown.
        preexec_fn=os.setsid,
    )
    return ActiveJob(
        run=run, stage_idx=stage_idx, popen=popen, port_base=port_base,
        profile=profile, started_at=datetime.now(), log_path=log_path,
    )


# ---------------------------------------------------------------------------
# `run` command — main dispatch loop
# ---------------------------------------------------------------------------

def cmd_run(args):
    def_path, exp_dir = resolve_exp_dir(args.exp)
    exp = cfg.load_experiment_def(def_path)
    machine = cfg.load_machine_config(MACHINES_DIR, override=args.machine)
    cfg.validate_against_machine(exp, machine)

    # CLI step-multiplier overrides the def-level one (with a warning if both set).
    step_multiplier = exp.step_multiplier
    if args.step_multiplier is not None:
        if step_multiplier != 1.0 and step_multiplier != args.step_multiplier:
            print(f"[scheduler] WARNING: --step-multiplier {args.step_multiplier} "
                  f"overrides def's step_multiplier={step_multiplier}")
        step_multiplier = args.step_multiplier

    n_stages = len(exp.stages)

    # --restart wipes the existing exp dir before starting. Useful when you
    # want a clean run instead of resume — equivalent to manually
    # `rm -rf results/experiments/<exp_id>/`.
    if getattr(args, "restart", False) and exp_dir.exists():
        print(f"[scheduler] --restart: wiping {exp_dir}")
        shutil.rmtree(exp_dir)

    exp_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the resolved def into the exp dir on first run (or if missing).
    snap_path = exp_dir / "experiment.yaml"
    if not snap_path.exists():
        with open(snap_path, "w") as f:
            yaml.safe_dump(cfg.snapshot_experiment(exp), f, sort_keys=False)
        print(f"[scheduler] snapshotted def → {snap_path}")

    state_path = exp_dir / "state.json"
    state = load_state(state_path)
    if state["started_at"] is None:
        state["started_at"] = datetime.now().isoformat(timespec="seconds")

    print(f"[scheduler] exp={exp.exp_id} mode={exp.mode} "
          f"n_stages={n_stages} step_multiplier={step_multiplier}")
    print(f"[scheduler] machine_config={machine.source} "
          f"resources={machine.resources}")

    kill_orphans(state)
    state["running"] = []
    save_state(state_path, state)

    runs = expand_runs(exp, exp_dir)
    for run in runs:
        run.run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[scheduler] {len(runs)} runs total "
          f"({len(exp.variations)} variations × {len(exp.methods)} methods × "
          f"{sum(m.n_seeds for m in exp.methods)} method-seeds)")

    rm = ResourceManager(machine.resources)
    port_alloc = PortAllocator(start=9100, window_size=10)
    active: list[ActiveJob] = []
    shutdown = {"requested": False}

    # In-memory failure counters and the resulting blocked set. Reset on
    # restart — typical workflow is "scheduler crashed → user fixed the typo
    # → re-run scheduler", so giving each (run, stage) a fresh retry budget
    # is what you want.
    consecutive_failures: dict[tuple[str, int], int] = {}
    blocked: set[tuple[str, int]] = set()

    def handle_signal(signum, frame):
        if shutdown["requested"]:
            print("\n[scheduler] second signal — force-killing children")
            for job in active:
                try:
                    os.killpg(job.popen.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            sys.exit(130)
        print(f"\n[scheduler] caught signal {signum} — terminating children, "
              f"will exit when they're done (Ctrl-C again to force)")
        for job in active:
            try:
                os.killpg(job.popen.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        shutdown["requested"] = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    while True:
        # ---- Reap finished jobs ----
        for job in list(active):
            rc = job.popen.poll()
            if rc is None:
                continue
            rm.release(job.profile.needs)
            port_alloc.release(job.port_base)
            active.remove(job)
            duration = (datetime.now() - job.started_at).total_seconds()
            outcome = "done" if rc == 0 else f"FAILED (exit {rc})"
            stage_done = runs_mod.stage_done(job.run.run_dir, job.stage_idx)
            print(f"[scheduler] {job.run.run_id} stage {job.stage_idx} "
                  f"{outcome} in {duration:.0f}s (stage_done={stage_done})")
            state["running"] = [
                e for e in state["running"]
                if not (e["run_id"] == job.run.run_id
                        and e["stage_idx"] == job.stage_idx)
            ]
            if rc != 0:
                state["failed"].append({
                    "run_id": job.run.run_id,
                    "stage_idx": job.stage_idx,
                    "exit_code": rc,
                    "log_path": str(job.log_path),
                    "at": datetime.now().isoformat(timespec="seconds"),
                })
                # Echo the tail of the child's log to the scheduler's stdout
                # so the user sees the error without opening the log file.
                tail = tail_log(job.log_path, ERROR_TAIL_LINES)
                if tail:
                    print(f"[scheduler]   --- last {ERROR_TAIL_LINES} log "
                          f"lines from {job.log_path.name} ---")
                    for ln in tail.splitlines():
                        print(f"    {ln}")
                    print(f"[scheduler]   --- end log tail ---")
                key = (job.run.run_id, job.stage_idx)
                consecutive_failures[key] = consecutive_failures.get(key, 0) + 1
                if consecutive_failures[key] >= MAX_CONSECUTIVE_FAILURES:
                    blocked.add(key)
                    print(f"[scheduler] BLOCKING {job.run.run_id} stage "
                          f"{job.stage_idx}: {consecutive_failures[key]} "
                          f"consecutive failures. Fix the issue (see tail "
                          f"above; full log at {job.log_path}) then re-run "
                          f"the scheduler to retry.")
            save_state(state_path, state)

        # ---- Drain mode: shutdown requested → don't spawn anything new ----
        if shutdown["requested"]:
            if not active:
                break
            time.sleep(POLL_INTERVAL)
            continue

        # ---- Try to dispatch as many as resources allow ----
        in_flight = {(j.run.run_id, j.stage_idx) for j in active}
        any_candidates = False
        for run, stage_idx in pick_candidates(runs, n_stages, exp.mode,
                                              in_flight, blocked):
            any_candidates = True
            profile = machine.method_profiles[run.method.name]
            if not rm.can_allocate(profile.needs):
                continue
            port_base = port_alloc.alloc()
            job = spawn_job(run, stage_idx, profile, port_base, exp, step_multiplier)
            rm.allocate(profile.needs)
            active.append(job)
            in_flight.add((run.run_id, stage_idx))
            state["running"].append({
                "run_id": run.run_id,
                "stage_idx": stage_idx,
                "pid": job.popen.pid,
                "port_base": port_base,
                "started_at": job.started_at.isoformat(timespec="seconds"),
                "log_path": str(job.log_path.relative_to(exp_dir)),
            })
            save_state(state_path, state)

        # ---- Termination check ----
        if not active and not any_candidates:
            n_done = sum(1 for r in runs if runs_mod.run_done(r.run_dir, n_stages))
            if blocked:
                print(f"[scheduler] stopping: {len(blocked)} (run, stage) pairs "
                      f"blocked after {MAX_CONSECUTIVE_FAILURES} consecutive "
                      f"failures, no other candidates available.")
                print(f"[scheduler] blocked: "
                      f"{sorted(blocked)}")
                print(f"[scheduler] inspect logs, fix the root cause, then "
                      f"re-run the scheduler — counters reset on restart.")
            else:
                print("[scheduler] all runs complete.")
                (exp_dir / "DONE").touch()
            break

        time.sleep(POLL_INTERVAL)

    n_done = sum(1 for r in runs if runs_mod.run_done(r.run_dir, n_stages))
    print(f"[scheduler] done runs: {n_done}/{len(runs)}")


# ---------------------------------------------------------------------------
# `status` command — read-only
# ---------------------------------------------------------------------------

def cmd_status(args):
    def_path, exp_dir = resolve_exp_dir(args.exp)
    exp = cfg.load_experiment_def(def_path)
    n_stages = len(exp.stages)
    state = load_state(exp_dir / "state.json")
    runs = expand_runs(exp, exp_dir)

    started_at = state.get("started_at")
    last_evt = state.get("last_event_at")
    started_age = _format_age(started_at)
    last_evt_age = _format_age(last_evt)
    print(f"Experiment: {exp.exp_id}")
    print(f"Source:     {exp.source}")
    print(f"Stages:     {n_stages}    Mode: {exp.mode}    "
          f"step_multiplier: {exp.step_multiplier}")
    print(f"Started:    {started_at}"
          f"{f'  ({started_age})' if started_age else ''}")
    print(f"Last activity: {last_evt}"
          f"{f'  ({last_evt_age})' if last_evt_age else ''}")
    print(f"  (= last time the scheduler dispatched or reaped a job; long "
          f"gaps just mean a stage is mid-training)")
    print()

    longest = max((len(r.run_id) for r in runs), default=0)
    print(f"{'run_id':<{longest}}  {'stages':>10}  progress")
    print("-" * (longest + 30))
    for run in runs:
        done = runs_mod.count_done_stages(run.run_dir, n_stages)
        bar = "█" * done + "·" * (n_stages - done)
        flag = " ✓" if done == n_stages else ""
        print(f"{run.run_id:<{longest}}  {done:>4}/{n_stages:<5}  {bar}{flag}")

    in_flight = state.get("running", [])
    if in_flight:
        print(f"\nIn flight ({len(in_flight)}):")
        for e in in_flight:
            alive = "alive" if pid_alive(e["pid"]) else "DEAD"
            print(f"  {e['run_id']} stage {e['stage_idx']}  "
                  f"pid={e['pid']} ({alive})  port={e.get('port_base')}  "
                  f"started={e.get('started_at')}")

    failed = state.get("failed", [])
    if failed:
        print(f"\nFailed (last {min(10, len(failed))} of {len(failed)}):")
        for e in failed[-10:]:
            print(f"  {e['run_id']} stage {e['stage_idx']}  "
                  f"exit={e['exit_code']}  at={e['at']}")
            if e.get("log_path"):
                print(f"    log: {e['log_path']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="scheduler",
        description="Run / resume / inspect ratsim experiment defs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run / resume an experiment")
    p_run.add_argument("exp", help="Experiment id (looked up in defs/) or path to a def yaml")
    p_run.add_argument(
        "--machine", default=os.environ.get("RATSIM_SCHEDULER_MACHINE"),
        help="Machine config: bare name (resolved against scheduler/machines/) "
             "or path. Defaults to scheduler/machines/default.yaml. "
             "Can also be set via $RATSIM_SCHEDULER_MACHINE.")
    p_run.add_argument(
        "--step-multiplier", type=float, default=None,
        help="Override the def's step_multiplier (e.g. 0.01 for smoke tests).")
    p_run.add_argument(
        "--restart", action="store_true",
        help="Wipe results/experiments/<exp_id>/ before starting "
             "(equivalent to rm -rf + run). Default behavior is to resume.")
    p_run.set_defaults(func=cmd_run)

    p_st = sub.add_parser("status", help="Show experiment status")
    p_st.add_argument("exp", help="Experiment id or path to a def yaml")
    p_st.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
