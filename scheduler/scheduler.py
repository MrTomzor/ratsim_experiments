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
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import yaml

# psutil is a soft dep — only needed if some profile sets max_ram_gb. We
# warn at startup if it's set but the import failed.
try:
    import psutil
except ImportError:
    psutil = None

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
    # Set when the scheduler kills the job for exceeding max_ram_gb. Reaping
    # logic uses this to skip the consecutive-failure increment so dreamer-
    # style runs that need many restarts don't get blocked.
    ram_killed: bool = False
    # True if this job got the PortAllocator's persistent slot (port 9000)
    # instead of a fresh 9100+ window. Build_command omits base_port= for
    # such jobs so train.py's allocate_unity_instances takes the
    # attach-to-9000 path; release() routing on the allocator also depends
    # on this.
    is_persistent: bool = False


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


def _process_tree_rss_gb(pid: int) -> float | None:
    """Return RSS of the process + all descendants in GB, or None if psutil
    isn't available or the process is gone. Used by the RAM watchdog.

    Counting descendants matters because train.py spawns Unity child
    instances; we want to attribute their RSS to the train process group."""
    if psutil is None:
        return None
    try:
        proc = psutil.Process(pid)
        rss = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                rss += child.memory_info().rss
            except psutil.NoSuchProcess:
                pass
        return rss / (1024 ** 3)
    except psutil.NoSuchProcess:
        return None


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


def _read_jsonl_episode_records(jsonl_path: Path) -> list[dict]:
    """Return per-episode records from a train_episodes.jsonl file.

    Each entry has stage_idx, steps, wall_time_s, total_score, objects_found.
    Skips empty lines and the final line if it's mid-write (incomplete JSON)
    — the env writes line by line so the tail is always either complete or
    a single half-written record at most. Lines missing any of the required
    fields are skipped."""
    out: list[dict] = []
    try:
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ep = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ("stage_idx" not in ep or "steps" not in ep
                        or "wall_time_s" not in ep):
                    continue
                out.append({
                    "stage_idx": int(ep["stage_idx"]),
                    "steps": int(ep["steps"]),
                    "wall_time_s": float(ep["wall_time_s"]),
                    "total_score": float(ep.get("total_score", 0.0)),
                    "objects_found": float(ep.get("objects_found", 0.0)),
                })
    except OSError:
        return []
    return out


def aggregate_fps_by_method(runs: list[Run], recent_window: int = 50) -> dict[str, dict]:
    """For each method, sum steps and wall_time across all runs (cumulative)
    and across the last `recent_window` episodes (recent). Returns
    {method: {steps, wall_time_s, fps, n_episodes, n_runs,
              recent_fps, recent_steps, recent_wall_s}}."""
    cumulative: dict[str, dict] = {}
    # Recent window pools the tail of each run's jsonl. Mixing tails across
    # runs of the same method is an approximation but cheap and representative
    # for "what's the typical recent throughput".
    per_method_recent_pool: dict[str, list[tuple[int, float]]] = {}
    for run in runs:
        records = _read_jsonl_episode_records(run.run_dir / "train_episodes.jsonl")
        if not records:
            continue
        method = run.method.name
        d = cumulative.setdefault(method, {
            "steps": 0, "wall_time_s": 0.0, "n_episodes": 0,
            "n_runs": set(),
        })
        for r in records:
            d["steps"] += r["steps"]
            d["wall_time_s"] += r["wall_time_s"]
            d["n_episodes"] += 1
            d["n_runs"].add(run.run_id)
        for r in records[-recent_window:]:
            per_method_recent_pool.setdefault(method, []).append(
                (r["steps"], r["wall_time_s"]))

    out: dict[str, dict] = {}
    for method, d in cumulative.items():
        if d["wall_time_s"] <= 0:
            continue
        recent = per_method_recent_pool.get(method, [])[-recent_window:]
        recent_steps = sum(s for s, _ in recent)
        recent_wall = sum(w for _, w in recent)
        out[method] = {
            "steps": d["steps"],
            "wall_time_s": d["wall_time_s"],
            "fps": d["steps"] / d["wall_time_s"],
            "n_episodes": d["n_episodes"],
            "n_runs": len(d["n_runs"]),
            "recent_fps": (recent_steps / recent_wall) if recent_wall > 0 else None,
            "recent_steps": recent_steps,
            "recent_wall_s": recent_wall,
            "recent_n": len(recent),
        }
    return out


def aggregate_per_job_heartbeat(in_flight: list[dict], runs: list[Run],
                                 exp_dir: Path,
                                 recent_window: int = 20) -> dict[tuple[str, int], dict]:
    """For each in-flight (run_id, stage_idx), measure recency + throughput
    from `train_episodes.jsonl`. Designed to surface stalled jobs in the
    status display — env-step rate alone misses "process is alive but not
    completing episodes" failures.

    Returns {(run_id, stage_idx): {last_ep_age_s, fps_recent, n_eps_stage,
    log_age_s}}. log_age_s is the mtime of the active scheduler log file
    (catches stalls within an episode, not just between them)."""
    run_dir_by_id = {r.run_id: r.run_dir for r in runs}
    out: dict[tuple[str, int], dict] = {}
    now = time.time()
    for entry in in_flight:
        run_id = entry["run_id"]
        stage_idx = int(entry["stage_idx"])
        run_dir = run_dir_by_id.get(run_id)
        if run_dir is None:
            continue
        jsonl = run_dir / "train_episodes.jsonl"
        last_ep_age = None
        if jsonl.exists():
            last_ep_age = now - jsonl.stat().st_mtime

        records = _read_jsonl_episode_records(jsonl)
        stage_eps = [r for r in records if r["stage_idx"] == stage_idx]
        recent = stage_eps[-recent_window:]
        recent_steps = sum(r["steps"] for r in recent)
        recent_wall = sum(r["wall_time_s"] for r in recent)
        fps_recent = (recent_steps / recent_wall) if recent_wall > 0 else None

        # log_age catches in-episode stalls — JSONL only updates on
        # terminate/truncate, but train.py's TaskTracker prints periodic
        # debug lines as it steps, so the log file mtime moves whenever
        # the env is still actually stepping.
        log_age = None
        log_path = entry.get("log_path")
        if log_path:
            try:
                log_age = now - (exp_dir / log_path).stat().st_mtime
            except OSError:
                pass

        out[(run_id, stage_idx)] = {
            "last_ep_age_s": last_ep_age,
            "log_age_s": log_age,
            "fps_recent": fps_recent,
            "n_eps_stage": len(stage_eps),
        }
    return out


def aggregate_perf_by_stage(runs: list[Run], recent_window: int = 50
                            ) -> dict[tuple[str, str, int], dict]:
    """Per (variation, method, stage_idx): mean total_score and objects_found
    over the last `recent_window` episodes within that stage, averaged across
    seeds (mean of per-run means — equal weight per seed regardless of
    episode count). Returns {(variation, method, stage_idx):
    {reward_mean, pickups_mean, n_seeds}}."""
    # First pass: per-run, per-stage means over the last N within each stage.
    per_run_stage: dict[str, dict[int, tuple[float, float]]] = {}
    run_meta = {r.run_id: (r.variation.name, r.method.name) for r in runs}

    for run in runs:
        records = _read_jsonl_episode_records(run.run_dir / "train_episodes.jsonl")
        if not records:
            continue
        # Bucket by stage_idx, then take the last N within each stage.
        by_stage: dict[int, list[dict]] = {}
        for r in records:
            by_stage.setdefault(r["stage_idx"], []).append(r)
        per_run_stage[run.run_id] = {}
        for s_idx, eps in by_stage.items():
            tail = eps[-recent_window:]
            if not tail:
                continue
            reward_mean = sum(r["total_score"] for r in tail) / len(tail)
            pickups_mean = sum(r["objects_found"] for r in tail) / len(tail)
            per_run_stage[run.run_id][s_idx] = (reward_mean, pickups_mean)

    # Second pass: aggregate by (variation, method, stage_idx).
    bucket: dict[tuple[str, str, int], list[tuple[float, float]]] = {}
    for run_id, stage_data in per_run_stage.items():
        var, method = run_meta[run_id]
        for s_idx, (rm, pm) in stage_data.items():
            bucket.setdefault((var, method, s_idx), []).append((rm, pm))

    out: dict[tuple[str, str, int], dict] = {}
    for key, pairs in bucket.items():
        rms = [p[0] for p in pairs]
        pms = [p[1] for p in pairs]
        out[key] = {
            "reward_mean": sum(rms) / len(rms),
            "pickups_mean": sum(pms) / len(pms),
            "n_seeds": len(pairs),
        }
    return out


def _format_elapsed(secs: float) -> str:
    s = int(secs)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m{s % 60:02d}s"
    h = s // 3600
    return f"{h}h{(s % 3600) // 60:02d}m"


def _format_si(n: int) -> str:
    """Compact SI: 1500 → '1.5k', 100000 → '100k', 10_000_000 → '10M'."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:g}M"
    if n >= 1_000:
        return f"{n / 1_000:g}k"
    return str(int(n))


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
                  step_multiplier: float,
                  is_persistent: bool = False) -> list[str]:
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
        f"n_envs={profile.n_envs}",
    ]
    # When the job is on the persistent slot, omit base_port — train.py's
    # allocate_unity_instances(n_envs=1) without a base_port hits the
    # "attach to PERSISTENT_PORT (9000) if alive, else spawn fresh on 9000"
    # path, which is exactly what we want.
    if not is_persistent:
        cmd.append(f"base_port={port_base}")
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
              step_multiplier: float,
              is_persistent: bool = False) -> ActiveJob:
    cmd = build_command(run, stage_idx, profile, port_base, exp, step_multiplier,
                        is_persistent=is_persistent)
    log_dir = run.run_dir / "scheduler_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"stage_{stage_idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    port_label = f"{port_base} (persistent)" if is_persistent else str(port_base)
    print(f"[scheduler] dispatch {run.run_id} stage {stage_idx} "
          f"port={port_label} n_envs={profile.n_envs}")
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
        is_persistent=is_persistent,
    )


def _is_unity_alive(port: int, host: str = "127.0.0.1",
                    timeout: float = 0.5) -> bool:
    """Quick TCP probe — true if something accepts on this port. Used to
    check whether a manually-launched Unity is listening on the persistent
    slot before the scheduler hands it to a job."""
    import socket
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (ConnectionRefusedError, socket.timeout, OSError):
        return False


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

    # Warn if any profile sets max_ram_gb but psutil isn't installed —
    # the watchdog will silently no-op otherwise.
    has_ram_limits = any(p.max_ram_gb is not None for p in machine.method_profiles.values())
    if has_ram_limits and psutil is None:
        print("[scheduler] WARNING: a method profile sets max_ram_gb but "
              "psutil is not importable — RAM watchdog will be inactive. "
              "`pip install psutil` in the SB3 venv to enable.")

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
    use_port_9000 = bool(getattr(args, "use_port_9000", False))
    port_alloc = PortAllocator(start=9100, window_size=10,
                               persistent_port=9000 if use_port_9000 else None)
    if use_port_9000:
        print(f"[scheduler] --use-port-9000 enabled: port 9000 will be "
              f"handed to one n_envs=1 dispatch when Unity is alive there. "
              f"Manually launch Unity on 9000 (e.g. start_ratsim_headless.sh) "
              f"to watch one training instance.")
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
        # ---- RAM watchdog: SIGTERM jobs that exceed max_ram_gb ----
        # The kill is recorded on the ActiveJob so the reaping path below
        # knows not to count it toward the consecutive-failure budget. We
        # only act once per job — once SIGTERM is sent, ram_killed is True
        # and the next iteration just waits for the process to exit.
        for job in active:
            if job.ram_killed or job.profile.max_ram_gb is None:
                continue
            rss_gb = _process_tree_rss_gb(job.popen.pid)
            if rss_gb is None:
                continue
            if rss_gb > job.profile.max_ram_gb:
                print(f"[scheduler] {job.run.run_id} stage {job.stage_idx} "
                      f"RAM={rss_gb:.1f}GB exceeds limit "
                      f"{job.profile.max_ram_gb:.1f}GB — SIGTERM (will resume "
                      f"from last in-stage checkpoint).")
                job.ram_killed = True
                try:
                    os.killpg(job.popen.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

        # ---- Reap finished jobs ----
        for job in list(active):
            rc = job.popen.poll()
            if rc is None:
                continue
            rm.release(job.profile.needs)
            port_alloc.release(job.port_base)
            active.remove(job)
            duration = (datetime.now() - job.started_at).total_seconds()
            if job.ram_killed:
                outcome = f"RAM-killed (exit {rc})"
            else:
                outcome = "done" if rc == 0 else f"FAILED (exit {rc})"
            stage_done = runs_mod.stage_done(job.run.run_dir, job.stage_idx)
            print(f"[scheduler] {job.run.run_id} stage {job.stage_idx} "
                  f"{outcome} in {duration:.0f}s (stage_done={stage_done})")
            state["running"] = [
                e for e in state["running"]
                if not (e["run_id"] == job.run.run_id
                        and e["stage_idx"] == job.stage_idx)
            ]
            if rc != 0 and not job.ram_killed:
                # Real failure: log + tail + bump the retry counter.
                state["failed"].append({
                    "run_id": job.run.run_id,
                    "stage_idx": job.stage_idx,
                    "exit_code": rc,
                    "log_path": str(job.log_path),
                    "at": datetime.now().isoformat(timespec="seconds"),
                })
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
            elif rc != 0 and job.ram_killed:
                # RAM-kill: also reset any prior consecutive-failure count
                # for this stage, since the process was healthy from the
                # scheduler's POV — we just told it to stop. Otherwise a
                # stage that got 1 real failure followed by N RAM-kills
                # could still hit the block threshold.
                consecutive_failures.pop(
                    (job.run.run_id, job.stage_idx), None)
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
            # Prefer the persistent slot (port 9000) for n_envs=1 dispatches
            # when the user has opted in AND Unity is actually alive there.
            # Falls back to a fresh 9100+ window otherwise.
            #
            # IMPORTANT: gate the TCP alive-probe on `persistent_in_use` first.
            # While an active job is on port 9000, opening extra TCP connections
            # to it (which is what _is_unity_alive does) disrupts the active
            # client's session — Unity's connector wasn't built for stray
            # parallel clients. The previous version probed on every candidate
            # every poll, which wedged the in-flight job after a few minutes.
            port_base = None
            is_persistent = False
            if (use_port_9000 and profile.n_envs == 1
                    and not port_alloc.persistent_in_use
                    and _is_unity_alive(9000)):
                cand = port_alloc.try_alloc_persistent()
                if cand is not None:
                    port_base = cand
                    is_persistent = True
            if port_base is None:
                port_base = port_alloc.alloc()
            job = spawn_job(run, stage_idx, profile, port_base, exp,
                            step_multiplier, is_persistent=is_persistent)
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
        heartbeat = aggregate_per_job_heartbeat(in_flight, runs, exp_dir)
        print(f"\nIn flight ({len(in_flight)}):")
        for e in in_flight:
            alive = "alive" if pid_alive(e["pid"]) else "DEAD"
            print(f"  {e['run_id']} stage {e['stage_idx']}  "
                  f"pid={e['pid']} ({alive})  port={e.get('port_base')}  "
                  f"started={e.get('started_at')}")
            hb = heartbeat.get((e["run_id"], int(e["stage_idx"])), {})
            # Two age signals:
            #   log_age = mtime of train.py's stdout log; bumps on every
            #             TaskTracker debug print, so even mid-episode the
            #             log moves while the env is still stepping.
            #   ep_age = mtime of train_episodes.jsonl; only bumps on
            #            episode terminate/truncate.
            # log_age stuck while ep_age is "young-ish" is normal (mid-ep).
            # log_age AND ep_age both stuck > 5 min = stalled.
            log_age = hb.get("log_age_s")
            ep_age = hb.get("last_ep_age_s")
            fps_recent = hb.get("fps_recent")
            n_eps_stage = hb.get("n_eps_stage", 0)

            stalled = (log_age is not None and log_age > 300
                       and (ep_age is None or ep_age > 300))
            tag = "  ⚠ STALLED" if stalled else ""

            parts = []
            if n_eps_stage > 0:
                parts.append(f"{n_eps_stage} eps this stage")
            if fps_recent is not None:
                parts.append(f"recent fps: {fps_recent:.1f}")
            if ep_age is not None:
                parts.append(f"last ep: {_format_elapsed(ep_age)} ago")
            else:
                parts.append("no episodes yet")
            if log_age is not None:
                parts.append(f"log: {_format_elapsed(log_age)} ago")
            print(f"    {' · '.join(parts)}{tag}")

    compact = bool(getattr(args, "compact", False))

    failed = state.get("failed", [])
    if failed:
        if compact:
            # Watch / compact mode: just count, the full list pollutes the
            # screen on every refresh. One-shot status (without --watch / --compact)
            # still shows the last 10 with log paths so you can scroll.
            print(f"\nFailed: {len(failed)} attempts so far. "
                  f"Re-run `python scheduler_status.py {args.exp}` "
                  f"(no --watch) to see the list.")
        else:
            print(f"\nFailed (last {min(10, len(failed))} of {len(failed)}):")
            for e in failed[-10:]:
                print(f"  {e['run_id']} stage {e['stage_idx']}  "
                      f"exit={e['exit_code']}  at={e['at']}")
                if e.get("log_path"):
                    print(f"    log: {e['log_path']}")

    # FPS by method — cumulative across the whole experiment so far, plus a
    # rolling "recent" window of the last 50 episodes per method. Recent FPS
    # catches slowdowns; cumulative gives the bulk throughput. Both are
    # env-step rates (sim throughput), not policy-update rates.
    fps_stats = aggregate_fps_by_method(runs)
    if fps_stats:
        print(f"\nFPS by method  (env-step rate from train_episodes.jsonl):")
        longest = max(len(m) for m in fps_stats)
        print(f"  {'method':<{longest}}  {'cumul fps':>9}  "
              f"{'recent fps':>10}  {'total steps':>12}  {'elapsed':>9}  "
              f"{'eps':>6}  runs")
        for m in sorted(fps_stats):
            d = fps_stats[m]
            recent = (f"{d['recent_fps']:.1f}" if d['recent_fps'] is not None
                      else "  —  ")
            print(f"  {m:<{longest}}  {d['fps']:>9.1f}  "
                  f"{recent:>10}  {d['steps']:>12,}  "
                  f"{_format_elapsed(d['wall_time_s']):>9}  "
                  f"{d['n_episodes']:>6}  {d['n_runs']}")
        print(f"  (recent fps = mean over last ~50 episodes per method)")

    # Per-stage performance tables: reward + pickups, columns = stages, rows
    # = (variation, method). Each cell is "mean (n_seeds)". Only stages where
    # at least one seed has data are shown.
    perf = aggregate_perf_by_stage(runs)
    if perf:
        single_var = len(exp.variations) == 1
        # Cumulative end-step per stage (uses def's step_multiplier — if you
        # ran with --step-multiplier the labels won't reflect that, but the
        # stage idx still does, which is the part that matters).
        cumulative = []
        cum = 0
        for stage in exp.stages:
            cum += int(stage.steps * exp.step_multiplier)
            cumulative.append(cum)

        stages_with_data = sorted({s_idx for (_, _, s_idx) in perf})
        row_keys = sorted({(v, m) for (v, m, _) in perf})

        def _row_label(v, m):
            return m if single_var else f"{v}/{m}"

        label_w = max(len(_row_label(v, m)) for v, m in row_keys)
        col_w = 11   # fits "12.34 (10)" + a bit

        def _print_table(title, value_key):
            print(f"\n=== {title} — mean of last ~50 eps per stage, averaged over seeds ===")
            line1 = " " * (label_w + 2)
            line2 = " " * (label_w + 2)
            for s_idx in stages_with_data:
                line1 += f"  {('s' + str(s_idx)):<{col_w}}"
                line2 += f"  {('@' + _format_si(cumulative[s_idx])):<{col_w}}"
            print(line1)
            print(line2)
            for v, m in row_keys:
                row = f"  {_row_label(v, m):<{label_w}}"
                for s_idx in stages_with_data:
                    key = (v, m, s_idx)
                    if key in perf:
                        d = perf[key]
                        cell = f"{d[value_key]:.2f} ({d['n_seeds']})"
                    else:
                        cell = "—"
                    row += f"  {cell:<{col_w}}"
                print(row)

        _print_table("Reward", "reward_mean")
        _print_table("Pickups", "pickups_mean")


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
    p_run.add_argument(
        "--use-port-9000", action="store_true", dest="use_port_9000",
        help="Also consider port 9000 as a single-slot port for one n_envs=1 "
             "dispatch at a time. The slot is only used when Unity is "
             "actually alive on 9000 (TCP probe at dispatch time). Useful "
             "for manually launching a Unity GUI on 9000 and watching one "
             "training instance live; other dispatches still go to 9100+ "
             "as usual.")
    p_run.set_defaults(func=cmd_run)

    p_st = sub.add_parser("status", help="Show experiment status")
    p_st.add_argument("exp", help="Experiment id or path to a def yaml")
    p_st.add_argument(
        "--compact", action="store_true",
        help="Hide the failed-runs list.")
    p_st.set_defaults(func=cmd_status)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
