#!/usr/bin/env python
"""Submit an experiment to SLURM without hand-typing resources.

    ./submit.sh gps_ablation_5house --time 1d
    ./submit.sh gps_ablation_5house --time 4h --mode bfs      # short taster
    ./submit.sh dreamer_ladder --time 3d --dry-run

You choose the experiment and how long you're willing to give it. Everything
else — partition, --cpus-per-task, --gres, --mem, and how the def splits across
jobs — is derived from the def and the machine configs, which are the things
that actually know.

Three jobs this does for you:

1. **Duration → partition.** `--time 4h` is `amdfast`, `1d` is `amd`, `3d` is
   `amdlong`. Partition names are lookup, not knowledge, and the existing
   scheduler_job.sbatch defaults to the 4 h smoke partition — a trap that kills
   real runs mid-stage (RCI_CLUSTER_PORT.md §0.1).

2. **Split by hardware class.** A def mixing PPO and dreamer cannot run as one
   job: one scheduler holds one allocation, and rci.yaml has no dreamer profile
   at all. So methods needing a GPU go to a GPU job and the rest to a CPU job,
   both feeding the same exp_id — one W&B group, one analysis. The split is read
   off the machine configs (`needs.gpu`), not hardcoded here, so changing a
   method's device is a one-line profile edit.

3. **Check the account ceiling before you hit it.** The per-partition-group CPU
   and GPU caps are *aggregate across all your running jobs*, not per-job
   (§0.55). Two 112-thread PPO jobs are 224 threads against a 200 cap — the
   second just pends, with nothing to tell you why. This warns up front.

Resuming is the default and needs no special flag: submit the same line again
and the scheduler skips stages that already have `.done` markers.
"""
from __future__ import annotations

import argparse
import getpass
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from experiment_defs import load_experiment_def, resolve_def_path

REPO = Path(__file__).resolve().parent
MACHINE_DIR = REPO / "scheduler" / "machines"

# Wall-clock tiers, longest-first match done by ascending limit. Names are the
# keys used under `sbatch.partitions` in the machine configs.
TIERS = [
    ("fast",       4 * 3600),
    ("day",       24 * 3600),
    ("long",      72 * 3600),
    ("extralong", 21 * 24 * 3600),
]

# Per-user AGGREGATE caps across all simultaneously running jobs in the group —
# not per-job limits. From RCI's web interface, recorded in §0.55.
CAPS = {
    "fast":      {"cpus": 700, "gpus": 8},
    "day":       {"cpus": 200, "gpus": 8},
    "long":      {"cpus": 200, "gpus": 6},
    "extralong": {"cpus": 200, "gpus": 8},
}

DEFAULT_CPU_MACHINE = "rci"
# rci_gpu2, not rci_gpu: the 2-card config is the one verified on hardware
# (§0.96), and the 1-card one fits only a single run at a time.
DEFAULT_GPU_MACHINE = "rci_gpu2"


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------

def parse_duration(s: str) -> int:
    """'4h' / '1d' / '90m' / '3-00:00:00' / '12:00:00' → seconds."""
    s = s.strip()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([smhd])", s, re.I)
    if m:
        n, unit = float(m.group(1)), m.group(2).lower()
        return int(n * {"s": 1, "m": 60, "h": 3600, "d": 86400}[unit])
    m = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d{2})(?::(\d{2}))?", s)
    if m:
        d, h, mi, sec = (int(g or 0) for g in m.groups())
        return d * 86400 + h * 3600 + mi * 60 + sec
    raise argparse.ArgumentTypeError(
        f"could not parse --time {s!r}; use 4h, 1d, 3d, 90m or 3-00:00:00")


def slurm_time(seconds: int) -> str:
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, sec = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:{sec:02d}" if d else f"{h:02d}:{m:02d}:{sec:02d}"


def pick_tier(seconds: int) -> str:
    for name, limit in TIERS:
        if seconds <= limit:
            return name
    raise SystemExit(
        f"[submit] --time exceeds the longest partition "
        f"({TIERS[-1][1] // 86400} days).")


# ---------------------------------------------------------------------------
# Machine configs
# ---------------------------------------------------------------------------

def load_machine_raw(name: str) -> dict:
    path = MACHINE_DIR / f"{name}.yaml" if "/" not in name else Path(name)
    if not path.exists():
        raise SystemExit(f"[submit] machine config not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    raw["_name"] = path.stem
    raw["_path"] = path
    return raw


def needs_gpu(machine_raw: dict, method: str) -> bool:
    """Does this method want a GPU, per the GPU machine's profile for it?

    Read off the config rather than hardcoded, so `ppo` moving to `cuda` (or
    dreamer to CPU) is a profile edit and this stays correct."""
    prof = (machine_raw.get("method_profiles") or {}).get(method)
    if prof is None:
        return False
    return int((prof.get("needs") or {}).get("gpu", 0)) > 0


# ---------------------------------------------------------------------------
# Account ceiling
# ---------------------------------------------------------------------------

def running_usage(partitions: set[str]) -> tuple[int, int] | None:
    """(cpus, gpus) already committed by your queued/running jobs in these
    partitions. Returns None if squeue is unavailable or unparseable — this is
    advisory, so a failure here must not block a submission."""
    if not shutil.which("squeue"):
        return None
    try:
        out = subprocess.run(
            ["squeue", "-h", "-u", getpass.getuser(),
             "--states=RUNNING,PENDING", "-o", "%P|%C|%b"],
            capture_output=True, text=True, timeout=20, check=True).stdout
    except (subprocess.SubprocessError, OSError):
        return None
    cpus = gpus = 0
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 3 or parts[0].strip().rstrip("*") not in partitions:
            continue
        try:
            cpus += int(parts[1])
        except ValueError:
            pass
        m = re.search(r"gpu[:\w]*?:(\d+)", parts[2])
        if m:
            gpus += int(m.group(1))
    return cpus, gpus


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------

class Job:
    def __init__(self, machine: dict, methods: list[str], n_runs: int,
                 emit_filter: bool):
        self.machine = machine
        # Always the real method list this job runs — concurrency() and the
        # printed plan both need it. Whether to pass --methods on the CLI is a
        # separate question (emit_filter): a job that happens to own the whole
        # def needs no filter, but still runs a specific set of methods. Keeping
        # these as one nullable field was a bug twice: it dropped the flag under
        # --only, and it made concurrency() fall back to the machine's full
        # profile list, which includes CPU-only methods and so silently
        # disabled the GPU bound below.
        self.methods = methods
        self.emit_filter = emit_filter
        self.n_runs = n_runs
        res = machine.get("resources") or {}
        sb = machine.get("sbatch") or {}
        self.cpus = int(res.get("cpu_slot", 1))
        self.gpus = int(res.get("gpu", 0))
        self.mem = sb.get("mem", "100G")
        self.partitions = sb.get("partitions") or {}

    def partition(self, tier: str) -> str:
        p = self.partitions.get(tier)
        if not p:
            raise SystemExit(
                f"[submit] {self.machine['_path']} has no sbatch.partitions."
                f"{tier} entry — add one or pick a different --time.")
        return p

    def concurrency(self) -> int:
        """How many of this job's runs fit at once. Reported so the printed plan
        says what you're actually buying.

        Bounded by GPUs as well as cpu_slot: a bucket where every method needs a
        card cannot exceed the card count, however many cores are free. Only
        applied when *every* method in the bucket needs one — a mixed bucket can
        pack CPU runs into the spare cores, and calling that out exactly would
        mean replaying the scheduler's bin-packing here."""
        profs = self.machine.get("method_profiles") or {}
        needs = [(profs.get(m) or {}).get("needs") or {} for m in self.methods]
        per_cpu = [int(n.get("cpu_slot", 0)) for n in needs]
        smallest = min([c for c in per_cpu if c > 0], default=self.cpus)
        fit = self.cpus // smallest
        if needs and all(int(n.get("gpu", 0)) > 0 for n in needs):
            fit = min(fit, self.gpus // min(int(n["gpu"]) for n in needs))
        return max(1, min(self.n_runs, fit))


def plan(exp, args) -> list[Job]:
    gpu_machine = load_machine_raw(args.gpu_machine)
    runs_per_method = {m.name: len(exp.variations) * m.n_seeds for m in exp.methods}

    if args.machine:
        # Explicit override: one job, everything on the named machine, no
        # --methods. This is the escape hatch for "put all 12 runs on the GPU
        # node" (PPO gets masked off the cards — verified in §0.96).
        mc = load_machine_raw(args.machine)
        return [Job(mc, list(runs_per_method), sum(runs_per_method.values()),
                    emit_filter=False)]

    gpu_methods = [m.name for m in exp.methods if needs_gpu(gpu_machine, m.name)]
    cpu_methods = [m.name for m in exp.methods if m.name not in gpu_methods]
    if args.only:
        gpu_methods = [m for m in gpu_methods if m in args.only]
        cpu_methods = [m for m in cpu_methods if m in args.only]
        if not gpu_methods and not cpu_methods:
            raise SystemExit(
                f"[submit] --only {','.join(args.only)} matched none of "
                f"{', '.join(m.name for m in exp.methods)}")

    # Pass --methods whenever a job owns less than the whole def. That covers
    # both the split (each job takes its hardware class) and --only (one job,
    # but deliberately not all the methods). Omitting it in the --only case
    # would hand the job the def's *other* methods too, and dreamer against
    # rci.yaml fails validation outright.
    all_methods = {m.name for m in exp.methods}

    def mk(machine, methods):
        owns_all = set(methods) == all_methods
        return Job(machine, methods, sum(runs_per_method[m] for m in methods),
                   emit_filter=not owns_all)

    jobs = []
    if cpu_methods:
        jobs.append(mk(load_machine_raw(args.cpu_machine), cpu_methods))
    if gpu_methods:
        jobs.append(mk(gpu_machine, gpu_methods))
    return jobs


def find_sbatch_script(override: str | None) -> Path:
    """Locate scheduler_job.sbatch, which lives in the meta-repo rather than
    here. `__file__` resolves through the meta-repo symlink to the real
    ratsim_experiments checkout, so look next to that as well as beside it."""
    if override:
        path = Path(override)
        if not path.exists():
            raise SystemExit(f"[submit] --sbatch-script not found: {path}")
        return path
    candidates = [
        REPO.parent / "meta_ratsim" / "rci_port_probes" / "scheduler_job.sbatch",
        REPO.parent / "rci_port_probes" / "scheduler_job.sbatch",
        REPO / "rci_port_probes" / "scheduler_job.sbatch",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(
        "[submit] could not find scheduler_job.sbatch; looked in\n  "
        + "\n  ".join(str(c) for c in candidates)
        + "\nPass --sbatch-script <path>.")


def main():
    p = argparse.ArgumentParser(
        prog="submit", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp", help="Experiment id (looked up in defs/) or path to a def yaml")
    p.add_argument("--time", required=True, type=str, metavar="DUR",
                   help="Wall clock: 4h, 1d, 3d, 90m, or 3-00:00:00. Picks the "
                        "partition. Resume is the default, so a short job you "
                        "later extend costs nothing.")
    p.add_argument("--machine", default=None,
                   help="Force every run onto one machine config instead of "
                        "splitting by hardware class.")
    p.add_argument("--only", type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
                   default=None, metavar="M1,M2",
                   help="Submit only these methods' jobs (e.g. stage the CPU "
                        "half now and the GPU half later).")
    p.add_argument("--mode", choices=("bfs", "dfs"), default=None,
                   help="Override the def's dispatch order. dfs finishes runs "
                        "one at a time; bfs advances every run through early "
                        "stages first. Neither is inferred from --time.")
    p.add_argument("--cpu-machine", default=DEFAULT_CPU_MACHINE, dest="cpu_machine")
    p.add_argument("--gpu-machine", default=DEFAULT_GPU_MACHINE, dest="gpu_machine")
    p.add_argument("--dry-run", action="store_true", dest="dry_run",
                   help="Print the sbatch lines without submitting.")
    p.add_argument("--sbatch-script", default=None, dest="sbatch_script",
                   help="Path to scheduler_job.sbatch (default: "
                        "../rci_port_probes/scheduler_job.sbatch).")
    args, passthrough = p.parse_known_args()

    seconds = parse_duration(args.time)
    tier = pick_tier(seconds)
    def_path = resolve_def_path(REPO / "defs", args.exp)
    if not def_path.exists():
        raise SystemExit(f"[submit] experiment def not found: {def_path}")
    exp = load_experiment_def(def_path)
    jobs = plan(exp, args)

    script = find_sbatch_script(args.sbatch_script)

    # Seeds are per-method (a def may give dreamer fewer than ppo), so the run
    # count is variations × sum-of-seeds, not a clean triple product.
    total_runs = len(exp.variations) * sum(m.n_seeds for m in exp.methods)
    n_stages = len(exp.stages)
    methods_desc = ", ".join(f"{m.name}×{m.n_seeds}" for m in exp.methods)
    print(f"{exp.exp_id}: {total_runs} runs "
          f"({len(exp.variations)} variations × [{methods_desc}] seeds), "
          f"{n_stages} stages × {exp.stages[0].steps:,} steps")
    print(f"  wall clock {slurm_time(seconds)} → {tier} partitions   "
          f"mode {args.mode or exp.mode}")

    cmds = []
    for job in jobs:
        part = job.partition(tier)
        cmd = ["sbatch", "-p", part, f"--time={slurm_time(seconds)}",
               f"--cpus-per-task={job.cpus}", f"--mem={job.mem}"]
        if job.gpus:
            cmd.append(f"--gres=gpu:{job.gpus}")
        cmd += [str(script), args.exp, "--machine", job.machine["_name"]]
        if job.emit_filter:
            cmd += ["--methods", ",".join(job.methods)]
        if args.mode:
            cmd += ["--mode", args.mode]
        cmd += passthrough
        cmds.append((job, part, cmd))

        gres = f", {job.gpus}×GPU" if job.gpus else ""
        what = ",".join(job.methods)
        print(f"  → {part:<18} {job.cpus}t{gres}, {job.mem}   "
              f"{what}  ({job.n_runs} runs, {job.concurrency()} concurrent)")

    # --- account ceiling ---------------------------------------------------
    cap = CAPS[tier]
    want_cpus = sum(j.cpus for j, _, _ in cmds)
    want_gpus = sum(j.gpus for j, _, _ in cmds)
    used = running_usage({part for _, part, _ in cmds})
    have_cpus, have_gpus = used if used else (0, 0)
    note = "" if used else "  (squeue unavailable — counting this submission only)"
    if have_cpus + want_cpus > cap["cpus"] or have_gpus + want_gpus > cap["gpus"]:
        print(f"\n  ⚠️  {tier} group cap: {cap['cpus']} CPUs / {cap['gpus']} GPUs, "
              f"aggregate across all your running jobs.{note}")
        print(f"      already committed {have_cpus} CPUs / {have_gpus} GPUs, "
              f"this adds {want_cpus} / {want_gpus}.")
        print(f"      Over the cap the extra job PENDS indefinitely with no "
              f"error. Consider a shorter --time (the 4 h group allows "
              f"{CAPS['fast']['cpus']} CPUs) or submitting halves with --only.")

    if args.dry_run:
        print("\n--dry-run, not submitting:")
        for _, _, cmd in cmds:
            print("  " + " ".join(cmd))
        return

    print()
    for _, _, cmd in cmds:
        r = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        if r.returncode != 0:
            raise SystemExit(f"[submit] sbatch failed ({r.returncode}); "
                             f"any jobs already submitted are still queued.")
    print(f"\nwandb group: {exp.exp_id}")
    print(f"monitor:     python scheduler_status.py {exp.exp_id} --watch 5   "
          f"(from login2, not login1)")


if __name__ == "__main__":
    main()
