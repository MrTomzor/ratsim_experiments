"""Record one trajectory per (experiment, method) for the trajectory figure.

Rows of the figure are experiments (each def fixes one world preset), columns
are methods. This script evaluates ONE run per method per experiment — a
randomly picked training seed unless --run-seed pins it — on
--n-worldseeds eval worlds with trajectory recording on, and writes a
manifest `<exp_dir>/trajectories.json` that plot_trajectories.py reads.

    python record_trajectories.py memory_3malls memory_orthomaze --methods ppo,dreamer,frontier
    python record_trajectories.py memory_3malls memory_orthomaze --methods human      # later, at the keyboard
    python record_trajectories.py memory_3malls --methods ppo --n-worldseeds 2 --run-seed 0 --difficulty 0.5
    python record_trajectories.py memory_3malls --methods ppo --snapshot ortho   # + overhead pictures
    python plot_trajectories.py memory_3malls memory_orthomaze [--background ortho]

How the worlds line up: RL methods go through eval_one_run(.py|_dreamer.py),
which draws world seeds from --eval-metaseed, so every RL method sees the
same worlds. The seeds are read back from the first RL run's JSONL and
frontier / human are then run on exactly those seeds through test.py
(eval_seeds=..., results in <exp_dir>/external/<method>/). If you only ask
for frontier/human, the seeds come from the existing manifest, or from
--worldseeds.

Experiment ids are looked up under results/rci/<exp> (pull_run.sh mirror; pull
checkpoints first). --local switches to results/experiments/<exp> (runs trained
here); there is no fallback between the two. A path is always used as given.
Needs $PPO_PYTHON_PATH / $DREAMER_PYTHON_PATH like the scheduler; frontier
needs ROS sourced in this shell; human needs you at the keyboard (test.py
waits for Enter before each episode). Unity: attaches to :9000, or
--spawn-unity. --snapshot VIEWS additionally renders each recorded world from
Unity (snapshot_worlds.py; needs a rendering Unity on --port) so
plot_trajectories.py --background <view> can draw on the real arena.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from analyze_experiment import (
    EVAL_SCRIPT_BY_METHOD,
    parse_run_id,
    python_for_method,
)

HERE = Path(__file__).parent
RESULTS_ROOTS = {"rci": HERE / "results" / "rci",          # pulled cluster runs (default)
                 "local": HERE / "results" / "experiments"}  # runs trained on this machine
EXTERNAL_METHODS = ("frontier", "human")
MANIFEST_NAME = "trajectories.json"
METHOD_ALIASES = {"dreamerv3": "dreamer"}


# ─────────────────────────────────────────────
#  Lookup
# ─────────────────────────────────────────────

def resolve_exp_dir(name: str, source: str = "rci") -> Path:
    """Experiment dir for `name`: an existing path is used as-is, otherwise the id is
    looked up under exactly one results root (`rci` = pulled cluster runs, the default;
    `local` = results/experiments). No fallback between roots, so a stale local copy can
    never shadow the cluster one."""
    p = Path(name)
    if p.is_dir():
        return p.resolve()
    root = RESULTS_ROOTS[source]
    if (root / name).is_dir():
        return (root / name).resolve()
    other = [k for k, r in RESULTS_ROOTS.items() if k != source and (r / name).is_dir()]
    hint = (f" (it exists under {other[0]!r}: pass --{other[0]} or the path)" if other
            else (f" (pull_run.sh {name} first?)" if source == "rci" else ""))
    sys.exit(f"ERROR: experiment '{name}' not found under {root}{hint}")


def completed_runs(exp_dir: Path, method: str) -> list[dict]:
    """Runs of `method` with at least one finished stage checkpoint."""
    out = []
    for d in sorted((exp_dir / "runs").iterdir()) if (exp_dir / "runs").is_dir() else []:
        parsed = parse_run_id(d.name)
        if not d.is_dir() or parsed is None:
            continue
        variation, m, seed = parsed
        if METHOD_ALIASES.get(m, m) != method:
            continue
        if not list((d / "checkpoints").glob("stage_*.done")):
            continue
        out.append({"run_dir": d, "run_id": d.name, "variation": variation,
                    "method": method, "seed": seed})
    return out


def load_manifest(exp_dir: Path) -> dict:
    p = exp_dir / MANIFEST_NAME
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {"exp_id": exp_dir.name, "exp_dir": str(exp_dir), "world_seeds": [],
            "methods": {}}


def save_manifest(exp_dir: Path, man: dict) -> Path:
    man["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
    p = exp_dir / MANIFEST_NAME
    with open(p, "w") as f:
        json.dump(man, f, indent=2, default=str)
    return p


def read_records(jsonl: Path) -> list[dict]:
    if not jsonl.exists():
        return []
    out = []
    with open(jsonl) as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def episodes_from_records(recs: list[dict], base: Path,
                          world_seeds: list[int] | None) -> list[dict]:
    """One manifest episode per world seed (latest record wins), absolute npz."""
    by_ws: dict[int, dict] = {}
    for r in recs:
        if not r.get("trajectory_file") or r.get("world_seed") is None:
            continue
        ws = int(r["world_seed"])
        if world_seeds is not None and ws not in world_seeds:
            continue
        npz = Path(r["trajectory_file"])
        npz = npz if npz.is_absolute() else base / npz
        by_ws[ws] = {
            "world_seed": ws,
            "episode_idx": r.get("episode_idx"),
            "npz": str(npz),
            "steps": r.get("steps"),
            "objects_found": r.get("objects_found"),
            "total_score": r.get("total_score"),
            "termination_reason": r.get("termination_reason"),
        }
    order = world_seeds if world_seeds is not None else list(by_ws)
    return [by_ws[ws] for ws in order if ws in by_ws]


def episodes_from_npz_dir(traj_dir: Path, world_seeds: list[int] | None) -> list[dict]:
    """Fallback when episodes.jsonl is missing (e.g. the run was interrupted
    after the bridge wrote the npz but before test.py logged the line): the
    npz meta block carries the same per-episode fields."""
    from ratsim.task_tracker.trajectory_record import load_trajectory
    recs = []
    for npz in sorted(traj_dir.glob("*.npz")) if traj_dir.is_dir() else []:
        try:
            m = load_trajectory(npz)["meta"]
        except Exception as e:  # unreadable file — skip, say so
            print(f"  [warn] cannot read {npz}: {e}")
            continue
        recs.append({**m, "trajectory_file": str(npz)})
    return episodes_from_records(recs, traj_dir, world_seeds)


def run(cmd: list[str], dry: bool) -> int:
    print("  $ " + " ".join(str(c) for c in cmd), flush=True)
    if dry:
        return 0
    return subprocess.run([str(c) for c in cmd]).returncode


# ─────────────────────────────────────────────
#  Per-method recording
# ─────────────────────────────────────────────

def record_rl(exp_dir: Path, method: str, args, rng: random.Random,
              man: dict) -> bool:
    runs = completed_runs(exp_dir, method)
    if not runs:
        print(f"  [skip] {method}: no run with a finished checkpoint under {exp_dir / 'runs'}")
        return False
    if args.run_seed is not None:
        runs = [r for r in runs if r["seed"] == args.run_seed]
        if not runs:
            print(f"  [skip] {method}: no run with seed {args.run_seed}")
            return False
    if args.variation:
        runs = [r for r in runs if r["variation"] == args.variation] or runs
    chosen = rng.choice(runs)
    print(f"  {method}: run {chosen['run_id']} "
          f"({'pinned' if args.run_seed is not None else 'random pick'} of {len(runs)})")
    try:
        python = python_for_method(method)
    except (ValueError, EnvironmentError) as e:
        print(f"  [skip] {method}: {e}")
        return False
    cmd = [python, EVAL_SCRIPT_BY_METHOD[method],
           "--run_dir", chosen["run_dir"], "--exp_dir", exp_dir,
           "--n_episodes", args.n_worldseeds,
           "--eval_metaseed", args.eval_metaseed, "--record-trajectories"]
    if args.difficulty is not None:
        cmd += ["--difficulty", args.difficulty]
    if args.deterministic:
        cmd.append("--deterministic")
    if args.spawn_unity:
        cmd.append("--spawn")
    rc = run(cmd, args.dry_run)
    if rc != 0:
        print(f"  [fail] {method}: eval exited {rc}")
        return False
    if args.dry_run:
        return True
    recs = read_records(chosen["run_dir"] / "eval_episodes.jsonl")
    eps = episodes_from_records(recs, chosen["run_dir"], None)[: args.n_worldseeds]
    if not eps:
        print(f"  [fail] {method}: no trajectory_file in eval_episodes.jsonl")
        return False
    seeds = [e["world_seed"] for e in eps]
    if man["world_seeds"] and man["world_seeds"][: len(seeds)] != seeds:
        print(f"  [warn] {method}: world seeds {seeds} differ from manifest "
              f"{man['world_seeds']} — different eval metaseed/difficulty? "
              f"Manifest keeps the earlier ones; this column will not line up.")
    elif not man["world_seeds"]:
        man["world_seeds"] = seeds
    wc = chosen["run_dir"] / "eval_world_config.json"
    if wc.exists():
        man["world_config_file"] = str(wc)
    man["methods"][method] = {
        "run_id": chosen["run_id"], "run_seed": chosen["seed"],
        "variation": chosen["variation"], "kind": "rl",
        "eval_metaseed": args.eval_metaseed, "difficulty": args.difficulty,
        "episodes": eps,
    }
    return True


def record_external(exp_dir: Path, method: str, args, man: dict,
                    method_args: dict) -> bool:
    seeds = args.worldseeds or man["world_seeds"]
    if not seeds and args.dry_run:
        seeds = ["<world_seeds from the RL eval>"]
    elif not seeds:
        print(f"  [skip] {method}: no world seeds yet — record an RL method first "
              f"(or pass --worldseeds).")
        return False
    seeds = seeds[: args.n_worldseeds] if not args.worldseeds else seeds
    if not man["world_seeds"]:
        man["world_seeds"] = list(seeds)
    exp_yaml = exp_dir / "experiment.yaml"
    if not exp_yaml.exists():
        print(f"  [skip] {method}: {exp_yaml} missing")
        return False
    out_dir = exp_dir / "external" / method
    try:
        python = python_for_method("ppo")   # test.py needs the sb3 venv
    except (ValueError, EnvironmentError):
        python = sys.executable
    cmd = [python, HERE / "test.py", f"def={exp_yaml}", f"method={method}",
           f"eval_seeds={','.join(str(s) for s in seeds)}", "episodes_per_seed=1",
           "record_trajectories=1", f"results_dir={out_dir}"]
    if args.difficulty is not None:
        cmd.append(f"difficulty={args.difficulty}")
    if method == "human":
        cmd.append(f"rtf={args.rtf}")
        print("  human: test.py waits for Enter before each episode — you drive.")
    for k, v in method_args.items():
        cmd.append(f"{method}.{k}={v}")
    if args.no_run:
        print(f"  {method}: --no-run, indexing what is already under {out_dir}")
    else:
        rc = run(cmd, args.dry_run)
        if rc != 0:
            print(f"  [fail] {method}: test.py exited {rc}")
            return False
        if args.dry_run:
            return True
    eps = episodes_from_records(read_records(out_dir / "episodes.jsonl"), out_dir, list(seeds))
    if not eps:
        eps = episodes_from_npz_dir(out_dir / "trajectories", list(seeds))
        if eps:
            print(f"  [note] {method}: no episodes.jsonl line for these seeds — "
                  f"indexed {len(eps)} episode(s) from the npz meta instead")
    if not eps:
        print(f"  [fail] {method}: no trajectories under {out_dir}")
        return False
    man["methods"][method] = {
        "run_id": f"external/{method}", "kind": "external",
        "difficulty": args.difficulty, "method_params": method_args,
        "episodes": eps,
    }
    return True


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

_SNAPSHOT_CONN = None


def snapshot_after_recording(exp_dir: Path, man: dict, args, views: list[str],
                             extra: dict) -> bool:
    """Overhead pictures of every world seed in the manifest (see snapshot_worlds.py)."""
    global _SNAPSHOT_CONN
    from snapshot_worlds import snapshot_exp
    seeds = man.get("world_seeds") or []
    if not seeds or (args.dry_run and not isinstance(seeds[0], int)):
        print(f"  [skip] snapshot: no world seeds recorded yet")
        return False
    try:
        if _SNAPSHOT_CONN is None and not args.dry_run:
            from ratsim.world_snapshot import connect_and_select_scene
            _SNAPSHOT_CONN = connect_and_select_scene(agent_preset=args.agent_preset, port=args.port)
        snapshot_exp(exp_dir, man, seeds, views, _SNAPSHOT_CONN, width=args.width,
                     height=args.height, extra=extra, difficulty=args.difficulty,
                     variation=None, dry=args.dry_run)
    except Exception as e:  # noqa: BLE001 — report and keep going with the next experiment
        print(f"  [fail] snapshot: {e}")
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exps", nargs="+", help="experiment ids (one figure row each) or paths")
    ap.add_argument("--methods", required=True,
                    help=f"comma-separated: {sorted(EVAL_SCRIPT_BY_METHOD)} and/or "
                         f"{list(EXTERNAL_METHODS)}")
    ap.add_argument("--n-worldseeds", type=int, default=1, dest="n_worldseeds",
                    help="eval worlds per experiment (default 1 = one row per experiment)")
    ap.add_argument("--run-seed", type=int, default=None, dest="run_seed",
                    help="training seed to evaluate (default: random pick among finished runs)")
    ap.add_argument("--rng-seed", type=int, default=None, dest="rng_seed",
                    help="seed for the random pick, to make it repeatable")
    ap.add_argument("--variation", default=None, help="prefer runs of this variation")
    ap.add_argument("--eval-metaseed", type=int, default=42, dest="eval_metaseed")
    ap.add_argument("--difficulty", type=float, default=None)
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--worldseeds", default=None,
                    help="explicit world seeds for frontier/human when no RL run was recorded")
    ap.add_argument("--rtf", type=float, default=1.0, help="human: real-time factor")
    ap.add_argument("--method-arg", action="append", default=[], metavar="METHOD.KEY=VALUE",
                    help="pass-through for external methods, e.g. frontier.grid_resolution=0.5")
    ap.add_argument("--spawn-unity", action="store_true", dest="spawn_unity")
    ap.add_argument("--no-run", action="store_true", dest="no_run",
                    help="frontier/human: don't run test.py, just (re)index what is already "
                         "under <exp_dir>/external/<method>/ into the manifest")
    ap.add_argument("--local", action="store_true",
                    help="look experiment ids up under results/experiments (runs trained on "
                         "this machine) instead of the default results/rci (pulled cluster runs)")
    ap.add_argument("--snapshot", default=None, metavar="VIEWS",
                    help="after recording, take overhead pictures of each recorded world "
                         "(comma-separated: ortho,persp — both straight down) into <exp_dir>/snapshots/ for "
                         "plot_trajectories.py --background; needs Unity rendering on --port")
    from snapshot_worlds import add_snapshot_args, parse_extra, parse_views
    add_snapshot_args(ap)
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    args = ap.parse_args()
    args.source = "local" if args.local else "rci"
    snap_views = parse_views(args.snapshot, ap) if args.snapshot else []
    snap_extra = parse_extra(args.set, ap)

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in EVAL_SCRIPT_BY_METHOD and m not in EXTERNAL_METHODS]
    if unknown:
        ap.error(f"unknown methods {unknown}")
    if args.worldseeds:
        args.worldseeds = [int(s) for s in args.worldseeds.split(",")]
    method_args: dict[str, dict] = {m: {} for m in EXTERNAL_METHODS}
    for item in args.method_arg:
        key, _, val = item.partition("=")
        m, _, k = key.partition(".")
        if m not in method_args or not k or not _:
            ap.error(f"--method-arg expects METHOD.KEY=VALUE with METHOD in "
                     f"{list(EXTERNAL_METHODS)}, got {item!r}")
        method_args[m][k] = val

    rng = random.Random(args.rng_seed)
    # RL first so the world seeds exist for frontier/human in the same call.
    ordered = [m for m in methods if m in EVAL_SCRIPT_BY_METHOD] + \
              [m for m in methods if m in EXTERNAL_METHODS]

    summary = []
    for exp in args.exps:
        exp_dir = resolve_exp_dir(exp, args.source)
        man = load_manifest(exp_dir)
        print(f"\n=== {exp_dir.name}  ({exp_dir})")
        for m in ordered:
            ok = (record_rl(exp_dir, m, args, rng, man) if m in EVAL_SCRIPT_BY_METHOD
                  else record_external(exp_dir, m, args, man, method_args[m]))
            summary.append((exp_dir.name, m, ok))
            if not args.dry_run:
                save_manifest(exp_dir, man)
        if snap_views:
            ok = snapshot_after_recording(exp_dir, man, args, snap_views, snap_extra)
            summary.append((exp_dir.name, "snapshot", ok))
            if not args.dry_run:
                save_manifest(exp_dir, man)
        if not args.dry_run:
            print(f"  manifest: {exp_dir / MANIFEST_NAME}  world_seeds={man['world_seeds']}")

    print("\nSummary:")
    for exp, m, ok in summary:
        print(f"  {exp:40s} {m:14s} {'ok' if ok else 'FAILED/skipped'}")
    print("\nNext: python plot_trajectories.py " + " ".join(args.exps))


if __name__ == "__main__":
    main()
