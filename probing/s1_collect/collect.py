"""Record the shared probing dataset: episodes from agent checkpoints and from
the scripted explorer, on fresh worlds, in N parallel Unity instances.

    $DREAMER_PYTHON_PATH -m probing.s1_collect.collect --out results/probing/pilot/dataset \
        --exp_dir results/experiments/memrssm_orthomaze_ladder \
        --variation L0_baseline \
        --policy runs/L0_baseline__dreamer__seed0:5:5 \
        --policy runs/L1_tbtt__dreamer__seed0:5:5 \
        --explorer 10 --workers 4 --spawn

The whole dataset is PLANNED first: episode i gets its source and its world
seed before anything runs, the seeds drawn from default_rng(--metaseed) in the
order env.py would draw them (one warm-up draw for the constructor's reset,
then one per episode -- the same worlds a single env would have produced).
The plan goes into meta.json; the episodes still missing are dealt
round-robin to --workers child processes, each with its own Unity
(allocate_unity_instances spawns them in this process, which keeps them alive
and kills them at exit). Because the plan fixes everything, the result does
not depend on the worker count, and re-running the same command after a
killed job only records the missing episodes.

The environment is read, never changed: the true pose and raw lidar come from
the connector's last reply (it keeps every message until the next read). A
worker does replace env.world_seed_generator after construction with a queue
of its planned seeds -- env.reset only ever calls .integers(0, 2**31 - 1) on
it (env.py:451-457) -- because that is the one way to hand a planned seed to
a reset that runs inside the embodied wrapper.

Format: probing/core/dataset.py.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import embodied  # noqa: F401 -- dreamerv3.main pulls runtime pieces from this
import dreamerv3  # noqa: F401
from dreamerv3.main import wrap_env

from ratsim.config_blender import blend_presets
from ratsim.transforms import yaw_from_quat
from ratsim_wildfire_gym_env.env import WildfireGymEnv

from experiment_defs import (find_variation, load_experiment_def,
                             resolve_agent_preset, resolve_stage_world,
                             resolve_task_preset)
from methods.dreamerv3.env_adapter import GymnasiumToEmbodied
from unity_attach import add_unity_attach_args, resolve_unity_port

from probing.core import agent_io, dataset
from probing.s1_collect.explorer import ScriptedExplorer

SEED_HIGH = 2**31 - 1  # env.py:454


def plan_episodes(sources: list, metaseed: int) -> list[dict]:
    """One entry per episode: index, source, world seed."""
    rng = np.random.default_rng(metaseed)
    rng.integers(0, SEED_HIGH)  # the env constructor's own reset() draws first
    plan = []
    for kind, run, stage, n in sources:
        for _ in range(n):
            plan.append({"idx": len(plan), "kind": kind, "run": run,
                         "stage": stage,
                         "seed": int(rng.integers(0, SEED_HIGH))})
    return plan


def pending(root: Path, plan: list[dict]) -> list[dict]:
    return [p for p in plan
            if not dataset.episode_path(root, p["idx"]).exists()]


class SeedQueue:
    """Stands in for env.world_seed_generator: hands out planned seeds."""

    def __init__(self, seeds):
        self.seeds = [int(s) for s in seeds]

    def integers(self, low, high):
        if not self.seeds:
            raise RuntimeError("more resets than planned episodes")
        return self.seeds.pop(0)


def snapshot(gym_env, obs) -> dict:
    """One step's record: the observation plus what Unity sent alongside it."""
    conn = gym_env.conn
    pose = conn.get_received_messages(gym_env.agent_gt_pose_msg_in_topic)
    lidar = conn.get_received_messages(gym_env.lidar_msg_in_topic)
    if not pose or not lidar:
        raise RuntimeError(
            f"no {gym_env.agent_gt_pose_msg_in_topic} / "
            f"{gym_env.lidar_msg_in_topic} in the last Unity reply")
    p = pose[0]
    return {
        "lidar": np.asarray(obs["lidar"], np.float32),
        "gps": np.asarray(obs["gps"], np.float32),
        "compass": np.asarray(obs["compass"], np.float32),
        "reward": np.float32(obs["reward"]),
        "is_first": bool(obs["is_first"]),
        "is_last": bool(obs["is_last"]),
        "is_terminal": bool(obs["is_terminal"]),
        "pose": np.array([p.x, p.y, yaw_from_quat(p.qx, p.qy, p.qz, p.qw)],
                         np.float32),
        "ranges": np.asarray(lidar[0].ranges, np.float32),
        "pickups": np.int16(gym_env.task_tracker.get_num_reward_objs_picked_up()),
    }


def reset_action(wrapped) -> dict:
    act = {k: np.zeros(s.shape, s.dtype)
           for k, s in wrapped.act_space.items() if k != "reset"}
    act["reset"] = True
    return act


def run_episode(gym_env, wrapped, choose) -> dict:
    """Roll one episode. `choose(step_record, obs)` returns the action dict
    (without `reset`) for the observation just recorded."""
    steps, actions = [], []
    obs = wrapped.step(reset_action(wrapped))
    while True:
        rec = snapshot(gym_env, obs)
        steps.append(rec)
        if rec["is_last"]:
            break
        act = choose(rec, obs)
        actions.append(act)
        obs = wrapped.step({**act, "reset": False})
    out = {k: np.stack([s[k] for s in steps]) for k in steps[0]}
    for key in dataset.ACT_KEYS:
        col = [int(a[key]) for a in actions] + [0]  # no action after the last obs
        out[key] = np.asarray(col, np.int32)
    return out


def policy_chooser(agent):
    carry = [agent.init_policy(1)]

    def choose(rec, obs):
        obs_b = {k: np.asarray(v)[None] for k, v in obs.items()
                 if not k.startswith("log/")}
        carry[0], act, _ = agent.policy(carry[0], obs_b, mode="eval")
        return {k: np.asarray(v)[0] for k, v in act.items()}
    return choose


def explorer_chooser(gym_env, seed: int):
    explorer = ScriptedExplorer(seed=seed)
    tracker = gym_env.task_tracker.exploration_tracker

    def choose(rec, obs):
        x, y, yaw = (float(v) for v in rec["pose"])
        a0, a1 = explorer.act(x, y, yaw, tracker)
        return {"action_0": np.int32(a0), "action_1": np.int32(a1)}
    return choose


def parse_policy(spec: str, exp_dir: Path) -> tuple[Path, int, int]:
    """RUN_DIR:STAGE:N_EPISODES; RUN_DIR may be relative to --exp_dir."""
    run, stage, n = spec.rsplit(":", 2)
    run_dir = Path(run)
    if not run_dir.is_absolute() and not run_dir.exists():
        run_dir = exp_dir / run
    return run_dir.resolve(), int(stage), int(n)


def make_env(meta: dict, port: int):
    presets = meta["presets"]
    gym_env = WildfireGymEnv(
        worldgen_config=blend_presets("world", presets["world"]),
        agent_config=blend_presets("agents", presets["agent"]),
        sensor_config={},
        action_config={"control_mode": "velocity"},
        task_config=blend_presets("task", presets["task"]),
        metaworldgen_config={"world_generation_metaseed": meta["metaseed"]},
        unity_port=port,
    )
    wrapped = wrap_env(GymnasiumToEmbodied(gym_env, obs_key="vector",
                                           act_key="action"), None)
    return gym_env, wrapped


def run_worker(root: Path, jobs: list[dict], port: int, platform, tag: str):
    """Record `jobs` (plan entries, in order) on the Unity at `port`."""
    meta = dataset.load_json(root / "meta.json")
    gym_env, wrapped = make_env(meta, port)
    tracker = gym_env.task_tracker.exploration_tracker
    if tracker is None:
        sys.exit(f"[{tag}] the task preset has no volumetric exploration; "
                 f"targets.py needs the ExplorationTracker grid geometry")
    seeds = SeedQueue(j["seed"] for j in jobs)
    gym_env.world_seed_generator = seeds

    obs_space = {k: v for k, v in wrapped.obs_space.items()
                 if not k.startswith("log/")}
    act_space = {k: v for k, v in wrapped.act_space.items() if k != "reset"}
    spaces = {"obs": dataset.spaces_to_json(obs_space),
              "act": dataset.spaces_to_json(act_space)}
    # First worker creates it, every other one must match (raises if not).
    dataset.publish_once(root / "spaces.json", spaces)

    agents = {}
    try:
        for job in jobs:
            t0 = time.time()
            run_dir = Path(job["run"]) if job["run"] else None
            if job["kind"] == "policy":
                key = (job["run"], job["stage"])
                if key not in agents:
                    config = agent_io.make_config(run_dir, platform=platform)
                    agents[key] = agent_io.build_agent(config, obs_space, act_space)
                    agent_io.load_checkpoint(
                        agents[key], agent_io.stage_checkpoint(run_dir, job["stage"]))
                    print(f"[{tag}] loaded {run_dir.name} stage {job['stage']}")
                choose = policy_chooser(agents[key])
            else:
                choose = explorer_chooser(gym_env, seed=job["seed"])
            left = len(seeds.seeds)
            ep = run_episode(gym_env, wrapped, choose)
            # Exactly one reset per episode, or this episode was generated
            # from another plan entry's seed and its world_seed label lies.
            if len(seeds.seeds) != left - 1:
                raise RuntimeError(f"ep{job['idx']:04d}: {left - len(seeds.seeds)} "
                                   f"resets in one episode")
            msg = gym_env.conn.get_received_messages(gym_env.lidar_msg_in_topic)[0]
            ep.update(source=np.str_(job["kind"]),
                      policy_run=np.str_(run_dir.name if run_dir else ""),
                      policy_stage=np.int32(job["stage"]),
                      world_seed=np.int64(job["seed"]),
                      lidar_angle_start_deg=np.float32(msg.angleStartDeg),
                      lidar_angle_increment_deg=np.float32(msg.angleIncrementDeg),
                      lidar_max_range=np.float32(msg.maxRange),
                      grid_world_width=np.float32(tracker.world_width),
                      grid_world_height=np.float32(tracker.world_height),
                      grid_resolution=np.float32(tracker.resolution))
            dataset.save_episode(dataset.episode_path(root, job["idx"]), ep)
            print(f"[{tag}] ep{job['idx']:04d} {job['kind']:8s} "
                  f"{run_dir.name if run_dir else '':32s} "
                  f"steps={len(ep['pose']) - 1} pickups={int(ep['pickups'][-1])} "
                  f"explored={gym_env.task_tracker.get_explored_area_m2():.0f}m2 "
                  f"seed={job['seed']} {time.time() - t0:.0f}s")
    finally:
        wrapped.close()


def finalize(root: Path, plan: list[dict]) -> None:
    """Check every planned episode is there, record the geometry, seal."""
    missing = pending(root, plan)
    if missing:
        sys.exit(f"[probe-collect] {len(missing)} planned episodes missing "
                 f"(first: ep{missing[0]['idx']:04d}); re-run to resume")
    ep0 = dataset.load_episode(dataset.episode_path(root, 0))
    meta = dataset.load_json(root / "meta.json")
    meta["lidar_geometry"] = {
        "angle_start_deg": float(ep0["lidar_angle_start_deg"]),
        "angle_increment_deg": float(ep0["lidar_angle_increment_deg"]),
        "max_range": float(ep0["lidar_max_range"])}
    meta["grid"] = {"world_width": float(ep0["grid_world_width"]),
                    "world_height": float(ep0["grid_world_height"]),
                    "resolution": float(ep0["grid_resolution"])}
    dataset.save_json(root / "meta.json", meta)
    fp = dataset.seal(root)
    print(f"[probe-collect] {len(plan)} episodes complete, sealed {fp[:12]} -> {root}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="dataset root")
    ap.add_argument("--exp_dir", help="scheduler experiment dir (experiment.yaml)")
    ap.add_argument("--variation",
                    help="variation whose agent/task presets the dataset uses")
    ap.add_argument("--policy", action="append", default=[],
                    metavar="RUN_DIR:STAGE:N", help="repeatable")
    ap.add_argument("--explorer", type=int, default=0, metavar="N")
    ap.add_argument("--metaseed", type=int, default=777,
                    help="world-seed RNG; must differ from training metaseeds "
                         "(0..n) and from eval's 42")
    ap.add_argument("--workers", type=int, default=1,
                    help="parallel Unity instances (needs RATSIM_UNITY_BIN if > 1)")
    ap.add_argument("--platform", default=None, help="override jax.platform")
    ap.add_argument("--worker_jobs", default=None, help=argparse.SUPPRESS)
    add_unity_attach_args(ap)
    args = ap.parse_args()
    root = Path(args.out).resolve()

    if args.worker_jobs is not None:  # child: record the given episodes
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        plan = dataset.load_json(root / "meta.json")["plan"]
        idx = {int(i) for i in args.worker_jobs.split(",")}
        run_worker(root, [p for p in plan if p["idx"] in idx], args.unity_port,
                   args.platform, tag=f"probe-collect w{os.getpid()}")
        return

    if not (args.exp_dir and args.variation):
        ap.error("--exp_dir and --variation are required")
    exp_dir = Path(args.exp_dir).resolve()
    exp = load_experiment_def(exp_dir / "experiment.yaml")
    variation = find_variation(exp, args.variation)
    presets = {
        "agent": resolve_agent_preset(exp, variation),
        "task": resolve_task_preset(exp, variation),
        "world": resolve_stage_world(exp.stages[-1], variation, exp),
    }
    presets = json.loads(json.dumps(presets))  # as meta.json will hold them
    sources = [("policy", str(run_dir), stage, n) for run_dir, stage, n in
               (parse_policy(spec, exp_dir) for spec in args.policy)]
    if args.explorer:
        sources.append(("explorer", "", -1, args.explorer))
    plan = plan_episodes(sources, args.metaseed)
    # One collector per dataset: a resubmitted job must not record the same
    # missing episodes again while the first is still writing them.
    with dataset.lock(root / ".collect.lock"):
        collect(root, args, exp_dir, presets, plan)


def collect(root: Path, args, exp_dir: Path, presets: dict, plan: list[dict]):
    # Temp files of a killed worker. Safe to delete only here: the lock
    # guarantees no live writer owns them.
    for stale in dataset.episode_dir(root).glob(".*.tmp"):
        stale.unlink()
    meta_path = root / "meta.json"
    meta = {"exp_dir": str(exp_dir), "variation": args.variation,
            "presets": presets, "metaseed": args.metaseed, "plan": plan}
    if meta_path.exists():
        old = dataset.load_json(meta_path)
        if old.get("plan") != plan or old.get("presets") != presets:
            sys.exit(f"[probe-collect] {root} holds a different plan; resume "
                     f"with the same arguments or pick another --out")
        meta = {**old, **meta}
    todo = pending(root, plan)
    if todo:  # about to change the episodes: no longer sealed until finalize
        meta.pop("fingerprint", None)
        meta.pop("n_episodes", None)
    dataset.save_json(meta_path, meta)

    print(f"[probe-collect] presets {presets}; {len(plan)} planned, "
          f"{len(todo)} to record, {args.workers} workers")
    n = max(1, min(args.workers, len(todo)))
    if len(todo) and n == 1:
        port = resolve_unity_port(args, tag="probe-collect")
        run_worker(root, todo, port, args.platform, tag="probe-collect")
    elif todo:
        from ratsim.unity_launcher import allocate_unity_instances
        ports = [u.port for u in allocate_unity_instances(n_envs=n)]
        procs = []
        for k, port in enumerate(ports):
            jobs = ",".join(str(p["idx"]) for p in todo[k::n])
            cmd = [sys.executable, "-u", "-m", "probing.s1_collect.collect", "--out", str(root),
                   "--worker_jobs", jobs, "--unity_port", str(port)]
            if args.platform:
                cmd += ["--platform", args.platform]
            procs.append(subprocess.Popen(cmd))
        while any(p.poll() is None for p in procs):
            if any(p.poll() for p in procs):  # one failed: stop the rest
                for p in procs:
                    if p.poll() is None:
                        p.terminate()
            time.sleep(5)
        codes = [p.returncode for p in procs]
        if any(codes):
            sys.exit(f"[probe-collect] worker exit codes {codes}; "
                     f"re-run the same command to resume")
    finalize(root, plan)


if __name__ == "__main__":
    main()
