"""
Evaluate a method on a run definition.

Usage:
    # Test a trained RL model
    python test.py def=default_forest_foraging model=results/my_run/checkpoints/final.zip

    # Test with recurrent PPO
    python test.py def=default_forest_foraging model=results/my_run/checkpoints/final.zip method=recurrent_ppo

    # Human evaluation
    python test.py def=default_forest_foraging method=human

    # Override eval seeds or number of episodes
    python test.py def=default_forest_foraging model=results/my_run/checkpoints/final.zip eval_seeds=42,123,456
    python test.py def=default_forest_foraging method=human episodes_per_seed=3

    # Experiment defs (defs/*.yaml) work too: agent/task come from the def (or
    # variation=<name>), the world is the FINAL stage's world preset, and
    # difficulty=D pins an adaptive-difficulty def at one rung.
    python test.py def=memory_orthomaze method=frontier eval_seeds=1834701,99123 difficulty=0.5

    # Trajectories for plot_trajectories.py: record_trajectories=1 writes one
    # npz per episode under results/<run>/trajectories/ and stamps
    # world_seed + trajectory_file into every JSONL line.
    python test.py def=memory_orthomaze method=human rtf=1.0 record_trajectories=1

    # Anything prefixed frontier. is forwarded verbatim to the ROS2 launch
    # file (see ratsim_ros2/launch/frontier_exploration.launch.py for keys).
    python test.py def=memory_orthomaze method=frontier frontier.grid_resolution=0.5
"""

import itertools
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

from feature_extractors import LidarCnnExtractor  # noqa: F401 — needed for model deserialization
from train import METHODS

from ratsim.roslike_unity_connector.connector import RoslikeUnityConnector
from ratsim.roslike_unity_connector.message_definitions import (
    BoolMessage,
    StringMessage,
)
from ratsim.config_blender import blend_presets, to_entries_json
from ratsim.config_blender.blender import flatten_config
from ratsim.task_tracker import TaskTracker
from ratsim.task_tracker.trajectory_record import save_trajectory
from ratsim.human_control_test import run_human_session

from experiment_defs import (
    find_variation,
    load_experiment_def,
    resolve_agent_preset,
    resolve_difficulty_overrides,
    resolve_stage_world,
    resolve_task_preset,
)

from ratsim_wildfire_gym_env.env import WildfireGymEnv


# -- Run definition loading --------------------------------------------------

def load_rundef(name_or_path: str, variation: str | None = None,
                difficulty: float | None = None) -> dict:
    """Load a run definition by name or path.

    Looks in rundefs/ first (legacy schema: agent_preset/task_preset/stages
    with world_presets lists). Falls back to defs/ (experiment defs, the
    schema train.py and the scheduler use) and converts to the same shape:
    one stage carrying the def's FINAL-stage world preset, so an eval here
    lands on the same world eval_one_run.py uses. `variation` picks the
    def's agent/task/world overrides; `difficulty` bakes an adaptive def's
    ranges at that rung into world_overrides.
    """
    path = Path(name_or_path)
    if path.suffix in (".yaml", ".yml") and path.exists():
        with open(path) as f:
            raw = yaml.safe_load(f)
        if "stages" in raw and "methods" not in raw:
            return raw
        return _rundef_from_experiment_def(path, variation, difficulty)
    rundef_dir = Path(__file__).parent / "rundefs"
    path = rundef_dir / f"{name_or_path}.yaml"
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    def_path = Path(__file__).parent / "defs" / f"{name_or_path}.yaml"
    if def_path.exists():
        return _rundef_from_experiment_def(def_path, variation, difficulty)
    available = sorted(f.stem for f in rundef_dir.glob("*.yaml"))
    available_defs = sorted(f.stem for f in def_path.parent.glob("*.yaml"))
    raise FileNotFoundError(
        f"Run definition '{name_or_path}' not found. rundefs/: {available}; "
        f"defs/: {available_defs}")


def _rundef_from_experiment_def(path: Path, variation: str | None,
                                difficulty: float | None) -> dict:
    exp = load_experiment_def(path)
    var = find_variation(exp, variation or exp.variations[0].name)
    world_overrides = {}
    if difficulty is not None:
        world_overrides = resolve_difficulty_overrides(exp, difficulty)
    print(f"Experiment def {exp.exp_id}: variation={var.name}, "
          f"world={resolve_stage_world(exp.stages[-1], var, exp)} (final stage)"
          + (f", difficulty={difficulty}" if difficulty is not None else ""))
    return {
        "agent_preset": resolve_agent_preset(exp, var),
        "task_preset": resolve_task_preset(exp, var),
        "stages": [{
            "world_presets": resolve_stage_world(exp.stages[-1], var, exp),
            "world_overrides": world_overrides,
        }],
    }


def _as_preset_list(v) -> list[str]:
    if v is None:
        return ["default"]
    if isinstance(v, str):
        return [v]
    return list(v)


# -- Config resolution -------------------------------------------------------

def resolve_world_config(stage: dict) -> dict:
    presets = stage.get("world_presets", ["default"])
    cfg = blend_presets("world", presets)
    cfg.update(stage.get("world_overrides", {}))
    return cfg


def resolve_task_config(rundef: dict, stage: dict | None = None) -> dict:
    """Load and optionally override task config.

    Rundef-level task_overrides are applied first, then stage-level task_overrides
    (if a stage dict is provided) take precedence.
    """
    cfg = blend_presets("task", _as_preset_list(rundef.get("task_preset", "default")))
    cfg.update(rundef.get("task_overrides", {}))
    if stage is not None:
        cfg.update(stage.get("task_overrides", {}))
    return cfg


def resolve_agent_config(rundef: dict) -> dict:
    return blend_presets("agents", _as_preset_list(rundef.get("agent_preset", "sphereagent_2d_lidar")))


# -- Episode result recording ------------------------------------------------

def make_episode_result(
    method: str,
    rundef_name: str,
    stage_idx: int,
    seed: int,
    episode_idx: int,
    tracker: TaskTracker,
    step_count: int,
    distance: float,
    wall_time: float,
    extra: dict | None = None,
) -> dict:
    result = {
        "method": method,
        "rundef": rundef_name,
        "stage_idx": stage_idx,
        "seed": seed,
        "episode_idx": episode_idx,
        "steps": step_count,
        "total_score": tracker.get_total_score(),
        "objects_found": tracker.get_num_reward_objs_picked_up(),
        "collisions": tracker.get_collision_count(),
        "termination_reason": tracker.get_termination_reason(),
        "distance_traveled": distance,
        "wall_time_s": wall_time,
    }
    if extra:
        result.update(extra)
    return result


def append_result(path: Path, result: dict):
    with open(path, "a") as f:
        f.write(json.dumps(result) + "\n")


def save_episode_trajectory(results_file: Path, traj: dict, result: dict,
                            world_config: dict) -> str:
    """Write one episode's pose buffer as
    <results_dir>/trajectories/<method>_seed<world_seed>_ep<idx>.npz and
    return the path relative to the results dir (goes into the JSONL as
    `trajectory_file`). Same npz format env.py writes for eval_one_run.py."""
    flat_world = flatten_config(world_config)
    meta = {**result,
            "world_width": flat_world.get("world_bounds/width"),
            "world_height": flat_world.get("world_bounds/height"),
            "frame": "ros: x forward, y left, z up; yaw ccw rad"}
    name = f"{result['method']}_seed{result['world_seed']}_ep{result['episode_idx']}.npz"
    out = save_trajectory(results_file.parent / "trajectories" / name, traj, meta)
    return str(out.relative_to(results_file.parent))


# -- RL evaluation ------------------------------------------------------------

def eval_rl(
    model,
    rundef: dict,
    rundef_name: str,
    method_name: str,
    stage_idx: int,
    world_config: dict,
    agent_config: dict,
    task_config: dict,
    seeds: list[int] | None,
    episodes_per_seed: int,
    results_file: Path,
    record_trajectories: bool = False,
):
    env = WildfireGymEnv(
        worldgen_config=world_config,
        agent_config=agent_config,
        sensor_config={},
        action_config={"control_mode": "velocity"},
        task_config=task_config,
        metaworldgen_config=None,
        record_trajectories=record_trajectories,
    )

    seed_iter = seeds if seeds is not None else itertools.count(1)

    for seed in seed_iter:
        for ep_idx in range(episodes_per_seed):
            actual_seed = seed + ep_idx * 10000

            t0 = time.time()
            obs, _ = env.reset(options={"seed": actual_seed})

            # For recurrent policies, need to track lstm states
            lstm_states = None
            episode_start = np.ones((1,), dtype=bool)

            terminated = False
            truncated = False

            while not terminated and not truncated:
                if hasattr(model, 'predict') and hasattr(model.policy, 'lstm'):
                    # RecurrentPPO
                    action, lstm_states = model.predict(
                        obs, state=lstm_states, episode_start=episode_start, deterministic=True
                    )
                    episode_start = np.zeros((1,), dtype=bool)
                else:
                    action, _ = model.predict(obs, deterministic=True)

                obs, reward, terminated, truncated, info = env.step(action)

            wall_time = time.time() - t0

            result = make_episode_result(
                method=method_name,
                rundef_name=rundef_name,
                stage_idx=stage_idx,
                seed=seed,
                episode_idx=ep_idx,
                tracker=env.task_tracker,
                step_count=env.step_count,
                distance=env.get_distance_traveled(),
                wall_time=wall_time,
                extra={"world_seed": env.current_world_seed},
            )
            if record_trajectories:
                traj = env.task_tracker.get_trajectory()
                if traj is not None:
                    result["trajectory_file"] = save_episode_trajectory(
                        results_file, traj, result, world_config)
            append_result(results_file, result)
            print(f"  seed={seed} ep={ep_idx}: objects={result['objects_found']}, "
                  f"steps={result['steps']}, score={result['total_score']:.2f}")

    env.close()


# -- Human evaluation ---------------------------------------------------------

def eval_human(
    rundef: dict,
    rundef_name: str,
    stage_idx: int,
    world_config: dict,
    agent_config: dict,
    task_config: dict,
    seeds: list[int] | None,
    episodes_per_seed: int,
    results_file: Path,
    rtf: float = 1.0,
    record_trajectories: bool = False,
    method_params: dict | None = None,
):
    conn = RoslikeUnityConnector(verbose=False)
    conn.connect()

    # Select scene and send agent config
    conn.publish(StringMessage(data="Wildfire"), "/sim_control/scene_select")
    conn.send_messages_and_step(enable_physics_step=False)
    conn.read_messages_from_unity()

    conn.publish(StringMessage(data=to_entries_json(agent_config)), "/sim_control/agent_config")
    conn.send_messages_and_step(enable_physics_step=False)
    conn.read_messages_from_unity()

    seed_iter = seeds if seeds is not None else itertools.count(1)

    for seed in seed_iter:
        for ep_idx in range(episodes_per_seed):
            wc = dict(world_config)
            actual_seed = seed + ep_idx * 10000

            print(f"\n{'='*60}")
            print(f"Human eval: seed={seed}, episode={ep_idx}")
            print(f"Press Enter in this terminal when ready to start...")
            input()

            t0 = time.time()
            metrics = run_human_session(conn, wc, agent_config, task_config, seed=actual_seed, rtf=rtf,
                                        record_trajectory=record_trajectories)
            wall_time = time.time() - t0

            # Build a TaskTracker-compatible result from the human session metrics
            result = {
                "method": "human",
                "rundef": rundef_name,
                "stage_idx": stage_idx,
                "seed": seed,
                "episode_idx": ep_idx,
                "world_seed": metrics.get("world_seed", actual_seed),
                "steps": metrics["steps"],
                "total_score": metrics["total_score"],
                "objects_found": metrics["objects_found"],
                "collisions": metrics["collisions"],
                "termination_reason": metrics["termination_reason"],
                "distance_traveled": 0.0,  # TODO: add distance tracking to human_control_test
                "wall_time_s": wall_time,
                "method_params": dict(method_params or {}),
            }
            if record_trajectories and metrics.get("trajectory") is not None:
                result["trajectory_file"] = save_episode_trajectory(
                    results_file, metrics["trajectory"], result, wc)
            append_result(results_file, result)
            print(f"  seed={seed} ep={ep_idx}: objects={result['objects_found']}, "
                  f"steps={result['steps']}, score={result['total_score']:.2f}")


# -- Frontier exploration evaluation -------------------------------------------

def eval_frontier(
    rundef: dict,
    rundef_name: str,
    stage_idx: int,
    world_config: dict,
    agent_config: dict,
    task_config: dict,
    seeds: list[int] | None,
    episodes_per_seed: int,
    results_file: Path,
    record_trajectories: bool = False,
    method_params: dict | None = None,
):
    """Evaluate the ROS2 frontier exploration method.

    Launches the ROS2 frontier_exploration launch file as a subprocess.
    The bridge node runs the episode loop internally and prints one JSON
    line per episode to stdout.  We read those lines and record them.

    `method_params` (from `frontier.<key>=<value>` CLI overrides) are
    forwarded verbatim as `<key>:=<value>` launch arguments — any key the
    launch file declares works; an unknown one fails loudly at launch.
    With `record_trajectories` the bridge writes one npz per episode into
    <results_dir>/trajectories/ and names it in its JSON result line.
    """
    seed_list = seeds if seeds is not None else list(range(1, 10001))
    seeds_str = ",".join(str(s) for s in seed_list)
    method_params = dict(method_params or {})

    world_json = json.dumps(world_config)
    agent_json = json.dumps(agent_config)
    task_json = json.dumps(task_config)

    cmd = [
        "ros2", "launch", "ratsim_ros2", "frontier_exploration.launch.py",
        f"world_config_json:={world_json}",
        f"agent_config_json:={agent_json}",
        f"task_config_json:={task_json}",
        f"seeds:={seeds_str}",
        f"episodes_per_seed:={episodes_per_seed}",
    ]
    traj_dir = results_file.parent / "trajectories"
    if record_trajectories:
        traj_dir.mkdir(parents=True, exist_ok=True)
        cmd.append(f"trajectory_dir:={traj_dir}")
    for k, v in method_params.items():
        cmd.append(f"{k}:={v}")

    print(f"Launching: {' '.join(cmd[:4])} ...")
    # stderr merged into stdout: an undrained stderr pipe stalls the launch
    # (and every node behind it) once the buffer fills. New session so the
    # whole launch process group can be signalled — `ros2 launch` never exits
    # by itself after "All episodes complete", it idles.
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # line-buffered
        start_new_session=True,
    )
    expected = None if seeds is None else len(seed_list) * episodes_per_seed
    n_results = 0
    # ros2 launch prefixes node output: "[unity_ros2_bridge-1] {...}"
    prefix_re = re.compile(r"^\[[^\]]+\]\s*")

    try:
        for raw in proc.stdout:
            line = prefix_re.sub("", raw.strip())
            if not line:
                continue

            # Try to parse as JSON (episode result from bridge node)
            episode_data = None
            if line.startswith("{"):
                try:
                    episode_data = json.loads(line)
                except json.JSONDecodeError:
                    episode_data = None
            if episode_data is None:
                print(f"  [ros2] {line}")
                if "All episodes complete" in line:
                    print("  bridge reports all episodes complete — shutting the launch down")
                    break
                continue

            # Build result in the standard JSONL schema
            result = {
                "method": "frontier",
                "rundef": rundef_name,
                "stage_idx": stage_idx,
                "seed": episode_data.get("seed", 0),
                "episode_idx": episode_data.get("episode_idx", 0),
                "world_seed": episode_data.get("world_seed"),
                "steps": episode_data.get("steps", 0),
                "total_score": episode_data.get("total_score", 0.0),
                "objects_found": episode_data.get("objects_found", 0),
                "collisions": episode_data.get("collisions", 0),
                "termination_reason": episode_data.get("termination_reason", "unknown"),
                "distance_traveled": 0.0,  # TODO: add distance tracking
                "wall_time_s": 0.0,  # TODO: add timing
                "method_params": method_params,
            }
            if episode_data.get("trajectory_file"):
                tf = Path(episode_data["trajectory_file"])
                try:
                    result["trajectory_file"] = str(tf.relative_to(results_file.parent))
                except ValueError:
                    result["trajectory_file"] = str(tf)
            append_result(results_file, result)
            n_results += 1
            print(
                f"  seed={result['seed']} ep={result['episode_idx']}: "
                f"objects={result['objects_found']}, steps={result['steps']}, "
                f"score={result['total_score']:.2f}"
                + (f"  [{n_results}/{expected}]" if expected else "")
            )
            if expected is not None and n_results >= expected:
                print("  all requested episodes recorded — shutting the launch down")
                break

    except KeyboardInterrupt:
        print("\nInterrupting ROS2 process...")
    finally:
        _stop_launch(proc)


def _stop_launch(proc: subprocess.Popen, grace: float = 20.0):
    """SIGINT the launch's process group (ros2 launch shuts its nodes down
    cleanly on Ctrl-C), escalate to SIGTERM / SIGKILL if it lingers, and drain
    stdout so the pipe never blocks the exit."""
    import signal
    if proc.poll() is not None:
        return
    for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, 5.0), (signal.SIGKILL, 5.0)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.communicate(timeout=wait)
            return
        except subprocess.TimeoutExpired:
            continue


# -- CLI -----------------------------------------------------------------------

def parse_overrides(override_list: list[str]) -> dict:
    result = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key=value")
        key, value = item.split("=", 1)
        try:
            value = yaml.safe_load(value)
        except yaml.YAMLError:
            pass
        result[key] = value
    return result


def main():
    overrides = parse_overrides(sys.argv[1:])

    rundef_name = overrides.pop("def", None)
    if rundef_name is None:
        print("Usage: python test.py def=<rundef_name> [model=<path>] [method=<name>] [eval_seeds=1,2,3]")
        sys.exit(1)
    rundef_name_clean = Path(rundef_name).stem

    method_name = overrides.pop("method", "ppo")
    model_path = overrides.pop("model", None)
    eval_seeds_raw = overrides.pop("eval_seeds", "1,2,3,4,5,6,7,8,9,10")
    episodes_per_seed = int(overrides.pop("episodes_per_seed", 1))
    run_name = overrides.pop("name", f"eval_{rundef_name_clean}_{method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    rtf = float(overrides.pop("rtf", 1.0))
    results_dir_override = overrides.pop("results_dir", None)
    record_trajectories = bool(overrides.pop("record_trajectories", False))
    variation = overrides.pop("variation", None)
    difficulty = overrides.pop("difficulty", None)
    if difficulty is not None:
        difficulty = float(difficulty)
        if not 0.0 <= difficulty <= 1.0:
            print(f"Error: difficulty must be in [0, 1], got {difficulty}")
            sys.exit(1)
    # Method-specific pass-through: frontier.<key>=<value> → launch arg key:=value
    method_params = {k.split(".", 1)[1]: v for k, v in overrides.items()
                     if k.startswith(f"{method_name}.")}
    for k in [k for k in overrides if "." in k]:
        overrides.pop(k)
    if overrides:
        print(f"Warning: unrecognised arguments ignored: {sorted(overrides)}")

    # Parse seeds
    if eval_seeds_raw == "inf":
        eval_seeds = None  # signals infinite mode
    elif isinstance(eval_seeds_raw, str):
        eval_seeds = [int(s.strip()) for s in eval_seeds_raw.split(",")]
    elif isinstance(eval_seeds_raw, list):
        eval_seeds = [int(s) for s in eval_seeds_raw]
    else:
        eval_seeds = [int(eval_seeds_raw)]

    # Load run definition
    rundef = load_rundef(rundef_name, variation=variation, difficulty=difficulty)
    agent_config = resolve_agent_config(rundef)

    # Output
    # results_dir=<path> puts the output exactly there (record_trajectories.py
    # uses <exp_dir>/external/<method>/); default is results/<run_name>/.
    results_dir = (Path(results_dir_override).resolve() if results_dir_override
                   else Path(__file__).parent / "results" / run_name)
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "episodes.jsonl"

    # Save eval config
    eval_meta = {
        "rundef": rundef_name_clean,
        "method": method_name,
        "model": model_path,
        "eval_seeds": "inf" if eval_seeds is None else eval_seeds,
        "episodes_per_seed": episodes_per_seed,
        "run_name": run_name,
        "variation": variation,
        "difficulty": difficulty,
        "record_trajectories": record_trajectories,
        "method_params": method_params,
        "agent_preset": rundef.get("agent_preset"),
        "task_preset": rundef.get("task_preset"),
        "stages": rundef.get("stages"),
    }
    with open(results_dir / "eval_config.json", "w") as f:
        json.dump(eval_meta, f, indent=2, default=str)

    print(f"Evaluating: method={method_name}, rundef={rundef_name}")
    print(f"Seeds: {'infinite (1,2,3,...)' if eval_seeds is None else eval_seeds}, episodes/seed: {episodes_per_seed}")
    print(f"Results: {results_file}")

    # Load model if RL (not needed for human or frontier)
    model = None
    if method_name not in ("human", "frontier"):
        if model_path is None:
            print("Error: model=<path> required for non-human/non-frontier methods")
            sys.exit(1)
        if method_name not in METHODS:
            print(f"Error: unknown method '{method_name}'. Available: {list(METHODS.keys())}")
            sys.exit(1)
        model = METHODS[method_name]["sb3_class"].load(model_path)
        print(f"Loaded model: {model_path}")

    # Evaluate each stage
    try:
        for stage_idx, stage in enumerate(rundef["stages"]):
            world_config = resolve_world_config(stage)
            task_config = resolve_task_config(rundef, stage)

            print(f"\n{'='*60}")
            print(f"Stage {stage_idx + 1}/{len(rundef['stages'])}: {stage.get('world_presets', ['?'])}")
            print(f"{'='*60}")

            # Resolved world config per stage, so the plot side can regenerate
            # the exact world from (config, world_seed) later.
            with open(results_dir / f"eval_world_config_stage{stage_idx}.json", "w") as f:
                json.dump({"world_config": world_config, "task_config": task_config,
                           "agent_config": agent_config}, f, indent=2, default=str)

            if method_name == "human":
                eval_human(
                    rundef, rundef_name_clean, stage_idx,
                    world_config, agent_config, task_config,
                    eval_seeds, episodes_per_seed, results_file,
                    rtf=rtf, record_trajectories=record_trajectories,
                    method_params=method_params,
                )
            elif method_name == "frontier":
                eval_frontier(
                    rundef, rundef_name_clean, stage_idx,
                    world_config, agent_config, task_config,
                    eval_seeds, episodes_per_seed, results_file,
                    record_trajectories=record_trajectories,
                    method_params=method_params,
                )
            else:
                eval_rl(
                    model, rundef, rundef_name_clean, method_name, stage_idx,
                    world_config, agent_config, task_config,
                    eval_seeds, episodes_per_seed, results_file,
                    record_trajectories=record_trajectories,
                )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    # Print summary
    print(f"\n{'='*60}")
    print("Evaluation complete.")
    print(f"Results written to: {results_file}")

    if results_file.exists():
        import pandas as pd
        df = pd.read_json(results_file, lines=True)
        print(f"\nSummary ({len(df)} episodes):")
        print(f"  objects_found: {df['objects_found'].mean():.1f} ± {df['objects_found'].std():.1f}")
        print(f"  total_score:   {df['total_score'].mean():.2f} ± {df['total_score'].std():.2f}")
        print(f"  steps:         {df['steps'].mean():.0f} ± {df['steps'].std():.0f}")


if __name__ == "__main__":
    main()
