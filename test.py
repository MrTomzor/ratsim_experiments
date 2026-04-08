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
"""

import itertools
import json
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
from ratsim.task_tracker import TaskTracker
from ratsim.human_control_test import run_human_session

from ratsim_wildfire_gym_env.env import WildfireGymEnv


# -- Run definition loading --------------------------------------------------

def load_rundef(name_or_path: str) -> dict:
    """Load a run definition YAML by name or file path."""
    path = Path(name_or_path)
    if path.suffix in (".yaml", ".yml") and path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    rundef_dir = Path(__file__).parent / "rundefs"
    path = rundef_dir / f"{name_or_path}.yaml"
    if not path.exists():
        available = [f.stem for f in rundef_dir.glob("*.yaml")]
        raise FileNotFoundError(f"Run definition '{name_or_path}' not found. Available: {available}")
    with open(path) as f:
        return yaml.safe_load(f)


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
    cfg = blend_presets("task", [rundef.get("task_preset", "default")])
    cfg.update(rundef.get("task_overrides", {}))
    if stage is not None:
        cfg.update(stage.get("task_overrides", {}))
    return cfg


def resolve_agent_config(rundef: dict) -> dict:
    return blend_presets("agents", [rundef.get("agent_preset", "sphereagent_2d_lidar")])


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
):
    env = WildfireGymEnv(
        worldgen_config=world_config,
        agent_config=agent_config,
        sensor_config={},
        action_config={"control_mode": "velocity"},
        task_config=task_config,
        metaworldgen_config=None,
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
            )
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
            metrics = run_human_session(conn, wc, agent_config, task_config, seed=actual_seed, rtf=rtf)
            wall_time = time.time() - t0

            # Build a TaskTracker-compatible result from the human session metrics
            result = {
                "method": "human",
                "rundef": rundef_name,
                "stage_idx": stage_idx,
                "seed": seed,
                "episode_idx": ep_idx,
                "steps": metrics["steps"],
                "total_score": metrics["total_score"],
                "objects_found": metrics["objects_found"],
                "collisions": metrics["collisions"],
                "termination_reason": metrics["termination_reason"],
                "distance_traveled": 0.0,  # TODO: add distance tracking to human_control_test
                "wall_time_s": wall_time,
            }
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
):
    """Evaluate the ROS2 frontier exploration method.

    Launches the ROS2 frontier_exploration launch file as a subprocess.
    The bridge node runs the episode loop internally and prints one JSON
    line per episode to stdout.  We read those lines and record them.
    """
    seed_list = seeds if seeds is not None else list(range(1, 10001))
    seeds_str = ",".join(str(s) for s in seed_list)

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

    print(f"Launching: {' '.join(cmd[:4])} ...")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
    )

    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue

            # Try to parse as JSON (episode result from bridge node)
            try:
                episode_data = json.loads(line)
            except json.JSONDecodeError:
                # Not JSON — print as log output
                print(f"  [ros2] {line}")
                continue

            # Build result in the standard JSONL schema
            result = {
                "method": "frontier",
                "rundef": rundef_name,
                "stage_idx": stage_idx,
                "seed": episode_data.get("seed", 0),
                "episode_idx": episode_data.get("episode_idx", 0),
                "steps": episode_data.get("steps", 0),
                "total_score": episode_data.get("total_score", 0.0),
                "objects_found": episode_data.get("objects_found", 0),
                "collisions": episode_data.get("collisions", 0),
                "termination_reason": episode_data.get("termination_reason", "unknown"),
                "distance_traveled": 0.0,  # TODO: add distance tracking
                "wall_time_s": 0.0,  # TODO: add timing
            }
            append_result(results_file, result)
            print(
                f"  seed={result['seed']} ep={result['episode_idx']}: "
                f"objects={result['objects_found']}, steps={result['steps']}, "
                f"score={result['total_score']:.2f}"
            )

    except KeyboardInterrupt:
        print("\nInterrupting ROS2 process...")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


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
    rundef = load_rundef(rundef_name)
    agent_config = resolve_agent_config(rundef)

    # Output
    results_dir = Path(__file__).parent / "results" / run_name
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
    }
    with open(results_dir / "eval_config.json", "w") as f:
        json.dump(eval_meta, f, indent=2)

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

            if method_name == "human":
                eval_human(
                    rundef, rundef_name_clean, stage_idx,
                    world_config, agent_config, task_config,
                    eval_seeds, episodes_per_seed, results_file,
                    rtf=rtf,
                )
            elif method_name == "frontier":
                eval_frontier(
                    rundef, rundef_name_clean, stage_idx,
                    world_config, agent_config, task_config,
                    eval_seeds, episodes_per_seed, results_file,
                )
            else:
                eval_rl(
                    model, rundef, rundef_name_clean, method_name, stage_idx,
                    world_config, agent_config, task_config,
                    eval_seeds, episodes_per_seed, results_file,
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
