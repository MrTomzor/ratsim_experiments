"""Evaluate a trained DreamerV3 agent on a run definition.

Mirrors test.py conventions (seeds, episodes_per_seed, JSONL results) but
uses DreamerV3's agent + checkpoint API. Activate the dreamer venv first:
``source ~/ratvenv/dreamer_venv/bin/activate``.

Usage
-----
    python test_dreamerv3.py def=houses_volex_1m \\
        model=results/<run>/checkpoints/stage_6

    # Or point at an embodied rolling checkpoint directly:
    python test_dreamerv3.py def=houses_volex_1m \\
        model=results/<run>/dreamer_logdir/ckpt/latest

    # Restrict to a single stage's world config, pick seeds explicitly:
    python test_dreamerv3.py def=houses_volex_1m model=... \\
        stage=last eval_seeds=1,2,3 episodes_per_seed=2

    # GPU
    python test_dreamerv3.py def=... model=... method.jax.platform=cuda

The ``model`` path must point to a directory containing ``agent.pkl``
(i.e. a checkpoint snapshot). A path ending in ``ckpt/latest`` is
resolved via the pointer file written by embodied.
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

import elements
import embodied
import dreamerv3  # noqa: F401 — needed to register agent modules
from dreamerv3.main import wrap_env

from ratsim_wildfire_gym_env.env import WildfireGymEnv

from methods.dreamerv3.env_adapter import GymnasiumToEmbodied

# Reuse the exact same helpers the trainer uses so eval builds an env
# matching the trained one.
from train_dreamerv3 import (
    load_rundef,
    resolve_world_config,
    resolve_task_config,
    resolve_agent_config,
    build_config,
    parse_overrides,
    DEFAULT_SIZE,
)


# -- Episode result recording (mirrors test.py schema) -----------------------

def make_episode_result(
    rundef_name: str,
    stage_idx: int,
    seed: int,
    episode_idx: int,
    tracker,
    step_count: int,
    distance: float,
    wall_time: float,
) -> dict:
    return {
        "method": "dreamerv3",
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


def append_result(path: Path, result: dict):
    with open(path, "a") as f:
        f.write(json.dumps(result) + "\n")


# -- Checkpoint path resolution ----------------------------------------------

def resolve_ckpt_path(model_arg: str) -> Path:
    """Accept either a stage-snapshot dir, a rolling ckpt/<timestamp> dir,
    or a ckpt/latest pointer — return the concrete directory with agent.pkl."""
    p = Path(model_arg).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"model path does not exist: {p}")
    # If it's the rolling pointer file itself, follow it.
    if p.is_file() and p.name == "latest":
        name = p.read_text().strip()
        p = p.parent / name
    # If it's a ckpt dir and contains a 'latest' pointer, follow it.
    elif p.is_dir() and (p / "latest").is_file():
        name = (p / "latest").read_text().strip()
        p = p / name
    if not (p / "agent.pkl").exists():
        raise FileNotFoundError(f"no agent.pkl under {p}")
    return p


# -- Eval loop (single env, manual stepping) ---------------------------------

def eval_stage(
    wrapped_env,
    gym_env,
    agent,
    rundef_name: str,
    stage_idx: int,
    seeds: list[int] | None,
    episodes_per_seed: int,
    results_file: Path,
) -> None:
    # Seed injection: the embodied adapter calls gym_env.reset() without
    # options. We patch reset() to consume a pending seed set by the eval loop
    # before each episode so every episode starts from the requested seed.
    pending = {"seed": None}
    original_reset = gym_env.reset

    def seeded_reset(*, seed=None, options=None):
        if pending["seed"] is not None:
            options = dict(options or {})
            options["seed"] = pending["seed"]
            pending["seed"] = None
        return original_reset(seed=seed, options=options)

    gym_env.reset = seeded_reset

    # Build a zero action template for the reset step: matches wrapped act_space.
    def zero_action_template():
        act = {}
        for name, space in wrapped_env.act_space.items():
            if name == "reset":
                continue
            if space.discrete:
                act[name] = np.zeros(space.shape, dtype=np.int32)
            else:
                act[name] = np.zeros(space.shape, dtype=np.float32)
        return act

    carry = agent.init_policy(1)

    seed_list = seeds if seeds is not None else list(range(1, 10_001))

    for seed in seed_list:
        for ep_idx in range(episodes_per_seed):
            actual_seed = seed + ep_idx * 10_000
            pending["seed"] = actual_seed

            t0 = time.time()
            # Trigger a reset through the wrapper stack. The adapter sees
            # reset=True, calls gym_env.reset() (which picks up our seed),
            # and returns the initial embodied-format obs with is_first=True.
            reset_act = zero_action_template()
            reset_act["reset"] = True
            obs = wrapped_env.step(reset_act)

            # init_policy gives a fresh carry; is_first=True inside obs also
            # signals the model to reset its internal state for this episode.
            carry = agent.init_policy(1)

            step_count = 0
            while not bool(obs["is_last"]):
                # Drop log/* keys and add batch dim for the agent.
                obs_batched = {
                    k: np.asarray(v)[None]
                    for k, v in obs.items()
                    if not k.startswith("log/")
                }
                carry, act, _ = agent.policy(carry, obs_batched, mode="eval")
                action = {k: np.asarray(v)[0] for k, v in act.items()}
                action["reset"] = False
                obs = wrapped_env.step(action)
                step_count += 1

            wall_time = time.time() - t0

            tracker = gym_env.task_tracker
            distance = (
                gym_env.get_distance_traveled()
                if hasattr(gym_env, "get_distance_traveled")
                else 0.0
            )
            result = make_episode_result(
                rundef_name=rundef_name,
                stage_idx=stage_idx,
                seed=seed,
                episode_idx=ep_idx,
                tracker=tracker,
                step_count=step_count,
                distance=distance,
                wall_time=wall_time,
            )
            append_result(results_file, result)
            print(
                f"  seed={seed} ep={ep_idx}: objects={result['objects_found']}, "
                f"steps={result['steps']}, score={result['total_score']:.2f}, "
                f"term={result['termination_reason']}"
            )


# -- Main --------------------------------------------------------------------

def main():
    overrides = parse_overrides(sys.argv[1:])

    rundef_name = overrides.pop("def", None)
    if rundef_name is None:
        print("Usage: python test_dreamerv3.py def=<rundef> model=<ckpt_dir> [...]")
        sys.exit(1)
    rundef_stem = Path(rundef_name).stem

    model_arg = overrides.pop("model", None)
    if model_arg is None:
        print("Error: model=<path> is required (directory containing agent.pkl)")
        sys.exit(1)

    eval_seeds_raw = overrides.pop("eval_seeds", "1,2,3,4,5,6,7,8,9,10")
    episodes_per_seed = int(overrides.pop("episodes_per_seed", 1))
    run_name = overrides.pop(
        "name",
        f"eval_{rundef_stem}_dreamerv3_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    metaseed = int(overrides.pop("metaseed", 0))
    size = overrides.pop("size", DEFAULT_SIZE)
    stage_arg = overrides.pop("stage", "all")  # "all", "last", or an int index

    # method.* passthrough for the dreamer config
    method_overrides = {}
    for key in list(overrides.keys()):
        if key.startswith("method."):
            method_overrides[key.removeprefix("method.")] = overrides.pop(key)

    if eval_seeds_raw == "inf":
        eval_seeds = None
    elif isinstance(eval_seeds_raw, str):
        eval_seeds = [int(s.strip()) for s in eval_seeds_raw.split(",")]
    elif isinstance(eval_seeds_raw, list):
        eval_seeds = [int(s) for s in eval_seeds_raw]
    else:
        eval_seeds = [int(eval_seeds_raw)]

    rundef = load_rundef(rundef_name)
    stages = rundef["stages"]

    if stage_arg == "all":
        stage_indices = list(range(len(stages)))
    elif stage_arg == "last":
        stage_indices = [len(stages) - 1]
    else:
        stage_indices = [int(stage_arg)]

    ckpt_path = resolve_ckpt_path(model_arg)
    print(f"Checkpoint: {ckpt_path}")

    results_dir = Path(__file__).parent / "results" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    results_file = results_dir / "episodes.jsonl"

    # Use a scratch logdir inside the eval results dir for the dreamer config;
    # the trainer's logdir is irrelevant here but the Agent wants a writable
    # one for any incidental output.
    scratch_logdir = results_dir / "dreamer_logdir"
    scratch_logdir.mkdir(exist_ok=True)

    with open(results_dir / "eval_config.json", "w") as f:
        json.dump(
            {
                "rundef": rundef_stem,
                "method": "dreamerv3",
                "model": str(ckpt_path),
                "eval_seeds": "inf" if eval_seeds is None else eval_seeds,
                "episodes_per_seed": episodes_per_seed,
                "stage_indices": stage_indices,
                "size": size,
                "method_overrides": method_overrides,
                "run_name": run_name,
            },
            f,
            indent=2,
        )

    print(f"Results: {results_file}")
    print(
        f"Seeds: {'inf' if eval_seeds is None else eval_seeds}, "
        f"episodes/seed: {episodes_per_seed}, stages: {stage_indices}"
    )

    # Build agent once — the obs/act spaces are the same across stages for
    # a given rundef (only world config changes). We pick the first requested
    # stage to construct the env for space inference.
    first_stage = stages[stage_indices[0]]

    config = build_config(method_overrides, scratch_logdir, total_steps=1, size=size)
    # Eval runs single-env, no train ratio relevance.
    config = config.update({"run.envs": 1, "run.eval_envs": 0})

    gym_env = WildfireGymEnv(
        worldgen_config=resolve_world_config(first_stage),
        agent_config=resolve_agent_config(rundef),
        sensor_config={},
        action_config={"control_mode": "velocity"},
        task_config=resolve_task_config(rundef, first_stage),
        metaworldgen_config=None,  # deterministic per-episode seeding instead
    )
    adapter = GymnasiumToEmbodied(gym_env, obs_key="vector", act_key="action")
    wrapped = wrap_env(adapter, config)

    from dreamerv3.agent import Agent

    notlog = lambda k: not k.startswith("log/")
    obs_space = {k: v for k, v in wrapped.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in wrapped.act_space.items() if k != "reset"}

    agent = Agent(
        obs_space,
        act_space,
        elements.Config(
            **config.agent,
            logdir=str(scratch_logdir),
            seed=config.seed,
            jax=config.jax,
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            replay_context=config.replay_context,
            report_length=config.report_length,
            replica=config.replica,
            replicas=config.replicas,
        ),
    )

    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(str(ckpt_path), keys=["agent"])
    print("Agent loaded.")

    try:
        for stage_idx in stage_indices:
            stage = stages[stage_idx]
            print(f"\n{'='*60}")
            print(
                f"Stage {stage_idx + 1}/{len(stages)}: "
                f"{stage.get('world_presets', ['?'])}"
            )
            print(f"{'='*60}")

            # Swap in this stage's world/task config without tearing down the
            # TCP connection. reset() reads these each episode.
            gym_env.worldgen_config = resolve_world_config(stage)
            gym_env.task_config = resolve_task_config(rundef, stage)
            # TaskTracker reads task_config on reset via env.reset(); re-init
            # defensively in case the env caches anything derived from it.
            from ratsim.task_tracker import TaskTracker
            gym_env.task_tracker = TaskTracker(gym_env.task_config)

            eval_stage(
                wrapped_env=wrapped,
                gym_env=gym_env,
                agent=agent,
                rundef_name=rundef_stem,
                stage_idx=stage_idx,
                seeds=eval_seeds,
                episodes_per_seed=episodes_per_seed,
                results_file=results_file,
            )
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        wrapped.close()

    print(f"\n{'='*60}")
    print("Evaluation complete.")
    print(f"Results: {results_file}")

    if results_file.exists():
        try:
            import pandas as pd

            df = pd.read_json(results_file, lines=True)
            if len(df):
                print(f"\nSummary ({len(df)} episodes):")
                print(
                    f"  objects_found: {df['objects_found'].mean():.1f} "
                    f"± {df['objects_found'].std():.1f}"
                )
                print(
                    f"  total_score:   {df['total_score'].mean():.2f} "
                    f"± {df['total_score'].std():.2f}"
                )
                print(
                    f"  steps:         {df['steps'].mean():.0f} "
                    f"± {df['steps'].std():.0f}"
                )
        except ImportError:
            pass


if __name__ == "__main__":
    main()
