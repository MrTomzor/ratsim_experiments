"""Evaluate a single scheduler run's latest checkpoint.

Spawned by analyze_experiment.py --run-eval. Runs in the venv appropriate
for the run's method (sb3 venv for PPO/RecurrentPPO; dreamer not yet
supported — separate dispatch path needed).

Reads:
  <run_dir>/run_config.json     — variation, method, metaseed, exp_id
  <exp_dir>/experiment.yaml     — for variation → agent/task/world resolution
  <run_dir>/checkpoints/        — picks final.zip if present, else highest
                                   stage_K.zip with a stage_K.done sibling
                                   (so partially-finished runs still eval).

Writes:
  <run_dir>/eval_episodes.jsonl — overwritten; same schema as
                                   train_episodes.jsonl (env.py writes it).

CLI:
    python eval_one_run.py --run_dir <path> --exp_dir <path> --n_episodes N
                           [--eval_metaseed M]

World preset is resolved from the FINAL stage of the experiment def — for
curricula, that's the curriculum endpoint. Task and agent presets come from
the variation, so e.g. a `no_gps` variation evals with its no-gps agent.

The world seed for each eval episode is drawn from
`np.random.default_rng(eval_metaseed)` inside the env (see env.py:reset).
That means:
  - Every run in an experiment evals on the same sequence of worlds, so
    eval scores across seeds / variations / methods are directly comparable.
  - A human run with the same eval_metaseed sees the same world sequence,
    episode 1, 2, 3, ... in the same order.

Default eval_metaseed=42 is well outside the typical training metaseed range
(0..n_seeds), so eval worlds are held out from training.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from ratsim.config_blender import blend_presets
from ratsim.unity_launcher import allocate_unity_instances
from ratsim_wildfire_gym_env.env import WildfireGymEnv

from experiment_defs import (
    find_variation,
    load_experiment_def,
    resolve_agent_preset,
    resolve_stage_world,
    resolve_task_preset,
)


SB3_METHODS = {"ppo", "recurrent_ppo", "cnn_ppo", "cnn_recurrent_ppo"}
RECURRENT_METHODS = {"recurrent_ppo", "cnn_recurrent_ppo"}


def latest_sb3_checkpoint(checkpoint_dir: Path) -> tuple[Path, int] | None:
    """Pick the checkpoint to eval. Prefers final.zip (whole run done);
    falls back to the highest stage_K.zip whose stage_K.done sibling exists
    (so half-finished runs still eval their last fully-saved stage).

    Returns (path, stage_idx). stage_idx is informational — used to stamp
    the eval JSONL records so the analyzer knows which stage was eval'd.
    """
    if not checkpoint_dir.is_dir():
        return None
    done_stages = sorted(
        int(p.stem.split("_")[1])
        for p in checkpoint_dir.glob("stage_*.done")
    )
    final = checkpoint_dir / "final.zip"
    if final.exists() and done_stages:
        return final, done_stages[-1]
    # No final.zip — pick highest-stage .zip with a matching .done.
    candidates = []
    for stage_idx in done_stages:
        zp = checkpoint_dir / f"stage_{stage_idx}.zip"
        if zp.exists():
            candidates.append((stage_idx, zp))
    if not candidates:
        return None
    stage_idx, ckpt = candidates[-1]
    return ckpt, stage_idx


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--n_episodes", type=int, required=True)
    ap.add_argument("--eval_metaseed", type=int, default=42,
                    help="World-generation metaseed used for eval. The env "
                         "draws each episode's world seed from "
                         "np.random.default_rng(this value), so the same "
                         "metaseed yields the same sequence of worlds across "
                         "all runs (and across a human eval). Default 42, "
                         "held out from typical training metaseeds (0..N).")
    ap.add_argument("--deterministic", action="store_true",
                    help="Use deterministic action selection (mean of the "
                         "policy distribution). Default is STOCHASTIC — "
                         "matches training-time behaviour, so eval scores "
                         "are directly comparable to training scores. "
                         "Memoryless policies (plain PPO) often get stuck "
                         "in oscillation loops or against walls under "
                         "deterministic eval, because the same observation "
                         "always yields the same action.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    exp_dir = Path(args.exp_dir).resolve()

    run_cfg_path = run_dir / "run_config.json"
    if not run_cfg_path.exists():
        print(f"[eval] ERROR: {run_cfg_path} doesn't exist; "
              f"can't determine variation/method.")
        sys.exit(1)
    with open(run_cfg_path) as f:
        run_cfg = json.load(f)

    variation_name = run_cfg["variation"]
    method_name = run_cfg["method"]
    seed = int(run_cfg["metaseed"])
    exp_id = run_cfg["exp_id"]

    if method_name not in SB3_METHODS:
        print(f"[eval] ERROR: method '{method_name}' not supported by this "
              f"script. SB3 methods only ({sorted(SB3_METHODS)}). "
              f"Dreamer needs a separate eval path in the dreamer venv.")
        sys.exit(1)

    exp_yaml = exp_dir / "experiment.yaml"
    if not exp_yaml.exists():
        print(f"[eval] ERROR: {exp_yaml} doesn't exist.")
        sys.exit(1)
    exp = load_experiment_def(exp_yaml)
    variation = find_variation(exp, variation_name)

    agent_preset = resolve_agent_preset(exp, variation)
    task_preset = resolve_task_preset(exp, variation)
    last_stage = exp.stages[-1]
    world_preset = resolve_stage_world(last_stage, variation, exp)

    print(f"[eval] run_dir:      {run_dir}")
    print(f"[eval] method/var/seed: {method_name} / {variation_name} / {seed}")
    print(f"[eval] agent_preset: {agent_preset}")
    print(f"[eval] task_preset:  {task_preset}")
    print(f"[eval] world_preset: {world_preset}  (from final stage)")

    ckpt_info = latest_sb3_checkpoint(run_dir / "checkpoints")
    if ckpt_info is None:
        print(f"[eval] ERROR: no completed checkpoints in "
              f"{run_dir / 'checkpoints'} (no final.zip and no stage_K.zip "
              f"with matching .done).")
        sys.exit(1)
    ckpt_path, stage_idx = ckpt_info
    print(f"[eval] checkpoint:   {ckpt_path}  (stage_idx={stage_idx})")

    agent_config = blend_presets("agents", agent_preset)
    task_config = blend_presets("task", task_preset)
    world_config = blend_presets("world", world_preset)

    instances = allocate_unity_instances(n_envs=1)
    port = instances[0].port
    print(f"[eval] unity_port:   {port}")

    # Wipe existing eval JSONL so episode_idx restarts at 1 — env.py counts
    # existing lines for its idx offset, and we want clean 1..N indexing here.
    eval_jsonl = run_dir / "eval_episodes.jsonl"
    if eval_jsonl.exists():
        eval_jsonl.unlink()

    run_metadata = {
        "method": method_name,
        "exp_id": exp_id,
        "variation": variation_name,
        "seed": seed,
        "stage_idx": stage_idx,
        "env_idx": 0,
    }

    # NOTE: metaseed here is the EVAL metaseed (default 42), not the run's
    # training metaseed. This is what makes eval directly comparable across
    # runs/variations/methods — every eval run draws from the same RNG, so
    # episode i of run A and episode i of run B see the same world. (The
    # training metaseed is in run_metadata for traceability only.)
    env = WildfireGymEnv(
        worldgen_config=world_config,
        agent_config=agent_config,
        sensor_config={},
        action_config={"control_mode": "velocity"},
        task_config=task_config,
        metaworldgen_config={"world_generation_metaseed": args.eval_metaseed},
        episode_log_path=eval_jsonl,
        run_metadata=run_metadata,
        unity_port=port,
    )

    if method_name in ("ppo", "cnn_ppo"):
        from stable_baselines3 import PPO
        model = PPO.load(str(ckpt_path), env=env)
    else:
        from sb3_contrib import RecurrentPPO
        model = RecurrentPPO.load(str(ckpt_path), env=env)

    is_recurrent = method_name in RECURRENT_METHODS
    mode = "deterministic" if args.deterministic else "stochastic"
    print(f"[eval] running {args.n_episodes} episodes "
          f"(eval_metaseed={args.eval_metaseed}, {mode})")

    for ep in range(args.n_episodes):
        t0 = time.time()
        # World seed for this episode is drawn from the metaworldgen RNG
        # (seeded with eval_metaseed at env construction). No options seed
        # needed — the env overwrites options["seed"] anyway when
        # world_seed_generator is set.
        obs, _ = env.reset()
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        terminated = truncated = False
        while not (terminated or truncated):
            if is_recurrent:
                action, lstm_states = model.predict(
                    obs, state=lstm_states,
                    episode_start=episode_start,
                    deterministic=args.deterministic)
                episode_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, _, terminated, truncated, _ = env.step(action)
        elapsed = time.time() - t0
        score = env.task_tracker.get_total_score()
        pickups = env.task_tracker.get_num_reward_objs_picked_up()
        print(f"[eval] ep {ep + 1}/{args.n_episodes}: "
              f"score={score:.2f}  pickups={pickups}  {elapsed:.1f}s")

    env.close()
    print(f"[eval] Done. Wrote {eval_jsonl}")


if __name__ == "__main__":
    main()
