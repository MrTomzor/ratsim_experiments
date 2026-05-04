"""Evaluate a single dreamer scheduler run's latest checkpoint.

Counterpart to eval_one_run.py for SB3 — same CLI, same JSONL output,
same eval-metaseed reproducibility model, but loads the agent through
embodied + dreamerv3.Agent and must run in the dreamer venv.

Spawned by analyze_experiment.py --run-eval N. Picks the dreamer venv via
$DREAMER_PYTHON_PATH (see scheduler/config.py:DEFAULT_PYTHON_ENV).

Reads:
  <run_dir>/run_config.json     — variation, method=dreamer, metaseed, exp_id,
                                   size, method_overrides
  <exp_dir>/experiment.yaml     — for variation → agent/task/world resolution
  <run_dir>/checkpoints/        — final/ dir if present, else highest
                                   stage_K/ dir with matching stage_K.done.
                                   (Dreamer checkpoints are directories
                                   containing agent.pkl, not zip files.)

Writes:
  <run_dir>/eval_episodes.jsonl — overwritten; same schema env.py uses for
                                   train_episodes.jsonl.
                                   With --ablate-memory the path is
                                   eval_episodes_ablated.jsonl instead, so
                                   the baseline eval persists alongside.

Reproducibility: every episode's world seed is drawn from
np.random.default_rng(eval_metaseed) inside env.py:reset, so the same
eval_metaseed produces the same world sequence across all runs (and
across a human eval).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import elements
import embodied  # noqa: F401 — dreamerv3.main pulls runtime pieces from this
import dreamerv3  # noqa: F401 — registers agent modules
from dreamerv3.main import wrap_env

from ratsim.config_blender import blend_presets
from ratsim.unity_launcher import allocate_unity_instances
from ratsim_wildfire_gym_env.env import WildfireGymEnv

from methods.dreamerv3.env_adapter import GymnasiumToEmbodied

from train_dreamerv3 import build_config, DEFAULT_SIZE
from experiment_defs import (
    find_variation,
    load_experiment_def,
    resolve_agent_preset,
    resolve_stage_world,
    resolve_task_preset,
)


def latest_dreamer_checkpoint(checkpoint_dir: Path) -> tuple[Path, int] | None:
    """Pick the dreamer checkpoint to eval. Prefer `final/` (whole run done);
    else the highest `stage_K/` whose `stage_K.done` sibling exists. Each
    candidate must contain `agent.pkl`.

    Returns (path, stage_idx). For `final`, stage_idx = last completed stage —
    purely informational, used to stamp the eval JSONL records."""
    if not checkpoint_dir.is_dir():
        return None
    done_stages = sorted(
        int(p.stem.split("_")[1])
        for p in checkpoint_dir.glob("stage_*.done")
    )
    final = checkpoint_dir / "final"
    if final.is_dir() and (final / "agent.pkl").exists():
        return final, (done_stages[-1] if done_stages else 0)
    for stage_idx in reversed(done_stages):
        d = checkpoint_dir / f"stage_{stage_idx}"
        if d.is_dir() and (d / "agent.pkl").exists():
            return d, stage_idx
    return None


def _zero_action(wrapped) -> dict:
    """Build a zero-valued action dict matching wrapped.act_space, minus
    `reset`. Caller sets `reset=True/False` separately."""
    act: dict = {}
    for name, space in wrapped.act_space.items():
        if name == "reset":
            continue
        if space.discrete:
            act[name] = np.zeros(space.shape, dtype=np.int32)
        else:
            act[name] = np.zeros(space.shape, dtype=np.float32)
    return act


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--n_episodes", type=int, required=True)
    ap.add_argument("--eval_metaseed", type=int, default=42,
                    help="World-generation metaseed (default 42). Same value "
                         "across runs → identical world sequence; matches "
                         "eval_one_run.py for SB3.")
    ap.add_argument("--deterministic", action="store_true",
                    help="Accepted for CLI parity with eval_one_run.py, but "
                         "currently a no-op: this dreamerv3 build's "
                         "Agent.policy() always samples from the action "
                         "distribution — there's no separate argmax path. "
                         "Stochastic eval also matches training behaviour, "
                         "so this isn't a wart we need to fix to compare.")
    ap.add_argument("--ablate-memory", action="store_true",
                    help="Memory ablation: force is_first=True on every step "
                         "so the agent resets its RSSM carry each step (the "
                         "same reset path it was trained to handle at episode "
                         "boundaries). In-distribution amnesia — tests "
                         "whether the trained policy is actually using "
                         "recurrent memory.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    exp_dir = Path(args.exp_dir).resolve()

    run_cfg_path = run_dir / "run_config.json"
    if not run_cfg_path.exists():
        print(f"[eval-dreamer] ERROR: {run_cfg_path} doesn't exist; "
              f"can't determine variation/method.")
        sys.exit(1)
    with open(run_cfg_path) as f:
        run_cfg = json.load(f)

    # Scheduler / analyze_experiment use "dreamer"; train_dreamerv3.py writes
    # "dreamerv3" into run_config.json. Accept both — they're the same method.
    method_name = run_cfg["method"]
    if method_name not in ("dreamer", "dreamerv3"):
        print(f"[eval-dreamer] ERROR: this script handles dreamer runs only; "
              f"got method='{method_name}'. SB3 runs go through "
              f"eval_one_run.py.")
        sys.exit(1)

    variation_name = run_cfg["variation"]
    seed = int(run_cfg["metaseed"])
    exp_id = run_cfg["exp_id"]
    size = run_cfg.get("size", DEFAULT_SIZE)
    method_overrides = dict(run_cfg.get("method_overrides", {}))

    exp_yaml = exp_dir / "experiment.yaml"
    if not exp_yaml.exists():
        print(f"[eval-dreamer] ERROR: {exp_yaml} doesn't exist.")
        sys.exit(1)
    exp = load_experiment_def(exp_yaml)
    variation = find_variation(exp, variation_name)

    agent_preset = resolve_agent_preset(exp, variation)
    task_preset = resolve_task_preset(exp, variation)
    last_stage = exp.stages[-1]
    world_preset = resolve_stage_world(last_stage, variation, exp)

    print(f"[eval-dreamer] run_dir:      {run_dir}")
    print(f"[eval-dreamer] var/seed:     {variation_name} / {seed}")
    print(f"[eval-dreamer] agent_preset: {agent_preset}")
    print(f"[eval-dreamer] task_preset:  {task_preset}")
    print(f"[eval-dreamer] world_preset: {world_preset}  (from final stage)")
    print(f"[eval-dreamer] size:         {size}")

    ckpt_info = latest_dreamer_checkpoint(run_dir / "checkpoints")
    if ckpt_info is None:
        print(f"[eval-dreamer] ERROR: no completed checkpoints in "
              f"{run_dir / 'checkpoints'} (no final/agent.pkl and no "
              f"stage_K/agent.pkl with matching stage_K.done).")
        sys.exit(1)
    ckpt_path, stage_idx = ckpt_info
    print(f"[eval-dreamer] checkpoint:   {ckpt_path}  (stage_idx={stage_idx})")

    agent_config = blend_presets("agents", agent_preset)
    task_config = blend_presets("task", task_preset)
    world_config = blend_presets("world", world_preset)

    instances = allocate_unity_instances(n_envs=1)
    port = instances[0].port
    print(f"[eval-dreamer] unity_port:   {port}")

    eval_jsonl = run_dir / (
        "eval_episodes_ablated.jsonl" if args.ablate_memory
        else "eval_episodes.jsonl")
    if eval_jsonl.exists():
        eval_jsonl.unlink()  # restart episode_idx at 0

    run_metadata = {
        "method": method_name,
        "exp_id": exp_id,
        "variation": variation_name,
        "seed": seed,
        "stage_idx": stage_idx,
        "env_idx": 0,
    }

    # eval_metaseed → metaworldgen RNG → per-episode world seeds. Reproducible
    # across runs (every method/variation/seed sees the same sequence). The
    # run's *training* metaseed is in run_metadata for traceability only.
    gym_env = WildfireGymEnv(
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

    # Scratch logdir for the dreamer Config — Agent wants a writable path
    # for any incidental output. Cleaned next time analyzer runs (via unlink
    # of eval_episodes.jsonl above we don't, but the scratch dir is small).
    scratch_logdir = run_dir / "eval_dreamer_scratch"
    scratch_logdir.mkdir(exist_ok=True)

    config = build_config(method_overrides, scratch_logdir,
                          total_steps=1, size=size)
    # Single-env eval; no train ratio relevance.
    config = config.update({"run.envs": 1, "run.eval_envs": 0})

    adapter = GymnasiumToEmbodied(gym_env, obs_key="vector", act_key="action")
    wrapped = wrap_env(adapter, config)

    from dreamerv3.agent import Agent

    notlog = lambda k: not k.startswith("log/")
    obs_space = {k: v for k, v in wrapped.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in wrapped.act_space.items() if k != "reset"}

    agent = Agent(obs_space, act_space, elements.Config(
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
    ))
    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(str(ckpt_path), keys=["agent"])
    print("[eval-dreamer] agent loaded")

    print(f"[eval-dreamer] running {args.n_episodes} episodes "
          f"(eval_metaseed={args.eval_metaseed})"
          f"{'  [ABLATE MEMORY]' if args.ablate_memory else ''}")

    try:
        for ep in range(args.n_episodes):
            t0 = time.time()

            # Trigger reset through the embodied wrapper stack. Adapter sees
            # reset=True → calls gym_env.reset() (which draws the next world
            # seed from the metaworldgen RNG) → returns initial embodied obs.
            reset_act = _zero_action(wrapped)
            reset_act["reset"] = True
            obs = wrapped.step(reset_act)

            carry = agent.init_policy(1)
            steps = 0
            while not bool(obs["is_last"]):
                obs_b = {k: np.asarray(v)[None] for k, v in obs.items()
                         if not k.startswith("log/")}
                if args.ablate_memory:
                    obs_b["is_first"] = np.ones_like(obs_b["is_first"])
                carry, act, _ = agent.policy(carry, obs_b, mode="eval")
                action = {k: np.asarray(v)[0] for k, v in act.items()}
                action["reset"] = False
                obs = wrapped.step(action)
                steps += 1

            elapsed = time.time() - t0
            score = gym_env.task_tracker.get_total_score()
            pickups = gym_env.task_tracker.get_num_reward_objs_picked_up()
            print(f"[eval-dreamer] ep {ep + 1}/{args.n_episodes}: "
                  f"score={score:.2f}  pickups={pickups}  steps={steps}  "
                  f"{elapsed:.1f}s")
    finally:
        wrapped.close()

    print(f"[eval-dreamer] Done. Wrote {eval_jsonl}")


if __name__ == "__main__":
    main()
