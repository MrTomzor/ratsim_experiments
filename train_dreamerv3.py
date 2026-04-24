"""Train DreamerV3 on a run definition.

Mirrors the CLI of train.py for parity with the SB3 methods:

    python train_dreamerv3.py def=default_forest_foraging
    python train_dreamerv3.py def=default_forest_foraging name=my_run metaseed=42
    python train_dreamerv3.py def=default_forest_foraging size=size12m step_multiplier=2.0
    python train_dreamerv3.py def=default_forest_foraging method.batch_size=8

Notes
-----
* Activate the dreamer venv first: ``source ~/ratvenv/dreamer_venv/bin/activate``.
* Multi-stage rundefs are supported: each stage re-enters the embodied train
  loop with a new env factory (potentially different world_presets) while the
  agent checkpoint and replay buffer carry over. The step counter is cumulative.
* Defaults assume CPU JAX (``jax.platform=cpu``); pass
  ``method.jax.platform=cuda`` on a GPU box.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from functools import partial as bind
from pathlib import Path

import numpy as np
import yaml

# DreamerV3 imports — these require the `dreamer_venv` environment.
import elements
import embodied
import ruamel.yaml as ryaml
import dreamerv3
from dreamerv3.main import make_logger, make_replay, make_stream, wrap_env

from ratsim.config_blender import blend_presets
from ratsim_wildfire_gym_env.env import WildfireGymEnv

from methods.dreamerv3.env_adapter import GymnasiumToEmbodied


# -- Run definition loading (mirrors train.py) -------------------------------

def load_rundef(name_or_path: str) -> dict:
    path = Path(name_or_path)
    if path.suffix in (".yaml", ".yml") and path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    rundef_dir = Path(__file__).parent / "rundefs"
    path = rundef_dir / f"{name_or_path}.yaml"
    if not path.exists():
        available = [f.stem for f in rundef_dir.glob("*.yaml")]
        raise FileNotFoundError(
            f"Run definition '{name_or_path}' not found at {path}\nAvailable: {available}"
        )
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_world_config(stage: dict) -> dict:
    cfg = blend_presets("world", stage.get("world_presets", ["default"]))
    cfg.update(stage.get("world_overrides", {}))
    return cfg


def resolve_task_config(rundef: dict, stage: dict) -> dict:
    cfg = blend_presets("task", [rundef.get("task_preset", "default")])
    cfg.update(rundef.get("task_overrides", {}))
    cfg.update(stage.get("task_overrides", {}))
    return cfg


def resolve_agent_config(rundef: dict) -> dict:
    return blend_presets("agents", [rundef.get("agent_preset", "sphereagent_2d_lidar")])


# -- CLI override parsing (mirrors train.py) ---------------------------------

def parse_overrides(items: list[str]) -> dict:
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key=value")
        k, v = item.split("=", 1)
        try:
            v = yaml.safe_load(v)
        except yaml.YAMLError:
            pass
        out[k] = v
    return out


def parse_args():
    p = argparse.ArgumentParser(description="Train DreamerV3 on a run definition.")
    p.add_argument("overrides", nargs="*")
    return p.parse_args()


# -- Env factory passed to embodied ------------------------------------------

def _build_gym_env(rundef: dict, stage: dict, metaseed: int,
                   episode_log_path: "Path | None" = None,
                   run_metadata: "dict | None" = None) -> WildfireGymEnv:
    return WildfireGymEnv(
        worldgen_config=resolve_world_config(stage),
        agent_config=resolve_agent_config(rundef),
        sensor_config={},
        action_config={"control_mode": "velocity"},
        task_config=resolve_task_config(rundef, stage),
        metaworldgen_config={"world_generation_metaseed": metaseed},
        episode_log_path=episode_log_path,
        run_metadata=run_metadata,
    )


def make_env(config, rundef, stage, metaseed, index,
             episode_log_path=None, run_metadata=None, **overrides):
    """embodied-style env factory. `index` is the parallel env index (we use 1)."""
    del overrides, index  # single-env, no per-index variation
    gym_env = _build_gym_env(rundef, stage, metaseed,
                             episode_log_path=episode_log_path,
                             run_metadata=run_metadata)
    env = GymnasiumToEmbodied(gym_env, obs_key="vector", act_key="action")
    return wrap_env(env, config)


def make_agent(config, rundef, stage, metaseed):
    """embodied-style agent factory; mirrors dreamerv3.main.make_agent."""
    from dreamerv3.agent import Agent
    # Agent factory builds a temporary env to read obs/act spaces; no logging needed here.
    env = make_env(config, rundef, stage, metaseed, 0)
    notlog = lambda k: not k.startswith("log/")
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != "reset"}
    env.close()
    if config.random_agent:
        return embodied.RandomAgent(obs_space, act_space)
    return Agent(obs_space, act_space, elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    ))


# -- Config assembly ---------------------------------------------------------

DEFAULT_SIZE = "size1m"


def build_config(method_overrides: dict, logdir: Path, total_steps: int, size: str):
    """Load dreamerv3's configs.yaml, apply size preset, then our overrides."""
    cfg_path = Path(dreamerv3.__file__).parent / "configs.yaml"
    raw = ryaml.YAML(typ="safe").load(cfg_path.read_text())
    config = elements.Config(raw["defaults"])
    config = config.update(raw[size])

    # Defaults appropriate for ratsim + CPU.
    config = config.update({
        "logdir": str(logdir),
        "task": "ratsim_wildfire",  # cosmetic; we bypass the suite switch in main.py
        "batch_size": 8,
        "batch_length": 32,
        "report_length": 32,
        "run.envs": 1,
        "run.eval_envs": 0,
        "run.steps": int(total_steps),
        "run.train_ratio": 32.0,
        "run.actor_threads": 1,
        "run.agent_process": False,
        "run.remote_envs": False,
        "run.remote_replay": False,
        "jax.platform": "cuda",
        "jax.compute_dtype": "float32",
        "jax.prealloc": False,
        "logger.outputs": ["jsonl", "tensorboard"],
    })

    # User overrides (`method.jax.platform=cuda`, `method.batch_size=4`, etc.)
    if method_overrides:
        config = config.update(method_overrides)

    return config


# -- Checkpoint snapshotting -------------------------------------------------

def snapshot_latest_ckpt(logdir: Path, dest: Path) -> None:
    """Copy embodied's rolling `ckpt/latest` into a stable per-stage dest.

    Mirrors PPO's per-stage checkpoint behavior so the eval script can load
    `results/<run>/checkpoints/stage_N/` directly.
    """
    latest_pointer = logdir / "ckpt" / "latest"
    if not latest_pointer.exists():
        print(f"[dreamerv3] WARNING: no ckpt/latest at {latest_pointer}; skipping snapshot")
        return
    latest_name = latest_pointer.read_text().strip()
    src = logdir / "ckpt" / latest_name
    if not src.exists():
        print(f"[dreamerv3] WARNING: ckpt/latest -> {src} does not exist; skipping snapshot")
        return
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    print(f"[dreamerv3] Snapshot: {dest}")


# -- Main --------------------------------------------------------------------

def main():
    args = parse_args()
    overrides = parse_overrides(args.overrides)

    rundef_name = overrides.pop("def", None)
    if rundef_name is None:
        print("Usage: python train_dreamerv3.py def=<rundef_name> [overrides...]")
        sys.exit(1)

    rundef_stem = Path(rundef_name).stem
    run_name = overrides.pop(
        "name", f"{rundef_stem}_dreamerv3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    step_multiplier = float(overrides.pop("step_multiplier", 1.0))
    metaseed = int(overrides.pop("metaseed", np.random.randint(0, 10000)))
    size = overrides.pop("size", DEFAULT_SIZE)

    method_overrides = {}
    method_config_file = overrides.pop("method_config", None)
    if method_config_file is not None:
        with open(method_config_file) as f:
            method_overrides.update(yaml.safe_load(f) or {})
    for key in list(overrides.keys()):
        if key.startswith("method."):
            method_overrides[key.removeprefix("method.")] = overrides.pop(key)

    rundef = load_rundef(rundef_name)
    stages = rundef["stages"]

    results_dir = Path(__file__).parent / "results" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    logdir = results_dir / "dreamer_logdir"
    logdir.mkdir(exist_ok=True)
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    episode_log_path = results_dir / "train_episodes.jsonl"
    base_run_metadata = {
        "method": "dreamerv3",
        "rundef": rundef_stem,
        "seed": int(metaseed),
    }

    # Cumulative step target across stages — embodied.run.train uses an
    # absolute step counter, so stage N trains from sum(stages[:N]) to
    # sum(stages[:N+1]).  The agent checkpoint + replay buffer persist
    # across calls because the logdir is the same.
    cumulative_steps = 0

    for stage_idx, stage in enumerate(stages):
        stage_steps = int(stage["steps"] * step_multiplier)
        cumulative_steps += stage_steps

        print(f"\n{'='*60}")
        print(f"[dreamerv3] Stage {stage_idx + 1}/{len(stages)}: "
              f"{stage.get('world_presets', ['?'])}")
        print(f"[dreamerv3] Stage steps: {stage_steps}  |  "
              f"Cumulative target: {cumulative_steps}")
        print(f"{'='*60}")

        # Skip stages already completed in a prior run. embodied.run.train uses
        # an absolute step counter, so re-entering a completed stage would load
        # the checkpoint, exit immediately (target already reached), and
        # overwrite the snapshot with itself — wasting time setting up the env,
        # agent and JAX compilation on each pass.
        stage_snapshot = ckpt_dir / f"stage_{stage_idx}"
        if stage_snapshot.exists():
            print(f"[dreamerv3] Snapshot exists at {stage_snapshot}; "
                  f"stage already complete, skipping.")
            continue

        with open(results_dir / "run_config.json", "w") as f:
            json.dump({
                "rundef": rundef_stem,
                "method": "dreamerv3",
                "size": size,
                "step_multiplier": step_multiplier,
                "metaseed": metaseed,
                "run_name": run_name,
                "method_overrides": method_overrides,
                "stage_index": stage_idx,
                "cumulative_steps": cumulative_steps,
            }, f, indent=2)

        config = build_config(method_overrides, logdir, cumulative_steps, size)
        config.save(elements.Path(str(logdir)) / "config.yaml")

        if stage_idx == 0:
            print(f"\n[dreamerv3] size preset: {size}")
            print(f"[dreamerv3]   rssm.deter={config.agent.dyn.rssm.deter}, "
                  f"hidden={config.agent.dyn.rssm.hidden}, "
                  f"classes={config.agent.dyn.rssm.classes}")
            print(f"[dreamerv3]   units={config.agent.policy.units}, "
                  f"enc.depth={config.agent.enc.simple.depth}")
            print(f"[dreamerv3]   batch_size={config.batch_size}, "
                  f"batch_length={config.batch_length}, "
                  f"train_ratio={config.run.train_ratio}")

        args_cfg = elements.Config(
            **config.run,
            replica=config.replica,
            replicas=config.replicas,
            logdir=str(logdir),
            batch_size=config.batch_size,
            batch_length=config.batch_length,
            report_length=config.report_length,
            consec_train=config.consec_train,
            consec_report=config.consec_report,
            replay_context=config.replay_context,
        )

        stage_run_metadata = {**base_run_metadata, "stage_idx": stage_idx}

        embodied.run.train(
            bind(make_agent, config, rundef, stage, metaseed),
            bind(make_replay, config, "replay"),
            bind(make_env, config, rundef, stage, metaseed,
                 episode_log_path=episode_log_path,
                 run_metadata=stage_run_metadata),
            bind(make_stream, config),
            bind(make_logger, config),
            args_cfg,
        )

        snapshot_latest_ckpt(logdir, ckpt_dir / f"stage_{stage_idx}")
        print(f"[dreamerv3] Stage {stage_idx + 1}/{len(stages)} complete.")

    snapshot_latest_ckpt(logdir, ckpt_dir / "final")
    (results_dir / "DONE").touch()
    print(f"\n[dreamerv3] All {len(stages)} stages complete. Logdir: {logdir}")


if __name__ == "__main__":
    main()
