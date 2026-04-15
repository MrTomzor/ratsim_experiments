"""Train DreamerV3 on a run definition.

Mirrors the CLI of train.py for parity with the SB3 methods:

    python train_dreamerv3.py def=default_forest_foraging
    python train_dreamerv3.py def=default_forest_foraging name=my_run metaseed=42
    python train_dreamerv3.py def=default_forest_foraging size=size12m step_multiplier=2.0
    python train_dreamerv3.py def=default_forest_foraging method.batch_size=8

Notes
-----
* Activate the dreamer venv first: ``source ~/ratvenv/dreamer_venv/bin/activate``.
* DreamerV3 doesn't natively do staged training the way SB3 does in train.py,
  so this trainer runs **stage 0 only** for now. Multi-stage support can be
  layered on later by re-entering the embodied train loop with a new env.
* Defaults assume CPU JAX (``jax.platform=cpu``); pass
  ``method.jax.platform=cuda`` on a GPU box.
"""
from __future__ import annotations

import argparse
import json
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

def _build_gym_env(rundef: dict, stage: dict, metaseed: int) -> WildfireGymEnv:
    return WildfireGymEnv(
        worldgen_config=resolve_world_config(stage),
        agent_config=resolve_agent_config(rundef),
        sensor_config={},
        action_config={"control_mode": "velocity"},
        task_config=resolve_task_config(rundef, stage),
        metaworldgen_config={"world_generation_metaseed": metaseed},
    )


def make_env(config, rundef, stage, metaseed, index, **overrides):
    """embodied-style env factory. `index` is the parallel env index (we use 1)."""
    del overrides, index  # single-env, no per-index variation
    gym_env = _build_gym_env(rundef, stage, metaseed)
    env = GymnasiumToEmbodied(gym_env, obs_key="vector", act_key="action")
    return wrap_env(env, config)


def make_agent(config, rundef, stage, metaseed):
    """embodied-style agent factory; mirrors dreamerv3.main.make_agent."""
    from dreamerv3.agent import Agent
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

DEFAULT_SIZE = "size12m"


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
        "jax.platform": "cpu",
        "jax.compute_dtype": "float32",
        "jax.prealloc": False,
        "logger.outputs": ["jsonl"],
    })

    # User overrides (`method.jax.platform=cuda`, `method.batch_size=4`, etc.)
    if method_overrides:
        config = config.update(method_overrides)

    return config


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
    if len(stages) > 1:
        print(f"[dreamerv3] WARNING: rundef has {len(stages)} stages; "
              "training stage 0 only (multi-stage not yet supported).")
    stage = stages[0]
    total_steps = int(stage["steps"] * step_multiplier)

    results_dir = Path(__file__).parent / "results" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    logdir = results_dir / "dreamer_logdir"
    logdir.mkdir(exist_ok=True)

    with open(results_dir / "run_config.json", "w") as f:
        json.dump({
            "rundef": rundef_stem,
            "method": "dreamerv3",
            "size": size,
            "step_multiplier": step_multiplier,
            "metaseed": metaseed,
            "run_name": run_name,
            "method_overrides": method_overrides,
            "stage_index": 0,
        }, f, indent=2)

    config = build_config(method_overrides, logdir, total_steps, size)
    config.save(elements.Path(str(logdir)) / "config.yaml")

    # embodied train loop assembly — same shape as dreamerv3.main.main.
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

    embodied.run.train(
        bind(make_agent, config, rundef, stage, metaseed),
        bind(make_replay, config, "replay"),
        bind(make_env, config, rundef, stage, metaseed),
        bind(make_stream, config),
        bind(make_logger, config),
        args_cfg,
    )

    print(f"\n[dreamerv3] Training complete. Logdir: {logdir}")


if __name__ == "__main__":
    main()
