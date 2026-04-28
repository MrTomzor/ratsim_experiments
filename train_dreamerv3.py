"""Train DreamerV3 on an experiment def.

Mirrors the CLI of train.py for parity with the SB3 methods:

    python train_dreamerv3.py def=method_compare
    python train_dreamerv3.py def=method_compare variation=baseline
    python train_dreamerv3.py def=method_compare run_folder=my_run metaseed=42
    python train_dreamerv3.py def=method_compare size=size12m step_multiplier=2.0
    python train_dreamerv3.py def=method_compare method.batch_size=8

Resuming an existing run (e.g. from the scheduler):
    python train_dreamerv3.py def=... run_folder=my_run start_stage=3 end_stage=4

Notes
-----
* Activate the dreamer venv first: ``source ~/ratvenv/dreamer_venv/bin/activate``.
* Multi-stage curricula are supported: each stage re-enters the embodied train
  loop with a new env factory while the agent checkpoint and replay buffer
  carry over. The step counter is cumulative.
* Defaults to CUDA JAX (``jax.platform=cuda``); pass
  ``method.jax.platform=cpu`` to force CPU on a box without a GPU.
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

import elements
import embodied
import ruamel.yaml as ryaml
import dreamerv3
from dreamerv3.main import make_logger, make_replay, make_stream, wrap_env

from ratsim.config_blender import blend_presets
from ratsim.unity_launcher import allocate_unity_instances
from ratsim_wildfire_gym_env.env import WildfireGymEnv

from methods.dreamerv3.env_adapter import GymnasiumToEmbodied

from experiment_defs import (
    ExperimentDef,
    StageSpec,
    VariationSpec,
    find_variation,
    load_experiment_def,
    resolve_agent_preset,
    resolve_def_path,
    resolve_stage_world,
    resolve_task_preset,
)


DEFS_DIR = Path(__file__).parent / "defs"


# -- CLI override parsing ----------------------------------------------------

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
    p = argparse.ArgumentParser(description="Train DreamerV3 on an experiment def.")
    p.add_argument("overrides", nargs="*")
    return p.parse_args()


# -- Env factory passed to embodied -----------------------------------------

def _build_gym_env(exp: ExperimentDef, variation: VariationSpec, stage: StageSpec,
                   metaseed: int,
                   episode_log_path: "Path | None" = None,
                   run_metadata: "dict | None" = None,
                   unity_port: int = 9000) -> WildfireGymEnv:
    world_preset_list = resolve_stage_world(stage, variation, exp)
    agent_preset_list = resolve_agent_preset(exp, variation)
    task_preset_list = resolve_task_preset(exp, variation)
    return WildfireGymEnv(
        worldgen_config=blend_presets("world", world_preset_list),
        agent_config=blend_presets("agents", agent_preset_list),
        sensor_config={},
        action_config={"control_mode": "velocity"},
        task_config=blend_presets("task", task_preset_list),
        metaworldgen_config={"world_generation_metaseed": metaseed},
        episode_log_path=episode_log_path,
        run_metadata=run_metadata,
        unity_port=unity_port,
    )


def make_env(config, exp, variation, stage, metaseed, index,
             episode_log_path=None, run_metadata=None,
             unity_ports=None, **overrides):
    """embodied-style env factory. `index` selects which Unity port to use."""
    del overrides
    if unity_ports is None:
        unity_ports = [9000]
    port = unity_ports[index % len(unity_ports)]
    md = run_metadata
    if md is not None:
        md = {**md, "env_idx": index}
    gym_env = _build_gym_env(exp, variation, stage, metaseed + index,
                             episode_log_path=episode_log_path,
                             run_metadata=md,
                             unity_port=port)
    env = GymnasiumToEmbodied(gym_env, obs_key="vector", act_key="action")
    return wrap_env(env, config)


def make_agent(config, exp, variation, stage, metaseed, unity_ports):
    """embodied-style agent factory; mirrors dreamerv3.main.make_agent."""
    from dreamerv3.agent import Agent
    env = make_env(config, exp, variation, stage, metaseed, 0, unity_ports=unity_ports)
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


def build_config(method_overrides: dict, logdir: Path, total_steps: int, size: str,
                 n_envs: int = 1):
    """Load dreamerv3's configs.yaml, apply size preset, then our overrides."""
    cfg_path = Path(dreamerv3.__file__).parent / "configs.yaml"
    raw = ryaml.YAML(typ="safe").load(cfg_path.read_text())
    config = elements.Config(raw["defaults"])
    config = config.update(raw[size])

    # replay.size: dreamerv3's default of 5M is sized for Atari; we hold full
    # RSSM state per step (~3 KB), so a 5M buffer OOMs the cloud box. 1M is
    # plenty and keeps in-memory replay around 5-10 GB.
    config = config.update({
        "logdir": str(logdir),
        "task": "ratsim_wildfire",  # cosmetic; we bypass the suite switch in main.py
        "batch_size": 8,
        "batch_length": 32,
        "report_length": 32,
        "replay.size": 1_000_000,
        "run.envs": int(n_envs),
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

    if method_overrides:
        config = config.update(method_overrides)
    return config


def snapshot_latest_ckpt(logdir: Path, dest: Path) -> None:
    """Copy embodied's rolling `ckpt/latest` into a stable per-stage dest."""
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


def main():
    args = parse_args()
    overrides = parse_overrides(args.overrides)

    def_arg = overrides.pop("def", None)
    if def_arg is None:
        print("Usage: python train_dreamerv3.py def=<exp_id_or_path> "
              "[variation=<name>] [overrides...]")
        sys.exit(1)
    variation_name = overrides.pop("variation", "baseline")

    def_path = resolve_def_path(DEFS_DIR, def_arg)
    if not def_path.exists():
        available = [f.stem for f in DEFS_DIR.glob("*.yaml")]
        print(f"[dreamerv3] ERROR: experiment def not found at {def_path}\n"
              f"            Available in {DEFS_DIR}: {available}")
        sys.exit(1)
    exp = load_experiment_def(def_path)
    variation = find_variation(exp, variation_name)

    run_folder = overrides.pop("run_folder", None)
    legacy_name = overrides.pop("name", None)
    if legacy_name is not None:
        if run_folder is None:
            print("[dreamerv3] WARNING: name= is deprecated, use run_folder= instead")
            run_folder = legacy_name
        else:
            print("[dreamerv3] WARNING: both name= and run_folder= given; using run_folder=")
    if run_folder is None:
        run_folder = (f"{exp.exp_id}_{variation_name}_dreamerv3_"
                      f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_name = run_folder

    step_multiplier = float(overrides.pop("step_multiplier", exp.step_multiplier))
    metaseed = int(overrides.pop("metaseed", np.random.randint(0, 10000)))
    size = overrides.pop("size", DEFAULT_SIZE)
    n_envs = int(overrides.pop("n_envs", 1))
    base_port_arg = overrides.pop("base_port", None)
    base_port = int(base_port_arg) if base_port_arg is not None else None

    start_stage = int(overrides.pop("start_stage", 0))
    end_stage_arg = overrides.pop("end_stage", None)

    # Method overrides priority (lowest → highest):
    #   variation.method_args (from def file)
    #   method_config file (if given)
    #   CLI method.X=Y overrides
    method_overrides: dict = dict(variation.method_args)
    method_config_file = overrides.pop("method_config", None)
    if method_config_file is not None:
        with open(method_config_file) as f:
            method_overrides.update(yaml.safe_load(f) or {})
    for key in list(overrides.keys()):
        if key.startswith("method."):
            method_overrides[key.removeprefix("method.")] = overrides.pop(key)

    n_stages = len(exp.stages)
    end_stage = n_stages if end_stage_arg is None else int(end_stage_arg)
    if start_stage < 0 or start_stage >= n_stages:
        print(f"[dreamerv3] ERROR: start_stage={start_stage} out of range [0, {n_stages})")
        sys.exit(1)
    if end_stage <= start_stage or end_stage > n_stages:
        print(f"[dreamerv3] ERROR: end_stage={end_stage} must be in ({start_stage}, {n_stages}]")
        sys.exit(1)

    results_dir = Path(__file__).parent / "results" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    logdir = results_dir / "dreamer_logdir"
    logdir.mkdir(exist_ok=True)
    ckpt_dir = results_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    episode_log_path = results_dir / "train_episodes.jsonl"
    base_run_metadata = {
        "method": "dreamerv3",
        "exp_id": exp.exp_id,
        "variation": variation_name,
        "seed": int(metaseed),
    }

    alloc_kwargs = {"n_envs": n_envs}
    if base_port is not None:
        alloc_kwargs["base_port"] = base_port
    unity_instances = allocate_unity_instances(**alloc_kwargs)
    unity_ports = [inst.port for inst in unity_instances]
    print(f"[dreamerv3] n_envs={n_envs}, unity_ports={unity_ports}")
    print(f"[dreamerv3] exp={exp.exp_id} variation={variation_name}")

    # Cumulative step target across stages — embodied.run.train uses an
    # absolute step counter, so stage N trains from sum(stages[:N]) to
    # sum(stages[:N+1]). The agent checkpoint + replay persist across calls
    # because the logdir is the same.
    cumulative_steps = 0

    for stage_idx, stage in enumerate(exp.stages):
        stage_steps = int(stage.steps * step_multiplier)
        cumulative_steps += stage_steps

        if stage_idx < start_stage or stage_idx >= end_stage:
            continue

        world_preset_list = resolve_stage_world(stage, variation, exp)
        print(f"\n{'='*60}")
        print(f"[dreamerv3] Stage {stage_idx + 1}/{n_stages}: world={world_preset_list}")
        print(f"[dreamerv3] Stage steps: {stage_steps}  |  "
              f"Cumulative target: {cumulative_steps}")
        print(f"{'='*60}")

        stage_done = ckpt_dir / f"stage_{stage_idx}.done"
        if stage_done.exists():
            print(f"[dreamerv3] {stage_done} exists; stage already complete, skipping.")
            continue

        with open(results_dir / "run_config.json", "w") as f:
            json.dump({
                "exp_id": exp.exp_id,
                "exp_def": str(exp.source),
                "variation": variation_name,
                "method": "dreamerv3",
                "size": size,
                "step_multiplier": step_multiplier,
                "metaseed": metaseed,
                "run_name": run_name,
                "method_overrides": method_overrides,
                "stage_index": stage_idx,
                "cumulative_steps": cumulative_steps,
            }, f, indent=2)

        config = build_config(method_overrides, logdir, cumulative_steps, size, n_envs=n_envs)
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
            bind(make_agent, config, exp, variation, stage, metaseed, unity_ports),
            bind(make_replay, config, "replay"),
            bind(make_env, config, exp, variation, stage, metaseed,
                 episode_log_path=episode_log_path,
                 run_metadata=stage_run_metadata,
                 unity_ports=unity_ports),
            bind(make_stream, config),
            bind(make_logger, config),
            args_cfg,
        )

        snapshot_latest_ckpt(logdir, ckpt_dir / f"stage_{stage_idx}")
        (ckpt_dir / f"stage_{stage_idx}.done").touch()
        print(f"[dreamerv3] Stage {stage_idx + 1}/{n_stages} complete.")

    all_stages_done = all(
        (ckpt_dir / f"stage_{i}.done").exists() for i in range(n_stages)
    )
    if all_stages_done:
        snapshot_latest_ckpt(logdir, ckpt_dir / "final")
        (results_dir / "DONE").touch()
        print(f"\n[dreamerv3] All {n_stages} stages complete. Logdir: {logdir}")
    else:
        completed = [i for i in range(n_stages)
                     if (ckpt_dir / f"stage_{i}.done").exists()]
        print(f"\n[dreamerv3] Partial run: stages done = {completed} / {n_stages}")


if __name__ == "__main__":
    main()
