"""Build a DreamerV3 agent for a finished scheduler run and load a checkpoint.

The same construction path as eval_one_run_dreamer.py (build_config from the
run's own method_overrides, Agent(...), elements.Checkpoint), factored out
because collect.py and encode.py both need it and eval_one_run_dreamer.main()
does it inline. That script itself is not used: in this tree it passes a
`record_trajectories=` kwarg that WildfireGymEnv does not accept.

Stock DreamerV3 and Dreamer (TBTT) runs need nothing beyond train_dreamerv3.
The architecture variants of the ratsim_metacentrum tree (gridmem, memrssm,
probeloss) are optional: where those packages exist, a variant run gets the
model it was trained with; where they don't, they are skipped.
"""
from __future__ import annotations

import importlib
import json
import tempfile
from pathlib import Path

import elements

from train_dreamerv3 import DEFAULT_SIZE, build_config


def _variant(name):
    try:
        return importlib.import_module(f"methods.dreamerv3.{name}")
    except ImportError:
        return None


VARIANTS = {n: m for n in ("gridmem", "memrssm", "probeloss") if (m := _variant(n))}


def run_config(run_dir) -> dict:
    return json.loads((Path(run_dir) / "run_config.json").read_text())


def stage_checkpoint(run_dir, stage: int) -> Path:
    """checkpoints/stage_K, accepted only with its stage_K.done marker: the
    marker is touched after the copy completes, so a dir without it may be a
    half-written snapshot."""
    ckpt = Path(run_dir) / "checkpoints" / f"stage_{stage}"
    done = Path(run_dir) / "checkpoints" / f"stage_{stage}.done"
    if not (done.exists() and (ckpt / "agent.pkl").exists()):
        raise FileNotFoundError(f"{ckpt}: no agent.pkl with a .done marker")
    return ckpt


def make_config(run_dir, platform: str | None = None, logdir=None):
    cfg = run_config(run_dir)
    logdir = Path(logdir or tempfile.mkdtemp(prefix="probe_agent_"))
    config = build_config(dict(cfg.get("method_overrides", {})), logdir,
                          total_steps=1, size=cfg.get("size", DEFAULT_SIZE))
    config = config.update({"run.envs": 1, "run.eval_envs": 0})
    if platform:
        config = config.update({"jax.platform": platform})
    return config


def build_agent(config, obs_space: dict, act_space: dict):
    obs_space = {k: v for k, v in obs_space.items() if not k.startswith("log/")}
    # Only a variant whose config block exists: build_config declares it next
    # to the variant's code, and an unpatched train_dreamerv3 has none.
    variants = {n: m for n, m in VARIANTS.items() if n in config}
    # A probe-loss run's checkpoint holds the probe head, which exists only if
    # the target key is in obs_space; the encoder never reads that key, so
    # observations without it encode exactly as at acting time.
    if "probeloss" in variants:
        obs_space = variants["probeloss"].with_target_space(obs_space, config)
    act_space = {k: v for k, v in act_space.items() if k != "reset"}
    # No-ops for the baseline and TBTT runs (no variant flag is set in their
    # method_overrides); kept so a memrssm run builds the model it was trained
    # with instead of failing to load.
    for module in variants.values():
        module.install(config)
    from dreamerv3.agent import Agent  # after install: probeloss swaps it
    return Agent(obs_space, act_space, elements.Config(
        **config.agent,
        logdir=str(config.logdir),
        seed=config.seed,
        # Inference only: skip compiling the train and report graphs, which
        # costs a minute and a few GB per agent and is never called here.
        jax={**config.jax, "precompile": False},
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    ))


def load_checkpoint(agent, ckpt_dir) -> None:
    cp = elements.Checkpoint()
    cp.agent = agent
    cp.load(str(ckpt_dir), keys=["agent"])
