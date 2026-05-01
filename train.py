"""
Train an RL agent.

Two ways to specify what to train on:

1. Saved experiment def (use this for anything you'll re-run / share):
    python train.py def=method_compare method=ppo
    python train.py def=method_compare method=ppo variation=baseline
    python train.py def=gps_ablation method=dreamer variation=no_gps
    python train.py def=method_compare method=ppo run_folder=my_run

2. Inline (no def file — give the same fields on the CLI):
    python train.py method=ppo agent_preset=sphereagent_2d_lidar \\
                    task_preset=default world_preset=maze_default \\
                    total_steps=1_000_000 n_stages=10
    # Minimal: defaults agent_preset=sphereagent_2d_lidar, task_preset=default,
    # n_stages=1. world_preset is required.
    python train.py method=ppo world_preset=maze_default total_steps=100_000

Resuming an existing run (e.g. from the scheduler):
    python train.py def=... method=ppo run_folder=my_run start_stage=3 end_stage=4

`def=` accepts either a bare name (looked up in defs/) or a path to a yaml.
`variation=` defaults to "baseline" — every experiment has at least that one
implicitly. `metaseed=N` controls the world-generation seed (random by default).
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from sb3_contrib import RecurrentPPO

from ratsim.config_blender import blend_presets
from ratsim.unity_launcher import allocate_unity_instances
from ratsim_wildfire_gym_env.env import WildfireGymEnv
from feature_extractors import LidarCnnExtractor

from experiment_defs import (
    ExperimentDef,
    StageSpec,
    VariationSpec,
    build_inline_def,
    find_variation,
    load_experiment_def,
    resolve_agent_preset,
    resolve_def_path,
    resolve_stage_world,
    resolve_task_preset,
)


DEFS_DIR = Path(__file__).parent / "defs"


# -- Methods ------------------------------------------------------------------

METHODS = {
    "ppo": {
        "sb3_class": PPO,
        "policy": "MultiInputPolicy",
    },
    "recurrent_ppo": {
        "sb3_class": RecurrentPPO,
        "policy": "MultiInputLstmPolicy",
    },
    "cnn_ppo": {
        "sb3_class": PPO,
        "policy": "MultiInputPolicy",
        "policy_kwargs": {
            "features_extractor_class": LidarCnnExtractor,
            "features_extractor_kwargs": {},
        },
    },
    "cnn_recurrent_ppo": {
        "sb3_class": RecurrentPPO,
        "policy": "MultiInputLstmPolicy",
        "policy_kwargs": {
            "features_extractor_class": LidarCnnExtractor,
            "features_extractor_kwargs": {},
        },
    },
}


# Recurrent PPO preset profiles. Select via method.recurrent_preset=<name>
# or override individual params with method.n_steps=... etc.
RECURRENT_PRESETS = {
    # A) Full-episode unroll — correct LSTM training, slow on CPU
    "full_episode": {
        "n_steps": "auto",      # set to max episode_max_steps
        "batch_size": "auto",   # set to n_steps
        "n_epochs": 15,
    },
    # B) SB3 defaults — fast but sequences split across batches
    "sb3_default": {
        "n_steps": 128,
        "batch_size": 128,
        "n_epochs": 10,
    },
    # C) Balanced — shorter unrolls + smaller LSTM, practical for CPU
    "balanced": {
        "n_steps": 512,
        "batch_size": 512,
        "n_epochs": 10,
        "policy_kwargs": {"lstm_hidden_size": 128, "n_lstm_layers": 1},
    },
    # D) Deep-memory — full-episode BPTT + long-horizon credit assignment.
    # For tasks where the policy must remember information from the start
    # of a long episode (e.g. which houses it has already visited in a
    # 2000-step foraging episode).
    #   n_steps == episode_max_steps  → unbroken BPTT from step 0 to terminal
    #   batch_size == n_steps         → each minibatch is one full env trajectory,
    #                                    so gradients flow through the entire episode
    #   n_epochs lower than full_episode (5 vs 15) — fewer passes over a small
    #     number of long sequences avoids overfitting one rollout
    #   gamma=0.999 → effective horizon ~1000 steps (0.99 was ~100, useless here)
    #   lr lower (1e-4) — long-BPTT updates are noisier; helps stability
    #   bigger LSTM (256) — capacity to actually retain per-house occupancy state
    "deep_memory": {
        "n_steps": "auto",
        "batch_size": "auto",
        "n_epochs": 5,
        "learning_rate": 1e-4,
        "gamma": 0.999,
        "gae_lambda": 0.95,
        "ent_coef": 0.01,
        "max_grad_norm": 0.5,
        "policy_kwargs": {"lstm_hidden_size": 256, "n_lstm_layers": 1},
    },
}
RECURRENT_PRESET_DEFAULT = "deep_memory"


def create_model(method_name: str, env, method_config: dict, tb_log_dir: str,
                 max_episode_steps: int | None = None):
    """Create an SB3 model from method name and config."""
    method = METHODS[method_name]
    is_recurrent = method["sb3_class"] is RecurrentPPO

    kwargs = {
        "policy": method["policy"],
        "env": env,
        "verbose": 1,
        "n_steps": 2048,
        "batch_size": 64,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "tensorboard_log": tb_log_dir,
    }

    if "policy_kwargs" in method:
        import copy
        kwargs["policy_kwargs"] = copy.deepcopy(method["policy_kwargs"])

    if "policy_kwargs" in kwargs and "features_extractor_kwargs" in kwargs.get("policy_kwargs", {}):
        ext_kwargs = kwargs["policy_kwargs"]["features_extractor_kwargs"]
        unwrapped = env.envs[0].unwrapped if hasattr(env, "envs") else env.unwrapped
        ext_kwargs.setdefault("n_rays", unwrapped.num_lidar_rays)
        ext_kwargs.setdefault("n_channels", unwrapped.num_lidar_channels)
        for pname in ("n_rays", "n_channels", "cnn_output_dim"):
            if pname in method_config:
                ext_kwargs[pname] = int(method_config.pop(pname))

    if is_recurrent:
        preset_name = method_config.pop("recurrent_preset", RECURRENT_PRESET_DEFAULT)
        preset = RECURRENT_PRESETS[preset_name]
        print(f"[RecurrentPPO] Using preset '{preset_name}': {preset}")
        for k, v in preset.items():
            if k not in method_config:
                if v == "auto" and max_episode_steps is not None:
                    v = max(2048, max_episode_steps)
                    print(f"[RecurrentPPO] Auto {k}={v} (max episode_max_steps={max_episode_steps})")
                if k == "policy_kwargs" and "policy_kwargs" in kwargs:
                    kwargs["policy_kwargs"].update(v)
                else:
                    kwargs[k] = v

    if "policy_kwargs" in method_config and "policy_kwargs" in kwargs:
        kwargs["policy_kwargs"].update(method_config.pop("policy_kwargs"))

    kwargs.update(method_config)

    model = method["sb3_class"](**kwargs)
    print_model_summary(model)
    return model


def print_model_summary(model):
    """Print key model size info for sanity checking."""
    policy = model.policy
    feat_dim = policy.features_extractor.features_dim
    extractor_name = type(policy.features_extractor).__name__
    print(f"\n{'='*50}")
    print(f"MODEL SUMMARY")
    print(f"{'='*50}")
    print(f"Features extractor: {extractor_name} -> {feat_dim}-dim")
    if hasattr(policy, "lstm_actor"):
        lstm = policy.lstm_actor
        print(f"LSTM (actor):  hidden_size={lstm.hidden_size}, num_layers={lstm.num_layers}")
    if hasattr(policy, "lstm_critic") and policy.lstm_critic is not None:
        lstm = policy.lstm_critic
        print(f"LSTM (critic): hidden_size={lstm.hidden_size}, num_layers={lstm.num_layers}")
    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Total parameters:     {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"{'='*50}\n")


# -- Stage transition resets ---------------------------------------------------

def reset_critic(model):
    """Reinitialize value function weights, keeping the policy (actor) intact."""
    policy = model.policy

    def reinit_module(module):
        if module is None:
            return
        for layer in module.modules():
            if isinstance(layer, th.nn.Linear):
                th.nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                th.nn.init.constant_(layer.bias, 0.0)
            elif isinstance(layer, th.nn.LSTM):
                for name, param in layer.named_parameters():
                    if "weight" in name:
                        th.nn.init.orthogonal_(param)
                    elif "bias" in name:
                        th.nn.init.constant_(param, 0.0)

    reinit_module(policy.mlp_extractor.value_net)
    th.nn.init.orthogonal_(policy.value_net.weight, gain=1.0)
    th.nn.init.constant_(policy.value_net.bias, 0.0)
    if hasattr(policy, "lstm_critic") and policy.lstm_critic is not None:
        reinit_module(policy.lstm_critic)
    if hasattr(policy, "critic") and policy.critic is not None:
        reinit_module(policy.critic)
    print("[StageTransition] Critic weights reinitialized")


def reset_optimizer(model):
    for state in model.policy.optimizer.state.values():
        state.clear()
    print("[StageTransition] Optimizer state cleared")


# -- Callback -----------------------------------------------------------------

class TrainingMetricsCallback(BaseCallback):
    """Logs custom metrics from the env to TensorBoard."""

    def __init__(self, log_freq=2048, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            all_distances = self.training_env.env_method("get_completed_episode_distances")
            all_pickups = self.training_env.env_method("get_completed_episode_pickups")
            all_explored = self.training_env.env_method("get_completed_episode_explored_area")
            longest_step_distance = self.training_env.env_method("get_longest_step_distance")

            flat_distances = [d for env_list in all_distances for d in env_list]
            flat_pickups = [p for env_list in all_pickups for p in env_list]
            flat_explored = [e for env_list in all_explored for e in env_list]

            if flat_distances:
                self.logger.record("custom/avg_distance_traveled", np.mean(flat_distances))
            if flat_pickups:
                self.logger.record("custom/avg_reward_pickups", np.mean(flat_pickups))
            if flat_explored:
                self.logger.record("custom/avg_explored_area_m2", np.mean(flat_explored))
            self.logger.record("custom/longest_step_distance", np.max(longest_step_distance))
        return True


# -- CLI ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train an RL agent on an experiment def.")
    parser.add_argument("overrides", nargs="*", help="key=value overrides")
    return parser.parse_args()


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
    args = parse_args()
    overrides = parse_overrides(args.overrides)

    # `def=` is optional — without it, train.py runs in inline mode and
    # builds a single-method, single-variation experiment from the CLI args.
    def_arg = overrides.pop("def", None)
    method_name = overrides.pop("method", "ppo")
    variation_name = overrides.pop("variation", "baseline")

    if def_arg is not None:
        def_path = resolve_def_path(DEFS_DIR, def_arg)
        if not def_path.exists():
            available = [f.stem for f in DEFS_DIR.glob("*.yaml")]
            print(f"[train] ERROR: experiment def not found at {def_path}\n"
                  f"        Available in {DEFS_DIR}: {available}")
            sys.exit(1)
        exp = load_experiment_def(def_path)
    else:
        # Inline mode — pop the experiment-shaping keys from overrides.
        try:
            exp = build_inline_def(method_name, overrides)
        except ValueError as e:
            print(f"[train] ERROR (inline def): {e}\n"
                  f"        Required: method=, world_preset=, total_steps=. "
                  f"Optional: agent_preset=, task_preset=, n_stages= (default 1).")
            sys.exit(1)
    variation = find_variation(exp, variation_name)

    # `run_folder` is canonical; `name=` is a deprecated alias.
    run_folder = overrides.pop("run_folder", None)
    legacy_name = overrides.pop("name", None)
    if legacy_name is not None:
        if run_folder is None:
            print("[train] WARNING: name= is deprecated, use run_folder= instead")
            run_folder = legacy_name
        else:
            print("[train] WARNING: both name= and run_folder= given; using run_folder=")
    if run_folder is None:
        run_folder = (f"{exp.exp_id}_{variation_name}_{method_name}_"
                      f"{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    run_name = run_folder

    step_multiplier = float(overrides.pop("step_multiplier", exp.step_multiplier))
    n_envs = int(overrides.pop("n_envs", 1))
    base_port_arg = overrides.pop("base_port", None)
    base_port = int(base_port_arg) if base_port_arg is not None else None

    # Stage range: [start_stage, end_stage). Defaults run all stages.
    start_stage = int(overrides.pop("start_stage", 0))
    end_stage_arg = overrides.pop("end_stage", None)
    # Optional initial-weights checkpoint. Only applied when start_stage==0
    # (start_stage>0 already auto-resumes from stage_{N-1}.zip in this run's
    # checkpoint dir, which takes precedence). Useful for extending a finished
    # inline run that has no further stages to advance into.
    resume_from = overrides.pop("resume_from", None)

    # Random metaseed by default — non-benchmark runs want some seed variability
    # between invocations.
    default_metaseed = np.random.randint(0, 10000)
    metaseed = int(overrides.pop("metaseed", default_metaseed))
    method_config_file = overrides.pop("method_config", None)

    # Method config: priority (lowest → highest):
    #   variation.method_args (from def file)
    #   method_config file (if given)
    #   CLI method.X=Y overrides
    method_config: dict = dict(variation.method_args)
    if method_config_file is not None:
        with open(method_config_file) as f:
            method_config.update(yaml.safe_load(f) or {})
    for key in list(overrides.keys()):
        if key.startswith("method."):
            method_config[key.removeprefix("method.")] = overrides.pop(key)

    # Validate stage range
    n_stages = len(exp.stages)
    end_stage = n_stages if end_stage_arg is None else int(end_stage_arg)
    if start_stage < 0 or start_stage >= n_stages:
        print(f"[train] ERROR: start_stage={start_stage} out of range [0, {n_stages})")
        sys.exit(1)
    if end_stage <= start_stage or end_stage > n_stages:
        print(f"[train] ERROR: end_stage={end_stage} must be in ({start_stage}, {n_stages}]")
        sys.exit(1)

    # Resolve agent + task configs (constant across stages within a variation).
    agent_preset_list = resolve_agent_preset(exp, variation)
    task_preset_list = resolve_task_preset(exp, variation)
    agent_config = blend_presets("agents", agent_preset_list)
    task_config_full = blend_presets("task", task_preset_list)
    max_episode_steps = int(task_config_full.get("episode_max_steps", 300))

    # Output directory
    results_dir = Path(__file__).parent / "results" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    run_meta = {
        "exp_id": exp.exp_id,
        "exp_def": str(exp.source),
        "variation": variation_name,
        "method": method_name,
        "agent_preset": agent_preset_list,
        "task_preset": task_preset_list,
        "method_config": method_config,
        "step_multiplier": step_multiplier,
        "metaseed": metaseed,
        "run_name": run_name,
        "overrides": overrides,
    }
    with open(results_dir / "run_config.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    tb_log_dir = str(results_dir / "tensorboard")
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    episode_log_path = results_dir / "train_episodes.jsonl"
    base_run_metadata = {
        "method": method_name,
        "exp_id": exp.exp_id,
        "variation": variation_name,
        "seed": int(metaseed),
    }

    # Allocate Unity instances once for the whole run; ports persist across stages.
    alloc_kwargs = {"n_envs": n_envs}
    if base_port is not None:
        alloc_kwargs["base_port"] = base_port
    unity_instances = allocate_unity_instances(**alloc_kwargs)
    unity_ports = [inst.port for inst in unity_instances]
    print(f"[train] n_envs={n_envs}, unity_ports={unity_ports}")
    print(f"[train] exp={exp.exp_id} variation={variation_name} method={method_name}")
    print(f"[train] agent_preset={agent_preset_list} task_preset={task_preset_list}")

    model = None
    total_steps_trained = 0

    # Resume from the previous stage's checkpoint if start_stage > 0.
    if start_stage > 0:
        prev_ckpt = checkpoint_dir / f"stage_{start_stage - 1}.zip"
        if not prev_ckpt.exists():
            print(f"[train] ERROR: start_stage={start_stage} but {prev_ckpt} does not exist")
            sys.exit(1)
        if resume_from is not None:
            print(f"[train] WARNING: ignoring resume_from={resume_from}; "
                  f"start_stage>0 takes precedence")
        sb3_class = METHODS[method_name]["sb3_class"]
        print(f"[train] Resuming from {prev_ckpt}")
        model = sb3_class.load(str(prev_ckpt), tensorboard_log=tb_log_dir)
    elif resume_from is not None:
        ckpt = Path(resume_from)
        # SB3's model.save / model.load accepts the path with or without .zip
        if not ckpt.exists() and not Path(str(ckpt) + ".zip").exists():
            print(f"[train] ERROR: resume_from path '{ckpt}' does not exist")
            sys.exit(1)
        sb3_class = METHODS[method_name]["sb3_class"]
        print(f"[train] Loading initial weights from {ckpt}")
        model = sb3_class.load(str(ckpt), tensorboard_log=tb_log_dir)

    for stage_idx in range(start_stage, end_stage):
        stage = exp.stages[stage_idx]
        world_preset_list = resolve_stage_world(stage, variation, exp)
        world_config = blend_presets("world", world_preset_list)
        task_config = task_config_full  # no per-stage task overrides in current schema
        stage_steps = int(stage.steps * step_multiplier)

        print(f"\n{'='*60}")
        print(f"Stage {stage_idx + 1}/{n_stages}: world={world_preset_list}")
        print(f"Steps: {stage_steps} (multiplier: {step_multiplier})")
        print(f"{'='*60}\n")

        stage_run_metadata = {**base_run_metadata, "stage_idx": stage_idx}

        def make_env_factory(port, env_idx, wc=world_config, tc=task_config):
            md = {**stage_run_metadata, "env_idx": env_idx}
            def _make():
                env = WildfireGymEnv(
                    worldgen_config=wc,
                    agent_config=agent_config,
                    sensor_config={},
                    action_config={"control_mode": "velocity"},
                    task_config=tc,
                    metaworldgen_config={"world_generation_metaseed": metaseed + env_idx},
                    episode_log_path=episode_log_path,
                    run_metadata=md,
                    unity_port=port,
                )
                return Monitor(env)
            return _make

        env_fns = [make_env_factory(p, i) for i, p in enumerate(unity_ports)]
        VecEnvCls = SubprocVecEnv if n_envs > 1 else DummyVecEnv
        env = VecEnvCls(env_fns)

        is_stage_transition = model is not None
        # NOTE: reset_on_stage_change is no longer in the experiment def schema.
        # If we need it back, add it as an exp-level field.
        reset_on_stage = False

        if not is_stage_transition:
            model = create_model(method_name, env, method_config, tb_log_dir,
                                 max_episode_steps=max_episode_steps)
        else:
            model.set_env(env)
            if reset_on_stage:
                reset_critic(model)
                reset_optimizer(model)

        model.learn(
            total_timesteps=stage_steps,
            callback=TrainingMetricsCallback(),
            reset_num_timesteps=is_stage_transition and reset_on_stage,
        )
        total_steps_trained += stage_steps

        checkpoint_path = checkpoint_dir / f"stage_{stage_idx}"
        model.save(str(checkpoint_path))
        (checkpoint_dir / f"stage_{stage_idx}.done").touch()
        print(f"Saved checkpoint: {checkpoint_path}")

        env.close()

    # Final model + top-level DONE only when *all* stages are present.
    all_stages_done = all(
        (checkpoint_dir / f"stage_{i}.done").exists() for i in range(n_stages)
    )
    if all_stages_done:
        final_path = checkpoint_dir / "final"
        model.save(str(final_path))
        (results_dir / "DONE").touch()
        print(f"\nTraining complete. Final model: {final_path}")
    else:
        completed = [i for i in range(n_stages)
                     if (checkpoint_dir / f"stage_{i}.done").exists()]
        print(f"\nPartial run: stages done = {completed} / {n_stages}")
    print(f"Total steps this invocation: {total_steps_trained}")
    print(f"Results dir: {results_dir}")


if __name__ == "__main__":
    main()
