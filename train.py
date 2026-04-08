"""
Train an RL agent on a run definition.

Usage:
    python train.py def=default_forest_foraging method=ppo
    python train.py def=default_forest_foraging method=recurrent_ppo
    python train.py def=default_forest_foraging method=cnn_ppo
    python train.py def=default_forest_foraging method=cnn_recurrent_ppo
    python train.py def=default_forest_foraging method=ppo name=my_run step_multiplier=2.0
    python train.py def=default_forest_foraging method=ppo metaseed=42
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO

from ratsim.config_blender import blend_presets
from ratsim_wildfire_gym_env.env import WildfireGymEnv
from feature_extractors import LidarCnnExtractor


# -- Run definition loading --------------------------------------------------

def load_rundef(name_or_path: str) -> dict:
    """Load a run definition YAML by name or file path."""
    path = Path(name_or_path)
    if path.suffix in (".yaml", ".yml") and path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    # Fall back to looking up by name in rundefs/
    rundef_dir = Path(__file__).parent / "rundefs"
    path = rundef_dir / f"{name_or_path}.yaml"
    if not path.exists():
        available = [f.stem for f in rundef_dir.glob("*.yaml")]
        raise FileNotFoundError(
            f"Run definition '{name_or_path}' not found at {path}\n"
            f"Available: {available}"
        )
    with open(path) as f:
        return yaml.safe_load(f)


# -- Config resolution -------------------------------------------------------

def resolve_world_config(stage: dict) -> dict:
    """Blend world presets and apply overrides for a stage."""
    presets = stage.get("world_presets", ["default"])
    cfg = blend_presets("world", presets)
    overrides = stage.get("world_overrides", {})
    cfg.update(overrides)
    return cfg


def resolve_task_config(rundef: dict, stage: dict | None = None) -> dict:
    """Load and optionally override task config.

    Rundef-level task_overrides are applied first, then stage-level task_overrides
    (if a stage dict is provided) take precedence.
    """
    task_preset = rundef.get("task_preset", "default")
    cfg = blend_presets("task", [task_preset])
    overrides = rundef.get("task_overrides", {})
    cfg.update(overrides)
    if stage is not None:
        stage_overrides = stage.get("task_overrides", {})
        cfg.update(stage_overrides)
    return cfg


def resolve_agent_config(rundef: dict) -> dict:
    """Load agent config from preset."""
    agent_preset = rundef.get("agent_preset", "sphereagent_2d_lidar")
    return blend_presets("agents", [agent_preset])


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


def get_max_episode_steps(rundef: dict, stages: list[dict]) -> int:
    """Return the maximum episode_max_steps across all stages."""
    base_task = blend_presets("task", [rundef.get("task_preset", "default")])
    base_task.update(rundef.get("task_overrides", {}))
    max_steps = base_task.get("episode_max_steps", 300)
    for stage in stages:
        stage_task = dict(base_task)
        stage_task.update(stage.get("task_overrides", {}))
        max_steps = max(max_steps, stage_task.get("episode_max_steps", max_steps))
    return max_steps


# Recurrent PPO preset profiles. Select via method.recurrent_preset=<name>
# or override individual params with method.n_steps=... etc.
RECURRENT_PRESETS = {
    # A) Full-episode unroll — correct LSTM training, slow on CPU
    "full_episode": {
        "n_steps": "auto",      # set to max episode_max_steps across stages
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
}
RECURRENT_PRESET_DEFAULT = "balanced"


def create_model(method_name: str, env, method_config: dict, tb_log_dir: str,
                 max_episode_steps: int | None = None):
    """Create an SB3 model from method name and config."""
    method = METHODS[method_name]
    is_recurrent = method["sb3_class"] is RecurrentPPO

    # Defaults that can be overridden by method_config
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

    # Carry over policy_kwargs defined in the METHODS entry (e.g. CNN extractor).
    if "policy_kwargs" in method:
        import copy
        kwargs["policy_kwargs"] = copy.deepcopy(method["policy_kwargs"])

    # Auto-populate CNN extractor params from the env.
    if "policy_kwargs" in kwargs and "features_extractor_kwargs" in kwargs.get("policy_kwargs", {}):
        ext_kwargs = kwargs["policy_kwargs"]["features_extractor_kwargs"]
        # Unwrap VecEnv (DummyVecEnv/SubprocVecEnv) to reach the actual Gym env.
        unwrapped = env.envs[0].unwrapped if hasattr(env, "envs") else env.unwrapped
        ext_kwargs.setdefault("n_rays", unwrapped.num_lidar_rays)
        ext_kwargs.setdefault("n_channels", unwrapped.num_lidar_channels)
        # Allow CLI overrides (method.cnn_output_dim, method.mlp_output_dim, etc.)
        for pname in ("n_rays", "n_channels", "cnn_output_dim"):
            if pname in method_config:
                ext_kwargs[pname] = int(method_config.pop(pname))

    # For recurrent policies: apply a preset profile, then let method_config override.
    if is_recurrent:
        preset_name = method_config.pop("recurrent_preset", RECURRENT_PRESET_DEFAULT)
        preset = RECURRENT_PRESETS[preset_name]
        print(f"[RecurrentPPO] Using preset '{preset_name}': {preset}")

        for k, v in preset.items():
            if k not in method_config:
                if v == "auto" and max_episode_steps is not None:
                    v = max(2048, max_episode_steps)
                    print(f"[RecurrentPPO] Auto {k}={v} (max episode_max_steps={max_episode_steps})")
                # Deep-merge policy_kwargs from preset into existing policy_kwargs
                if k == "policy_kwargs" and "policy_kwargs" in kwargs:
                    kwargs["policy_kwargs"].update(v)
                else:
                    kwargs[k] = v

    # Deep-merge policy_kwargs from method_config into existing policy_kwargs
    if "policy_kwargs" in method_config and "policy_kwargs" in kwargs:
        kwargs["policy_kwargs"].update(method_config.pop("policy_kwargs"))

    kwargs.update(method_config)

    return method["sb3_class"](**kwargs)


# -- Stage transition resets ---------------------------------------------------

def reset_critic(model):
    """Reinitialize value function weights, keeping the policy (actor) intact.

    Resets: mlp_extractor critic path, value_net head, and for RecurrentPPO
    the lstm_critic and critic projection if they exist.
    """
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

    # MLP extractor critic path
    reinit_module(policy.mlp_extractor.value_net)
    # Final value head (Linear -> 1)
    th.nn.init.orthogonal_(policy.value_net.weight, gain=1.0)
    th.nn.init.constant_(policy.value_net.bias, 0.0)
    # RecurrentPPO: separate critic LSTM and projection
    if hasattr(policy, "lstm_critic") and policy.lstm_critic is not None:
        reinit_module(policy.lstm_critic)
    if hasattr(policy, "critic") and policy.critic is not None:
        reinit_module(policy.critic)

    print("[StageTransition] Critic weights reinitialized")


def reset_optimizer(model):
    """Reset optimizer state (Adam momentum/variance) so stale statistics
    from the previous stage don't interfere with the new distribution."""
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
            distances = self.training_env.env_method("get_distance_traveled")
            pickups = self.training_env.env_method("get_reward_pickups")
            longest_step_distance = self.training_env.env_method("get_longest_step_distance")

            self.logger.record("custom/avg_distance_traveled", np.mean(distances))
            self.logger.record("custom/avg_reward_pickups", np.mean(pickups))
            self.logger.record("custom/longest_step_distance", np.max(longest_step_distance))

        return True


# -- Main ---------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train an RL agent on a run definition.")
    parser.add_argument("overrides", nargs="*", help="key=value overrides")
    return parser.parse_args()


def parse_overrides(override_list: list[str]) -> dict:
    """Parse key=value pairs from the command line."""
    result = {}
    for item in override_list:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}', expected key=value")
        key, value = item.split("=", 1)
        # Try to parse as number/bool
        try:
            value = yaml.safe_load(value)
        except yaml.YAMLError:
            pass
        result[key] = value
    return result


def main():
    args = parse_args()
    overrides = parse_overrides(args.overrides)

    # Required args
    rundef_name = overrides.pop("def", None)
    method_name = overrides.pop("method", "ppo")
    if rundef_name is None:
        print("Usage: python train.py def=<rundef_name> method=<method_name> [overrides...]")
        sys.exit(1)
    rundef_name_clean = Path(rundef_name).stem  # strip path and .yaml extension

    # Optional args
    run_name = overrides.pop("name", f"{rundef_name_clean}_{method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    step_multiplier = float(overrides.pop("step_multiplier", 1.0))
    # If no metaseed provided, generate random number (not a benchmark with specific seed -> want some variability between runs)
    default_metaseed = np.random.randint(0, 10000)
    metaseed = overrides.pop("metaseed", default_metaseed)
    method_config_file = overrides.pop("method_config", None)

    # Load method config from file if specified, then apply remaining overrides
    method_config = {}
    if method_config_file is not None:
        with open(method_config_file) as f:
            method_config = yaml.safe_load(f)

    # Any remaining overrides that start with "method." go into method_config
    for key in list(overrides.keys()):
        if key.startswith("method."):
            method_config[key.removeprefix("method.")] = overrides.pop(key)

    # Load run definition
    rundef = load_rundef(rundef_name)
    stages = rundef["stages"]

    # Resolve configs
    agent_config = resolve_agent_config(rundef)

    # Output directory
    results_dir = Path(__file__).parent / "results" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save the full resolved config for reproducibility
    run_meta = {
        "rundef": rundef_name_clean,
        "method": method_name,
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

    # Train each stage sequentially
    model = None
    total_steps_trained = 0
    max_episode_steps = get_max_episode_steps(rundef, stages)

    for stage_idx, stage in enumerate(stages):
        world_config = resolve_world_config(stage)
        task_config = resolve_task_config(rundef, stage)
        stage_steps = int(stage["steps"] * step_multiplier)

        print(f"\n{'='*60}")
        print(f"Stage {stage_idx + 1}/{len(stages)}: {stage.get('world_presets', ['?'])}")
        print(f"Steps: {stage_steps} (multiplier: {step_multiplier})")
        print(f"{'='*60}\n")

        # Build env with this stage's world config
        def make_env(wc=world_config):
            return WildfireGymEnv(
                worldgen_config=wc,
                agent_config=agent_config,
                sensor_config={},
                action_config={"control_mode": "velocity"},
                task_config=task_config,
                metaworldgen_config={"world_generation_metaseed": metaseed},
            )

        env = make_vec_env(make_env, n_envs=1)

        is_stage_transition = model is not None
        reset_on_stage = rundef.get("reset_on_stage_change", False)

        if not is_stage_transition:
            model = create_model(method_name, env, method_config, tb_log_dir,
                                max_episode_steps=max_episode_steps)
        else:
            # Continue training with new env
            model.set_env(env)

            # Stage transition resets
            if reset_on_stage:
                reset_critic(model)
                reset_optimizer(model)

        model.learn(
            total_timesteps=stage_steps,
            callback=TrainingMetricsCallback(),
            # Reset LR schedule on stage change so it doesn't continue decayed
            reset_num_timesteps=is_stage_transition and reset_on_stage,
        )

        total_steps_trained += stage_steps

        # Save checkpoint after each stage
        checkpoint_path = checkpoint_dir / f"stage_{stage_idx}"
        model.save(str(checkpoint_path))
        print(f"Saved checkpoint: {checkpoint_path}")

        env.close()

    # Save final model
    final_path = checkpoint_dir / "final"
    model.save(str(final_path))
    print(f"\nTraining complete. Final model: {final_path}")
    print(f"Total steps: {total_steps_trained}")
    print(f"Results dir: {results_dir}")


if __name__ == "__main__":
    main()
