"""
Train an RL agent on a run definition.

Usage:
    python train.py def=default_forest_foraging method=ppo
    python train.py def=default_forest_foraging method=recurrent_ppo
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
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from sb3_contrib import RecurrentPPO

from ratsim.config_blender import blend_presets
from ratsim_wildfire_gym_env.env import WildfireGymEnv


# -- Run definition loading --------------------------------------------------

def load_rundef(name: str) -> dict:
    """Load a run definition YAML by name from the rundefs/ directory."""
    rundef_dir = Path(__file__).parent / "rundefs"
    path = rundef_dir / f"{name}.yaml"
    if not path.exists():
        available = [f.stem for f in rundef_dir.glob("*.yaml")]
        raise FileNotFoundError(
            f"Run definition '{name}' not found at {path}\n"
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


def resolve_task_config(rundef: dict) -> dict:
    """Load and optionally override task config."""
    task_preset = rundef.get("task_preset", "default")
    cfg = blend_presets("task", [task_preset])
    overrides = rundef.get("task_overrides", {})
    cfg.update(overrides)
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
}


def create_model(method_name: str, env, method_config: dict, tb_log_dir: str):
    """Create an SB3 model from method name and config."""
    method = METHODS[method_name]

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
    kwargs.update(method_config)

    return method["sb3_class"](**kwargs)


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

    # Optional args
    run_name = overrides.pop("name", f"{rundef_name}_{method_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    step_multiplier = float(overrides.pop("step_multiplier", 1.0))
    metaseed = overrides.pop("metaseed", 1)
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
    task_config = resolve_task_config(rundef)
    agent_config = resolve_agent_config(rundef)

    # Output directory
    results_dir = Path(__file__).parent / "results" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save the full resolved config for reproducibility
    run_meta = {
        "rundef": rundef_name,
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

    for stage_idx, stage in enumerate(stages):
        world_config = resolve_world_config(stage)
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

        if model is None:
            model = create_model(method_name, env, method_config, tb_log_dir)
        else:
            # Continue training with new env
            model.set_env(env)

        model.learn(
            total_timesteps=stage_steps,
            callback=TrainingMetricsCallback(),
            reset_num_timesteps=False,  # keep global step counter across stages
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
