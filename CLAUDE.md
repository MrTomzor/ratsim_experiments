# CLAUDE.md — ratsim_experiments

## Purpose

This repo is the experiment automation layer for the ratsim project. It exists to run reproducible, comparable experiments across multiple methods (RL, frontier exploration, human, VLM, etc.) for paper-ready results. It sits at the top of the dependency stack:

```
ratsim_experiments
    ├── imports ratsim (SDK: presets, config blender, TaskTracker, connector, human_control_test)
    ├── imports ratsim_wildfire_gym_env (Gym env for RL evaluation)
    └── launches ratsim_ros2 (for frontier exploration baseline)
```

## Core Design

### Three layers of configuration

| Layer | Location | Purpose |
|-------|----------|---------|
| **Presets** | `ratsim/config_blender/` (core repo) | Sim vocabulary — world types, agent bodies, task rules. Reusable across all consumers. |
| **Run definitions** | `rundefs/` (this repo) | Named sequences of (world presets + overrides, steps). Referenced by train.py and test.py. |
| **Orchestration** | CLI args + queue files (this repo) | Which rundef × method × seeds to run. Step multipliers, method configs, eval seeds. |

### Run definitions

A run definition (`rundefs/*.yaml`) specifies a sequence of stages, each with world presets and step counts. Agent and task presets are set once for the whole definition. Run definitions are method-agnostic — the same rundef can be used for training (train.py) and evaluation (test.py).

```yaml
agent_preset: sphereagent_2d_lidar
task_preset: default
stages:
  - world_presets: [default]
    world_overrides:
      vegetation/tree1/density: 0.01
    steps: 1_000_000
```

### Method-invariant evaluation

All methods produce the same JSONL output (`results/<run_name>/episodes.jsonl`), one JSON object per episode:

```json
{"method": "ppo", "rundef": "...", "stage_idx": 0, "seed": 42, "episode_idx": 0, "steps": 300, "total_score": 15.0, "objects_found": 3, "collisions": 1, "termination_reason": "max_steps", "distance_traveled": 450.2, "wall_time_s": 12.3}
```

TaskTracker (from the core ratsim repo) is the single source of truth for episode metrics regardless of method.

### Human evaluation

Human control is handled via `/enable_human_control` topic sent to Unity. The core function `ratsim.human_control_test.run_human_session()` manages the sim loop — Python ticks the sim, Unity handles human input, TaskTracker records metrics. test.py imports this for human eval runs.

## File Structure

```
ratsim_experiments/
├── train.py              # Train RL agents on a run definition
├── test.py               # Evaluate any method on a run definition (RL, human, etc.)
├── rundefs/              # Named run definitions (YAML)
│   └── *.yaml
├── results/              # Output directory (gitignored)
│   └── <run_name>/
│       ├── run_config.json or eval_config.json
│       ├── episodes.jsonl
│       ├── tensorboard/
│       └── checkpoints/
├── pyproject.toml
└── .gitignore
```

## Usage

```bash
# Train
python train.py def=default_forest_foraging method=ppo name=my_run
python train.py def=default_forest_foraging method=recurrent_ppo step_multiplier=2.0

# Evaluate trained model
python test.py def=default_forest_foraging model=results/my_run/checkpoints/final.zip
python test.py def=default_forest_foraging model=results/my_run/checkpoints/final.zip eval_seeds=42,123,456

# Human evaluation
python test.py def=default_forest_foraging method=human

# Method config overrides
python train.py def=default_forest_foraging method=ppo method.learning_rate=1e-4
python train.py def=default_forest_foraging method=recurrent_ppo method_config=configs/lstm256.yaml
```

## Seeds

- **Eval seeds**: fixed list passed to test.py, shared across all methods for fair comparison. Default: 42,123,456,789,1337.
- **Training seeds (metaseed)**: controls world generation randomness during training. Pass `metaseed=N` to train.py. Eval seeds must never appear in training.
- **Training run seeds**: run the same config multiple times with different metaseeds to get error bars.

## Adding a new method

1. If it uses the Gym env (like RL): add the SB3 class to `METHODS` dict in train.py, add load/predict logic in test.py.
2. If it bypasses the Gym env (like human, frontier): add an `eval_<method>()` function in test.py that manages its own sim loop but records the same JSONL schema.

## Modifiers (planned)

Named config overrides (e.g., `double_tree_density`, `add_walls_1000m_box`) that can be applied on top of world presets at the orchestration layer.
