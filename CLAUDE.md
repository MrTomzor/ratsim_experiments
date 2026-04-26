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

All methods produce the same JSONL schema, one JSON object per episode:

```json
{"method": "ppo", "rundef": "...", "stage_idx": 0, "seed": 42, "episode_idx": 1, "steps": 300, "total_score": 15.0, "objects_found": 3, "collisions": 1, "termination_reason": "max_steps", "distance_traveled": 450.2, "wall_time_s": 12.3}
```

- **Training**: `results/<run_name>/train_episodes.jsonl` — written by the Gym env itself (see `ratsim_wildfire_gym_env/env.py`'s `episode_log_path` / `run_metadata` kwargs), so PPO and DreamerV3 produce identical schemas for free. `episode_idx` is **cumulative across stages** — on env construction, the env counts existing JSONL lines and offsets from there, so resumed runs keep monotonically increasing indices. With `n_envs>1`, all parallel envs append to the same JSONL: each line carries an `env_idx` field so you can group/dedupe per-env, and `episode_idx` is per-env (i.e. unique within an `env_idx` but not globally).
- **Evaluation**: `results/<run_name>/eval_episodes.jsonl` — written by `test.py` via `make_episode_result()`.
- **DONE marker**: `results/<run_name>/DONE` (empty file) is touched at the end of a successful run. `analyze_run_data.py` warns on any run dir missing it (run crashed or still in progress).

TaskTracker (from the core ratsim repo) is the single source of truth for episode metrics regardless of method.

### Checkpoints

- **PPO** (`train.py`): saves `checkpoints/stage_<i>.zip` after each stage, plus `checkpoints/final.zip`.
- **DreamerV3** (`train_dreamerv3.py`): embodied's rolling `ckpt/latest` pointer is snapshotted into `checkpoints/stage_<i>/` after each stage and `checkpoints/final/` at the end, mirroring PPO's per-stage layout.

### Analysis

`analyze_run_data.py` loads one or more `train_episodes.jsonl` files and emits terminal summaries + PNGs. Accepts any mix of run dirs, parent dirs, and symlinks (recursive, follows symlinks). Validates schema/monotonicity/DONE marker and flags issues.

```bash
# Single run
python analyze_run_data.py results/my_run/

# A whole batch (parent dir, recursive)
python analyze_run_data.py results/batch_20260420/ --out analysis_output/ --rolling 20

# Cherry-picked runs via symlink farm
python analyze_run_data.py symlinks/comparison_A/
```

Needs the sb3 venv (pandas + matplotlib): `~/ratvenv/venv/bin/python analyze_run_data.py ...`.

### Human evaluation

Human control is handled via `/enable_human_control` topic sent to Unity. The core function `ratsim.human_control_test.run_human_session()` manages the sim loop — Python ticks the sim, Unity handles human input, TaskTracker records metrics. test.py imports this for human eval runs.

## File Structure

```
ratsim_experiments/
├── train.py                 # Train PPO / RecurrentPPO on a run definition
├── train_dreamerv3.py       # Train DreamerV3 (separate venv — jax/embodied)
├── test.py                  # Evaluate a method on a run definition (RL, human, etc.)
├── analyze_run_data.py      # Load train_episodes.jsonl(s) → tables + plots
├── overnight_batch.sh       # Example bash queue for long unattended runs
├── rundefs/                 # Named run definitions (YAML)
│   └── *.yaml
├── results/                 # Output directory (gitignored)
│   └── <run_name>/
│       ├── run_config.json or eval_config.json
│       ├── train_episodes.jsonl    # training episodes, written by env
│       ├── eval_episodes.jsonl     # eval episodes, written by test.py
│       ├── DONE                    # empty marker file, touched on clean finish
│       ├── tensorboard/
│       └── checkpoints/
│           ├── stage_0.zip / stage_0/    # PPO: zip; Dreamer: dir
│           ├── ...
│           └── final.zip / final/
├── pyproject.toml
└── .gitignore
```

## Usage

The `def=` argument accepts either a name (looked up in `rundefs/`) or a file path (tab-completable). File paths are sanitized automatically for result folder naming.

```bash
# Train (by name or path)
python train.py def=default_forest_foraging method=ppo name=my_run
python train.py def=rundefs/default_forest_foraging.yaml method=ppo
python train.py def=default_forest_foraging method=recurrent_ppo step_multiplier=2.0

# Evaluate trained model
python test.py def=default_forest_foraging model=results/my_run/checkpoints/final.zip
python test.py def=default_forest_foraging model=results/my_run/checkpoints/final.zip eval_seeds=42,123,456

# Infinite eval (runs seeds 1,2,3,... until Ctrl+C, then prints summary)
python test.py def=default_forest_foraging model=results/my_run/checkpoints/final.zip eval_seeds=inf

# Human evaluation (with real-time factor for smooth visuals)
python test.py def=default_forest_foraging method=human rtf=1.0

# Method config overrides
python train.py def=default_forest_foraging method=ppo method.learning_rate=1e-4
python train.py def=default_forest_foraging method=recurrent_ppo method_config=configs/lstm256.yaml

# Vectorized training (requires RATSIM_UNITY_BIN; spawns n_envs Unity instances on 9100+)
python train.py def=default_forest_foraging method=ppo n_envs=8
python train_dreamerv3.py def=default_forest_foraging n_envs=2 method.jax.platform=cuda

# Two parallel runs on the same box: pass non-overlapping base_port
python train.py def=default_forest_foraging method=ppo n_envs=4 base_port=9100  # uses 9100-9103
python train.py def=default_forest_foraging method=ppo n_envs=4 base_port=9110  # uses 9110-9113
```

Result folders are named `<rundef>_<method>_<YYYYMMDD_HHMMSS>` (e.g., `default_forest_foraging_ppo_20260401_143022`).

## Seeds

- **Eval seeds**: fixed list passed to test.py, shared across all methods for fair comparison. Default: 1-10. Use `eval_seeds=inf` for continuous evaluation.
- **Training seeds (metaseed)**: controls world generation randomness during training. Pass `metaseed=N` to train.py. Eval seeds must never appear in training.
- **Training run seeds**: run the same config multiple times with different metaseeds to get error bars.

## Adding a new method

1. If it uses the Gym env (like RL): add the SB3 class to `METHODS` dict in train.py, add load/predict logic in test.py.
2. If it bypasses the Gym env (like human, frontier): add an `eval_<method>()` function in test.py that manages its own sim loop but records the same JSONL schema.

## Modifiers (planned)

Named config overrides (e.g., `double_tree_density`, `add_walls_1000m_box`) that can be applied on top of world presets at the orchestration layer.

## Unity instance management

Train and test scripts get their Unity ports from
`ratsim.unity_launcher.allocate_unity_instances(n_envs, fresh=...)`. Two tiers:

- **`RATSIM_UNITY_BIN` unset**: launch Unity manually (Editor Play or
  `start_ratsim_headless.sh`) on port 9000. Only `n_envs=1` works; the script
  attaches to the running instance. Trying `n_envs>1` errors out with a
  message pointing at the env var.
- **`RATSIM_UNITY_BIN=/path/to/build`**: scripts auto-spawn Unity on demand.
  `n_envs=1` reuses port 9000 if alive (so debug runs share your interactive
  instance); `n_envs>1` always allocates fresh on ports 9100+. Spawned
  instances die with the parent Python process (via `atexit`); SIGKILL or
  power loss leaves orphans — clean them up with
  `./scripts/stop_ratsim_headless.sh --all`.

See `ratsim/CLAUDE.md` for the full launcher contract.

## Vectorization vs parallel runs

Two orthogonal ways to use multiple Unity instances. Don't conflate them:

**Vectorization (`n_envs=N` within one run)** — one algorithm, N parallel envs.
PPO/RecurrentPPO concatenates rollouts from all N envs into a single batch
each update; DreamerV3 fills its replay buffer N× faster. Single shared
policy, single optimizer. Use this to speed up *one* training run.

```bash
# One PPO run, 8 parallel envs (ports 9100–9107)
python train.py def=default_forest_foraging method=ppo n_envs=8

# One DreamerV3 run, 2 parallel envs (GPU recommended)
python train_dreamerv3.py def=default_forest_foraging n_envs=2 method.jax.platform=cuda
```

**Parallel runs** — N independent training processes, each with its own
policy, optimizer, results dir, and tensorboard. Use this for seed
sweeps / hyperparam sweeps / different methods at the same time. Pass
non-overlapping `base_port` to each run.

```bash
# Two seeds in parallel, each using 4 envs
python train.py def=default_forest_foraging method=ppo n_envs=4 \
    base_port=9100 metaseed=1 name=ppo_seed1 > logs/run1.log 2>&1 &
python train.py def=default_forest_foraging method=ppo n_envs=4 \
    base_port=9110 metaseed=2 name=ppo_seed2 > logs/run2.log 2>&1 &
```

The 10-port gap is just a convention; what matters is the ranges don't
overlap (run A: 9100–9103, run B: 9110–9113). Sanity-check RAM first:
each Unity instance is ~500 MB, so 2 runs × 4 envs ≈ 4 GB just for Unity.

## Headless display lifecycle

`setup_headless_display.sh` installs `xorg-ratsim.service` (Xorg on `:99`),
which auto-starts on boot. To stop or disable it:

```bash
sudo systemctl stop xorg-ratsim          # stop now
sudo systemctl disable xorg-ratsim       # don't auto-start on boot
sudo systemctl enable --now xorg-ratsim  # re-enable later
```

Idle Xorg costs ~20–50 MB RAM and near-zero CPU, so leaving it running is
also fine. Re-run `setup_headless_display.sh` only after NVIDIA driver
changes (the script caches its config path on first run).
