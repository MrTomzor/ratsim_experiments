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
| **Experiment defs** | `defs/` (this repo) | Named experiment specs: agent + task + stages, methods × variations × seeds, BFS/DFS. The unit train.py and the scheduler both consume. |
| **Orchestration** | scheduler + machine config | Which experiment to run, on which machine, at what step multiplier. |

### Experiment defs

An experiment def (`defs/*.yaml`) is a single file declaring everything an
experiment needs: agent preset, task preset, world preset(s), the curriculum
of stages, the methods to compare, the variations (override bundles) to test,
seed counts, and BFS/DFS mode. Same file used by ad-hoc training (`train.py
def=<name>`) and the scheduler (`python -m scheduler.scheduler run <name>`).

```yaml
agent_preset: sphereagent_2d_lidar
task_preset: volumetric_exploration_2000_collision_penalty
world_preset: maze_default

total_steps: 10_000_000     # short form: equal stages of total/n
n_stages: 10
mode: bfs

methods:
  - {name: ppo}
  - {name: dreamer}
seeds: 3

variations:                  # optional; default is [{name: baseline}]
  - {name: with_gps}
  - {name: no_gps, agent_preset: sphereagent_2d_lidar_no_gps}
```

Long form: replace `total_steps:` + `n_stages:` with an explicit `stages:`
list (one entry per stage with its own `world_preset` + `steps`). See
`scheduler/README.md` for the full schema and the override resolution rules.

For one-off training without a saved def, `train.py` accepts the same fields
inline on the CLI — see the **Usage** section below. Defs are for things
you'll re-run, share, or feed to the scheduler.

### Method-invariant evaluation

All methods produce the same JSONL schema, one JSON object per episode:

```json
{"method": "ppo", "rundef": "...", "stage_idx": 0, "seed": 42, "episode_idx": 1, "steps": 300, "total_score": 15.0, "objects_found": 3, "collisions": 1, "termination_reason": "max_steps", "distance_traveled": 450.2, "wall_time_s": 12.3}
```

- **Training**: `results/<run_name>/train_episodes.jsonl` — written by the Gym env itself (see `ratsim_wildfire_gym_env/env.py`'s `episode_log_path` / `run_metadata` kwargs), so PPO and DreamerV3 produce identical schemas for free. The `run_metadata` carries `exp_id`, `variation`, `method`, `seed`, `stage_idx`, `env_idx` so per-line filtering / grouping is trivial. `episode_idx` is **cumulative across stages** — on env construction, the env counts existing JSONL lines and offsets from there, so resumed runs keep monotonically increasing indices. With `n_envs>1`, all parallel envs append to the same JSONL: each line carries an `env_idx` field so you can group/dedupe per-env, and `episode_idx` is per-env (i.e. unique within an `env_idx` but not globally).
- **Evaluation**: `results/<run_name>/eval_episodes.jsonl` — written by `test.py` via `make_episode_result()`.
- **DONE marker**: `results/<run_name>/DONE` (empty file) is touched at the end of a successful run. `analyze_run_data.py` warns on any run dir missing it (run crashed or still in progress).

TaskTracker (from the core ratsim repo) is the single source of truth for episode metrics regardless of method.

### Checkpoints

- **PPO** (`train.py`): saves `checkpoints/stage_<i>.zip` after each stage, plus `checkpoints/final.zip`.
- **DreamerV3** (`train_dreamerv3.py`): embodied's rolling `ckpt/latest` pointer is snapshotted into `checkpoints/stage_<i>/` after each stage and `checkpoints/final/` at the end, mirroring PPO's per-stage layout.
- **Per-stage marker**: `checkpoints/stage_<i>.done` is touched after the checkpoint write/copy fully completes, so the scheduler can distinguish "stage finished" from "process killed mid-save". Top-level `DONE` is only touched when every stage has its `.done` marker — a partial / resumed run won't falsely claim full completion.

### Resuming a run

Both training scripts accept `start_stage=K end_stage=K+1` (end exclusive) to run a single stage of an existing run, loading the previous stage's checkpoint:

```bash
python train.py def=method_compare method=ppo run_folder=my_run start_stage=3 end_stage=4
python train_dreamerv3.py def=method_compare run_folder=my_run start_stage=3 end_stage=4
```

This is the primitive the scheduler uses to interleave stages across methods × variations × seeds. PPO loads `checkpoints/stage_<K-1>.zip`; Dreamer relies on embodied's rolling `ckpt/latest` in the same `dreamer_logdir/` and re-targets `run.steps` to the cumulative end of stage K.

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

### Experiment tracking (Weights & Biases)

`wandb_integration.py` adds W&B as an **extra sink alongside TensorBoard**, not a
replacement. Both frameworks fan one metric stream out to a list of outputs, so
nothing about *what* is recorded changes — every existing `logger.record()` call
site (`custom/*`, `timing/*`, `rollout/*`, `train/*`) is mirrored untouched. TB
keeps writing (12 KB–5 MB per run, noise next to a 225 MB replay dir) and stays
the local fallback. `train_episodes.jsonl` is in neither path, so the
method-invariant schema and every analysis script are unaffected.

Off by default. Opt in per-run or globally:

```bash
python train.py def=smoke_test method=ppo wandb=1
python train.py def=smoke_test method=ppo wandb=1 wandb_project=ratsim-dev
export RATSIM_WANDB=1     # shell rc locally, rci_env.sh on the cluster
```

Credentials come from `WANDB_API_KEY` or `~/.netrc`. **Never put the key in
`rci_env.sh`** — that file is tracked in git.

**Run identity is the part that matters.** The scheduler dispatches one *process
per stage* (`start_stage=K end_stage=K+1`), so a 10-stage run is 10 invocations.
A plain `wandb.init()` would give 10 disconnected runs per seed on top of the
7-wide seed fan-out. Instead the W&B run id is generated once and persisted to
`<run_dir>/wandb_id.txt`, and later stages reopen it with `resume="allow"`. The
id is stored on disk rather than hashed from the run path deliberately: a path
hash would silently resume a *stale* W&B run if the results dir were deleted and
the experiment re-run, appending new data onto old curves.

This works because both frameworks already carry a monotone global step across
stages — SB3 runs `reset_num_timesteps=False` and restores `num_timesteps` from
the checkpoint zip; Dreamer re-targets `run.steps` and restores from
`ckpt/latest`. Verified locally: stage 0 ended at `total_timesteps=6144`, and
stage 1 in a *separate process* continued to 12288 under the same run id.

Grouping, so 7-wide sweeps collapse into comparable charts rather than 7
unrelated lines:

| W&B field | Value |
|---|---|
| `project` | `ratsim` (override with `wandb_project=`) |
| `group` | `exp_id` |
| `job_type` | method (ppo / recurrent_ppo / dreamer) |
| `name` | run leaf (`<variation>__<method>__seed<n>`) |
| `tags` | method, variation, machine (`rci` under SLURM, else hostname) |
| `config` | the existing `run_meta` / `run_config.json` payload |

W&B state is written to `<run_dir>/wandb/` (via `dir=`), **not** `$TMPDIR` —
`rci_env.sh` makes that per-job and wipes it on job exit.

RCI compute nodes have full outbound HTTPS (measured on a07: no proxy, DNS
resolves, `curl_exit=0` against `api.wandb.ai`), so runs stream live and the
`WANDB_MODE=offline` + `wandb sync` fallback is not needed there.

Both frameworks are wired, each through its own logger:

| Path | Hook | Where |
|---|---|---|
| SB3 (ppo, recurrent_ppo) | `KVWriter` appended to `model.logger.output_formats` | `make_sb3_callback` |
| DreamerV3 | an `elements.logger` output appended in a local `make_logger` | `make_elements_output` |

The SB3 writer attaches in `_on_training_start`, not at construction: SB3 builds
the real logger inside `learn()` (`_setup_learn` → `configure_logger`), so
anything attached earlier is discarded.

`train_dreamerv3.py` carries its **own copy** of `make_logger` instead of
importing `dreamerv3.main.make_logger`. `~/git/dreamerv3` is a pristine upstream
clone and upstream's version builds its `WandBOutput` with `name=` only — no
project, group, job_type or config, i.e. exactly the run identity the 7-wide
one-process-per-stage layout needs. Copying ~15 lines beats carrying a fork.

⚠️ **Do not "simplify" `make_elements_output` into a subclass of
`elements.logger.WandBOutput`.** Its image branch is broken upstream: it reads
`value.shape[3]` on a 2/3-D array (copy-pasted from the 4-D video branch below
it), so *any* image summary raises `IndexError` — verified, both the 2-D and 3-D
paths raise. Inheriting `__call__` would turn the first Dreamer report containing
an image into a crashed training run. Our version fixes the index and leaves
images in the H×W×C layout `wandb.Image` documents.

Verified end-to-end on both paths, each run as **two separate processes** to
mirror the scheduler:

| Path | Stage 0 | Stage 1 (separate process) |
|---|---|---|
| PPO | `total_timesteps=6144` | continued to `12288`, same run id |
| DreamerV3 | — | embodied step counter 870 → 1880, `resumed=True` online |

Unrelated gotcha that will bite you when re-running a stage by hand: the client
socket leaves the Unity port in `TIME_WAIT` (~60 s on Linux), while
`_wait_port_bindable` defaults to 20 s, so a back-to-back invocation on the same
`base_port` fails with "port N still in use after waiting". Pre-existing, and
already suspected in that function's own docstring. Wait for
`ratsim.unity_launcher._port_bindable(port)` before relaunching.

### Memory ablation (DreamerV3)

Test-time probe for whether a trained DreamerV3 agent is actually using its
recurrent (RSSM) memory. Mechanism: force `is_first=True` into the obs dict
on every policy step. The agent was trained to handle this signal at episode
boundaries, so it resets its RSSM carry the same way it learned to —
**in-distribution amnesia**, not OOD zeroing of the hidden state.

Caveat: this is an eval-time intervention, not a like-for-like comparison
with a memoryless agent. Strong evidence if performance is unchanged
(memory wasn't load-bearing); weaker evidence in the other direction (the
ablated policy could degrade for reasons unrelated to information loss).
For a clean "memory matters" claim, train a feedforward baseline.

DreamerV3-only: `dreamerv3.Agent.policy()` always samples from the action
distribution — there's no separate argmax/deterministic path — so the
`is_first` mechanism is the cleanest amnesia we have. SB3 paths don't
expose anything analogous and are skipped by the analyzer dispatcher.

Three entry points, all sharing the same mechanism:

| Use case | Script | Flag |
|---|---|---|
| Quick check on one checkpoint | `test_dreamerv3.py` | `ablate-memory=1` |
| One scheduler run | `eval_one_run_dreamer.py` | `--ablate-memory` |
| All dreamer runs in an experiment | `analyze_experiment.py --run-eval N` | `--ablate-memory` |

```bash
# Single checkpoint, full eval-style sweep over seeds. Auto-suffixes the
# results dir with `_ablated` so baseline + ablated runs land separately.
python test_dreamerv3.py def=houses_volex_1m \
    model=results/<run>/checkpoints/stage_6 ablate-memory=1

# Whole experiment. Writes eval_episodes_ablated.jsonl alongside the baseline
# eval_episodes.jsonl in each run dir; produces eval_<metric>_ablation.png
# paired-bar charts (baseline solid, ablated hatched) under analysis/.
python analyze_experiment.py gps_ablation_5house --run-eval 10 --ablate-memory

# Replot only — without --run-eval, --ablate-memory just re-renders from
# any cached eval_episodes_ablated.jsonl files on disk.
python analyze_experiment.py gps_ablation_5house --ablate-memory
```

Workflow for a paired comparison: run `--run-eval N` once for the baseline,
then again with `--ablate-memory`. Both JSONLs persist; analyzer emits both
the standard `eval_<metric>.png` and the comparison `eval_<metric>_ablation.png`.

### Human evaluation

Human control is handled via `/enable_human_control` topic sent to Unity. The core function `ratsim.human_control_test.run_human_session()` manages the sim loop — Python ticks the sim, Unity handles human input, TaskTracker records metrics. test.py imports this for human eval runs.

## File Structure

```
ratsim_experiments/
├── train.py                 # Train PPO / RecurrentPPO on an experiment def
├── train_dreamerv3.py       # Train DreamerV3 (separate venv — jax/embodied)
├── test.py                  # Evaluate a method on a run definition (RL, human, etc.)
├── analyze_run_data.py      # Load train_episodes.jsonl(s) → tables + plots
├── analyze_experiment.py    # Scheduler-experiment analyzer: train curves +
│                            #   eval bar charts + memory-ablation plot
├── eval_one_run.py          # Per-run eval helper for SB3 (PPO / RecurrentPPO)
├── eval_one_run_dreamer.py  # Per-run eval helper for DreamerV3 (embodied venv)
├── overnight_batch.sh       # Example bash queue for long unattended runs
├── experiment_defs.py       # Schema + loader for defs/*.yaml (shared by train + scheduler)
├── defs/                    # Experiment definitions (YAML)
│   └── *.yaml
├── scheduler/               # Multi-run orchestrator (BFS/DFS, resume, machine configs)
│   ├── scheduler.py
│   ├── config.py
│   └── machines/*.yaml
├── results/                 # Output directory (gitignored)
│   ├── <run_name>/                       # Ad-hoc training runs
│   └── experiments/<exp_id>/             # Scheduler-driven experiments
│       ├── experiment.yaml               # snapshot of the def
│       ├── state.json                    # mutable: pids, failures
│       ├── DONE
│       └── runs/<variation>__<method>__seed<i>/
│           └── ... (same layout as ad-hoc)
├── pyproject.toml
└── .gitignore
```

## Usage

Two ways to specify what to train on: a saved experiment def (`def=<name>`), or inline CLI args (no `def=`). Both go through the same `ExperimentDef` schema internally; the inline form just builds a single-method, single-variation def from the CLI.

```bash
# --- With a saved def (lookup by name in defs/, or pass a path) ---
python train.py def=method_compare method=ppo run_folder=my_run
python train.py def=defs/method_compare.yaml method=ppo
python train.py def=method_compare method=ppo step_multiplier=2.0
python train.py def=gps_ablation method=ppo variation=no_gps

# --- Inline (no def file — for quick ad-hoc training) ---
# Minimal: world and total_steps are required; agent / task / n_stages
# default to sphereagent_2d_lidar / default / 1. (CLI keys `world`/`task`/
# `agent` map to YAML schema `world_preset`/`task_preset`/`agent_preset`.)
python train.py method=ppo world=maze_default total_steps=100_000

# Full inline:
python train.py method=ppo \
    agent=sphereagent_2d_lidar \
    task=volumetric_exploration_2000_collision_penalty \
    world=maze_default \
    total_steps=1_000_000 n_stages=10 metaseed=42

# Dreamer (inline mode infers method=dreamer):
python train_dreamerv3.py world=maze_default total_steps=500_000 n_stages=5

# --- Evaluate trained model (test.py still uses the old rundef path; needs porting) ---
python test.py def=default_forest_foraging model=results/my_run/checkpoints/final.zip
python test.py def=default_forest_foraging method=human rtf=1.0

# Method config overrides (work with both saved-def and inline)
python train.py def=method_compare method=ppo method.learning_rate=1e-4
python train.py def=method_compare method=recurrent_ppo method_config=configs/lstm256.yaml

# Vectorized training (requires RATSIM_UNITY_BIN; spawns n_envs Unity instances on 9100+)
python train.py def=method_compare method=ppo n_envs=8
python train_dreamerv3.py def=method_compare n_envs=2  # CUDA by default; use method.jax.platform=cpu

# Two parallel runs on the same box: pass non-overlapping base_port
python train.py def=method_compare method=ppo n_envs=4 base_port=9100  # 9100-9103
python train.py def=method_compare method=ppo n_envs=4 base_port=9110  # 9110-9113

# Or just let the scheduler do all of this for you:
python -m scheduler.scheduler run method_compare
```

Ad-hoc result folders are named `<exp_id>_<variation>_<method>_<YYYYMMDD_HHMMSS>` (inline mode uses `cli_run` as the exp_id). Scheduler-driven runs go to `results/experiments/<exp_id>/runs/<variation>__<method>__seed<n>/`.

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
  `./scripts/stop_ratsim_headless.sh --all` (well-behaved case, uses
  pidfiles) or `./kill_all_unity.sh` (sledgehammer; matches by command-line
  pattern, doesn't trust pidfiles, and also accepts `-9` for SIGKILL).

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

# One DreamerV3 run, 2 parallel envs (CUDA is default)
python train_dreamerv3.py def=default_forest_foraging n_envs=2
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
