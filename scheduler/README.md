# Scheduler

Runs an *experiment* — a batch of (variation × method × seed) training runs
sharing the same curriculum of stages — in either DFS (one run at a time, all
stages) or BFS (one stage of every run before the next stage) order. Resumable:
kill and restart with the same command, it picks up from per-stage `.done`
markers on disk.

## Quick start

```bash
# 0. (one time, in ~/.bashrc) point the scheduler at your venvs
export PPO_PYTHON_PATH=/home/tom/ratvenv/venv/bin/python
export DREAMER_PYTHON_PATH=/home/tom/ratvenv/dreamer_venv/bin/python

# 1. browse the example experiment defs
ls defs/
#   bptt_length.yaml
#   gps_ablation.yaml
#   method_compare.yaml
#   openfield_to_houses_curriculum.yaml
#   volex_reward_sweep.yaml

# 2. run / resume — uses scheduler/machines/default.yaml (CPU only)
cd ratsim_experiments
python scheduler_run.py method_compare

# 2b. on a GPU box, point at the gpu_example config:
python scheduler_run.py method_compare --machine gpu_example
# or persist via shell:
export RATSIM_SCHEDULER_MACHINE=gpu_example

# 2c. smoke test: 1% of all step counts
python scheduler_run.py method_compare --step-multiplier 0.01

# 3. status, in another terminal
python scheduler_status.py method_compare
```

The `run` command is idempotent: stop it (Ctrl-C), restart with the same
command. The scheduler scans `checkpoints/stage_<i>.done` markers under each
run dir and dispatches whatever's missing next.

## Folder layout

```
ratsim_experiments/
├── defs/                                   ← experiment specs (you author here)
│   └── method_compare.yaml
├── results/
│   └── experiments/
│       └── <exp_id>/                       ← created on first run
│           ├── experiment.yaml             — snapshot of the def, write-once
│           ├── state.json                  — running pids, failure log
│           ├── DONE                        — touched once every run completes
│           └── runs/
│               └── <variation>__<method>__seed<i>/
│                   ├── checkpoints/
│                   │   ├── stage_0.zip          (PPO) or stage_0/ (Dreamer)
│                   │   ├── stage_0.done         — sibling marker, source of truth
│                   │   └── ...
│                   ├── train_episodes.jsonl
│                   ├── tensorboard/
│                   ├── run_config.json
│                   └── scheduler_logs/
│                       └── stage_<i>_<timestamp>.log
```

## Experiment def format

A single YAML file in `defs/<exp_id>.yaml` declares everything:

```yaml
agent_preset: sphereagent_2d_lidar              # default, may be string or list
task_preset: volumetric_exploration_2000_collision_penalty
world_preset: maze_default                      # default world for stages

# Either short form (single-world experiments):
total_steps: 10_000_000
n_stages: 10                                    # → 10 equal stages of 1M each

# OR long form (curriculum):
# stages:
#   - {world_preset: easy_maze,  steps: 1_000_000}
#   - {world_preset: hard_maze,  steps: 5_000_000}

mode: bfs                                       # bfs | dfs

methods:
  - name: ppo                                   # 3 seeds (from `seeds:` below)
  - name: dreamer
  - name: recurrent_ppo
    n_seeds: 1                                  # override per method
seeds: 3

# Optional. Default: [{name: baseline}]. Each variation overrides experiment-
# level presets / method args.
variations:
  - name: with_gps                              # baseline — no overrides
  - name: no_gps
    agent_preset: sphereagent_2d_lidar_no_gps   # full preset swap
  - name: bptt_512
    method_args:                                # method.X=Y for SB3 or Dreamer
      n_steps: 512
      batch_size: 512
  - name: zero_volex
    task_preset:                                # preset list = compose overlays
      - volumetric_exploration_2000_collision_penalty
      - volex_zero_overlay
```

### Override resolution

Per (variation, stage), resolved at dispatch time:

| Layer        | Source (highest precedence first)                                  |
|--------------|--------------------------------------------------------------------|
| agent_preset | variation.agent_preset → exp.agent_preset                          |
| task_preset  | variation.task_preset  → exp.task_preset                           |
| world_preset | stage.world_preset → variation.world_preset → exp.world_preset     |
| method args  | CLI `method.X=Y` → method_config file → variation.method_args      |

Each preset field is a **list**, blended via `blend_presets()`. The blender does
shallow top-level merge (later list entries override earlier ones), so an
"overlay" preset that only respecifies one nested block (e.g. just
`volumetric_exploration_settings:`) cleanly overrides that block while leaving
the rest of the base preset intact.

## Two yaml files in flight

### `defs/<exp_id>.yaml` — *what* to run (machine-agnostic)

The experiment def, authored by you. Snapshotted into
`results/experiments/<exp_id>/experiment.yaml` on first dispatch for
reproducibility.

### Machine config — *how this box runs things*

Declares total resource capacity (`gpu`, `cpu_slot`, ...) and per-method
resource needs + device args. Same def works on any machine — actual
concurrency is determined by which machine config is in effect.

Two configs ship in `scheduler/machines/`:

  * **`default.yaml`** — CPU-only, used when no override is given. Runs
    everything (PPO / RecurrentPPO / Dreamer) through `cpu_slot` with
    CPU device args. Adjust `cpu_slot` capacity for your laptop.
  * **`gpu_example.yaml`** — one GPU + several cpu_slots. Plain PPO uses
    cpu_slot, RecurrentPPO and Dreamer both contend for the gpu slot.

Selection precedence:

  1. `--machine <name|path>` CLI flag, if given
  2. `$RATSIM_SCHEDULER_MACHINE` env var
  3. `scheduler/machines/default.yaml`

A bare name like `gpu_example` is resolved against `scheduler/machines/`;
anything containing a slash or ending in `.yaml`/`.yml` is treated as a
direct path. Drop additional configs into `scheduler/machines/` and
reference them by name — they're gitignored unless explicitly tracked.

## How resume works

There's no scheduler-managed progress state. The training scripts write
`checkpoints/stage_<i>.done` after each stage's checkpoint is fully saved.
The scheduler:

1. On startup, kills any pids recorded in `state.json["running"]` from a
   previous invocation that may still be alive (e.g. scheduler crashed but
   child kept going), then clears the list.
2. For each run × stage_idx, scans `.done` markers to find what's not done.
3. Dispatches in BFS or DFS order, skipping anything whose `.done` exists.

Half-saved checkpoints are not a problem — the marker is only touched after
the save returns successfully, so a killed-mid-save stage simply isn't
considered done and gets re-run from the previous stage's checkpoint.

## Concurrency model

Each method profile declares `needs: {<resource>: <count>}`. The scheduler
keeps a running tally of reserved resources and dispatches a candidate iff
its needs fit in the remaining capacity. With:

```yaml
resources: {gpu: 1, cpu_slot: 4}
method_profiles:
  ppo:           {needs: {cpu_slot: 1}, ...}
  recurrent_ppo: {needs: {gpu: 1},      ...}
  dreamer:       {needs: {gpu: 1},      ...}
```

the box runs up to 4 PPOs in parallel and at most one of {recurrent_ppo,
dreamer}, plus PPOs alongside that GPU job. To restrict more aggressively
on a smaller box, lower `cpu_slot` capacity.

## Unity ports

Every dispatch gets a fresh non-overlapping port window starting at 9100,
stepping by 10 (matching the convention documented in
`ratsim_experiments/CLAUDE.md`). Released windows are reused. The scheduler
always passes `base_port=` so `n_envs=1` jobs spawn fresh too — they don't
attach to the persistent :9000 instance.

This means **`RATSIM_UNITY_BIN` must be set** for any scheduler-driven run
(otherwise auto-spawn fails). Manual `start_ratsim_headless.sh` is only for
attaching to a single interactive run.

## CLI reference

```
python scheduler_run.py    <exp_id_or_path> [--machine <name|path>] [--step-multiplier <x>]
python scheduler_status.py <exp_id_or_path>
```

`<exp_id_or_path>` accepts either a path (absolute or relative) or a bare
exp_id (resolved against `defs/<id>.yaml`). For tab-completion in the shell,
type the path form: `python scheduler_run.py defs/met<TAB>` →
`python scheduler_run.py defs/method_compare.yaml`.

`--machine` falls back to `$RATSIM_SCHEDULER_MACHINE` if unset, and to
`default.yaml` if both are unset. `--step-multiplier` overrides the def's
`step_multiplier:` (default 1.0) — use 0.01 for smoke tests, but note that
resuming with a different multiplier than the original is a footgun: resumed
stages target the new step counts but load checkpoints trained at the old
counts.

Equivalent module-style invocations:
`python -m scheduler.scheduler run <exp>` and
`python -m scheduler.scheduler status <exp>`. The two wrapper scripts are
just thin shims that skip the `run` / `status` subcommand.

## Required presets for the example defs

Most of the bundled defs use existing presets in `ratsim/config_blender/`.
Two need new files you'll have to write:

### For `gps_ablation.yaml`

`ratsim/ratsim/config_blender/agents_presets/sphereagent_2d_lidar_no_gps.yaml`
— copy of `sphereagent_2d_lidar.yaml` with `relative_pose` and `compass`
sensors removed:

```yaml
prefab_name: SphereAgent
name_prefix: rat1
sensors:
  - name: lidar2d
    maxRange: 20.0
    angleStartDeg: -90
    angleEndDeg: 90
    angleIncrementDeg: 10
    occlusionRegion: none
    occlusionDistance: 0.5
    semanticSet: reward_and_boundary_only
  - name: odom
  - name: collision
actuators: velocity
velocity/steeringBias: 0.0
velocity/blockLeftTurn: false
velocity/blockRightTurn: false
max_health: 100.0
```

### For `volex_reward_sweep.yaml`

Two task overlay files. Each contains the FULL
`volumetric_exploration_settings:` block (because the blender does shallow
merge); other top-level blocks (`foraging_settings`, `collision_settings`,
`termination_settings`) are inherited from the baseline preset.

`ratsim/ratsim/config_blender/task_presets/volex_zero_overlay.yaml`:

```yaml
volumetric_exploration_settings:
  reward_per_m2: 0.0
  grid_resolution: 1
  visualize: false
  debug: 0
  debug_every: 10
```

`ratsim/ratsim/config_blender/task_presets/volex_double_overlay.yaml`:

```yaml
volumetric_exploration_settings:
  reward_per_m2: 0.02
  grid_resolution: 1
  visualize: false
  debug: 0
  debug_every: 10
```
