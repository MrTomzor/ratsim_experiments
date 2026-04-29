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
#   bptt_length.yaml                       gps_ablation_5house.yaml
#   compare_5houses.yaml                   gps_ablation_maze_default.yaml
#   compare_loopymaze.yaml                 method_compare.yaml
#   dreamers_maze_smoke.yaml               openfield_to_houses_curriculum.yaml
#   smoke_test.yaml                        volex_reward_sweep.yaml

# 2. run / resume — uses scheduler/machines/default.yaml (CPU only)
cd ratsim_experiments
python scheduler_run.py method_compare

# 2b. on a GPU box, point at the gpu_example config:
python scheduler_run.py method_compare --machine gpu_example
# or persist via shell:
export RATSIM_SCHEDULER_MACHINE=gpu_example

# 2c. smoke test: 1% of all step counts
python scheduler_run.py method_compare --step-multiplier 0.01

# 2d. wipe and start fresh (instead of resuming)
python scheduler_run.py method_compare --restart

# 3. status, in another terminal
python scheduler_status.py method_compare              # one-shot
python scheduler_status.py method_compare --watch      # live, refresh every 2s
python scheduler_status.py method_compare --watch 5    # refresh every 5s
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

### Skipping the def for ad-hoc runs

The scheduler always loads from `defs/*.yaml`, but `train.py` (and
`train_dreamerv3.py`) accept the same fields inline on the CLI when you don't
want to save a file:

```bash
python train.py method=ppo world_preset=maze_default total_steps=100_000
python train.py method=ppo agent_preset=sphereagent_2d_lidar \
    task_preset=volumetric_exploration_2000_collision_penalty \
    world_preset=maze_default total_steps=1_000_000 n_stages=10 metaseed=42
python train_dreamerv3.py world_preset=maze_default total_steps=500_000 n_stages=5
```

Defaults for inline mode: `agent_preset=sphereagent_2d_lidar`,
`task_preset=default`, `n_stages=1`. `world_preset` and `total_steps` are
required. For curricula or variation sweeps, write a def — that's exactly
the case where saving the file pays off.

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

## RAM watchdog (max_ram_gb)

Each method profile can set an optional `max_ram_gb`. If the dispatched
process's *process tree* RSS (the train process + all descendants, including
Unity envs) exceeds this, the scheduler SIGTERMs the job and re-dispatches
it. The job's last in-stage checkpoint is what gets resumed from — for
dreamer that's `dreamer_logdir/ckpt/latest`, written periodically by
embodied (~every 10 min by default), so at most that much progress is lost
per kill.

```yaml
method_profiles:
  dreamer:
    needs: {gpu: 1, cpu_slot: 1}
    n_envs: 1
    max_ram_gb: 30      # ← SIGTERM if RSS exceeds 30 GB
```

**RAM-kills don't count toward `MAX_CONSECUTIVE_FAILURES`.** Otherwise dreamer
(which has a known leak — see `/home/tom/dreamer_crash_summary.md`) would get
blocked after 2 OOMs in a long stage. A real crash (segfault, traceback,
non-RAM-kill nonzero exit) still counts normally. RAM-kills also reset any
prior failure counter for that stage.

`psutil` is a soft dep — if a profile sets `max_ram_gb` but psutil isn't
importable, the watchdog stays inactive and a warning is printed at
startup. `pip install psutil` in the SB3 venv to enable.

The default machine config doesn't set `max_ram_gb` anywhere; only
`gpu_example.yaml` enables it for dreamer.

## Vectorization (n_envs)

Each method profile also declares `n_envs:` (default 1) — the number of
parallel Unity envs each dispatched job spawns. Lives on the *machine*
profile, not the experiment def, because "how many envs make sense"
depends on the box (cores, RAM, GPU), not the experiment. The scheduler
passes `n_envs=<N>` to train.py at dispatch time.

```yaml
method_profiles:
  ppo:
    needs: {cpu_slot: 4}      # ← bump in lockstep with n_envs
    n_envs: 4
  dreamer:
    needs: {gpu: 1, cpu_slot: 2}
    n_envs: 2
```

**`needs` does not auto-track `n_envs`** — you have to bump them together.
Each Unity env is roughly one `cpu_slot` of CPU work for the sim side, plus
the policy/learning compute. If you set `n_envs: 4` but leave
`needs: {cpu_slot: 1}`, the scheduler will run multiple of these in
parallel and oversubscribe the box.

`n_envs` is in `RESERVED_ARGS` — putting it in `args:` of the profile or in
def-level `common_args:` is ignored with a warning. The machine profile is
the only place to set it for scheduler-driven runs.

**Hard cap**: `n_envs ≤ 10` per job (each dispatch gets a 10-wide Unity
port window starting at `base_port`). Validation catches this at startup.

For inline ad-hoc training (`python train.py method=ppo ... n_envs=8`),
just pass `n_envs=N` on the CLI — the train scripts default it to 1.

## Unity ports

Every dispatch gets a fresh non-overlapping port window starting at 9100,
stepping by 10 (matching the convention documented in
`ratsim_experiments/CLAUDE.md`). Released windows are reused. The scheduler
always passes `base_port=` so `n_envs=1` jobs spawn fresh too — they don't
attach to the persistent :9000 instance.

This means **`RATSIM_UNITY_BIN` must be set** for any scheduler-driven run
(otherwise auto-spawn fails). Manual `start_ratsim_headless.sh` is only for
attaching to a single interactive run.

### Watching one training instance live (`--use-port-9000`)

Pass `--use-port-9000` to opt port 9000 into the allocator as a single-slot
port for one n_envs=1 dispatch at a time:

```bash
# 1. Manually launch Unity on 9000 (with the headless display setup so you
#    can VNC into :99 and toggle the camera to follow the agent)
./start_ratsim_headless.sh /path/to/build

# 2. Run the scheduler with the flag
python scheduler_run.py method_compare --use-port-9000
```

Behavior:
- The slot is only handed out to **n_envs=1** dispatches and only when
  Unity is **actually alive on 9000** (TCP probe at dispatch time). If
  Unity isn't running there yet, the scheduler dispatches to 9100+ as
  usual; start Unity and the next eligible dispatch will pick up the slot.
- At most one job uses the slot at a time. Other parallel jobs (including
  multi-env ones) still allocate fresh 9100+ windows.
- The dispatched job's `train.py` gets no `base_port=` arg, so its
  `allocate_unity_instances(n_envs=1)` call falls through to the
  attach-or-spawn-on-9000 path — attaching to your manually-launched
  instance.

## Cleaning up zombie Unity processes

If the scheduler dies ungracefully (kernel OOM-kill, SIGKILL, terminal
closed without Ctrl-C), Unity children may outlive it. Symptoms: `state.json`
lists pids that are dead but Unity is still pinning ports / RAM, or the next
scheduler run can't bind its port window.

```bash
./kill_all_unity.sh              # SIGTERM matches; basename of $RATSIM_UNITY_BIN or 'SARBench'
./kill_all_unity.sh -9           # SIGKILL (use if SIGTERM didn't take)
./kill_all_unity.sh -p MyBuild   # custom command-line pattern
./kill_all_unity.sh -n           # dry-run — show what would be killed
```

Matches by command-line pattern (so it catches both the Unity binary and
its `start_ratsim_headless.sh` launcher) and cleans up stale `/tmp/ratsim_*.pid`
files for already-dead pids. For the well-behaved single-instance case use
`stop_ratsim_headless.sh --all` instead.

## CLI reference

```
python scheduler_run.py    <exp_id_or_path> [--machine <name|path>]
                                            [--step-multiplier <x>]
                                            [--restart]
python scheduler_status.py <exp_id_or_path> [--watch [SECONDS]] [--compact]
```

`<exp_id_or_path>` accepts either a path (absolute or relative) or a bare
exp_id (resolved against `defs/<id>.yaml`). For tab-completion in the shell,
type the path form: `python scheduler_run.py defs/met<TAB>` →
`python scheduler_run.py defs/method_compare.yaml`.

**`run` flags:**
- `--machine` falls back to `$RATSIM_SCHEDULER_MACHINE` if unset, and to
  `default.yaml` if both are unset.
- `--step-multiplier` overrides the def's `step_multiplier:` (default 1.0)
  — use 0.01 for smoke tests, but note that resuming with a different
  multiplier than the original is a footgun: resumed stages target the new
  step counts but load checkpoints trained at the old counts.
- `--restart` wipes `results/experiments/<exp_id>/` before starting,
  equivalent to `rm -rf` + run. Default is to resume from `.done` markers.

**`status` flags:**
- `--watch [SECONDS]` clears the screen and re-prints status every SECONDS
  (default 2). Auto-enables `--compact`. Ctrl-C to exit.
- `--compact` hides the failed-runs list (the list pollutes the screen on
  every refresh in watch mode). One-shot mode shows the last 10 failures
  with log paths so you can scroll back.

Equivalent module-style invocations:
`python -m scheduler.scheduler run <exp>` and
`python -m scheduler.scheduler status <exp>`. The two wrapper scripts are
just thin shims that skip the `run` / `status` subcommand. `--watch` is only
on the wrapper (`scheduler_status.py`); the module form is one-shot.

## Status output

`scheduler_status.py` reads `state.json` + `.done` markers + each run's
`train_episodes.jsonl` and prints, in order:

1. **Header**: exp_id, source, n_stages, mode, step_multiplier, started-at,
   last-activity-at (with relative `(Xm ago)` for the last dispatch/reap).
2. **Per-run progress bar**: one row per `<variation>__<method>__seed<i>`,
   with stage-completion as `███···` blocks, `✓` once all stages are done.
3. **In flight**: pids, port_base, started-at for any currently-running
   stages. `alive` / `DEAD` based on `kill -0` to the pid.
4. **Failed**: count + (last 10 with log paths in non-compact mode).
5. **FPS by method**: cumulative env-step rate + a rolling "recent fps"
   over the last ~50 episodes per method, total steps, elapsed wall-time,
   episode count, and contributing run count. Drops slowdowns from
   per-stage averaging — if recent fps drifts much below cumulative, the
   box is degrading (renderer fell back, RAM pressure, etc.).
6. **Per-stage performance tables** — reward + pickups, columns are stages
   (with cumulative end-step labels), rows are `(variation, method)` or
   just `method` if there's only one variation. Each cell is `mean (n_seeds)`
   over the last ~50 episodes per seed. Only stages where at least one seed
   has data are shown.

The FPS / perf tables use only `train_episodes.jsonl`, written by the env
on episode terminate/truncate. Smoke runs with stage_steps < episode_max_steps
may produce zero rows — that's expected, not a logging bug.

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
