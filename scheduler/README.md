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

## On the cluster — `submit.sh`

Don't hand-write sbatch lines. You pick the experiment and how long you're
willing to give it; partition, `--cpus-per-task`, `--gres`, `--mem` and any
CPU/GPU split are derived from the def and `machines/*.yaml`:

```bash
./submit.sh gps_ablation_5house --time 1d
./submit.sh method_compare      --time 4h --mode bfs     # short taster
./submit.sh dreamer_ladder      --time 3d --dry-run      # print, don't submit
```

```
method_compare: 7 runs (1 variations × [ppo×3, dreamer×3, recurrent_ppo×1] seeds),
                10 stages × 1,000,000 steps
  wall clock 3-00:00:00 → long partitions   mode bfs
  → amdlong      112t, 200G          ppo                    (3 runs, 3 concurrent)
  → amdgpulong   62t, 2×GPU, 200G    dreamer,recurrent_ppo  (4 runs, 3 concurrent)
```

Notes:

- **`--time` picks the partition.** 4h → `amdfast`, 1d → `amd`, 3d → `amdlong`,
  beyond → `amdextralong` (GPU equivalents `amdgpu*`). `scheduler_job.sbatch`
  defaults to the 4 h smoke partition, which kills real runs mid-stage — this
  removes that trap.
- **Resume is the default**, so a 4 h taster you later extend to 3 days costs
  nothing: same exp_id, same `.done` markers, same `wandb_id.txt`, so the W&B
  curves continue rather than restarting.
- **The account CPU/GPU caps are aggregate across all your running jobs**, not
  per-job (RCI_CLUSTER_PORT.md §0.55). 200 CPUs on the 1-day and 3-day groups
  means one 112-thread PPO job plus one 62-thread GPU job fits, and two
  112-thread jobs don't — the second just pends forever with no error.
  `submit.sh` checks `squeue` and warns before you hit it.
- **`--mode` is never inferred.** Short job ≠ automatically bfs.
- **Run it from login2/3/4**, not login1 (CentOS 7 / glibc 2.17 can't run the
  venv's interpreter).

`--machine <name>` forces everything into one job; `--only ppo` submits just one
method's share; `--variations consec4` runs just one cell of a ladder def.

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

# Give any TWO of steps_per_stage / total_steps / n_stages; the third follows.
# Best pairing — state the budget and the granularity:
steps_per_stage: 1_000_000
total_steps: 10_000_000                         # → 10 stages

# Also fine:  steps_per_stage + n_stages
# ⚠️ Legacy:  total_steps + n_stages  (derives the stage size — see below)

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

### ⚠️ Always pin `steps_per_stage`

Any two of `steps_per_stage` / `total_steps` / `n_stages` determine the third, so
all three pairings parse. But they are not equally safe:

| Pairing | Derived | Safe to edit later? |
|---|---|---|
| `steps_per_stage` + `total_steps` | `n_stages` | ✅ best — raise the total, stages append |
| `steps_per_stage` + `n_stages` | `total_steps` | ✅ raise `n_stages`, stages append |
| `total_steps` + `n_stages` | **stage size** | ⚠️ legacy — see below |

`checkpoints/stage_<i>.done` records *that stage K finished* — never how big it
was. So stage size is the one quantity resume silently depends on. Derive it and
changing either other number resizes **every** stage, including ones already
marked done: bump `total_steps: 12M → 18M` on a run that's finished 30 of 40
stages and those thirty markers now claim 450k steps of training that never
happened, which resume then builds on without complaint.

Pin `steps_per_stage` and that can't happen — raising `total_steps` only appends
stages. The two must divide evenly; if they don't you get an error naming the
two nearest usable totals, so there's no arithmetic to do by hand.

Giving all three is allowed and cross-checked (a contradiction is an error, not
a silent winner). The only safe edit to a legacy def is to scale both numbers
together so the ratio holds — or convert it, which is free when the derived size
is already what you want (`12M / 40` → `steps_per_stage: 300_000`, identical
stages, existing markers stay valid).

### Splitting one def across jobs — `--methods`

A def mixing PPO and dreamer can't run as one job: a scheduler process holds one
allocation, and `machines/rci.yaml` has no dreamer profile at all (correctly —
the `amd*` partitions have no accelerators). `--methods` lets each job claim its
share, both feeding the same exp dir:

```bash
# CPU partition
python scheduler_run.py gps_ablation --machine rci      --methods ppo
# GPU partition, separate sbatch
python scheduler_run.py gps_ablation --machine rci_gpu2 --methods dreamer
```

One exp_id, one W&B group, one `analyze_experiment.py`. Run dirs are disjoint
because the method is part of `run_id`.

Each filter gets **its own state file** (`state.ppo.json`, `state.dreamer.json`).
That's load-bearing, not cosmetic: `state["running"]` holds child pids and the
scheduler reaps every pid it finds there on startup, unable to distinguish a
stale pid from its own crashed invocation from a live child of the sibling job.
With one shared file, submitting the GPU job after the CPU job would kill all
seven running PPO children. `scheduler_status.py` merges every `state*.json`, so
status still shows one experiment.

Under `--methods`, `--restart` wipes only that job's runs.

### Running one cell of a ladder — `--variations`

Same mechanism, one axis over. A ladder def is several variants of one method
(`variations:`), and often only one of them is worth continuing — giving the
promising cell more steps without paying for the rest:

```bash
python scheduler_run.py ortho_wells_adaptive_nohomeprime_bigdreamer_ladder \
    --machine rci_gpu2 --variations consec4
```

Run dirs are `<variation>__<method>__seed<i>`, so a filtered job writes into the
same exp dir and the same W&B group; the other cells simply don't advance. It
combines with `--methods`, and the pair picks the state file:
`state.dreamer.json`, `state.v.consec4.json`, `state.dreamer.v.consec4.json` —
so a ladder job started later doesn't reap a sibling's children. `--restart`
under either filter wipes only that job's runs.

Don't derive these commands by hand — `submit.sh` does it (below).

### Skipping the def for ad-hoc runs

The scheduler always loads from `defs/*.yaml`, but `train.py` (and
`train_dreamerv3.py`) accept the same fields inline on the CLI when you don't
want to save a file:

```bash
python train.py method=ppo world=maze_default total_steps=100_000
python train.py method=ppo agent=sphereagent_2d_lidar \
    task=volumetric_exploration_2000_collision_penalty \
    world=maze_default total_steps=1_000_000 n_stages=10 metaseed=42
python train_dreamerv3.py world=maze_default total_steps=500_000 n_stages=5
```

The CLI keys `world` / `task` / `agent` map to the YAML schema
`world_preset` / `task_preset` / `agent_preset` when assembling an inline
def. Defaults for inline mode: `agent=sphereagent_2d_lidar`,
`task=default`, `n_stages=1`. `world` and `total_steps` are required.
For curricula or variation sweeps, write a def — that's exactly the case
where saving the file pays off.

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
    needs: {gpu: 1, cpu_slot: 16}
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

`n_envs` — the number of parallel Unity envs each dispatched job spawns —
lives on the **experiment def**, per method, and defaults to **4**:

```yaml
methods:
  - name: ppo
    n_seeds: 3
    n_envs: 4        # optional; 4 is the default
```

It used to live on the machine profile, on the theory that "how many envs
make sense" is a property of the box. That was wrong: fewer envs means less
trajectory diversity per update, so a 1-env run and a 4-env run of the same
def are **different experiments**, not the same experiment at different
speeds. Keeping one value across every machine is what makes a laptop run
comparable with a cluster run. A machine config that still sets `n_envs:`
is now a hard error telling you to move it.

The scheduler passes `n_envs=<N>` to train.py at dispatch time, and always
records it in the def snapshot.

### Sizing: ~4 cpu_slots per env

`cpu_slot` is counted in **threads**. The scheduler warns at startup when a
method's `needs.cpu_slot` is below `4 × n_envs`:

```
[scheduler] WARNING: method 'ppo' runs n_envs=4 on cpu_slot=8. Below ~4
slots per env (16 here) throughput drops sharply — ...
```

It's a warning, not an error — an undersized box still produces valid
results. But this is a cliff, not a slope. The same 20k-step PPO run at
`n_envs=4`: **17 fps** on 4 threads, **251** on 8, **688** on 16.

The corollary is that packing more runs by shrinking `needs.cpu_slot` makes
things worse, not better: 4 runs sharing one 16-thread job took 1265 s,
against 339 s when only 2 ran at a time.

`n_envs` is in `RESERVED_ARGS` — putting it in a profile's `args:` or in
def-level `common_args:` is ignored with a warning. The `methods:` entry is
the only place to set it for scheduler-driven runs.

**Hard cap**: `n_envs ≤ 10` per job (each dispatch gets a 10-wide Unity
port window starting at `base_port`). Validation catches this at startup.

For inline ad-hoc training (`python train.py method=ppo ... n_envs=8`),
just pass `n_envs=N` on the CLI — the train scripts default it to 1.

## GPUs

Children are plain subprocesses inside one job, so without intervention they'd
all inherit the same `CUDA_VISIBLE_DEVICES` and pile onto device 0 while the
other cards sat idle — the port-collision problem, one resource over.
`GpuAllocator` (`scheduler/gpus.py`) is the fix: it hands each dispatch its own
device and sets `CUDA_VISIBLE_DEVICES` for that child alone.

The pool comes from the job's own `CUDA_VISIBLE_DEVICES` when set. Under SLURM
with cgroup device isolation the job only *sees* the GPUs it was granted and
that variable lists exactly them, so each entry means the same device to a child
as it does to the scheduler. With the variable unset it falls back to
`range(resources.gpu)`. Ids are passed through as strings, so GPU-UUID form
works too.

```
[scheduler] GPU pool: 0, 1, 2, 3 (CUDA_VISIBLE_DEVICES is set per child; ...)
[scheduler] dispatch baseline__dreamer__seed0 stage 0 port=9100 n_envs=4 gpu=0
[scheduler] dispatch baseline__ppo__seed0     stage 0 port=9110 n_envs=4 gpu=none(masked)
```

Three behaviours worth knowing:

- **A method with no `needs.gpu` is masked off the GPUs entirely**
  (`CUDA_VISIBLE_DEVICES=""`), not left to inherit the job's list. Otherwise a
  CPU-profile PPO run still opens a context on whichever card it likes and takes
  memory from the dreamer beside it. **If a method genuinely needs the GPU,
  declare `needs: {gpu: 1}`** — that declaration is what this reads.
- **On a machine config that declares no GPUs, nothing is touched at all.** The
  child inherits the environment exactly as before, so laptop runs are
  unaffected.
- **Capacity is the smaller of `resources.gpu` and the GPUs actually granted.**
  Declaring fewer than `--gres` gave keeps a deliberate cap (the scheduler will
  not widen it because the sbatch over-asked); declaring more gets clamped so
  the loop doesn't keep offering candidates the pool can't serve. Either
  mismatch warns at startup.

To run N GPU jobs at once, `--gres=gpu:N`, `resources.gpu: N` and enough
`cpu_slot` for N runs all have to move together — `cpu_slot` binds
independently, so `gpu: 4` with `cpu_slot: 31` still only fits one 16-slot run.

## Unity ports

Every dispatch gets a fresh non-overlapping port window starting at 9100,
stepping by 10 (matching the convention documented in
`ratsim_experiments/CLAUDE.md`). The scheduler always passes `base_port=` so
`n_envs=1` jobs spawn fresh too — they don't attach to the persistent :9000
instance.

### Released windows cool before reuse

A finished window is **not** returned straight to the pool. The scheduler reaps
a job when its *train process* exits, but that process spawned `n_envs` Unity
children who outlive it by seconds — so the window's ports are still held at the
moment the scheduler considers it free. Handing it to the next dispatch
immediately produced, in job 11325473:

```
RuntimeError: port 9640 still in use after waiting
```

which killed 4 of 4 dreamer re-dispatches (0 of 4 PPO ones — PPO's Unity
children die inside the launcher's 20 s grace, dreamer's don't).

So `release()` puts the window in a `cooling` set, and `alloc()` only returns a
cooling window once **every port in it actually binds**. Otherwise it skips to
the next window and logs:

```
[scheduler] port windows still cooling, skipped: [9640] — using 9660
```

Nothing waits and no timeout is guessed — there is always another window, so the
next run starts immediately and the cooled one rejoins the pool the moment it is
genuinely clear. The launcher's own 20 s wait stays as a backstop for the
residual race (the allocator tests the port, the child binds it a moment later).

The whole window is tested, not just the first `n_envs` ports, since the
allocator can't know how many envs the *next* run will want. A window with a
stranger squatting on any of its ports keeps getting skipped, which is correct.

`alloc()` scans at most 100 windows (9100–10099) and then raises rather than
looping forever — hitting that means leaked Unity processes, and
`scripts/stop_ratsim_headless.sh --all` is the cleanup.

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
