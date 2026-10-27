# probing/ — offline probes of DreamerV3's hidden state

Does a trained Dreamer's recurrent state *remember* the maze? The pipeline
records episodes, computes what the agent could know (the map it has seen,
the true walls and boxes), extracts the hidden state of each checkpoint along
those episodes, and trains small probes to decode the one from the other. The
protocol follows Memory Maze (Pasukonis et al. 2022): probes on frozen states,
scored per memory age (how long ago a cell was last in view), against an
observation-only and an untrained-network control.

The folders are the pipeline, in order:

```
probing/
  core/          shared by every step
    dataset.py     episode files, sealing (content fingerprint), locks, splits
    agent_io.py    build a Dreamer agent from a run's config, load a stage checkpoint
  s1_collect/    1. record episodes in Unity
    collect.py     policy episodes from checkpoints (+ optional scripted explorer), N workers
    explorer.py    the scripted explorer
  s2_targets/    2. what a probe must predict
    targets.py     seen-map windows, coverage, distance to start -- from the agent's own lidar
    layouts.py     true maze + boxes per episode from the Unity layout dump, checked against the lidar
    global_targets.py  true-map targets (glayout: walls, gboxes: boxes) in an agent-centred crop
  s3_encode/     3. the hidden state of every checkpoint along the recorded episodes
    encode.py
  s4_probe/      4. one probe per (states, target)
    train_probe.py   train + score (early-stopped MLP, split by episode)
    global_score.py  scoring of the true-map probes by memory age
  s5_analysis/   5. tables and figures over all probes
    analyze.py, analyze_global.py
  tools/         optional: inspection figures, diagnostics, decoded dreams, coverage per arm
  jobs/          the whole pipeline as one cluster job
    probe_job.slurm  RCI (SLURM)
    probe_job.qsub   Metacentrum (PBS) -- the version it was translated from
```

Dependencies only point forward: a step imports `core/` and earlier steps,
never a later one.

## What each step reads and writes

Everything lives under `results/probing/<name>/`; every step skips work whose
output already exists and was computed from the dataset as it is sealed now,
so re-running resumes.

| Step | Reads | Writes |
|---|---|---|
| 1 collect | `results/experiments/<exp>/experiment.yaml`, `runs/<cell>__dreamer__seed<s>/{run_config.json, checkpoints/stage_<k>/}` | `dataset/episodes/ep*.npz`, `dataset/meta.json` (plan, presets, fingerprint), `dataset/spaces.json` |
| 2 targets | the episodes | `dataset/targets/` (seen map, coverage, start) |
| 2 layouts | the episodes + a Unity build with the layout dump | `dataset/layouts/` (true mask + boxes per episode) |
| 2 global_targets | episodes + layouts | `dataset/gtargets/` (glayout, gboxes) |
| 3 encode | dataset + checkpoints | `states/<cell>__dreamer__seed<s>__stage<k>.npz`, `states/random_init*.npz` |
| 4 train_probe | dataset + targets + one states file (or `obs` / `const` controls) | `probes/<model>/<target>[__pseed<k>]/{metrics.json, test_predictions.npz}` |
| 5 analyze | all probes | `probes/analysis/` (summary.csv, probes.csv, forgetting curves, paired comparisons) |

## Running it

On a cluster, one job runs every step (from `ratsim_experiments/`):

```bash
# RCI: parameters are environment variables, see the header of the script
PROBE_NAME=pilot PROBE_EXP=<exp> PROBE_CELLS="baseline" sbatch -p amdgpufast --time=04:00:00 probing/jobs/probe_job.slurm
```

By hand, the same order (main venv = torch/scipy, dreamer venv = JAX):

```bash
$DREAMER_PYTHON_PATH -m probing.s1_collect.collect --help
$PPO_PYTHON_PATH     -m probing.s2_targets.targets --help
$PPO_PYTHON_PATH     -m probing.s2_targets.layouts --help          # RATSIM_UNITY_BIN = layout-dump build
$PPO_PYTHON_PATH     -m probing.s2_targets.global_targets --help
$DREAMER_PYTHON_PATH -m probing.s3_encode.encode --help
$PPO_PYTHON_PATH     -m probing.s4_probe.train_probe --help
$PPO_PYTHON_PATH     -m probing.s5_analysis.analyze --help
```

## Things that bite

- **The layout build must be the episodes' build.** `layouts.py` re-generates
  each world from its seed in a Unity build that carries the layout dump
  (`LayoutDump.cs` + the `MazeLayoutLoader` hooks). If its world generation
  differs from the build that recorded the episodes, the verification against
  the lidar fails -- which is the point of it.
- **Boxes and chunk streaming.** Builds that stream chunks (ratsim_unity_project
  `human-ui-styles`, `chunk_width: 64`) create a room's boxes only when its
  chunk loads, so a reset-time dump lists only the boxes near the spawn. Dump
  with `chunk_width: 256` (one chunk) before probing `gboxes`. Walls are not
  affected.
- **One job per `<name>`.** A lock refuses a second job on the same name; more
  cells on an existing sealed dataset: `PROBE_COLLECT=0`.
- **Seeds.** Collection draws worlds from metaseed 777 -- apart from training
  (0, 1, ...) and the paper's eval (42).
- **Stock and TBTT runs need nothing extra.** `core/agent_io.py` also builds the
  ratsim_metacentrum architecture variants (gridmem, memrssm, probeloss) when
  those packages exist, and skips them when they don't.

## Tests (`tests/`)

| File | Venv | What |
|---|---|---|
| `test_probe_targets.py` | main | seen-map targets, windows, pose scoring |
| `test_probe_global.py` | main | layouts, true-map targets, scoring, pipeline on CPU |
| `test_probe_concurrency.py` | main | atomic files, locks, sealing under real processes |
| `test_probing.py` | dreamer | offline encode == the acting state; explorer; plan/split |
| `test_probe_unity.py` | dreamer | real Unity: parallel = serial, lock, walltime kill + resume, full pipeline (`RATSIM_UNITY_BIN=...`) |

Each file runs directly: `python tests/<file>.py`.
