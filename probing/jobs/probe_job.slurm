#!/bin/bash
# Offline probing of DreamerV3's hidden state (probing/), RCI / SLURM version of
# probe_job.qsub (the Metacentrum PBS job it was translated from).
#
# One job runs the whole pipeline over checkpoints that ALREADY EXIST in
# results/experiments/<PROBE_EXP>/runs/<cell>__dreamer__seed<s>/checkpoints/stage_<k>/;
# nothing is trained. Steps, each skipped when its output exists, so submitting
# the same command again resumes:
#
#   collect   dreamer venv, PROBE_WORKERS Unity instances in parallel: episodes
#             from every seed's checkpoint at PROBE_POLICY_STAGE (+ optional
#             scripted-explorer episodes). Planned up front; a frozen Unity fails
#             its worker and collect is retried in place (PROBE_COLLECT_TRIES).
#   targets   main venv: seen-map windows + geodesic distance to start
#   layouts   main venv + a Unity build WITH the layout dump (LayoutDump.cs and
#             the MazeLayoutLoader hooks): the true maze + boxes of every
#             episode, verified against its lidar. Only for glayout/gboxes.
#             RATSIM_LAYOUT_BIN = that build (default: RATSIM_UNITY_BIN). It
#             must be the same Unity code the episodes came from. On a
#             chunk-streaming build (human-ui-styles) boxes exist only in loaded
#             chunks, so dump with chunk_width 256 before probing gboxes.
#   gtargets  main venv: ground-truth targets + masks (same condition)
#   encode    dreamer venv, no Unity: every cell x seed x stage, + random init
#   probe     main venv (torch): one probe per state file and target, plus the
#             obs-only and constant controls
#   analyze   main venv: tables + forgetting curves
#
# Parameters are environment variables; sbatch passes the submitting shell's
# environment to the job, so lists stay space-separated. From the login node
# (login2/3/4), in meta_ratsim/ratsim_experiments:
#
#   PROBE_NAME=pilot PROBE_EXP=memory_orthomaze sbatch -p amdgpufast --time=04:00:00 probing/jobs/probe_job.slurm
#
#   PROBE_NAME=full PROBE_EXP=memory_orthomaze PROBE_CELLS="baseline" PROBE_SEEDS="0 1 2" \
#   PROBE_STAGES="0 3 6 9 13 19" PROBE_POLICY_STAGE=19 PROBE_EPISODES=34 PROBE_EXPLORER=0 \
#   PROBE_TARGETS="layout coverage start" PROBE_PSEEDS="0 1 2" PROBE_PSEEDS_STAGE=19 \
#   sbatch probing/jobs/probe_job.slurm
#
# More cells on the same sealed dataset (e.g. another experiment's runs):
# PROBE_COLLECT=0 PROBE_EXP=<other exp> PROBE_CELLS=... PROBE_MODELS="<cell>__*".
# One job per PROBE_NAME at a time (flock); chain a second with
# --dependency=afterok:<jobid>.
#
# Resources, as on Metacentrum: 3 collect workers (Unity + Python ~2 cores
# each) fill 8 CPUs; RAM peaked ~7 GB, 3 parallel probes need ~4 GB each. The
# GPU is busy only in encode and probe. A 204-episode, 36-checkpoint,
# 4-target run took ~2.5 h on an A40 (collect ~0.5 h, probes ~1.3 h).
#
# Results: results/probing/<PROBE_NAME>/
#   dataset/  states/  probes/<model>/<target>[__pseed<k>]/  probes/analysis/

#SBATCH --job-name=ratsim-probe
#SBATCH --partition=amdgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/personal/%u/logs/probe-%j.out

set -uo pipefail

PROBE_NAME="${PROBE_NAME:-pilot}"
PROBE_EXP="${PROBE_EXP:?set PROBE_EXP (an experiment under results/experiments/)}"
PROBE_CELLS="${PROBE_CELLS:-baseline}"     # variations; run dirs <cell>__dreamer__seed<s>
PROBE_SEEDS="${PROBE_SEEDS:-0}"
PROBE_STAGES="${PROBE_STAGES:-0}"
PROBE_POLICY_STAGE="${PROBE_POLICY_STAGE:-0}"
PROBE_EPISODES="${PROBE_EPISODES:-5}"      # per (cell, seed in PROBE_SEEDS)
PROBE_EXPLORER="${PROBE_EXPLORER:-10}"
PROBE_METASEED="${PROBE_METASEED:-777}"    # != training metaseeds, != eval's 42
PROBE_WORKERS="${PROBE_WORKERS:-3}"        # parallel Unity instances for collect
PROBE_PARALLEL="${PROBE_PARALLEL:-3}"      # probes trained at once on the one GPU
PROBE_COLLECT_TRIES="${PROBE_COLLECT_TRIES:-3}"  # collect attempts within this job (resume in place)
PROBE_COLLECT="${PROBE_COLLECT:-1}"        # 0: probe more cells on the existing
                                           # sealed dataset of PROBE_NAME, no Unity
PROBE_TARGETS="${PROBE_TARGETS:-layout coverage start}"  # + glayout gboxes (ground truth)
PROBE_MODELS="${PROBE_MODELS:-*}"          # globs over model names (states files, obs, const)
PROBE_PSEEDS="${PROBE_PSEEDS:-0}"          # probe seeds for PSEEDS_STAGE checkpoints + controls
PROBE_PSEEDS_STAGE="${PROBE_PSEEDS_STAGE:-}"
PROBE_WINDOW="${PROBE_WINDOW:-64}"         # glayout/gboxes crop side in cells; 0 = whole map
PROBE_EXPECT_BOXES="${PROBE_EXPECT_BOXES:-}"

# shellcheck disable=SC1091
. "${RCI_ENV:-$HOME/rci_env.sh}" || exit 2  # modules, paths, per-job TMPDIR, *_PYTHON_PATH
ratsim_activate || exit 2
export RATSIM_XVFB=1
# No RATSIM_RUNDIR / ports: the launcher derives both from $SLURM_JOB_ID.
EXPERIMENTS_DIR="${EXPERIMENTS_DIR:-$RATSIM_GIT_DIR/meta_ratsim/ratsim_experiments}"
STOP_UNITY="${STOP_UNITY:-$RATSIM_GIT_DIR/meta_ratsim/ratsim/scripts/stop_ratsim_headless.sh}"
LOG_DIR="${LOG_DIR:-/mnt/personal/$USER/logs}"
LAYOUT_BIN="${RATSIM_LAYOUT_BIN:-$RATSIM_UNITY_BIN}"
mkdir -p "$LOG_DIR"
# Where the launcher keeps pidfiles + Unity logs for this job (kept on a failed
# collect attempt, since the per-job TMPDIR goes away with the job).
UNITY_RUNDIR="$("$PPO_PYTHON_PATH" -c 'from ratsim.unity_launcher import _rundir; print(_rundir())')"

cd "$EXPERIMENTS_DIR" || exit 2
EXP_DIR="results/experiments/$PROBE_EXP"
OUT="results/probing/$PROBE_NAME"
DS="$OUT/dataset"
FIRST_CELL="${PROBE_CELLS%% *}"

echo "=== $(date -Is)  probe job ${SLURM_JOB_ID:-?} on $(hostname) ==="
echo "name/exp:   $PROBE_NAME / $PROBE_EXP"
echo "cells:      $PROBE_CELLS   seeds: $PROBE_SEEDS   stages: $PROBE_STAGES"
echo "dataset:    $PROBE_EPISODES policy eps/cell (stage $PROBE_POLICY_STAGE) + $PROBE_EXPLORER explorer, metaseed $PROBE_METASEED"
echo "targets:    $PROBE_TARGETS   models: $PROBE_MODELS   probe seeds: $PROBE_PSEEDS${PROBE_PSEEDS_STAGE:+ (stage $PROBE_PSEEDS_STAGE + controls)}"
echo "unity:      $RATSIM_UNITY_BIN   layouts: $LAYOUT_BIN"
echo "out:        $PWD/$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "no GPU"

step() { echo; echo "=== $(date -Is)  $*"; }
fail() { echo "FAILED: $*"; "$STOP_UNITY" --all || true; sleep 1; exit 1; }
# current FILE: exists AND was computed from the dataset as sealed now. A
# skipped step must never reuse output of an earlier version of the episodes.
current() { "$PPO_PYTHON_PATH" -m probing.core.dataset current "$DS" "$1"; }

# --- one job per PROBE_NAME ---------------------------------------------------
# Re-submitting while an earlier job on the same name is still running would
# have two jobs rewriting the same files -- the second could compute targets
# from one version of an episode and states from another. The lock is held on
# fd 9 for the life of this script; a second job fails here at once. (flock on
# NFSv4 is enforced by the server across nodes.)
mkdir -p "$OUT"
exec 9>"$OUT/.job.lock"
if ! flock -n 9; then
  echo "FATAL: another job holds $OUT/.job.lock -- one job per PROBE_NAME"
  exit 3
fi

# --- collect ----------------------------------------------------------------
# Always run: it resumes the episodes still missing and re-seals the dataset;
# on a complete, sealed dataset it records nothing and starts no Unity.
# PROBE_COLLECT=0 probes other cells (other PROBE_CELLS) on the dataset that
# is already there: the plan would differ, so collect must not run at all.
if [[ "$PROBE_COLLECT" == 0 ]]; then
  step "collect: skipped, using the existing dataset"
  "$PPO_PYTHON_PATH" -c "from probing.core import dataset; print('sealed', dataset.verify('$DS')[:12])" \
      || fail "PROBE_COLLECT=0 needs a complete, sealed dataset at $DS"
else
step "collect"
{
  POLICY_ARGS=()
  # Every seed's policy, not seed0's alone: a dataset driven by one model
  # is that model's state distribution, and every other checkpoint is probed
  # on it.
  for cell in $PROBE_CELLS; do
    for s in $PROBE_SEEDS; do
      POLICY_ARGS+=(--policy "runs/${cell}__dreamer__seed${s}:${PROBE_POLICY_STAGE}:${PROBE_EPISODES}")
    done
  done
  # A Unity instance now and then stops answering mid-episode: the connector's
  # timeout fails that worker and collect stops the others (once in ~115 episodes on
  # Metacentrum, right after a reset). collect resumes the missing episodes, so
  # the attempt is retried here on fresh Unity instances. Each failed attempt's
  # Unity logs are kept in logs/ -- the per-job TMPDIR goes at job end.
  for attempt in $(seq 1 "$PROBE_COLLECT_TRIES"); do
    "$DREAMER_PYTHON_PATH" -u -m probing.s1_collect.collect --out "$DS" --exp_dir "$EXP_DIR" \
        --variation "$FIRST_CELL" "${POLICY_ARGS[@]}" --explorer "$PROBE_EXPLORER" \
        --metaseed "$PROBE_METASEED" --workers "$PROBE_WORKERS" --spawn && break
    keep="$LOG_DIR/unity_${SLURM_JOB_ID:-local}_collect$attempt"
    mkdir -p "$keep" && cp "$UNITY_RUNDIR"/ratsim_*.log "$keep/" 2>/dev/null
    (( attempt < PROBE_COLLECT_TRIES )) || fail "collect ($attempt attempts; Unity logs in $keep)"
    "$STOP_UNITY" --all || true
    step "collect: attempt $attempt failed (Unity logs in $keep), resuming"
  done
  "$STOP_UNITY" --all || true
}
fi

wants() { [[ " $PROBE_TARGETS " == *" $1 "* ]]; }
WANT_OLD=0; WANT_GT=0
for t in layout coverage start; do wants "$t" && WANT_OLD=1; done
for t in glayout gboxes; do wants "$t" && WANT_GT=1; done

# --- targets ----------------------------------------------------------------
if (( ! WANT_OLD )); then
  step "targets: no seen-map target requested, skipping"
elif current "$DS/targets/targets.json"; then
  step "targets: done, skipping"
else
  step "targets"
  # One process per CPU, each single-threaded: the work is Python loops.
  OMP_NUM_THREADS=1 "$PPO_PYTHON_PATH" -u -m probing.s2_targets.targets --dataset "$DS" \
      --workers "${SLURM_CPUS_PER_TASK:-8}" || fail targets
fi

# --- layouts + ground-truth targets ------------------------------------------
if (( WANT_GT )); then
  if current "$DS/layouts/layouts.json"; then
    step "layouts: done, skipping"
  else
    step "layouts"
    # A build with the layout dump, from the same Unity code as the episodes.
    RATSIM_UNITY_BIN="$LAYOUT_BIN" \
      "$PPO_PYTHON_PATH" -u -m probing.s2_targets.layouts --dataset "$DS" --spawn \
        ${PROBE_EXPECT_BOXES:+--expect_boxes "$PROBE_EXPECT_BOXES"} || fail layouts
    "$STOP_UNITY" --all || true
  fi
  if current "$DS/gtargets/gtargets.json"; then
    step "gtargets: done, skipping"
  else
    step "gtargets"
    OMP_NUM_THREADS=1 "$PPO_PYTHON_PATH" -u -m probing.s2_targets.global_targets --dataset "$DS" \
        --workers "${SLURM_CPUS_PER_TASK:-8}" || fail gtargets
  fi
fi

# --- encode -----------------------------------------------------------------
for cell in $PROBE_CELLS; do
  runs=(); missing=0
  for s in $PROBE_SEEDS; do
    runs+=("runs/${cell}__dreamer__seed${s}")
    for k in $PROBE_STAGES; do
      current "$OUT/states/${cell}__dreamer__seed${s}__stage${k}.npz" || missing=1
    done
  done
  if (( missing )); then
    step "encode $cell"
    # shellcheck disable=SC2086
    "$DREAMER_PYTHON_PATH" -u -m probing.s3_encode.encode --dataset "$DS" --out "$OUT/states" \
        --exp_dir "$EXP_DIR" --runs "${runs[@]}" --stages $PROBE_STAGES || fail "encode $cell"
  else
    step "encode $cell: done, skipping"
  fi
done
# Untrained control per architecture: the first cell's config, no checkpoint.
# The baseline's keeps its original name; any other is random_init__<cell>.
RI_NAME="random_init"
[[ "$FIRST_CELL" == L0_baseline ]] || RI_NAME="random_init__${FIRST_CELL}"
if ! current "$OUT/states/${RI_NAME}.npz"; then
  step "encode $RI_NAME"
  "$DREAMER_PYTHON_PATH" -u -m probing.s3_encode.encode --dataset "$DS" --out "$OUT/states" \
      --exp_dir "$EXP_DIR" --runs "runs/${FIRST_CELL}__dreamer__seed0" \
      --random_init --random_init_name "$RI_NAME" || fail "encode $RI_NAME"
fi

# --- probes -----------------------------------------------------------------
# Each probe is a small MLP that leaves the GPU mostly idle, so PROBE_PARALLEL
# of them run at once. Missing ones only, so a resubmit picks up the rest.
QUEUE="$OUT/probe_queue.txt"
: > "$QUEUE"
read -ra MODEL_PATTERNS <<< "$PROBE_MODELS"   # read -ra: no glob expansion here
wanted() {  # wanted <model>: matches one of the PROBE_MODELS globs
  local pat
  for pat in "${MODEL_PATTERNS[@]}"; do
    # shellcheck disable=SC2053
    [[ "$1" == $pat ]] && return 0
  done
  return 1
}
probe() {  # probe <model> <features> [--states FILE]
  local model="$1" feats="$2"; shift 2
  wanted "$model" || return 0
  # Extra probe seeds: checkpoints of PROBE_PSEEDS_STAGE and the learnt
  # controls (obs, random_init*). const has no randomness.
  local pseeds="0"
  if [[ -n "$PROBE_PSEEDS_STAGE" && "$model" == *__stage"$PROBE_PSEEDS_STAGE" ]] \
     || [[ "$model" == obs || "$model" == random_init* ]]; then
    pseeds="$PROBE_PSEEDS"
  fi
  local target ps tdir extra
  for target in $PROBE_TARGETS; do
    tdir="$target"; extra=""
    if [[ "$target" == glayout || "$target" == gboxes ]]; then
      tdir="${target}_w${PROBE_WINDOW}"; extra=" --window $PROBE_WINDOW"
    fi
    for ps in $pseeds; do
      local out="$OUT/probes/$model/$tdir" seedarg=""
      if [[ "$ps" != 0 ]]; then out+="__pseed$ps"; seedarg=" --seed $ps"; fi
      current "$out/metrics.json" && continue
      echo "\"$PPO_PYTHON_PATH\" -u -m probing.s4_probe.train_probe --dataset \"$DS\" --features $feats --target $target --out \"$out\"$extra$seedarg $*" >> "$QUEUE"
    done
  done
}
for f in "$OUT"/states/*.npz; do
  # A states file left over from another dataset version is not probed.
  current "$f" || { echo "skipping stale $f"; continue; }
  probe "$(basename "$f" .npz)" state --states "$f"
done
probe obs obs
probe const const
step "probes: $(wc -l < "$QUEUE") to train, $PROBE_PARALLEL at a time"
xargs -P "$PROBE_PARALLEL" -d '\n' -I{} bash -c '{}' < "$QUEUE" \
    || fail "probes (some failed; resubmit to retry only those)"

# --- analyze ----------------------------------------------------------------
step "analyze"
"$PPO_PYTHON_PATH" -u -m probing.s5_analysis.analyze --probes "$OUT/probes" --dataset "$DS" \
    || fail analyze

"$STOP_UNITY" --all || true
step "done -> $PWD/$OUT/probes/analysis"
sleep 1
