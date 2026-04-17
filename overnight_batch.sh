#!/bin/bash
# Overnight batch runner. Run under tmux/nohup:
#   tmux new -s batch
#   ./overnight_batch.sh
#   Ctrl-B D to detach
#
# No `set -e` — a single failed run should not abort the whole queue.

STAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/batch_${STAMP}"
mkdir -p "$LOG_DIR"

run() {
    local name="$1"
    shift
    echo "[$(date +%H:%M:%S)] START $name" | tee -a "$LOG_DIR/_summary.log"
    python train.py name="$name" "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
    local rc=${PIPESTATUS[0]}
    echo "[$(date +%H:%M:%S)] END   $name  rc=$rc" | tee -a "$LOG_DIR/_summary.log"
}

# --- queue ---

run lawn_dreamer_s1          def=lawnmower_shortsighted method=dreamerv3 metaseed=1
run lawn_dreamer_s2          def=lawnmower_shortsighted method=dreamerv3 metaseed=2
run lawn_dreamer_s3          def=lawnmower_shortsighted method=dreamerv3 metaseed=3

run lawn_olfaction_dreamer_s1          def=lawnmower_shortsighted_sectorsignal method=dreamerv3 metaseed=1
run lawn_olfaction_dreamer_s2          def=lawnmower_shortsighted_sectorsignal method=dreamerv3 metaseed=2
run lawn_olfaction_dreamer_s3          def=lawnmower_shortsighted_sectorsignal method=dreamerv3 metaseed=3

echo "[$(date +%H:%M:%S)] batch done. Summary: $LOG_DIR/_summary.log"
