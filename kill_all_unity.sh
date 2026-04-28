#!/usr/bin/env bash
# Sledgehammer for killing zombie Unity processes left behind when the
# scheduler / a train.py / tmux died ungracefully and didn't unwind its
# children. Doesn't rely on /tmp/ratsim_*.pid pidfiles — finds processes
# by command-line match.
#
# Usage:
#   ./kill_all_unity.sh              # SIGTERM matches; uses $RATSIM_UNITY_BIN basename or 'SARBench'
#   ./kill_all_unity.sh -9           # SIGKILL (use if SIGTERM didn't take)
#   ./kill_all_unity.sh -p MyBuild   # custom command-line pattern
#   ./kill_all_unity.sh -n           # dry-run — show what would be killed
#
# Also cleans up stale /tmp/ratsim_*.pid files for dead pids. Doesn't touch
# pidfiles for processes still running — use stop_ratsim_headless.sh --all
# for the well-behaved case.

set -u

SIGNAL="-TERM"
PATTERN=""
DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -9|--kill)    SIGNAL="-KILL"; shift ;;
    -p|--pattern) PATTERN="$2"; shift 2 ;;
    -n|--dry-run) DRY_RUN=1; shift ;;
    -h|--help)    sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1 (try -h)"; exit 2 ;;
  esac
done

if [[ -z "$PATTERN" ]]; then
  if [[ -n "${RATSIM_UNITY_BIN:-}" ]]; then
    PATTERN="$(basename "$RATSIM_UNITY_BIN")"
  else
    PATTERN="SARBench"
  fi
fi

echo "[kill_all_unity] pattern: $PATTERN  signal: $SIGNAL  dry_run: $DRY_RUN"

# pgrep -f matches the full command line, so this catches both the Unity
# binary itself AND the start_ratsim_headless.sh launcher that wraps it.
PIDS=$(pgrep -f "$PATTERN" || true)
if [[ -z "$PIDS" ]]; then
  echo "  no matching processes"
else
  echo "  matched processes:"
  for pid in $PIDS; do
    cmd=$(ps -o cmd= -p "$pid" 2>/dev/null | head -c 120)
    echo "    pid=$pid  $cmd"
  done
  if [[ "$DRY_RUN" -eq 0 ]]; then
    # shellcheck disable=SC2086
    kill $SIGNAL $PIDS 2>/dev/null || true
    echo "  signal sent."
  fi
fi

# Clean up stale pidfiles for already-dead pids. Skip live ones — those are
# being managed by stop_ratsim_headless.sh / the running scheduler.
shopt -s nullglob
for pf in /tmp/ratsim_*.pid; do
  pid=$(cat "$pf" 2>/dev/null)
  if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
    if [[ "$DRY_RUN" -eq 0 ]]; then
      rm -f "$pf"
    fi
    echo "  stale pidfile: $pf (pid=$pid)"
  fi
done
