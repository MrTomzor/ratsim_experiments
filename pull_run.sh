#!/usr/bin/env bash
#
# Pull checkpoints for one experiment def from the RCI cluster to this machine,
# so you can watch the trained policy in the Unity Editor.
#
#   ./pull_run.sh memory_orthomaze                       # latest ckpt, every method
#   ./pull_run.sh memory_orthomaze -m dreamer            # only dreamer runs
#   ./pull_run.sh memory_orthomaze -m ppo,recurrent_ppo -a   # every stage ckpt
#   ./pull_run.sh memory_orthomaze -A                    # whole run dirs (replay incl.)
#   ./pull_run.sh memory_orthomaze -o /data/ckpts        # custom destination
#
# Default destination: results/rci/<def>/  (mirrors the remote
# results/experiments/<def>/ layout, so eval_one_run.py can be pointed at it
# directly -- see the "next steps" the script prints when it finishes).
#
# WHY A FILE LIST INSTEAD OF --include RULES: picking "the latest checkpoint"
# needs to know which stage_K is the highest one with a matching stage_K.done,
# and only the remote side knows that. So we ssh once, build an explicit
# relative-path list there, and feed it to rsync --files-from. It also means a
# dreamer run costs ~8 MB (one stage dir) instead of ~480 MB, because
# dreamer_logdir/replay never enters the list unless you ask for -A.
#
# The .done markers are ALWAYS pulled (they are empty files). eval_one_run.py
# decides which checkpoint to load from them: final.zip only when every stage
# of the def is done, else the highest stage_K with a .done sibling. Pulling
# all the markers but only the latest payload keeps that decision consistent
# with what actually landed on disk.

set -euo pipefail

HOST="rci"
REMOTE_ROOT=""            # empty => /mnt/personal/$USER/git/ratsim_experiments/results/experiments (resolved remotely)
METHODS=""                # empty => all
MODE="latest"             # latest | all | full
DEST=""
DRY_RUN=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
    cat <<EOF
usage: $(basename "$0") <def> [options]

  <def>                     experiment id, e.g. memory_orthomaze
                            (remote results/experiments/<def>/)

  -m, --methods LIST        comma-separated methods to pull, matched against
                            the method field of <variation>__<method>__seed<N>
                            run dirs. Default: all methods.
  -a, --all-checkpoints     pull every stage checkpoint, not just the latest.
  -A, --all-data            pull the ENTIRE run directories -- replay buffer,
                            tensorboard, scheduler logs. Hundreds of MB per
                            dreamer run. Off by default.
  -o, --dest DIR            destination. Default: $SCRIPT_DIR/results/rci/<def>
  -H, --host HOST           ssh host. Default: $HOST
  -R, --remote-root PATH    remote results/experiments dir. Default:
                            /mnt/personal/\$USER/git/ratsim_experiments/results/experiments
  -n, --dry-run             list what would be pulled; transfer nothing.
  -h, --help                this message.
EOF
}

DEF=""
while [ $# -gt 0 ]; do
    case "$1" in
        -m|--methods)       METHODS="$2"; shift 2 ;;
        -a|--all-checkpoints) MODE="all"; shift ;;
        -A|--all-data)      MODE="full"; shift ;;
        -o|--dest)          DEST="$2"; shift 2 ;;
        -H|--host)          HOST="$2"; shift 2 ;;
        -R|--remote-root)   REMOTE_ROOT="$2"; shift 2 ;;
        -n|--dry-run)       DRY_RUN=1; shift ;;
        -h|--help)          usage; exit 0 ;;
        -*)                 echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
        *)
            if [ -n "$DEF" ]; then
                echo "unexpected extra argument: $1" >&2; exit 2
            fi
            DEF="$1"; shift ;;
    esac
done

if [ -z "$DEF" ]; then
    usage >&2; exit 2
fi
# -a and -A are mutually exclusive in effect; -A wins and says so.
if [ "$MODE" = "full" ]; then
    :
fi
if [ -z "$DEST" ]; then
    DEST="$SCRIPT_DIR/results/rci/$DEF"
fi

# Strip whitespace from the methods list so `-m "ppo, dreamer"` works.
METHODS="$(echo "$METHODS" | tr -d '[:space:]')"

# --- remote listing ---------------------------------------------------------
# Emits one relative path per line on stdout (relative to <root>/<def>/), plus
# PULLRUN_* diagnostics on stderr. Deliberately POSIX-ish: login1 is CentOS 7.
REMOTE_LISTER=$(cat <<'EOS'
root="$1"; def="$2"; methods="$3"; mode="$4"
if [ -z "$root" ]; then
    root="/mnt/personal/${USER:-$(id -un)}/git/ratsim_experiments/results/experiments"
fi
exp="$root/$def"
if [ ! -d "$exp" ]; then
    echo "PULLRUN_ERR: no such experiment dir: $exp" >&2
    if [ -d "$root" ]; then
        echo "PULLRUN_ERR: available:" >&2
        ls -1 "$root" >&2
    fi
    exit 3
fi
for f in experiment.yaml state.json; do
    if [ -f "$exp/$f" ]; then echo "$f"; fi
done
if [ ! -d "$exp/runs" ]; then
    echo "PULLRUN_ERR: no runs/ under $exp" >&2
    exit 3
fi
matched=0
for rd in "$exp"/runs/*/; do
    if [ ! -d "$rd" ]; then continue; fi
    run=$(basename "$rd")
    # run id is <variation>__<method>__seed<N>: method is the field between
    # the last two '__' separators (so cnn_recurrent_ppo survives intact).
    case "$run" in
        *__*__*) m=${run%__*}; m=${m##*__} ;;
        *)       m="" ;;
    esac
    if [ -n "$methods" ]; then
        case ",$methods," in
            *",$m,"*) : ;;
            *) continue ;;
        esac
    fi
    matched=$((matched + 1))
    if [ "$mode" = "full" ]; then
        echo "runs/$run"
        continue
    fi
    if [ -f "$rd/run_config.json" ]; then echo "runs/$run/run_config.json"; fi
    if [ ! -d "$rd/checkpoints" ]; then
        echo "PULLRUN_WARN: $run has no checkpoints/ yet" >&2
        continue
    fi
    # .done markers are empty files; always take all of them.
    for p in "$rd"checkpoints/*.done; do
        if [ -e "$p" ]; then echo "runs/$run/checkpoints/$(basename "$p")"; fi
    done
    payload=0
    if [ "$mode" = "all" ]; then
        for p in "$rd"checkpoints/*; do
            case "$p" in *.done) continue ;; esac
            if [ -e "$p" ]; then
                echo "runs/$run/checkpoints/$(basename "$p")"
                payload=$((payload + 1))
            fi
        done
    else
        # final.zip (sb3) / final/ (dreamer): whole-run snapshot, small.
        for p in "$rd"checkpoints/final.zip "$rd"checkpoints/final; do
            if [ -e "$p" ]; then
                echo "runs/$run/checkpoints/$(basename "$p")"
                payload=$((payload + 1))
            fi
        done
        best=""
        for dn in "$rd"checkpoints/stage_*.done; do
            if [ ! -e "$dn" ]; then continue; fi
            k=$(basename "$dn"); k=${k#stage_}; k=${k%.done}
            case "$k" in ''|*[!0-9]*) continue ;; esac
            if [ -z "$best" ] || [ "$k" -gt "$best" ]; then best="$k"; fi
        done
        if [ -n "$best" ]; then
            for p in "$rd"checkpoints/stage_$best.zip "$rd"checkpoints/stage_$best; do
                if [ -e "$p" ]; then
                    echo "runs/$run/checkpoints/$(basename "$p")"
                    payload=$((payload + 1))
                fi
            done
        fi
    fi
    if [ "$payload" -eq 0 ]; then
        echo "PULLRUN_WARN: $run has no completed checkpoint (no final, no stage_K with a .done) -- nothing to eval" >&2
    fi
done
if [ "$matched" -eq 0 ]; then
    echo "PULLRUN_ERR: no run dirs matched methods='$methods' under $exp/runs" >&2
    echo "PULLRUN_ERR: present:" >&2
    ls -1 "$exp/runs" >&2
    exit 4
fi
echo "PULLRUN_INFO: root=$root matched_runs=$matched mode=$mode" >&2
EOS
)

LIST_FILE="$(mktemp -t pull_run_files.XXXXXX)"
ERR_FILE="$(mktemp -t pull_run_err.XXXXXX)"
trap 'rm -f "$LIST_FILE" "$ERR_FILE"' EXIT

echo "[pull_run] listing $HOST:${REMOTE_ROOT:-<default>}/$DEF (methods=${METHODS:-all}, mode=$MODE)"
if ! ssh "$HOST" "bash -s -- '$REMOTE_ROOT' '$DEF' '$METHODS' '$MODE'" \
        <<<"$REMOTE_LISTER" >"$LIST_FILE" 2>"$ERR_FILE"; then
    cat "$ERR_FILE" >&2
    echo "[pull_run] remote listing failed (see PULLRUN_ERR above)" >&2
    exit 1
fi
cat "$ERR_FILE" >&2

N_PATHS=$(wc -l <"$LIST_FILE" | tr -d ' ')
if [ "$N_PATHS" -eq 0 ]; then
    echo "[pull_run] nothing to pull." >&2
    exit 1
fi

# Take the root the lister actually resolved, rather than re-deriving it here:
# rsync does NOT expand `$USER` in a remote path (it quotes the arg for the
# remote shell), so a literal /mnt/personal/$USER/... fails with a confusing
# "change_dir ... No such file or directory".
RESOLVED_ROOT=$(sed -n 's/^PULLRUN_INFO: root=\([^ ]*\).*/\1/p' "$ERR_FILE" | tail -1)
if [ -z "$RESOLVED_ROOT" ]; then
    RESOLVED_ROOT="$REMOTE_ROOT"
fi
if [ -z "$RESOLVED_ROOT" ]; then
    echo "[pull_run] could not resolve the remote root from the listing" >&2
    exit 1
fi
SRC="$HOST:$RESOLVED_ROOT/$DEF/"

echo "[pull_run] $N_PATHS path(s) selected:"
sed 's/^/           /' "$LIST_FILE"
echo "[pull_run] src:  $SRC"
echo "[pull_run] dest: $DEST"

# -r is NOT redundant with -a here: --files-from turns recursion off, and -a's
# implied -r does not count as "explicitly specified". Without it a dreamer
# checkpoint (a DIRECTORY holding agent.pkl) arrives as an empty dir and the
# transfer looks like a suspiciously fast success.
RSYNC_ARGS=(-azr --info=stats1,progress2 --partial --files-from="$LIST_FILE")
if [ "$DRY_RUN" -eq 1 ]; then
    echo "[pull_run] DRY RUN -- rsync -n"
    RSYNC_ARGS+=(-n)
fi

mkdir -p "$DEST"
rsync "${RSYNC_ARGS[@]}" "$SRC" "$DEST/"

if [ "$DRY_RUN" -eq 1 ]; then
    exit 0
fi

echo
echo "[pull_run] pulled $(du -sh "$DEST" | cut -f1) into $DEST"
echo
echo "Next: press Play in the Unity Editor (it listens on :9000). The eval"
echo "scripts attach to it and WAIT for it -- they never spawn a build unless"
echo "you pass --spawn -- so you can start them in either order. Then:"
echo
for rd in "$DEST"/runs/*/; do
    [ -d "$rd" ] || continue
    run=$(basename "$rd")
    case "$run" in
        *__*__*) m=${run%__*}; m=${m##*__} ;;
        *)       m="" ;;
    esac
    case "$m" in
        dreamer)
            echo "  \$DREAMER_PYTHON_PATH $SCRIPT_DIR/eval_one_run_dreamer.py \\"
            ;;
        *)
            echo "  python $SCRIPT_DIR/eval_one_run.py \\"
            ;;
    esac
    echo "      --run_dir $rd --exp_dir $DEST --n_episodes 5"
done
echo
echo "  # or every pulled run in sequence, plus the plots:"
echo "  python $SCRIPT_DIR/analyze_experiment.py $DEST --run-eval 5"
