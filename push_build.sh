#!/usr/bin/env bash
#
# Push the local Unity build to the RCI cluster, so the jobs there run the
# simulator you just built. The mirror image of ./pull_run.sh.
#
#   ./push_build.sh                     # ~/ForagerSimBuildV1 -> rci:/mnt/personal/$USER/ForagerSimBuildV1
#   ./push_build.sh -n                  # dry run: show every file that would change
#   ./push_build.sh -s ~/OtherBuild     # a different local build dir
#   ./push_build.sh -f                  # push anyway while jobs are running (read the warning first)
#   ./push_build.sh --no-delete         # add/overwrite only; leave remote-only files alone
#
# Default destination: rci:/mnt/personal/<remote-user>/<basename of source>, which
# is exactly what rci_env.sh points RATSIM_UNITY_BIN at:
#
#     /mnt/personal/$USER/ForagerSimBuildV1/ForagerSimBuildV1.x86_64
#
# WHY THE REMOTE ROOT IS RESOLVED OVER SSH: the local user is not the cluster
# user (tom vs musilto8), and rsync does NOT expand `$USER` in a remote path --
# it quotes the argument for the remote shell, so a literal /mnt/personal/$USER
# fails with a confusing "change_dir ... No such file or directory". One ssh
# up front asks the remote side who it is, and everything after uses that.
#
# WHY --delete IS THE DEFAULT: this is a whole-build mirror, not an update.
# A Unity build directory is a unit -- Assembly-CSharp.dll, the level*/shared-
# assets* blobs and ScriptingAssemblies.json have to come from the same build.
# Leaving a file behind from an older build is how you get a remote sim that
# loads a scene the new code no longer matches. The guard is that the script
# refuses to --delete into anything that does not already look like a Unity
# build dir (no *.x86_64 inside), so a typo'd -d cannot mirror-delete a data
# directory.
#
# WHY IT REFUSES WHILE JOBS ARE QUEUED OR RUNNING: the scheduler launches a
# fresh Unity instance per env, throughout the job -- not once at the start.
# Already-running instances keep the old inode (rsync writes a temp file and
# renames), but every instance launched after the push gets the new build, so a
# single experiment would silently span two simulators. Wait for the jobs, or
# push to a versioned directory (-d) and point that def's RATSIM_UNITY_BIN there.

set -euo pipefail

HOST="rci"
SRC="$HOME/ForagerSimBuildV1"
DEST=""                   # empty => /mnt/personal/<remote-user>/<basename SRC>
REMOTE_ROOT=""            # empty => /mnt/personal/<remote-user>
DELETE=1
FORCE=0
DRY_RUN=0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# For the staleness check only; a missing project is not an error.
UNITY_ASSETS="$SCRIPT_DIR/../ratsim_unity_project/RatsimUnityProject/Assets"

usage() {
    cat <<EOF
usage: $(basename "$0") [options]

  -s, --src DIR             local build directory. Default: $SRC
  -d, --dest PATH           absolute remote build dir. Default:
                            <remote-root>/<basename of --src>
  -R, --remote-root PATH    remote parent dir. Default: /mnt/personal/\$USER
  -H, --host HOST           ssh host. Default: $HOST
      --no-delete           do not remove remote-only files (default is a mirror)
  -f, --force               push even when SLURM jobs are queued/running, and
                            even when the destination does not look like a build
  -n, --dry-run             show what would transfer; change nothing
  -h, --help                this message
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        -s|--src)         SRC="$2"; shift 2 ;;
        -d|--dest)        DEST="$2"; shift 2 ;;
        -R|--remote-root) REMOTE_ROOT="$2"; shift 2 ;;
        -H|--host)        HOST="$2"; shift 2 ;;
        --no-delete)      DELETE=0; shift ;;
        -f|--force)       FORCE=1; shift ;;
        -n|--dry-run)     DRY_RUN=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *)                echo "unexpected argument: $1" >&2; usage >&2; exit 2 ;;
    esac
done

SRC="${SRC%/}"
BUILD_NAME="$(basename "$SRC")"

# --- local preflight --------------------------------------------------------
if [ ! -d "$SRC" ]; then
    echo "[push_build] no such build directory: $SRC" >&2
    exit 2
fi
# A Unity Linux player is <name>.x86_64 + UnityPlayer.so + <name>_Data/. Check
# for all three: pushing half a build wastes 88 MB and fails at launch, not here.
BIN_PATH="$(find "$SRC" -maxdepth 1 -name '*.x86_64' -type f | head -1)"
if [ -z "$BIN_PATH" ] || [ ! -f "$SRC/UnityPlayer.so" ] || [ ! -d "$SRC/${BUILD_NAME}_Data" ]; then
    echo "[push_build] $SRC does not look like a Unity Linux build" >&2
    echo "[push_build] expected <name>.x86_64, UnityPlayer.so and ${BUILD_NAME}_Data/ inside it" >&2
    exit 2
fi
if [ ! -x "$BIN_PATH" ]; then
    echo "[push_build] WARNING: $(basename "$BIN_PATH") is not executable locally; rsync -a will copy that bit as-is" >&2
fi

# Newest file in the build, used both for the report and the staleness check.
# Not the .x86_64 stub -- an incremental rebuild rewrites _Data/ and leaves the
# launcher stub untouched, so the stub's mtime can be weeks older than the build.
BUILD_NEWEST="$(find "$SRC" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1)"
BUILD_NEWEST_PATH="${BUILD_NEWEST#* }"
BUILD_MTIME="$(date -d "@${BUILD_NEWEST%% *}" '+%Y-%m-%d %H:%M' 2>/dev/null || echo '?')"
BUILD_SIZE="$(du -sh "$SRC" | cut -f1)"

echo "[push_build] src:  $SRC  ($BUILD_SIZE, newest file $BUILD_MTIME)"

# --- is the build stale? ----------------------------------------------------
# Warning only. The script cannot know whether those edits matter, but a build
# older than the C# it is supposed to contain is worth one line of noise.
if [ -d "$UNITY_ASSETS" ] && [ -n "$BUILD_NEWEST_PATH" ]; then
    NEWER_CS="$(find "$UNITY_ASSETS" -name '*.cs' -newer "$BUILD_NEWEST_PATH" 2>/dev/null | wc -l | tr -d ' ')"
    if [ "$NEWER_CS" -gt 0 ]; then
        echo "[push_build] WARNING: $NEWER_CS C# file(s) under $UNITY_ASSETS are newer than this build."
        echo "[push_build]          You may be pushing a build that predates your latest Unity changes."
        find "$UNITY_ASSETS" -name '*.cs' -newer "$BUILD_NEWEST_PATH" 2>/dev/null \
            | head -5 | sed 's|^|             |'
        if [ "$NEWER_CS" -gt 5 ]; then echo "             ... and $((NEWER_CS - 5)) more"; fi
    fi
fi

# --- remote preflight -------------------------------------------------------
# One ssh for everything the remote side knows: who we are, what is already at
# the destination, and whether any of our jobs are live.
REMOTE_PROBE=$(cat <<'EOS'
root="$1"; name="$2"; dest="$3"
me="${USER:-$(id -un)}"
if [ -z "$root" ]; then root="/mnt/personal/$me"; fi
if [ -z "$dest" ]; then dest="$root/$name"; fi
echo "PUSH_USER=$me"
echo "PUSH_DEST=$dest"
if [ -d "$dest" ]; then
    echo "PUSH_EXISTS=1"
    n=$(ls -1 "$dest"/*.x86_64 2>/dev/null | wc -l)
    echo "PUSH_ISBUILD=$n"
    echo "PUSH_OLD_SIZE=$(du -sh "$dest" 2>/dev/null | cut -f1)"
    echo "PUSH_OLD_MTIME=$(find "$dest" -type f -printf '%TY-%Tm-%Td %TH:%TM\n' 2>/dev/null | sort | tail -1)"
else
    echo "PUSH_EXISTS=0"
    echo "PUSH_ISBUILD=0"
    parent=$(dirname "$dest")
    if [ ! -d "$parent" ]; then echo "PUSH_NOPARENT=$parent"; fi
fi
if command -v squeue >/dev/null 2>&1; then
    echo "PUSH_JOBS=$(squeue -u "$me" -h -t RUNNING,PENDING 2>/dev/null | wc -l)"
    squeue -u "$me" -h -t RUNNING,PENDING -o 'PUSH_JOB=%.10i %.9P %.30j %.2t %.10M' 2>/dev/null
else
    echo "PUSH_JOBS=?"
fi
EOS
)

echo "[push_build] probing $HOST ..."
PROBE_OUT="$(ssh "$HOST" "bash -s -- '$REMOTE_ROOT' '$BUILD_NAME' '$DEST'" <<<"$REMOTE_PROBE")"

get() { echo "$PROBE_OUT" | sed -n "s/^$1=//p" | tail -1; }
REMOTE_USER="$(get PUSH_USER)"
RESOLVED_DEST="$(get PUSH_DEST)"
EXISTS="$(get PUSH_EXISTS)"
ISBUILD="$(get PUSH_ISBUILD)"
OLD_SIZE="$(get PUSH_OLD_SIZE)"
OLD_MTIME="$(get PUSH_OLD_MTIME)"
NOPARENT="$(get PUSH_NOPARENT)"
JOBS="$(get PUSH_JOBS)"

if [ -z "$RESOLVED_DEST" ]; then
    echo "[push_build] could not resolve the remote destination (ssh $HOST failed?)" >&2
    exit 1
fi
if [ -n "$NOPARENT" ]; then
    echo "[push_build] remote parent directory does not exist: $NOPARENT" >&2
    echo "[push_build] create it there first, or pass -R/-d" >&2
    exit 1
fi

if [ "$EXISTS" = "1" ]; then
    echo "[push_build] dest: $HOST:$RESOLVED_DEST  (replacing $OLD_SIZE, newest file ${OLD_MTIME:-?})"
else
    echo "[push_build] dest: $HOST:$RESOLVED_DEST  (new directory)"
fi

# --- the two refusals -------------------------------------------------------
if [ "$DELETE" = "1" ] && [ "$EXISTS" = "1" ] && [ "${ISBUILD:-0}" = "0" ] && [ "$FORCE" = "0" ]; then
    echo "[push_build] REFUSING: $RESOLVED_DEST exists but has no *.x86_64 in it, so it is" >&2
    echo "[push_build] probably not a build directory -- and --delete would mirror-delete" >&2
    echo "[push_build] whatever is in there. Use --no-delete, fix -d, or -f to override." >&2
    exit 1
fi

if [ "${JOBS:-0}" != "0" ] && [ "$JOBS" != "?" ] && [ "$FORCE" = "0" ] && [ "$DRY_RUN" = "0" ]; then
    echo "[push_build] REFUSING: $JOBS SLURM job(s) of yours are queued or running:" >&2
    echo "$PROBE_OUT" | sed -n 's/^PUSH_JOB=/             /p' >&2
    echo "[push_build]" >&2
    echo "[push_build] Those jobs launch a fresh Unity instance per env as they go, so a" >&2
    echo "[push_build] push now would leave one experiment straddling two builds. Wait for" >&2
    echo "[push_build] them, or push to a versioned dir (-d /mnt/personal/$REMOTE_USER/${BUILD_NAME}_new)" >&2
    echo "[push_build] and repoint RATSIM_UNITY_BIN. -f overrides if you know it is fine." >&2
    exit 1
fi
if [ "${JOBS:-0}" != "0" ] && [ "$JOBS" != "?" ]; then
    echo "[push_build] NOTE: $JOBS job(s) queued/running (proceeding: ${DRY_RUN:+dry run}${FORCE:+--force})"
fi

# --- transfer ---------------------------------------------------------------
# -a keeps the executable bits (without them Unity is just an unlaunchable
# 39 MB blob on the far side). -z earns its keep here: the .so and the assets
# blobs are the bulk, and RCI is a WAN hop away.
RSYNC_ARGS=(-az --info=stats1,progress2 --partial)
if [ "$DELETE" = "1" ]; then
    RSYNC_ARGS+=(--delete)
fi
if [ "$DRY_RUN" = "1" ]; then
    echo "[push_build] DRY RUN -- rsync -n, itemising changes"
    RSYNC_ARGS+=(-n -i)
fi

# Trailing slash on the source: copy the CONTENTS into $RESOLVED_DEST, rather
# than nesting a second ForagerSimBuildV1/ inside it.
rsync "${RSYNC_ARGS[@]}" "$SRC/" "$HOST:$RESOLVED_DEST/"

if [ "$DRY_RUN" = "1" ]; then
    echo "[push_build] dry run only -- nothing was written."
    exit 0
fi

# --- verify -----------------------------------------------------------------
BIN_NAME="$(basename "$BIN_PATH")"
echo
echo "[push_build] verifying ..."
ssh "$HOST" "bash -s -- '$RESOLVED_DEST' '$BIN_NAME'" <<'EOS'
dest="$1"; bin="$2"
if [ ! -x "$dest/$bin" ]; then
    echo "[push_build] VERIFY FAILED: $dest/$bin missing or not executable" >&2
    exit 1
fi
echo "[push_build] remote: $(du -sh "$dest" | cut -f1) in $(find "$dest" -type f | wc -l) files"
ls -l "$dest/$bin"
EOS

echo
echo "Next: the cluster picks this up automatically -- rci_env.sh already exports"
echo
echo "  RATSIM_UNITY_BIN=$RESOLVED_DEST/$BIN_NAME"
echo
echo "so a job submitted from meta_ratsim/rci_port_probes runs the build you just pushed:"
echo
echo "  ./submit.sh <def> --time 4h"
echo
echo "If you pushed to a non-default -d, export RATSIM_UNITY_BIN in the job instead."
