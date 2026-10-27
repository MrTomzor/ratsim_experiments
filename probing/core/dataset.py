"""On-disk layout of a probing dataset, and the only code that writes it.

    <root>/meta.json             what produced the dataset (presets, metaseed,
                                 the episode plan, lidar and grid geometry,
                                 and the fingerprint once it is complete)
    <root>/spaces.json           the agent's obs/act spaces, so encode.py can
                                 rebuild the agent without a simulator
    <root>/episodes/ep0000.npz   one episode per file

An episode with N observations (index 0 = the reset observation) stores, per
step t:

    lidar, gps, compass         the observation exactly as the agent saw it
    reward, is_first, is_last, is_terminal
    action_0, action_1          the action taken AFTER observing step t; the
                                last row has none and holds 0. Dreamer's
                                prevact for step t is therefore action[t-1].
    pose                        (x, y, yaw) from /<agent>/gt_pose: metres, ROS
                                frame, origin = arena centre. Privileged; never
                                an agent input.
    ranges                      raw lidar ranges in metres (-1 = no return),
                                from the same Unity reply as the observation
    pickups                     cumulative objects collected

plus per-episode scalars: source, policy_run, policy_stage, world_seed, and
the lidar/grid geometry.

Files never contain object arrays: they load with allow_pickle=False in any
numpy (the dreamer venv pins numpy<2, the main venv runs numpy 2).

Concurrency. Several processes write here at once (parallel collect workers,
parallel probes), and a job can be resubmitted while an earlier one is still
alive. The rules every writer follows:

  * Every file is written to a temp name unique to the writer (pid + thread),
    which no glob of real files can match, then renamed over the target.
    rename() is atomic on one filesystem, so a reader sees the old file or
    the new one, never a truncated one, and two writers of the same file never
    share a temp file.
  * spaces.json is published create-if-absent (link(), which fails if the
    target exists): the first worker wins, every later one must match it.
  * A complete dataset is SEALED with a fingerprint over every episode file.
    targets.py, encode.py and train_probe.py refuse an unsealed dataset or a
    fingerprint that no longer matches, and record the fingerprint they used,
    so states and targets computed from two versions of an episode can never
    be paired silently.
  * lock() gives one process exclusive use of a directory (flock; on NFSv4
    Linux maps it to a POSIX lock, which the server enforces across nodes).
"""
from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import re
import threading
from pathlib import Path

import numpy as np

OBS_KEYS = ("lidar", "gps", "compass")
ACT_KEYS = ("action_0", "action_1")
STEP_KEYS = (*OBS_KEYS, "reward", "is_first", "is_last", "is_terminal",
             *ACT_KEYS, "pose", "ranges", "pickups")
EPISODE_RE = re.compile(r"^ep(\d{4})\.npz$")


def episode_dir(root) -> Path:
    return Path(root) / "episodes"


def episode_path(root, idx: int) -> Path:
    return episode_dir(root) / f"ep{idx:04d}.npz"


def episode_id(path) -> int:
    m = EPISODE_RE.match(Path(path).name)
    if not m:
        raise ValueError(f"not an episode file: {path}")
    return int(m.group(1))


def list_episodes(root) -> list[Path]:
    """Episode files only -- never a temp file left by a killed writer."""
    d = episode_dir(root)
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if EPISODE_RE.match(p.name))


# -- atomic writes --------------------------------------------------------------

def _tmp_name(path: Path) -> Path:
    # Leading dot and trailing .tmp: matches neither EPISODE_RE nor *.npz.
    return path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")


def _write_atomic(path, write) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_name(path)
    try:
        with open(tmp, "wb") as f:
            write(f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _check_no_objects(arrays: dict) -> None:
    for key, value in arrays.items():
        if np.asarray(value).dtype == object:
            raise TypeError(f"{key}: object arrays would need allow_pickle")


def save_npz(path, arrays: dict) -> None:
    _check_no_objects(arrays)
    _write_atomic(path, lambda f: np.savez_compressed(f, **arrays))


save_episode = save_npz


def load_episode(path) -> dict:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def save_json(path, obj) -> None:
    data = json.dumps(obj, indent=2, default=str).encode()
    _write_atomic(path, lambda f: f.write(data))


def load_json(path):
    return json.loads(Path(path).read_text())


def publish_once(path, obj) -> bool:
    """Write `obj` to `path` unless it exists; then it must equal `obj`.

    Returns True if this call created the file. link() is atomic and fails if
    the target exists, so of N racing processes exactly one creates it and
    the rest compare against complete content."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_name(path)
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    try:
        os.link(tmp, path)
        return True
    except FileExistsError:
        existing = load_json(path)
        if existing != json.loads(json.dumps(obj, default=str)):
            raise ValueError(f"{path} exists with different content")
        return False
    finally:
        tmp.unlink(missing_ok=True)


@contextlib.contextmanager
def lock(path):
    """Exclusive, non-blocking: a second holder fails at once, it never waits
    behind a job that may run for hours."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                f"{path} is held by another process -- a second job on the "
                f"same output? Wait for it, or use another name.") from None
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# -- sealing ------------------------------------------------------------------

def fingerprint_files(paths) -> str:
    """sha256 over the files' names and contents, in the order given."""
    h = hashlib.sha256()
    for p in paths:
        h.update(Path(p).name.encode())
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def fingerprint(root) -> str:
    """sha256 over every episode file's name and content."""
    return fingerprint_files(list_episodes(root))


def seal(root) -> str:
    """Record the fingerprint of a complete dataset in meta.json."""
    root = Path(root)
    meta = load_json(root / "meta.json")
    meta["fingerprint"] = fingerprint(root)
    meta["n_episodes"] = len(list_episodes(root))
    save_json(root / "meta.json", meta)
    return meta["fingerprint"]


def verify(root) -> str:
    """Fingerprint of a sealed, complete, unchanged dataset; raises otherwise.

    Complete = episodes are exactly ep0000..ep{n-1} (encode.py, targets.py and
    train_probe.py all index episodes by that id) and, when the dataset has a
    plan, n is the plan's length."""
    root = Path(root)
    meta = load_json(root / "meta.json")
    ids = [episode_id(p) for p in list_episodes(root)]
    if ids != list(range(len(ids))):
        missing = sorted(set(range(max(ids, default=-1) + 1)) - set(ids))
        raise ValueError(f"{root}: episode ids not contiguous, missing {missing[:5]}")
    if "plan" in meta and len(ids) != len(meta["plan"]):
        raise ValueError(f"{root}: {len(ids)} episodes, plan has {len(meta['plan'])}")
    if "fingerprint" not in meta:
        raise ValueError(f"{root}: not sealed (collect did not finish)")
    fp = fingerprint(root)
    if fp != meta["fingerprint"]:
        raise ValueError(f"{root}: episodes changed since the dataset was sealed")
    return fp


def spaces_to_json(spaces: dict) -> dict:
    """elements.Space dict -> plain JSON (dtype, shape, low, high)."""
    return {k: {"dtype": np.dtype(s.dtype).str, "shape": list(s.shape),
                "low": np.asarray(s.low).tolist(),
                "high": np.asarray(s.high).tolist()}
            for k, s in spaces.items()}


def spaces_from_json(obj: dict) -> dict:
    import elements
    out = {}
    for k, s in obj.items():
        dtype = np.dtype(s["dtype"])
        out[k] = elements.Space(dtype, tuple(s["shape"]),
                                np.asarray(s["low"], dtype),
                                np.asarray(s["high"], dtype))
    return out


def split_episodes(n: int, seed: int = 0, fracs=(0.7, 0.15, 0.15)) -> dict:
    """Episode indices per split. Splitting by episode, never by step: every
    episode is a fresh maze, so a step-level split would let the probe see
    the test maze during training."""
    order = np.random.default_rng(seed).permutation(n)
    n_train = int(round(fracs[0] * n))
    n_val = max(1, int(round(fracs[1] * n))) if n >= 3 else 0
    return {"train": np.sort(order[:n_train]),
            "val": np.sort(order[n_train:n_train + n_val]),
            "test": np.sort(order[n_train + n_val:])}


def is_current(output, root) -> bool:
    """True if `output` (targets.json, a states .npz or a metrics.json)
    exists and records the fingerprint the dataset is sealed with now. Reads
    meta.json only -- no re-hash -- so the job script can call it per file.

    An output derived from another derived product (a global-target probe from
    gtargets/, gtargets from layouts/) also records `<name>_fingerprint`; each
    must equal the `fingerprint` in <root>/<name>/<name>.json now."""
    output = Path(output)
    try:
        sealed = load_json(Path(root) / "meta.json").get("fingerprint")
        if output.suffix == ".npz":
            with np.load(output, allow_pickle=False) as f:
                made_from = str(f["dataset_fingerprint"])
            record = {}
        else:
            record = load_json(output)
            made_from = record.get("dataset_fingerprint")
        for key, value in record.items():
            name = key.removesuffix("_fingerprint")
            if key.endswith("_fingerprint") and name != "dataset":
                if load_json(Path(root) / name / f"{name}.json").get("fingerprint") != value:
                    return False
    except (OSError, KeyError, ValueError):
        return False
    return sealed is not None and made_from == sealed


if __name__ == "__main__":
    # python -m probing.core.dataset current <dataset> <output>  -> exit 0 if current
    import sys
    if len(sys.argv) == 4 and sys.argv[1] == "current":
        sys.exit(0 if is_current(sys.argv[3], sys.argv[2]) else 1)
    sys.exit("usage: python -m probing.core.dataset current <dataset> <output>")
