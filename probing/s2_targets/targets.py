"""Probe targets, computed once per dataset (not per checkpoint).

    $PPO_PYTHON_PATH -m probing.s2_targets.targets --dataset results/probing/pilot/dataset

The seen-map is not stored by collect.py; it is rebuilt here by replaying every
step's raw lidar scan and true pose through ratsim's own ExplorationTracker --
the class the volumetric reward is computed with -- so it is that map exactly,
plus the reset observation's scan, which TaskTracker skips but the agent saw.
The field of view at step t is the scan's fan -- every cell whose centre lies
within the bearing span of the rays and closer than the range interpolated
between the two neighbouring rays -- plus the cells the rays themselves touch
(a second tracker, reset every step). The fan matters: 19 rays 10 degrees
apart leave gaps wider than a 2 m cell beyond ~11 m, and a cell between two
free rays is something a model WITHOUT memory can infer, so it must not count
as memory.

Per sampled step t (every --stride steps), written to <out>/epXXXX.npz:

    t           (n,)          step index into the episode
    map_label   (n, 16, 16)   int8  0 unknown / 1 free / 2 wall; a 32 x 32 m
                              window of 2 m cells, centred on the agent, axes
                              of the world (row = +y, col = +x); a coarse cell
                              is wall if any 1 m cell in it is, else free if
                              any is, else unknown
    map_inview  (n, 16, 16)   bool  some 1 m cell of it is in the current fan
    map_age     (n, 16, 16)   int32 steps since any of its cells was last in a
                              scan; -1 = never seen
    geo_start   (n,)          float32 shortest path from the start cell to the
                              agent's cell over seen-free cells and cells the
                              agent has been in, metres; NaN if they do not
                              connect. Visited cells count because the seen
                              map never clears OCCUPIED: a wall return landing
                              in a 1 m cell the agent drives through would
                              otherwise cut its own path.
    euc_start   (n,)          float32 straight-line distance to start, metres;
                              equals |gps| * 300 and is in the agent's input,
                              so it is a positive control, not a memory test

The memory cells the headline metric uses are `map_label != 0 & ~map_inview`:
known, and not visible now -- the only cells a model without memory cannot
get right except by chance.
"""
from __future__ import annotations

import argparse
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ratsim.task_tracker.exploration_tracker import (FREE, OCCUPIED, UNKNOWN,
                                                     ExplorationTracker)

from probing.core import dataset

WINDOW_CELLS = 32   # 1 m cells per side -> 32 m
# Memory-age bins in steps (10 steps = 1 s): how long ago a cell was last in
# view (map_age, last_seen). Every scored or plotted age uses these edges.
AGE_BINS = [1, 10, 30, 100, 300, 1000, 3000, 10_000]
COARSE = 2          # 2 m cells -> 16 x 16
LABEL_UNKNOWN, LABEL_FREE, LABEL_WALL = 0, 1, 2


def blocks(fine: np.ndarray) -> np.ndarray:
    """(h, w) fine grid -> (h/COARSE, COARSE, w/COARSE, COARSE) view of its 2 m cells."""
    h, w = fine.shape
    return fine.reshape(h // COARSE, COARSE, w // COARSE, COARSE)


def coarse_label(fine: np.ndarray) -> np.ndarray:
    b = blocks(fine)
    wall = (b == OCCUPIED).any((1, 3))
    free = (b == FREE).any((1, 3))
    return np.where(wall, LABEL_WALL,
                    np.where(free, LABEL_FREE, LABEL_UNKNOWN)).astype(np.int8)


def coarse_any(fine: np.ndarray) -> np.ndarray:
    return blocks(fine).any((1, 3))


def coarse_max(fine: np.ndarray) -> np.ndarray:
    return blocks(fine).max((1, 3))


def window(grid: np.ndarray, row: int, col: int, fill,
           size: int = WINDOW_CELLS) -> np.ndarray:
    """size x size slice with the agent's cell at the centre corner
    (row, col) -> [half, half]; out-of-grid cells read `fill`. The same crop as
    global_score.IndexedTargets (maps padded by size // 2, corner = agent cell)."""
    half = size // 2
    out = np.full((size, size), fill, grid.dtype)
    r0, c0 = row - half, col - half
    rs, cs = max(r0, 0), max(c0, 0)
    re = min(r0 + size, grid.shape[0])
    ce = min(c0 + size, grid.shape[1])
    if rs < re and cs < ce:
        out[rs - r0:re - r0, cs - c0:ce - c0] = grid[rs:re, cs:ce]
    return out


def geodesic(free: np.ndarray, start: tuple[int, int], goal: tuple[int, int],
             resolution: float = 1.0) -> float:
    """Shortest 8-connected path over `free` cells (no corner cutting), metres."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra

    h, w = free.shape
    inside = lambda rc: 0 <= rc[0] < h and 0 <= rc[1] < w
    # A negative index would wrap to the far side of the grid, silently.
    if not (inside(start) and inside(goal) and free[start] and free[goal]):
        return float("nan")
    ids = np.full(free.shape, -1, np.int64)
    ids[free] = np.arange(int(free.sum()))
    rows, cols, wts = [], [], []

    def link(a, b, ok, w):
        rows.append(ids[a][ok]); cols.append(ids[b][ok])
        wts.append(np.full(int(ok.sum()), w))

    s = np.s_
    link(s[:, :-1], s[:, 1:], free[:, :-1] & free[:, 1:], 1.0)
    link(s[:-1, :], s[1:, :], free[:-1, :] & free[1:, :], 1.0)
    diag = math.sqrt(2.0)
    link(s[:-1, :-1], s[1:, 1:], free[:-1, :-1] & free[1:, 1:]
         & free[:-1, 1:] & free[1:, :-1], diag)
    link(s[:-1, 1:], s[1:, :-1], free[:-1, 1:] & free[1:, :-1]
         & free[:-1, :-1] & free[1:, 1:], diag)
    n = int(free.sum())
    graph = coo_matrix((np.concatenate(wts),
                        (np.concatenate(rows), np.concatenate(cols))),
                       shape=(n, n)).tocsr()
    dist = dijkstra(graph, directed=False, indices=int(ids[start]))
    d = dist[int(ids[goal])]
    return float(d * resolution) if np.isfinite(d) else float("nan")


def fan(tracker, x, y, yaw, ranges, angle_start_rad, angle_increment_rad,
        max_range) -> np.ndarray:
    """Bool grid: cells whose centre is inside the current scan's fan."""
    res = tracker.resolution
    h, w = tracker.grid.shape
    col, row = tracker.world_to_cell(x, y)
    n = int(math.ceil(max_range / res)) + 1
    r0, r1 = max(row - n, 0), min(row + n + 1, h)
    c0, c1 = max(col - n, 0), min(col + n + 1, w)
    out = np.zeros((h, w), bool)
    if r0 >= r1 or c0 >= c1:
        return out
    cy = tracker.origin_y + (np.arange(r0, r1) + 0.5) * res
    cx = tracker.origin_x + (np.arange(c0, c1) + 0.5) * res
    dx, dy = cx[None, :] - x, cy[:, None] - y
    dist = np.hypot(dx, dy)
    bearing = (np.arctan2(dy, dx) - yaw + np.pi) % (2 * np.pi) - np.pi
    ranges = np.asarray(ranges, np.float64)
    reach = np.where((ranges > 0) & (ranges < max_range), ranges, max_range)
    angles = angle_start_rad + np.arange(len(ranges)) * angle_increment_rad
    order = np.argsort(angles)
    limit = np.interp(bearing, angles[order], reach[order],
                      left=np.nan, right=np.nan)
    with np.errstate(invalid="ignore"):
        inside = (dist <= limit + 0.5 * res) | (dist < 0.5 * res)
    out[r0:r1, c0:c1] = inside
    return out


def scan_geometry(meta: dict) -> dict:
    """The tracker's lidar keyword arguments. TaskTracker._update_exploration
    negates Unity's lidar angles (Unity's angle grows clockwise); the same
    conversion, so every map built here agrees with the reward's."""
    lg = meta["lidar_geometry"]
    return dict(angle_start_rad=-math.radians(lg["angle_start_deg"]),
                angle_increment_rad=-math.radians(lg["angle_increment_deg"]),
                max_range=lg["max_range"])


def new_tracker(meta: dict) -> ExplorationTracker:
    g = meta["grid"]
    return ExplorationTracker(g["world_width"], g["world_height"], g["resolution"])


def replay(ep: dict, meta: dict):
    """Step through an episode's scans; after integrating step t's scan yield
    (t, seen, inview, last_seen): the seen-map tracker, the 1 m cells in the
    current field of view (rays + fan), and per 1 m cell the last step it was
    in view (-1 = never). The arrays are updated in place: copy to keep."""
    scan = scan_geometry(meta)
    seen, view = new_tracker(meta), new_tracker(meta)
    last_seen = np.full(seen.grid.shape, -1, np.int32)
    pose, ranges = ep["pose"].astype(np.float64), ep["ranges"]
    for t in range(len(pose)):
        x, y, yaw = pose[t]
        seen.update_from_lidar(x, y, yaw, ranges[t], **scan)
        view.reset()
        view.update_from_lidar(x, y, yaw, ranges[t], **scan)
        inview = (view.grid != UNKNOWN) | fan(seen, x, y, yaw, ranges[t], **scan)
        last_seen[inview] = t
        yield t, seen, inview, last_seen


def episode_targets(ep: dict, meta: dict, stride: int) -> dict:
    pose = ep["pose"].astype(np.float64)
    geometry = new_tracker(meta)
    visited = np.zeros(geometry.grid.shape, bool)
    sc, sr = geometry.world_to_cell(pose[0, 0], pose[0, 1])
    out = {k: [] for k in ("t", "map_label", "map_inview", "map_age",
                           "geo_start", "euc_start")}
    for t, seen, inview, last_seen in replay(ep, meta):
        x, y, yaw = pose[t]
        col, row = seen.world_to_cell(x, y)
        if 0 <= row < visited.shape[0] and 0 <= col < visited.shape[1]:
            visited[row, col] = True
        if t % stride:
            continue
        last = coarse_max(window(last_seen, row, col, -1))
        out["t"].append(t)
        out["map_label"].append(coarse_label(window(seen.grid, row, col, UNKNOWN)))
        out["map_inview"].append(coarse_any(window(inview, row, col, False)))
        out["map_age"].append(np.where(last >= 0, t - last, -1).astype(np.int32))
        out["geo_start"].append(geodesic((seen.grid == FREE) | visited,
                                         (sr, sc), (row, col), seen.resolution))
        out["euc_start"].append(math.hypot(x - pose[0, 0], y - pose[0, 1]))
    return {"t": np.asarray(out["t"], np.int32),
            "map_label": np.stack(out["map_label"]),
            "map_inview": np.stack(out["map_inview"]),
            "map_age": np.stack(out["map_age"]),
            "geo_start": np.asarray(out["geo_start"], np.float32),
            "euc_start": np.asarray(out["euc_start"], np.float32)}


def _one(job):
    path, out_path, meta, stride = job
    tg = episode_targets(dataset.load_episode(path), meta, stride)
    dataset.save_episode(out_path, tg)
    known = tg["map_label"] != LABEL_UNKNOWN
    memory = known & ~tg["map_inview"]
    return (path.name, len(tg["t"]), float(memory.mean()),
            float(np.isnan(tg["geo_start"]).mean()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None, help="default: <dataset>/targets")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    root = Path(args.dataset)
    out = Path(args.out or root / "targets")
    fp = dataset.verify(root)  # sealed, complete, unchanged
    meta = dataset.load_json(root / "meta.json")
    (out / "targets.json").unlink(missing_ok=True)  # incomplete until rewritten
    jobs = [(p, out / p.name, meta, args.stride)
            for p in dataset.list_episodes(root)]
    with ProcessPoolExecutor(args.workers) as pool:
        for name, n, mem, nan in pool.map(_one, jobs):
            print(f"[probe-targets] {name}: {n} samples, "
                  f"memory cells {100 * mem:.1f}% of window, "
                  f"geo_start NaN {100 * nan:.1f}%")
    # Written last: its presence means every episode's targets are complete,
    # and dataset_fingerprint says which version of the episodes they are of.
    dataset.save_json(out / "targets.json", {"stride": args.stride,
                                             "window_m": WINDOW_CELLS,
                                             "cell_m": COARSE,
                                             "dataset_fingerprint": fp})


if __name__ == "__main__":
    main()
