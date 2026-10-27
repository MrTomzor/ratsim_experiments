"""Global probe targets from the ground-truth layouts: the whole maze and the
boxes still in it, on a fixed world grid, plus what the agent had seen.

    $PPO_PYTHON_PATH -m probing.s2_targets.global_targets --dataset results/probing/full/dataset

Needs <dataset>/layouts (probing.s2_targets.layouts, verified). The grid is the
ExplorationTracker grid (row = +y, col = +x, ROS frame) in COARSE x COARSE m
cells: 100 x 100 cells of 2 m for the 200 x 200 m orthomaze. The seen-map and
field of view are targets.py's (the same replay), so "memory" means the same
thing for the old window probes and these.

Per episode, <out>/epXXXX.npz, n = samples every --stride steps:

    t                (n,)         step index into the episode
    gage             (n, H, W)    uint16  steps since some 1 m cell of the cell
                                  was last in the field of view; 0 = in view
                                  now, AGE_NEVER = never
    gknown_first     (H, W)       int32   first sampled step at which the
                                  seen-map knew some 1 m cell of it (free or
                                  wall); -1 = never
    glayout          (H, W)       float32 wall fraction of the cell (the four
                                  1 m cells' fractions averaged)
    box_cell         (B,)         int32   flat index (row * W + col) of each box
    box_collected_t  (B,)         int32   step the box was picked up; NEVER = not
    box_first_hit_t  (B,)         int32   first step a lidar ray returned off
                                  it (the semantic channel); NEVER = never

Masks at sample k (train_probe): inview = gage == 0; known = 0 <= gknown_first
<= t; memory = known & ~inview; never_seen = ~known & ~inview. The box map at
t is the cells of boxes with box_collected_t > t; a box is observed at t once
box_first_hit_t <= t.

<out>/gtargets.json is written last: stride, grid, dataset_fingerprint,
layouts_fingerprint and the fingerprint of these files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ratsim.task_tracker.exploration_tracker import UNKNOWN

from probing.core import dataset
from probing.s2_targets import layouts, targets

COARSE = targets.COARSE
AGE_NEVER = np.iinfo(np.uint16).max
NEVER = np.iinfo(np.int32).max


def coarse_rc(xy: np.ndarray, meta: dict) -> np.ndarray:
    """(n, 2) int64 (row, col) of the coarse cells holding the (n, 2) points;
    float32 points stay float32 here, as in every caller."""
    tr = targets.new_tracker(meta)
    cell = COARSE * tr.resolution
    return np.stack([np.floor((xy[:, 1] - tr.origin_y) / cell),
                     np.floor((xy[:, 0] - tr.origin_x) / cell)], 1).astype(np.int64)


def glayout(world: dict) -> np.ndarray:
    """(H, W) wall fraction of each coarse cell (mean of its 1 m cells)."""
    return targets.blocks(world["wall"]).mean((1, 3)).astype(np.float32)


def box_cells(world: dict, meta: dict) -> np.ndarray:
    tr = targets.new_tracker(meta)
    cell = COARSE * tr.resolution
    col = np.floor((world["box_xy"][:, 0] - tr.origin_x) / cell).astype(np.int64)
    row = np.floor((world["box_xy"][:, 1] - tr.origin_y) / cell).astype(np.int64)
    w = tr.cells_x // COARSE
    if ((row < 0) | (row >= tr.cells_y // COARSE) | (col < 0) | (col >= w)).any():
        raise ValueError("a box lies outside the grid")
    return (row * w + col).astype(np.int32)


def episode_global_targets(ep: dict, world: dict, meta: dict, stride: int,
                           pickup_radius: float) -> dict:
    classes = [str(c) for c in world["semantic_classes"]]
    rs = layouts.rays(ep, meta, classes.index(layouts.REWARD_CLASS), len(classes))
    boxes = layouts.box_timeline(ep, world, rs, pickup_radius)
    if boxes["unmatched_pickups"]:
        raise ValueError(f"pickups with no box within reach at t={boxes['unmatched_pickups'][:5]}")

    ts, ages, known_first = [], [], None
    for t, seen, inview, last_seen in targets.replay(ep, meta):
        if t % stride:
            continue
        last = targets.coarse_max(last_seen)
        known = targets.coarse_any(seen.grid != UNKNOWN)
        if known_first is None:
            known_first = np.full(known.shape, -1, np.int32)
        known_first[known & (known_first < 0)] = t
        ages.append(np.where(last >= 0, np.minimum(t - last, AGE_NEVER - 1),
                             AGE_NEVER).astype(np.uint16))
        ts.append(t)
    to_i32 = lambda a: np.minimum(a, NEVER).astype(np.int32)
    return {"t": np.asarray(ts, np.int32),
            "gage": np.stack(ages),
            "gknown_first": known_first,
            "glayout": glayout(world),
            "box_cell": box_cells(world, meta),
            "box_collected_t": to_i32(boxes["collected_t"]),
            "box_first_hit_t": to_i32(boxes["first_hit_t"])}


def _one(job):
    root, layouts_dir, out, e, meta, stride, radius = job
    ep = dataset.load_episode(dataset.episode_path(root, e))
    world = dataset.load_episode(layouts_dir / f"ep{e:04d}.npz")
    tg = episode_global_targets(ep, world, meta, stride, radius)
    dataset.save_npz(out / f"ep{e:04d}.npz", tg)
    memory = (tg["gknown_first"][None] >= 0) & (tg["gknown_first"][None] <= tg["t"][:, None, None]) \
        & (tg["gage"] > 0)
    return (e, len(tg["t"]), float(memory[-1].mean()),
            int((tg["box_first_hit_t"] < NEVER).sum()),
            int((tg["box_collected_t"] < NEVER).sum()))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None, help="default: <dataset>/gtargets")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    root = Path(args.dataset)
    out = Path(args.out or root / "gtargets")
    layouts_dir = root / "layouts"
    fp = dataset.verify(root)  # sealed, complete, unchanged
    meta = dataset.load_json(root / "meta.json")
    lay = dataset.load_json(layouts_dir / "layouts.json")  # absent = not verified
    files = [layouts_dir / p.name for p in dataset.list_episodes(root)]
    if lay["dataset_fingerprint"] != fp or dataset.fingerprint_files(files) != lay["fingerprint"]:
        raise SystemExit("[probe-gtargets] layouts are not of this dataset version, or "
                         "changed since they were verified: re-run probing.s2_targets.layouts")
    with dataset.lock(out / ".gtargets.lock"):
        (out / "gtargets.json").unlink(missing_ok=True)  # incomplete until rewritten
        jobs = [(root, layouts_dir, out, dataset.episode_id(p), meta, args.stride,
                 lay["pickup_radius"]) for p in dataset.list_episodes(root)]
        with ProcessPoolExecutor(args.workers) as pool:
            for e, n, mem, seen, got in pool.map(_one, jobs):
                print(f"[probe-gtargets] ep{e:04d}: {n} samples, memory cells at the end "
                      f"{100 * mem:.1f}% of the map, boxes observed {seen}, collected {got}")
        h, w = (s // COARSE for s in targets.new_tracker(meta).grid.shape)
        # Written last: its presence means every episode's targets are complete.
        dataset.save_json(out / "gtargets.json", {
            "stride": args.stride, "cell_m": COARSE * meta["grid"]["resolution"],
            "grid_shape": [h, w], "age_never": int(AGE_NEVER), "never": int(NEVER),
            "dataset_fingerprint": fp, "layouts_fingerprint": lay["fingerprint"],
            "fingerprint": dataset.fingerprint_files(
                [out / p.name for p in dataset.list_episodes(root)])})
        print(f"[probe-gtargets] {len(jobs)} episodes -> {out}")


if __name__ == "__main__":
    main()
