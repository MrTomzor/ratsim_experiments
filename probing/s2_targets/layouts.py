"""Ground-truth maze layout and boxes of every dataset episode, checked against
what the episode itself recorded.

    RATSIM_UNITY_BIN=$RATSIM_ROOT/build_layout/ForagerSimBuild.x86_64 \\
    $PPO_PYTHON_PATH -m probing.s2_targets.layouts --dataset results/probing/full/dataset --spawn

Where the layout comes from. The training build cannot export its maze. The
layout-dump build (ratsim_metacentrum/build_layout, built from unity_project/:
the training build's own Unity source plus LayoutDump.cs) is reset once per
episode with the dataset's world presets and the episode's world_seed, and
publishes MazeLayoutLoader's wall mask and every live reward on
/sim_control/layout_dump. Same source, same Unity version, same Mono runtime,
hence the same world -- which is checked here, not assumed:

  * reset match: the dump build's reset pose and lidar ranges must equal the
    episode's step 0 (--reset_tol);
  * every lidar return must lie within HIT_TOL_M of a wall cell (wall_ok; a
    ray that clips a block's convex corner returns AT the corner) or, when the
    semantic channel says reward, on a box still there (reward_ok); every ray
    must pass through free cells up to its return (free_ok). The same wall
    test with the lidar angles mirrored is a negative control
    (wall_ok_mirrored): it must come out far lower, or the test is blind to
    the flip it is meant to catch;
  * the agent itself must stand in a free cell at every step (pose_ok);
  * every pickup must happen within --pickup_radius of a box still there.
Steps where the agent sits inside a collider (every ray reports the same
near-zero range: SemanticLidarSensor's embedded fallback) carry no geometry;
they are skipped and counted (embedded_frac).

Per episode, <out>/epXXXX.npz:

    wall          (H, W) float32  wall fraction of each 1 m cell of the
                                  ExplorationTracker grid (row = +y, col = +x,
                                  ROS frame), supersampled from Unity's mask
    box_xy        (B, 2) float32  box centres, ROS frame, metres
    box_half      (B, 2) float32  box half extents along ROS x / y
    mask          (mask_h, mask_w) bool  Unity's mask as dumped, [j, i]
    mask_origin, mask_cell        its placement (Unity x, z of mask[0, 0]'s corner)
    world_seed, reset_pose, reset_ranges, build_guid, semantic_classes

<out>/verify_report.json   every episode's checks; written even on failure
<out>/layouts.json         written last and only if every episode passed:
                           dataset_fingerprint, fingerprint (of the layout
                           files), build_guid, thresholds, summary
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from probing.core import dataset
from probing.s2_targets import targets

TOPIC = "/sim_control/layout_dump"
REWARD_CLASS = "reward_pickup"
NEVER = np.iinfo(np.int32).max      # box_timeline: "not collected / never hit"
HIT_TOL_M = 0.1                     # a wall return may sit this far off a wall cell
BOX_TOL_M = 0.15                    # a reward return may sit this far off the box
FREE_FROM_M, FREE_MARGIN_M, FREE_STEP_M = 0.6, 0.3, 0.25


# -- geometry -------------------------------------------------------------------

def dump_mask(dump: dict) -> np.ndarray:
    """(mask_h, mask_w) bool, [j, i]: j along Unity Z, i along Unity X."""
    chars = np.frombuffer(dump["mask"].encode(), np.uint8)
    return (chars == ord("1")).reshape(dump["mask_h"], dump["mask_w"])


def wall_fraction(mask, origin, cell, meta: dict, supersample: int = 4) -> np.ndarray:
    """Rasterise Unity's mask onto the ExplorationTracker grid: per 1 m cell the
    fraction of supersample x supersample points that fall on a wall. Unity X
    = -ROS y and Unity Z = ROS x (CoordConversion.cs). Outside the mask is
    outside the world, i.e. wall."""
    tr = targets.new_tracker(meta)
    off = (np.arange(supersample) + 0.5) / supersample
    xs = tr.origin_x + (np.arange(tr.cells_x)[:, None] + off).ravel() * tr.resolution
    ys = tr.origin_y + (np.arange(tr.cells_y)[:, None] + off).ravel() * tr.resolution
    i = np.floor((-ys[:, None] - origin[0]) / cell).astype(np.int64)  # Unity X
    j = np.floor((xs[None, :] - origin[1]) / cell).astype(np.int64)   # Unity Z
    i, j = np.broadcast_arrays(i, j)
    inside = (i >= 0) & (i < mask.shape[1]) & (j >= 0) & (j < mask.shape[0])
    wall = np.ones(i.shape, bool)
    wall[inside] = mask[j[inside], i[inside]]
    s = supersample
    return wall.reshape(tr.cells_y, s, tr.cells_x, s).mean((1, 3)).astype(np.float32)


def world_from_dump(dump: dict, meta: dict) -> dict:
    mask = dump_mask(dump)
    origin = np.array([dump["origin_x"], dump["origin_z"]], np.float64)
    boxes = dump["rewards"]
    return {
        "wall": wall_fraction(mask, origin, dump["cell_size"], meta),
        "box_xy": np.array([[b["z"], -b["x"]] for b in boxes], np.float32).reshape(-1, 2),
        "box_half": np.array([[b["size_z"] / 2, b["size_x"] / 2] for b in boxes],
                             np.float32).reshape(-1, 2),
        "mask": mask,
        "mask_origin": origin,
        "mask_cell": np.float64(dump["cell_size"]),
        "semantic_classes": np.array(dump["semantic_classes"], dtype=np.str_),
        "build_guid": np.str_(dump["build_guid"]),
    }


def is_wall(world: dict, meta: dict, x, y) -> np.ndarray:
    """Wall (any wall in the 1 m cell, or outside the grid) at world points."""
    tr = targets.new_tracker(meta)
    col = np.floor((np.asarray(x) - tr.origin_x) / tr.resolution).astype(np.int64)
    row = np.floor((np.asarray(y) - tr.origin_y) / tr.resolution).astype(np.int64)
    h, w = world["wall"].shape
    inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    out = np.ones(np.shape(x), bool)
    out[inside] = world["wall"][row[inside], col[inside]] > 0
    return out


def near_wall(world: dict, meta: dict, x, y, tol: float = HIT_TOL_M) -> np.ndarray:
    """A wall cell within `tol` (per axis) of each point."""
    out = np.zeros(np.shape(x), bool)
    for dx in (-tol, 0.0, tol):
        for dy in (-tol, 0.0, tol):
            out |= is_wall(world, meta, np.asarray(x) + dx, np.asarray(y) + dy)
    return out


def rays(ep: dict, meta: dict, reward_class: int, n_classes: int,
         mirrored: bool = False) -> dict:
    """Every ray of every non-embedded step: origin, world angle, range,
    whether it returned, whether the semantic channel says reward."""
    pose, ranges = ep["pose"].astype(np.float64), ep["ranges"].astype(np.float64)
    n_steps, n_rays = ranges.shape
    if ep["lidar"].shape[1] != n_rays * (1 + n_classes):
        raise ValueError(f"lidar width {ep['lidar'].shape[1]} != {n_rays} rays "
                         f"x (1 depth + {n_classes} semantic classes)")
    scan = targets.scan_geometry(meta)
    local = scan["angle_start_rad"] + np.arange(n_rays) * scan["angle_increment_rad"]
    if mirrored:
        local = -local
    embedded = (np.ptp(ranges, axis=1) < 1e-6) & (ranges[:, 0] > 0) & (ranges[:, 0] < 1.0)
    # Semantics are one-hot per ray, ray-major (SemanticLidarSensor descriptors).
    sem = ep["lidar"][:, n_rays:].reshape(n_steps, n_rays, n_classes)
    t = np.repeat(np.flatnonzero(~embedded), n_rays)
    k = np.tile(np.arange(n_rays), int((~embedded).sum()))
    r = ranges[t, k]
    return {"t": t, "ray": k, "x0": pose[t, 0], "y0": pose[t, 1],
            "angle": pose[t, 2] + local[k], "range": r,
            "hit": (r > 0) & (r < scan["max_range"]),
            "reward": sem[t, k, reward_class] > 0.5,
            "max_range": scan["max_range"], "embedded": embedded}


def _point(rs: dict, dist) -> tuple[np.ndarray, np.ndarray]:
    return (rs["x0"] + np.cos(rs["angle"]) * dist,
            rs["y0"] + np.sin(rs["angle"]) * dist)


def _segment_dist(p, a, b) -> np.ndarray:
    """Distance from points p (B, 2) to the segment a-b."""
    ab = b - a
    u = np.clip(((p - a) @ ab) / max(float(ab @ ab), 1e-12), 0.0, 1.0)
    return np.linalg.norm(p - (a + u[:, None] * ab), axis=1)


def assign_pickups(xy, collected, a, b, n: int, radius: float, t: int) -> tuple[list, list]:
    """A pickup of `n` boxes at step t: the n boxes still there (collected ==
    NEVER) nearest to the path a -> b; those within `radius` get collected = t
    (in place). Returns their distances and one t per pickup out of reach."""
    alive = np.flatnonzero(collected == NEVER)
    dist = _segment_dist(xy[alive], a, b)
    dists, unmatched = [], []
    for k in np.argsort(dist)[:n]:
        dists.append(float(dist[k]))
        if dist[k] <= radius:
            collected[alive[k]] = t
        else:
            unmatched.append(int(t))
    return dists, unmatched


def hit_matrix(hx, hy, xy, half) -> np.ndarray:
    """(hits, boxes): the hit point lies on the box (+BOX_TOL_M)."""
    return ((np.abs(hx[:, None] - xy[:, 0]) <= half[:, 0] + BOX_TOL_M)
            & (np.abs(hy[:, None] - xy[:, 1]) <= half[:, 1] + BOX_TOL_M))


def box_timeline(ep: dict, world: dict, rs: dict, radius: float) -> dict:
    """When each box was collected and first returned a lidar hit.

    A pickup at step t (the cumulative count rises) takes the boxes still
    there that are nearest to the path pose[t-1] -> pose[t]; the agent touches
    a box at a centre distance of 1.4-1.75 m, so --pickup_radius 2.0 m.
    A reward return at step t belongs to a box with collected_t >= t that
    contains the hit point (+BOX_TOL_M): Unity's Destroy is deferred to the
    end of the frame, so the box picked up at step t can still be seen at t."""
    xy, half = world["box_xy"].astype(np.float64), world["box_half"].astype(np.float64)
    pose, pickups = ep["pose"][:, :2].astype(np.float64), ep["pickups"].astype(np.int64)
    collected = np.full(len(xy), NEVER, np.int64)
    pickup_dist, unmatched = [], []
    for t in np.flatnonzero(np.diff(pickups) > 0) + 1:
        d, u = assign_pickups(xy, collected, pose[t - 1], pose[t],
                              int(pickups[t] - pickups[t - 1]), radius, int(t))
        pickup_dist += d
        unmatched += u

    first_hit = np.full(len(xy), NEVER, np.int64)
    sel = rs["hit"] & rs["reward"]
    hx, hy = _point({k: v[sel] for k, v in rs.items() if k in ("x0", "y0", "angle")},
                    rs["range"][sel])
    ts = rs["t"][sel]
    inside = hit_matrix(hx, hy, xy, half) & (collected[None, :] >= ts[:, None])
    for b in range(len(xy)):
        if inside[:, b].any():
            first_hit[b] = ts[inside[:, b]].min()
    return {"collected_t": collected, "first_hit_t": first_hit,
            "reward_hit_ok": inside.any(1), "pickup_dist": np.array(pickup_dist),
            "unmatched_pickups": unmatched}


def verify(ep: dict, world: dict, meta: dict, radius: float,
           detail: bool = False) -> dict:
    """The per-episode checks (module docstring). With detail=True also the
    per-ray arrays and the box timeline, for inspect_layouts."""
    classes = [str(c) for c in world["semantic_classes"]]
    rc = classes.index(REWARD_CLASS)
    rs = rays(ep, meta, rc, len(classes))
    boxes = box_timeline(ep, world, rs, radius)

    wall_rays = rs["hit"] & ~rs["reward"]
    hx, hy = _point(rs, rs["range"])
    wall_ok = near_wall(world, meta, hx[wall_rays], hy[wall_rays])
    mrs = rays(ep, meta, rc, len(classes), mirrored=True)
    mhx, mhy = _point(mrs, mrs["range"])
    mirrored_ok = near_wall(world, meta, mhx[wall_rays], mhy[wall_rays])

    # Free space: points along each ray up to just before its return.
    end = np.where(rs["hit"], rs["range"], rs["max_range"]) - FREE_MARGIN_M
    steps = np.arange(FREE_FROM_M, rs["max_range"], FREE_STEP_M)
    along = steps[None, :] <= end[:, None]
    ray_idx, step_idx = np.nonzero(along)
    fx, fy = _point({k: rs[k][ray_idx] for k in ("x0", "y0", "angle")}, steps[step_idx])
    free_bad = is_wall(world, meta, fx, fy)
    ray_free_ok = np.ones(len(rs["t"]), bool)
    np.logical_and.at(ray_free_ok, ray_idx, ~free_bad)

    frac = lambda ok: float(np.mean(ok)) if np.size(ok) else 1.0
    out = {
        "n_boxes": int(len(world["box_xy"])),
        "n_rays": int(len(rs["t"])),
        "wall_ok": frac(wall_ok),
        "wall_ok_mirrored": frac(mirrored_ok),
        "reward_ok": frac(boxes["reward_hit_ok"]),
        "n_reward_hits": int(np.size(boxes["reward_hit_ok"])),
        "free_ok": frac(ray_free_ok),
        "pose_ok": frac(~is_wall(world, meta, ep["pose"][:, 0], ep["pose"][:, 1])),
        "embedded_frac": float(rs["embedded"].mean()),
        "n_pickups": int(ep["pickups"][-1]) - int(ep["pickups"][0]),
        "pickups_at_reset": int(ep["pickups"][0]),
        "max_pickup_dist": float(boxes["pickup_dist"].max()) if len(boxes["pickup_dist"]) else 0.0,
        "unmatched_pickups": boxes["unmatched_pickups"],
        "boxes_seen": int((boxes["first_hit_t"] < NEVER).sum()),
    }
    if detail:
        cls = np.full(len(rs["t"]), "free_ray", dtype=object)
        cls[wall_rays] = np.where(wall_ok, "wall_ok", "wall_bad")
        rew = rs["hit"] & rs["reward"]
        cls[rew] = np.where(boxes["reward_hit_ok"], "reward_ok", "reward_bad")
        cls[~rs["hit"]] = "no_return"
        cls[~ray_free_ok] = "free_bad"
        out["detail"] = {"rays": rs, "hit_xy": _point(rs, rs["range"]),
                         "ray_class": cls, "boxes": boxes}
    return out


def check(stats: dict, limits: dict) -> list[str]:
    """Reasons an episode fails; empty if it passes."""
    bad = []
    for key in ("wall_ok", "reward_ok", "free_ok", "pose_ok"):
        if stats[key] < limits[f"min_{key}"]:
            bad.append(f"{key} {stats[key]:.4f} < {limits[f'min_{key}']}")
    if stats["wall_ok_mirrored"] > stats["wall_ok"] - limits["min_mirror_gap"]:
        bad.append(f"mirrored control {stats['wall_ok_mirrored']:.3f} too close to "
                   f"wall_ok {stats['wall_ok']:.3f}: the test cannot see a flip")
    if stats["unmatched_pickups"]:
        bad.append(f"pickups with no box within reach at t={stats['unmatched_pickups'][:5]}")
    if stats["pickups_at_reset"]:
        bad.append(f"{stats['pickups_at_reset']} pickups at reset")
    if limits["expect_boxes"] is not None and stats["n_boxes"] != limits["expect_boxes"]:
        bad.append(f"{stats['n_boxes']} boxes, expected {limits['expect_boxes']}")
    if not stats.get("reset_match", True):
        bad.append(f"reset differs from the dataset's step 0 "
                   f"(pose {stats['reset_pose_err']:.2e}, ranges {stats['reset_ranges_err']:.2e})")
    return bad


# -- Unity ----------------------------------------------------------------------

def binary_guid(binary) -> str | None:
    """build-guid from a Unity player's <name>_Data/boot.config."""
    binary = Path(binary)
    boot = binary.with_name(binary.stem + "_Data") / "boot.config"
    if not boot.exists():
        return None
    for line in boot.read_text().splitlines():
        if line.startswith("build-guid="):
            return line.split("=", 1)[1].strip()
    return None


def make_env(meta: dict, port: int):
    """The collection env (collect.make_env) without the dreamer wrapper and
    with the dump switched on. No metaseed: the seed is passed per reset."""
    from ratsim.config_blender import blend_presets
    from ratsim_wildfire_gym_env.env import WildfireGymEnv
    presets = meta["presets"]
    return WildfireGymEnv(
        worldgen_config={**blend_presets("world", presets["world"]),
                         "layout_dump/enabled": 1},
        agent_config=blend_presets("agents", presets["agent"]),
        sensor_config={},
        action_config={"control_mode": "velocity"},
        task_config=blend_presets("task", presets["task"]),
        metaworldgen_config=None,
        unity_port=port,
    )


def fetch(env, seed: int) -> tuple[dict, np.ndarray, np.ndarray]:
    """Reset to `seed`; the dump plus the reset pose and ranges."""
    from ratsim.transforms import yaw_from_quat
    env.reset(options={"seed": int(seed)})
    conn = env.conn
    dumps = conn.get_received_messages(TOPIC)
    if not dumps:
        raise RuntimeError(f"no {TOPIC} in Unity's reply: is this the layout-dump "
                           f"build (build_layout/), not the training build?")
    dump = json.loads(dumps[-1].data)
    if int(dump["seed"]) != int(seed):
        raise RuntimeError(f"dump is of seed {dump['seed']}, asked for {seed}")
    p = conn.get_received_messages(env.agent_gt_pose_msg_in_topic)[0]
    lidar = conn.get_received_messages(env.lidar_msg_in_topic)[0]
    pose = np.array([p.x, p.y, yaw_from_quat(p.qx, p.qy, p.qz, p.qw)], np.float32)
    return dump, pose, np.asarray(lidar.ranges, np.float32)


def fetch_all(root: Path, out: Path, meta: dict, todo: list[int], args) -> None:
    from unity_attach import resolve_unity_port
    expected_guid = binary_guid(os.environ.get("RATSIM_UNITY_BIN", ""))
    env = make_env(meta, resolve_unity_port(args, tag="probe-layouts"))
    try:
        for e in todo:
            ep = dataset.load_episode(dataset.episode_path(root, e))
            seed = int(ep["world_seed"])
            dump, pose, ranges = fetch(env, seed)
            if expected_guid and dump["build_guid"] != expected_guid:
                raise RuntimeError(f"Unity on the port is build {dump['build_guid']}, "
                                   f"RATSIM_UNITY_BIN is {expected_guid}: a stale instance?")
            world = world_from_dump(dump, meta)
            world.update(world_seed=np.int64(seed), reset_pose=pose, reset_ranges=ranges)
            dataset.save_npz(out / f"ep{e:04d}.npz", world)
            print(f"[probe-layouts] ep{e:04d} seed {seed}: {len(dump['rewards'])} boxes, "
                  f"walls {100 * world['wall'].mean():.1f}%")
    finally:
        env.close()


def _verify_one(job):
    root, out, e, meta, radius, tol = job
    ep = dataset.load_episode(dataset.episode_path(root, e))
    world = dataset.load_episode(out / f"ep{e:04d}.npz")
    if int(world["world_seed"]) != int(ep["world_seed"]):
        raise ValueError(f"ep{e:04d}: layout of seed {int(world['world_seed'])}, "
                         f"episode has {int(ep['world_seed'])}")
    stats = verify(ep, world, meta, radius)
    stats["reset_pose_err"] = float(np.abs(world["reset_pose"] - ep["pose"][0]).max())
    stats["reset_ranges_err"] = float(np.abs(world["reset_ranges"] - ep["ranges"][0]).max())
    stats["reset_match"] = max(stats["reset_pose_err"], stats["reset_ranges_err"]) <= tol
    stats["source"] = str(ep["source"])
    stats["build_guid"] = str(world["build_guid"])
    stats["mask_sha"] = hashlib.sha256(np.packbits(world["mask"]).tobytes()).hexdigest()[:16]
    return e, stats


def main() -> None:
    from unity_attach import add_unity_attach_args
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", default=None, help="default: <dataset>/layouts")
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    help="only these (a smoke run: layouts.json is never written)")
    ap.add_argument("--verify_only", action="store_true",
                    help="no Unity: verify the layouts already on disk")
    ap.add_argument("--workers", type=int, default=4, help="verification processes")
    ap.add_argument("--pickup_radius", type=float, default=2.0)
    ap.add_argument("--reset_tol", type=float, default=1e-3)
    ap.add_argument("--min_wall_ok", type=float, default=0.99)
    ap.add_argument("--min_reward_ok", type=float, default=0.99)
    ap.add_argument("--min_free_ok", type=float, default=0.99)
    ap.add_argument("--min_pose_ok", type=float, default=0.99)
    ap.add_argument("--min_mirror_gap", type=float, default=0.2)
    ap.add_argument("--expect_boxes", type=int, default=None)
    add_unity_attach_args(ap)
    args = ap.parse_args()

    root = Path(args.dataset)
    out = Path(args.out or root / "layouts")
    fp = dataset.verify(root)  # sealed, complete, unchanged
    meta = dataset.load_json(root / "meta.json")
    n = len(dataset.list_episodes(root))
    episodes = list(range(n)) if args.episodes is None else args.episodes
    limits = {k: getattr(args, k) for k in ("min_wall_ok", "min_reward_ok", "min_free_ok",
                                            "min_pose_ok", "min_mirror_gap", "expect_boxes")}
    with dataset.lock(out / ".layouts.lock"):
        if args.episodes is None:  # a smoke run never rewrites the marker, so keeps it
            (out / "layouts.json").unlink(missing_ok=True)  # incomplete until rewritten
        for stale in out.glob(".*.tmp"):
            stale.unlink()
        todo = [e for e in episodes if not (out / f"ep{e:04d}.npz").exists()]
        if todo and args.verify_only:
            sys.exit(f"[probe-layouts] --verify_only, but {len(todo)} layouts missing")
        if todo:
            fetch_all(root, out, meta, todo, args)

        jobs = [(root, out, e, meta, args.pickup_radius, args.reset_tol) for e in episodes]
        with ProcessPoolExecutor(args.workers) as pool:
            report = dict(pool.map(_verify_one, jobs))
        failed = {}
        for e, stats in report.items():
            stats["fail"] = check(stats, limits)
            if stats["fail"]:
                failed[e] = stats["fail"]
        dataset.save_json(out / "verify_report.json",
                          {"dataset_fingerprint": fp, "limits": limits,
                           "pickup_radius": args.pickup_radius, "reset_tol": args.reset_tol,
                           "episodes": {f"ep{e:04d}": s for e, s in report.items()}})
        summarise(report)
        if failed:
            for e, why in list(failed.items())[:20]:
                print(f"[probe-layouts] FAIL ep{e:04d}: {'; '.join(why)}")
            sys.exit(f"[probe-layouts] {len(failed)}/{len(report)} episodes failed; "
                     f"see {out / 'verify_report.json'} and probing.tools.inspect_layouts")
        if args.episodes is not None:
            print("[probe-layouts] smoke run passed; layouts.json not written")
            return

        guids = {s["build_guid"] for s in report.values()}
        masks = [s["mask_sha"] for s in report.values()]
        seeds = [int(dataset.load_episode(dataset.episode_path(root, e))["world_seed"])
                 for e in episodes]
        if len(guids) != 1:
            sys.exit(f"[probe-layouts] layouts from {len(guids)} different builds: {guids}")
        if len(set(seeds)) != len(seeds) or len(set(masks)) != len(masks):
            sys.exit("[probe-layouts] two episodes share a world: the episode split "
                     "would leak a maze from train into test")
        files = [out / f"ep{e:04d}.npz" for e in episodes]
        # Written last: its presence means every layout is there and verified.
        dataset.save_json(out / "layouts.json", {
            "dataset_fingerprint": fp,
            "fingerprint": dataset.fingerprint_files(files),
            "build_guid": guids.pop(),
            "limits": limits, "pickup_radius": args.pickup_radius,
            "reset_tol": args.reset_tol, "n_episodes": len(files)})
        print(f"[probe-layouts] {len(files)} layouts verified -> {out}")


def summarise(report: dict) -> None:
    keys = ("wall_ok", "wall_ok_mirrored", "reward_ok", "free_ok", "pose_ok", "embedded_frac",
            "max_pickup_dist", "reset_pose_err", "reset_ranges_err")
    rows = np.array([[s[k] for k in keys] for s in report.values()])
    print(f"[probe-layouts] {len(report)} episodes   " + "   ".join(
        f"{k} min {rows[:, i].min():.4g} / med {np.median(rows[:, i]):.4g} / max {rows[:, i].max():.4g}"
        for i, k in enumerate(keys)))


if __name__ == "__main__":
    main()
