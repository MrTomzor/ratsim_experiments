"""Per episode of a probing dataset: seen area / explorable area of ITS maze,
overall and per arm (4 orthogonal sectors beyond r=25 m of the centre), by
replaying pose + lidar through the reward's own ExplorationTracker.
Explorable = free cells 4-connected to the spawn + the wall cells bordering them
(true layout from layouts/ep*.npz); every seen cell lies inside it.

    python -m probing.tools.coverage_arms <dataset dir> <out.json>
"""
import glob, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
from scipy import ndimage
from probing.s2_targets import targets
from ratsim.task_tracker.exploration_tracker import UNKNOWN

# Set by main() from argv before the pool forks; module-level so the workers
# (fork) see them without pickling the dataset meta per task.
DS = meta = scan = None

def explorable(lay):
    wall = lay["wall"] >= 0.5
    tr = targets.new_tracker(meta)
    x, y, _ = lay["reset_pose"]
    c, r = tr.world_to_cell(x, y)
    lab, _ = ndimage.label(~wall)
    reach = lab == lab[r, c]
    return reach | (wall & ndimage.binary_dilation(reach, np.ones((3, 3), bool)))

H = W = 200
yy, xx = np.mgrid[0:H, 0:W]
cx, cy = xx - W / 2, yy - H / 2
RAD = np.hypot(cx, cy)
SEC = np.where(np.abs(cx) > np.abs(cy), np.where(cx > 0, 0, 2), np.where(cy > 0, 1, 3))

def one(f):
    ep = dict(np.load(f, allow_pickle=False))
    lay = np.load(DS / "layouts" / Path(f).name)
    assert int(lay["world_seed"]) == int(ep["world_seed"])
    tr = targets.new_tracker(meta)
    for (x, y, yaw), rg in zip(ep["pose"].astype(np.float64), ep["ranges"]):
        tr.update_from_lidar(x, y, yaw, rg, **scan)
    seen = tr.grid != UNKNOWN
    ex = explorable(lay)
    arms = [float((seen & ex & (RAD > 25) & (SEC == s)).sum() / max((ex & (RAD > 25) & (SEC == s)).sum(), 1)) for s in range(4)]
    run = str(ep["policy_run"]) if str(ep["source"]) == "policy" else "explorer"
    return {"run": run, "seen": int(seen.sum()), "seen_in_ex": int((seen & ex).sum()), "explorable": int(ex.sum()),
            "arms": arms, "pickups": int(ep["pickups"].max()) if ep["pickups"].size else 0}

def main():
    global DS, meta, scan
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    DS, out = Path(sys.argv[1]), sys.argv[2]
    meta = json.load(open(DS / "meta.json"))
    scan = targets.scan_geometry(meta)
    files = sorted(glob.glob(str(DS / "episodes" / "ep*.npz")))
    with Pool(14) as p:
        res = p.map(one, files, chunksize=4)
    json.dump(res, open(out, "w"))
    print("episodes", len(res))


if __name__ == "__main__":
    main()
