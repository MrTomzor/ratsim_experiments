"""Tests for the ground-truth probes: probing/s2_targets/layouts.py, global_targets.py,
global_score.py and their use in train_probe / analyze / the inspect CLIs
(main venv: scipy, torch, sklearn, matplotlib).

    ~/ratvenv/venv/bin/python tests/test_probe_global.py

The world is NOT square (60 m along ROS x, 40 m along ROS y): on the real
200 x 200 m orthomaze a transposed or mirrored layout has the right shape and
only the verification against the lidar would catch it. Episodes are
synthesised by a ray-marcher in Unity's lidar convention, so verify() is
tested against geometry it did not produce.
"""
from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probing.core import dataset  # noqa: E402
from probing.s4_probe import global_score as gs  # noqa: E402
from probing.s2_targets import global_targets, layouts, targets  # noqa: E402

META = {"grid": {"world_width": 40.0, "world_height": 60.0, "resolution": 1.0},
        "lidar_geometry": {"angle_start_deg": -90.0, "angle_increment_deg": 10.0,
                           "max_range": 20.0},
        "presets": {"world": [], "agent": [], "task": []}}
CLASSES = ["reward_pickup", "arena_wall"]
BOX_HALF = 0.8825


def _wall_ros(i: int) -> np.ndarray:
    """(40 rows = ROS y, 60 cols = ROS x): border + one block, placed per episode."""
    w = np.zeros((40, 60), bool)
    w[0, :] = w[-1, :] = w[:, 0] = w[:, -1] = True
    w[25:35, 30 + i:33 + i] = True          # y in [5, 15), x in [i, i + 3)
    return w


def _dump(i: int, boxes_ros) -> dict:
    """What LayoutDump.cs would publish for that world: mask[j, i] with i along
    Unity X = -ROS y and j along Unity Z = ROS x (the mapping under test)."""
    mask = _wall_ros(i)[::-1, :].T          # (60 = Unity Z, 40 = Unity X)
    return {"seed": 1000 + i, "build_guid": "test", "mask_w": 40, "mask_h": 60,
            "cell_size": 1.0, "origin_x": -20.0, "origin_z": -30.0,
            "mask": "".join("1" if v else "0" for v in mask.ravel()),
            "rewards": [{"x": -y, "z": x, "size_x": 2 * BOX_HALF, "size_z": 2 * BOX_HALF}
                        for x, y in boxes_ros],
            "semantic_classes": CLASSES}


def _march(world, x, y, a, boxes, alive, max_range=20.0, step=0.005):
    """Range and whether a box was hit, along world angle a (5 mm steps)."""
    d = np.arange(0.0, max_range, step)
    px, py = x + math.cos(a) * d, y + math.sin(a) * d
    stop = layouts.is_wall(world, META, px, py)
    box = np.zeros(len(d), bool)
    for k, (bx, by) in enumerate(boxes):
        if alive[k]:
            box |= (np.abs(px - bx) <= BOX_HALF) & (np.abs(py - by) <= BOX_HALF)
    first = np.flatnonzero(stop | box)
    if not len(first):
        return -1.0, False
    return float(d[first[0]]), bool(box[first[0]])


def _episode(i: int, n: int = 61, embedded_at=(15,)) -> tuple[dict, dict]:
    """Agent drives along y = 0 (+x, 0.5 m/step) and passes box 0 at 1.5 m."""
    boxes = [(-12.0, 1.5), (18.0, -10.0)]
    world = layouts.world_from_dump(_dump(i, boxes), META)
    pose = np.stack([-25.0 + 0.5 * np.arange(n), np.zeros(n), np.zeros(n)], 1)
    scan = targets.scan_geometry(META)
    local = scan["angle_start_rad"] + np.arange(19) * scan["angle_increment_rad"]
    collect_t = int(np.argmin(np.abs(pose[:, 0] - boxes[0][0])))
    ranges = np.zeros((n, 19), np.float32)
    sem = np.zeros((n, 19, 2), np.float32)
    pickups = (np.arange(n) >= collect_t).astype(np.int16)
    for t in range(n):
        alive = [t <= collect_t, True]   # Destroy is deferred: visible at its pickup step
        for k in range(19):
            r, box = _march(world, pose[t, 0], pose[t, 1], pose[t, 2] + local[k], boxes, alive)
            ranges[t, k] = r
            sem[t, k, 0] = float(box)
        if t in embedded_at:
            ranges[t] = 0.5
            sem[t] = 0
    depth = np.where(ranges > 0, ranges / 20.0, 1.0)
    ep = {"lidar": np.concatenate([depth, sem.reshape(n, -1)], 1).astype(np.float32),
          "gps": (pose[:, :2] / 300).astype(np.float32),
          "compass": np.zeros((n, 1), np.float32),
          "pose": pose.astype(np.float32), "ranges": ranges, "pickups": pickups,
          "is_first": np.arange(n) == 0, "action_0": np.zeros(n, np.int32),
          "action_1": np.zeros(n, np.int32), "world_seed": np.int64(1000 + i),
          "source": np.str_("explorer" if i % 2 else "policy"),
          "policy_run": np.str_("" if i % 2 else f"L{i % 4 // 2}_test__dreamer__seed0")}
    return ep, world


def _run(*args):
    return subprocess.run([sys.executable, "-m", *args], cwd=ROOT, check=True,
                          capture_output=True, text=True)


def _refused(args, text):
    try:
        _run(*args)
    except subprocess.CalledProcessError as e:
        assert text in e.stderr + e.stdout, (e.stderr + e.stdout)[-800:]
    else:
        raise AssertionError(f"accepted, expected a refusal with {text!r}")


# -- layouts ----------------------------------------------------------------------

def test_dump_orientation_on_a_non_square_world():
    mask = np.zeros((60, 40), bool)          # [j, i]
    mask[10, 5] = True                       # Unity x = -14.5, z = -19.5
    dump = {"mask_w": 40, "mask_h": 60, "cell_size": 1.0, "origin_x": -20.0,
            "origin_z": -30.0, "mask": "".join("1" if v else "0" for v in mask.ravel()),
            "rewards": [{"x": -14.5, "z": -19.5, "size_x": 2.0, "size_z": 1.0}],
            "semantic_classes": CLASSES, "build_guid": "t"}
    w = layouts.world_from_dump(dump, META)
    assert w["wall"].shape == (40, 60)       # rows = ROS y (Unity X extent)
    # ROS x = Unity z = -19.5 -> col 10; ROS y = -Unity x = 14.5 -> row 34
    assert w["wall"][34, 10] == 1.0 and w["wall"].sum() == 1.0, np.argwhere(w["wall"])
    assert np.allclose(w["box_xy"], [[-19.5, 14.5]])
    assert np.allclose(w["box_half"], [[0.5, 1.0]])   # along ROS x = Unity z
    # the helper the other tests build worlds with agrees
    assert np.array_equal(layouts.world_from_dump(_dump(2, []), META)["wall"] > 0, _wall_ros(2))


def test_wall_fraction_of_an_offset_mask():
    # A 39-cell mask shifted by half a cell (memory_maze trims to odd sizes).
    mask = np.zeros((60, 39), bool)
    mask[:, 20] = True                       # Unity x in [0.5, 1.5)
    f = layouts.wall_fraction(mask, np.array([-19.5, -30.0]), 1.0, META)
    rows = 1 + np.flatnonzero(f[1:39, 30] > 0)   # ROS y = -Unity x in (-1.5, -0.5]
    assert rows.tolist() == [18, 19] and np.allclose(f[rows, 30], 0.5), f[:, 30]
    # The edge rows half-overlap the mask's border, half lie outside it (wall).
    assert f[0, 30] == 0.5 and f[39, 30] == 0.5, f[:, 30]


def test_verify_passes_and_its_controls_bite():
    ep, world = _episode(0)
    v = layouts.verify(ep, world, META, radius=2.0)
    assert v["wall_ok"] == 1.0 and v["reward_ok"] == 1.0, v
    assert v["free_ok"] == 1.0 and v["pose_ok"] == 1.0, v
    assert v["wall_ok_mirrored"] < v["wall_ok"] - 0.2, v
    assert v["n_reward_hits"] > 0 and v["boxes_seen"] == 2, v
    assert abs(v["embedded_frac"] - 1 / 61) < 1e-9, v
    # The same episode against a mirrored layout must fail.
    flipped = dict(world, wall=world["wall"][::-1].copy())
    bad = layouts.verify(ep, flipped, META, radius=2.0)
    assert bad["wall_ok"] < 0.9 or bad["free_ok"] < 0.9, bad
    assert layouts.check(v, {"min_wall_ok": .99, "min_reward_ok": .99, "min_free_ok": .99,
                             "min_pose_ok": .99, "min_mirror_gap": .2, "expect_boxes": 2}) == []


def test_box_timeline_pickups():
    _, world = _episode(0)
    world = dict(world, box_xy=np.array([[0, 1.5], [0, -1.5], [10, 0]], np.float32),
                 box_half=np.full((3, 2), BOX_HALF, np.float32))
    n = 10
    ep = {"pose": np.stack([np.linspace(-4, 14, n), np.zeros(n), np.zeros(n)], 1),
          "pickups": np.array([0, 0, 0, 2, 2, 2, 2, 2, 3, 3], np.int16)}
    no_rays = {"hit": np.zeros(0, bool), "reward": np.zeros(0, bool), "x0": np.zeros(0),
               "y0": np.zeros(0), "angle": np.zeros(0), "range": np.zeros(0), "t": np.zeros(0, int)}
    b = layouts.box_timeline(ep, world, no_rays, radius=2.0)
    assert b["collected_t"].tolist() == [3, 3, 8], b      # a double pickup at t=3
    assert b["unmatched_pickups"] == []
    ep["pickups"] = np.array([0, 0, 0, 3, 3, 3, 3, 3, 3, 3], np.int16)   # box 2 out of reach
    b = layouts.box_timeline(ep, world, no_rays, radius=2.0)
    assert b["unmatched_pickups"] == [3] and b["collected_t"][:2].tolist() == [3, 3], b


# -- global targets / scoring -----------------------------------------------------

def test_global_targets_masks_and_boxes():
    ep, world = _episode(1)
    tg = global_targets.episode_global_targets(ep, world, META, stride=1, pickup_radius=2.0)
    assert tg["glayout"].shape == (20, 30) and set(np.unique(tg["glayout"])) <= {0, .25, .5, .75, 1}
    for t, seen, inview, last in targets.replay(ep, META):
        assert np.array_equal(tg["gage"][t] == 0, targets.coarse_any(inview)), t
    assert (np.diff(tg["gage"].astype(int), axis=0)[(tg["gage"][1:] > 0) & (tg["gage"][:-1] > 0)
            & (tg["gage"][1:] < global_targets.AGE_NEVER)] == 1).all()
    collected = tg["box_collected_t"][0]
    assert 0 < collected < 61 and tg["box_collected_t"][1] == global_targets.NEVER
    assert tg["box_first_hit_t"][0] <= collected
    known = tg["gknown_first"]
    assert (known[known >= 0] <= 60).all() and (known >= 0).any()


def test_indexed_targets_crop_uncrop_and_mean():
    import torch
    rng = np.random.default_rng(0)
    maps = rng.random((3, 20, 30)).astype(np.float32)
    index = np.array([0, 1, 2, 2])
    rc = np.array([[0, 0], [10, 15], [19, 29], [5, 3]])
    it = gs.IndexedTargets(maps, index, rc, window=8, fill=1.0)
    y = it[np.arange(4)].reshape(4, 8, 8)
    assert np.allclose(y[1], maps[1, 6:14, 11:19], atol=1e-3)
    assert (y[0][:4, :4] == 1.0).all() and np.allclose(y[0][4:, 4:], maps[0, :4, :4], atol=1e-3)
    on_cpu = it.to("cpu")[torch.arange(4)].numpy()
    assert np.array_equal(on_cpu, it[np.arange(4)])
    assert np.allclose(it.mean(np.arange(4), chunk=3), it[np.arange(4)].mean(0))
    full, inside = gs.uncrop(it[np.arange(4)], rc, 8, (20, 30))
    assert np.allclose(full[1].reshape(20, 30)[6:14, 11:19], maps[1, 6:14, 11:19], atol=1e-3)
    assert inside[0].sum() == 16 and np.isnan(full[0]).sum() == 600 - 16
    whole = gs.IndexedTargets(maps, index, rc, window=None, fill=1.0)
    assert np.allclose(whole[np.arange(4)], maps[index].reshape(4, -1), atol=1e-3)


def test_ap_is_exact_for_float16_scores():
    from sklearn.metrics import average_precision_score
    rng = np.random.default_rng(1)
    p = rng.random(5000).astype(np.float16)            # many ties after rounding
    y = rng.random(5000) < 0.1 + 0.5 * p.astype(float)
    h = gs.APHist()
    h.add(p[:2000], y[:2000])
    h.add(p[2000:], y[2000:])
    assert abs(h.ap() - average_precision_score(y, p.astype(np.float64))) < 1e-12


def _one_episode_targets(tmp: Path):
    ep, world = _episode(1)
    tg1 = global_targets.episode_global_targets(ep, world, META, stride=1, pickup_radius=2.0)
    dataset.save_npz(tmp / "ep0000.npz", tg1)
    tr = targets.new_tracker(META)
    rc = np.stack([np.floor((ep["pose"][:, 1] - tr.origin_y) / 2),
                   np.floor((ep["pose"][:, 0] - tr.origin_x) / 2)], 1).astype(int)
    tg = {"dir": tmp, "names": ["ep0000.npz"], "t": tg1["t"],
          "episode": np.zeros(len(tg1["t"]), int), "glayout": tg1["glayout"][None],
          "gknown_first": tg1["gknown_first"][None], "agent_rc": rc, "window": None,
          **{k: [tg1[k]] for k in ("box_cell", "box_collected_t", "box_first_hit_t")}}
    return ep, tg, tg1


def test_box_hit_rate_and_collected_false_alarm():
    with tempfile.TemporaryDirectory() as tmp:
        ep, tg, tg1 = _one_episode_targets(Path(tmp))
        n, cells = len(tg1["t"]), tg1["glayout"].size
        took = int(tg1["box_cell"][0])       # box 0: collected
        alive = gs.alive_at(tg1["box_collected_t"], tg1["t"])
        perfect = np.zeros((n, cells), np.float16)
        for i in range(n):                   # point at every box that is there
            perfect[i, tg1["box_cell"][alive[i]]] = 1.0
        acc = gs.GlobalScore("gboxes", 30, tg1["glayout"].shape)
        acc.add_episode(0, "explorer", perfect, tg)
        m = acc.metrics()
        assert m["n_memory_boxes"] > 0 and m["box_hit_rate"] == 1.0, m
        assert m["collected_fa_rate"] == 0.0, m
        stale = np.zeros((n, cells), np.float16)
        stale[:, took] = 1.0                 # keeps pointing at the box it took
        acc = gs.GlobalScore("gboxes", 30, tg1["glayout"].shape)
        acc.add_episode(0, "explorer", stale, tg)
        m = acc.metrics()
        assert m["collected_fa_rate"] == 1.0, m


def test_analyze_global_plots_many_cells():
    # Run 2 brought 8 model families; a palette of 4 crashed analyze (KeyError).
    from probing.s5_analysis.analyze_global import paired_bootstrap, plot_global
    per_ep = {"0": [10, 6, 10, 7], "1": [12, 8, 9, 5]}
    box_ep = {"0": {"boxes": 4, "hits": 1}, "1": {"boxes": 3, "hits": 2}}
    cells = ["L0_baseline", "L1_tbtt", "L1w128_tbtt", "L1w256_tbtt", "L2_read",
             "L2c_noaddr", "L6_probe_base", "L6_probe_tbtt", "L9_extra"]
    rows = []
    for cell in cells:
        for target in ("glayout", "gboxes"):
            test = {"bal_memory_by_age": [0.6] * 7, "box_hit_rate_by_age": [0.1] * 7,
                    "by_source": {"explorer": {"bal_memory": 0.55, "ap_memory": 0.01}},
                    "per_episode": per_ep if target == "glayout" else box_ep}
            rows.append({"model": f"{cell}__dreamer__seed0__stage5", "cell": cell, "seed": 0,
                         "stage": 5, "target": target, "_tests": [test], "bal_memory": 0.55,
                         "bal_inview": 0.7, "ap_memory": 0.01, "box_hit_rate": 0.09,
                         "box_hit_chance": 0.03})
    with tempfile.TemporaryDirectory() as d:
        plot_global(rows, Path(d))
        assert len(list(Path(d).glob("global_*.png"))) == 3
    assert len(paired_bootstrap(rows, n_boot=20)) == 2 * (len(cells) - 1)
    # A chosen subset, in order: the first is the reference, files get the tag.
    pick = ["L1_tbtt", "L7_missing", "L6_probe_tbtt", "L2_read"]
    with tempfile.TemporaryDirectory() as d:
        plot_global(rows, Path(d), cells=pick, tag="_sub")
        assert sorted(p.name for p in Path(d).glob("global_*_sub.png")) == [
            "global_by_source_sub.png", "global_forgetting_sub.png", "global_vs_stage_sub.png"]
    sub = paired_bootstrap(rows, n_boot=20, cells=pick)
    assert {r["minus"] for r in sub} == {"L1_tbtt"} and {r["cell"] for r in sub} == {"L6_probe_tbtt", "L2_read"}
    assert all("p_seeds_welch" in r for r in sub)


# -- end to end ---------------------------------------------------------------------

def _synthetic_layout_dataset(root: Path, n_eps: int = 8) -> str:
    for i in range(n_eps):
        ep, world = _episode(i)
        dataset.save_episode(dataset.episode_path(root, i), ep)
        world.update(world_seed=np.int64(1000 + i), reset_pose=ep["pose"][0],
                     reset_ranges=ep["ranges"][0])
        dataset.save_npz(root / "layouts" / f"ep{i:04d}.npz", world)
    dataset.save_json(root / "meta.json", META)
    return dataset.seal(root)


def test_pipeline_end_to_end_on_cpu():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        root = tmp / "dataset"
        fp = _synthetic_layout_dataset(root)
        _run("probing.s2_targets.layouts", "--dataset", str(root), "--verify_only", "--workers", "1",
             "--expect_boxes", "2")
        lay = dataset.load_json(root / "layouts" / "layouts.json")
        assert lay["dataset_fingerprint"] == fp
        _run("probing.s2_targets.global_targets", "--dataset", str(root), "--workers", "1")
        assert dataset.is_current(root / "gtargets" / "gtargets.json", root)

        probes = tmp / "probes"
        for feats in ("const", "obs"):
            for target in ("glayout", "gboxes"):
                out = probes / feats / f"{target}_w16"
                _run("probing.s4_probe.train_probe", "--dataset", str(root), "--features", feats,
                     "--target", target, "--window", "16", "--out", str(out),
                     "--epochs", "2", "--device", "cpu", "--save_pred_per_source", "1")
                m = json.loads((out / "metrics.json").read_text())
                head = "bal_inview" if target == "glayout" else "ap_all"
                assert np.isfinite(m["test"][head]), m["test"]
                assert m["test"]["n_memory_cells" if target == "glayout"
                                 else "n_collected_cells"] > 0, m["test"]
                assert m["window"] == 16 and m["gtargets_fingerprint"]
                assert dataset.is_current(out / "metrics.json", root)
                pr = np.load(out / "test_predictions.npz")
                assert pr["pred"].shape[1:] == (16, 16) and len(pr["agent_rc"]) == len(pr["t"])
        # The whole (non-square) map works too.
        whole = tmp / "whole"
        _run("probing.s4_probe.train_probe", "--dataset", str(root), "--features", "obs", "--target",
             "gboxes", "--window", "0", "--out", str(whole), "--epochs", "1", "--device", "cpu")
        assert np.load(whole / "test_predictions.npz")["pred"].shape[1:] == (20, 30)

        # Deterministic: the same seed again gives the same numbers and predictions.
        again = tmp / "again"
        _run("probing.s4_probe.train_probe", "--dataset", str(root), "--features", "obs", "--target",
             "glayout", "--window", "16", "--out", str(again), "--epochs", "2", "--device", "cpu",
             "--save_pred_per_source", "1")
        a = json.loads((probes / "obs" / "glayout_w16" / "metrics.json").read_text())
        b = json.loads((again / "metrics.json").read_text())
        assert a["test"] == b["test"] and a["val_history"] == b["val_history"]
        assert np.array_equal(np.load(probes / "obs" / "glayout_w16" / "test_predictions.npz")["pred"],
                              np.load(again / "test_predictions.npz")["pred"])

        _run("probing.s5_analysis.analyze", "--probes", str(probes), "--dataset", str(root))
        for name in ("summary.csv", "global_vs_stage.png", "global_forgetting.png",
                     "global_by_source.png", "global_paired.csv"):
            assert (probes / "analysis" / name).exists(), name
        _run("probing.tools.inspect_layouts", "--dataset", str(root), "--episodes", "0")
        _run("probing.tools.inspect_targets", "--dataset", str(root), "--episodes", "0", "--t", "30")
        _run("probing.tools.inspect_probes", "--probes", str(probes), "--dataset", str(root),
             "--target", "gboxes")
        for png in (root / "layouts" / "inspect" / "ep0000.png",
                    root / "gtargets" / "inspect" / "ep0000.png",
                    probes / "analysis" / "inspect" / "curves_gboxes.png"):
            assert png.stat().st_size > 10_000, png

        # A layout changed after verification: refused downstream.
        w = dataset.load_episode(root / "layouts" / "ep0001.npz")
        w["wall"] = w["wall"][::-1].copy()
        dataset.save_npz(root / "layouts" / "ep0001.npz", w)
        _refused(["probing.s2_targets.global_targets", "--dataset", str(root), "--workers", "1"],
                 "changed since they were verified")
        # Re-verified, it fails (the flip is caught) and leaves no layouts.json.
        _refused(["probing.s2_targets.layouts", "--dataset", str(root), "--verify_only", "--workers", "1"],
                 "episodes failed")
        assert not (root / "layouts" / "layouts.json").exists()
        # So the probes are no longer current, and train_probe refuses.
        assert not dataset.is_current(probes / "obs" / "glayout_w16" / "metrics.json", root)
        _refused(["probing.s4_probe.train_probe", "--dataset", str(root), "--features", "const",
                  "--target", "glayout", "--out", str(tmp / "bad"), "--device", "cpu"],
                 "layouts.json")


def test_diagnose_end_to_end_on_cpu():
    """probing.tools.diagnose plans the 33 runs, trains a small version of them on
    synthetic states, reproduces the probes/ probe of the same config exactly,
    summarises, plots, and skips everything on a rerun."""
    from probing.tools.diagnose import plan
    assert len(plan(Path("s"), ["L0_baseline", "L1_tbtt"], [0, 1, 2], 5)) == 33
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "x"
        root = base / "dataset"
        fp = _synthetic_layout_dataset(root)
        _run("probing.s2_targets.layouts", "--dataset", str(root), "--verify_only", "--workers", "1",
             "--expect_boxes", "2")
        _run("probing.s2_targets.global_targets", "--dataset", str(root), "--workers", "1")
        _run("probing.s2_targets.targets", "--dataset", str(root), "--stride", "10", "--workers", "1")
        grid = [dataset.load_episode(root / "gtargets" / f"ep{i:04d}.npz")["t"] for i in range(8)]
        assert all(np.array_equal(g, dataset.load_episode(root / "targets" / f"ep{i:04d}.npz")["t"])
                   for i, g in enumerate(grid))
        ep = np.concatenate([np.full(len(g), i, np.int32) for i, g in enumerate(grid)])
        t = np.concatenate(grid).astype(np.int32)
        rng = np.random.default_rng(0)
        (base / "states").mkdir()
        for name in ("L0_test__dreamer__seed0__stage5", "L0_test__dreamer__seed1__stage5",
                     "random_init"):
            np.savez(base / "states" / f"{name}.npz", deter=rng.random((len(t), 8)).astype(np.float16),
                     stoch=rng.integers(0, 4, (len(t), 3)).astype(np.uint8), episode=ep, t=t,
                     classes=np.int32(4), dataset_fingerprint=np.str_(fp))
        model = "L0_test__dreamer__seed0__stage5"
        ref = base / "probes" / model / "glayout_w16"
        _run("probing.s4_probe.train_probe", "--dataset", str(root), "--features", "state", "--states",
             str(base / "states" / f"{model}.npz"), "--target", "glayout", "--window", "16",
             "--out", str(ref), "--epochs", "1", "--device", "cpu")
        args = ["probing.tools.diagnose", "--root", tmp, "--name", "x", "--cells", "L0_test",
                "--seeds", "0", "1", "--window", "16", "--epochs", "1", "--parallel", "3"]
        _run(*args)
        out = base / "diagnostics"
        with open(out / "summary.csv") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 15, len(rows)
        cap = next(r for r in rows if r["name"] == f"glayout__4x1024__{model}")
        assert cap["ref_test_bal_memory"] and cap["ref_test_bal_memory"] == cap["test_bal_memory"]
        assert cap["train_bal_memory"] and cap["train_bal_inview"]
        lin = next(r for r in rows if r["name"] == f"glayout__0x1024__{model}")
        assert lin["layers"] == "0" and not lin.get("ref_test_bal_memory")
        loc = [r for r in rows if r["group"] == "localization"]
        assert len(loc) == 7 and all(float(r["test_pos_err_mean_m"]) >= 0 for r in loc)
        for png in ("localization.png", "underfitting.png"):
            assert (out / png).stat().st_size > 20_000, png
        rerun = _run(*args)
        assert "15 runs, 0 to train" in rerun.stdout, rerun.stdout


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=6)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
