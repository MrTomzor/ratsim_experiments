"""Tests for probing/s2_targets/targets.py and probing/s4_probe/train_probe.py (main venv: scipy,
torch).

    ~/ratvenv/venv/bin/python tests/test_probe_targets.py

What they pin down: which way the map window is oriented, that "memory
cells" exclude everything the current scan shows, that the geodesic respects
walls, and that train_probe pairs every state with the target of its own step.
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ratsim.task_tracker.exploration_tracker import (FREE, OCCUPIED,  # noqa: E402
                                                     UNKNOWN)

from probing.core import dataset  # noqa: E402
from probing.s2_targets import targets  # noqa: E402
from probing.s4_probe.train_probe import score, score_pose  # noqa: E402

META = {"grid": {"world_width": 60.0, "world_height": 60.0, "resolution": 1.0},
        "lidar_geometry": {"angle_start_deg": -90.0,
                           "angle_increment_deg": 10.0, "max_range": 20.0}}


def _episode(poses, ranges=None):
    poses = np.asarray(poses, np.float32)
    n = len(poses)
    if ranges is None:
        ranges = np.full((n, 19), -1.0, np.float32)  # no returns: all free
    return {"pose": poses, "ranges": np.asarray(ranges, np.float32)}


def test_window_is_world_aligned():
    grid = np.full((60, 60), UNKNOWN, np.int8)
    grid[30, 35] = OCCUPIED                 # agent cell (30, 30) + 5 m in +x
    grid[38, 30] = OCCUPIED                 # + 8 m in +y
    lab = targets.coarse_label(targets.window(grid, 30, 30, UNKNOWN))
    assert lab.shape == (16, 16)
    assert lab[8, 10] == targets.LABEL_WALL      # x -> column
    assert lab[12, 8] == targets.LABEL_WALL      # y -> row
    assert (lab == targets.LABEL_WALL).sum() == 2


def test_window_pads_outside_the_grid():
    grid = np.full((60, 60), FREE, np.int8)
    lab = targets.coarse_label(targets.window(grid, 2, 2, UNKNOWN))
    assert (lab[:7, :] == targets.LABEL_UNKNOWN).all()
    assert (lab[8:, 8:] == targets.LABEL_FREE).all()


def test_reset_scan_counts_and_nothing_is_memory_at_step_zero():
    tg = targets.episode_targets(_episode([[0, 0, 0]]), META, stride=1)
    known = tg["map_label"][0] != targets.LABEL_UNKNOWN
    assert known.any(), "the reset observation's scan was dropped"
    assert not (known & ~tg["map_inview"][0]).any()


def test_turning_around_turns_the_view_into_memory():
    # Face +x, then rotate in place to face -x: what was ahead is behind.
    yaws = np.linspace(0.0, math.pi, 31)
    tg = targets.episode_targets(_episode([[0, 0, y] for y in yaws]), META,
                                 stride=30)
    label, inview, age = (tg[k][-1] for k in ("map_label", "map_inview", "map_age"))
    memory = (label != targets.LABEL_UNKNOWN) & ~inview
    assert memory[:, 12:].any(), "cells ahead at t=0 should be memory now"
    assert not memory[:, :6].any(), "cells in the current scan are not memory"
    assert (age[inview] == 0).all() and (age[memory] > 0).all()


def test_fan_and_rays_put_a_wall_on_the_same_side():
    # Unity ray 0 is at -90 deg, clockwise-positive, i.e. the agent's LEFT;
    # facing +x, left is +y. Rays 0 and 1 (90 and 80 deg left) hit at 5 m,
    # nothing elsewhere -- two rays, so the fan between them is unambiguous.
    ranges = np.full((1, 19), -1.0, np.float32)
    ranges[0, :2] = 5.0
    tg = targets.episode_targets(_episode([[0, 0, 0]], ranges), META, stride=1)
    lab, inview = tg["map_label"][0], tg["map_inview"][0]
    # Agent at the corner of coarse cells 7/8; +5 m in y -> coarse row 10.
    assert lab[10, 8] == targets.LABEL_WALL, np.argwhere(lab == 2)
    assert not inview[13, 8], "behind the wall must be outside the fan"
    assert inview[3, 8], "the free side (-y) is inside the fan"


def test_geodesic_respects_walls():
    grid = np.full((40, 40), FREE, np.int8)
    straight = targets.geodesic(grid == FREE, (20, 5), (20, 35))
    assert abs(straight - 30.0) < 1e-6
    grid[:, 20] = OCCUPIED
    grid[2, 20] = FREE                          # the only gap
    around = targets.geodesic(grid == FREE, (20, 5), (20, 35))
    assert around > 40.0, around
    grid[2, 20] = OCCUPIED
    assert math.isnan(targets.geodesic(grid == FREE, (20, 5), (20, 35)))
    assert math.isnan(targets.geodesic(grid == FREE, (20, 20), (20, 35)))
    # Off the grid is NaN, not numpy's wrap-around to the opposite edge.
    grid[:, 20] = FREE
    assert math.isnan(targets.geodesic(grid == FREE, (-1, 5), (20, 35)))
    assert math.isnan(targets.geodesic(grid == FREE, (20, 5), (20, 40)))


def test_geodesic_never_loses_the_agents_own_path():
    # Every ray returns at 5 cm: the tracker marks the agent's own cell
    # OCCUPIED at every step (endpoint cell == agent cell), so on the seen
    # map alone there is no path at all. The cells driven through are.
    poses = [[x, 0.0, 0.0] for x in np.arange(0.0, 10.0, 0.2)]
    ranges = np.full((len(poses), 19), 0.05, np.float32)
    tg = targets.episode_targets(_episode(poses, ranges), META, stride=49)
    assert abs(tg["geo_start"][-1] - 9.0) < 1e-6, tg["geo_start"]


def _grid_case():
    """One sample: cols 0-7 free, 8-15 unknown; cols 0-3 in view, so cols
    4-7 are the memory cells; rows 0-3 are walls in both parts, so each part
    has both classes (balanced accuracy is undefined with one)."""
    label = np.zeros((1, 16, 16), np.int8)
    label[0, :, :8] = targets.LABEL_FREE
    label[0, :4, :8] = targets.LABEL_WALL
    inview = np.zeros((1, 16, 16), bool)
    inview[0, :, :4] = True
    return {"map_label": label, "map_inview": inview, "t": np.array([0])}


def test_layout_scores_only_known_cells_and_balances_classes():
    tg = _grid_case()
    wall = tg["map_label"] == targets.LABEL_WALL
    pred = np.ones((1, 16, 16), np.int8)          # "wall" everywhere
    m = score("layout", pred, tg, np.array([0]), half_step=10)
    assert m["n_memory_cells"] == 16 * 4          # unknown cells never count
    assert m["acc_memory_wall"] == 1.0 and m["acc_memory_free"] == 0.0
    assert m["bal_memory"] == 0.5                 # all-wall is chance, not 25%
    m = score("layout", wall.astype(np.int8), tg, np.array([0]), half_step=10)
    assert m["bal_memory"] == 1.0 and m["bal_inview"] == 1.0


def test_coverage_scores_known_vs_unknown_outside_the_view():
    tg = _grid_case()
    known = (tg["map_label"] != 0).astype(np.int8)
    m = score("coverage", known, tg, np.array([0]), half_step=10)
    assert m["bal_outview"] == 1.0 and m["n_outview_cells"] == 16 * 12
    m = score("coverage", np.zeros_like(known), tg, np.array([0]), half_step=10)
    assert m["bal_outview"] == 0.5


def test_start_scores_the_detour_not_the_straight_line():
    from probing.s4_probe.train_probe import build_targets
    tg = {"geo_start": np.array([30.0, 12.0, np.nan], np.float32),
          "euc_start": np.array([10.0, 12.0, 5.0], np.float32),
          "t": np.array([0, 10, 20])}
    y = build_targets("start", tg) * 100
    assert np.allclose(y[:2], [[20, 10], [0, 12]]) and np.isnan(y[2, 0])
    rows = np.arange(3)
    m = score("start", np.nan_to_num(y), tg, rows, half_step=100)
    assert m["mae_detour_m"] == 0 and m["mae_geo_m"] == 0 and m["mae_euc_m"] == 0
    straight = np.stack([np.zeros(3), tg["euc_start"]], 1)  # "no detour" guess
    assert score("start", straight, tg, rows, half_step=100)["mae_detour_m"] == 10


def test_layout_loss_ignores_unknown_cells():
    import torch
    from probing.s4_probe.train_probe import build_targets, loss_fn
    tg = _grid_case()
    y = torch.from_numpy(build_targets("layout", tg))
    assert torch.isnan(y[0, :, 8:]).all() and not torch.isnan(y[0, :, :8]).any()
    logits = torch.zeros(1, 256)
    base = loss_fn("layout", logits, y)
    logits.view(1, 16, 16)[0, :, 8:] = 50.0       # wild guesses on unknown cells
    assert torch.isclose(loss_fn("layout", logits, y), base)


def _synthetic_dataset(root: Path, n_eps=6, n=41, sources=False):
    rng = np.random.default_rng(0)
    for i in range(n_eps):
        ang = np.cumsum(rng.uniform(-0.1, 0.1, n))
        pos = np.cumsum(np.stack([np.cos(ang), np.sin(ang)], 1) * 0.5, 0)
        extra = {"source": np.str_("explorer" if i % 2 else "policy"),
                 "policy_run": np.str_("" if i % 2 else "L0_test__dreamer__seed0")} if sources else {}
        dataset.save_episode(dataset.episode_path(root, i), {**extra,
            "lidar": rng.random((n, 57)).astype(np.float32),
            "gps": (pos / 300).astype(np.float32),
            "compass": (ang / np.pi)[:, None].astype(np.float32),
            "pose": np.concatenate([pos, ang[:, None]], 1).astype(np.float32),
            "ranges": rng.uniform(2, 25, (n, 19)).astype(np.float32),
            "is_first": np.arange(n) == 0,
            "action_0": np.zeros(n, np.int32), "action_1": np.zeros(n, np.int32)})
    dataset.save_json(root / "meta.json", META)
    return dataset.seal(root)


def _run(*args):
    subprocess.run([sys.executable, "-m", *args], cwd=ROOT, check=True,
                   capture_output=True, text=True)


def _refused(args, text):
    try:
        _run(*args)
    except subprocess.CalledProcessError as e:
        assert text in e.stderr, e.stderr[-800:]
    else:
        raise AssertionError(f"accepted, expected a refusal with {text!r}")


def test_pipeline_end_to_end_on_cpu():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "dataset"
        fp = _synthetic_dataset(root)
        _run("probing.s2_targets.targets", "--dataset", str(root), "--stride", "10",
             "--workers", "1")
        tg = dataset.load_episode(root / "targets" / "ep0000.npz")
        assert tg["t"].tolist() == [0, 10, 20, 30, 40]

        # A states file on the targets' (episode, t) grid, as encode.py writes.
        ep = np.repeat(np.arange(6, dtype=np.int32), 5)
        t = np.tile(np.arange(0, 41, 10, dtype=np.int32), 6)
        states = Path(tmp) / "s.npz"
        np.savez(states, deter=np.random.rand(30, 8).astype(np.float16),
                 stoch=np.random.randint(0, 4, (30, 3)).astype(np.uint8),
                 episode=ep, t=t, classes=np.int32(4),
                 dataset_fingerprint=np.str_(fp))
        for feats, extra in (("const", []), ("obs", []),
                             ("state", ["--states", str(states)])):
            for target in ("layout", "coverage", "start"):
                out = Path(tmp) / feats / target
                _run("probing.s4_probe.train_probe", "--dataset", str(root),
                     "--features", feats, "--target", target, "--out", str(out),
                     "--epochs", "2", "--device", "cpu", *extra)
                m = json.loads((out / "metrics.json").read_text())
                assert m["n_samples"]["test"] == 5, m["n_samples"]
                assert m["dataset_fingerprint"] == fp
                key = {"layout": "bal_known", "coverage": "acc_all",
                       "start": "mae_euc_m"}[target]
                assert np.isfinite(m["test"][key]), (feats, target, m["test"])

        _run("probing.s5_analysis.analyze", "--probes", tmp, "--dataset", str(root))
        for name in ("probes.csv", "summary.csv", "forgetting.png",
                     "over_episode.png"):
            assert (Path(tmp) / "analysis" / name).exists(), name

        probe = ["probing.s4_probe.train_probe", "--dataset", str(root), "--features",
                 "state", "--states", str(states), "--target", "layout",
                 "--out", str(Path(tmp) / "bad"), "--epochs", "1", "--device", "cpu"]

        # A states file on another grid must be refused, not silently paired.
        np.savez(states, deter=np.zeros((30, 8), np.float16),
                 stoch=np.zeros((30, 3), np.uint8), episode=ep, t=t + 1,
                 classes=np.int32(4), dataset_fingerprint=np.str_(fp))
        _refused(probe, "grid differs")

        # States encoded from another version of the episodes: refused.
        np.savez(states, deter=np.zeros((30, 8), np.float16),
                 stoch=np.zeros((30, 3), np.uint8), episode=ep, t=t,
                 classes=np.int32(4), dataset_fingerprint=np.str_("0" * 64))
        _refused(probe, "encoded from dataset")

        # An episode changed after sealing: every consumer refuses the dataset.
        ep0 = dataset.load_episode(dataset.episode_path(root, 0))
        ep0["pose"] = ep0["pose"] + 1.0
        dataset.save_npz(dataset.episode_path(root, 0), ep0)
        _refused(probe, "changed since")
        _refused(["probing.s2_targets.targets", "--dataset", str(root), "--workers", "1"],
                 "changed since")

        # Re-sealed: the old targets are now of another version, refused too.
        fp2 = dataset.seal(root)
        np.savez(states, deter=np.zeros((30, 8), np.float16),
                 stoch=np.zeros((30, 3), np.uint8), episode=ep, t=t,
                 classes=np.int32(4), dataset_fingerprint=np.str_(fp2))
        _refused(probe, "recompute the targets")

        # And analyze never mixes probes of two dataset versions in a table.
        rec = json.loads((Path(tmp) / "obs" / "layout" / "metrics.json").read_text())
        rec["dataset_fingerprint"] = fp2
        (Path(tmp) / "obs" / "layout" / "metrics.json").write_text(json.dumps(rec))
        _refused(["probing.s5_analysis.analyze", "--probes", tmp, "--dataset", str(root)],
                 "dataset versions")


def test_pose_score_is_exact_and_wraps_the_heading():
    deg = np.radians
    y = np.array([[0.0, 0.0, np.sin(deg(179)), np.cos(deg(179))],
                  [0.1, -0.2, np.sin(deg(10)), np.cos(deg(10))]])
    t, groups = np.array([5, 700]), np.array(["explorer", "L0"])
    m = score_pose(y.copy(), y, t, groups)
    assert m["pos_err_mean_m"] == 0 and m["heading_err_mean_deg"] < 1e-9
    assert m["r2_x"] == 1 and m["r2_y"] == 1
    # 3-4-5 m off, and -179 deg for 179 deg is 2 deg off, not 358.
    p = y.copy()
    p[0, :2] += (0.03, 0.04)
    p[0, 2:] = np.sin(deg(-179)), np.cos(deg(-179))
    m = score_pose(p, y, t, groups)
    assert math.isclose(m["by_source"]["explorer"]["pos_err_mean_m"], 5.0, abs_tol=1e-9)
    assert math.isclose(m["by_source"]["explorer"]["heading_err_mean_deg"], 2.0, abs_tol=1e-6)
    assert m["by_source"]["L0"]["pos_err_mean_m"] < 1e-9
    # Time bins: step 5 is in the first bin (0-100), step 700 in the third (500-1500).
    assert math.isclose(m["by_time"][0]["pos_err_mean_m"], 5.0, abs_tol=1e-9)
    assert m["by_time"][1] == {} and m["by_time"][2]["pos_err_mean_m"] < 1e-9


def test_localization_and_diagnostic_options_end_to_end():
    """latent / deter / lidar carry no gps or compass; --target pose is the
    offset from the episode start; --layers 0 is linear; --train_frac shrinks
    only the training episodes; --score_train adds train_eval."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "dataset"
        fp = _synthetic_dataset(root, sources=True)
        _run("probing.s2_targets.targets", "--dataset", str(root), "--stride", "10", "--workers", "1")
        ep = np.repeat(np.arange(6, dtype=np.int32), 5)
        t = np.tile(np.arange(0, 41, 10, dtype=np.int32), 6)
        states = Path(tmp) / "s.npz"
        np.savez(states, deter=np.random.rand(30, 8).astype(np.float16),
                 stoch=np.random.randint(0, 4, (30, 3)).astype(np.uint8),
                 episode=ep, t=t, classes=np.int32(4), dataset_fingerprint=np.str_(fp))
        base = ["probing.s4_probe.train_probe", "--dataset", str(root), "--target", "pose",
                "--epochs", "2", "--device", "cpu"]
        n_in = {"state": 8 + 12 + 4, "latent": 8 + 12, "deter": 8, "lidar": 57, "obs": 57 + 4,
                "const": 0}
        for feats, n in n_in.items():
            out = Path(tmp) / feats
            extra = ["--states", str(states)] if feats in ("state", "latent", "deter") else []
            _run(*base, "--features", feats, "--out", str(out), *extra)
            m = json.loads((out / "metrics.json").read_text())
            assert m["n_inputs"] == n, (feats, m["n_inputs"])
            assert np.isfinite(m["test"]["pos_err_mean_m"]) and len(m["test"]["by_time"]) == 5
            assert set(m["test"]["by_source"]) <= {"explorer", "L0_test"}
            assert dataset.is_current(out / "metrics.json", root)
        # The target is the true offset from the start: the const probe's
        # prediction is the training mean of exactly that.
        split = dataset.split_episodes(6, seed=0)
        eps = [dataset.load_episode(dataset.episode_path(root, e)) for e in split["train"]]
        mean = np.mean([(e["pose"][k, :2] - e["pose"][0, :2]) / 100 for e in eps
                        for k in range(0, 41, 10)], 0)
        pred = np.load(Path(tmp) / "const" / "test_predictions.npz")["pred"]
        assert np.allclose(pred[:, :2], mean, atol=1e-6), (pred[0], mean)

        out = Path(tmp) / "linear"
        _run(*base, "--features", "latent", "--states", str(states), "--out", str(out),
             "--layers", "0", "--train_frac", "0.5", "--score_train")
        m = json.loads((out / "metrics.json").read_text())
        full = json.loads((Path(tmp) / "latent" / "metrics.json").read_text())
        assert m["layers"] == 0 and m["train_frac"] == 0.5
        assert len(m["split"]["train"]) == len(full["split"]["train"]) // 2
        assert set(m["split"]["train"]) < set(full["split"]["train"])
        assert m["split"]["val"] == full["split"]["val"] and m["split"]["test"] == full["split"]["test"]
        assert set(m["split"]["train_eval"]) <= set(m["split"]["train"])
        assert len(m["split"]["train_eval"]) == len(m["split"]["test"])
        assert np.isfinite(m["train_eval"]["pos_err_mean_m"])
        # The default run is unchanged by the new options: no train_eval, 4x1024.
        assert "train_eval" not in full and full["layers"] == 4 and full["units"] == 1024


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_")]
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
