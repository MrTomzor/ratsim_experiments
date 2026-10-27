"""Train one probe and score it on held-out episodes.

    $PPO_PYTHON_PATH -m probing.s4_probe.train_probe --dataset D --targets D/targets \
        --features state --states S/L0_baseline__dreamer__seed0__stage5.npz \
        --target layout --out P/L0_baseline__dreamer__seed0__stage5/layout

Features (the probe's input):
    latent  concat(deter, one-hot stoch) WITHOUT gps / compass, and
    deter   deter alone: for the localization probe (--target pose), which
            asks whether the state itself keeps where the agent is
    lidar   the current lidar alone (no gps / compass): its control
    state   concat(deter, one-hot stoch) from encode.py, plus the agent's own
            current gps and compass (sin, cos). The paper also feeds the
            agent's position and orientation: the probe should not have to
            re-derive what the agent is told every step.
    obs     the current observation only (lidar + gps + compass): the
            no-memory control. Whatever it gets right needs no memory.
    const   no input; the training-set marginal (per map cell, or the mean
            distance). The floor every probe must clear.

Targets (from targets.py's 16 x 16 window, 2 m cells, world-aligned):
    layout    wall vs free, one sigmoid per cell, BCE on KNOWN cells only.
              What the maze looks like where the agent has looked: the
              analogue of the paper's Walls benchmark. Headline = accuracy on
              memory cells (known, outside the current fan).
    coverage  known vs unknown, one sigmoid per cell, BCE on all cells. Where
              the agent has looked: the memory the volumetric reward is about.
    start     [detour, euclidean] distance to the start, metres / 100, MSE.
              detour = geodesic - euclidean: the part of the way home that
              depends on the maze. In pilot 2 the geodesic itself was decoded
              as well from the current observation as from any state (7.0 vs
              7.3 m) because it is mostly the straight-line distance, which is
              |gps| * 300 and in every input. euclidean stays as the positive
              control. NaN geodesics are masked out of loss and metrics.

Ground-truth targets (global_targets.py: the TRUE maze from the verified
layouts, 2 m cells; the probe predicts a --window x --window crop centred on
the agent, world axes -- see global_score.py for why not the whole map):
    glayout   wall fraction per cell, soft BCE on ALL cells of the crop.
              Headline = balanced accuracy on memory cells (seen before, not
              in view).
    gboxes    1 where a box is still there, BCE on all cells. Headline =
              average precision on memory cells (positives: boxes observed).

Localization target (probe diagnostics, probing/tools/diagnose.py):
    pose      [dx, dy] / 100 (metres from the episode start = gps * 300) and
              [sin, cos] of the heading, MSE. Scored as position error (m) and
              heading error (deg), by time since start and by source.

Diagnostic options: --units / --layers (0 = a linear probe), --train_frac
(a seeded fraction of the training episodes), --score_train (also score a
seeded subset of training episodes, as many as test: metrics["train_eval"]).

layout and coverage were one 3-class head in the first pilot. Unknown is the
majority class there, so the probe learnt mostly where the explored region
ends and scored 0.29-0.36 on KNOWN cells -- below chance for wall vs free.

Probe = Tab. E.3 of the paper: MLP, 4 x 1024, LayerNorm (eps 1e-3), ELU,
Adam 3e-4 / eps 1e-5; early-stopped on validation loss. Splits are by
episode (dataset.split_episodes), each episode a different maze.

Writes <out>/metrics.json and <out>/test_predictions.npz (episode, t, pred;
for the ground-truth targets only --save_pred_per_source test episodes per
source, pred = float16 probabilities of the crop (n, window, window), and
agent_rc, the agent's cell = the crop's centre). Training is deterministic for a given
--seed on the same GPU / torch (torch.use_deterministic_algorithms).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from probing.core import dataset
from probing.s4_probe.global_score import GLOBAL_TARGETS

DIST_SCALE = 100.0
POSE_SCALE = 100.0
GRID_TARGETS = ("layout", "coverage")
STATE_FEATURES = ("state", "latent", "deter")
TIME_BINS = [0, 100, 500, 1500, 3000, 5001]
WALL = 2  # targets.LABEL_WALL; 0 unknown, 1 free


def pose_features(ep: dict, t: np.ndarray) -> np.ndarray:
    ang = ep["compass"][t, 0] * np.pi
    return np.concatenate([ep["gps"][t], np.sin(ang)[:, None],
                           np.cos(ang)[:, None]], 1).astype(np.float32)


def load_targets(targets_dir: Path, paths: list[Path], fp: str) -> dict:
    """Targets of exactly these episode files, from this dataset version."""
    info = targets_dir / "targets.json"
    if not info.exists():
        raise ValueError(f"{targets_dir}: incomplete (no targets.json)")
    made_from = dataset.load_json(info).get("dataset_fingerprint")
    if made_from != fp:
        raise ValueError(f"{targets_dir}: computed from dataset {str(made_from)[:12]}, "
                         f"dataset is now {fp[:12]}; recompute the targets")
    parts = [dataset.load_episode(targets_dir / p.name) for p in paths]
    out = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    out["episode"] = np.concatenate(
        [np.full(len(part["t"]), dataset.episode_id(p), np.int32)
         for p, part in zip(paths, parts)])
    return out


def build_features(kind: str, tg: dict, episodes: list[dict],
                   states_path: str | None, fp: str = "") -> np.ndarray | None:
    if kind == "const":
        return None
    per_episode = lambda f: np.concatenate([f(episodes[e], tg["t"][tg["episode"] == e])
                                            for e in range(len(episodes))])
    pose = per_episode(pose_features) if kind in ("state", "obs") else None
    if kind in ("obs", "lidar"):
        lidar = per_episode(lambda ep, t: ep["lidar"][t]).astype(np.float32)
        return lidar if kind == "lidar" else np.concatenate([lidar, pose], 1)
    st = dataset.load_episode(states_path)
    made_from = str(st.get("dataset_fingerprint", ""))
    if made_from != fp:
        raise ValueError(f"{states_path}: encoded from dataset {made_from[:12]}, "
                         f"dataset is now {fp[:12]}; re-encode")
    # encode.py and targets.py sample the same (episode, t) grid; check it
    # rather than trust it -- a stride mismatch would silently pair a state
    # with another step's target.
    if not (np.array_equal(st["episode"], tg["episode"])
            and np.array_equal(st["t"], tg["t"])):
        raise ValueError(f"{states_path}: (episode, t) grid differs from the "
                         f"targets'; encode and targets need the same --stride")
    parts = [st["deter"].astype(np.float32)]
    if kind != "deter":
        onehot = np.eye(int(st["classes"]), dtype=np.float32)[st["stoch"]]
        parts.append(onehot.reshape(len(onehot), -1))
    if kind == "state":
        parts.append(pose)
    return np.concatenate(parts, 1)


def build_targets(kind: str, tg: dict, episodes: list[dict] | None = None):
    if kind == "pose":  # where the agent is, relative to where it started
        out = []
        for e, ep in enumerate(episodes):
            p = ep["pose"][tg["t"][tg["episode"] == e]].astype(np.float64)
            d = (p[:, :2] - ep["pose"][0, :2]) / POSE_SCALE
            out.append(np.column_stack([d, np.sin(p[:, 2]), np.cos(p[:, 2])]))
        return np.concatenate(out).astype(np.float32)
    if kind == "layout":  # NaN = unknown cell, masked out of the loss
        label = tg["map_label"]
        return np.where(label == 0, np.nan, label == WALL).astype(np.float32)
    if kind == "coverage":
        return (tg["map_label"] != 0).astype(np.float32)
    detour = tg["geo_start"] - tg["euc_start"]
    return (np.stack([detour, tg["euc_start"]], 1)
            / DIST_SCALE).astype(np.float32)


def make_mlp(n_in: int, n_out: int, units: int = 1024, layers: int = 4):
    """layers=0: a linear probe."""
    import torch.nn as nn
    mods, d = [], n_in
    for _ in range(layers):
        mods += [nn.Linear(d, units), nn.LayerNorm(units, eps=1e-3), nn.ELU()]
        d = units
    mods.append(nn.Linear(d, n_out))
    return nn.Sequential(*mods)


def loss_fn(kind: str, out, y):
    import torch
    import torch.nn.functional as F
    mask = ~torch.isnan(y)
    if kind in GRID_TARGETS or kind in GLOBAL_TARGETS:
        bce = F.binary_cross_entropy_with_logits(
            out.view(y.shape), torch.nan_to_num(y), reduction="none")
        return (bce * mask).sum() / mask.sum().clamp(min=1)
    return ((torch.nan_to_num(out - y)) ** 2 * mask).sum() / mask.sum().clamp(min=1)


def predict(kind: str, model, x, batch=4096):
    import torch
    outs = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(x), batch):
            outs.append(model(x[i:i + batch]).float().cpu())
    out = torch.cat(outs)
    if kind in GLOBAL_TARGETS:
        return torch.sigmoid(out).numpy().astype(np.float16)
    out = out.numpy()
    if kind in GRID_TARGETS:
        return (out.reshape(len(out), 16, 16) > 0).astype(np.int8)
    if kind == "pose":
        return out
    return out * DIST_SCALE


def train(kind, x, y, idx, args):
    import torch
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    dev = torch.device(args.device)
    mean = x[idx["train"]].mean(0)
    std = x[idx["train"]].std(0) + 1e-6
    xt = torch.from_numpy((x - mean) / std).to(dev)
    if kind in GLOBAL_TARGETS:  # IndexedTargets: maps + a map index per sample
        yt, n_out = y.to(dev), y.n_out
    else:
        yt = torch.from_numpy(y).to(dev)
        n_out = 16 * 16 if kind in GRID_TARGETS else 4 if kind == "pose" else 2
    model = make_mlp(x.shape[1], n_out, args.units, args.layers).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
    rng = np.random.default_rng(args.seed)
    tr = torch.from_numpy(idx["train"]).to(dev)
    va = torch.from_numpy(idx["val"]).to(dev)
    assert len(va), "no validation episodes to early-stop on"

    best, best_state, bad, history = np.inf, None, 0, []
    for epoch in range(args.epochs):
        model.train()
        perm = tr[torch.from_numpy(rng.permutation(len(tr))).to(dev)]
        for i in range(0, len(perm), args.batch):
            b = perm[i:i + args.batch]
            loss = loss_fn(kind, model(xt[b]), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val = float(np.mean([loss_fn(kind, model(xt[va[i:i + 4096]]),
                                         yt[va[i:i + 4096]]).item()
                                 for i in range(0, len(va), 4096)]))
        history.append(val)
        if val < best - 1e-5:
            best, bad = val, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                break
    model.load_state_dict(best_state)
    return lambda rows: predict(kind, model, xt[torch.from_numpy(rows).to(dev)]), \
        {"epochs": len(history), "best_val_loss": best, "val_history": history}


def const_predictor(kind, y, train_rows):
    if kind in GLOBAL_TARGETS:  # per-cell training-set mean, as a probability
        mean = y.mean(train_rows).astype(np.float16)
        return lambda rows: np.broadcast_to(mean, (len(rows), len(mean))).copy()
    if kind in GRID_TARGETS:  # per-cell majority over the training split
        mode = (np.nan_to_num(np.nanmean(y[train_rows], 0), nan=0.0)
                > 0.5).astype(np.int8)
        return lambda rows: np.broadcast_to(mode, (len(rows), 16, 16)).copy()
    if kind == "pose":
        mean = y[train_rows].mean(0)
        return lambda rows: np.broadcast_to(mean, (len(rows), 4)).copy()
    mean = np.nanmean(y[train_rows], 0) * DIST_SCALE
    return lambda rows: np.broadcast_to(mean, (len(rows), 2)).copy()


def score(kind: str, pred: np.ndarray, tg: dict, rows: np.ndarray,
          half_step: int) -> dict:
    late = tg["t"][rows] >= half_step
    if kind in GRID_TARGETS:
        label = tg["map_label"][rows]
        inview = tg["map_inview"][rows]
        late3 = late[:, None, None]
        acc = lambda hit, m: float(hit[m].mean()) if m.any() else float("nan")
        known = label != 0
        if kind == "layout":
            # Accuracy, and balanced accuracy (mean of the per-class
            # accuracies): walls outnumber free cells, so plain accuracy
            # rewards answering "wall" everywhere.
            wall = label == WALL
            hit = pred == wall
            memory = known & ~inview
            bal = lambda m: 0.5 * (acc(hit, m & wall) + acc(hit, m & ~wall))
            return {"acc_memory": acc(hit, memory),
                    "bal_memory": bal(memory),
                    "bal_memory_2nd_half": bal(memory & late3),
                    "acc_memory_wall": acc(hit, memory & wall),
                    "acc_memory_free": acc(hit, memory & ~wall),
                    "bal_inview": bal(known & inview),
                    "bal_known": bal(known),
                    "n_memory_cells": int(memory.sum())}
        # coverage: outside the current fan, "have I looked there?" is memory.
        hit = pred == known
        out = ~inview
        bal = lambda m: 0.5 * (acc(hit, m & known) + acc(hit, m & ~known))
        return {"acc_outview": acc(hit, out), "bal_outview": bal(out),
                "bal_outview_2nd_half": bal(out & late3),
                "acc_all": acc(hit, np.ones_like(hit)),
                "n_outview_cells": int(out.sum())}
    geo = tg["geo_start"][rows]
    euc = tg["euc_start"][rows]
    detour = geo - euc
    ok = ~np.isnan(geo)
    mae = lambda e, m: float(np.abs(e[m]).mean()) if m.any() else float("nan")
    return {"mae_detour_m": mae(pred[:, 0] - detour, ok),
            "mae_detour_m_2nd_half": mae(pred[:, 0] - detour, ok & late),
            "mae_geo_m": mae(pred[:, 0] + pred[:, 1] - geo, ok),
            "mae_euc_m": mae(pred[:, 1] - euc, np.ones_like(ok)),
            "geo_nan_frac": float(1 - ok.mean())}


def score_pose(pred: np.ndarray, y: np.ndarray, t: np.ndarray, groups: np.ndarray) -> dict:
    """Position error (m) and heading error (deg) of pose predictions, overall,
    by time since the episode start and by source."""
    pos = np.hypot(*((pred[:, :2] - y[:, :2]) * POSE_SCALE).T)
    yaw = lambda v: np.arctan2(v[:, 2], v[:, 3])
    head = np.degrees(np.abs((yaw(pred) - yaw(y) + np.pi) % (2 * np.pi) - np.pi))
    r2 = lambda k: float(1 - ((pred[:, k] - y[:, k]) ** 2).sum() / max(((y[:, k] - y[:, k].mean()) ** 2).sum(), 1e-12))

    def summary(m):
        if not m.any():
            return {}
        return {"pos_err_mean_m": float(pos[m].mean()), "pos_err_median_m": float(np.median(pos[m])),
                "pos_err_p90_m": float(np.percentile(pos[m], 90)),
                "heading_err_mean_deg": float(head[m].mean()),
                "heading_err_median_deg": float(np.median(head[m])), "n": int(m.sum())}
    out = {**summary(np.ones(len(pos), bool)), "r2_x": r2(0), "r2_y": r2(1)}
    tbin = np.digitize(t, TIME_BINS) - 1
    out["time_bins"] = TIME_BINS
    out["by_time"] = [summary(tbin == b) for b in range(len(TIME_BINS) - 1)]
    out["by_source"] = {g: summary(groups == g) for g in sorted(set(groups))}
    return out


def score_global(kind, predictor, tg, episodes, split, half, per_source,
                 names=("val", "test")):
    """Stream the `names` splits' episodes through global_score; keep the
    fixed test episodes' predictions."""
    from probing.s4_probe import global_score as gs
    groups = {e: gs.source_group(ep) for e, ep in enumerate(episodes)}
    keep = set(gs.fixed_episodes(split["test"], groups, per_source))
    shape = tg["glayout"].shape[1:]
    crop = (tg["window"], tg["window"]) if tg["window"] else shape
    metrics, saved = {}, {"episode": [], "t": [], "agent_rc": [], "pred": []}
    for name in names:
        acc = gs.GlobalScore(kind, half, shape)
        for e in split[name]:
            e = int(e)
            rows = np.flatnonzero(tg["episode"] == e)
            p = predictor(rows)
            acc.add_episode(e, groups[e], p, tg)
            if name == "test" and e in keep:
                saved["episode"].append(np.full(len(rows), e, np.int32))
                saved["t"].append(tg["t"][rows])
                saved["agent_rc"].append(tg["agent_rc"][rows])
                saved["pred"].append(p.reshape(len(rows), *crop))
        metrics[name] = acc.metrics()
    return metrics, {k: np.concatenate(v) for k, v in saved.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--targets", default=None,
                    help="default: <dataset>/targets, or <dataset>/gtargets for glayout/gboxes")
    ap.add_argument("--features", choices=("state", "obs", "const", "latent", "deter", "lidar"),
                    required=True)
    ap.add_argument("--states", default=None, help="encode.py output (state)")
    ap.add_argument("--target", choices=(*GRID_TARGETS, "start", *GLOBAL_TARGETS, "pose"),
                    required=True)
    ap.add_argument("--units", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=4, help="0 = a linear probe")
    ap.add_argument("--train_frac", type=float, default=1.0,
                    help="train on this seeded fraction of the training episodes")
    ap.add_argument("--score_train", action="store_true",
                    help="also score a seeded subset of training episodes (as many as test)")
    ap.add_argument("--window", type=int, default=64,
                    help="glayout/gboxes: crop side in cells around the agent; 0 = whole map")
    ap.add_argument("--save_pred_per_source", type=int, default=2,
                    help="global targets: test episodes per source whose predictions are saved")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    # Before the first cuBLAS call: deterministic matmuls need a fixed workspace.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    if args.features in STATE_FEATURES and not args.states:
        ap.error(f"--features {args.features} needs --states")
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    root = Path(args.dataset)
    # verify() guarantees ids 0..n-1, so list position == episode id below.
    fp = dataset.verify(root)
    paths = dataset.list_episodes(root)
    episodes = [dataset.load_episode(p) for p in paths]
    if args.target in GLOBAL_TARGETS:
        from probing.s4_probe import global_score as gs
        tg = gs.load_global_targets(Path(args.targets or root / "gtargets"), root, paths,
                                    fp, episodes, args.window)
        y = gs.build_global_targets(args.target, tg)
    else:
        tg = load_targets(Path(args.targets or root / "targets"), paths, fp)
        y = build_targets(args.target, tg, episodes)
    x = build_features(args.features, tg, episodes, args.states, fp)

    split = dataset.split_episodes(len(episodes), seed=args.split_seed)
    if args.train_frac < 1.0:  # val and test stay the same episodes
        keep = np.random.default_rng(args.split_seed + 1).choice(
            split["train"], max(1, int(round(args.train_frac * len(split["train"])))), replace=False)
        split["train"] = np.sort(keep)
    if args.score_train:
        split["train_eval"] = np.sort(np.random.default_rng(args.split_seed + 2).choice(
            split["train"], min(len(split["test"]), len(split["train"])), replace=False))
    rows = {k: np.flatnonzero(np.isin(tg["episode"], v)) for k, v in split.items()}
    scored = ("val", "test", "train_eval") if args.score_train else ("val", "test")
    if args.features == "const":
        predictor, info = const_predictor(args.target, y, rows["train"]), {}
    else:
        predictor, info = train(args.target, x, y, rows, args)

    half = max(len(ep["pose"]) for ep in episodes) // 2
    out = Path(args.out)
    extra = {}
    if args.target in GLOBAL_TARGETS:
        metrics, saved = score_global(args.target, predictor, tg, episodes, split,
                                      half, args.save_pred_per_source, scored)
        dataset.save_npz(out / "test_predictions.npz", saved)
        import torch
        extra = {"window": args.window,
                 "gtargets_fingerprint": tg["info"]["fingerprint"],
                 "layouts_fingerprint": tg["info"]["layouts_fingerprint"],
                 "torch_version": torch.__version__, "deterministic": True,
                 "device_name": (torch.cuda.get_device_name()
                                 if args.device.startswith("cuda") else "cpu")}
    else:
        preds = {name: predictor(rows[name]) for name in scored}
        if args.target == "pose":
            from probing.s4_probe.global_score import source_group
            groups = np.array([source_group(ep) for ep in episodes])[tg["episode"]]
            metrics = {name: score_pose(preds[name], y[rows[name]], tg["t"][rows[name]],
                                        groups[rows[name]]) for name in preds}
        else:
            metrics = {name: score(args.target, preds[name], tg, rows[name], half)
                       for name in preds}
        dataset.save_npz(out / "test_predictions.npz",
                         {"episode": tg["episode"][rows["test"]],
                          "t": tg["t"][rows["test"]], "pred": preds["test"]})
    record = {"features": args.features, "target": args.target,
              "states": str(args.states or ""), "seed": args.seed,
              "units": args.units, "layers": args.layers, "train_frac": args.train_frac,
              "split": {k: v.tolist() for k, v in split.items()},
              "n_samples": {k: int(len(v)) for k, v in rows.items()},
              "n_inputs": 0 if x is None else int(x.shape[1]),
              "half_step": half, "dataset_fingerprint": fp, **extra, **info, **metrics,
              "seconds": round(time.time() - t0, 1)}
    # Last and atomic: probe_job.qsub treats an existing metrics.json as done.
    dataset.save_json(out / "metrics.json", record)
    head = {k: v for k, v in metrics["test"].items()
            if not isinstance(v, (dict, list))}
    print(f"[probe-train] {args.features}/{args.target} "
          f"{Path(args.states).stem if args.states else ''}: " + json.dumps(head))


if __name__ == "__main__":
    main()
