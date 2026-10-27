"""Probe diagnostics on the existing states of one probing run (main venv).

Two questions, one driver:

    localization   does the hidden state know where the agent is? Position
                   (relative to the episode start, what gps encodes) and
                   heading decoded from the state ALONE: `latent` (deter +
                   stoch) and `deter`, no gps/compass input. Controls: the
                   current lidar alone, the training mean (const), an
                   untrained network.
    underfitting   is the 4x1024 probe too small or too short of data, or is
                   the information not in the state? Train vs test scores,
                   probe capacity (linear .. 8x1024), training-set size.

    python -m probing.tools.diagnose --name full --parallel 3          # cluster: diag_job.qsub
    python -m probing.tools.diagnose --name full --dry_run             # list the runs only

Reads results/probing/<name>/{dataset,states,probes}; nothing is encoded.
Every run is one probing.s4_probe.train_probe process; a run whose metrics.json is
current is skipped, so a resubmit finishes the rest. Writes
results/probing/<name>/diagnostics/:

    runs/<run>/metrics.json   train_probe output (+ log.txt)
    summary.csv               one row per run; ref_test_* = the same probe in
                              probes/ (identical config), a reproducibility check
    localization.png          position / heading error vs time since start
    underfitting.png          train vs test, capacity curve, data curve
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from probing.core import dataset
from probing.s5_analysis.analyze_global import CELL_COLORS, CONTROL_STYLE, GREY, INK, _style

ARCHS = [(0, 1024), (4, 1024), (4, 2048), (8, 1024)]  # (layers, units); 0 = linear
FRACS = [0.25, 0.5]
KEYS = {"pose": ("pos_err_mean_m", "pos_err_median_m", "pos_err_p90_m",
                 "heading_err_mean_deg", "heading_err_median_deg", "r2_x", "r2_y"),
        "glayout": ("bal_memory", "bal_inview"),
        "gboxes": ("ap_memory", "box_hit_rate")}


def plan(states: Path, cells: list[str], seeds: list[int], stage: int) -> list[dict]:
    """The runs, in the order they are executed."""
    model = lambda cell, s: f"{cell}__dreamer__seed{s}__stage{stage}"
    runs = []

    def add(group, name, target, features, mdl="", **kw):
        runs.append({"group": group, "name": name, "target": target, "features": features,
                     "model": mdl, "states": str(states / f"{mdl}.npz") if mdl else "",
                     "layers": 4, "units": 1024, "train_frac": 1.0, "score_train": False, **kw})

    for feats in ("latent", "deter"):
        for cell in cells:
            for s in seeds:
                add("localization", f"pose__{feats}__{model(cell, s)}", "pose", feats, model(cell, s))
    add("localization", "pose__lidar", "pose", "lidar")
    add("localization", "pose__const", "pose", "const")
    add("localization", "pose__latent__random_init", "pose", "latent", "random_init")
    for cell in cells:
        for layers, units in ARCHS:
            add("capacity", f"glayout__{layers}x{units}__{model(cell, seeds[0])}", "glayout",
                "state", model(cell, seeds[0]), layers=layers, units=units, score_train=True)
    for cell in cells:
        for frac in FRACS:
            add("data", f"glayout__frac{frac}__{model(cell, seeds[0])}", "glayout", "state",
                model(cell, seeds[0]), train_frac=frac, score_train=True)
    for cell in cells:
        for s in seeds[1:]:
            add("train_vs_test", f"glayout__4x1024__{model(cell, s)}", "glayout", "state",
                model(cell, s), score_train=True)
    for cell in cells:
        add("train_vs_test", f"gboxes__4x1024__{model(cell, seeds[0])}", "gboxes", "state",
            model(cell, seeds[0]), score_train=True)
    return runs


def command(run: dict, ds: Path, out: Path, window: int, epochs: int | None) -> list[str]:
    cmd = [sys.executable, "-u", "-m", "probing.s4_probe.train_probe", "--dataset", str(ds),
           "--features", run["features"], "--target", run["target"], "--out", str(out),
           "--layers", str(run["layers"]), "--units", str(run["units"]),
           "--train_frac", str(run["train_frac"])]
    if run["states"]:
        cmd += ["--states", run["states"]]
    if run["target"] != "pose":
        cmd += ["--window", str(window)]
    if run["score_train"]:
        cmd.append("--score_train")
    if epochs:
        cmd += ["--epochs", str(epochs)]
    return cmd


def execute(runs: list[dict], ds: Path, rundir: Path, window: int, parallel: int,
            epochs: int | None) -> list[str]:
    """Train the runs that are not current; return the names that failed."""
    todo = []
    for run in runs:
        out = rundir / run["name"]
        if dataset.is_current(out / "metrics.json", ds):
            continue
        if run["states"] and not dataset.is_current(run["states"], ds):
            print(f"[diagnose] {run['name']}: states missing or stale: {run['states']}")
            todo.append((run, None))
            continue
        todo.append((run, out))
    print(f"[diagnose] {len(runs)} runs, {len(todo)} to train, {parallel} at a time", flush=True)

    def one(item):
        run, out = item
        if out is None:
            return run["name"]
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "log.txt", "w") as log:
            code = subprocess.run(command(run, ds, out, window, epochs), stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        print(f"[diagnose] {'done' if code == 0 else 'FAILED'} {run['name']}", flush=True)
        return None if code == 0 else run["name"]

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        return [name for name in pool.map(one, todo) if name]


def summarise(runs: list[dict], rundir: Path, probes: Path, window: int) -> list[dict]:
    """One row per finished run: config, train / val / test metrics, and the
    test metrics of the identical probe in probes/ where there is one."""
    rows = []
    for run in runs:
        path = rundir / run["name"] / "metrics.json"
        if not path.exists():
            continue
        m = json.loads(path.read_text())
        row = {k: run[k] for k in ("group", "name", "target", "features", "model",
                                   "layers", "units", "train_frac")}
        row.update({"epochs": m.get("epochs"), "n_train": m["n_samples"]["train"],
                    "_m": m})
        for split in ("train_eval", "val", "test"):
            for key in KEYS[run["target"]]:
                row[f"{split.split('_')[0]}_{key}"] = m.get(split, {}).get(key)
        ref = probes / run["model"] / f"{run['target']}_w{window}" / "metrics.json"
        same = (run["target"] != "pose" and run["layers"] == 4 and run["units"] == 1024
                and run["train_frac"] == 1.0)
        if same and ref.exists():
            r = json.loads(ref.read_text())
            for key in KEYS[run["target"]]:
                row[f"ref_test_{key}"] = r["test"].get(key)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    cols = list(dict.fromkeys(k for r in rows for k in r if not k.startswith("_")))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _control_test(probes: Path, model: str, target: str, window: int, key: str):
    path = probes / model / f"{target}_w{window}" / "metrics.json"
    return json.loads(path.read_text())["test"].get(key) if path.exists() else None


def plot_localization(rows: list[dict], cells: list[str], out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    loc = [r for r in rows if r["group"] == "localization"]
    if not loc:
        return
    bins = loc[0]["_m"]["test"]["time_bins"]
    labels = [f"{bins[i] / 10:.0f}-{bins[i + 1] / 10:.0f} s" for i in range(len(bins) - 1)]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="white")
    panels = [("pos_err_mean_m", "position error (m, mean): |decoded - true| offset from the start"),
              ("heading_err_mean_deg", "heading error (deg, mean)")]
    for ax, (key, title) in zip(axes, panels):
        _style(ax, title)
        for i, cell in enumerate(cells):
            for feats, ls, mk in (("latent", "-", "o"), ("deter", "--", "s")):
                rs = [r for r in loc if r["features"] == feats and r["model"].startswith(cell + "__")]
                if not rs:
                    continue
                curves = np.array([[b.get(key, np.nan) for b in r["_m"]["test"]["by_time"]]
                                   for r in rs], float)
                mean = np.nanmean(curves, 0)
                color = CELL_COLORS[i] if i < len(CELL_COLORS) else GREY
                ax.plot(x, mean, ls=ls, marker=mk, color=color, lw=2,
                        label=f"{cell} {feats} ({len(rs)} seeds)")
                if len(rs) > 1:
                    sd = np.nanstd(curves, 0, ddof=1)
                    ax.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.15, lw=0)
        for name, style in (("pose__lidar", (INK, "--")), ("pose__const", CONTROL_STYLE["const"]),
                            ("pose__latent__random_init", CONTROL_STYLE["random_init"])):
            r = next((r for r in loc if r["name"] == name), None)
            if r:
                ax.plot(x, [b.get(key, np.nan) for b in r["_m"]["test"]["by_time"]], ls=style[1],
                        color=style[0], lw=1.4, label=name.removeprefix("pose__").replace("__", " "))
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_xlabel("time since the episode start (bin)")
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7, framealpha=0.9)
    fig.suptitle("Localization probe: pose decoded from the state alone (no gps / compass input), "
                 "test episodes", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_underfitting(rows: list[dict], cells: list[str], probes: Path, window: int,
                      out: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    color = {c: CELL_COLORS[i] if i < len(CELL_COLORS) else GREY for i, c in enumerate(cells)}
    full = lambda r: r["layers"] == 4 and r["units"] == 1024 and r["train_frac"] == 1.0
    fig, axes = plt.subplots(3, 3, figsize=(16, 13), facecolor="white")
    for row_axes, (target, key, what) in zip(axes, [
            ("glayout", "bal_memory", "walls, memory cells: balanced acc."),
            ("glayout", "bal_inview", "walls, cells in view: balanced acc."),
            ("gboxes", "box_hit_rate", "boxes: remembered box in top-k")]):
        refs = [(m, _control_test(probes, m, target, window, key)) for m in ("obs", "const")]

        def controls(ax):
            for m, v in refs:
                if v is not None:
                    ax.axhline(v, color=CONTROL_STYLE[m][0], ls=CONTROL_STYLE[m][1], lw=1.2,
                               label=f"{m} (test)")

        # train vs test, the 4x1024 probe of probes/, per checkpoint
        ax = row_axes[0]
        _style(ax, f"{what}\ntrain vs test, 4x1024 probe")
        rs = [r for r in rows if r["target"] == target and full(r) and r["train_" + key] is not None]
        rs.sort(key=lambda r: r["model"])
        for j, r in enumerate(rs):
            c = color.get(r["model"].split("__")[0], GREY)
            ax.plot([j, j], [r["test_" + key], r["train_" + key]], color=c, lw=1)
            ax.plot(j, r["train_" + key], "o", mfc="white", mec=c, ms=8,
                    label="train episodes" if j == 0 else None)
            ax.plot(j, r["test_" + key], "o", color=c, ms=8, label="test episodes" if j == 0 else None)
        ax.set_xticks(range(len(rs)))
        ax.set_xticklabels([r["model"].split("__dreamer__")[0] + "\n" +
                            r["model"].split("__")[-2] for r in rs], fontsize=7)
        controls(ax)
        ax.legend(fontsize=7, framealpha=0.9)
        if target == "gboxes":
            for a in row_axes[1:]:
                a.axis("off")
            continue

        # capacity and data curves: one checkpoint per cell (the first seed);
        # the full-data point of the data curve is the 4x1024 capacity run
        for ax, xs, xlabel, title in (
                (row_axes[1], ARCHS, "probe (layers x units)", "probe capacity"),
                (row_axes[2], FRACS + [1.0], "fraction of the training episodes",
                 "training-set size")):
            _style(ax, f"{what}\n{title}")
            for cell in cells:
                rs = [r for r in rows if r["target"] == target and r["model"].startswith(cell + "__")]
                find = lambda ok: next((r for r in rs if ok(r)), None)
                if xs is ARCHS:
                    pick = [find(lambda r, a=a: r["group"] == "capacity"
                                 and (r["layers"], r["units"]) == a) for a in xs]
                    labels = ["linear" if a[0] == 0 else f"{a[0]}x{a[1]}" for a in xs]
                else:
                    pick = [find(lambda r, f=f: r["group"] == "data" and r["train_frac"] == f)
                            for f in FRACS]
                    pick.append(find(lambda r: r["group"] == "capacity" and full(r)))
                    labels = [f"{f:g}" for f in xs]
                for split, ls, mfc in (("test", "-", color[cell]), ("train", "--", "white")):
                    ys = [np.nan if r is None or r[f"{split}_{key}"] is None else r[f"{split}_{key}"]
                          for r in pick]
                    ax.plot(range(len(xs)), ys, ls=ls, marker="o", color=color[cell], mfc=mfc,
                            lw=2, ms=7, label=f"{cell} {split}")
            ax.set_xticks(range(len(xs)))
            ax.set_xticklabels(labels)
            ax.set_xlabel(xlabel)
            controls(ax)
            ax.legend(fontsize=7, framealpha=0.9)
    fig.suptitle("Probe underfitting diagnostics: open = train episodes, filled = test episodes; "
                 "train >> test = not enough data, both low and flat in capacity = not in the state",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--name", default="full", help="results/probing/<name>")
    ap.add_argument("--root", default="results/probing")
    ap.add_argument("--cells", nargs="+", default=["L0_baseline", "L1_tbtt"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--stage", type=int, default=5)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--parallel", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=None, help="smoke tests only")
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    base = Path(args.root) / args.name
    ds, probes, outdir = base / "dataset", base / "probes", base / "diagnostics"
    runs = plan(base / "states", args.cells, args.seeds, args.stage)
    if args.dry_run:
        for r in runs:
            print(f"{r['group']:14s} {r['name']}")
        print(f"[diagnose] {len(runs)} runs")
        return
    dataset.verify(ds)
    failed = execute(runs, ds, outdir / "runs", args.window, args.parallel, args.epochs)
    rows = summarise(runs, outdir / "runs", probes, args.window)
    write_csv(outdir / "summary.csv", rows)
    plot_localization(rows, args.cells, outdir / "localization.png")
    plot_underfitting(rows, args.cells, probes, args.window, outdir / "underfitting.png")
    print(f"[diagnose] {len(rows)}/{len(runs)} runs summarised -> {outdir}")
    if failed:
        sys.exit(f"[diagnose] FAILED ({len(failed)}): {' '.join(failed)}")


if __name__ == "__main__":
    main()
