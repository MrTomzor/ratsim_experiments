"""Tables and curves from a directory of trained probes.

    $PPO_PYTHON_PATH -m probing.s5_analysis.analyze --probes results/probing/pilot/probes \
        --dataset results/probing/pilot/dataset

Expects <probes>/<model>/<target>/{metrics.json, test_predictions.npz}, where
<model> is `<cell>__dreamer__seed<i>__stage<k>` for checkpoints and anything
else (random_init, obs, const) for controls. Writes into <probes>/analysis/:

    probes.csv          one row per (model, target), test-split metrics
    summary.csv         checkpoints grouped by (cell, stage): mean, sd, n seeds
    forgetting.png      layout: balanced accuracy on memory cells vs steps
                        since the cell was last in view -- the forgetting curve
    over_episode.png    layout (memory cells), coverage (outside the view) and
                        detour error vs episode step

and, for the ground-truth targets (glayout, gboxes; global_score.py), with
probe seeds averaged first (<target>[__pseed<k>] dirs) and model seeds second:

    global_vs_stage.png    headline metrics vs training stage, L0/L1 cells
                           (mean +- sd over seeds) against the controls
    global_forgetting.png  last stage: memory metrics vs steps since last seen
    global_by_source.png   last stage: headline metrics per episode source
    global_paired.csv      per stage: cell - first cell, 95% paired bootstrap
                           over test episodes
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from probing.core import dataset
from probing.s2_targets.targets import AGE_BINS

CKPT = re.compile(r"^(?P<cell>.+)__dreamer__seed(?P<seed>\d+)__stage(?P<stage>\d+)$")
STEP_BINS = 10
# Headline columns per target; the first is what the tables lead with.
TABLE = {"layout": ("bal_memory", "bal_memory_2nd_half", "bal_inview"),
         "coverage": ("bal_outview", "bal_outview_2nd_half", "acc_all"),
         "start": ("mae_detour_m", "mae_detour_m_2nd_half", "mae_euc_m"),
         "glayout": ("bal_memory", "bal_memory_2nd_half", "bal_inview", "bal_never_seen"),
         "gboxes": ("ap_memory", "box_hit_rate", "box_hit_chance", "collected_fa_rate")}
OLD_TARGETS = ("layout", "coverage", "start")
COLS = tuple(dict.fromkeys(c for cols in TABLE.values() for c in cols))


def balanced(hit: np.ndarray, mask: np.ndarray, positive: np.ndarray) -> float:
    a = [hit[mask & cls].mean() for cls in (positive, ~positive)
         if (mask & cls).any()]
    return float(np.mean(a)) if len(a) == 2 else np.nan


def group_of(model: str) -> tuple[str, int | None, int | None]:
    m = CKPT.match(model)
    if not m:
        return model, None, None
    return m["cell"], int(m["seed"]), int(m["stage"])


def load_rows(probes: Path) -> list[dict]:
    rows, fps = [], set()
    for f in sorted(probes.glob("*/*/metrics.json")):
        rec = json.loads(f.read_text())
        fps.add(rec.get("dataset_fingerprint"))
        model = f.parent.parent.name
        cell, seed, stage = group_of(model)
        row = {"model": model, "cell": cell, "seed": seed, "stage": stage,
               "target": rec["target"], "pseed": rec.get("seed", 0),
               "window": rec.get("window", ""), "dir": f.parent,
               **{k: rec["test"].get(k) for k in COLS}}
        if rec["target"] not in OLD_TARGETS:
            row["_test"] = rec["test"]  # per-age / per-source / per-episode
        rows.append(row)
    if len(fps) > 1:  # never put numbers from two dataset versions in a table
        raise SystemExit(f"{probes}: probes from {len(fps)} dataset versions "
                         f"{sorted(str(f)[:12] for f in fps)}")
    return rows


def collapse_probe_seeds(rows: list[dict]) -> list[dict]:
    """One row per (model, target, window): the mean over probe seeds, with
    n_pseeds and the probe-seed sd of each column as <col>_psd."""
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["model"], r["target"], r["window"]), []).append(r)
    out = []
    for rs in groups.values():
        rs = sorted(rs, key=lambda r: r["pseed"])
        rec = {k: v for k, v in rs[0].items() if k not in COLS and k != "pseed"}
        rec["n_pseeds"] = len(rs)
        rec["_tests"] = [r["_test"] for r in rs if "_test" in r]
        for k in COLS:
            v = np.array([r[k] for r in rs if r[k] is not None], float)
            v = v[np.isfinite(v)]
            rec[k] = float(v.mean()) if len(v) else None
            if len(v) > 1:
                rec[k + "_psd"] = float(v.std(ddof=1))
        out.append(rec)
    return out


def summarise(rows: list[dict]) -> list[dict]:
    groups: dict = {}
    for r in rows:
        groups.setdefault((r["cell"], r["stage"], r["target"], r.get("window", "")),
                          []).append(r)
    out = []
    for (cell, stage, target, window), rs in sorted(groups.items(),
                                                    key=lambda kv: str(kv[0])):
        rec = {"cell": cell, "stage": stage, "target": target, "n": len(rs)}
        if window != "":
            rec["window"] = window
        for k in COLS:
            v = np.array([r[k] for r in rs if r[k] is not None], float)
            v = v[np.isfinite(v)]
            if len(v):
                rec[k] = round(float(v.mean()), 4)
                rec[k + "_sd"] = round(float(v.std(ddof=1)), 4) if len(v) > 1 else ""
        out.append(rec)
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for r in rows for k in r
                              if k != "dir" and not k.startswith("_")))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def curves(rows: list[dict], targets_dir: Path, half: int):
    """Per model, balanced accuracies by age bin and by episode-step bin
(layout: memory cells; coverage: cells outside the view), detour MAE by
episode-step bin."""
    cache: dict = {}

    def target_of(ep):
        if ep not in cache:
            cache[ep] = dataset.load_episode(targets_dir / f"ep{ep:04d}.npz")
        return cache[ep]

    out = {}
    edges = np.linspace(0, 2 * half, STEP_BINS + 1)
    for r in rows:
        pr = np.load(r["dir"] / "test_predictions.npz")
        lab, inv, age, geo, t = [], [], [], [], pr["t"]
        for ep, tt in zip(pr["episode"], pr["t"]):
            tg = target_of(int(ep))
            i = int(np.searchsorted(tg["t"], tt))
            lab.append(tg["map_label"][i]); inv.append(tg["map_inview"][i])
            age.append(tg["map_age"][i])
            geo.append(tg["geo_start"][i] - tg["euc_start"][i])  # detour
        sbin = np.clip(np.digitize(t, edges) - 1, 0, STEP_BINS - 1)
        res = {}
        if r["target"] in ("layout", "coverage"):
            lab, inv, age = np.stack(lab), np.stack(inv), np.stack(age)
            known = lab != 0
            if r["target"] == "layout":
                positive, mask = lab == 2, known & ~inv
            else:
                positive, mask = known, ~inv
            hit = pr["pred"] == positive
            abin = np.digitize(age, AGE_BINS) - 1
            res["by_age"] = [balanced(hit, mask & (abin == b), positive)
                             for b in range(len(AGE_BINS) - 1)]
            res["by_step"] = [balanced(hit, mask & (sbin == b)[:, None, None],
                                       positive) for b in range(STEP_BINS)]
        else:
            err = np.abs(pr["pred"][:, 0] - np.asarray(geo))
            res["by_step"] = [float(np.nanmean(err[sbin == b]))
                              if np.isfinite(err[sbin == b]).any() else np.nan
                              for b in range(STEP_BINS)]
        out[(r["model"], r["target"])] = res
    return out, edges


def plot(rows, cur, edges, outdir: Path) -> None:
    import warnings
    warnings.filterwarnings("ignore", "Mean of empty slice")  # empty bins -> gaps
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def label_of(r):
        return r["cell"] if r["stage"] is None else f"{r['cell']} s{r['stage']}"

    # Average curves over seeds of the same (cell, stage).
    by_label: dict = {}
    for r in rows:
        if (r["model"], r["target"]) in cur:
            by_label.setdefault((label_of(r), r["target"]), []).append(
                cur[(r["model"], r["target"])])
    mid_age = np.sqrt(np.array(AGE_BINS[:-1]) * np.array(AGE_BINS[1:]))
    mid_step = 0.5 * (edges[:-1] + edges[1:])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for (lab, target), cs in sorted(by_label.items()):
        if target == "layout":
            ax.plot(mid_age, np.nanmean([c["by_age"] for c in cs], 0), "o-",
                    label=f"{lab} (n={len(cs)})")
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("steps since the cell was last in view (0.1 s each)")
    ax.set_ylabel("wall vs free, balanced accuracy on memory cells")
    ax.set_title("Forgetting curve (test episodes)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "forgetting.png", dpi=150)
    plt.close(fig)

    panel = {"layout": 0, "coverage": 1, "start": 2}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
    for (lab, target), cs in sorted(by_label.items()):
        axes[panel[target]].plot(mid_step, np.nanmean([c["by_step"] for c in cs], 0),
                                 "o-", label=f"{lab} (n={len(cs)})")
    axes[0].set_ylabel("layout: balanced accuracy, memory cells")
    axes[1].set_ylabel("coverage: balanced accuracy, outside the view")
    axes[2].set_ylabel("detour to start (geodesic - straight), MAE [m]")
    for ax in axes:
        ax.set_xlabel("episode step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(outdir / "over_episode.png", dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--targets", default=None, help="default: <dataset>/targets")
    ap.add_argument("--cells", nargs="*", default=None,
                    help="ground-truth figures/CSV: only these cells, in this order; the first "
                         "is the reference of global_paired (default: all, sorted)")
    ap.add_argument("--tag", default="", help="suffix of the ground-truth figure/CSV names")
    args = ap.parse_args()

    probes = Path(args.probes)
    outdir = probes / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(probes)
    if not rows:
        raise SystemExit(f"no */*/metrics.json under {probes}")
    summary = summarise(collapse_probe_seeds(rows))
    write_csv(outdir / "probes.csv", rows)
    write_csv(outdir / "summary.csv", summary)

    for target, cols in TABLE.items():
        print(f"\n== {target} ==")
        print(f"{'cell':28s} {'stage':>5s} {'n':>2s} " +
              " ".join(f"{c:>24s}" for c in cols))
        for s in summary:
            if s["target"] != target:
                continue
            vals = " ".join(
                f"{s.get(c, float('nan')):>14.4f} ± {s.get(c + '_sd') or 0:<7.4f}"
                for c in cols)
            print(f"{s['cell']:28s} {str(s['stage'] if s['stage'] is not None else '-'):>5s} "
                  f"{s['n']:>2d} {vals}")

    old = [r for r in rows if r["target"] in OLD_TARGETS]
    if old:
        half = json.loads((old[0]["dir"] / "metrics.json").read_text())["half_step"]
        cur, edges = curves(old, Path(args.targets or Path(args.dataset) / "targets"),
                            half)
        plot(old, cur, edges, outdir)
    gt = [r for r in collapse_probe_seeds(rows) if r["target"] not in OLD_TARGETS]
    if gt:
        from probing.s5_analysis.analyze_global import paired_bootstrap, plot_global
        plot_global(gt, outdir, args.cells, args.tag)
        write_csv(outdir / f"global_paired{args.tag}.csv", paired_bootstrap(gt, cells=args.cells))
    print(f"\n[probe-analyze] wrote {outdir}")


if __name__ == "__main__":
    main()
