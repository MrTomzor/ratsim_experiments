"""Figures and the paired bootstrap of the ground-truth probes (glayout, gboxes)
for analyze.py. Input: analyze.collapse_probe_seeds rows, each carrying its
probe seeds' test metrics in `_tests` (global_score.GlobalScore.metrics)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from probing.s2_targets.targets import AGE_BINS

INK, PAPER, GREY = "#1A1A1A", "#F4F4F2", "#6B6B6B"
# The 8-slot categorical palette in its fixed order (a 9th cell would be grey,
# never a recycled hue).
CELL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
               "#e87ba4", "#008300", "#6250d6", "#e34948"]
CONTROL_STYLE = {"obs": (INK, "--"), "const": (GREY, ":"), "random_init": ("#9C9C9A", "-.")}
PANELS = [("glayout", "bal_memory", "walls: balanced acc., memory cells"),
          ("glayout", "bal_inview", "walls: balanced acc., cells in view"),
          ("gboxes", "ap_memory", "boxes: average precision, memory cells"),
          ("gboxes", "box_hit_rate", "boxes: remembered box in top-k (+-1 cell)")]


def _split(rows, cells=None):
    """Checkpoint rows, control rows, cell order: `cells` as given (only those
    present; the first is the reference), else all cells sorted."""
    ckpt = [r for r in rows if r["stage"] is not None]
    ctrl = [r for r in rows if r["stage"] is None]
    present = {r["cell"] for r in ckpt}
    if cells:
        cells = [c for c in cells if c in present]
        ckpt = [r for r in ckpt if r["cell"] in cells]
    else:
        cells = sorted(present)
    return ckpt, ctrl, cells


def _control_style(model: str):
    return CONTROL_STYLE.get(model.split("__")[0], (GREY, "-."))


def _style(ax, title):
    ax.set_facecolor(PAPER)
    ax.set_title(title, loc="left", fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _mean_list(tests: list, key: str) -> np.ndarray:
    return np.nanmean(np.array([t[key] for t in tests], float), 0)


def plot_global(rows: list[dict], outdir: Path, cells=None, tag: str = "") -> None:
    import warnings
    warnings.filterwarnings("ignore", "Mean of empty slice")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ckpt, ctrl, cells = _split(rows, cells)
    color = {cell: CELL_COLORS[i] if i < len(CELL_COLORS) else GREY
             for i, cell in enumerate(cells)}
    last = max((r["stage"] for r in ckpt), default=None)

    # 1. headline metrics vs stage
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), facecolor="white")
    for ax, (target, key, title) in zip(axes.ravel(), PANELS):
        _style(ax, title)
        for cell in cells:
            rs = [r for r in ckpt if r["cell"] == cell and r["target"] == target]
            stages = sorted({r["stage"] for r in rs})
            vals = [[r[key] for r in rs if r["stage"] == k and r[key] is not None] for k in stages]
            mean = np.array([np.mean(v) if v else np.nan for v in vals])
            sd = np.array([np.std(v, ddof=1) if len(v) > 1 else 0.0 for v in vals])
            n = max((len(v) for v in vals), default=0)
            ax.plot(stages, mean, "o-", color=color[cell], lw=2, ms=6, label=f"{cell} (n={n} seeds)")
            ax.fill_between(stages, mean - sd, mean + sd, color=color[cell], alpha=0.18, lw=0)
        for r in ctrl:
            if r["target"] == target and r[key] is not None:
                c, ls = _control_style(r["model"])
                ax.axhline(r[key], color=c, ls=ls, lw=1.3, label=f"{r['model']}")
        if target == "gboxes" and key == "box_hit_rate":
            chance = [r["box_hit_chance"] for r in rows
                      if r["target"] == target and r.get("box_hit_chance") is not None]
            if chance:
                ax.axhline(np.mean(chance), color=GREY, lw=0.8, label="random ranking")
        ax.set_xlabel("training stage")
        ax.legend(fontsize=7, framealpha=0.9)
    fig.suptitle("ground-truth probes vs training stage (test episodes)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(outdir / f"global_vs_stage{tag}.png", dpi=130)
    plt.close(fig)

    # 2. forgetting: memory metrics vs steps since last seen, last stage
    mid = np.sqrt(np.array(AGE_BINS[:-1], float) * np.array(AGE_BINS[1:]))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor="white")
    for ax, (target, key, title) in zip(axes, [
            ("glayout", "bal_memory_by_age", "walls: balanced acc. on memory cells"),
            ("gboxes", "box_hit_rate_by_age", "boxes: remembered box in top-k")]):
        _style(ax, f"{title}, stage {last}")
        for cell in cells:
            tests = [t for r in ckpt if r["cell"] == cell and r["target"] == target
                     and r["stage"] == last for t in r["_tests"]]
            if tests:
                ax.plot(mid, _mean_list(tests, key), "o-", color=color[cell], lw=2,
                        label=f"{cell} ({len(tests)} probes)")
        for r in ctrl:
            if r["target"] == target and r["_tests"]:
                c, ls = _control_style(r["model"])
                ax.plot(mid, _mean_list(r["_tests"], key), ls=ls, color=c, lw=1.3, label=r["model"])
        ax.set_xscale("log")
        ax.set_xlabel("steps since the cell was last in view (0.1 s each)")
        ax.legend(fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outdir / f"global_forgetting{tag}.png", dpi=130)
    plt.close(fig)

    # 3. headline per episode source, last stage
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), facecolor="white")
    for ax, (target, key, title) in zip(axes, [PANELS[0], PANELS[2]]):
        models = [(cell, [r for r in ckpt if r["cell"] == cell and r["stage"] == last
                          and r["target"] == target]) for cell in cells]
        models += [(r["model"], [r]) for r in ctrl if r["target"] == target]
        _style(ax, f"{title}, stage {last}, per source")
        sources = sorted({s for _, rs in models for r in rs if r["target"] == target
                          for t in r["_tests"] for s in t.get("by_source", {})})
        for j, (name, rs) in enumerate(models):
            vals = [[t["by_source"][s][key] for r in rs if r["target"] == target
                     for t in r["_tests"] if s in t.get("by_source", {})] for s in sources]
            xs = np.arange(len(sources)) + (j - len(models) / 2) * 0.12
            c = color.get(name) or _control_style(name)[0]
            ax.plot(xs, [np.nanmean(v) if v else np.nan for v in vals], "o", color=c, ms=8, label=name)
        ax.set_xticks(range(len(sources)))
        ax.set_xticklabels(sources)
        ax.legend(fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(outdir / f"global_by_source{tag}.png", dpi=130)
    plt.close(fig)


def _episode_metric(target: str, runs: list[dict], eps: np.ndarray) -> float:
    """Headline over the resampled episodes, averaged over the runs."""
    vals = []
    for per_ep in runs:
        if target == "glayout":
            c = np.sum([per_ep[e] for e in eps], 0)
            vals.append(0.5 * (c[1] / c[0] + c[3] / c[2]) if c[0] and c[2] else np.nan)
        else:
            b = np.sum([[per_ep[e]["boxes"], per_ep[e]["hits"]] for e in eps], 0)
            vals.append(b[1] / b[0] if b[0] else np.nan)
    return float(np.nanmean(vals))


def _welch_p(a, b) -> float:
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    from scipy import stats
    return round(float(stats.ttest_ind(a, b, equal_var=False).pvalue), 4)


def paired_bootstrap(rows: list[dict], n_boot: int = 2000, seed: int = 0,
                     cells=None) -> list[dict]:
    """Per target and stage: each cell minus the first cell on the headline
    (glayout bal_memory, gboxes box_hit_rate), resampling TEST EPISODES (the
    same resample for both cells) and averaging over seeds x probe seeds.
    The bootstrap holds the trained models fixed, so it leaves out seed-to-seed
    variation; p_seeds_welch is the Welch t-test over the training seeds (each
    seed's metric averaged over its probe seeds) -- read the two together."""
    ckpt, _, cells = _split(rows, cells)
    rng = np.random.default_rng(seed)
    out = []
    if len(cells) < 2:
        return out
    ref = cells[0]
    for target in ("glayout", "gboxes"):
        metric = "bal_memory" if target == "glayout" else "box_hit_rate"
        for stage in sorted({r["stage"] for r in ckpt if r["target"] == target}):
            runs = {c: [t["per_episode"] for r in ckpt if r["cell"] == c and r["stage"] == stage
                        and r["target"] == target for t in r["_tests"]] for c in cells}
            if not runs[ref]:
                continue
            eps = np.array(sorted(runs[ref][0]))
            boots = rng.integers(0, len(eps), (n_boot, len(eps)))
            for cell in cells[1:]:
                if not runs[cell]:
                    continue
                diff = _episode_metric(target, runs[cell], eps) - _episode_metric(target, runs[ref], eps)
                bd = [_episode_metric(target, runs[cell], eps[b]) - _episode_metric(target, runs[ref], eps[b])
                      for b in boots]
                lo, hi = np.nanpercentile(bd, [2.5, 97.5])
                per_seed = lambda c: [r[metric] for r in ckpt if r["cell"] == c and r["stage"] == stage
                                      and r["target"] == target and r.get(metric) is not None]
                out.append({"target": target, "metric": metric, "stage": stage, "cell": cell,
                            "minus": ref, "diff": round(diff, 5), "ci95_lo": round(lo, 5),
                            "ci95_hi": round(hi, 5),
                            "p_seeds_welch": _welch_p(per_seed(cell), per_seed(ref)),
                            "n_seeds": len(per_seed(cell)), "n_seeds_ref": len(per_seed(ref)),
                            "n_runs": len(runs[cell]), "n_runs_ref": len(runs[ref]),
                            "n_test_episodes": len(eps)})
    return out
