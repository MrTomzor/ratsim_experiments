"""Paper figures of final performance: one bar chart per world, a bar per method
(height = mean, error bar = +-1 std, individual samples overlaid; --no-points hides them).

    ~/ratvenv/venv/bin/python make_final_boxplots.py                         # total_score
    ~/ratvenv/venv/bin/python make_final_boxplots.py --metrics total_score,objects_found
    ~/ratvenv/venv/bin/python make_final_boxplots.py --grid                  # + all worlds in one 2x4 figure
    ~/ratvenv/venv/bin/python make_final_boxplots.py --grid --layout tall    # 4x2 instead

    # any def(s), paper row or not
    ~/ratvenv/venv/bin/python make_final_boxplots.py hardsar superdynamic
    ~/ratvenv/venv/bin/python make_final_boxplots.py bigdreamer_ladder:consec4 --local

With no positional argument the figures are the paper worlds; naming experiment
ids instead (`EXP[:VARIATION]`, `--local` for results/experiments) boxplots any
def the same way, with whatever methods its runs contain.

Each (world, method) cell is resolved by make_results_table.py's own
resolve_rl_cell / resolve_external_cell with the same paper/results_table.yaml
(rows, `per_method`, `na`, n_eval, eval_metaseed, pinned difficulty), but
held-out eval only -- unlike the table, never the training-episode fallback:

  ppo / dreamer     one point per eval episode (the first n_eval of each seed),
                    pooled over seeds -- the spread is across worlds (and seeds);
                    seeds with fewer than n_eval eval episodes are left out
  human / frontier  one point per episode (the first n_eval)

One seed with a complete eval is enough for a cell. The figure does not mark
cells that have fewer seeds than expected; the console note names the left-out
seeds per cell. Methods in a row's `na:` are left out; a method with no
data gets a "no data" mark in its slot so the method order stays the same in
every world; human is always drawn last, as in the table. Points are overlaid on the bars by default (mean +- std alone
hides how few samples there are); --no-points turns them off.

Output: <out>/final_boxplots/<metric>/<world label>.{png,pdf}, and with --grid
<out>/final_boxplots_<metric>[_<name>].{png,pdf}; --out defaults to
`figures_out:` in paper/results_table.yaml, else results/analysis/paper/.
Needs pandas + matplotlib + pyyaml (sb3 venv).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from analyze_experiment import BASELINE_COLORS  # noqa: E402
from make_results_table import (  # noqa: E402
    DEFAULT_CONFIG,
    cell_spec,
    load_config,
    resolve_external_cell,
    resolve_rl_cell,
)
from make_training_figure import (  # noqa: E402
    DEFAULT_OUT,
    METRIC_LABELS,
    add_exp_args,
    add_layout_args,
    figure_legend,
    grid_shape,
    method_colors,
    panel_rows,
)
from record_trajectories import EXTERNAL_METHODS  # noqa: E402


def resolve_row(row: dict, methods: list[str], metrics: list[str], cfg: dict) -> dict:
    """{method: cell} with the table's rules; `na` methods -> status na."""
    na = set(row.get("na") or [])
    cells = {}
    for m in methods:
        if m in na:
            cells[m] = {"status": "na", "samples": {}}
            continue
        spec = cell_spec(row, m)
        if not spec.get("exp"):
            cells[m] = {"status": "missing", "samples": {}, "note": "row has no `exp`"}
        elif m in EXTERNAL_METHODS:
            cells[m] = resolve_external_cell(spec, m, cfg, metrics)
        else:
            cells[m] = resolve_rl_cell(spec, m, cfg, metrics, eval_only=True)
    return cells


def all_colors(methods: list[str]) -> dict[str, tuple | str]:
    colors = method_colors(methods)
    for m in methods:
        if m in EXTERNAL_METHODS:
            colors[m] = BASELINE_COLORS.get(m, "gray")
    return colors


def draw_box(ax, row: dict, cells: dict, methods: list[str], metric: str, cfg: dict,
             colors: dict, args, show_ticklabels: bool = True) -> None:
    shown = [m for m in methods if cells[m]["status"] != "na"]
    rng = np.random.default_rng(0)
    for i, m in enumerate(shown):
        vals = np.asarray(cells[m].get("samples", {}).get(metric) or [], dtype=float)
        if len(vals) == 0:
            ax.text(i, 0.5, "no data", ha="center", va="center", rotation=90, color="0.55",
                    fontsize=args.fontsize - 1, transform=ax.get_xaxis_transform())
            continue
        c = colors[m]
        # bar from 0 to the mean, error bar = +-1 std over the samples
        ax.bar(i, vals.mean(), width=0.6, color=matplotlib.colors.to_rgba(c, 0.35),
               edgecolor=c, lw=1.0, zorder=2)
        ax.errorbar(i, vals.mean(), yerr=vals.std(), fmt="none", ecolor="black",
                    elinewidth=1.1, capsize=4, zorder=4)
        if args.points:
            jitter = rng.uniform(-0.15, 0.15, len(vals)) if len(vals) > 1 else np.zeros(1)
            ax.scatter(i + jitter, vals, s=12, color=c, edgecolor="white", lw=0.4, zorder=3)
    ax.set_xticks(range(len(shown)))
    ax.set_xticklabels([cfg["methods"][m] for m in shown] if show_ticklabels else [],
                       fontsize=args.fontsize - 1, rotation=args.rotate,
                       ha="right" if args.rotate else "center")
    ax.set_xlim(-0.6, len(shown) - 0.4)
    ax.set_title(row["label"], fontsize=args.fontsize + 1)
    ax.tick_params(axis="y", labelsize=args.fontsize - 1)
    ax.grid(True, axis="y", alpha=0.3)


def slug(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="paper/results_table.yaml")
    ap.add_argument("--metrics", default="total_score",
                    help="comma-separated: total_score, objects_found (figures for each)")
    ap.add_argument("--n", type=int, default=None, help="eval episodes per cell (n_eval)")
    ap.add_argument("--rows", default=None, help="subset of table rows (labels or exp ids)")
    ap.add_argument("--methods", default=None, help="subset of the methods drawn")
    add_exp_args(ap)
    ap.add_argument("--grid", action="store_true",
                    help="also write all worlds in one figure (see --layout / --cols)")
    ap.add_argument("--no-single", action="store_true", dest="no_single",
                    help="skip the per-world figures")
    add_layout_args(ap, panel_w=2.6, panel_h=2.4)
    ap.add_argument("--no-points", action="store_false", dest="points",
                    help="don't overlay the individual samples on the bars")
    ap.add_argument("--rotate", type=float, default=0, help="x tick label rotation (degrees)")
    ap.add_argument("--fontsize", type=float, default=9.0)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--name", default="", help="suffix for the grid figure's file name")
    ap.add_argument("--out", default=None,
                    help="output dir (default: yaml figures_out, else results/analysis/paper)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    if args.n is not None:
        cfg["n_eval"] = args.n
    rows, found = panel_rows(cfg, args)
    methods = found if found is not None else list(cfg["methods"])
    if args.methods:
        keep = [m.strip() for m in args.methods.split(",") if m.strip()]
        bad = [m for m in keep if m not in methods]
        if bad:
            sys.exit(f"ERROR: --methods {bad} not among {methods}")
        methods = [m for m in methods if m in keep]
    if "human" in methods:  # human is the reference, so it goes last (as in the table)
        methods = [m for m in methods if m != "human"] + ["human"]
    metrics =[m.strip() for m in args.metrics.split(",") if m.strip()]
    bad = [m for m in metrics if m not in METRIC_LABELS]
    if bad:
        sys.exit(f"ERROR: --metrics {bad}; known: {list(METRIC_LABELS)}")
    colors = all_colors(methods)

    print(f"[final_boxplots] n_eval={cfg['n_eval']} (held-out eval only)")
    resolved = []
    for row in rows:
        cells = resolve_row(row, methods, metrics, cfg)
        resolved.append(cells)
        print(f"  {row['label']:<24} " + "  ".join(
            f"{m}={c['status']}({len(c.get('samples', {}).get(metrics[0]) or [])})"
            for m, c in cells.items() if c["status"] != "na"))
        for m, c in cells.items():
            if c["status"] != "na" and c.get("note"):
                print(f"    {m}: {c['note']}")

    out = Path(args.out or cfg.get("figures_out") or DEFAULT_OUT).expanduser()
    for metric in metrics:
        ylabel = METRIC_LABELS[metric]
        if not args.no_single:
            single_dir = out / "final_boxplots" / metric
            single_dir.mkdir(parents=True, exist_ok=True)
            for row, cells in zip(rows, resolved):
                # wide enough that the method names under the bars don't collide
                n_shown = sum(c["status"] != "na" for c in cells.values())
                fig, ax = plt.subplots(figsize=(max(args.panel_w, 1.05 * n_shown + 0.9),
                                                args.panel_h))
                draw_box(ax, row, cells, methods, metric, cfg, colors, args)
                ax.set_ylabel(ylabel, fontsize=args.fontsize)
                fig.tight_layout()
                stem = single_dir / slug(row["label"])
                fig.savefig(stem.with_suffix(".png"), dpi=args.dpi)
                fig.savefig(stem.with_suffix(".pdf"))
                plt.close(fig)
            print(f"[final_boxplots] -> {single_dir}/  ({len(rows)} worlds, .png + .pdf)")

        if args.grid:
            nrows, ncols = grid_shape(len(rows), args)
            fig, axes = plt.subplots(nrows, ncols, squeeze=False,
                                     figsize=(args.panel_w * ncols, args.panel_h * nrows))
            for k, (row, cells) in enumerate(zip(rows, resolved)):
                ax = axes[k // ncols][k % ncols]
                draw_box(ax, row, cells, methods, metric, cfg, colors, args,
                         show_ticklabels=False)
                if k % ncols == 0:
                    ax.set_ylabel(ylabel, fontsize=args.fontsize)
            for k in range(len(rows), nrows * ncols):
                axes[k // ncols][k % ncols].set_visible(False)
            handles = [Patch(facecolor=matplotlib.colors.to_rgba(colors[m], 0.25),
                             edgecolor=colors[m]) for m in methods]
            bottom = figure_legend(fig, handles, [cfg["methods"][m] for m in methods],
                                   args.fontsize)
            fig.tight_layout(rect=(0, bottom, 1, 1))
            out.mkdir(parents=True, exist_ok=True)
            stem = out / (f"final_boxplots_{metric}" + (f"_{args.name}" if args.name else ""))
            fig.savefig(stem.with_suffix(".png"), dpi=args.dpi)
            fig.savefig(stem.with_suffix(".pdf"))
            plt.close(fig)
            print(f"[final_boxplots] -> {stem}.png\n[final_boxplots] -> {stem}.pdf")


if __name__ == "__main__":
    main()
