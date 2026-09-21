"""Paper figure of the training curves: one panel per paper world, 2x4 (wide) by default.

    ~/ratvenv/venv/bin/python make_training_figure.py                        # total_score
    ~/ratvenv/venv/bin/python make_training_figure.py --metrics objects_found
    ~/ratvenv/venv/bin/python make_training_figure.py --metrics total_score,objects_found
    ~/ratvenv/venv/bin/python make_training_figure.py --xmax def             # x-axis to the def's target
    ~/ratvenv/venv/bin/python make_training_figure.py --min-seeds 1 --per-seed
    ~/ratvenv/venv/bin/python make_training_figure.py --layout tall          # 4x2 portrait (or --cols N)

    # any def(s), paper row or not -- to see whether it belongs in the paper
    ~/ratvenv/venv/bin/python make_training_figure.py hardsar superdynamic --name try
    ~/ratvenv/venv/bin/python make_training_figure.py bigdreamer_ladder:consec4 --local

With no positional argument the panels are the paper worlds; naming experiment
ids instead (`EXP[:VARIATION]`, `--local` for results/experiments) plots any
def the same way, with whatever methods its runs contain -- handy for judging
whether a def is worth adding to the paper. Otherwise:

Same worlds, names, order and data as make_results_table.py / make_world_figure.py:
panels come from paper/results_table.yaml (row `label`, `exp`, `variation`,
`per_method`, `na`, `source`), methods and their paper names from its
`methods:`. RL methods (ppo, dreamer) are training curves read from each run's
train_episodes.jsonl (pull_run.sh -t mirror under results/rci); human /
frontier are horizontal lines at the mean of the first `n_eval` episodes of
<exp_dir>/external/<method>/episodes.jsonl -- the very numbers in the results
table -- with a faint +/-1 std band. Methods listed in a row's `na:` are left out.

Curves: per seed, a rolling mean over --rolling episodes against cumulative env
steps; per method, the mean across seeds on a shared step grid with a +/-1 std
band across seeds. By default the mean stops where the shortest seed stops
(same rule as analyze_experiment.py); `--min-seeds K` continues it while at
least K seeds still have data, the fewer-seeds tail drawn lighter so it is
obvious. `--per-seed` adds the thin individual-seed lines.

X-axis extent (`--xmax`):
  longest  the furthest a drawn curve reaches in that panel (default)
  def      the def's total training steps (sum of stage steps), taken from
           defs/<exp>.yaml -- the CURRENT target, which may have been
           lengthened since the pulled results/rci/<exp>/experiment.yaml was
           written -- falling back to that experiment.yaml. Curves already past
           the target are cut there.

One figure per metric: <out>/training_curves_<metric>[_<name>].{png,pdf};
--out defaults to `figures_out:` in paper/results_table.yaml, else
results/analysis/paper/. Needs pandas + matplotlib + pyyaml (sb3 venv).
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from analyze_experiment import (  # noqa: E402
    BASELINE_COLORS,
    BASELINE_STYLES,
    load_jsonl,
    parse_run_id,
)
from experiment_defs import load_experiment_def  # noqa: E402
from make_results_table import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFS_DIR,
    cell_spec,
    find_exp_dir,
    first_n,
    load_config,
    runs_for,
    stage_steps,
)
from record_trajectories import EXTERNAL_METHODS  # noqa: E402

HERE = Path(__file__).parent
DEFAULT_OUT = HERE / "results" / "analysis" / "paper"
METRIC_LABELS = {"total_score": "score", "objects_found": "objects found"}
N_GRID = 400


# ─────────────────────────────────────────────
#  Data
# ─────────────────────────────────────────────

def def_target_steps(exp: str, exp_dir: Path | None) -> int | None:
    """Total training steps of the def: defs/<exp>.yaml first (current target),
    then the experiment.yaml snapshot next to the pulled results."""
    cands = [DEFS_DIR / f"{exp}.yaml"]
    if exp_dir is not None:
        cands.append(exp_dir / "experiment.yaml")
    for p in cands:
        if p.exists():
            try:
                steps = stage_steps(load_experiment_def(p))
            except Exception as e:  # noqa: BLE001 — a broken def shouldn't kill the figure
                print(f"  WARN: could not parse {p}: {e}")
                continue
            if steps:
                return sum(steps)
    return None


def seed_curves(spec: dict, method: str, metric: str, rolling: int) -> list[tuple]:
    """[(cum_steps, rolling-mean metric)] per seed of `method` in this cell."""
    exp_dir = find_exp_dir(spec["exp"], spec["source"])
    if exp_dir is None:
        return []
    try:
        runs = runs_for(exp_dir)
    except FileNotFoundError:
        return []
    out = []
    for r in runs:
        df = r["train_df"]
        if (r["method"] != method or r["variation"] != spec["variation"] or df is None
                or metric not in df.columns or not df[metric].notna().any()):
            continue
        df = df.sort_values("episode_idx", kind="stable")
        x = df["steps"].cumsum().to_numpy(dtype=float)
        y = df[metric].astype(float).rolling(rolling, min_periods=1).mean().to_numpy()
        out.append((x, y))
    return out


def external_values(spec: dict, method: str, metric: str, n_eval: int) -> np.ndarray | None:
    exp_dir = find_exp_dir(spec["exp"], spec["source"])
    if exp_dir is None:
        return None
    df = load_jsonl(exp_dir / "external" / method / "episodes.jsonl")
    if df is None or metric not in df.columns:
        return None
    vals = first_n(df, n_eval)[metric].dropna().to_numpy(dtype=float)
    return vals if len(vals) else None


def mean_curve(curves: list[tuple], min_seeds: int) -> dict | None:
    """Mean / std across seeds on a shared grid. Grid runs from the latest
    seed start to the point where fewer than `min_seeds` seeds have data
    (min_seeds=0 -> all seeds, i.e. the shortest seed's end)."""
    n = len(curves)
    need = n if min_seeds <= 0 else min(min_seeds, n)
    ends = sorted((x[-1] for x, _ in curves), reverse=True)
    x_lo = max(x[0] for x, _ in curves)
    x_hi = ends[need - 1]
    if x_hi <= x_lo:
        return None
    grid = np.linspace(x_lo, x_hi, N_GRID)
    ys = np.full((n, N_GRID), np.nan)
    for i, (x, y) in enumerate(curves):
        inside = grid <= x[-1]
        ys[i, inside] = np.interp(grid[inside], x, y)
    count = np.sum(~np.isnan(ys), axis=0)
    mean = np.nanmean(ys, axis=0)
    std = np.zeros(N_GRID)
    multi = count > 1
    std[multi] = np.nanstd(ys[:, multi], axis=0, ddof=1)
    return {"x": grid, "mean": mean, "std": std, "count": count, "n": n}


# ─────────────────────────────────────────────
#  Drawing
# ─────────────────────────────────────────────

def method_colors(methods: list[str]) -> dict[str, tuple]:
    cmap = plt.colormaps.get_cmap("tab10")
    rl = [m for m in methods if m not in EXTERNAL_METHODS]
    return {m: cmap(i % 10) for i, m in enumerate(rl)}


def draw_panel(ax, row: dict, methods: list[str], metric: str, args, cfg: dict,
               colors: dict) -> dict:
    """Draws one world; returns {"x_end": furthest drawn step, "target": def steps,
    "notes": [...]} for the x-limit and the console summary."""
    na = set(row.get("na") or [])
    x_end, notes, drew_rl = 0.0, [], False
    for m in methods:
        if m in na or m in EXTERNAL_METHODS:
            continue
        spec = cell_spec(row, m)
        curves = seed_curves(spec, m, metric, args.rolling)
        if not curves:
            notes.append(f"{m}: no train data")
            continue
        mc = mean_curve(curves, args.min_seeds)
        c = colors[m]
        if args.per_seed:
            for x, y in curves:
                ax.plot(x / 1e6, y, color=c, lw=0.6, alpha=0.3)
                x_end = max(x_end, float(x[-1]))
        if mc is None:
            notes.append(f"{m}: seeds don't overlap")
            continue
        drew_rl = True
        full = mc["count"] == mc["n"]
        # fewer-seeds tail lighter; overlap one point so the line is continuous
        tail = ~full
        if tail.any() and full.any():
            tail[np.argmax(tail) - 1] = True
        for mask, alpha in ((full, 1.0), (tail, 0.45)):
            if not mask.any():
                continue
            x = mc["x"][mask] / 1e6
            ax.plot(x, mc["mean"][mask], color=c, lw=1.8, alpha=alpha)
            ax.fill_between(x, (mc["mean"] - mc["std"])[mask], (mc["mean"] + mc["std"])[mask],
                            color=c, alpha=0.18 * alpha, lw=0)
        x_end = max(x_end, float(mc["x"][-1]))
        notes.append(f"{m}: {mc['n']} seeds, to {mc['x'][-1] / 1e6:.1f}M")

    for m in methods:
        if m in na or m not in EXTERNAL_METHODS:
            continue
        vals = external_values(cell_spec(row, m), m, metric, cfg["n_eval"])
        if vals is None:
            notes.append(f"{m}: no episodes")
            continue
        mean = float(vals.mean())
        color = BASELINE_COLORS.get(m, "gray")
        ax.axhline(mean, color=color, linestyle=BASELINE_STYLES.get(m, "-."), lw=1.4)
        if len(vals) > 1:
            std = float(vals.std(ddof=1))
            ax.axhspan(mean - std, mean + std, color=color, alpha=0.07, lw=0)
        notes.append(f"{m}: n={len(vals)}")

    exp = row.get("exp")
    target = def_target_steps(exp, find_exp_dir(exp, row.get("source", "rci"))) if exp else None
    if not drew_rl:
        ax.text(0.5, 0.15, "no training data", ha="center", va="center",
                transform=ax.transAxes, color="0.5", fontsize=args.fontsize)
    return {"x_end": x_end, "target": target, "notes": notes}


def legend_entries(methods: list[str], colors: dict, cfg: dict) -> tuple[list, list[str]]:
    handles = []
    for m in methods:
        if m in EXTERNAL_METHODS:
            handles.append(Line2D([], [], color=BASELINE_COLORS.get(m, "gray"),
                                  linestyle=BASELINE_STYLES.get(m, "-."), lw=1.4))
        else:
            handles.append((Patch(color=colors[m], alpha=0.18, lw=0),
                            Line2D([], [], color=colors[m], lw=1.8)))
    return handles, [cfg["methods"][m] for m in methods]


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def add_layout_args(ap: argparse.ArgumentParser, panel_w: float, panel_h: float) -> None:
    """--layout / --cols / panel size; shared with make_final_boxplots.py."""
    ap.add_argument("--layout", choices=["wide", "tall"], default="wide",
                    help="wide: 4 panels per row (8 worlds -> 2x4, default); "
                         "tall: 2 per row (-> 4x2)")
    ap.add_argument("--cols", type=int, default=None,
                    help="panels per figure row; overrides --layout")
    ap.add_argument("--panel-w", type=float, default=panel_w, dest="panel_w", help="inches")
    ap.add_argument("--panel-h", type=float, default=panel_h, dest="panel_h", help="inches")


def grid_shape(n_panels: int, args) -> tuple[int, int]:
    ncols = min(args.cols or (4 if args.layout == "wide" else 2), n_panels)
    return -(-n_panels // ncols), ncols


def select_rows(cfg: dict, only: str | None) -> list[dict]:
    """Table rows in config order, or the --rows subset (labels or exp ids) in given order."""
    rows = [r for g in cfg["groups"] for r in g.get("rows", [])]
    if not only:
        return rows
    by = {r["label"]: r for r in rows} | {r["exp"]: r for r in rows}
    want = [k.strip() for k in only.split(",") if k.strip()]
    unknown = [k for k in want if k not in by]
    if unknown:
        sys.exit(f"ERROR: unknown rows {unknown}; known: {[r['label'] for r in rows]}")
    return [by[k] for k in want]


def add_exp_args(ap: argparse.ArgumentParser) -> None:
    """Ad-hoc mode: panels for defs that are not paper rows. Shared with
    make_final_boxplots.py."""
    ap.add_argument("exps", nargs="*", metavar="EXP[:VARIATION]",
                    help="experiment ids to plot instead of the paper rows, e.g. "
                         "`bigdreamer_ladder:consec4 superdynamic` (default: the rows in "
                         "paper/results_table.yaml)")
    ap.add_argument("--local", action="store_true",
                    help="look ad-hoc experiment ids up under results/experiments instead of "
                         "results/rci (same as record_trajectories.py --local)")


def rows_from_exps(items: list[str], source: str) -> list[dict]:
    """`EXP[:VARIATION]` -> table-row-shaped panels (label = exp, or
    `exp / variation` when it isn't `baseline`). No `na:`, so every method that
    has data is drawn."""
    rows = []
    for item in items:
        exp, _, variation = item.partition(":")
        variation = variation or "baseline"
        if find_exp_dir(exp, source) is None:
            sys.exit(f"ERROR: no results dir for '{exp}' under results/"
                     f"{'experiments' if source == 'local' else 'rci'} "
                     f"(pull_run.sh -t {exp}? --local?)")
        rows.append({"label": exp if variation == "baseline" else f"{exp} / {variation}",
                     "exp": exp, "variation": variation, "source": source})
    return rows


def methods_present(rows: list[dict], cfg: dict) -> list[str]:
    """Methods that actually have data in these experiments, config methods
    first (paper order / names) and anything else after under its own id --
    so an ad-hoc def's recurrent_ppo runs aren't silently dropped, and a def
    with no human run doesn't get an empty human slot."""
    found = []
    for row in rows:
        exp_dir = find_exp_dir(row["exp"], row.get("source", "rci"))
        if exp_dir is None:
            continue
        runs_dir = exp_dir / "runs"
        if runs_dir.is_dir():
            for p in sorted(runs_dir.iterdir()):
                parsed = parse_run_id(p.name) if p.is_dir() else None
                if parsed and parsed[0] == row["variation"] and parsed[1] not in found:
                    found.append(parsed[1])
        ext = exp_dir / "external"
        if ext.is_dir():
            found += [p.name for p in sorted(ext.iterdir())
                      if p.is_dir() and p.name not in found]
    known = [m for m in cfg["methods"] if m in found]
    extra = [m for m in found if m not in cfg["methods"]]
    cfg["methods"] = dict(cfg["methods"]) | {m: m for m in extra}
    rl = [m for m in known + extra if m not in EXTERNAL_METHODS]
    return rl + [m for m in known + extra if m in EXTERNAL_METHODS]


def panel_rows(cfg: dict, args) -> tuple[list[dict], list[str] | None]:
    """(rows, methods) -- ad-hoc `exps` if given, else the paper rows; methods
    is None for paper rows (the caller keeps the config's method list)."""
    if getattr(args, "exps", None):
        rows = rows_from_exps(args.exps, "local" if args.local else "rci")
        return rows, methods_present(rows, cfg)
    return select_rows(cfg, args.rows), None


def figure_legend(fig, handles: list, labels: list[str], fontsize: float) -> float:
    """Shared legend under the panels; returns the bottom fraction to keep free
    for fig.tight_layout(rect=(0, that, 1, 1)). Entries wrap onto more rows
    rather than off the edge when the figure is narrow (a 2-panel grid with
    five methods, say)."""
    ncol = max(1, min(len(handles), int(fig.get_figwidth() // 1.7)))
    n_legend_rows = -(-len(handles) // ncol)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False,
               fontsize=fontsize, bbox_to_anchor=(0.5, 0.0))
    return (0.15 + 0.25 * n_legend_rows) / fig.get_figheight()


def make_figure(rows: list[dict], methods: list[str], metric: str, args, cfg: dict):
    nrows, ncols = grid_shape(len(rows), args)
    colors = method_colors(methods)
    fig, axes = plt.subplots(nrows, ncols, figsize=(args.panel_w * ncols, args.panel_h * nrows),
                             squeeze=False)
    print(f"\n[training_figure] {metric}  (rolling={args.rolling}, xmax={args.xmax}, "
          f"min_seeds={args.min_seeds or 'all'})")
    for k, row in enumerate(rows):
        ax = axes[k // ncols][k % ncols]
        info = draw_panel(ax, row, methods, metric, args, cfg, colors)
        if args.xmax == "def" and info["target"]:
            right = info["target"]
        else:
            right = info["x_end"] or info["target"] or 1.0
        ax.set_xlim(0, right / 1e6)
        ax.set_title("\n".join(textwrap.wrap(row["label"], 32)), fontsize=args.fontsize + 1)
        ax.tick_params(labelsize=args.fontsize - 1)
        ax.grid(True, alpha=0.3)
        if k // ncols == nrows - 1 or k + ncols >= len(rows):
            ax.set_xlabel("env steps (M)", fontsize=args.fontsize)
        if k % ncols == 0:
            ax.set_ylabel(METRIC_LABELS.get(metric, metric), fontsize=args.fontsize)
        tgt = f"{info['target'] / 1e6:.1f}M" if info["target"] else "?"
        print(f"  {row['label']:<24} x to {right / 1e6:5.1f}M (def {tgt})  " + "; ".join(info["notes"]))
    for k in range(len(rows), nrows * ncols):
        axes[k // ncols][k % ncols].set_visible(False)

    handles, labels = legend_entries(methods, colors, cfg)
    bottom = figure_legend(fig, handles, labels, args.fontsize)
    if args.title:
        fig.suptitle(args.title, fontsize=args.fontsize + 2)
    fig.tight_layout(rect=(0, bottom, 1, 1))
    return fig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="paper/results_table.yaml")
    ap.add_argument("--metrics", default="total_score",
                    help="comma-separated: total_score, objects_found (one figure each)")
    ap.add_argument("--xmax", choices=["longest", "def"], default="longest",
                    help="x extent per panel: longest drawn curve, or the def's total steps")
    ap.add_argument("--rolling", type=int, default=100, help="rolling-mean window in episodes")
    ap.add_argument("--min-seeds", type=int, default=0, dest="min_seeds",
                    help="continue the mean while >= K seeds have data (0 = all seeds)")
    ap.add_argument("--per-seed", action="store_true", dest="per_seed",
                    help="also draw thin per-seed lines")
    ap.add_argument("--rows", default=None, help="subset of table rows (labels or exp ids)")
    ap.add_argument("--methods", default=None, help="subset of the methods drawn")
    add_exp_args(ap)
    add_layout_args(ap, panel_w=3.4, panel_h=2.5)
    ap.add_argument("--fontsize", type=float, default=9.0)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--title", default="")
    ap.add_argument("--name", default="", help="suffix for the output file name")
    ap.add_argument("--out", default=None,
                    help="output dir (default: yaml figures_out, else results/analysis/paper)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    rows, found = panel_rows(cfg, args)
    methods = found if found is not None else list(cfg["methods"])
    if args.methods:
        keep = [m.strip() for m in args.methods.split(",") if m.strip()]
        bad = [m for m in keep if m not in methods]
        if bad:
            sys.exit(f"ERROR: --methods {bad} not among {methods}")
        methods = [m for m in methods if m in keep]
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    bad = [m for m in metrics if m not in METRIC_LABELS]
    if bad:
        sys.exit(f"ERROR: --metrics {bad}; known: {list(METRIC_LABELS)}")

    out = Path(args.out or cfg.get("figures_out") or DEFAULT_OUT).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    for metric in metrics:
        fig = make_figure(rows, methods, metric, args, cfg)
        stem = f"training_curves_{metric}" + (f"_{args.name}" if args.name else "")
        png, pdf = out / f"{stem}.png", out / f"{stem}.pdf"
        fig.savefig(png, dpi=args.dpi)
        fig.savefig(pdf)
        plt.close(fig)
        print(f"[training_figure] -> {png}\n[training_figure] -> {pdf}")


if __name__ == "__main__":
    main()
