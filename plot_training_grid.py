"""Training curves of several experiments side by side, with baseline lines.

    python plot_training_grid.py memory_orthomaze memory_3malls memory_fencemaze memory_dynahouses
    python plot_training_grid.py memory_3malls memory_fencemaze --name mazes --rolling 100 \\
        --methods ppo,dreamer --variations baseline,consec4
    python plot_training_grid.py memory_3malls memory_dynahouses \\
        --baseline memory_dynahouses:human=1100 --baseline memory_dynahouses:human.objects_found=18
    python plot_training_grid.py compare_5houses --local

Columns = experiments in the order given; rows = `total_score`, `objects_found`
and, when any experiment's runs carry it, `difficulty`. Each panel is the same
drawing `analyze_experiment.py` puts in `train_<metric>.png` (per-seed thin
lines, per-(variation, method) mean bold, colour = variation, linestyle =
method), so the two never disagree.

Horizontal lines are external baselines — human / frontier episodes under
`<exp_dir>/external/<method>/episodes.jsonl`, i.e. whatever
`record_trajectories.py --methods frontier,human` (or `test.py ...
results_dir=<exp_dir>/external/<method>`) has recorded for that experiment.
Nothing has to be typed in: the line sits at the mean over all recorded
episodes, the band is +/-1 std, and the legend says `n=` and, for adaptive
defs, the difficulty rung the baseline was pinned at. `--baseline
EXP:METHOD[.METRIC]=VALUE` (metric defaults to total_score) is the escape
hatch for a def with no external data yet; such lines are labelled `manual`.
Baseline lines are drawn on the score rows only, never on `difficulty`.

Experiment ids are looked up under results/rci (pulled cluster runs, see
pull_run.sh -t); `--local` looks under results/experiments instead; a path is
used as-is. Output: <out>/training_grid[_<name>].png, default out =
results/analysis/<exp1>+<exp2>+.../ (same convention as plot_trajectories.py).
Needs pandas + matplotlib (sb3 venv).
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from analyze_experiment import (  # noqa: E402
    OPTIONAL_TRAIN_METRICS,
    TRAIN_METRICS,
    baseline_label,
    discover_runs,
    draw_baselines,
    draw_training_curve,
    load_external_baselines,
    method_linestyles,
    variation_colors,
)
from record_trajectories import resolve_exp_dir  # noqa: E402

HERE = Path(__file__).parent


def _csv(s: str | None) -> list[str] | None:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_manual_baselines(items: list[str]) -> dict[str, list[dict]]:
    """`EXP:METHOD[.METRIC]=VALUE` (METRIC defaults to total_score) ->
    {exp: [baseline dict with a one-row df holding just that metric]}. Same
    shape as `load_external_baselines` returns, so `draw_baselines` treats
    both alike and skips the metrics a manual entry doesn't carry; the label
    says `manual` instead of `eval, n=`. Several entries for the same method
    (e.g. `.total_score` and `.objects_found`) are merged into one line set."""
    out: dict[str, list[dict]] = {}
    for item in items:
        try:
            exp, rest = item.split(":", 1)
            key, value = rest.split("=", 1)
            method, _, metric = key.partition(".")
            metric = metric or "total_score"
            value = float(value)
        except ValueError:
            sys.exit(f"ERROR: --baseline expects EXP:METHOD[.METRIC]=VALUE, got {item!r}")
        if metric not in TRAIN_METRICS:
            sys.exit(f"ERROR: --baseline metric must be one of {TRAIN_METRICS}, got {metric!r}")
        entries = out.setdefault(exp.strip(), [])
        for b in entries:
            if b["method"] == method.strip():
                b["df"][metric] = [value]
                break
        else:
            entries.append({"method": method.strip(), "df": pd.DataFrame({metric: [value]}),
                            "n": 1, "difficulty": None, "manual": True})
    return out


def load_column(name: str, source: str, methods: list[str] | None,
                variations: list[str] | None) -> dict:
    exp_dir = resolve_exp_dir(name, source)
    runs = discover_runs(exp_dir)
    if methods:
        runs = [r for r in runs if r["method"] in methods]
    if variations:
        runs = [r for r in runs if r["variation"] in variations]
    return {"name": name, "exp_dir": exp_dir, "runs": runs,
            "baselines": load_external_baselines(exp_dir)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exps", nargs="+", help="experiment ids in column order (or paths)")
    ap.add_argument("--methods", default=None,
                    help="comma-separated RL methods to keep (default: all)")
    ap.add_argument("--variations", default=None,
                    help="comma-separated variations to keep (default: all)")
    ap.add_argument("--rolling", type=int, default=50,
                    help="rolling-mean window in episodes (default 50)")
    ap.add_argument("--baseline", action="append", default=[],
                    metavar="EXP:METHOD[.METRIC]=VALUE",
                    help="manual horizontal line for one experiment; METRIC "
                         "defaults to total_score, e.g. memory_3malls:human=1100 "
                         "memory_3malls:human.objects_found=18 (repeatable)")
    ap.add_argument("--no-baselines", action="store_true", dest="no_baselines",
                    help="skip the external/ baseline lines")
    ap.add_argument("--sharey", action="store_true",
                    help="share the y-axis within each metric row")
    ap.add_argument("--panel", type=float, default=4.2, help="panel width in inches")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--title", default=None)
    ap.add_argument("--name", default=None, help="suffix for the output file name")
    ap.add_argument("--out", default=None, help="output dir (default results/analysis/<exps>/)")
    ap.add_argument("--local", action="store_true",
                    help="look experiment ids up under results/experiments instead of the "
                         "default results/rci (same as record_trajectories.py --local)")
    args = ap.parse_args()

    source = "local" if args.local else "rci"
    methods, variations = _csv(args.methods), _csv(args.variations)
    manual = parse_manual_baselines(args.baseline)

    cols = []
    for name in args.exps:
        try:
            col = load_column(name, source, methods, variations)
        except FileNotFoundError as e:
            sys.exit(f"ERROR: {e}")
        keys = {name, col["exp_dir"].name}      # id as typed, or the dir name
        col["baselines"] = ([] if args.no_baselines else col["baselines"]) \
            + [b for k in sorted(keys) for b in manual.get(k, [])]
        cols.append(col)
        n_train = sum(1 for r in col["runs"] if r["train_df"] is not None)
        bl = ", ".join(baseline_label(b) for b in col["baselines"]) or "none"
        print(f"{name}: {len(col['runs'])} run(s), {n_train} with train data; baselines: {bl}")

    metrics = list(TRAIN_METRICS)
    for m in OPTIONAL_TRAIN_METRICS:
        if any(r["train_df"] is not None and m in r["train_df"].columns
               and r["train_df"][m].notna().any()
               for c in cols for r in c["runs"]):
            metrics.append(m)

    # One colour / linestyle map across the whole figure so `baseline / ppo`
    # looks the same in every column.
    all_runs = [r for c in cols for r in c["runs"] if r["train_df"] is not None]
    var_color = variation_colors([r["variation"] for r in all_runs])
    meth_style = method_linestyles([r["method"] for r in all_runs])

    n_rows, n_cols = len(metrics), len(cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(args.panel * n_cols, 3.4 * n_rows),
                             squeeze=False, sharey="row" if args.sharey else False)
    for j, col in enumerate(cols):
        for i, metric in enumerate(metrics):
            ax = axes[i][j]
            drawn = draw_training_curve(ax, col["runs"], metric, args.rolling,
                                        var_color, meth_style)
            if drawn and metric not in OPTIONAL_TRAIN_METRICS and col["baselines"]:
                draw_baselines(ax, col["baselines"], metric)
            if not drawn:
                ax.text(0.5, 0.5, f"no {metric} data", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                ax.set_xticks([]); ax.set_yticks([])
            if i == 0:
                ax.set_title("\n".join(textwrap.wrap(col["exp_dir"].name, 28)),
                             fontsize=10)
            if j != 0:
                ax.set_ylabel("")
            if i != n_rows - 1:
                ax.set_xlabel("")
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=7, loc="best")
    if args.title:
        fig.suptitle(args.title)
    fig.tight_layout()

    out_dir = (Path(args.out) if args.out
               else HERE / "results" / "analysis" / "+".join(c["exp_dir"].name for c in cols))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"training_grid{'_' + args.name if args.name else ''}.png"
    fig.savefig(out_path, dpi=args.dpi)
    plt.close(fig)
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
