"""Analyze ratsim training runs from `train_episodes.jsonl` files.

Usage
-----
    python analyze_run_data.py <path> [<path> ...] [--out <dir>] [--rolling N]

Each <path> can be:
  - a single run dir (one containing `train_episodes.jsonl`)
  - a parent dir (searched recursively, symlinks followed)
  - a symlink to either of the above
  - any mix

Outputs
-------
- Terminal: validation warnings, per-run summary, per-(method, rundef) summary,
  global termination-reason breakdown.
- PNGs in <out>/ (default: `analysis_output/`): learning curves for several
  metrics, a curve vs cumulative env steps, and a termination-reason bar chart.

Run from the sb3 venv (needs pandas + matplotlib).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


REQUIRED_FIELDS = [
    "method", "rundef", "stage_idx", "seed", "episode_idx",
    "steps", "total_score", "objects_found", "collisions",
    "termination_reason", "distance_traveled", "wall_time_s",
]


# -- Discovery --------------------------------------------------------------

def discover_jsonls(paths: list[Path]) -> list[Path]:
    """Recursively find train_episodes.jsonl under each path; follow symlinks."""
    found: list[Path] = []
    for p in paths:
        if not p.exists():
            print(f"WARN: path does not exist: {p}")
            continue
        if p.is_file() and p.name == "train_episodes.jsonl":
            found.append(p.resolve())
            continue
        for root, _dirs, files in os.walk(p, followlinks=True):
            if "train_episodes.jsonl" in files:
                found.append((Path(root) / "train_episodes.jsonl").resolve())
    return sorted(set(found))


# -- Loading + validation ---------------------------------------------------

def load_run(jsonl_path: Path) -> pd.DataFrame:
    run_dir = jsonl_path.parent
    df = pd.read_json(jsonl_path, lines=True)
    df["run_dir"] = str(run_dir)
    df["run_name"] = run_dir.name
    df["done_marker"] = (run_dir / "DONE").exists()
    return df


def validate_run(df: pd.DataFrame, run_name: str) -> list[str]:
    """Sanity checks that replace the manual head -1 / episode_idx checks."""
    issues: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in df.columns:
            issues.append(f"[{run_name}] missing schema field: {field}")
    present = [f for f in REQUIRED_FIELDS if f in df.columns]
    for f in present:
        n_null = df[f].isnull().sum()
        if n_null:
            issues.append(f"[{run_name}] {n_null} nulls in '{f}'")
    if "episode_idx" in df.columns and len(df) > 0:
        idxs = df["episode_idx"].tolist()
        if idxs != sorted(idxs):
            issues.append(f"[{run_name}] episode_idx NOT monotonic across file")
        if len(set(idxs)) != len(idxs):
            issues.append(f"[{run_name}] episode_idx has duplicates")
        if min(idxs) < 1:
            issues.append(f"[{run_name}] episode_idx starts at {min(idxs)} (expected 1+)")
    if not df["done_marker"].iloc[0]:
        issues.append(f"[{run_name}] missing DONE marker (run may be incomplete)")
    return issues


# -- Terminal summaries ------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    print("\n" + "=" * 72)
    print("PER-RUN SUMMARY")
    print("=" * 72)
    g = df.groupby("run_name", sort=False).agg(
        method=("method", "first"),
        rundef=("rundef", "first"),
        seed=("seed", "first"),
        stages=("stage_idx", "nunique"),
        episodes=("episode_idx", "count"),
        mean_score=("total_score", "mean"),
        mean_objects=("objects_found", "mean"),
        mean_coll=("collisions", "mean"),
        mean_dist=("distance_traveled", "mean"),
        wall_s=("wall_time_s", "sum"),
        done=("done_marker", "first"),
    )
    with pd.option_context("display.max_rows", None, "display.width", 200,
                           "display.float_format", lambda x: f"{x:.2f}"):
        print(g.to_string())

    print("\n" + "=" * 72)
    print("BY (method, rundef) — aggregated across seeds")
    print("=" * 72)
    mr = df.groupby(["method", "rundef"]).agg(
        n_runs=("run_name", "nunique"),
        n_episodes=("episode_idx", "count"),
        mean_score=("total_score", "mean"),
        median_score=("total_score", "median"),
        std_score=("total_score", "std"),
        mean_objects=("objects_found", "mean"),
        mean_coll=("collisions", "mean"),
    )
    with pd.option_context("display.width", 200,
                           "display.float_format", lambda x: f"{x:.2f}"):
        print(mr.to_string())

    print("\n" + "=" * 72)
    print("TERMINATION REASONS (global, % of episodes)")
    print("=" * 72)
    reasons = df["termination_reason"].value_counts(normalize=True) * 100
    for r, p in reasons.items():
        print(f"  {r:28s} {p:6.2f}%")


# -- Plotting ----------------------------------------------------------------

def _method_colors(df: pd.DataFrame) -> dict:
    methods = sorted(df["method"].unique())
    cmap = plt.colormaps.get_cmap("tab10")
    return {m: cmap(i % 10) for i, m in enumerate(methods)}


def plot_curve(df: pd.DataFrame, out: Path, metric: str,
               x: str = "episode_idx", rolling: int = 20) -> None:
    """One line per run; x is either `episode_idx` or cumulative env steps."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = _method_colors(df)

    for _, rdf in df.groupby("run_name", sort=False):
        rdf = rdf.sort_values("episode_idx").copy()
        if x == "cum_steps":
            rdf["cum_steps"] = rdf["steps"].cumsum()
            xs = rdf["cum_steps"]
            xlab = "cumulative env steps"
        else:
            xs = rdf["episode_idx"]
            xlab = "episode_idx (cumulative across stages)"
        vals = rdf[metric].rolling(rolling, min_periods=1).mean()
        color = colors[rdf["method"].iloc[0]]
        label = f"{rdf['method'].iloc[0]} / {rdf['rundef'].iloc[0]} / s={rdf['seed'].iloc[0]}"
        ax.plot(xs, vals, color=color, alpha=0.75, lw=1.2, label=label)

    # dedupe legend entries
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), fontsize=7, loc="best")
    ax.set_xlabel(xlab)
    ax.set_ylabel(f"{metric} (rolling mean, w={rolling})")
    ax.set_title(f"{metric} vs {xlab}")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    suffix = "_steps" if x == "cum_steps" else ""
    path = out / f"curve_{metric}{suffix}.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  -> {path}")


def plot_termination_reasons(df: pd.DataFrame, out: Path) -> None:
    ct = pd.crosstab(df["run_name"], df["termination_reason"], normalize="index") * 100
    fig, ax = plt.subplots(figsize=(10, max(3, 0.35 * len(ct))))
    ct.plot(kind="barh", stacked=True, ax=ax, colormap="tab20")
    ax.set_xlabel("% of episodes")
    ax.set_ylabel("")
    ax.set_title("Termination reasons by run")
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    path = out / "termination_reasons.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"  -> {path}")


# -- Main --------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+",
                    help="Run dirs, their parents, or symlinks (recursive, follows symlinks).")
    ap.add_argument("--out", default="analysis_output",
                    help="Output dir for plots (default: analysis_output/)")
    ap.add_argument("--rolling", type=int, default=20,
                    help="Rolling window for learning curves (default: 20 episodes)")
    args = ap.parse_args()

    jsonls = discover_jsonls([Path(p) for p in args.paths])
    if not jsonls:
        print("ERROR: no train_episodes.jsonl files found under given paths.")
        sys.exit(1)
    print(f"Found {len(jsonls)} JSONL file(s):")
    for j in jsonls:
        print(f"  {j}")

    dfs: list[pd.DataFrame] = []
    issues: list[str] = []
    for j in jsonls:
        try:
            rdf = load_run(j)
        except Exception as e:
            issues.append(f"[{j.parent.name}] failed to load: {e}")
            continue
        if len(rdf) == 0:
            issues.append(f"[{j.parent.name}] JSONL is empty")
            continue
        issues.extend(validate_run(rdf, j.parent.name))
        dfs.append(rdf)

    if issues:
        print("\n" + "=" * 72)
        print("VALIDATION ISSUES")
        print("=" * 72)
        for i in issues:
            print("  " + i)

    if not dfs:
        print("\nERROR: no loadable runs; aborting.")
        sys.exit(1)

    df = pd.concat(dfs, ignore_index=True)
    print(f"\nLoaded {len(df)} episodes across {df['run_name'].nunique()} run(s).")

    print_summary(df)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 72)
    print(f"PLOTS -> {out_dir}/")
    print("=" * 72)
    for metric in ("total_score", "objects_found", "collisions", "distance_traveled"):
        plot_curve(df, out_dir, metric=metric, x="episode_idx", rolling=args.rolling)
    plot_curve(df, out_dir, metric="total_score", x="cum_steps", rolling=args.rolling)
    plot_termination_reasons(df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
