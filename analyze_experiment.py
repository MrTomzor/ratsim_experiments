"""Analyze a scheduler-driven experiment.

Usage:
    python analyze_experiment.py compare_5houses
    python analyze_experiment.py results/experiments/compare_5houses/
    python analyze_experiment.py compare_5houses --out my_plots/
    python analyze_experiment.py compare_5houses --rolling 100

Walks `<exp_dir>/runs/<variation>__<method>__seed<i>/`, loads each
`train_episodes.jsonl`, and writes:

  - `train_<metric>.png`   — rolling-mean curves vs cumulative env steps,
                             colored per variation, linestyle per method.
                             Per-seed lines drawn thin; per-(variation,method)
                             mean drawn bold.
  - `eval_<metric>.png`    — bar chart per metric, x-axis grouped by method,
                             bars colored per variation, error bars = std
                             across per-seed means (RL-paper standard).
                             Drawn only if any run has an eval_episodes.jsonl.

`eval_episodes.jsonl` is the cache: re-run this script without `--run-eval`
and it just re-reads whatever's on disk. Run with `--run-eval N` to (re)write
those files — one subprocess per run in the method's venv (the same env-var
convention the scheduler uses: $PPO_PYTHON_PATH / $DREAMER_PYTHON_PATH).
Dreamer eval is not yet implemented; those runs are skipped with a warning.

Output dir defaults to `<exp_dir>/analysis/` (or pass `--out`). Run from the
sb3 venv (needs pandas + matplotlib).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


REPO_ROOT = Path(__file__).parent
EXP_ROOT = REPO_ROOT / "results" / "experiments"
EVAL_SCRIPT = REPO_ROOT / "eval_one_run.py"

# Metrics we plot from train + eval JSONLs. Always present in the schema
# (see env.py's _log_episode_jsonl and test.py's make_episode_result).
TRAIN_METRICS = ["total_score", "objects_found"]
EVAL_METRICS = ["total_score", "objects_found"]


# -- Discovery --------------------------------------------------------------

def resolve_exp_dir(arg: str) -> Path:
    p = Path(arg)
    if p.is_dir():
        return p.resolve()
    cand = EXP_ROOT / arg
    if cand.is_dir():
        return cand.resolve()
    raise FileNotFoundError(
        f"Couldn't find experiment dir for '{arg}'. Tried:\n"
        f"  - direct path: {p}\n"
        f"  - {cand}\n"
        f"Available under {EXP_ROOT}: "
        f"{sorted(p.name for p in EXP_ROOT.iterdir() if p.is_dir()) if EXP_ROOT.is_dir() else '<none>'}")


def parse_run_id(run_dir_name: str) -> tuple[str, str, int] | None:
    """run dir name = `<variation>__<method>__seed<n>`. Split by `__` (double
    underscore) so single-underscore names like `with_gps` / `recurrent_ppo`
    survive."""
    parts = run_dir_name.split("__")
    if len(parts) < 3 or not parts[-1].startswith("seed"):
        return None
    seed_str = parts[-1][len("seed"):]
    if not seed_str.isdigit():
        return None
    method = parts[-2]
    variation = "__".join(parts[:-2])
    return variation, method, int(seed_str)


def load_jsonl(p: Path) -> pd.DataFrame | None:
    if not p.exists() or p.stat().st_size == 0:
        return None
    df = pd.read_json(p, lines=True)
    return df if len(df) else None


def discover_runs(exp_dir: Path) -> list[dict]:
    runs_dir = exp_dir / "runs"
    if not runs_dir.is_dir():
        raise FileNotFoundError(
            f"{runs_dir} doesn't exist — is this a scheduler exp dir? "
            f"Expected layout: <exp_dir>/runs/<variation>__<method>__seed<i>/")
    out = []
    for run_dir in sorted(runs_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        parsed = parse_run_id(run_dir.name)
        if parsed is None:
            print(f"[analyze] WARN: skipping unrecognized run dir: {run_dir.name}")
            continue
        variation, method, seed = parsed
        out.append({
            "run_dir": run_dir,
            "run_id": run_dir.name,
            "variation": variation,
            "method": method,
            "seed": seed,
            "train_df": load_jsonl(run_dir / "train_episodes.jsonl"),
            "eval_df": load_jsonl(run_dir / "eval_episodes.jsonl"),
            "done": (run_dir / "DONE").exists(),
        })
    return out


# -- Color / style assignment ----------------------------------------------

def variation_colors(variations) -> dict[str, tuple]:
    cmap = plt.colormaps.get_cmap("tab10")
    return {v: cmap(i % 10) for i, v in enumerate(sorted(set(variations)))}


def method_linestyles(methods) -> dict[str, str]:
    styles = ["-", "--", "-.", ":"]
    return {m: styles[i % len(styles)] for i, m in enumerate(sorted(set(methods)))}


# -- Plotting ---------------------------------------------------------------

def _add_cum_steps(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("episode_idx").copy()
    df["cum_steps"] = df["steps"].cumsum()
    return df


def plot_training_curve(runs: list[dict], metric: str, rolling: int,
                        out_dir: Path) -> None:
    """One image per metric, rolling mean vs cum_steps. Per-seed thin lines +
    per-(variation, method) mean bold line, with the mean computed on a shared
    interpolated x-grid (since seeds don't end at the same cum_steps)."""
    have = [r for r in runs if r["train_df"] is not None
            and metric in r["train_df"].columns]
    if not have:
        print(f"  -> skipping train_{metric}: no data")
        return

    var_color = variation_colors([r["variation"] for r in have])
    meth_style = method_linestyles([r["method"] for r in have])

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for r in have:
        df = _add_cum_steps(r["train_df"])
        smooth = df[metric].rolling(rolling, min_periods=1).mean()
        ax.plot(df["cum_steps"], smooth,
                color=var_color[r["variation"]],
                linestyle=meth_style[r["method"]],
                alpha=0.25, lw=0.8)

    groups = defaultdict(list)
    for r in have:
        groups[(r["variation"], r["method"])].append(r)
    for (var, meth), rs in groups.items():
        xs_per_seed, ys_per_seed = [], []
        for r in rs:
            df = _add_cum_steps(r["train_df"])
            xs_per_seed.append(df["cum_steps"].to_numpy())
            ys_per_seed.append(df[metric].rolling(rolling, min_periods=1).mean().to_numpy())
        # Common x-grid clipped to the shortest seed's run.
        x_lo = max(x[0] for x in xs_per_seed)
        x_hi = min(x[-1] for x in xs_per_seed)
        if x_hi <= x_lo:
            continue
        x_grid = np.linspace(x_lo, x_hi, 200)
        y_stack = np.stack([np.interp(x_grid, x, y)
                            for x, y in zip(xs_per_seed, ys_per_seed)])
        ax.plot(x_grid, y_stack.mean(axis=0),
                color=var_color[var], linestyle=meth_style[meth],
                alpha=1.0, lw=2.2,
                label=f"{var} / {meth} (n={len(rs)})")

    ax.set_xlabel("cumulative env steps")
    ax.set_ylabel(f"{metric} (rolling mean, w={rolling})")
    ax.set_title(f"Training: {metric}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / f"train_{metric}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  -> {out_path}")


def plot_eval_bar(runs: list[dict], metric: str, out_dir: Path) -> None:
    """Paper-style bar chart. X-axis = methods, bars-per-group = variations,
    error bars = std across per-seed means. Per-seed mean is the standard
    RL-paper unit because raw-episode std conflates within-run noise with
    between-seed variability."""
    have = [r for r in runs if r["eval_df"] is not None
            and metric in r["eval_df"].columns]
    if not have:
        return

    seed_means: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in have:
        seed_means[(r["variation"], r["method"])].append(
            float(r["eval_df"][metric].mean()))

    methods = sorted({m for (_, m) in seed_means})
    variations = sorted({v for (v, _) in seed_means})
    var_color = variation_colors(variations)

    n_var = len(variations)
    n_meth = len(methods)
    bar_w = 0.8 / max(n_var, 1)
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * n_meth + 1), 5))

    for i, var in enumerate(variations):
        xs, means, stds, ns = [], [], [], []
        for j, meth in enumerate(methods):
            vals = seed_means.get((var, meth), [])
            xs.append(j + (i - (n_var - 1) / 2) * bar_w)
            if vals:
                means.append(np.mean(vals))
                stds.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
                ns.append(len(vals))
            else:
                means.append(np.nan)
                stds.append(0.0)
                ns.append(0)
        ax.bar(xs, means, bar_w, yerr=stds, capsize=4,
               color=var_color[var], label=var,
               edgecolor="black", lw=0.5)
        for x, m, n in zip(xs, means, ns):
            if not np.isnan(m) and n > 0:
                ax.text(x, m, f"n={n}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(np.arange(n_meth))
    ax.set_xticklabels(methods)
    ax.set_ylabel(f"eval {metric}  (mean of per-seed means; err = std)")
    ax.set_title(f"Eval: {metric}")
    ax.legend(title="variation", fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = out_dir / f"eval_{metric}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  -> {out_path}")


# -- Eval dispatch ---------------------------------------------------------

def python_for_method(method_name: str) -> str:
    """Look up the python interpreter for a method via the same env-var
    convention the scheduler uses ($PPO_PYTHON_PATH / $DREAMER_PYTHON_PATH).
    Imported lazily so plot-only runs don't drag in scheduler config."""
    from scheduler.config import DEFAULT_PYTHON_ENV
    var = DEFAULT_PYTHON_ENV.get(method_name)
    if var is None:
        raise ValueError(
            f"no python_env mapping for method '{method_name}' in "
            f"scheduler/config.py:DEFAULT_PYTHON_ENV.")
    val = os.environ.get(var)
    if not val:
        raise EnvironmentError(
            f"env var ${var} (python for method '{method_name}') is unset. "
            f"Add `export {var}=...` to your shell rc.")
    return val


def run_eval_for_runs(runs: list[dict], exp_dir: Path, n_episodes: int,
                      deterministic: bool = False,
                      eval_metaseed: int = 42) -> None:
    """Sequentially shell out to eval_one_run.py for each run in its method's
    venv. Subprocess inherits stdout/stderr so per-episode progress streams
    live. Failures are reported but don't abort the loop — other runs still
    get a shot."""
    succeeded, skipped, failed = [], [], []
    for r in runs:
        method = r["method"]
        if method == "dreamer":
            print(f"\n[eval] SKIP {r['run_id']}: dreamer eval not yet "
                  f"implemented (needs the dreamer venv + embodied loader).")
            skipped.append(r["run_id"])
            continue
        try:
            python = python_for_method(method)
        except (ValueError, EnvironmentError) as e:
            print(f"\n[eval] SKIP {r['run_id']}: {e}")
            skipped.append(r["run_id"])
            continue
        cmd = [python, str(EVAL_SCRIPT),
               "--run_dir", str(r["run_dir"]),
               "--exp_dir", str(exp_dir),
               "--n_episodes", str(n_episodes),
               "--eval_metaseed", str(eval_metaseed)]
        if deterministic:
            cmd.append("--deterministic")
        print(f"\n[eval] === {r['run_id']} ===")
        print(f"[eval] cmd: {' '.join(cmd)}")
        rc = subprocess.run(cmd).returncode
        if rc == 0:
            succeeded.append(r["run_id"])
        else:
            print(f"[eval] FAILED (exit={rc})")
            failed.append(r["run_id"])

    print(f"\n[eval] Summary: ok={len(succeeded)}  failed={len(failed)}  "
          f"skipped={len(skipped)}")
    if failed:
        print(f"  failed: {failed}")
    if skipped:
        print(f"  skipped: {skipped}")


# -- Summary table ---------------------------------------------------------

def print_summary(runs: list[dict]) -> None:
    rows = []
    for r in runs:
        td, ed = r["train_df"], r["eval_df"]
        rows.append({
            "run_id": r["run_id"],
            "done": "✓" if r["done"] else "·",
            "train_eps": 0 if td is None else len(td),
            "train_score": "" if td is None else f"{td['total_score'].mean():.2f}",
            "train_pickups": "" if td is None else f"{td['objects_found'].mean():.2f}",
            "eval_eps": 0 if ed is None else len(ed),
            "eval_score": "" if ed is None else f"{ed['total_score'].mean():.2f}",
            "eval_pickups": "" if ed is None else f"{ed['objects_found'].mean():.2f}",
        })
    df = pd.DataFrame(rows)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(df.to_string(index=False))


# -- Main -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp", help="Experiment id (looked up under results/experiments/) "
                                "or path to an exp dir")
    ap.add_argument("--out", default=None,
                    help="Output dir for plots (default: <exp_dir>/analysis/)")
    ap.add_argument("--rolling", type=int, default=50,
                    help="Rolling window for training curves, in episodes "
                         "(default: 50)")
    ap.add_argument("--run-eval", type=int, default=None, metavar="N",
                    dest="run_eval",
                    help="Before plotting, run N eval episodes per run on "
                         "the latest checkpoint. Spawns one subprocess per "
                         "run in the method's venv; overwrites any existing "
                         "eval_episodes.jsonl. Re-running without this flag "
                         "just re-plots from the cached JSONLs.")
    ap.add_argument("--deterministic", action="store_true",
                    help="Pass --deterministic to eval_one_run.py "
                         "(default is stochastic, which matches training-time "
                         "behaviour). Only meaningful with --run-eval.")
    ap.add_argument("--eval-metaseed", type=int, default=42, dest="eval_metaseed",
                    help="World-generation metaseed for eval (default 42). "
                         "All runs in the experiment evaluate on the same "
                         "world sequence drawn from this seed; pass the same "
                         "value to a human-control session to reproduce. Only "
                         "meaningful with --run-eval.")
    args = ap.parse_args()

    if args.run_eval is not None and args.run_eval < 1:
        ap.error("--run-eval N requires N >= 1")

    exp_dir = resolve_exp_dir(args.exp)
    print(f"Experiment dir: {exp_dir}")

    runs = discover_runs(exp_dir)
    if not runs:
        print("ERROR: no runs found under runs/.")
        sys.exit(1)

    if args.run_eval is not None:
        mode = "deterministic" if args.deterministic else "stochastic"
        print(f"\nRunning {args.run_eval} {mode} eval episode(s) per run "
              f"(eval_metaseed={args.eval_metaseed}, sequential, may take a while)...")
        run_eval_for_runs(runs, exp_dir, args.run_eval,
                          deterministic=args.deterministic,
                          eval_metaseed=args.eval_metaseed)
        # Re-discover so newly written eval_episodes.jsonl files are picked up.
        runs = discover_runs(exp_dir)
    elif args.deterministic:
        print("[analyze] WARN: --deterministic only takes effect with --run-eval; "
              "ignoring.")

    n_train = sum(1 for r in runs if r["train_df"] is not None)
    n_eval = sum(1 for r in runs if r["eval_df"] is not None)
    print(f"\nFound {len(runs)} run(s): {n_train} with train data, "
          f"{n_eval} with eval data.\n")
    print_summary(runs)

    out_dir = Path(args.out) if args.out else (exp_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nPlots -> {out_dir}/")

    for m in TRAIN_METRICS:
        plot_training_curve(runs, m, args.rolling, out_dir)

    if n_eval:
        for m in EVAL_METRICS:
            plot_eval_bar(runs, m, out_dir)
    else:
        print("  (no eval_episodes.jsonl found; skipping eval bar charts. "
              "Run with --run-eval N to populate them.)")

    print("\nDone.")


if __name__ == "__main__":
    main()
