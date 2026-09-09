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
                             mean drawn bold. Wrapper-added fields like
                             `difficulty` (adaptive defs) are plotted too
                             whenever any run's JSONL carries them. Any
                             external baseline under `<exp_dir>/external/
                             <method>/episodes.jsonl` (human / frontier, as
                             written by record_trajectories.py or test.py
                             `results_dir=`) is drawn as a horizontal line at
                             its mean with a +/-1 std band.
  - `eval_<metric>.png`    — bar chart per metric, x-axis grouped by method,
                             bars colored per variation, error bars = std
                             across per-seed means (RL-paper standard).
                             Drawn only if any run has an eval_episodes.jsonl.

`eval_episodes.jsonl` is the cache: re-run this script without `--run-eval`
and it just re-reads whatever's on disk. Run with `--run-eval N` to (re)write
those files — one subprocess per run in the method's venv (the same env-var
convention the scheduler uses: $PPO_PYTHON_PATH / $DREAMER_PYTHON_PATH).
SB3 runs go through `eval_one_run.py`; dreamer runs go through
`eval_one_run_dreamer.py` (loads via embodied + dreamerv3.Agent).

Add `--difficulty D` (0..1) with `--run-eval` for a def with an
`adaptive_difficulty:` block to evaluate every run at one fixed rung of the
difficulty ladder instead of at the def's base world config. It overwrites the
same `eval_episodes.jsonl`, so a run at a different D replaces the previous
eval; each record carries the `difficulty` field that produced it.

Add `--ablate-memory` (dreamer-only) to also run an RSSM amnesia eval; results
land in `eval_episodes_ablated.jsonl` and produce
`eval_<metric>_ablation.png` paired bars next to the baseline plots.

Output dir defaults to `<exp_dir>/analysis/` (or pass `--out`). Run from the
sb3 venv (needs pandas + matplotlib).
"""
from __future__ import annotations

import argparse
import json
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
EVAL_SCRIPT_DREAMER = REPO_ROOT / "eval_one_run_dreamer.py"

# Method → eval script. Splits because dreamer needs the dreamer venv +
# embodied agent loader, while sb3 methods load via stable_baselines3.
EVAL_SCRIPT_BY_METHOD = {
    "ppo": EVAL_SCRIPT,
    "recurrent_ppo": EVAL_SCRIPT,
    "cnn_ppo": EVAL_SCRIPT,
    "cnn_recurrent_ppo": EVAL_SCRIPT,
    "dreamer": EVAL_SCRIPT_DREAMER,
}

# Metrics we plot from train + eval JSONLs. Always present in the schema
# (see env.py's _log_episode_jsonl and test.py's make_episode_result).
TRAIN_METRICS = ["total_score", "objects_found"]
EVAL_METRICS = ["total_score", "objects_found"]
# Plotted only when at least one run's train JSONL actually carries the field
# (wrappers add these via extra_log_fields — e.g. 'difficulty' from
# AdaptiveDifficultyWrapper on defs with an `adaptive_difficulty:` block).
OPTIONAL_TRAIN_METRICS = ["difficulty"]


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
    """Load a JSONL file, skipping unparseable lines with a warning.

    Concurrent appends from n_envs>1 could tear lines on network filesystems
    (fixed writer-side with flock in env.py, but files written before that fix
    carry a few torn records). A whole-file pd.read_json would refuse the
    entire run over one bad line; parsing per line just drops it."""
    if not p.exists() or p.stat().st_size == 0:
        return None
    records, bad = [], 0
    with open(p) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except ValueError:
                bad += 1
    if bad:
        print(f"  WARN: {p}: skipped {bad} corrupt line(s) "
              f"({len(records)} good)")
    return pd.DataFrame.from_records(records) if records else None


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
            "eval_df_ablated": load_jsonl(
                run_dir / "eval_episodes_ablated.jsonl"),
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


def draw_training_curve(ax, runs: list[dict], metric: str, rolling: int,
                        var_color: dict | None = None,
                        meth_style: dict | None = None) -> bool:
    """Paint rolling-mean training curves for one metric onto `ax`.

    Per-seed lines drawn thin; per-(variation, method) mean drawn bold, with
    the mean computed on a shared interpolated x-grid (seeds don't end at the
    same cum_steps). Returns False (and draws nothing) when no run carries the
    metric. `var_color` / `meth_style` let a caller keep colours consistent
    across several axes; by default they are derived from `runs`."""
    have = [r for r in runs if r["train_df"] is not None
            and metric in r["train_df"].columns
            and r["train_df"][metric].notna().any()]
    if not have:
        return False

    var_color = var_color or variation_colors([r["variation"] for r in have])
    meth_style = meth_style or method_linestyles([r["method"] for r in have])

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
    ax.grid(True, alpha=0.3)
    return True


def plot_training_curve(runs: list[dict], metric: str, rolling: int,
                        out_dir: Path,
                        baselines: list[dict] | None = None) -> None:
    """One image per metric: `draw_training_curve` plus, for the score-like
    metrics, horizontal lines for any external baselines (human / frontier
    eval episodes under `<exp_dir>/external/`, see `load_external_baselines`)."""
    fig, ax = plt.subplots(figsize=(10, 5.5))
    if not draw_training_curve(ax, runs, metric, rolling):
        plt.close(fig)
        print(f"  -> skipping train_{metric}: no data")
        return
    if baselines and metric not in OPTIONAL_TRAIN_METRICS:
        draw_baselines(ax, baselines, metric)
    ax.set_title(f"Training: {metric}")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    out_path = out_dir / f"train_{metric}.png"
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  -> {out_path}")


# -- External baselines (human / frontier) ---------------------------------

# Colours for the horizontal baseline lines; deliberately outside tab10, which
# `variation_colors` hands to the RL curves, so a baseline never looks like a
# variation. Unknown external methods fall back to a grey.
BASELINE_COLORS = {"human": "black", "frontier": "dimgray"}
BASELINE_STYLES = {"human": "-", "frontier": "--"}


def load_external_baselines(exp_dir: Path) -> list[dict]:
    """Episodes of non-RL methods evaluated on this experiment's world, as
    written by `test.py ... results_dir=<exp_dir>/external/<method>/` (which
    is what `record_trajectories.py --methods frontier,human` runs). One dict
    per method: `method`, `df` (all episodes, method-invariant schema), `n`,
    `difficulty` (the fixed rung the eval was pinned at, from eval_config.json;
    None for non-adaptive defs or an unpinned eval). Empty list when there is
    no `external/` dir."""
    ext = exp_dir / "external"
    if not ext.is_dir():
        return []
    out = []
    for mdir in sorted(ext.iterdir()):
        if not mdir.is_dir():
            continue
        df = load_jsonl(mdir / "episodes.jsonl")
        if df is None:
            continue
        difficulty = None
        cfg = mdir / "eval_config.json"
        if cfg.exists():
            try:
                difficulty = json.loads(cfg.read_text()).get("difficulty")
            except ValueError:
                pass
        if "difficulty" in df.columns and df["difficulty"].notna().any():
            difficulty = float(df["difficulty"].dropna().iloc[-1])
        out.append({"method": mdir.name, "df": df, "n": len(df),
                    "difficulty": difficulty})
    return out


def baseline_label(b: dict) -> str:
    """`frontier (eval, n=10, d=1.00)`; a hand-typed value (plot_training_grid
    --baseline) is marked `(manual)` so it can't pass for measured data."""
    if b.get("manual"):
        return f"{b['method']} (manual)"
    d = "" if b.get("difficulty") is None else f", d={b['difficulty']:.2f}"
    return f"{b['method']} (eval, n={b['n']}{d})"


def draw_baselines(ax, baselines: list[dict], metric: str) -> None:
    """Horizontal line at each external method's mean `metric`, with a faint
    band of +/- one std across its episodes (no band for n=1). Labels carry
    the episode count and, for adaptive defs, the difficulty rung, so the
    figure says what the line is worth."""
    for b in baselines:
        if metric not in b["df"].columns:
            continue
        vals = b["df"][metric].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        color = BASELINE_COLORS.get(b["method"], "gray")
        style = BASELINE_STYLES.get(b["method"], "-.")
        mean = float(vals.mean())
        ax.axhline(mean, color=color, linestyle=style, lw=1.6,
                   label=baseline_label(b))
        if len(vals) > 1:
            std = float(vals.std(ddof=1))
            ax.axhspan(mean - std, mean + std, color=color, alpha=0.08, lw=0)


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


def plot_eval_ablation_bar(runs: list[dict], metric: str,
                           out_dir: Path) -> None:
    """Paired bar chart: baseline vs memory-ablated, grouped by
    (variation, method). Drawn only if at least one run has ablated data on
    disk. Bars use the same per-variation color scheme as the standard eval
    plot; ablated bars are hatched and slightly transparent so the
    comparison is readable in a single glance."""
    have_baseline_or_ablated = [
        r for r in runs
        if (r["eval_df"] is not None
            and metric in r["eval_df"].columns)
        or (r["eval_df_ablated"] is not None
            and metric in r["eval_df_ablated"].columns)
    ]
    if not any(r["eval_df_ablated"] is not None
               and metric in (r["eval_df_ablated"].columns
                              if r["eval_df_ablated"] is not None else [])
               for r in have_baseline_or_ablated):
        return  # no ablated data → nothing to compare

    base_seed_means: dict[tuple[str, str], list[float]] = defaultdict(list)
    abl_seed_means: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in have_baseline_or_ablated:
        key = (r["variation"], r["method"])
        if r["eval_df"] is not None and metric in r["eval_df"].columns:
            base_seed_means[key].append(float(r["eval_df"][metric].mean()))
        if (r["eval_df_ablated"] is not None
                and metric in r["eval_df_ablated"].columns):
            abl_seed_means[key].append(
                float(r["eval_df_ablated"][metric].mean()))

    keys = sorted(set(base_seed_means) | set(abl_seed_means))
    if not keys:
        return
    variations = sorted({v for (v, _) in keys})
    var_color = variation_colors(variations)

    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(keys) + 1), 5))
    xs = np.arange(len(keys))
    bar_w = 0.4

    def _stats(vals: list[float]) -> tuple[float, float, int]:
        if not vals:
            return float("nan"), 0.0, 0
        return (float(np.mean(vals)),
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                len(vals))

    for i, key in enumerate(keys):
        var, _ = key
        bm, bs, bn = _stats(base_seed_means.get(key, []))
        am, asd, an = _stats(abl_seed_means.get(key, []))
        ax.bar(xs[i] - bar_w / 2, bm, bar_w, yerr=bs, capsize=4,
               color=var_color[var], edgecolor="black", lw=0.5,
               label="baseline" if i == 0 else None)
        ax.bar(xs[i] + bar_w / 2, am, bar_w, yerr=asd, capsize=4,
               color=var_color[var], edgecolor="black", lw=0.5,
               hatch="//", alpha=0.6,
               label="ablated (is_first=True every step)" if i == 0 else None)
        if not np.isnan(bm) and bn:
            ax.text(xs[i] - bar_w / 2, bm, f"n={bn}",
                    ha="center", va="bottom", fontsize=7)
        if not np.isnan(am) and an:
            ax.text(xs[i] + bar_w / 2, am, f"n={an}",
                    ha="center", va="bottom", fontsize=7)

    ax.set_xticks(xs)
    ax.set_xticklabels([f"{v}\n{m}" for (v, m) in keys],
                       rotation=15, ha="right", fontsize=9)
    ax.set_ylabel(f"eval {metric}  (mean of per-seed means; err = std)")
    ax.set_title(f"Memory ablation: {metric}")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = out_dir / f"eval_{metric}_ablation.png"
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
                      eval_metaseed: int = 42,
                      ablate_memory: bool = False,
                      difficulty: "float | None" = None,
                      spawn_unity: bool = False,
                      record_trajectories: bool = False) -> None:
    """Sequentially shell out to eval_one_run.py for each run in its method's
    venv. Subprocess inherits stdout/stderr so per-episode progress streams
    live. Failures are reported but don't abort the loop — other runs still
    get a shot.

    --ablate-memory is dreamer-only (RSSM amnesia via is_first=True every
    step). Non-dreamer runs are skipped when this flag is on so we don't
    silently produce a non-ablated 'ablation' result.

    `difficulty` (0..1) is forwarded to every eval script and pins an
    adaptive-difficulty def's world at that fixed d, so all runs are compared
    on the same rung of the ladder rather than each at wherever its own
    difficulty walk ended."""
    succeeded, skipped, failed = [], [], []
    for r in runs:
        method = r["method"]
        eval_script = EVAL_SCRIPT_BY_METHOD.get(method)
        if eval_script is None:
            print(f"\n[eval] SKIP {r['run_id']}: no eval script for method "
                  f"'{method}'. Known: {sorted(EVAL_SCRIPT_BY_METHOD)}.")
            skipped.append(r["run_id"])
            continue
        if ablate_memory and method != "dreamer":
            print(f"\n[eval] SKIP {r['run_id']}: --ablate-memory is "
                  f"dreamer-only; method='{method}'.")
            skipped.append(r["run_id"])
            continue
        try:
            python = python_for_method(method)
        except (ValueError, EnvironmentError) as e:
            print(f"\n[eval] SKIP {r['run_id']}: {e}")
            skipped.append(r["run_id"])
            continue
        cmd = [python, str(eval_script),
               "--run_dir", str(r["run_dir"]),
               "--exp_dir", str(exp_dir),
               "--n_episodes", str(n_episodes),
               "--eval_metaseed", str(eval_metaseed)]
        if deterministic:
            cmd.append("--deterministic")
        if difficulty is not None:
            cmd += ["--difficulty", str(difficulty)]
        if ablate_memory:
            cmd.append("--ablate-memory")
        if record_trajectories:
            cmd.append("--record-trajectories")
        if spawn_unity:
            # Without this each eval waits for a Unity on :9000 (see
            # unity_attach.py) — right when you're watching in the Editor,
            # wrong for an unattended batch on a headless box.
            cmd.append("--spawn")
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
    ap.add_argument("--difficulty", type=float, default=None, metavar="D",
                    help="Evaluate every run at a FIXED difficulty d in "
                         "[0, 1] (0 = easiest, 1 = hardest), forwarded to the "
                         "eval scripts. Only for defs with an "
                         "`adaptive_difficulty:` block: its ranges are "
                         "interpolated once at d and merged over the final "
                         "stage's world config, so all runs are compared on "
                         "the same rung instead of each at wherever its own "
                         "difficulty walk ended. The value is stamped into "
                         "every eval JSONL record as 'difficulty'. Only "
                         "meaningful with --run-eval.")
    ap.add_argument("--ablate-memory", action="store_true",
                    dest="ablate_memory",
                    help="Memory-ablation eval: forwards --ablate-memory to "
                         "eval_one_run_dreamer.py (RSSM amnesia via "
                         "is_first=True every step). Writes "
                         "eval_episodes_ablated.jsonl alongside the baseline "
                         "eval_episodes.jsonl, then emits an "
                         "eval_<metric>_ablation.png paired-bar chart. "
                         "Dreamer-only — non-dreamer runs are skipped. Only "
                         "meaningful with --run-eval; without --run-eval the "
                         "ablation plot is still drawn from any cached "
                         "eval_episodes_ablated.jsonl files on disk.")
    ap.add_argument("--record-trajectories", action="store_true",
                    dest="record_trajectories",
                    help="With --run-eval: also record the agent's pose every "
                         "step (one npz per episode next to the eval JSONL) "
                         "for plot_trajectories.py. Keep N small.")
    ap.add_argument("--spawn-unity", action="store_true", dest="spawn_unity",
                    help="Let each eval spawn its own headless Unity build "
                         "(needs $RATSIM_UNITY_BIN) instead of attaching to a "
                         "running one on :9000. Default is to ATTACH and wait, "
                         "so you can watch the policy in the Editor; use this "
                         "for unattended batch eval. Only meaningful with "
                         "--run-eval; automatic under SLURM.")
    args = ap.parse_args()

    if args.run_eval is not None and args.run_eval < 1:
        ap.error("--run-eval N requires N >= 1")
    if args.difficulty is not None and not 0.0 <= args.difficulty <= 1.0:
        ap.error(f"--difficulty must be in [0, 1], got {args.difficulty}")

    exp_dir = resolve_exp_dir(args.exp)
    print(f"Experiment dir: {exp_dir}")

    runs = discover_runs(exp_dir)
    if not runs:
        print("ERROR: no runs found under runs/.")
        sys.exit(1)

    if args.run_eval is not None:
        mode = "deterministic" if args.deterministic else "stochastic"
        ablation_tag = "  [ABLATE MEMORY]" if args.ablate_memory else ""
        difficulty_tag = ("" if args.difficulty is None
                          else f", difficulty={args.difficulty:.3f}")
        print(f"\nRunning {args.run_eval} {mode} eval episode(s) per run "
              f"(eval_metaseed={args.eval_metaseed}{difficulty_tag}, "
              f"sequential, may take a while)"
              f"{ablation_tag}...")
        run_eval_for_runs(runs, exp_dir, args.run_eval,
                          deterministic=args.deterministic,
                          eval_metaseed=args.eval_metaseed,
                          ablate_memory=args.ablate_memory,
                          difficulty=args.difficulty,
                          spawn_unity=args.spawn_unity,
                          record_trajectories=args.record_trajectories)
        # Re-discover so newly written eval_episodes(_ablated).jsonl files are picked up.
        runs = discover_runs(exp_dir)
    else:
        if args.deterministic:
            print("[analyze] WARN: --deterministic only takes effect with "
                  "--run-eval; ignoring.")
        if args.difficulty is not None:
            print("[analyze] WARN: --difficulty only takes effect with "
                  "--run-eval (it changes the worlds an eval is run on, not "
                  "how cached results are plotted); ignoring.")
        if args.record_trajectories:
            print("[analyze] WARN: --record-trajectories only takes effect "
                  "with --run-eval; ignoring.")
        if args.ablate_memory:
            print("[analyze] --ablate-memory without --run-eval: re-plotting "
                  "from any cached eval_episodes_ablated.jsonl files; not "
                  "running new ablated eval.")

    n_train = sum(1 for r in runs if r["train_df"] is not None)
    n_eval = sum(1 for r in runs if r["eval_df"] is not None)
    print(f"\nFound {len(runs)} run(s): {n_train} with train data, "
          f"{n_eval} with eval data.\n")
    print_summary(runs)

    out_dir = Path(args.out) if args.out else (exp_dir / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nPlots -> {out_dir}/")

    baselines = load_external_baselines(exp_dir)
    if baselines:
        print("  external baselines: "
              + ", ".join(baseline_label(b) for b in baselines))
    for m in TRAIN_METRICS:
        plot_training_curve(runs, m, args.rolling, out_dir, baselines)
    for m in OPTIONAL_TRAIN_METRICS:
        plot_training_curve(runs, m, args.rolling, out_dir)

    if n_eval:
        for m in EVAL_METRICS:
            plot_eval_bar(runs, m, out_dir)
    else:
        print("  (no eval_episodes.jsonl found; skipping eval bar charts. "
              "Run with --run-eval N to populate them.)")

    n_ablated = sum(1 for r in runs if r["eval_df_ablated"] is not None)
    if n_ablated:
        for m in EVAL_METRICS:
            plot_eval_ablation_bar(runs, m, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
