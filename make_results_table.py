"""Generate the paper's results table from whatever data is on disk.

    ~/ratvenv/venv/bin/python make_results_table.py
    ~/ratvenv/venv/bin/python make_results_table.py --n 10 --rows 3buildings,5houses-sar
    ~/ratvenv/venv/bin/python make_results_table.py --methods ppo,dreamer --out /tmp/t.tex
    ~/ratvenv/venv/bin/python make_results_table.py --strict      # fail unless all black
    ~/ratvenv/venv/bin/python make_results_table.py --only-score  # no OBJ columns

Rows, paper names, method columns and the "fully baked" threshold come from
paper/results_table.yaml (see the comments there); CLI flags override it.
The human column is always moved last, whatever its place in the config (the
config order is shared with the figure scripts).
Each cell resolves to one of:

  eval     black   at least one training seed has >= n_eval episodes in its
                   eval_episodes.jsonl (eval_one_run.py output, held-out
                   worlds from --eval_metaseed), at the pinned difficulty;
                   seeds with fewer are left out (listed in the note).
                   For human / frontier: >= n_eval episodes under
                   <exp_dir>/external/<method>/episodes.jsonl.
  train    blue    fallback: last `train_window` episodes of each run's
                   train_episodes.jsonl (pull_run.sh -t). Training worlds,
                   not held-out ones.
  partial  orange  some eval episodes but fewer than n_eval, and nothing to
                   fall back on (external methods, or RL with no train data).
  missing  red     nothing on disk.
  n/a      dash    method doesn't apply to the row (`na:` in the config).

Only the FIRST n_eval eval episodes (by episode_idx) of each run are used, so
every cell is scored on the same n_eval worlds even where one run was
evaluated on more.

Mean and std are over EPISODES, all qualifying seeds pooled (RL: n_seeds x
n_eval episodes; human / frontier: their n_eval). So every +- is the same
thing -- spread across eval worlds (and, for RL, runs) -- and an RL cell is
comparable with the human / frontier cells, which have no seeds to take a
spread over. With n_eval episodes per seed the pooled mean equals the mean of
per-seed means. Training-run variability alone is NOT what the +- shows.

Writes the tabular-only .tex (the paper keeps its own table env + caption and
pulls it in with \\input), a status markdown next to it, and prints the same
status to the console: seeds present / expected, training steps done / target,
eval episodes per seed, so it is obvious what still needs baking.

The mean is always typeset black; only the \\pm std subscript is coloured, so a
cell's provenance is visible without making the number itself hard to read.
In each row, the best (highest) mean of every metric listed under `bold_best:`
in the config (default: total_score) is set in bold -- compared at the printed
precision, so a tie bolds every tied cell. Every cell with a value competes,
whatever its colour.

The .tex defines \\cellEval, \\cellTrain, \\cellPartial, \\cellMissing, \\cellNA
with \\providecommand (the first three wrap the subscript, in math mode) --
define them in the paper BEFORE the \\input to override (e.g. all black for
camera-ready).

Gotcha: eval_one_run.py OVERWRITES eval_episodes.jsonl, and
record_trajectories.py calls it with --n_episodes = --n-worldseeds (default
1). Recording a trajectory figure after a 5-episode eval silently shrinks that
run's eval to 1 episode; the status column shows it.

Needs pandas + pyyaml (sb3 venv).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from analyze_experiment import discover_runs, load_jsonl  # noqa: E402
from experiment_defs import load_experiment_def  # noqa: E402
from record_trajectories import EXTERNAL_METHODS, RESULTS_ROOTS  # noqa: E402

HERE = Path(__file__).parent
DEFAULT_CONFIG = HERE / "paper" / "results_table.yaml"
DEFS_DIR = HERE / "defs"

_RUNS_CACHE: dict[Path, list[dict]] = {}


def runs_for(exp_dir: Path) -> list[dict]:
    """discover_runs, once per experiment dir (every method column of a row
    shares the same dir, and the JSONL loads are the slow part)."""
    if exp_dir not in _RUNS_CACHE:
        _RUNS_CACHE[exp_dir] = discover_runs(exp_dir)
    return _RUNS_CACHE[exp_dir]


STATUS_MACRO = {
    "eval": "\\cellEval",
    "train": "\\cellTrain",
    "partial": "\\cellPartial",
}


# ─────────────────────────────────────────────
#  Config
# ─────────────────────────────────────────────

def load_config(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    for key in ("metrics", "methods", "groups"):
        if key not in cfg:
            sys.exit(f"ERROR: {path}: missing `{key}:`")
    cfg.setdefault("n_eval", 5)
    cfg.setdefault("train_window", 100)
    cfg.setdefault("eval_metaseed", 42)
    cfg.setdefault("decimals", {})
    cfg.setdefault("bold_best", ["total_score"])
    return cfg


def cell_spec(row: dict, method: str) -> dict:
    """Row-level keys merged with the row's per_method override for `method`."""
    spec = {
        "exp": row.get("exp"),
        "variation": row.get("variation", "baseline"),
        "difficulty": row.get("difficulty"),
        "source": row.get("source", "rci"),
        "eval_metaseed": row.get("eval_metaseed"),
    }
    spec.update((row.get("per_method") or {}).get(method) or {})
    return spec


def find_exp_dir(exp: str, source: str) -> Path | None:
    """Like record_trajectories.resolve_exp_dir, but a missing dir is a
    table cell state (red), not a fatal error."""
    p = Path(os.path.expanduser(exp))
    if p.is_dir():
        return p.resolve()
    root = RESULTS_ROOTS.get(source)
    if root is None:
        sys.exit(f"ERROR: unknown source '{source}' (use {list(RESULTS_ROOTS)})")
    cand = root / exp
    return cand.resolve() if cand.is_dir() else None


def load_def_for(exp_dir: Path, exp: str):
    """The def that actually ran (exp_dir/experiment.yaml), else defs/<exp>.yaml."""
    for cand in (exp_dir / "experiment.yaml", DEFS_DIR / f"{exp}.yaml"):
        if cand.exists():
            try:
                return load_experiment_def(cand)
            except Exception as e:  # noqa: BLE001 — a broken def shouldn't kill the table
                print(f"  WARN: could not parse {cand}: {e}")
    return None


# ─────────────────────────────────────────────
#  Per-run facts
# ─────────────────────────────────────────────

def stage_steps(exp_def) -> list[int]:
    return [int(s.steps) for s in exp_def.stages] if exp_def else []


def steps_done(run_dir: Path, stages: list[int]) -> int | None:
    """Env steps completed = sum of the stages that have a stage_K.done marker.
    run_config.json's cumulative_steps is written at stage START, so it
    overstates an in-progress run by one stage; the markers don't."""
    ck = run_dir / "checkpoints"
    if ck.is_dir() and stages:
        done = []
        for p in ck.glob("stage_*.done"):
            k = p.name[len("stage_"):-len(".done")]
            if k.isdigit() and int(k) < len(stages):
                done.append(int(k))
        if done:
            return sum(stages[k] for k in done)
    rc = run_dir / "run_config.json"
    if rc.exists():
        try:
            return int(json.loads(rc.read_text()).get("cumulative_steps") or 0)
        except ValueError:
            pass
    return None


def eval_provenance(run_dir: Path) -> tuple[int | None, float | None]:
    """(eval_metaseed, difficulty) the run's eval was produced with."""
    p = run_dir / "eval_world_config.json"
    if not p.exists():
        return None, None
    try:
        d = json.loads(p.read_text())
    except ValueError:
        return None, None
    return d.get("eval_metaseed"), d.get("difficulty")


def same_difficulty(a, b) -> bool:
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False
    return math.isclose(float(a), float(b), abs_tol=1e-6)


def first_n(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if "episode_idx" in df.columns:
        df = df.sort_values("episode_idx", kind="stable")
    return df.head(n)


def pooled_episodes(dfs: list[pd.DataFrame], metric: str) -> list[float]:
    """Every episode's value, all seeds together (one sample per world x seed)."""
    return [float(v) for df in dfs for v in df[metric]]


def fmt_steps(v: int | None) -> str:
    if v is None:
        return "?"
    return f"{v / 1e6:.1f}M" if v >= 1e5 else str(v)


# ─────────────────────────────────────────────
#  Cell resolution
# ─────────────────────────────────────────────

def agg(values: list[float]) -> tuple[float, float]:
    """(mean, std) of the per-episode samples (all seeds pooled); the cell
    keeps the samples too, under `samples`, for make_final_boxplots.py."""
    arr = np.asarray(values, dtype=float)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return float(arr.mean()), std


def resolve_rl_cell(spec: dict, method: str, cfg: dict, metrics: list[str],
                    eval_only: bool = False) -> dict:
    """Samples are the individual episodes pooled over the seeds (eval: first
    n_eval of every seed with a complete, provenance-checked eval; train: last
    train_window of every run) -- spread across worlds, not seeds. Status `eval`
    as soon as ONE seed has a complete eval; seeds without one are left out and
    named in the note. eval_only (make_final_boxplots.py): never fall back to
    training data."""
    n_eval, window = cfg["n_eval"], cfg["train_window"]
    out = {"status": "missing", "values": {}, "samples": {}, "seeds": "0/?", "steps": "-",
           "eval_eps": "-", "note": ""}
    exp_dir = find_exp_dir(spec["exp"], spec["source"])
    if exp_dir is None:
        out["note"] = f"no results dir for {spec['exp']} (pull_run.sh -t {spec['exp']}?)"
        return out
    out["exp_dir"] = exp_dir
    exp_def = load_def_for(exp_dir, spec["exp"])
    stages = stage_steps(exp_def)
    target = sum(stages) if stages else None
    expected = None
    if exp_def:
        for m in exp_def.methods:
            if m.name == method:
                expected = m.n_seeds

    try:
        runs = runs_for(exp_dir)
    except FileNotFoundError as e:
        out["note"] = str(e)
        return out
    runs = [r for r in runs if r["method"] == method and r["variation"] == spec["variation"]]
    if not runs:
        out["seeds"] = f"0/{expected if expected is not None else '?'}"
        out["note"] = f"no runs for {spec['variation']}__{method} under {exp_dir.name}/runs"
        return out
    if expected is None:
        expected = len(runs)
    out["seeds"] = f"{len(runs)}/{expected}"

    done = [steps_done(r["run_dir"], stages) for r in runs]
    known = [d for d in done if d is not None]
    if known:
        lo, hi = min(known), max(known)
        span = fmt_steps(lo) if lo == hi else f"{fmt_steps(lo)}..{fmt_steps(hi)}"
        out["steps"] = f"{span}/{fmt_steps(target)}"

    # -- eval: first n_eval episodes per run, provenance checked --
    eval_ok, eval_counts, notes = [], [], []
    want_ms = (spec["eval_metaseed"] if spec.get("eval_metaseed") is not None
               else cfg["eval_metaseed"])
    for r in runs:
        ed = r["eval_df"]
        n = 0 if ed is None else len(ed)
        eval_counts.append(n)
        if ed is None:
            continue
        ms, d = eval_provenance(r["run_dir"])
        if ms is not None and int(ms) != int(want_ms):
            notes.append(f"seed{r['seed']}: eval_metaseed {ms} != {want_ms}")
            continue
        if not same_difficulty(d, spec["difficulty"]):
            notes.append(f"seed{r['seed']}: eval at d={d}, table wants d={spec['difficulty']}")
            continue
        if n < n_eval:
            if (r["run_dir"] / "eval_episodes_trajectories").is_dir():
                notes.append(f"seed{r['seed']}: {n}<{n_eval} eval eps (record_trajectories overwrote it?)")
            continue
        eval_ok.append(first_n(ed, n_eval))
    out["eval_eps"] = ",".join(str(c) for c in eval_counts)

    if eval_ok:
        out["status"] = "eval"
        for m in metrics:
            out["samples"][m] = pooled_episodes(eval_ok, m)
            out["values"][m] = agg(out["samples"][m])
        if len(eval_ok) < expected:
            incomplete = [f"seed{r['seed']}({c})" for r, c in zip(runs, eval_counts) if c < n_eval]
            if incomplete:
                notes.insert(0, f"left out, <{n_eval} eval eps: {', '.join(incomplete)}")
            notes.insert(0, f"eval on {len(eval_ok)}/{expected} seeds")
        out["note"] = "; ".join(notes)
        return out

    if eval_only:
        if any(eval_counts):
            notes.insert(0, f"no seed with >= {n_eval} eval eps ({', '.join(map(str, eval_counts))})")
        out["note"] = "; ".join(notes)
        return out

    # -- train fallback: tail of each run --
    train = [r["train_df"] for r in runs if r["train_df"] is not None]
    if train:
        out["status"] = "train"
        for m in metrics:
            out["samples"][m] = pooled_episodes([df.tail(window) for df in train], m)
            out["values"][m] = agg(out["samples"][m])
        if any(("difficulty" in df.columns) for df in train):
            ds = [float(df.tail(window)["difficulty"].mean()) for df in train
                  if "difficulty" in df.columns]
            notes.append(f"train tail at mean d={np.mean(ds):.2f}")
        out["note"] = "; ".join(notes)
        return out

    if any(eval_counts):
        out["status"] = "partial"
        pool = [first_n(r["eval_df"], n_eval) for r in runs if r["eval_df"] is not None]
        for m in metrics:
            out["samples"][m] = pooled_episodes(pool, m)
            out["values"][m] = agg(out["samples"][m])
        notes.insert(0, f"eval < {n_eval} eps on every seed, no train data")
    else:
        notes.insert(0, "no train_episodes.jsonl / eval_episodes.jsonl in any run")
    out["note"] = "; ".join(notes)
    return out


def resolve_external_cell(spec: dict, method: str, cfg: dict, metrics: list[str]) -> dict:
    n_eval = cfg["n_eval"]
    out = {"status": "missing", "values": {}, "samples": {}, "seeds": "-", "steps": "-",
           "eval_eps": "0", "note": ""}
    exp_dir = find_exp_dir(spec["exp"], spec["source"])
    if exp_dir is None:
        out["note"] = f"no results dir for {spec['exp']}"
        return out
    out["exp_dir"] = exp_dir
    mdir = exp_dir / "external" / method
    df = load_jsonl(mdir / "episodes.jsonl")
    if df is None:
        out["note"] = (f"no external/{method}/episodes.jsonl "
                       f"(record_trajectories.py {spec['exp']} --methods {method} "
                       f"--n-worldseeds {n_eval})")
        return out
    # difficulty provenance: eval_config.json, or the per-record field
    d = None
    cfgp = mdir / "eval_config.json"
    if cfgp.exists():
        try:
            d = json.loads(cfgp.read_text()).get("difficulty")
        except ValueError:
            pass
    if "difficulty" in df.columns and df["difficulty"].notna().any():
        d = float(df["difficulty"].dropna().iloc[-1])
    if not same_difficulty(d, spec["difficulty"]):
        out["note"] = f"external eval at d={d}, table wants d={spec['difficulty']}"
        out["eval_eps"] = str(len(df))
        return out
    out["eval_eps"] = str(len(df))
    use = first_n(df, n_eval)
    for m in metrics:
        out["samples"][m] = use[m].astype(float).tolist()
        out["values"][m] = agg(out["samples"][m])
    if len(df) >= n_eval:
        out["status"] = "eval"
    else:
        out["status"] = "partial"
        out["note"] = f"{len(df)}<{n_eval} episodes"
    return out


# ─────────────────────────────────────────────
#  Output
# ─────────────────────────────────────────────

def fmt_value(mean: float, std: float, decimals: int, macro: str, bold: bool = False) -> str:
    """Mean is always plain (black); only the +-std subscript carries the provenance colour."""
    num = f"{mean:.{decimals}f}"
    if bold:
        num = f"\\mathbf{{{num}}}"
    return (f"${num}_{{{macro}{{\\pm {std:.{decimals}f}}}}}$")


def best_in_row(row_label: str, methods: list[str], met: str, decimals: int,
                cells: dict) -> float | None:
    """Highest mean of `met` across the row's methods, rounded as printed."""
    vals = [round(cells[(row_label, m)]["values"][met][0], decimals) for m in methods
            if met in cells[(row_label, m)].get("values", {})]
    return max(vals) if vals else None


def render_tex(cfg: dict, groups: list[dict], methods: list[str], metrics: list[str],
               cells: dict) -> str:
    L = []
    L.append("% AUTO-GENERATED by ratsim_experiments/make_results_table.py -- do not edit;")
    L.append(f"% regenerate with: python make_results_table.py  (n_eval={cfg['n_eval']}, "
             f"train_window={cfg['train_window']})")
    L.append("% Means are always black; the \\pm std subscript carries the provenance")
    L.append("% colour: black = held-out eval, blue = training-episode fallback,")
    L.append("% orange = partial eval, red = no data. \\newcommand these BEFORE the")
    L.append("% \\input to override (e.g. all black for camera-ready).")
    L.append("\\providecommand{\\cellEval}[1]{#1}")
    L.append("\\providecommand{\\cellTrain}[1]{{\\color{blue}#1}}")
    L.append("\\providecommand{\\cellPartial}[1]{{\\color{orange}#1}}")
    L.append("\\providecommand{\\cellMissing}{\\textcolor{red}{--}}")
    L.append("\\providecommand{\\cellNA}{--}")
    ncol = len(methods) * len(metrics)
    # numbers left-aligned; a vertical rule separates each method's block of metrics
    L.append("\\begin{tabular}{l " + " | ".join("l" * len(metrics) for _ in methods) + "}")
    L.append("\\toprule")
    # one metric per method (--only-score): the method name is the whole header
    two_rows = len(metrics) > 1
    head = ["\\multirow{2}{*}{\\textbf{World}}" if two_rows else "\\textbf{World}"]
    for m in methods:
        # multicolumn overrides the column spec, so carry the "|" here too
        bar = "" if m == methods[-1] else "|"
        head.append(f"\\multicolumn{{{len(metrics)}}}{{c{bar}}}"
                    f"{{\\textbf{{{cfg['methods'][m]}}}}}")
    L.append(" & ".join(head) + " \\\\")
    if two_rows:
        rules = []
        for i in range(len(methods)):
            a = 2 + i * len(metrics)
            rules.append(f"\\cmidrule(lr){{{a}-{a + len(metrics) - 1}}}")
        L.append(" ".join(rules))
        L.append(" & " + " & ".join(cfg["metrics"][met] for _ in methods for met in metrics) + " \\\\")
    L.append("\\midrule")
    for gi, g in enumerate(groups):
        if gi:
            L.append("\\midrule")
        for row in g["rows"]:
            parts = [row["label"]]
            best = {met: best_in_row(row["label"], methods, met,
                                     int(cfg["decimals"].get(met, 1)), cells)
                    for met in metrics if met in cfg["bold_best"]}
            for m in methods:
                c = cells[(row["label"], m)]
                for met in metrics:
                    if c["status"] == "na":
                        parts.append("\\cellNA")
                    elif c["status"] == "missing":
                        parts.append("\\cellMissing")
                    else:
                        mean, std = c["values"][met]
                        dec = int(cfg["decimals"].get(met, 1))
                        bold = best.get(met) is not None and round(mean, dec) == best[met]
                        parts.append(fmt_value(mean, std, dec, STATUS_MACRO[c["status"]], bold))
            L.append(" & ".join(parts) + " \\\\")
    L.append("\\bottomrule")
    L.append("\\end{tabular}")
    assert ncol == len(methods) * len(metrics)
    return "\n".join(L) + "\n"


def status_frame(groups: list[dict], methods: list[str], cells: dict) -> pd.DataFrame:
    rows = []
    for g in groups:
        for row in g["rows"]:
            for m in methods:
                c = cells[(row["label"], m)]
                spec = cell_spec(row, m)
                rows.append({
                    "row": row["label"],
                    "method": m,
                    "exp": spec["exp"] if c["status"] != "na" else "-",
                    "status": c["status"],
                    "seeds": c.get("seeds", "-"),
                    "steps": c.get("steps", "-"),
                    "eval_eps": c.get("eval_eps", "-"),
                    "note": c.get("note", ""),
                })
    return pd.DataFrame(rows)


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "\\|") for c in cols) + " |")
    return "\n".join(lines) + "\n"


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--n", type=int, default=None, help="eval episodes per cell (n_eval)")
    ap.add_argument("--train-window", type=int, default=None, dest="train_window",
                    help="training episodes per run in the blue fallback")
    ap.add_argument("--rows", default=None, help="comma-separated row labels to keep")
    ap.add_argument("--methods", default=None, help="comma-separated method columns to keep")
    ap.add_argument("--out", default=None, help="output .tex (default: `out:` in the config, "
                                                "else results/analysis/results_table.tex)")
    ap.add_argument("--strict", action="store_true",
                    help="exit 1 unless every cell is eval (black) or n/a")
    ap.add_argument("--only-score", action="store_true", dest="only_score",
                    help="only the total_score column per method (drop objects_found)")
    args = ap.parse_args()

    cfg = load_config(Path(args.config))
    if args.n is not None:
        cfg["n_eval"] = args.n
    if args.train_window is not None:
        cfg["train_window"] = args.train_window

    methods = list(cfg["methods"])
    if args.methods:
        keep = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = [m for m in keep if m not in methods]
        if unknown:
            sys.exit(f"ERROR: --methods {unknown} not in config methods {methods}")
        methods = keep
    if "human" in methods:  # human is the reference, so it goes last
        methods = [m for m in methods if m != "human"] + ["human"]
    metrics = list(cfg["metrics"])
    if args.only_score:
        metrics = [m for m in metrics if m == "total_score"]

    groups = []
    want = None if not args.rows else {r.strip() for r in args.rows.split(",")}
    for g in cfg["groups"]:
        rows = [r for r in g.get("rows", []) if want is None or r["label"] in want]
        if rows:
            groups.append({"name": g.get("name", ""), "rows": rows})
    if want:
        seen = {r["label"] for g in groups for r in g["rows"]}
        if want - seen:
            sys.exit(f"ERROR: --rows {sorted(want - seen)} not in config")

    out_path = Path(os.path.expanduser(args.out or cfg.get("out")
                                       or str(HERE / "results" / "analysis" / "results_table.tex")))

    cells = {}
    for g in groups:
        for row in g["rows"]:
            na = set(row.get("na") or [])
            for m in methods:
                if m in na:
                    cells[(row["label"], m)] = {"status": "na", "values": {}}
                    continue
                spec = cell_spec(row, m)
                if not spec.get("exp"):
                    cells[(row["label"], m)] = {"status": "missing", "values": {},
                                                "note": "row has no `exp`"}
                    continue
                if m in EXTERNAL_METHODS:
                    cells[(row["label"], m)] = resolve_external_cell(spec, m, cfg, metrics)
                else:
                    cells[(row["label"], m)] = resolve_rl_cell(spec, m, cfg, metrics)

    status = status_frame(groups, methods, cells)
    print(f"\n[results_table] n_eval={cfg['n_eval']} train_window={cfg['train_window']} "
          f"eval_metaseed={cfg['eval_metaseed']}")
    with pd.option_context("display.max_rows", None, "display.width", 250,
                           "display.max_colwidth", 90):
        print(status.to_string(index=False))
    counts = status["status"].value_counts().to_dict()
    print(f"\n[results_table] cells: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    tex = render_tex(cfg, groups, methods, metrics, cells)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex)
    md_path = out_path.with_suffix(".status.md")
    md_path.write_text(f"# results table status (n_eval={cfg['n_eval']}, "
                       f"train_window={cfg['train_window']})\n\n" + to_markdown(status))
    print(f"[results_table] -> {out_path}\n[results_table] -> {md_path}")

    if args.strict:
        bad = status[~status["status"].isin(["eval", "na"])]
        if len(bad):
            print(f"[results_table] STRICT: {len(bad)} cell(s) not fully baked", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
