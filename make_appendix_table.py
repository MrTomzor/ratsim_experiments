"""Appendix results table: one row per (world x method), columns score /
objects / coverage, so it fits a page where the main table's 4 methods x N
metrics layout does not.

    ~/ratvenv/venv/bin/python make_appendix_table.py
    ~/ratvenv/venv/bin/python make_appendix_table.py --cov-scale 1 --out /tmp/app.tex

Rows, methods, n_eval, eval_metaseed and cell provenance are exactly those of
make_results_table.py (same paper/results_table.yaml, same resolve_*_cell), so
the two tables never disagree on score / objects. `na:` methods are left out
of a world's block instead of printing a dash row.

Coverage = the TaskTracker's lidar-explored area (`explored_area_m2`), the
quantity the volumetric-exploration reward pays for. The RL evals log it;
test.py (human / frontier) does not, so there it is recovered from the score:
with the task presets the paper uses, the score is exactly

    total_score = pickup * objects_found - flat * collisions + per_m2 * area

(flat per-collision penalty, no velocity term, no all-rewards-collected bonus),
so area = (total_score - pickup*objects + flat*collisions) / per_m2. The
coefficients are read from the run's blended task preset; if the preset has a
term the formula doesn't cover, the reconstruction refuses (NaN + note) rather
than guess. Episodes that ended `all_rewards_collected` are refused too. Where
both the logged value and the reconstruction exist they are compared, and any
mismatch > 1 m^2 is printed -- the check that the formula still holds.

Printed in units of --cov-scale m^2 (default 100: with reward_per_m2 = 0.01
that is the coverage part of the score).

Aggregation per cell is the main table's: mean / std over episodes, all
seeds pooled, so every +- is the spread across eval worlds.

Only reads results; writes the .tex (default: results_appendix.tex next to the
main table's `out:`) and a .status.md beside it.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from ratsim.config_blender import blend_presets

import make_results_table as mrt
from analyze_experiment import load_jsonl
from experiment_defs import resolve_task_preset
from record_trajectories import EXTERNAL_METHODS

COV = "coverage_m2"
METRICS = ["total_score", "objects_found"]


# ─────────────────────────────────────────────
#  Coverage: logged, else recovered from the score
# ─────────────────────────────────────────────

def score_coefficients(exp_dir: Path, exp: str, variation: str) -> tuple[dict | None, str]:
    """(pickup, flat, per_m2) of the run's task preset, or None + why the score
    isn't a clean pickup/collision/area sum under it."""
    exp_def = mrt.load_def_for(exp_dir, exp)
    if exp_def is None:
        return None, "no def to read the task preset from"
    var = next((v for v in exp_def.variations if v.name == variation), None)
    if var is None:
        return None, f"variation {variation} not in def"
    t = blend_presets("task", resolve_task_preset(exp_def, var))
    fs, cs = t.get("foraging_settings", {}), t.get("collision_settings", {})
    ts, vs = t.get("termination_settings", {}), t.get("volumetric_exploration_settings", {})
    per_m2 = float(vs.get("reward_per_m2", 0.0))
    if per_m2 == 0:
        return None, "no volumetric reward in the task preset"
    if not cs.get("penalize_collisions", True):
        flat = 0.0
    elif (cs.get("penalization_variable") == "velocity"
          and float(cs.get("collision_penalty_modifier", 0.3)) != 0):
        return None, "velocity-scaled collision penalty"
    else:
        flat = float(cs.get("collision_flat_penalty", 0.0))
        if cs.get("penalization_variable") != "velocity":
            flat += float(cs.get("collision_penalty_modifier", 0.3))
    for k in ("collision_termination_reward", "zero_battery_termination_reward",
              "zero_health_termination_reward"):
        if float(ts.get(k, 0.0)) != 0:
            return None, f"{k} != 0"
    if float(fs.get("wrong_well_penalty_fraction", 0.0)) != 0:
        return None, "wrong-well penalty"
    return {"pickup": float(fs.get("reward_object_pickup_modifier", 1.0)),
            "flat": flat, "per_m2": per_m2}, ""


def add_coverage(df: pd.DataFrame | None, coef: dict | None, where: str) -> None:
    if df is None or COV in df.columns:
        return
    logged = (df["explored_area_m2"].astype(float) if "explored_area_m2" in df.columns
              else pd.Series(np.nan, index=df.index))
    rec = pd.Series(np.nan, index=df.index)
    if coef is not None and {"total_score", "objects_found", "collisions"} <= set(df.columns):
        rec = ((df["total_score"] - coef["pickup"] * df["objects_found"]
                + coef["flat"] * df["collisions"]) / coef["per_m2"])
        if "termination_reason" in df.columns:
            rec[df["termination_reason"] == "all_rewards_collected"] = np.nan
    both = logged.notna() & rec.notna()
    bad = (logged[both] - rec[both]).abs() > 1.0
    if bad.any():
        print(f"  WARN: {where}: score reconstruction off from logged explored_area_m2 on "
              f"{int(bad.sum())}/{int(both.sum())} episodes (max "
              f"{(logged[both] - rec[both]).abs().max():.0f} m^2) -- formula doesn't hold")
    df[COV] = logged.fillna(rec)


_AUGMENTED: set[tuple] = set()


def augment_runs(exp_dir: Path, spec: dict) -> None:
    """Give every cached run's eval_df / train_df a coverage column, so
    resolve_rl_cell aggregates it with the same seed selection as the rest."""
    key = (exp_dir, spec["variation"])
    if key in _AUGMENTED:
        return
    _AUGMENTED.add(key)
    try:
        runs = mrt.runs_for(exp_dir)
    except FileNotFoundError:
        return
    coef, _ = score_coefficients(exp_dir, spec["exp"], spec["variation"])
    for r in runs:
        if r["variation"] != spec["variation"]:
            continue
        add_coverage(r["eval_df"], coef, f"{r['run_id']} eval")
        add_coverage(r["train_df"], coef, f"{r['run_id']} train")


def nan_agg(samples: list[float]) -> tuple[tuple[float, float] | None, str]:
    ok = [s for s in samples if not np.isnan(s)]
    if not ok:
        return None, "no coverage" if samples else ""
    note = f"coverage on {len(ok)}/{len(samples)} episodes" if len(ok) < len(samples) else ""
    return mrt.agg(ok), note


# ─────────────────────────────────────────────
#  Cells
# ─────────────────────────────────────────────

def resolve_cell(spec: dict, method: str, cfg: dict) -> dict:
    if method in EXTERNAL_METHODS:
        c = mrt.resolve_external_cell(spec, method, cfg, METRICS)
        samples, why = [], ""
        if c["status"] in ("eval", "partial"):
            mdir = c["exp_dir"] / "external" / method
            df = mrt.first_n(load_jsonl(mdir / "episodes.jsonl"), cfg["n_eval"]).copy()
            coef, why = score_coefficients(c["exp_dir"], spec["exp"], spec["variation"])
            add_coverage(df, coef, f"{spec['exp']}/external/{method}")
            samples = df[COV].astype(float).tolist()
    else:
        exp_dir = mrt.find_exp_dir(spec["exp"], spec["source"])
        why = ""
        if exp_dir is not None:
            augment_runs(exp_dir, spec)
        c = mrt.resolve_rl_cell(spec, method, cfg, METRICS + [COV])
        # episodes without coverage are NaN; nan_agg drops them
        samples = c["samples"].pop(COV, [])
        c["values"].pop(COV, None)
    val, note = nan_agg(samples)
    if val is not None:
        c["values"][COV] = val
    note = "; ".join(n for n in (note, why if val is None else "") if n)
    if note and c["status"] not in ("missing", "na"):
        c["note"] = "; ".join(n for n in (c.get("note"), note) if n)
    return c


# ─────────────────────────────────────────────
#  Output
# ─────────────────────────────────────────────

def render_tex(cfg: dict, groups: list[dict], methods: list[str], cells: dict,
               cov_scale: float, cov_dec: int) -> str:
    cols = METRICS + [COV]
    dec = {m: int(cfg["decimals"].get(m, 1)) for m in METRICS}
    dec[COV] = cov_dec
    L = [
        "% AUTO-GENERATED by ratsim_experiments/make_appendix_table.py -- do not edit;",
        f"% n_eval={cfg['n_eval']}, train_window={cfg['train_window']}, coverage = lidar-explored "
        f"area (logged, else recovered from score - pickups + collisions), units of {cov_scale:g} m^2.",
        "% Same provenance colours / \\providecommand overrides as make_results_table.py.",
        "\\providecommand{\\cellEval}[1]{#1}",
        "\\providecommand{\\cellTrain}[1]{{\\color{blue}#1}}",
        "\\providecommand{\\cellPartial}[1]{{\\color{orange}#1}}",
        "\\providecommand{\\cellMissing}{\\textcolor{red}{--}}",
        "\\providecommand{\\cellNA}{--}",
        "\\begin{tabular}{ll lll}",
        "\\toprule",
    ]
    unit = "m$^2$" if cov_scale == 1 else f"$10^{{{int(round(np.log10(cov_scale)))}}}$\\,m$^2$"
    L.append("\\textbf{World} & \\textbf{Method} & "
             f"{cfg['metrics'].get('total_score', 'Score')} & "
             f"{cfg['metrics'].get('objects_found', 'OBJ')} & "
             f"Coverage [{unit}] \\\\")
    for g in groups:
        L.append("\\midrule")
        for row in g["rows"]:
            ms = [m for m in methods if cells[(row["label"], m)]["status"] != "na"]
            best = {}
            for met in cols:
                if met not in cfg["bold_best"]:
                    continue
                vals = [round(cells[(row["label"], m)]["values"][met][0] / (cov_scale if met == COV else 1), dec[met])
                        for m in ms if met in cells[(row["label"], m)].get("values", {})]
                best[met] = max(vals) if vals else None
            if row is not g["rows"][0]:
                L.append("\\cmidrule(lr){1-5}")
            for i, m in enumerate(ms):
                c = cells[(row["label"], m)]
                world = (f"\\multirow{{{len(ms)}}}{{*}}{{{row['label']}}}" if i == 0 else "")
                parts = [world, cfg["methods"][m]]
                for met in cols:
                    if c["status"] == "missing":
                        parts.append("\\cellMissing")
                    elif met not in c["values"]:
                        parts.append("\\cellNA")
                    else:
                        mean, std = c["values"][met]
                        if met == COV:
                            mean, std = mean / cov_scale, std / cov_scale
                        bold = best.get(met) is not None and round(mean, dec[met]) == best[met]
                        parts.append(mrt.fmt_value(mean, std, dec[met],
                                                   mrt.STATUS_MACRO[c["status"]], bold))
                L.append(" & ".join(parts) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    return "\n".join(L) + "\n"


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(mrt.DEFAULT_CONFIG))
    ap.add_argument("--n", type=int, default=None, help="eval episodes per cell (n_eval)")
    ap.add_argument("--rows", default=None, help="comma-separated row labels to keep")
    ap.add_argument("--methods", default=None, help="comma-separated methods to keep")
    ap.add_argument("--out", default=None,
                    help="output .tex (default: results_appendix.tex next to the config's `out:`)")
    ap.add_argument("--cov-scale", type=float, default=100.0, dest="cov_scale",
                    help="coverage printed in units of this many m^2 (1 = plain m^2)")
    ap.add_argument("--cov-decimals", type=int, default=1, dest="cov_decimals")
    args = ap.parse_args()

    cfg = mrt.load_config(Path(args.config))
    if args.n is not None:
        cfg["n_eval"] = args.n

    methods = list(cfg["methods"])
    if args.methods:
        keep = [m.strip() for m in args.methods.split(",") if m.strip()]
        unknown = [m for m in keep if m not in methods]
        if unknown:
            sys.exit(f"ERROR: --methods {unknown} not in config methods {methods}")
        methods = keep
    if "human" in methods:
        methods = [m for m in methods if m != "human"] + ["human"]

    want = None if not args.rows else {r.strip() for r in args.rows.split(",")}
    groups = []
    for g in cfg["groups"]:
        rows = [r for r in g.get("rows", []) if want is None or r["label"] in want]
        if rows:
            groups.append({"name": g.get("name", ""), "rows": rows})
    if want and want - {r["label"] for g in groups for r in g["rows"]}:
        sys.exit(f"ERROR: --rows {sorted(want - {r['label'] for g in groups for r in g['rows']})} "
                 f"not in config")

    if args.out:
        out_path = Path(os.path.expanduser(args.out))
    elif cfg.get("out"):
        out_path = Path(os.path.expanduser(cfg["out"])).with_name("results_appendix.tex")
    else:
        out_path = mrt.HERE / "results" / "analysis" / "results_appendix.tex"

    cells = {}
    for g in groups:
        for row in g["rows"]:
            na = set(row.get("na") or [])
            for m in methods:
                spec = mrt.cell_spec(row, m)
                if m in na:
                    cells[(row["label"], m)] = {"status": "na", "values": {}}
                elif not spec.get("exp"):
                    cells[(row["label"], m)] = {"status": "missing", "values": {},
                                                "note": "row has no `exp`"}
                else:
                    cells[(row["label"], m)] = resolve_cell(spec, m, cfg)

    status = mrt.status_frame(groups, methods, cells)
    status.insert(4, "coverage", [
        f"{cells[(r, m)]['values'][COV][0]:.0f}" if COV in cells[(r, m)].get("values", {}) else "-"
        for r, m in zip(status["row"], status["method"])])
    print(f"\n[appendix_table] n_eval={cfg['n_eval']} coverage [m^2] = explored_area_m2 "
          f"(logged, else recovered from the score)")
    with pd.option_context("display.max_rows", None, "display.width", 250,
                           "display.max_colwidth", 90):
        print(status.to_string(index=False))

    tex = render_tex(cfg, groups, methods, cells, args.cov_scale, args.cov_decimals)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(tex)
    md_path = out_path.with_suffix(".status.md")
    md_path.write_text(f"# appendix table status (n_eval={cfg['n_eval']}, coverage = explored_area_m2)"
                       f"\n\n" + mrt.to_markdown(status))
    print(f"[appendix_table] -> {out_path}\n[appendix_table] -> {md_path}")


if __name__ == "__main__":
    main()
