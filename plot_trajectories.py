"""Trajectory figure: rows = experiments (one world config each), columns = methods.

Reads the manifests record_trajectories.py writes (`<exp_dir>/trajectories.json`)
and draws each trajectory on a blank top-down slate with
`ratsim.ratsim_vis.trajectory_plot`, or with `--background <view>` on the
rendered world snapshot of that view (ortho / persp, both straight down; taken by
snapshot_worlds.py or record_trajectories.py --snapshot, indexed in the
manifest under `snapshots`).

    python plot_trajectories.py memory_3malls memory_orthomaze
    python plot_trajectories.py memory_3malls memory_orthomaze --methods ppo,dreamer,frontier,human
    python plot_trajectories.py memory_3malls memory_orthomaze --overlay --colour-by-time
    python plot_trajectories.py memory_3malls --out results/analysis/paper --name fig3
    python plot_trajectories.py memory_3malls memory_orthomaze --show     # open the PNG when done
    python plot_trajectories.py memory_3malls --background ortho            # on the rendered world

Rows follow the experiment order given; an experiment recorded with
--n-worldseeds N contributes N rows. Columns are the methods in --methods
order (default: every method any manifest has, in first-seen order). A panel
without data says so instead of failing.

Output: <out>/trajectories_<layout>[_<view>][_<name>].png, default out =
results/analysis/. Needs matplotlib (sb3 venv).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ratsim.ratsim_vis.trajectory_plot import default_colors, plot_trajectories  # noqa: E402
from ratsim.task_tracker.trajectory_record import load_trajectory  # noqa: E402

from ratsim.world_snapshot import load_snapshot  # noqa: E402
from record_trajectories import MANIFEST_NAME, resolve_exp_dir  # noqa: E402


# ─────────────────────────────────────────────
#  Rows
# ─────────────────────────────────────────────

def load_rows(exps: list[str], source: str = "rci") -> list[dict]:
    """One row per (experiment, world seed): {exp, world_seed, cells: {method: npz}}."""
    rows = []
    for exp in exps:
        exp_dir = resolve_exp_dir(exp, source)
        mp = exp_dir / MANIFEST_NAME
        if not mp.exists():
            print(f"  [warn] {exp_dir.name}: no {MANIFEST_NAME} — run record_trajectories.py first")
            continue
        with open(mp) as f:
            man = json.load(f)
        seeds = man.get("world_seeds") or []
        if not seeds:
            print(f"  [warn] {exp_dir.name}: manifest has no world seeds")
            continue
        for ws in seeds:
            cells = {}
            for method, entry in man.get("methods", {}).items():
                for ep in entry.get("episodes", []):
                    if int(ep["world_seed"]) == int(ws) and Path(ep["npz"]).exists():
                        cells[method] = ep["npz"]
            rows.append({"exp": exp_dir.name, "world_seed": int(ws), "cells": cells,
                         "multi": len(seeds) > 1,
                         "snapshots": man.get("snapshots", {}).get(str(ws), {})})
    return rows


def world_bounds_of(t: dict) -> tuple | None:
    m = t.get("meta", {})
    if m.get("world_width") and m.get("world_height"):
        return float(m["world_width"]), float(m["world_height"])
    return None


def panel_title(t: dict, label: str) -> str:
    m = t.get("meta", {})
    bits = [label]
    if m.get("objects_found") is not None:
        bits.append(f"{m['objects_found']} obj")
    if m.get("total_score") is not None:
        bits.append(f"score {float(m['total_score']):.0f}")
    return " · ".join(bits)


_BG_CACHE: dict = {}


def row_background(row: dict, view: str | None) -> tuple:
    """(image, view_proj) of the row's snapshot in `view`, or (None, None) with a warning."""
    if not view:
        return None, None
    png = row["snapshots"].get(view)
    if not png or not Path(png).exists():
        print(f"  [warn] {row['exp']} world {row['world_seed']}: no '{view}' snapshot — "
              f"blank slate (snapshot_worlds.py {row['exp']} --views {view})")
        return None, None
    if png not in _BG_CACHE:
        image, meta = load_snapshot(png)
        _BG_CACHE[png] = (image, meta["view_proj"])
    return _BG_CACHE[png]


def row_label(row: dict) -> str:
    return f"{row['exp']}\nworld {row['world_seed']}" if row["multi"] else row["exp"]


# ─────────────────────────────────────────────
#  Figures
# ─────────────────────────────────────────────

def draw_grid(rows, labels, args, out: Path) -> Path:
    n_rows, n_cols = len(rows), len(labels)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(args.panel * n_cols, args.panel * n_rows),
                             squeeze=False)
    colors = dict(zip(labels, default_colors(len(labels))))
    for r, row in enumerate(rows):
        bg, vp = row_background(row, args.background)
        for c, label in enumerate(labels):
            ax = axes[r][c]
            npz = row["cells"].get(label)
            if npz is None:
                if bg is not None:
                    plot_trajectories(ax, [], background=bg, view_proj=vp)
                ax.text(0.5, 0.5, "no trajectory", ha="center", va="center",
                        transform=ax.transAxes, color="0.5")
                ax.set_xticks([]); ax.set_yticks([])
                if r == 0:
                    ax.set_title(label)
                continue
            t = load_trajectory(npz)
            plot_trajectories(ax, [{"xyz": t["xyz"], "steps": t["steps"],
                                    "pickup_steps": t["pickup_steps"], "color": colors[label]}],
                              world_bounds=world_bounds_of(t), background=bg, view_proj=vp,
                              colour_by_time=args.colour_by_time, subsample=args.subsample,
                              show_pickups=not args.no_pickups, linewidth=args.linewidth,
                              title=panel_title(t, label))
            if c > 0:
                ax.set_ylabel("")
            if r < n_rows - 1:
                ax.set_xlabel("")
        axes[r][0].set_ylabel(row_label(row) if bg is not None
                              else f"{row_label(row)}\nz (Unity, m)")
    if args.title:
        fig.suptitle(args.title, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return out


def draw_overlay(rows, labels, args, out: Path) -> Path:
    n = len(rows)
    n_cols = min(n, args.overlay_cols)
    n_rows = int(np.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(args.panel * 1.3 * n_cols, args.panel * 1.3 * n_rows),
                             squeeze=False)
    colors = dict(zip(labels, default_colors(len(labels))))
    for i, row in enumerate(rows):
        ax = axes[i // n_cols][i % n_cols]
        items, bounds = [], None
        for label in labels:
            npz = row["cells"].get(label)
            if npz is None:
                continue
            t = load_trajectory(npz)
            bounds = bounds or world_bounds_of(t)
            items.append({"xyz": t["xyz"], "steps": t["steps"],
                          "pickup_steps": t["pickup_steps"], "color": colors[label],
                          "label": label})
        bg, vp = row_background(row, args.background)
        plot_trajectories(ax, items, world_bounds=bounds, background=bg, view_proj=vp,
                          subsample=args.subsample, show_pickups=not args.no_pickups,
                          alpha=0.8, linewidth=args.linewidth, legend=True,
                          title=row_label(row).replace("\n", " · "))
    for j in range(n, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")
    if args.title:
        fig.suptitle(args.title, y=1.0)
    fig.tight_layout()
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    return out


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exps", nargs="+", help="experiment ids in row order (or paths)")
    ap.add_argument("--methods", default=None,
                    help="comma-separated column order (default: all recorded, first seen)")
    ap.add_argument("--overlay", action="store_true",
                    help="one panel per row with all methods overlaid, instead of the grid")
    ap.add_argument("--overlay-cols", type=int, default=2)
    ap.add_argument("--colour-by-time", action="store_true", dest="colour_by_time")
    ap.add_argument("--subsample", type=int, default=1, help="keep every k-th pose")
    ap.add_argument("--no-pickups", action="store_true", dest="no_pickups")
    ap.add_argument("--background", default=None, choices=["ortho", "persp"],
                    help="draw on the world snapshot of this view (snapshot_worlds.py / "
                         "record_trajectories.py --snapshot) instead of a blank slate")
    ap.add_argument("--linewidth", type=float, default=1.2)
    ap.add_argument("--panel", type=float, default=4.0, help="panel size in inches")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--title", default=None)
    ap.add_argument("--name", default=None, help="suffix for the output file name")
    ap.add_argument("--out", default=None, help="output dir (default results/analysis/)")
    ap.add_argument("--show", action="store_true",
                    help="open the written PNG in the system image viewer (xdg-open)")
    ap.add_argument("--local", action="store_true",
                    help="look experiment ids up under results/experiments instead of the "
                         "default results/rci (same as record_trajectories.py --local)")
    args = ap.parse_args()

    rows = load_rows(args.exps, "local" if args.local else "rci")
    if not rows:
        sys.exit("ERROR: nothing to plot.")
    found = []
    for row in rows:
        for m in row["cells"]:
            if m not in found:
                found.append(m)
    if args.methods:
        labels = [m.strip() for m in args.methods.split(",") if m.strip()]
        missing = [m for m in labels if m not in found]
        if missing:
            print(f"  [warn] no trajectories for {missing} (recorded: {found})")
    else:
        labels = found
    if not labels:
        sys.exit("ERROR: no methods recorded in these manifests.")

    print(f"Rows:    {[row_label(r).replace(chr(10), ' ') for r in rows]}")
    print(f"Columns: {labels}")
    out_dir = Path(args.out) if args.out else Path(__file__).parent / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = (f"_{args.background}" if args.background else "") + (f"_{args.name}" if args.name else "")
    layout = "overlay" if args.overlay else "grid"
    out = out_dir / f"trajectories_{layout}{suffix}.png"
    (draw_overlay if args.overlay else draw_grid)(rows, labels, args, out)
    print(f"Wrote {out}")
    if args.show:
        open_file(out)


def open_file(path: Path) -> None:
    """Open with the platform viewer, detached so the script returns at once."""
    import platform
    import subprocess
    opener = {"Darwin": ["open"], "Windows": ["cmd", "/c", "start", ""]}.get(
        platform.system(), ["xdg-open"])
    try:
        subprocess.Popen(opener + [str(path)], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
    except FileNotFoundError:
        print(f"  [warn] could not find {opener[0]} to open {path}")


if __name__ == "__main__":
    main()
