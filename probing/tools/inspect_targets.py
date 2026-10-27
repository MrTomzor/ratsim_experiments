"""Look at the global probe targets and their masks (PNG, headless).

    $PPO_PYTHON_PATH -m probing.tools.inspect_targets --dataset results/probing/full/dataset
    $PPO_PYTHON_PATH -m probing.tools.inspect_targets --dataset ... --episodes 0 --t 500 2500 5000

Writes to <dataset>/gtargets/inspect/ (or --out):

    epXXXX.png   one row per step t (the nearest sample at or before it):
                 1. glayout, the wall fraction the probe must predict, with the
                    trajectory so far
                 2. gboxes, the boxes still there (yellow; ringed once a ray has
                    returned off them), collected ones as x
                 3. which group each cell is scored in: in view now, memory
                    (seen before, not in view), never seen
                 4. memory cells by steps since last in view
    summary.png  over all episodes and per source: the share of the map in each
                 group, and observed / collected boxes, against t

Default episodes: the lowest id of every source (explorer, each policy run).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap, LogNorm  # noqa: E402

from probing.core import dataset  # noqa: E402
from probing.tools.inspect_layouts import (INK, PAPER, TIME_CMAP, YELLOW,  # noqa: E402
                                     default_episodes, extent, source_group)

BLUE, PANEL = "#2a78d6", "#EDEDEB"
GROUPS = ("in view", "memory", "never seen")
GROUP_CMAP = ListedColormap([BLUE, YELLOW, PANEL])


def masks(tg: dict, k: int) -> dict:
    """The scoring groups at sample k (see global_targets)."""
    t = tg["t"][k]
    inview = tg["gage"][k] == 0
    known = (tg["gknown_first"] >= 0) & (tg["gknown_first"] <= t)
    return {"in view": inview, "memory": known & ~inview,
            "never seen": ~known & ~inview}


def box_state(tg: dict, t: int) -> tuple[np.ndarray, np.ndarray]:
    alive = tg["box_collected_t"] > t
    observed = tg["box_first_hit_t"] <= t
    return alive, observed


def cell_xy(cells: np.ndarray, shape, ext) -> np.ndarray:
    h, w = shape
    cw, ch = (ext[1] - ext[0]) / w, (ext[3] - ext[2]) / h
    return np.stack([ext[0] + (cells % w + 0.5) * cw,
                     ext[2] + (cells // w + 0.5) * ch], 1)


def plot_episode(e: int, root: Path, out: Path, meta: dict, steps) -> Path:
    ep = dataset.load_episode(dataset.episode_path(root, e))
    tg = dataset.load_episode(root / "gtargets" / f"ep{e:04d}.npz")
    ext = extent(meta)
    shape = tg["glayout"].shape
    n = len(ep["pose"])
    steps = steps or [n // 10, n // 3, 2 * n // 3, n - 1]
    rows = [int(np.searchsorted(tg["t"], min(s, tg["t"][-1]), side="right") - 1)
            for s in steps]

    fig, axes = plt.subplots(len(rows), 4, figsize=(20, 5 * len(rows)),
                             facecolor="white", squeeze=False)
    for r, k in enumerate(rows):
        t = int(tg["t"][k])
        m = masks(tg, k)
        traj = ep["pose"][:t + 1]
        alive, observed = box_state(tg, t)
        bxy = cell_xy(tg["box_cell"], shape, ext)

        ax = axes[r, 0]
        im = ax.imshow(tg["glayout"], origin="lower", extent=ext, cmap="Greys",
                       vmin=0, vmax=1, interpolation="nearest")
        ax.scatter(traj[:, 0], traj[:, 1], c=np.arange(len(traj)), cmap=TIME_CMAP,
                   s=0.5, vmin=0, vmax=n, rasterized=True)
        ax.plot(*traj[-1, :2], "o", ms=8, mfc=YELLOW, mec=INK)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01).set_label("wall fraction")
        ax.set_title(f"t={t}: glayout (target) + trajectory", loc="left", fontsize=10)

        ax = axes[r, 1]
        ax.imshow(tg["glayout"] >= 0.5, origin="lower", extent=ext,
                  cmap=ListedColormap(["white", "#d6d6d4"]), interpolation="nearest")
        ax.scatter(*bxy[alive & ~observed].T, s=28, c=YELLOW, edgecolors="none",
                   label=f"box, not observed yet ({(alive & ~observed).sum()})")
        ax.scatter(*bxy[alive & observed].T, s=28, c=YELLOW, edgecolors=INK, lw=1.2,
                   label=f"box, observed ({(alive & observed).sum()})")
        ax.scatter(*bxy[~alive].T, marker="x", s=28, c=INK, lw=1.2,
                   label=f"collected ({(~alive).sum()})")
        ax.plot(*traj[-1, :2], "o", ms=8, mfc=YELLOW, mec=INK)
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
        ax.set_title(f"t={t}: gboxes (target: cells with a box still there)",
                     loc="left", fontsize=10)

        ax = axes[r, 2]
        group = np.select([m["in view"], m["memory"]], [0, 1], 2)
        ax.imshow(group, origin="lower", extent=ext, cmap=GROUP_CMAP, vmin=-0.5,
                  vmax=2.5, interpolation="nearest")
        ax.contour(tg["glayout"] >= 0.5, levels=[0.5], colors=INK, linewidths=0.4,
                   extent=ext, origin="lower")
        ax.plot(*traj[-1, :2], "o", ms=8, mfc="white", mec=INK)
        handles = [plt.Rectangle((0, 0), 1, 1, fc=c, ec=INK, lw=0.4)
                   for c in (BLUE, YELLOW, PANEL)]
        ax.legend(handles, [f"{g} ({100 * m[g].mean():.1f}%)" for g in GROUPS],
                  loc="upper right", fontsize=7, framealpha=0.9)
        ax.set_title(f"t={t}: scoring groups (wall outline)", loc="left", fontsize=10)

        ax = axes[r, 3]
        age = np.where(m["memory"], tg["gage"][k], np.nan).astype(float)
        ax.imshow(np.where(m["never seen"], 1.0, np.nan), origin="lower", extent=ext,
                  cmap=ListedColormap([PANEL]), interpolation="nearest")
        im = ax.imshow(age, origin="lower", extent=ext, cmap="Blues_r",
                       norm=LogNorm(1, max(n, 10)), interpolation="nearest")
        ax.plot(*traj[-1, :2], "o", ms=8, mfc=YELLOW, mec=INK)
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.01).set_label("steps since last in view")
        ax.set_title(f"t={t}: memory cells by age", loc="left", fontsize=10)
        for ax in axes[r]:
            ax.set_aspect("equal")
    fig.suptitle(f"ep{e:04d}  ({source_group(ep)}, seed {int(ep['world_seed'])})  "
                 f"global targets {shape[0]}x{shape[1]}", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    path = out / f"ep{e:04d}.png"
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def plot_summary(root: Path, out: Path) -> Path:
    curves: dict[str, list] = {}
    for p in dataset.list_episodes(root):
        with np.load(p, allow_pickle=False) as f:
            g = source_group({"policy_run": f["policy_run"], "source": f["source"]})
        tg = dataset.load_episode(root / "gtargets" / p.name)
        t = tg["t"]
        known = (tg["gknown_first"][None] >= 0) & (tg["gknown_first"][None] <= t[:, None, None])
        inview = tg["gage"] == 0
        row = {"t": t,
               "in view": inview.mean((1, 2)),
               "memory": (known & ~inview).mean((1, 2)),
               "never seen": (~known & ~inview).mean((1, 2)),
               "observed": (tg["box_first_hit_t"][None] <= t[:, None]).sum(1),
               "collected": (tg["box_collected_t"][None] <= t[:, None]).sum(1)}
        curves.setdefault(g, []).append(row)

    panels = [("memory", "memory cells, share of the map"),
              ("in view", "in-view cells, share of the map"),
              ("never seen", "never-seen cells, share of the map"),
              ("observed", "boxes observed (of 36)"),
              ("collected", "boxes collected (of 36)")]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e87ba4"]
    fig, axes = plt.subplots(1, len(panels), figsize=(5 * len(panels), 4.2), facecolor="white")
    for ax, (key, title) in zip(axes, panels):
        for color, (g, rows) in zip(colors, sorted(curves.items())):
            t = rows[0]["t"]
            vals = np.stack([r[key] for r in rows])
            lo, mid, hi = np.percentile(vals, [25, 50, 75], axis=0)
            ax.plot(t, mid, color=color, lw=2, label=f"{g} (n={len(rows)})")
            ax.fill_between(t, lo, hi, color=color, alpha=0.15, lw=0)
        ax.set_facecolor(PAPER)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("step")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(fontsize=8, title="median, IQR band", title_fontsize=8)
    fig.suptitle("global targets over the episode, per source", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    path = out / "summary.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--t", type=int, nargs="*", default=None, help="steps (rows)")
    ap.add_argument("--out", default=None, help="default: <dataset>/gtargets/inspect")
    ap.add_argument("--no_summary", action="store_true")
    args = ap.parse_args()

    root = Path(args.dataset)
    out = Path(args.out or root / "gtargets" / "inspect")
    out.mkdir(parents=True, exist_ok=True)
    meta = dataset.load_json(root / "meta.json")
    episodes = args.episodes if args.episodes is not None else default_episodes(root, 1)
    for e in episodes:
        print(f"[inspect-targets] {plot_episode(e, root, out, meta, args.t)}")
    if not args.no_summary:
        print(f"[inspect-targets] {plot_summary(root, out)}")


if __name__ == "__main__":
    main()
