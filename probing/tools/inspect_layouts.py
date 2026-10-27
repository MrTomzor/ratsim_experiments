"""Look at the ground-truth layouts and their verification (PNG, headless).

    $PPO_PYTHON_PATH -m probing.tools.inspect_layouts --dataset results/probing/full/dataset
    $PPO_PYTHON_PATH -m probing.tools.inspect_layouts --dataset ... --episodes 0 250 --t 1200

Writes to <dataset>/layouts/inspect/ (or --out):

    epXXXX.png   1. the reference map: walls, boxes (yellow; a collected box is
                    hollow, with an x where the agent was when it took it), the
                    trajectory coloured by time
                 2. every lidar return of every 5th step on the map, coloured by
                    its check: wall ok (grey), box ok (yellow), mismatch (red)
                 3. a 40 m zoom at step --t: the scan's rays, coloured the same
    summary.png  the per-episode checks of verify_report.json, all episodes

Default episodes: the two lowest ids of every source (explorer, each policy run).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

from probing.core import dataset  # noqa: E402
from probing.s2_targets import layouts, targets  # noqa: E402

# design.md tokens: ink on paper, yellow as the one accent (boxes); status red
# for failed checks only.
INK, PAPER, GREY, YELLOW, YELLOW_DEEP = "#1A1A1A", "#F4F4F2", "#9C9C9A", "#F6C633", "#EAB308"
CRITICAL, SERIOUS = "#d03b3b", "#ec835a"
TIME_CMAP = LinearSegmentedColormap.from_list(   # Blues without its near-white end
    "time", plt.cm.Blues(np.linspace(0.3, 1.0, 256)))
CLASS_STYLE = {  # ray class -> (colour, marker size, z-order, label)
    "wall_ok": (GREY, 1.0, 2, "wall return ok"),
    "reward_ok": (YELLOW_DEEP, 3.0, 3, "box return ok"),
    "wall_bad": (CRITICAL, 12.0, 5, "wall return off any wall"),
    "reward_bad": (CRITICAL, 12.0, 5, "box return off any box"),
    "free_bad": (SERIOUS, 12.0, 4, "ray crosses a wall"),
}


def source_group(ep: dict) -> str:
    run = str(ep["policy_run"])
    return run.split("__")[0] if run else str(ep["source"])


def default_episodes(root: Path, per_group: int = 2) -> list[int]:
    seen: dict[str, list[int]] = {}
    for p in dataset.list_episodes(root):
        with np.load(p, allow_pickle=False) as f:
            g = source_group({"policy_run": f["policy_run"], "source": f["source"]})
        if len(seen.setdefault(g, [])) < per_group:
            seen[g].append(dataset.episode_id(p))
    return sorted(e for ids in seen.values() for e in ids)


def extent(meta: dict) -> tuple[float, float, float, float]:
    tr = targets.new_tracker(meta)
    return (tr.origin_x, tr.origin_x + tr.cells_x * tr.resolution,
            tr.origin_y, tr.origin_y + tr.cells_y * tr.resolution)


def draw_map(ax, world: dict, meta: dict) -> None:
    ax.imshow(world["wall"], origin="lower", extent=extent(meta),
              cmap=ListedColormap(["white", INK]), vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")


def draw_boxes(ax, world: dict, boxes: dict, ep: dict) -> None:
    xy, collected = world["box_xy"], boxes["collected_t"] < layouts.NEVER
    ax.scatter(*xy[~collected].T, s=40, c=YELLOW, edgecolors=INK, lw=0.8, zorder=6,
               label=f"box ({(~collected).sum()} left)")
    ax.scatter(*xy[collected].T, s=40, facecolors="none", edgecolors=YELLOW_DEEP,
               lw=1.2, zorder=6, label=f"box collected ({collected.sum()})")
    t = boxes["collected_t"][collected]
    ax.scatter(*ep["pose"][t, :2].T, marker="x", s=30, c=INK, lw=1.2, zorder=7,
               label="agent at pickup")


def plot_episode(e: int, root: Path, out: Path, meta: dict, report: dict,
                 radius: float, t_zoom: int | None) -> Path:
    ep = dataset.load_episode(dataset.episode_path(root, e))
    world = dataset.load_episode(root / "layouts" / f"ep{e:04d}.npz")
    v = layouts.verify(ep, world, meta, radius, detail=True)
    d = v["detail"]
    rs, cls, boxes = d["rays"], d["ray_class"], d["boxes"]
    hx, hy = d["hit_xy"]
    pose = ep["pose"]
    n = len(pose)
    t_zoom = n // 2 if t_zoom is None else min(t_zoom, n - 1)

    fig, axes = plt.subplots(1, 3, figsize=(21, 7.4), facecolor="white",
                             gridspec_kw={"width_ratios": [1, 1, 0.85]})
    # 1. reference map + trajectory + boxes
    ax = axes[0]
    draw_map(ax, world, meta)
    sc = ax.scatter(pose[:, 0], pose[:, 1], c=np.arange(n), cmap=TIME_CMAP, s=0.6,
                    zorder=4, rasterized=True)
    ax.plot(*pose[0, :2], "o", ms=8, mfc="white", mec=INK, zorder=8, label="start")
    draw_boxes(ax, world, boxes, ep)
    fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.01).set_label("step")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.set_title(f"reference map  ep{e:04d}  ({source_group(ep)}, seed {int(ep['world_seed'])})",
                 loc="left", fontsize=10)

    # 2. every lidar return of every 5th step, coloured by its check
    ax = axes[1]
    draw_map(ax, world, meta)
    every5 = rs["t"] % 5 == 0
    for name, (color, size, z, label) in CLASS_STYLE.items():
        sel = cls == name
        if not name.endswith("_bad"):
            sel &= every5  # mismatches: all of them
        if name == "free_bad":
            px, py = rs["x0"][sel], rs["y0"][sel]   # from where the bad ray left
        else:
            px, py = hx[sel], hy[sel]
        ax.scatter(px, py, s=size, c=color, lw=0, zorder=z, rasterized=True,
                   label=f"{label} ({int((cls == name).sum())})")
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9, markerscale=3)
    ax.set_title(f"lidar returns  wall_ok {v['wall_ok']:.4f} (mirrored {v['wall_ok_mirrored']:.3f})  "
                 f"box_ok {v['reward_ok']:.4f}  free_ok {v['free_ok']:.4f}", loc="left", fontsize=10)

    # 3. zoom at t: the rays of one scan
    ax = axes[2]
    draw_map(ax, world, meta)
    x, y = pose[t_zoom, :2]
    ax.set_xlim(x - 20, x + 20)
    ax.set_ylim(y - 20, y + 20)
    at = rs["t"] == t_zoom
    if at.any():
        ends = np.where(rs["hit"][at], rs["range"][at], rs["max_range"])
        ex, ey = layouts._point({k: rs[k][at] for k in ("x0", "y0", "angle")}, ends)
        for k in range(at.sum()):
            c = cls[at][k]
            color = CLASS_STYLE.get(c, (GREY,))[0] if c != "no_return" else "#c8c8c6"
            ax.plot([x, ex[k]], [y, ey[k]], color=color, lw=1.4, zorder=4)
            if c != "no_return":
                ax.plot(ex[k], ey[k], "o", ms=3.5, color=color, zorder=5)
    else:
        ax.text(x, y + 2, "embedded step: no geometry", ha="center", color=CRITICAL)
    ax.plot(x, y, "o", ms=9, mfc=YELLOW, mec=INK, zorder=6)
    ax.annotate("", xy=(x + 3 * np.cos(pose[t_zoom, 2]), y + 3 * np.sin(pose[t_zoom, 2])),
                xytext=(x, y), arrowprops=dict(arrowstyle="->", color=INK, lw=1.5), zorder=7)
    draw_boxes(ax, world, boxes, ep)
    ax.set_title(f"scan at step {t_zoom}  (pickups so far {int(ep['pickups'][t_zoom])})",
                 loc="left", fontsize=10)

    rep = report.get(f"ep{e:04d}", {})
    verdict = "FAIL: " + "; ".join(rep["fail"]) if rep.get("fail") else "all checks pass"
    fig.suptitle(f"ep{e:04d}   {v['n_boxes']} boxes, {v['n_pickups']} pickups "
                 f"(max dist {v['max_pickup_dist']:.2f} m), embedded steps "
                 f"{100 * v['embedded_frac']:.1f}%   reset match "
                 f"{rep.get('reset_match', '?')}   ->  {verdict}",
                 x=0.01, ha="left", fontsize=11,
                 color=CRITICAL if rep.get("fail") else INK)
    fig.tight_layout()
    path = out / f"ep{e:04d}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def plot_summary(report: dict, out: Path) -> Path:
    eps = report["episodes"]
    panels = [("wall_ok", "wall returns on a wall"),
              ("wall_ok_mirrored", "mirrored-angle control (must be low)"),
              ("reward_ok", "box returns on a live box"),
              ("free_ok", "rays through free cells only"),
              ("pose_ok", "agent positions in free cells"),
              ("max_pickup_dist", f"max pickup distance [m] (radius {report['pickup_radius']})"),
              ("embedded_frac", "embedded steps (skipped)")]
    fig, axes = plt.subplots(2, 4, figsize=(19, 7.5), facecolor="white")
    axes.ravel()[-1].set_visible(False)
    for ax, (key, title) in zip(axes.ravel(), panels):
        vals = np.array([s[key] for s in eps.values()])
        ax.set_facecolor(PAPER)
        if np.ptp(vals) == 0:  # a histogram of one value is just an axis
            ax.text(0.5, 0.5, f"all {len(vals)} episodes = {vals[0]:.4g}", ha="center",
                    va="center", transform=ax.transAxes, fontsize=12, color=INK)
            ax.set_xticks([])
            ax.set_yticks([])
        else:
            ax.hist(vals, bins=40, color=YELLOW, edgecolor="white", lw=0.5)
        ax.set_title(f"{title}\nmin {vals.min():.4g}  median {np.median(vals):.4g}  "
                     f"max {vals.max():.4g}", loc="left", fontsize=9)
        ax.set_ylabel("episodes")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    n_fail = sum(bool(s["fail"]) for s in eps.values())
    reset = sum(bool(s["reset_match"]) for s in eps.values())
    fig.suptitle(f"layout verification: {len(eps)} episodes, {n_fail} failed, "
                 f"reset match {reset}/{len(eps)}", x=0.01, ha="left", fontsize=12,
                 color=CRITICAL if n_fail else INK)
    fig.tight_layout()
    path = out / "summary.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episodes", type=int, nargs="*", default=None)
    ap.add_argument("--t", type=int, default=None, help="step of the zoom panel (default: middle)")
    ap.add_argument("--out", default=None, help="default: <dataset>/layouts/inspect")
    args = ap.parse_args()

    root = Path(args.dataset)
    out = Path(args.out or root / "layouts" / "inspect")
    out.mkdir(parents=True, exist_ok=True)
    meta = dataset.load_json(root / "meta.json")
    report_path = root / "layouts" / "verify_report.json"
    report = dataset.load_json(report_path) if report_path.exists() else {}
    radius = report.get("pickup_radius", 2.0)
    for e in (args.episodes if args.episodes is not None else default_episodes(root)):
        print(f"[inspect-layouts] {plot_episode(e, root, out, meta, report.get('episodes', {}), radius, args.t)}")
    if report:
        print(f"[inspect-layouts] {plot_summary(report, out)}")


if __name__ == "__main__":
    main()
