"""Look at what the ground-truth probes predict (PNG / GIF, headless).

    $PPO_PYTHON_PATH -m probing.tools.inspect_probes --probes results/probing/full/probes \\
        --dataset results/probing/full/dataset --target glayout
    $PPO_PYTHON_PATH -m probing.tools.inspect_probes ... --target gboxes --episode 35 \\
        --models L0_baseline__dreamer__seed0__stage5 obs --t 1000 3000 --video

Writes to <probes>/analysis/inspect/:

    curves_<target>.png       validation loss per epoch of every probe
    <target>_epXXXX.png       one row per step t; columns: the truth, then each
                              model's predicted crop pasted into the whole map
                              (outside the crop: grey). Yellow outline: memory
                              cells (seen before, not in view); the crop's frame
                              and the agent are drawn on every panel. gboxes:
                              boxes still there circled, collected ones x, and
                              the k highest-scored cells (k = boxes in the crop)
                              outlined
    <target>_epXXXX.gif       --video: every saved step of that episode

Predictions exist for the fixed test episodes train_probe saved
(--save_pred_per_source). Default models: seed 0 of each cell at its last
stage, plus the controls.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from probing.core import dataset  # noqa: E402
from probing.s4_probe import global_score as gs  # noqa: E402
from probing.tools.inspect_layouts import INK, YELLOW, extent  # noqa: E402

CKPT = re.compile(r"^(?P<cell>.+)__dreamer__seed(?P<seed>\d+)__stage(?P<stage>\d+)$")
OUTSIDE = "#d6d6d4"


def target_dir(model_dir: Path, target: str) -> Path | None:
    """<target>_w<window> (probe_job.qsub) or plain <target>; probe seed 0."""
    hits = sorted(p for p in model_dir.glob(f"{target}*")
                  if (p / "metrics.json").exists() and "__pseed" not in p.name)
    return hits[0] if hits else None


def default_models(probes: Path, target: str) -> list[str]:
    last: dict = {}
    controls = []
    for d in sorted(p for p in probes.iterdir() if p.is_dir() and p.name != "analysis"):
        if target_dir(d, target) is None:
            continue
        m = CKPT.match(d.name)
        if not m:
            controls.append(d.name)
        elif m["seed"] == "0" and int(m["stage"]) >= last.get(m["cell"], (-1, ""))[0]:
            last[m["cell"]] = (int(m["stage"]), d.name)
    return [name for _, name in sorted(last.values(), key=lambda v: v[1])] + controls


def plot_curves(probes: Path, target: str, out: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")
    for f in sorted(probes.glob(f"*/{target}*/metrics.json")):
        rec = json.loads(f.read_text())
        if "val_history" in rec:
            ax.plot(rec["val_history"], lw=1, alpha=0.7, label=f"{f.parent.parent.name}/{f.parent.name}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation loss (BCE)")
    ax.set_title(f"{target}: early stopping on validation loss (best epoch = minimum)", loc="left")
    if len(ax.lines) <= 12:
        ax.legend(fontsize=6)
    fig.tight_layout()
    path = out / f"curves_{target}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


class Episode:
    """Truth, masks and every model's predictions of one test episode."""

    def __init__(self, root: Path, probes: Path, target: str, e: int, models: list[str]):
        self.target, self.e = target, e
        self.meta = dataset.load_json(root / "meta.json")
        self.ext = extent(self.meta)
        self.pose = dataset.load_episode(dataset.episode_path(root, e))["pose"]
        self.tg = dataset.load_episode(root / "gtargets" / f"ep{e:04d}.npz")
        self.shape = self.tg["glayout"].shape
        self.preds, self.models = {}, []
        for m in models:
            d = target_dir(probes / m, target)
            if d is None:
                continue
            with np.load(d / "test_predictions.npz", allow_pickle=False) as f:
                sel = f["episode"] == e
                if sel.any():
                    self.preds[m] = {"t": f["t"][sel], "rc": f["agent_rc"][sel],
                                     "pred": f["pred"][sel].astype(np.float32)}
                    self.window = json.loads((d / "metrics.json").read_text()).get("window") or 0
                    self.models.append(m)
        if not self.models:
            raise SystemExit(f"no saved predictions of ep{e:04d} for {models}")
        self.t = self.preds[self.models[0]]["t"]
        # One colour scale for every model's p(box): comparable panels.
        self.vmax = 1.0 if target == "glayout" else max(float(np.nanpercentile(
            np.concatenate([p["pred"].ravel() for p in self.preds.values()]), 99.9)), 1e-3)

    def masks(self, k: int) -> dict:
        t = self.t[k]
        i = int(np.searchsorted(self.tg["t"], t))
        inview = self.tg["gage"][i] == 0
        known = (self.tg["gknown_first"] >= 0) & (self.tg["gknown_first"] <= t)
        return {"memory": known & ~inview, "inview": inview}

    def pasted(self, model: str, k: int) -> np.ndarray:
        p = self.preds[model]
        full, _ = gs.uncrop(p["pred"][k].reshape(1, -1), p["rc"][k:k + 1],
                            self.window or None, self.shape)
        return full.reshape(self.shape)

    def truth(self, k: int) -> np.ndarray:
        if self.target == "glayout":
            return self.tg["glayout"]
        m = np.zeros(self.tg["glayout"].size, np.float32)
        alive = self.tg["box_collected_t"] > self.t[k]
        m[self.tg["box_cell"][alive]] = 1
        return m.reshape(self.shape)


def draw(ax, ep: Episode, k: int, model: str | None) -> None:
    """One panel: the truth (model None) or a model's pasted prediction."""
    cmap = plt.get_cmap("Greys" if ep.target == "glayout" else "Purples").copy()
    cmap.set_bad(OUTSIDE)
    img = ep.truth(k) if model is None else ep.pasted(model, k)
    vmax = 1.0 if model is None else ep.vmax
    ax.imshow(img, origin="lower", extent=ep.ext, cmap=cmap, vmin=0, vmax=vmax,
              interpolation="nearest")
    m = ep.masks(k)
    ax.contour(m["memory"], levels=[0.5], colors=YELLOW, linewidths=0.9,
               extent=ep.ext, origin="lower")
    if ep.target == "gboxes":
        ax.contour(ep.tg["glayout"] >= 0.5, levels=[0.5], colors="#bdbdbb",
                   linewidths=0.3, extent=ep.ext, origin="lower")
        h, w = ep.shape
        cw = (ep.ext[1] - ep.ext[0]) / w
        xy = np.stack([ep.ext[0] + (ep.tg["box_cell"] % w + 0.5) * cw,
                       ep.ext[2] + (ep.tg["box_cell"] // w + 0.5) * cw], 1)
        alive = ep.tg["box_collected_t"] > ep.t[k]
        ax.scatter(*xy[alive].T, s=60, facecolors="none", edgecolors=INK, lw=1.0)
        ax.scatter(*xy[~alive].T, marker="x", s=30, c=INK, lw=1.0)
        if model is not None:
            inside = ~np.isnan(img)
            kk = int((ep.truth(k).astype(bool) & inside).sum())
            if kk:
                flat = np.where(inside, img, -np.inf).ravel()
                for c in np.argpartition(-flat, kk - 1)[:kk]:
                    r, q = divmod(int(c), w)
                    ax.add_patch(Rectangle((ep.ext[0] + q * cw, ep.ext[2] + r * cw), cw, cw,
                                           fill=False, ec="#0ca30c", lw=1.0))
    t = ep.t[k]
    x, y = ep.pose[t, :2]
    if ep.window:
        half = ep.window * (ep.ext[1] - ep.ext[0]) / ep.shape[1] / 2
        ax.add_patch(Rectangle((x - half, y - half), 2 * half, 2 * half, fill=False,
                               ec=INK, lw=0.8, ls="--"))
    ax.plot(x, y, "o", ms=6, mfc=YELLOW, mec=INK)
    ax.set_xlim(ep.ext[0], ep.ext[1])  # a crop sticking out must not rescale the map
    ax.set_ylim(ep.ext[2], ep.ext[3])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")


def plot_episode(ep: Episode, steps, out: Path) -> Path:
    ks = [int(np.argmin(np.abs(ep.t - s))) for s in steps] if steps else \
        list(np.linspace(len(ep.t) // 10, len(ep.t) - 1, 4).astype(int))
    cols = [None] + ep.models
    fig, axes = plt.subplots(len(ks), len(cols), figsize=(3.6 * len(cols), 3.7 * len(ks)),
                             facecolor="white", squeeze=False)
    for r, k in enumerate(ks):
        for c, model in enumerate(cols):
            draw(axes[r, c], ep, k, model)
            name = "truth" if model is None else model.replace("__dreamer__", " ")
            axes[r, c].set_title(f"t={ep.t[k]}  {name}", fontsize=8, loc="left")
    what = "wall fraction" if ep.target == "glayout" else f"p(box), colour scale 0-{ep.vmax:.2f}"
    fig.suptitle(f"{ep.target} ep{ep.e:04d}: truth vs predicted {what} "
                 f"(crop {ep.window or 'whole map'}; yellow = memory cells)",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    path = out / f"{ep.target}_ep{ep.e:04d}.png"
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def write_video(ep: Episode, out: Path, every: int = 2) -> Path:
    from matplotlib.animation import FuncAnimation, PillowWriter
    cols = [None] + ep.models
    fig, axes = plt.subplots(1, len(cols), figsize=(3.6 * len(cols), 3.9), facecolor="white")
    axes = np.atleast_1d(axes)

    def frame(k):
        for ax, model in zip(axes, cols):
            ax.clear()
            draw(ax, ep, k, model)
            ax.set_title(f"t={ep.t[k]}  {'truth' if model is None else model}", fontsize=8, loc="left")
    anim = FuncAnimation(fig, frame, frames=range(0, len(ep.t), every))
    path = out / f"{ep.target}_ep{ep.e:04d}.gif"
    anim.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probes", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--target", choices=gs.GLOBAL_TARGETS, required=True)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--episode", type=int, default=None,
                    help="default: every episode with saved predictions")
    ap.add_argument("--t", type=int, nargs="*", default=None, help="steps (rows)")
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--out", default=None, help="default: <probes>/analysis/inspect")
    args = ap.parse_args()

    probes, root = Path(args.probes), Path(args.dataset)
    out = Path(args.out or probes / "analysis" / "inspect")
    out.mkdir(parents=True, exist_ok=True)
    print(f"[inspect-probes] {plot_curves(probes, args.target, out)}")
    models = args.models or default_models(probes, args.target)
    have = {m: target_dir(probes / m, args.target) for m in models}
    missing = [m for m, d in have.items() if d is None or not (d / "test_predictions.npz").exists()]
    if missing:
        # A pull with --exclude=test_predictions.npz brings metrics but no predictions.
        print(f"[inspect-probes] no saved predictions for {missing} under {probes}; "
              f"copy <model>/{args.target}_w*/test_predictions.npz from the cluster to plot them")
    models = [m for m in models if m not in missing]
    if not models:
        raise SystemExit("[inspect-probes] nothing to plot")
    if args.episode is not None:
        episodes = [args.episode]
    else:
        d = target_dir(probes / models[0], args.target)
        with np.load(d / "test_predictions.npz", allow_pickle=False) as f:
            episodes = sorted(set(int(e) for e in f["episode"]))
    for e in episodes:
        ep = Episode(root, probes, args.target, e, models)
        print(f"[inspect-probes] {plot_episode(ep, args.t, out)}")
        if args.video:
            print(f"[inspect-probes] {write_video(ep, out)}")


if __name__ == "__main__":
    main()
