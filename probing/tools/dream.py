"""Dreamer's dreams, decoded: what the world model imagines the next seconds look
like, against what actually happened (dreamer venv; GIF / PNG, headless).

    $DREAMER_PYTHON_PATH -m probing.tools.dream --run RUN_DIR --stage 5 \\
        --dataset results/probing/full/dataset --episode 207 --t0 1500 --horizon 100

The model observes the episode from its first step up to t0 (posterior, the
state the agent acts from), then imagines `horizon` steps with the actions the
episode really took (prior only, no observations: DreamerV3's open-loop report,
here from the true episode start). The decoder turns every imagined state into
an observation: 19 lidar ranges + box semantics, gps and compass (pose = start
+ 300 * gps, yaw = pi * compass). The decoder renders position too coarsely to
draw (memory_orthomaze stage 12: 11 m median gps error even on OBSERVED steps,
8 m jumps between dream steps; gps is tiny next to 57 lidar values in its loss),
so the GIF shows what the model dreams well -- the egocentric scan -- against
the real scan of the same moment, from the real pose. The --starts summary keeps
the decoded-motion error as a curve.

    <out>/dream_<run>_ep<E>_t<T0>.gif   per imagined step: the true maze (verified
                                         layout), the real scan from the real pose
                                         (grey) and the dreamed scan from the
                                         dreamed pose (orange); ray ranges; errors
    <out>/dream_<run>_errors.png         --starts N: mean error vs seconds ahead
                                         over N random (episode, t0) of the test
                                         split, against "nothing changes" (the
                                         observation at t0 held fixed)

For orientation: the actor is trained on dreams of imag_length = 15 steps
(1.5 s); acting itself never dreams.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from probing.core import agent_io, dataset

STEP_S = 0.1
N_RAYS = 19
RANGE_M = 20.0
INK, PAPER, GREY, ORANGE, YELLOW = "#1A1A1A", "#F4F4F2", "#9C9C9A", "#eb6834", "#F6C633"


def dream_fn(agent, horizon: int):
    """Jitted: observe a prefix, imagine `horizon` steps with given actions,
    decode both. Returns (recon of the prefix, dream), each {key: pred}."""
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    m = agent.model

    def run(obs, prevact, future):
        b = obs["is_first"].shape[0]
        enc_c, dyn_c, dec_c = m.enc.initial(b), m.dyn.initial(b), m.dec.initial(b)
        enc_c, _, tokens = m.enc(enc_c, obs, obs["is_first"], False)
        dyn_c, _, obsfeat = m.dyn.observe(dyn_c, tokens, prevact, obs["is_first"], False)
        _, imgfeat, _ = m.dyn.imagine(dyn_c, future, horizon, False)
        dec_c, _, rec = m.dec(dec_c, obsfeat, obs["is_first"], False)
        _, _, img = m.dec(dec_c, imgfeat, jnp.zeros((b, horizon), bool), False)
        pick = lambda d: {k: d[k].pred() for k in ("lidar", "gps", "compass")}
        return pick(rec), pick(img)

    pure = nj.pure(run)
    return jax.jit(lambda params, obs, pa, fut, seed: pure(
        params, obs, pa, fut, seed=seed, create=False)[1])


def load(run_dir: Path, stage: int, root: Path):
    sp = dataset.load_json(root / "spaces.json")
    obs_space, act_space = dataset.spaces_from_json(sp["obs"]), dataset.spaces_from_json(sp["act"])
    agent = agent_io.build_agent(agent_io.make_config(run_dir), obs_space, act_space)
    agent_io.load_checkpoint(agent, agent_io.stage_checkpoint(run_dir, stage))
    return agent


def dream(agent, fn, ep: dict, t0: int, horizon: int, seed: int = 0) -> dict:
    """Prefix 0..t0 observed, steps t0+1..t0+horizon imagined with the
    episode's actions (prevact[t] = action[t-1], dataset.py)."""
    import jax
    t1 = t0 + horizon
    obs = {k: ep[k][None, :t0 + 1] for k in ("lidar", "gps", "compass")}
    obs["is_first"] = (np.arange(t0 + 1) == 0)[None]
    for k in ("reward", "is_last", "is_terminal"):
        obs[k] = np.zeros((1, t0 + 1), np.float32 if k == "reward" else bool)
    prev = {k: np.concatenate([[0], ep[k][:-1]]).astype(np.int32) for k in dataset.ACT_KEYS}
    prevact = {k: v[None, :t0 + 1] for k, v in prev.items()}
    future = {k: v[None, t0 + 1:t1 + 1] for k, v in prev.items()}
    put = lambda t: jax.tree.map(jax.device_put, t)
    rec, img = jax.device_get(fn(agent.params, put(obs), put(prevact), put(future),
                                 jax.device_put(np.array([seed, 0], np.uint32))))
    raw = np.concatenate([np.concatenate([300 * np.asarray(d["gps"][0], np.float64),
                                          np.pi * np.asarray(d["compass"][0], np.float64)], 1)
                          for d in (rec, img)])[t0:]          # step t0 (reconstructed), then the dream
    true_pose = ep["pose"][t0:t1 + 1].astype(np.float64)
    motion = raw - raw[:1]                                  # dreamed displacement / turn since t0
    dream_pose = true_pose[:1] + motion
    dream_pose[:, 2] = (dream_pose[:, 2] + np.pi) % (2 * np.pi) - np.pi
    dream_range = np.clip(np.concatenate([rec["lidar"][0, -1:, :N_RAYS], img["lidar"][0, :, :N_RAYS]]), 0, 1) * RANGE_M
    return {"t0": t0, "horizon": horizon, "true_pose": true_pose,
            "true_range": np.where(ep["ranges"][t0:t1 + 1] > 0, ep["ranges"][t0:t1 + 1], RANGE_M),
            "dream_pose": dream_pose, "dream_range": dream_range}


def errors(d: dict) -> dict:
    """Per step ahead (0 = t0, the posterior reconstruction)."""
    return {"range_mae_m": np.abs(d["dream_range"] - d["true_range"]).mean(1),
            "pos_err_m": np.hypot(*(d["dream_pose"][:, :2] - d["true_pose"][:, :2]).T),
            "hold_range_mae_m": np.abs(d["true_range"][:1] - d["true_range"]).mean(1),
            "hold_recon_range_mae_m": np.abs(d["dream_range"][:1] - d["true_range"]).mean(1),
            "hold_pos_err_m": np.hypot(*(d["true_pose"][:1, :2] - d["true_pose"][:, :2]).T)}


def ray_ends(pose, ranges, meta):
    from probing.s2_targets.targets import scan_geometry
    s = scan_geometry(meta)
    a = pose[2] + s["angle_start_rad"] + np.arange(len(ranges)) * s["angle_increment_rad"]
    return pose[0] + np.cos(a) * ranges, pose[1] + np.sin(a) * ranges


def plot_gif(d: dict, wall: np.ndarray, meta: dict, title: str, path: Path, every: int = 1):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import ListedColormap
    from probing.tools.inspect_layouts import extent
    e = errors(d)
    sec = np.arange(d["horizon"] + 1) * STEP_S
    fig = plt.figure(figsize=(13, 6.2), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    ax_map, ax_ray, ax_err = fig.add_subplot(gs[:, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
    cx, cy = d["true_pose"][0, :2]

    def frame(h):
        for ax in (ax_map, ax_ray, ax_err):
            ax.clear()
        ax_map.imshow(wall, origin="lower", extent=extent(meta),
                      cmap=ListedColormap(["white", INK]), vmin=0, vmax=1, interpolation="nearest")
        ax_map.plot(*d["true_pose"][:, :2].T, color=GREY, lw=1, ls=":", label="real path")
        for pose, rng, c, lab in ((d["true_pose"][h], d["true_range"][h], GREY, "real scan"),
                                  (d["true_pose"][h], d["dream_range"][h], ORANGE, "dreamed scan (same moment)")):
            ex, ey = ray_ends(pose, rng, meta)
            for x, y in zip(ex, ey):
                ax_map.plot([pose[0], x], [pose[1], y], color=c, lw=0.8, alpha=0.8)
            ax_map.scatter(ex, ey, s=10, color=c, zorder=4, label=lab)
        ax_map.plot(*d["true_pose"][h, :2], "o", ms=9, mfc=GREY, mec=INK, zorder=5)
        ax_map.set_xlim(cx - 25, cx + 25); ax_map.set_ylim(cy - 25, cy + 25); ax_map.set_aspect("equal")
        ax_map.set_title(f"{sec[h]:.1f} s into the dream" + ("  (t0: observed)" if h == 0 else ""),
                         loc="left", fontsize=11)
        ax_map.legend(fontsize=7, loc="upper right")
        deg = np.degrees(-np.linspace(-np.pi / 2, np.pi / 2, N_RAYS))
        ax_ray.plot(deg, d["true_range"][h], "o-", color=GREY, label="real")
        ax_ray.plot(deg, d["dream_range"][h], "o-", color=ORANGE, label="dreamed")
        ax_ray.set_ylim(0, RANGE_M + 1); ax_ray.set_xlabel("ray angle (deg, + = left)")
        ax_ray.set_ylabel("range [m]"); ax_ray.legend(fontsize=7); ax_ray.set_facecolor(PAPER)
        ax_err.plot(sec[:h + 1], e["range_mae_m"][:h + 1], color=ORANGE, label="dream: ray range error [m]")
        ax_err.plot(sec[:h + 1], e["hold_range_mae_m"][:h + 1], color=GREY, ls="--", label="'nothing changes': ray error")
        ax_err.plot(sec[:h + 1], e["hold_recon_range_mae_m"][:h + 1], color=GREY, ls=":", label="hold the t0 reconstruction")
        ax_err.axvline(1.5, color=YELLOW, lw=2, label="actor trains on 1.5 s dreams")
        ax_err.set_xlim(0, sec[-1]); ax_err.set_ylim(0, max(10, float(np.nanmax(e["range_mae_m"])) + 1))
        ax_err.set_xlabel("seconds ahead"); ax_err.legend(fontsize=7, loc="upper left"); ax_err.set_facecolor(PAPER)
        fig.suptitle(title, x=0.01, ha="left", fontsize=11)

    anim = FuncAnimation(fig, frame, frames=range(0, d["horizon"] + 1, every))
    anim.save(path, writer=PillowWriter(fps=10))
    frame(min(15, d["horizon"]))
    fig.savefig(path.with_suffix(".png"), dpi=100)
    plt.close(fig)


def plot_errors(curves: dict, horizon: int, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sec = np.arange(horizon + 1) * STEP_S
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), facecolor="white")
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#e34948"]
    for (name, es), c in zip(curves.items(), colors):
        for ax, key, hold in ((axes[0], "range_mae_m", "hold_range_mae_m"), (axes[1], "pos_err_m", "hold_pos_err_m")):
            v = np.stack([x[key] for x in es])
            lo, mid, hi = np.percentile(v, [25, 50, 75], 0)
            ax.plot(sec, mid, color=c, lw=2, label=f"{name} dream (median, n={len(es)})")
            ax.fill_between(sec, lo, hi, color=c, alpha=0.15, lw=0)
    first = next(iter(curves.values()))
    axes[0].plot(sec, np.median(np.stack([x["hold_recon_range_mae_m"] for x in first]), 0), color=GREY,
                 ls=":", lw=1.5, label=f"hold the t0 reconstruction ({next(iter(curves))})")
    for ax, hold in ((axes[0], "hold_range_mae_m"), (axes[1], "hold_pos_err_m")):
        ax.plot(sec, np.median(np.stack([x[hold] for x in first]), 0), color=GREY, ls="--",
                lw=1.5, label="'nothing changes' (obs at t0 held)")
        ax.axvline(1.5, color=YELLOW, lw=2, label="actor trains on 1.5 s dreams")
        ax.set_xlabel("seconds ahead"); ax.set_facecolor(PAPER); ax.legend(fontsize=7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel("mean |ray range error| [m]"); axes[1].set_ylabel("decoded motion error since t0 [m]")
    axes[0].set_title("lidar scan: what the dream shows", loc="left", fontsize=10)
    axes[1].set_title("position: decoded too coarsely to trust (error even when observed)", loc="left", fontsize=10)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", nargs="+", required=True, help="run dirs (run_config.json + checkpoints/)")
    ap.add_argument("--stage", type=int, nargs="+", required=True, help="one per run, or one for all")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--episode", type=int, default=None)
    ap.add_argument("--t0", type=int, default=1500)
    ap.add_argument("--horizon", type=int, default=100)
    ap.add_argument("--starts", type=int, default=0, help="error curves over N random test starts")
    ap.add_argument("--out", default="../../charts/dreams")
    args = ap.parse_args()
    root, out = Path(args.dataset), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = dataset.load_json(root / "meta.json")
    stages = args.stage * len(args.run) if len(args.stage) == 1 else args.stage
    test = dataset.split_episodes(len(dataset.list_episodes(root)), 0)["test"]
    rng = np.random.default_rng(0)
    starts = [(int(rng.choice(test)), int(rng.integers(200, 5000 - args.horizon - 1)))
              for _ in range(args.starts)]
    curves = {}
    for run, stage in zip(args.run, stages):
        run = Path(run).resolve()
        name = f"{run.name.split('__')[0]} s{stage}"
        agent = load(run, stage, root)
        fn = dream_fn(agent, args.horizon)
        if args.episode is not None:
            ep = dataset.load_episode(dataset.episode_path(root, args.episode))
            wall = dataset.load_episode(root / "layouts" / f"ep{args.episode:04d}.npz")["wall"]
            d = dream(agent, fn, ep, args.t0, args.horizon)
            gif = out / f"dream_{run.name}_s{stage}_ep{args.episode:04d}_t{args.t0}.gif"
            plot_gif(d, wall, meta, f"Dream of {name} (ep{args.episode}, t0={args.t0}): orange = imagined, "
                     f"grey = real", gif)
            print(f"[dream] {gif}")
        if starts:
            es = []
            for e, t0 in starts:
                es.append(errors(dream(agent, fn, dataset.load_episode(dataset.episode_path(root, e)), t0, args.horizon)))
            curves[name] = es
    if curves:
        p = out / f"dream_errors_{'_vs_'.join(n.replace(' ', '') for n in curves)}.png"
        plot_errors(curves, args.horizon, p, f"How far do the dreams hold? {args.starts} random test starts, "
                    f"imagined with the real actions")
        print(f"[dream] {p}")


if __name__ == "__main__":
    main()
