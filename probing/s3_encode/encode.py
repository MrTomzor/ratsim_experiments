"""Teacher-force the shared dataset through checkpoints and keep the state.

    $DREAMER_PYTHON_PATH -m probing.s3_encode.encode --dataset results/probing/pilot/dataset \
        --out results/probing/pilot/states --exp_dir results/experiments/memrssm_orthomaze_ladder \
        --runs runs/L0_baseline__dreamer__seed0 --stages 5
    ... --runs runs/L0_baseline__dreamer__seed0 --random_init

Each episode runs from zero state at step 0 to its last step, the state
carried across the whole episode, through the ACTING code path: the encoder
over the chunk (it has no recurrence), then dyn.observe(single=True) one step
at a time inside a jitted scan -- what agent.policy calls every step.
tests/test_probing.py checks the two agree step for step, for the stock RSSM
and for MemRSSM.

Why not dyn.observe over the whole chunk at once: for the stock RSSM that is
the same per-step scan, but MemRSSM pushes a window of T steps into its
512-slot bank by evicting the T oldest slots, so step t of a 500-step chunk
would read 12 + t past slots where the acting agent reads 512. Encoding a
memory model in chunks would have silently erased most of its memory. The Memory Maze paper does the same at
inference: both RSSM and RSSM (TBTT) carry over the complete trajectory, the
difference between them is only in how they were trained.

prevact at step t is the action recorded at t-1 (zero at t=0), the same
alignment agent.train builds in _apply_replay_context.

The posterior sample is drawn per chunk from --seed. It is a sample, like the
one the acting agent drew; with the same params and inputs the deterministic
path (deter) differs from acting time only through that draw.

Output, one file per (run, stage) or random_init:
    deter    (n, D) float16   feat['deter'] (2048 for the baseline)
    stoch    (n, S) uint8     index of the sampled class per latent group
    classes  ()     int32     classes per group, to rebuild the one-hot
    episode  (n,)   int32     the episode file's id (epXXXX)
    dataset_fingerprint       which version of the dataset this was run on
    t        (n,)   int32     every --stride-th step, aligned with targets.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from probing.core import agent_io, dataset


def observe_fn(agent):
    """Jitted enc + dyn.observe over a chunk, built once per agent: params
    are an argument, so loading another checkpoint reuses the compiled code
    instead of recompiling the 2048-wide scan for every one."""
    cached = getattr(agent, "_probe_observe_fn", None)
    if cached is not None:
        return cached
    import jax
    import jax.numpy as jnp
    import ninjax as nj
    model = agent.model

    def step(dyn_carry, inputs):
        tokens, prevact, reset = inputs
        dyn_carry, _, feat = model.dyn.observe(
            dyn_carry, tokens, prevact, reset, False, single=True)
        return dyn_carry, {
            "deter": feat["deter"],
            "stoch": jnp.argmax(feat["stoch"], -1).astype(jnp.uint8)}

    def observe(carry, obs, prevact):
        enc_carry, dyn_carry = carry
        reset = obs["is_first"]
        enc_carry, _, tokens = model.enc(enc_carry, obs, reset, False)
        dyn_carry, out = nj.scan(step, dyn_carry, (tokens, prevact, reset),
                                 axis=1)
        return (enc_carry, dyn_carry), out

    pure = nj.pure(observe)
    fn = jax.jit(lambda params, carry, obs, pa, seed: pure(
        params, carry, obs, pa, seed=seed, create=False)[1])
    agent._probe_observe_fn = fn
    return fn


def initial_carry(agent, batch: int):
    import jax
    m = agent.model
    make = jax.jit(lambda: (m.enc.initial(batch), m.dyn.initial(batch)))
    return make()


def pad_batch(episodes: list[dict], batch: int, length: int, obs_space: dict,
              act_space: dict):
    """Episodes -> (obs, prevact) arrays of shape (batch, length, ...),
    zero-padded in time and in batch. Padding comes after an episode's last
    step, so it cannot affect any step that is kept."""
    obs = {k: np.zeros((batch, length, *obs_space[k].shape), obs_space[k].dtype)
           for k in (*dataset.OBS_KEYS, "is_first")}
    prevact = {k: np.zeros((batch, length), act_space[k].dtype)
               for k in dataset.ACT_KEYS}
    for b, ep in enumerate(episodes):
        n = len(ep["is_first"])
        for k in dataset.OBS_KEYS:
            obs[k][b, :n] = ep[k]
        obs["is_first"][b, 0] = True
        for k in dataset.ACT_KEYS:
            prevact[k][b, 1:n] = ep[k][:n - 1]
    return obs, prevact


def encode_episodes(agent, episodes: list[dict], obs_space: dict,
                    act_space: dict, stride: int = 10, chunk: int = 500,
                    batch: int = 16, seed: int = 0) -> dict:
    import jax
    fn = observe_fn(agent)
    out = {"deter": [], "stoch": [], "episode": [], "t": []}
    ids = [int(ep.get("episode_id", i)) for i, ep in enumerate(episodes)]
    for b0 in range(0, len(episodes), batch):
        group = episodes[b0:b0 + batch]
        longest = max(len(ep["is_first"]) for ep in group)
        length = -(-longest // chunk) * chunk
        obs, prevact = pad_batch(group, batch, length, obs_space, act_space)
        carry = initial_carry(agent, batch)
        parts = []
        for c in range(0, length, chunk):
            sl = lambda x: jax.device_put(np.ascontiguousarray(x[:, c:c + chunk]))
            key = jax.device_put(np.array([seed, b0 * 100_000 + c], np.uint32))
            carry, res = fn(agent.params, carry,
                            {k: sl(v) for k, v in obs.items()},
                            {k: sl(v) for k, v in prevact.items()}, key)
            parts.append(jax.device_get(res))
        deter = np.concatenate([p["deter"] for p in parts], 1)
        stoch = np.concatenate([p["stoch"] for p in parts], 1)
        for i, ep in enumerate(group):
            ts = np.arange(0, len(ep["is_first"]), stride)
            out["deter"].append(deter[i, ts].astype(np.float16))
            out["stoch"].append(stoch[i, ts])
            out["episode"].append(np.full(len(ts), ids[b0 + i], np.int32))
            out["t"].append(ts.astype(np.int32))
    return {k: np.concatenate(v) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True, help="directory for state files")
    ap.add_argument("--exp_dir", default=None,
                    help="resolve relative --runs against this")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="run dirs sharing one architecture (e.g. the seeds "
                         "of one cell); the first one's config builds the agent")
    ap.add_argument("--stages", nargs="*", type=int, default=[])
    ap.add_argument("--random_init", action="store_true",
                    help="encode with freshly initialised params instead")
    ap.add_argument("--random_init_name", default="random_init",
                    help="output name for --random_init; one per architecture, "
                         "since an untrained memory model can leak past "
                         "observations through its random read")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--platform", default=None, help="override jax.platform")
    args = ap.parse_args()
    if not args.stages and not args.random_init:
        sys.exit("[probe-encode] give --stages, --random_init, or both")

    root = Path(args.dataset)
    out_dir = Path(args.out)
    resolve = lambda r: (Path(r) if Path(r).exists() or not args.exp_dir
                         else Path(args.exp_dir) / r).resolve()
    runs = [resolve(r) for r in args.runs]
    # One agent is built and every checkpoint is loaded into it, so the runs
    # must share the parameter tree. tbtt/replay_context only change how a
    # cell was trained, not its params; these prefixes change the params.
    arch = lambda r: sorted(
        (k, str(v)) for k, v in agent_io.run_config(r).get(
            "method_overrides", {}).items()
        if k.split(".")[0] in ("agent", "gridmem", "memrssm", "probeloss"))
    assert all(arch(r) == arch(runs[0]) for r in runs), \
        "--runs mix architectures; encode one cell per call"

    spaces = dataset.load_json(root / "spaces.json")
    obs_space = dataset.spaces_from_json(spaces["obs"])
    act_space = dataset.spaces_from_json(spaces["act"])
    fp = dataset.verify(root)  # sealed, complete, unchanged
    episodes = [{**dataset.load_episode(p), "episode_id": dataset.episode_id(p)}
                for p in dataset.list_episodes(root)]
    print(f"[probe-encode] {len(episodes)} episodes from {root} ({fp[:12]})")

    config = agent_io.make_config(runs[0], platform=args.platform)
    agent = agent_io.build_agent(config, obs_space, act_space)
    kw = dict(stride=args.stride, chunk=args.chunk, batch=args.batch,
              seed=args.seed)

    classes = np.int32(config.agent.dyn.rssm.classes)

    def save(name, res, **attrs):
        dataset.save_npz(out_dir / f"{name}.npz",
                         {**res, **attrs, "classes": classes,
                          "dataset_fingerprint": np.str_(fp)})
        print(f"[probe-encode] {name}: {len(res['t'])} states, "
              f"deter {res['deter'].shape[1]}, stoch {res['stoch'].shape[1]}")

    if args.random_init:  # before any load: params are still the init draw
        res = encode_episodes(agent, episodes, obs_space, act_space, **kw)
        save(args.random_init_name, res, run=np.str_(runs[0].name),
             stage=np.int32(-1), random_init=np.bool_(True))
    for run in runs:
        for stage in args.stages:
            agent_io.load_checkpoint(agent, agent_io.stage_checkpoint(run, stage))
            res = encode_episodes(agent, episodes, obs_space, act_space, **kw)
            save(f"{run.name}__stage{stage}", res, run=np.str_(run.name),
                 stage=np.int32(stage), random_init=np.bool_(False))


if __name__ == "__main__":
    main()
