"""JAX-level tests for Dreamer + TBTT: does the carried RSSM state actually flow
through `agent.train` / `agent.report`, and is `is_first` the only thing that
resets it?

Runs on CPU with a tiny model (a few seconds of jit each). Run with either:

    ~/ratvenv/dreamer_venv/bin/python -m pytest tests/test_tbtt_agent.py -q
    ~/ratvenv/dreamer_venv/bin/python tests/test_tbtt_agent.py

What is checked, and why it settles the "is TBTT implemented right" question:

1. Reset semantics (exact): the training loss on chunk 2 with the carry from
   chunk 1 differs from the zero-carry loss, but once `is_first[:, 0]` is
   forced True both are identical. So the carry is consumed, and the
   is_first mask is the only reset -- which is also why the stream must not
   force is_first at the top of every window.
2. Carry hand-off (exact, RSSM level): the carry a chunk returns is its last
   per-step state, and feeding the single-pass state at L-1 as the carry of
   chunk 2 reproduces the single pass's deterministic state (`deter`) and
   posterior logits at step L. Stochastic latents are re-sampled per step so
   the two runs cannot share samples beyond that, but this is exactly what a
   correct truncated-BPTT hand-off must preserve.
3. End to end: the real `agent.train` consumes batches from the sequential
   stream (K=0 and K=16), produces finite metrics, and the K=16 path returns
   replay updates that `replay.update` accepts.
4. Plumbing: `build_config` forces consec_train=1, `make_stream_ratsim` returns
   the sequential stream for training and the stock stream for reports, and
   `make_replay` sizes windows as L + K.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import elements  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import ninjax as nj  # noqa: E402
from embodied.core.replay import Replay  # noqa: E402
from embodied.jax import internal  # noqa: E402

import train_dreamerv3 as td  # noqa: E402
from methods.dreamerv3.tbtt_stream import SequentialReplayStream  # noqa: E402

B, L = 4, 8
OBS_DIM, ACT_DIM = 6, 2

_CACHE: dict = {}


def small_config(K: int, extra: dict | None = None):
    tmp = Path(tempfile.mkdtemp(prefix="tbtt_test_"))
    overrides = {
        "tbtt.enabled": True,
        "consec_train": 4,           # must get forced back to 1
        "replay_context": K,
        "batch_size": B,
        "batch_length": L,
        "report_length": L,
        "replay.size": 20_000,
        "jax.platform": "cpu",
        "jax.prealloc": False,
        "run.envs": 1,
        **(extra or {}),
    }
    config = td.build_config(overrides, tmp, total_steps=1000, size="size1m")
    # Shrink the model the way upstream's `debug` preset does.
    config = config.update({"agent": {
        r".*\.rssm": {"deter": 32, "hidden": 16, "stoch": 4, "classes": 4, "blocks": 4},
        r".*\.units": 16,
        r".*\.depth": 2,
        r".*\.layers": 1,
        r".*\.bins": 5,
    }})
    return config


def spaces():
    obs_space = {
        "vector": elements.Space(np.float32, (OBS_DIM,)),
        "reward": elements.Space(np.float32),
        "is_first": elements.Space(bool),
        "is_last": elements.Space(bool),
        "is_terminal": elements.Space(bool),
    }
    act_space = {"action": elements.Space(np.float32, (ACT_DIM,), -1.0, 1.0)}
    return obs_space, act_space


def make_agent(config):
    from dreamerv3.agent import Agent
    obs_space, act_space = spaces()
    return Agent(obs_space, act_space, elements.Config(
        **config.agent,
        logdir=config.logdir,
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    ))


def agent_for(K: int):
    """One agent per K, cached: jit compile dominates test time."""
    if K not in _CACHE:
        config = small_config(K)
        _CACHE[K] = (config, make_agent(config))
    return _CACHE[K]


def fill_replay(agent, K, n_workers=2, n_episodes=6, ep_len=40, seed=0):
    rng = np.random.default_rng(seed)
    replay = Replay(length=L + K, capacity=None, chunksize=64, seed=seed)
    for ep in range(n_episodes):
        for w in range(n_workers):
            for t in range(ep_len):
                step = {}
                for k, sp in agent.spaces.items():
                    if k in ("consec", "stepid"):
                        continue  # stream / replay add these
                    step[k] = np.zeros(sp.shape, sp.dtype)
                step["vector"] = rng.standard_normal(OBS_DIM).astype(np.float32)
                step["action"] = rng.uniform(-1, 1, ACT_DIM).astype(np.float32)
                step["reward"] = np.float32(rng.standard_normal())
                step["is_first"] = np.bool_(t == 0)
                step["is_last"] = np.bool_(t == ep_len - 1)
                step["is_terminal"] = np.bool_(False)
                replay.add(step, worker=w)
    return replay


def with_seed(agent, data, counter):
    # Mirrors agent.stream(): explicit device_put (the jax transfer guard is
    # set to 'disallow', so implicit host->device transfers raise).
    data = internal.device_put(dict(data), agent.train_sharded)
    return {**data, "seed": agent._seeds(counter, agent.train_mirrored)}


def scalar_metrics(mets):
    out = {}
    for k, v in mets.items():
        if k == "params/summary":
            continue
        v = np.asarray(v)
        if np.issubdtype(v.dtype, np.floating) and v.ndim == 0:
            out[k] = float(v)
    return out


def to_np(tree):
    return jax.device_get(tree)  # explicit device->host, allowed by the guard


def to_dev(tree):
    return jax.tree.map(jax.device_put, tree)


SEED0 = None


def seed0():
    global SEED0
    if SEED0 is None:
        SEED0 = jax.device_put(np.array([0, 0], np.uint32))
    return SEED0


# ---------------------------------------------------------------------------
# Pure, jitted views of the model. Eager calls trip the jax transfer guard on
# Python constants, and upstream's `report` only returns image metrics (its
# `mets.update(mets)` is a no-op), so both tests go through `model.loss` /
# `model.dyn.observe` directly, exactly as `agent.train` does inside its jit.

def _init_carry(agent, with_dec=True):
    # Under jit: eager jnp.zeros transfers its fill value host->device, which
    # the transfer guard forbids.
    m = agent.model
    if with_dec:
        return jax.jit(lambda: (m.enc.initial(B), m.dyn.initial(B), m.dec.initial(B)))()
    return jax.jit(lambda: (m.enc.initial(B), m.dyn.initial(B)))()


def _loss_fn(agent):
    model = agent.model

    def lossfn(carry, obs, prevact):
        loss, (carry, _, outs, metrics) = model.loss(carry, obs, prevact, False)
        return loss, carry, {k: v.mean() for k, v in outs["losses"].items()}

    pure = nj.pure(lossfn)
    return jax.jit(lambda params, carry, obs, pa, seed: pure(
        params, carry, obs, pa, seed=seed, create=False)[1])


def _observe_fn(agent):
    model = agent.model

    def observe(carry, obs, prevact):
        enc_carry, dyn_carry = carry
        reset = obs["is_first"]
        enc_carry, _, tokens = model.enc(enc_carry, obs, reset, False)
        dyn_carry, entries, feat = model.dyn.observe(
            dyn_carry, tokens, prevact, reset, False)
        return (enc_carry, dyn_carry), entries, feat

    pure = nj.pure(observe)
    return jax.jit(lambda params, carry, obs, pa, seed: pure(
        params, carry, obs, pa, seed=seed, create=False)[1])


def _split_batch(agent, data, prev_last_act):
    """Mirror Agent._apply_replay_context for K=0: obs dict + prevact shifted
    by one, with the previous chunk's last action in front."""
    obs = {k: data[k] for k in agent.obs_space}
    act = data["action"]
    prevact = {"action": np.concatenate([prev_last_act[:, None], act[:, :-1]], 1)}
    return to_dev(obs), to_dev(prevact)


def _force_first(obs):
    obs = dict(obs)
    is_first = to_np(obs["is_first"]).copy()
    is_first[:, 0] = True
    obs["is_first"] = jax.device_put(is_first)
    return obs


# ---------------------------------------------------------------------------
# 1. Reset semantics through the training loss (exact)

def test_loss_carry_matters_and_is_first_is_the_only_reset():
    K = 0
    config, agent = agent_for(K)
    replay = fill_replay(agent, K)
    stream = SequentialReplayStream(replay, B, L, prefix=K, seed=1)
    b1 = next(stream)
    b2 = next(stream)
    assert (b2["consec"][:, 0] == 1).all(), "expected no restarts on a fresh full buffer"
    assert not b2["is_first"][:, 0].any(), "stream must not force is_first on carried rows"

    fn = _loss_fn(agent)
    params, seed = agent.params, seed0()
    zero = _init_carry(agent)
    obs1, pa1 = _split_batch(agent, b1, np.zeros((B, ACT_DIM), np.float32))
    _, carry1, _ = fn(params, zero, obs1, pa1, seed)
    obs2, pa2 = _split_batch(agent, b2, b1["action"][:, -1])

    l_carried, _, m_carried = to_np(fn(params, carry1, obs2, pa2, seed))
    l_zero, _, m_zero = to_np(fn(params, zero, obs2, pa2, seed))
    assert m_carried, "no per-key losses"
    assert not np.isclose(l_carried, l_zero), (l_carried, l_zero)
    assert any(not np.isclose(m_carried[k], m_zero[k]) for k in m_carried), m_carried

    # Same chunk with is_first forced at position 0: the RSSM masks the carry
    # (and prevact), so carried and zero must agree exactly.
    obs2f = _force_first(obs2)
    l_cf, _, m_cf = to_np(fn(params, carry1, obs2f, pa2, seed))
    l_zf, _, m_zf = to_np(fn(params, zero, obs2f, pa2, seed))
    assert np.allclose(l_cf, l_zf, rtol=1e-6, atol=1e-7), (l_cf, l_zf)
    for k in m_cf:
        assert np.allclose(m_cf[k], m_zf[k], rtol=1e-6, atol=1e-7), (k, m_cf[k], m_zf[k])
    # ... and forcing is_first changed the carried result, i.e. a stream that
    # forced is_first per window would have erased the carry.
    assert not np.isclose(l_carried, l_cf), (l_carried, l_cf)


# ---------------------------------------------------------------------------
# 2. Exact carry hand-off at the RSSM level

def test_rssm_carry_handoff_matches_single_pass():
    K = 0
    config, agent = agent_for(K)
    replay = fill_replay(agent, K, seed=3)
    stream = SequentialReplayStream(replay, B, L, prefix=K, seed=2)
    b1, b2 = next(stream), next(stream)
    carried_rows = b2["consec"][:, 0] == 1
    assert carried_rows.sum() >= 2

    obs_keys = list(agent.obs_space)
    full = {k: np.concatenate([b1[k], b2[k]], 1) for k in obs_keys + ["action"]}
    acts = full["action"]
    prevact_full = np.concatenate([np.zeros_like(acts[:, :1]), acts[:, :-1]], 1)
    obs_full = to_dev({k: full[k] for k in obs_keys})
    obs_1 = to_dev({k: full[k][:, :L] for k in obs_keys})
    obs_2 = to_dev({k: full[k][:, L:] for k in obs_keys})
    pa_full = to_dev({"action": prevact_full})
    pa_1 = to_dev({"action": prevact_full[:, :L]})
    pa_2 = to_dev({"action": prevact_full[:, L:]})

    fn = _observe_fn(agent)
    params, seed = agent.params, seed0()
    init = _init_carry(agent, with_dec=False)

    # (a) The carry returned by a chunk is exactly its last per-step entry.
    carry_mid, ent_1, _ = fn(params, init, obs_1, pa_1, seed)
    for k in ("deter", "stoch"):
        assert np.array_equal(to_np(carry_mid[1][k]), to_np(ent_1[k])[:, -1]), k

    # (b) Feeding an in-sequence state as the carry reproduces the next step of
    # the single pass exactly. Stochastic latents are re-sampled per step with
    # per-call seeds, so the two runs cannot share samples; the deterministic
    # transition deter_L = f(deter_{L-1}, stoch_{L-1}, a_L) and the posterior
    # logits at L are the exact quantities a correct hand-off must preserve.
    _, ent_full, feat_full = fn(params, init, obs_full, pa_full, seed)
    ent_full_np = to_np(ent_full)
    carry_from_full = (carry_mid[0], to_dev({
        "deter": ent_full_np["deter"][:, L - 1],
        "stoch": ent_full_np["stoch"][:, L - 1],
    }))
    _, ent_2, feat_2 = fn(params, carry_from_full, obs_2, pa_2, seed)

    d_full = ent_full_np["deter"][:, L]
    d_2 = to_np(ent_2["deter"])[:, 0]
    lg_full = to_np(feat_full["logit"])[:, L]
    lg_2 = to_np(feat_2["logit"])[:, 0]
    assert np.allclose(d_full, d_2, rtol=1e-5, atol=1e-6), np.abs(d_full - d_2).max()
    assert np.allclose(lg_full, lg_2, rtol=1e-5, atol=1e-6), np.abs(lg_full - lg_2).max()

    # (c) Control: force a reset at the top of chunk 2 -> state no longer
    # matches the single pass on the carried rows ...
    obs_2f = _force_first(obs_2)
    _, ent_2f, _ = fn(params, carry_from_full, obs_2f, pa_2, seed)
    d_2f = to_np(ent_2f["deter"])[:, 0]
    assert not np.allclose(d_full[carried_rows], d_2f[carried_rows], atol=1e-4)
    # ... and equals the zero-carry result exactly (is_first is the reset).
    _, ent_2z, _ = fn(params, init, obs_2f, pa_2, seed)
    assert np.allclose(d_2f, to_np(ent_2z["deter"])[:, 0], rtol=1e-6, atol=1e-7)


# ---------------------------------------------------------------------------
# 3. End to end through agent.train

def _train_some(K, n_batches=5):
    config, agent = agent_for(K)
    replay = fill_replay(agent, K, seed=5)
    stream = SequentialReplayStream(replay, B, L, prefix=K, seed=4)
    it = iter(agent.stream(stream))
    carry = agent.init_train(B)
    n_updates_before = replay.metrics["updates"]
    all_mets = []
    for _ in range(n_batches):
        carry, outs, mets = agent.train(carry, next(it))
        if "replay" in outs:
            replay.update(outs["replay"])
        all_mets.append(mets)
    # Metrics are returned one step late (async fetch); the last two are real.
    finite = [m for m in all_mets if m]
    assert finite, "no metrics came back"
    for m in finite:
        for k, v in scalar_metrics(m).items():
            assert np.isfinite(v), (k, v)
    dyn_carry = to_np(carry[1])
    assert np.abs(dyn_carry["deter"]).max() > 0, "carry never left the zero state"
    return replay.metrics["updates"] - n_updates_before, stream.stats()


def test_train_end_to_end_k0():
    n_updates, st = _train_some(0)
    assert n_updates == 0, "K=0 stores no latents, so no replay updates expected"
    # agent.stream() wraps the source in a Prefetch (one batch ahead).
    assert st["tbtt_batches"] >= 5, st
    assert st["tbtt_stretch_max"] >= 5 * L, st


def test_train_end_to_end_k16_writes_latents_back():
    K = 16
    n_updates, st = _train_some(K)
    assert n_updates > 0, "K>0 path must write refreshed latents into the replay"
    assert st["tbtt_batches"] >= 5, st


# ---------------------------------------------------------------------------
# 4. Plumbing

def test_build_config_forces_consec_and_stream_factory_dispatch():
    from dreamerv3.main import make_replay
    from embodied.core.streams import Consec, Stateless

    K = 0
    config, _ = agent_for(K)
    assert config.tbtt.enabled
    assert config.consec_train == 1, config.consec_train
    assert config.replay_context == K

    replay = make_replay(config, "replay")
    assert replay.length == L + K, replay.length
    train_stream = td.make_stream_ratsim(config, replay, "train")
    assert isinstance(train_stream, SequentialReplayStream)
    report_stream = td.make_stream_ratsim(config, replay, "report")
    assert isinstance(report_stream, Consec)
    assert "tbtt_restart_frac" in replay.stats()

    off = small_config(16, {"tbtt.enabled": False, "consec_train": 4})
    assert off.consec_train == 4
    replay_off = make_replay(off, "replay")
    assert replay_off.length == 4 * L + 16
    assert isinstance(td.make_stream_ratsim(off, replay_off, "train"), Consec)
    assert not isinstance(td.make_stream_ratsim(off, replay_off, "train"),
                          SequentialReplayStream)
    del Stateless


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            traceback.print_exc()
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
