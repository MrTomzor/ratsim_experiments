"""Tests for probing/ that need the dreamer venv (JAX, ratsim, no scipy/torch).

    ~/ratvenv/dreamer_venv/bin/python tests/test_probing.py

The one that matters is test_offline_encode_is_the_acting_state: the whole
probing protocol rests on encode.py reproducing the state the agent had while
acting. It checks that with the posterior draw made deterministic, so any
difference is plumbing -- prevact alignment, reset handling, which obs keys
reach the encoder, the carry across chunks -- and not sampling noise.
"""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import elements  # noqa: E402
import jax  # noqa: E402

from probing.core import agent_io, dataset  # noqa: E402
from probing.s3_encode.encode import encode_episodes  # noqa: E402
from probing.s1_collect.explorer import (FORWARD, OCCUPIED, FREE, ScriptedExplorer,  # noqa: E402
                              bfs)
import train_dreamerv3 as td  # noqa: E402

LIDAR, T = 12, 11
DT = 0.1  # seconds per env step


def small_config():
    """A tiny stock-DreamerV3 config on CPU: just enough agent to encode with."""
    tmp = Path(tempfile.mkdtemp(prefix="probing_test_"))
    overrides = {"replay_context": 0, "batch_size": 2, "batch_length": 6,
                 "report_length": 6, "replay.size": 20_000, "jax.platform": "cpu",
                 "jax.prealloc": False, "run.envs": 1}
    config = td.build_config(overrides, tmp, total_steps=1000, size="size1m")
    return config.update({"agent": {
        r".*\.rssm": {"deter": 32, "hidden": 16, "stoch": 4, "classes": 4,
                      "blocks": 4},
        r".*\.units": 16,
        r".*\.depth": 2,
        r".*\.layers": 1,
        r".*\.bins": 5,
    }})


def spaces():
    obs = {
        "lidar": elements.Space(np.float32, (LIDAR,), 0.0, 1.0),
        "gps": elements.Space(np.float32, (2,), -1.0, 1.0),
        "compass": elements.Space(np.float32, (1,), -1.0, 1.0),
        "reward": elements.Space(np.float32),
        "is_first": elements.Space(bool),
        "is_last": elements.Space(bool),
        "is_terminal": elements.Space(bool),
    }
    act = {"action_0": elements.Space(np.int32, (), 0, 3),
           "action_1": elements.Space(np.int32, (), 0, 5)}
    return obs, act


def _deterministic_posterior():
    """Swap the categorical draw for its mode; returns the undo."""
    from embodied.jax import outs
    orig = outs.OneHot.sample
    outs.OneHot.sample = lambda self, seed, shape=(): self.pred()
    return lambda: setattr(outs.OneHot, "sample", orig)


def _act_and_record(agent, obs_space, rng, steps=T):
    """Run agent.policy step by step, as collect.py does, recording the
    episode in dataset format and the carry after every step."""
    carry = agent.init_policy(1)
    ep = {k: [] for k in (*dataset.OBS_KEYS, *dataset.ACT_KEYS, "is_first")}
    deter, stoch = [], []
    for t in range(steps):
        obs = {"lidar": rng.random(LIDAR).astype(np.float32),
               "gps": rng.uniform(-1, 1, 2).astype(np.float32),
               "compass": rng.uniform(-1, 1, 1).astype(np.float32),
               "reward": np.float32(0), "is_first": t == 0,
               "is_last": False, "is_terminal": False}
        obs_b = {k: np.asarray(v, obs_space[k].dtype)[None]
                 for k, v in obs.items()}
        carry, act, _ = agent.policy(carry, obs_b, mode="eval")
        dyn = jax.device_get(carry[1])
        deter.append(np.asarray(dyn["deter"][0]))
        stoch.append(np.asarray(dyn["stoch"][0]).argmax(-1))
        for k in dataset.OBS_KEYS:
            ep[k].append(obs[k])
        ep["is_first"].append(t == 0)
        for k in dataset.ACT_KEYS:
            ep[k].append(int(np.asarray(act[k])[0]))
    ep = {k: np.asarray(v) for k, v in ep.items()}
    for k in dataset.ACT_KEYS:
        ep[k][-1] = 0  # collect.py: no action after the last observation
        ep[k] = ep[k].astype(np.int32)
    return ep, np.stack(deter), np.stack(stoch)


def test_offline_encode_is_the_acting_state():
    undo = _deterministic_posterior()
    try:
        obs_space, act_space = spaces()
        agent = agent_io.build_agent(small_config(),
                                     obs_space, act_space)
        ep, deter, stoch = _act_and_record(agent, obs_space,
                                           np.random.default_rng(0))
        # chunk 4 over 11 steps = 3 chunks with padding, batch 2 with one
        # padded row: the carry hand-off and both paddings are exercised.
        res = encode_episodes(agent, [ep], obs_space, act_space,
                              stride=1, chunk=4, batch=2)
        assert (res["t"] == np.arange(T)).all()
        np.testing.assert_allclose(res["deter"].astype(np.float32), deter,
                                   rtol=2e-3, atol=2e-3)
        assert (res["stoch"] == stoch).all()

        # Sensitivity: the same episode with prevact off by one step must
        # NOT reproduce the acting state, or the check above proves nothing.
        bad = dict(ep)
        for k in dataset.ACT_KEYS:
            bad[k] = np.roll(ep[k], -1)
        wrong = encode_episodes(agent, [bad], obs_space, act_space,
                                stride=1, chunk=4, batch=2)
        assert not np.allclose(wrong["deter"].astype(np.float32), deter,
                               atol=2e-3), "deter ignores the actions"
    finally:
        undo()


def _window_encode(agent, ep, obs_space, act_space, chunk):
    """The OLD encode: dyn.observe over a whole chunk at once."""
    import ninjax as nj
    from probing.s3_encode.encode import initial_carry, pad_batch
    model = agent.model

    def observe(carry, obs, prevact):
        enc_carry, dyn_carry = carry
        enc_carry, _, tokens = model.enc(enc_carry, obs, obs["is_first"], False)
        dyn_carry, _, feat = model.dyn.observe(
            dyn_carry, tokens, prevact, obs["is_first"], False)
        return (enc_carry, dyn_carry), feat["deter"]

    pure = nj.pure(observe)
    fn = jax.jit(lambda p, c, o, a, s: pure(p, c, o, a, seed=s, create=False)[1])
    n = len(ep["is_first"])
    length = -(-n // chunk) * chunk
    obs, prevact = pad_batch([ep], 1, length, obs_space, act_space)
    carry, out = initial_carry(agent, 1), []
    for c in range(0, length, chunk):
        sl = lambda x: jax.device_put(np.ascontiguousarray(x[:, c:c + chunk]))
        carry, d = fn(agent.params, carry, {k: sl(v) for k, v in obs.items()},
                      {k: sl(v) for k, v in prevact.items()},
                      jax.device_put(np.array([0, c], np.uint32)))
        out.append(jax.device_get(d))
    return np.concatenate(out, 1)[0, :n]


def _acting_feats(agent, ep):
    """feat at every step, computed the way agent.policy computes it: encoder
    and dyn.observe with single=True, one call per step, carry handed on.
    (agent.policy itself returns only the carry, and for MemRSSM the carry
    lacks the memory read, which goes to the heads and not into the
    recurrence -- so the read is exactly what a carry comparison misses.)"""
    import ninjax as nj
    from probing.s3_encode.encode import initial_carry
    model = agent.model

    def step(carry, obs, prevact):
        enc_carry, dyn_carry = carry
        enc_carry, _, tokens = model.enc(enc_carry, obs, obs["is_first"], False,
                                         single=True)
        dyn_carry, _, feat = model.dyn.observe(
            dyn_carry, tokens, prevact, obs["is_first"], False, single=True)
        return (enc_carry, dyn_carry), feat

    pure = nj.pure(step)
    fn = jax.jit(lambda p, c, o, a, s: pure(p, c, o, a, seed=s, create=False)[1])
    carry, feats = initial_carry(agent, 1), []
    for t in range(len(ep["is_first"])):
        obs = {k: jax.device_put(np.asarray(ep[k][t])[None])
               for k in (*dataset.OBS_KEYS, "is_first")}
        pa = {k: jax.device_put(np.asarray(ep[k][t - 1] if t else 0, np.int32)[None])
              for k in dataset.ACT_KEYS}
        carry, f = fn(agent.params, carry, obs, pa,
                      jax.device_put(np.array([0, t], np.uint32)))
        feats.append(jax.device_get(f))
    return {k: np.concatenate([f[k] for f in feats]) for k in ("deter", "stoch")}


def test_offline_encode_is_the_acting_state_for_the_memory_model():
    """MemRSSM keeps a FIFO bank of past steps. Pushing a whole chunk evicts
    as many old slots as the chunk is long, so chunked observe reads fewer
    past steps than the acting agent. The encoder must follow the acting
    path; the old chunked path must be shown to differ, or this test would
    not guard anything."""
    try:  # the memory model is a ratsim_metacentrum variant, absent here
        from methods.dreamerv3 import memrssm
        from test_memrssm_agent import DETER, MEMSIZE, small_config as mem_config
    except ImportError:
        print("SKIP test_offline_encode_is_the_acting_state_for_the_memory_model: "
              "no methods/dreamerv3/memrssm in this tree")
        return
    steps = 3 * MEMSIZE  # the bank fills and evicts
    undo = _deterministic_posterior()
    try:
        config = mem_config()
        with memrssm.installed(config):
            obs_space, act_space = spaces()
            agent = agent_io.build_agent(config, obs_space, act_space)
            ep, carry_deter, stoch = _act_and_record(
                agent, obs_space, np.random.default_rng(3), steps)
            ref = _acting_feats(agent, ep)
            # The reference IS the policy's computation: its recurrent part is
            # the carry agent.policy handed on, step for step.
            np.testing.assert_allclose(ref["deter"][:, :DETER], carry_deter,
                                       rtol=1e-5, atol=1e-5)
            assert ref["deter"].shape[1] > DETER, "no memory read in feat"
            res = encode_episodes(agent, [ep], obs_space, act_space,
                                  stride=1, chunk=8, batch=2)
            np.testing.assert_allclose(res["deter"].astype(np.float32),
                                       ref["deter"], rtol=2e-3, atol=2e-3)
            assert (res["stoch"] == ref["stoch"].argmax(-1)).all()
            old = _window_encode(agent, ep, obs_space, act_space, chunk=8)
            np.testing.assert_allclose(old[:, :DETER], ref["deter"][:, :DETER],
                                       rtol=2e-3, atol=2e-3)  # recurrence unaffected
            assert not np.allclose(old[MEMSIZE:, DETER:], ref["deter"][MEMSIZE:, DETER:],
                                   atol=2e-3), \
                "chunked observe matched acting: the bank never evicted"
    finally:
        undo()


def test_stride_keeps_aligned_steps():
    obs_space, act_space = spaces()
    agent = agent_io.build_agent(small_config(),
                                 obs_space, act_space)
    rng = np.random.default_rng(1)
    eps = []
    for n in (23, 7):
        eps.append({"lidar": rng.random((n, LIDAR)).astype(np.float32),
                    "gps": rng.uniform(-1, 1, (n, 2)).astype(np.float32),
                    "compass": rng.uniform(-1, 1, (n, 1)).astype(np.float32),
                    "is_first": np.arange(n) == 0,
                    "action_0": rng.integers(0, 3, n).astype(np.int32),
                    "action_1": rng.integers(0, 5, n).astype(np.int32)})
    res = encode_episodes(agent, eps, obs_space, act_space, stride=5,
                          chunk=8, batch=4)
    assert res["episode"].tolist() == [0] * 5 + [1] * 2
    assert res["t"].tolist() == [0, 5, 10, 15, 20, 0, 5]
    assert res["deter"].dtype == np.float16 and res["stoch"].dtype == np.uint8


def test_episode_files_need_no_pickle():
    with tempfile.TemporaryDirectory() as tmp:
        path = dataset.episode_path(tmp, 0)
        dataset.save_episode(path, {"pose": np.zeros((3, 3), np.float32),
                                    "source": np.str_("explorer"),
                                    "world_seed": np.int64(12345)})
        ep = dataset.load_episode(path)  # allow_pickle=False inside
        assert str(ep["source"]) == "explorer" and int(ep["world_seed"]) == 12345
        try:
            dataset.save_episode(path, {"bad": np.array([{}], object)})
        except TypeError:
            pass
        else:
            raise AssertionError("object array was written")


def test_plan_reproduces_the_pilots_worlds():
    """The planned seeds are the ones env.py drew in pilot 1 (metaseed 777,
    one warm-up draw for the constructor's reset), read off its log."""
    from probing.s1_collect.collect import plan_episodes
    plan = plan_episodes([("policy", "a", 5, 5), ("policy", "b", 5, 5),
                          ("explorer", "", -1, 10)], 777)
    assert [p["seed"] for p in plan[:2]] == [1312314220, 757058591]
    assert plan[5]["seed"] == 975065321 and plan[5]["run"] == "b"
    assert plan[10]["seed"] == 723790941 and plan[10]["kind"] == "explorer"
    assert plan[19]["seed"] == 2031875823 and [p["idx"] for p in plan] == list(range(20))


def test_workers_share_out_exactly_the_missing_episodes():
    from probing.s1_collect.collect import pending, plan_episodes, SeedQueue
    plan = plan_episodes([("explorer", "", -1, 9)], 1)
    with tempfile.TemporaryDirectory() as tmp:
        for i in (0, 4):
            dataset.save_episode(dataset.episode_path(tmp, i), {"x": np.zeros(1)})
        todo = pending(Path(tmp), plan)
        assert [p["idx"] for p in todo] == [1, 2, 3, 5, 6, 7, 8]
        shards = [todo[k::3] for k in range(3)]
        assert sorted(p["idx"] for s in shards for p in s) == [1, 2, 3, 5, 6, 7, 8]
    q = SeedQueue(p["seed"] for p in shards[1])
    assert [q.integers(0, 2**31 - 1) for _ in shards[1]] == [p["seed"] for p in shards[1]]
    try:
        q.integers(0, 2**31 - 1)
    except RuntimeError:
        pass
    else:
        raise AssertionError("an unplanned reset got a seed")


def test_json_is_never_seen_half_written():
    """Readers racing a writer (4 collect workers on spaces.json) must see
    the old or the new content, never an empty or partial file."""
    import threading
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spaces.json"
        big = {"k": list(range(20000))}
        dataset.save_json(path, big)
        errors, stop = [], threading.Event()

        def read():
            while not stop.is_set():
                try:
                    assert dataset.load_json(path) == big
                except Exception as e:  # noqa: BLE001
                    errors.append(e)

        readers = [threading.Thread(target=read) for _ in range(3)]
        for r in readers:
            r.start()
        for _ in range(200):
            dataset.save_json(path, big)
        stop.set()
        for r in readers:
            r.join()
        assert not errors, errors[:1]
        assert [p.name for p in Path(tmp).iterdir()] == ["spaces.json"]


def test_spaces_round_trip():
    obs_space, act_space = spaces()
    back = dataset.spaces_from_json(dataset.spaces_to_json(
        {**obs_space, **act_space}))
    for k, s in {**obs_space, **act_space}.items():
        assert back[k].dtype == s.dtype and back[k].shape == s.shape, k
        assert np.array_equal(back[k].low, s.low), k
        assert np.array_equal(back[k].high, s.high), k


def test_split_is_by_episode_and_disjoint():
    sp = dataset.split_episodes(20, seed=0)
    allidx = np.concatenate(list(sp.values()))
    assert sorted(allidx.tolist()) == list(range(20))
    assert len(sp["train"]) == 14 and len(sp["val"]) == 3 and len(sp["test"]) == 3


class _Grid:
    """The ExplorationTracker surface the explorer uses, on a fixed map."""

    def __init__(self, grid, origin=(-30.0, -30.0), res=1.0):
        self.grid, self.resolution = grid, res
        self.origin_x, self.origin_y = origin

    def world_to_cell(self, x, y):
        return (int((x - self.origin_x) / self.resolution),
                int((y - self.origin_y) / self.resolution))


def _wall_map():
    grid = np.full((60, 60), FREE, np.int8)
    grid[:, 30] = OCCUPIED       # a wall at x = 0 ...
    grid[8:12, 30] = FREE        # ... with one gap near y = -20
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = OCCUPIED
    return _Grid(grid)


def test_bfs_routes_through_the_gap():
    m = _wall_map()
    dist, _ = bfs(m.grid == FREE, (30, 15))
    goal = 30 * 60 + 45
    assert dist[goal] > 30, "BFS went through the wall"


def test_explorer_drives_to_a_goal_around_a_wall():
    """Kinematics as measured in the pilot dataset: velocity set directly,
    +angular_z = counter-clockwise = yaw grows, one env step = 0.1 s (1.0 m
    forward, 0.149 rad of turn per step)."""
    m = _wall_map()
    ex = ScriptedExplorer(seed=0)
    x, y, yaw = -15.0, 0.0, 0.0
    goal = (15.0, 0.0)
    lin, ang = (-1.0, 0.0, 1.0), (-1.0, -0.5, 0.0, 0.5, 1.0)
    for step in range(600):
        a0, a1 = ex.act(x, y, yaw, m, goal_xy=goal)
        v, w = lin[a0] * 10.0, ang[a1] * 1.5
        yaw = math.atan2(math.sin(yaw + w * DT), math.cos(yaw + w * DT))
        x, y = x + v * math.cos(yaw) * DT, y + v * math.sin(yaw) * DT
        c, r = m.world_to_cell(x, y)
        assert m.grid[r, c] != OCCUPIED, f"drove into a wall at step {step}"
        if math.hypot(x - goal[0], y - goal[1]) < 1.5:
            break
    else:
        raise AssertionError(f"never reached the goal, ended at {x:.1f},{y:.1f}")


def test_explorer_finds_a_way_out_of_a_real_corridor():
    """Regression, pilot episode ep0010 at step 79: the seen map shows the
    corridor as one free column and the agent's own cell as OCCUPIED. With
    walls inflated by one cell the reachable set was 1 cell and the explorer
    spun in place for the rest of the episode."""
    f = np.load(Path(__file__).parent / "data" / "explorer_ep0010_t79.npz")
    m = _Grid(f["grid"], origin=tuple(f["origin"]), res=float(f["resolution"]))
    ex = ScriptedExplorer(seed=0)
    ex.visited = {tuple(rc) for rc in f["visited"].tolist()}
    x, y, _ = (float(v) for v in f["pose"])
    c, r = m.world_to_cell(x, y)
    assert m.grid[r, c] == OCCUPIED, "fixture no longer shows the agent on a wall"
    assert ex.plan(m.grid, (r, c)), "no plan from the pilot's stuck state"
    assert len(ex.path) > 5
    path = np.array(ex.path)
    on_wall = m.grid[path[:, 0], path[:, 1]] == OCCUPIED
    visited = np.array([tuple(p) in ex.visited for p in ex.path])
    assert not (on_wall & ~visited).any(), "planned through a wall not driven"


def test_explorer_turns_the_short_way():
    left = ScriptedExplorer.steer(0, 0, 0.0, 0.0, 5.0)    # target at +90 deg
    right = ScriptedExplorer.steer(0, 0, 0.0, 0.0, -5.0)  # target at -90 deg
    ahead = ScriptedExplorer.steer(0, 0, 0.0, 5.0, 0.0)
    assert left[1] > 2 and right[1] < 2, (left, right)
    assert ahead == (FORWARD, 2)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=6)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
