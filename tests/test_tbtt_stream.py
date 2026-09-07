"""Unit tests for methods/dreamerv3/tbtt_stream.py against a real embodied Replay.

No JAX needed. Run with either:

    ~/ratvenv/dreamer_venv/bin/python -m pytest tests/test_tbtt_stream.py -q
    ~/ratvenv/dreamer_venv/bin/python tests/test_tbtt_stream.py

Synthetic steps encode (worker, episode, t, marker) in the observation so every
assertion below is about *which* steps came back, not just their shapes.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embodied.core.replay import Replay  # noqa: E402

from methods.dreamerv3.tbtt_stream import SequentialReplayStream, attach_stats  # noqa: E402

MARKER = 12345.0
L = 8      # batch_length
B = 4      # batch_size


def make_step(worker, episode, t, ep_len, abandoned=False):
    return {
        "obs": np.array([worker, episode, t, MARKER], np.float32),
        "action": np.array([worker + 0.5, t], np.float32),
        "reward": np.float32(0.0),
        "is_first": np.bool_(t == 0),
        "is_last": np.bool_(t == ep_len - 1 and not abandoned),
        "is_terminal": np.bool_(False),
    }


def fill(replay, n_workers, n_episodes, ep_len, abandoned_episodes=()):
    for ep in range(n_episodes):
        for w in range(n_workers):
            for t in range(ep_len):
                replay.add(make_step(w, ep, t, ep_len, abandoned=ep in abandoned_episodes),
                           worker=w)


def make_replay(length, chunksize=32, capacity=None):
    return Replay(length=length, capacity=capacity, chunksize=chunksize, seed=0)


def wid(data):  # worker id per (row, step)
    return data["obs"][..., 0]


def tix(data):  # time index per (row, step)
    return data["obs"][..., 2]


def eid(data):
    return data["obs"][..., 1]


# ---------------------------------------------------------------------------

def test_shapes_and_keys():
    K = 0
    replay = make_replay(L + K)
    fill(replay, n_workers=2, n_episodes=3, ep_len=40)
    stream = SequentialReplayStream(replay, B, L, prefix=K, seed=1)
    data = next(stream)
    assert set(data) == {"obs", "action", "reward", "is_first", "is_last",
                         "is_terminal", "stepid", "consec"}, set(data)
    for k, v in data.items():
        assert v.shape[:2] == (B, L + K), (k, v.shape)
    assert data["consec"].dtype == np.int32
    assert (data["consec"] == 0).all(), "very first batch must be all restarts"


def test_first_batch_restart_flags_and_is_first_forcing_k0():
    replay = make_replay(L)
    fill(replay, 2, 3, 40)
    stream = SequentialReplayStream(replay, B, L, prefix=0, seed=1)
    data = next(stream)
    # K=0: restarted rows get is_first forced at position 0 (zero RSSM state
    # at the random offset) ...
    assert data["is_first"][:, 0].all()
    # ... and nowhere else unless it is a genuine episode start.
    genuine = tix(data) == 0
    assert np.array_equal(data["is_first"][:, 1:], genuine[:, 1:])


def test_no_is_first_forcing_with_prefix():
    K = 3
    replay = make_replay(L + K)
    fill(replay, 2, 3, 40)
    stream = SequentialReplayStream(replay, B, L, prefix=K, seed=1)
    for _ in range(5):
        data = next(stream)
        # With a replay-context prefix the stored latent seeds restarted rows,
        # so is_first must be exactly the env's own flag everywhere.
        assert np.array_equal(data["is_first"], tix(data) == 0)


def test_contiguity_across_batches_and_chunk_boundaries():
    K = 2
    replay = make_replay(L + K, chunksize=32)   # 32 % 8 == 0 but 32 % (L+K) != 0
    fill(replay, 3, 6, 50)                        # plenty of chunk boundaries
    stream = SequentialReplayStream(replay, B, L, prefix=K, seed=3)
    prev = next(stream)
    n_checked = 0
    for _ in range(30):
        data = next(stream)
        for b in range(B):
            if data["consec"][b, 0] == 0:
                continue  # restarted row: no continuity expected
            # Window i+1 starts exactly L steps after window i.
            assert np.array_equal(data["stepid"][b, 0], prev["stepid"][b, L]), b
            assert np.array_equal(data["obs"][b, :K], prev["obs"][b, L:L + K]), b
            # Time index increases by 1 within the window except at episode
            # starts (t wraps to 0 there); never a jump elsewhere.
            t = tix(data)[b]
            dt = np.diff(t)
            ok = (dt == 1) | (t[1:] == 0)
            assert ok.all(), (t,)
            n_checked += 1
        prev = data
    assert n_checked > 50, n_checked


def test_rows_never_mix_workers():
    replay = make_replay(L, chunksize=16)
    fill(replay, 4, 5, 30)
    stream = SequentialReplayStream(replay, B, L, prefix=0, seed=5)
    for _ in range(40):
        data = next(stream)
        w = wid(data)
        assert (w == w[:, :1]).all(), w
        assert (data["obs"][..., 3] == MARKER).all(), "read past the write head"


def test_head_is_never_read_and_partial_chunk_is_handled():
    """Add data while reading: the stream must never return unwritten memory."""
    K = 0
    replay = make_replay(L + K, chunksize=16)
    fill(replay, 1, 1, 40)
    stream = SequentialReplayStream(replay, B, L, prefix=K, seed=7)
    t = 0
    for it in range(60):
        data = next(stream)
        assert (data["obs"][..., 3] == MARKER).all(), it
        # keep the writer moving: 5 more steps of a long episode each iteration
        for _ in range(5):
            replay.add(make_step(0, 99, t, ep_len=10_000), worker=0)
            t += 1
    st = stream.stats()
    # One 40-step episode + a growing tail with 4 readers walking 8 steps per
    # batch against a writer adding 5: cursors keep catching the head, so
    # there must be restarts, and the stream must have survived them.
    assert st["tbtt_restart_frac"] > 0.02, st
    assert st["tbtt_batches"] == 60


def test_eviction_causes_restart_not_crash():
    K = 0
    replay = make_replay(L + K, chunksize=16, capacity=64)
    fill(replay, 2, 2, 40)
    stream = SequentialReplayStream(replay, B, L, prefix=K, seed=9)
    for i in range(40):
        data = next(stream)
        assert (data["obs"][..., 3] == MARKER).all()
        # churn the buffer so the cursors' chunks get evicted underneath them
        for t in range(10):
            replay.add(make_step(1, 500 + i, t, 10), worker=1)
    st = stream.stats()
    assert 0.0 < st["tbtt_restart_frac"] < 1.0, st


def test_stretch_grows_far_beyond_one_window():
    """Long episodes, big buffer, no writer: cursors should carry for hundreds
    of steps, which is the whole point versus consec4's fixed 192."""
    replay = make_replay(L, chunksize=64)
    fill(replay, 2, 4, 200)   # 800 steps per worker, episodes of 200
    stream = SequentialReplayStream(replay, B, L, prefix=0, seed=11)
    for _ in range(40):
        next(stream)
    st = stream.stats()
    assert st["tbtt_stretch_max"] >= 160, st
    assert st["tbtt_restart_frac"] < 0.5, st
    # boundaries occur roughly once per 200 steps -> ~L/200 of rows
    assert 0.0 < st["tbtt_episode_boundary_frac"] < 0.2, st


def test_is_last_repair_for_abandoned_episode():
    replay = make_replay(L, chunksize=64)
    # episode 0 is abandoned (no is_last), episode 1 follows with is_first
    fill(replay, 1, 3, 20, abandoned_episodes=(0,))
    stream = SequentialReplayStream(replay, B, L, prefix=0, seed=13)
    seen = False
    for _ in range(30):
        data = next(stream)
        first = data["is_first"]
        last = data["is_last"]
        for b in range(B):
            for t in range(1, L):
                if first[b, t] and eid(data)[b, t] == 1:
                    assert last[b, t - 1], "abandoned episode must get is_last"
                    seen = True
    assert seen


def test_consec_zero_exactly_on_restart():
    replay = make_replay(L, chunksize=16, capacity=48)
    fill(replay, 1, 2, 40)
    stream = SequentialReplayStream(replay, B, L, prefix=0, seed=17)
    prev = next(stream)
    for i in range(30):
        data = next(stream)
        for b in range(B):
            # K=0: window i+1 starts right after the last step of window i,
            # i.e. same worker and either t advances by one or a new episode
            # of that worker starts (t wraps to 0 with the env's own is_first).
            same_worker = wid(data)[b, 0] == wid(prev)[b, L - 1]
            t0, t_prev = tix(data)[b, 0], tix(prev)[b, L - 1]
            contiguous = same_worker and (
                t0 == t_prev + 1
                or (t0 == 0 and eid(data)[b, 0] != eid(prev)[b, L - 1]))
            if data["consec"][b, 0] == 1:
                assert contiguous, "consec=1 row must continue the previous window"
            else:
                assert data["is_first"][b, 0], "K=0 restart must force is_first"
        for t in range(6):
            replay.add(make_step(0, 700 + i, t, 6), worker=0)
        prev = data


def test_attach_stats_merges_into_replay_stats():
    replay = make_replay(L)
    fill(replay, 1, 2, 40)
    stream = SequentialReplayStream(replay, B, L, prefix=0)
    attach_stats(replay, stream)
    next(stream)
    st = replay.stats()
    assert "tbtt_restart_frac" in st and "inserts" in st, st


def test_replay_length_mismatch_is_rejected():
    replay = make_replay(L + 5)
    try:
        SequentialReplayStream(replay, B, L, prefix=0)
    except AssertionError:
        return
    raise AssertionError("expected an AssertionError for replay.length != L + K")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
