"""Concurrency and crash-safety of probing/core/dataset.py, with real processes.

    ~/ratvenv/dreamer_venv/bin/python tests/test_probe_concurrency.py
    ~/ratvenv/venv/bin/python tests/test_probe_concurrency.py

Every test here is a way the first full run could have produced WRONG numbers
instead of crashing: a reader seeing half a file, two writers sharing a temp
file, a killed writer's leftovers counted as an episode, two jobs on one
dataset, or states and targets computed from two versions of an episode.
numpy only, so it runs in both venvs.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probing.core import dataset  # noqa: E402

CTX = mp.get_context("fork")
BIG = 400_000  # floats per array: ~1.6 MB compressed-incompressible payload


def _payload(writer: int, it: int) -> dict:
    """Self-checking content: every value encodes (writer, it)."""
    rng = np.random.default_rng(writer * 1000 + it)
    noise = rng.random(BIG).astype(np.float32)  # incompressible -> slow writes
    return {"writer": np.int32(writer), "it": np.int32(it),
            "noise": noise, "sum": np.float64(noise.sum(dtype=np.float64))}


def _consistent(d: dict) -> bool:
    ok = d["noise"].sum(dtype=np.float64) == d["sum"]
    again = _payload(int(d["writer"]), int(d["it"]))["noise"]
    return bool(ok and np.array_equal(again, d["noise"]))


def _writer(path, writer, n, start):
    start.wait()
    for it in range(n):
        dataset.save_npz(path, _payload(writer, it))


def _reader(path, stop, bad, reads):
    # One message per reader, sent at the end: a process with unread items in
    # a multiprocessing.Queue cannot be joined, so per-error puts deadlock.
    errors, first = 0, ""
    while not stop.is_set():
        try:
            d = dataset.load_episode(path)
        except FileNotFoundError:
            continue
        except Exception as e:  # noqa: BLE001 -- any parse error is a torn read
            errors += 1
            first = first or f"{type(e).__name__}: {e}"
            continue
        with reads.get_lock():
            reads.value += 1
        if not _consistent(d):
            errors += 1
            first = first or f"mixed content writer={int(d['writer'])}"
    bad.put((errors, first))


def test_writers_and_readers_of_one_file_never_see_a_torn_episode():
    """8 processes rewrite one episode file while 4 others read it."""
    with tempfile.TemporaryDirectory() as tmp:
        path = dataset.episode_path(tmp, 7)
        dataset.save_npz(path, _payload(99, 0))
        start, stop = CTX.Event(), CTX.Event()
        bad, reads = CTX.Queue(), CTX.Value("i", 0)
        writers = [CTX.Process(target=_writer, args=(path, w, 6, start))
                   for w in range(8)]
        readers = [CTX.Process(target=_reader, args=(path, stop, bad, reads))
                   for _ in range(4)]
        for p in writers + readers:
            p.start()
        start.set()
        for p in writers:
            p.join(120)
        stop.set()
        reports = [bad.get(timeout=60) for _ in readers]  # drain, then join
        for p in readers:
            p.join(30)
        n_err = sum(n for n, _ in reports)
        assert all(p.exitcode == 0 for p in writers), \
            f"writers crashed: {[p.exitcode for p in writers]}"
        assert n_err == 0, f"{n_err} torn reads, e.g. {[m for _, m in reports if m][:1]}"
        assert reads.value > 20, f"readers barely ran ({reads.value} reads)"
        assert _consistent(dataset.load_episode(path))
        leftovers = [p.name for p in Path(path).parent.iterdir() if p != path]
        assert not leftovers, leftovers


def _write_forever(path, ready):
    it = 0
    while True:
        dataset.save_npz(path, _payload(1, it))
        it += 1
        ready.set()


def test_a_writer_killed_mid_write_leaves_a_whole_file_or_none():
    """SIGKILL during writes, repeatedly: the episode path always holds one
    complete payload, and the killed writer's temp file is never listed."""
    with tempfile.TemporaryDirectory() as tmp:
        path = dataset.episode_path(tmp, 0)
        for trial in range(8):
            ready = CTX.Event()
            p = CTX.Process(target=_write_forever, args=(path, ready))
            p.start()
            ready.wait(30)
            time.sleep(0.01 + 0.03 * trial)  # land at different points of a write
            os.kill(p.pid, signal.SIGKILL)
            p.join()
            assert _consistent(dataset.load_episode(path)), f"trial {trial}"
            assert dataset.list_episodes(tmp) == [path]


def test_stale_temp_files_are_never_episodes():
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(3):
            dataset.save_npz(dataset.episode_path(tmp, i), {"x": np.zeros(1)})
        d = dataset.episode_dir(tmp)
        for name in (".ep0003.npz.4242.1.tmp",   # this code's temp names
                     "ep0003.tmp.npz",           # the old code's temp name
                     "ep0003.npz.part", "notes.txt"):
            (d / name).write_bytes(b"garbage")
        assert [p.name for p in dataset.list_episodes(tmp)] == \
            ["ep0000.npz", "ep0001.npz", "ep0002.npz"]


def _publish(path, obj, start, results):
    start.wait()
    try:
        results.put(("created" if dataset.publish_once(path, obj) else "matched"))
    except Exception as e:  # noqa: BLE001
        results.put(f"error {type(e).__name__}: {e}")


def test_publish_once_has_exactly_one_creator_and_catches_a_mismatch():
    obj = {"obs": {"lidar": {"shape": [57], "low": list(range(500))}}}
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spaces.json"
        start, results = CTX.Event(), CTX.Queue()
        procs = [CTX.Process(target=_publish, args=(path, obj, start, results))
                 for _ in range(16)]
        for p in procs:
            p.start()
        start.set()
        for p in procs:
            p.join()
        out = sorted(results.get() for _ in procs)
        assert out.count("created") == 1 and out.count("matched") == 15, out
        assert dataset.load_json(path) == obj
        assert [p.name for p in Path(tmp).iterdir()] == ["spaces.json"]
        try:
            dataset.publish_once(path, {"obs": {}})
        except ValueError:
            pass
        else:
            raise AssertionError("a different spaces.json was accepted")


def _hold(path, held, release):
    with dataset.lock(path):
        held.set()
        release.wait(30)


def _try_lock(path, results):
    try:
        with dataset.lock(path):
            results.put("acquired")
    except RuntimeError:
        results.put("refused")


def test_lock_refuses_a_second_process_and_dies_with_its_holder():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / ".collect.lock"
        held, release, results = CTX.Event(), CTX.Event(), CTX.Queue()
        holder = CTX.Process(target=_hold, args=(path, held, release))
        holder.start()
        assert held.wait(10)
        t0 = time.time()
        p = CTX.Process(target=_try_lock, args=(path, results))
        p.start(); p.join()
        assert results.get() == "refused"
        assert time.time() - t0 < 5, "a second holder must fail fast, not wait"
        # A holder killed by the scheduler must not leave the lock taken.
        os.kill(holder.pid, signal.SIGKILL)
        holder.join()
        p = CTX.Process(target=_try_lock, args=(path, results))
        p.start(); p.join()
        assert results.get() == "acquired"


def _sealed_dataset(tmp, n=4, plan=True):
    root = Path(tmp) / "ds"
    for i in range(n):
        dataset.save_npz(dataset.episode_path(root, i), {"x": np.full(3, i)})
    meta = {"plan": [{"idx": i} for i in range(n)]} if plan else {}
    dataset.save_json(root / "meta.json", meta)
    dataset.seal(root)
    return root


def _raises(fn, text):
    try:
        fn()
    except ValueError as e:
        assert text in str(e), str(e)
    else:
        raise AssertionError(f"expected ValueError containing {text!r}")


def test_verify_refuses_unsealed_changed_and_incomplete_datasets():
    with tempfile.TemporaryDirectory() as tmp:
        root = _sealed_dataset(tmp)
        fp = dataset.verify(root)
        assert len(fp) == 64

        dataset.save_npz(dataset.episode_path(root, 2), {"x": np.full(3, 9)})
        _raises(lambda: dataset.verify(root), "changed since")
        dataset.seal(root)

        dataset.episode_path(root, 1).unlink()
        _raises(lambda: dataset.verify(root), "not contiguous")

        meta = dataset.load_json(root / "meta.json")
        meta.pop("fingerprint")
        dataset.save_json(root / "meta.json", meta)
        dataset.save_npz(dataset.episode_path(root, 1), {"x": np.full(3, 1)})
        _raises(lambda: dataset.verify(root), "not sealed")

        dataset.seal(root)
        dataset.save_npz(dataset.episode_path(root, 4), {"x": np.zeros(3)})
        dataset.seal(root)
        _raises(lambda: dataset.verify(root), "plan has 4")


def test_is_current_tracks_the_sealed_fingerprint():
    with tempfile.TemporaryDirectory() as tmp:
        root = _sealed_dataset(tmp)
        fp = dataset.verify(root)
        js, npz = Path(tmp) / "m.json", Path(tmp) / "s.npz"
        dataset.save_json(js, {"dataset_fingerprint": fp})
        dataset.save_npz(npz, {"dataset_fingerprint": np.str_(fp)})
        assert dataset.is_current(js, root) and dataset.is_current(npz, root)
        assert not dataset.is_current(Path(tmp) / "missing.json", root)
        dataset.save_npz(dataset.episode_path(root, 0), {"x": np.ones(3)})
        dataset.seal(root)  # the dataset moved on: old outputs are stale
        assert not dataset.is_current(js, root)
        assert not dataset.is_current(npz, root)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f}s)")
        except Exception as e:  # noqa: BLE001
            failed += 1
            import traceback
            print(f"FAIL {name}: {type(e).__name__}: {e}")
            traceback.print_exc(limit=6)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
