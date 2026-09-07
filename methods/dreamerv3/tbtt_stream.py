"""Sequential replay stream for Dreamer + TBTT (MemoryMaze-style).

Reference: Pasukonis, Lillicrap & Hafner, "Evaluating Long-Term Memory in 3D
Mazes" (arXiv 2210.13383), §4.1, "Dreamer (TBTT)":

    "we implement TBTT training in Dreamer by replaying complete trajectories
    sequentially and passing the RSSM state from one batch to the next. We
    alleviate the potential problem of correlated batches by forming each
    T x B batch from B different episodes, and we start replaying the first
    episode from a random offset to avoid synchronized episode starts."

How this maps onto dreamerv3 (danijar main, b65cf81):

* The run loop (``embodied.run.train``) already threads the carry returned by
  ``agent.train`` into the next call, and the RSSM already zeroes its state on
  ``is_first`` (``rssm._observe`` masks the carry with ``~reset``). Each
  ``agent.train`` call is a separate jitted step, so the carried state is a
  concrete array: gradients still truncate at ``batch_length`` exactly as in
  the paper (their sequence length was 48 too).
* The stock sampler (``Stateless(replay.sample)`` + ``Consec``) draws an
  independent window every ``consec_train`` chunks. This stream replaces it
  with ``batch_size`` cursors that each walk forward through the replay buffer
  ``batch_length`` steps at a time, forever. Consecutive batches are therefore
  contiguous per row, and the carry is valid across the whole stored
  trajectory of that env worker (episode boundaries included; ``is_first``
  resets the state there).
* A cursor restarts at a random buffer position when its data was evicted or
  when it catches up with the live write head. On restart the row gets
  ``consec = 0``; otherwise ``consec = 1``. The agent's
  ``_apply_replay_context`` branches on that per row: with ``replay_context``
  K > 0 it seeds the carry from the stored RSSM latent (consec4 behaviour),
  with K = 0 the flag is ignored and we force ``is_first`` on the restarted
  row so the RSSM starts from zero state at the random offset -- the paper's
  exact behaviour.

The one trap: ``Replay._annotate_batch`` forces ``is_first[:, 0] = True`` on
every sampled window. Under the stock sampler that lands in the discarded
replay-context prefix (harmless); in a sequential stream it would zero the
carry at the top of every chunk and silently turn TBTT back into vanilla
Dreamer. This stream deliberately does not do that; only the ``is_last``
repair for abandoned episodes is kept.

Compute per gradient step is unchanged (batch shape stays ``(B, L + K)``), so
this is not slower per step. What changes is the sampling distribution:
batches are time-correlated within a stretch, and the ``online`` fresh-data
queue of the stock replay is bypassed (the paper had neither).
"""
from __future__ import annotations

import threading

import numpy as np
from embodied.core import limiters


class SequentialReplayStream:
    """Batch iterator that walks ``batch_size`` cursors through a replay buffer.

    Yields dicts of shape ``(batch_size, length + prefix)`` with the same keys
    as ``Replay.sample`` plus ``consec`` (int32, 0 on the first chunk after a
    restart, 1 otherwise). Duck-types ``embodied.core.base.Stream``.
    """

    def __init__(self, replay, batch_size: int, length: int, prefix: int = 0,
                 seed: int = 0):
        assert replay.length == length + prefix, (
            f"replay.length={replay.length} must equal length + prefix = "
            f"{length} + {prefix}; build the replay with consec_train=1")
        self.replay = replay
        self.batch_size = int(batch_size)
        self.length = int(length)
        self.prefix = int(prefix)
        self.rng = np.random.default_rng(seed)
        # Per row: (chunkid, index) of the next window to read, or None to restart.
        self.cursors: list = [None] * self.batch_size
        # Steps carried since the row's last restart (for stats).
        self.carried = np.zeros(self.batch_size, np.int64)
        self._lock = threading.Lock()
        self._reset_counters()

    # -- Stream protocol ------------------------------------------------------

    def __iter__(self):
        return self

    def __next__(self):
        replay = self.replay
        limiters.wait(lambda: len(replay.sampler) > 0,
                      f"Replay buffer {replay.name} is empty")
        seqs = []
        restarted = np.zeros(self.batch_size, bool)
        for b in range(self.batch_size):
            seq, r = self._read_row(b)
            seqs.append(seq)
            restarted[b] = r
        data = replay._assemble_batch(seqs, 0, replay.length)
        data = self._annotate(data, restarted)
        self._count(data, restarted)
        return data

    def save(self):
        # Cursors are cheap to lose: on resume every row simply restarts at a
        # random offset, which is what the paper does at the start anyway.
        return None

    def load(self, state):
        pass

    # -- Cursor handling ------------------------------------------------------

    def _read_row(self, b: int):
        """Return (seq_parts, restarted) for row ``b`` and advance its cursor."""
        restarted = False
        while True:
            if self.cursors[b] is None:
                start = self._random_start()
                if start is None:
                    limiters.wait(lambda: len(self.replay.sampler) > 0,
                                  f"Replay buffer {self.replay.name} is empty")
                    continue
                self.cursors[b] = start
                self.carried[b] = 0
                restarted = True
            chunkid, index = self.cursors[b]
            try:
                seq = self.replay._getseq(chunkid, index, concat=False)
            except (KeyError, AssertionError, AttributeError):
                # Chunk evicted (KeyError), or the window runs past the live
                # write head: the live chunk's succ is UUID(0) (KeyError), or
                # a just-created successor chunk with no data yet
                # (AttributeError on chunk.data=None). Either way: restart.
                self.cursors[b] = None
                continue
            self.cursors[b] = self._advance(chunkid, index, self.length)
            self.carried[b] += self.length
            return seq, restarted

    def _random_start(self):
        """A random (chunkid, index) that had a full window when it was indexed."""
        replay = self.replay
        try:
            if len(replay.sampler) == 0:
                return None
            itemid = replay.sampler()
            return replay.items[itemid]
        except (KeyError, ValueError):
            return None

    def _advance(self, chunkid, index: int, n: int):
        """Move ``n`` steps forward along the chunk chain; None if it runs out."""
        index += n
        while True:
            chunk = self.replay.chunks.get(chunkid)
            if chunk is None:
                return None
            if index < chunk.length:
                return (chunkid, index)
            if chunk.length < chunk.size:
                # Live (incomplete) chunk and we are past what was written:
                # we caught the write head. Restart rather than wait.
                return None
            index -= chunk.size
            chunkid = chunk.succ

    # -- Batch annotation -----------------------------------------------------

    def _annotate(self, data: dict, restarted: np.ndarray) -> dict:
        data = dict(data)
        is_first = np.array(data["is_first"], copy=True)
        if self.prefix == 0:
            # Paper-exact: zero RSSM state at the random restart offset. With
            # prefix > 0 the stored latent seeds the state instead (consec=0).
            is_first[restarted, 0] = True
        data["is_first"] = is_first
        if "is_last" in data:
            # Same repair as Replay._annotate_batch: an episode abandoned
            # without is_last (env reset mid-episode) gets is_last set on the
            # step before the next is_first.
            next_is_first = np.roll(is_first, shift=-1, axis=1)
            next_is_first[:, -1] = False
            data["is_last"] = data["is_last"] | next_is_first
        consec = np.where(restarted, 0, 1).astype(np.int32)[:, None]
        data["consec"] = np.ascontiguousarray(
            np.broadcast_to(consec, is_first.shape))
        return data

    # -- Stats ----------------------------------------------------------------

    def _reset_counters(self):
        self._n_batches = 0
        self._n_rows = 0
        self._n_restarts = 0
        self._n_boundaries = 0
        self._stretch_sum = 0
        self._stretch_max = 0

    def _count(self, data: dict, restarted: np.ndarray):
        # Episode boundaries inside the trained part of the window, excluding
        # an is_first we forced ourselves at position 0 of a restarted row.
        trained = data["is_first"][:, self.prefix:]
        natural = trained.copy()
        if self.prefix == 0:
            natural[restarted, 0] = False
        with self._lock:
            self._n_batches += 1
            self._n_rows += self.batch_size
            self._n_restarts += int(restarted.sum())
            self._n_boundaries += int(natural.any(axis=1).sum())
            self._stretch_sum += int(self.carried.sum())
            self._stretch_max = max(self._stretch_max, int(self.carried.max()))

    def stats(self) -> dict:
        """Counters since the previous call (logged under ``replay/`` by the
        train loop when attached via ``attach_stats``)."""
        with self._lock:
            rows = max(self._n_rows, 1)
            out = {
                "tbtt_restart_frac": self._n_restarts / rows,
                "tbtt_episode_boundary_frac": self._n_boundaries / rows,
                "tbtt_stretch_mean": self._stretch_sum / rows,
                "tbtt_stretch_max": float(self._stretch_max),
                "tbtt_batches": float(self._n_batches),
            }
            self._reset_counters()
        return out


def attach_stats(replay, stream: SequentialReplayStream) -> None:
    """Merge the stream's counters into ``replay.stats()``.

    ``embodied.run.train`` logs ``replay.stats()`` every ``log_every`` and
    knows nothing about the stream, so this is the least invasive way to get
    the TBTT counters into TensorBoard / W&B without copying the run loop.
    """
    original = replay.stats

    def stats():
        return {**original(), **stream.stats()}

    replay.stats = stats


def make_stream(config, replay, mode: str, fallback):
    """Drop-in for ``dreamerv3.main.make_stream``.

    Returns the sequential stream for training when ``config.tbtt.enabled``,
    otherwise defers to ``fallback(config, replay, mode)`` (the stock
    ``Stateless`` + ``Consec`` pair). Report streams always use the fallback.
    """
    if mode != "train" or not config.tbtt.enabled:
        return fallback(config, replay, mode)
    assert config.consec_train == 1, (
        "tbtt.enabled requires consec_train=1 (build_config forces this)")
    stream = SequentialReplayStream(
        replay,
        batch_size=config.batch_size,
        length=config.batch_length,
        prefix=config.replay_context,
        seed=config.seed,
    )
    attach_stats(replay, stream)
    return stream
