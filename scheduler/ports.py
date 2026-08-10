"""Unity port window allocator.

Each running training subprocess gets a non-overlapping window of consecutive
ports starting at some `base_port`. We pick window size larger than the largest
expected `n_envs` so windows never overlap. The 10-port gap convention
documented in CLAUDE.md is what we follow.

Optionally tracks a single "persistent port" slot (default disabled). When
enabled, the scheduler can hand it out to one n_envs=1 dispatch at a time —
intended for the user's manually-launched Unity GUI on port 9000, so they
can watch one training instance while others run headless on 9100+.

Allocation is in-memory only — windows are released when the subprocess
finishes. The scheduler is the sole writer.

Released windows do NOT go straight back into circulation; see the "cooling"
discussion on `release()`.
"""
from __future__ import annotations

from typing import Callable

from ratsim.unity_launcher import _port_bindable


# Refuse to scan forever if every window looks busy. 100 windows is 9100-10099,
# far more than any node can host (a run needs ~4 threads per env, so a
# 382-thread node tops out around 24 concurrent runs). Hitting this means
# something is systematically wrong — a leaked pool of Unity processes, or
# another user's job sitting across the range — and a clear error beats an
# infinite loop.
MAX_WINDOWS = 100


class PortAllocator:
    def __init__(self, start: int = 9100, window_size: int = 10,
                 persistent_port: int | None = None,
                 port_is_free: Callable[[int], bool] | None = None):
        self.start = start
        self.window_size = window_size
        self.in_use: set[int] = set()  # set of base ports
        # Base ports released by a finished job whose Unity processes may not
        # have let go of the sockets yet. Re-tested lazily on each alloc().
        self.cooling: set[int] = set()
        # Injectable so tests don't need real sockets.
        self.port_is_free = port_is_free or _port_bindable
        # Single-port slot, disjoint from the windowed range. When None, the
        # scheduler behaves like always (only 9100+ windows). When set (e.g.
        # to 9000), `try_alloc_persistent()` may hand it to one n_envs=1 job.
        self.persistent_port = persistent_port
        self.persistent_in_use = False

    def _window_is_free(self, base: int) -> bool:
        """True if every port in the window can be bound right now.

        The whole window is tested, not just the first n_envs ports, because
        the allocator doesn't know how many envs the *next* run will want —
        and a window with a stranger squatting on its 8th port is one we should
        keep skipping regardless.
        """
        return all(self.port_is_free(base + k) for k in range(self.window_size))

    def alloc(self) -> int:
        """Hand out the lowest window that is both unreserved and actually free.

        A window in `cooling` is only returned once its ports genuinely bind,
        which is what stops the next run inheriting the previous one's
        still-shutting-down Unity instances.
        """
        skipped = []
        for i in range(MAX_WINDOWS):
            port = self.start + i * self.window_size
            if port in self.in_use:
                continue
            if port in self.cooling:
                if not self._window_is_free(port):
                    skipped.append(port)
                    continue
                # Previous holder is gone — back into normal circulation.
                self.cooling.discard(port)
            if skipped:
                print(f"[scheduler] port windows still cooling, skipped: "
                      f"{skipped} — using {port}")
            self.in_use.add(port)
            return port
        raise RuntimeError(
            f"no free Unity port window in {self.start}-"
            f"{self.start + MAX_WINDOWS * self.window_size - 1} "
            f"({len(self.in_use)} in use, {len(self.cooling)} cooling). "
            f"Something is holding ports that should be free — check for "
            f"leaked Unity processes (scripts/stop_ratsim_headless.sh --all).")

    def try_alloc_persistent(self) -> int | None:
        """Reserve the persistent slot if it's enabled and free; else return
        None. Caller is responsible for confirming liveness / suitability
        first (e.g. n_envs == 1, Unity actually running on this port)."""
        if self.persistent_port is None or self.persistent_in_use:
            return None
        self.persistent_in_use = True
        return self.persistent_port

    def release(self, port: int, cooled: bool = True) -> None:
        """Return a window to the pool — but into `cooling`, not straight to free.

        Pass `cooled=False` when the window was reserved but never actually
        launched into (the caller bailed before spawning), since there is then
        no Unity teardown to wait for.

        The scheduler reaps a job when its *train process* exits, and that
        process spawned n_envs Unity children who outlive it. So for some
        seconds after "done", the window's ports are still held. Handing the
        same window to the next dispatch immediately is what produced

            RuntimeError: port 9640 still in use after waiting

        in job 11325473, killing 4 of 4 dreamer re-dispatches while 0 of 4 PPO
        ones failed — PPO's Unity children die inside the launcher's 20 s
        grace, dreamer's don't.

        Cooling fixes the cause rather than lengthening that grace: alloc()
        tests the window and simply uses a different one if it isn't clear, so
        nothing blocks and no timeout has to be guessed. The launcher's wait
        stays as the backstop for the residual bind race (we test, then the
        child binds a moment later).
        """
        if port == self.persistent_port and self.persistent_in_use:
            self.persistent_in_use = False
        else:
            self.in_use.discard(port)
            if cooled:
                self.cooling.add(port)
