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
"""
from __future__ import annotations


class PortAllocator:
    def __init__(self, start: int = 9100, window_size: int = 10,
                 persistent_port: int | None = None):
        self.start = start
        self.window_size = window_size
        self.in_use: set[int] = set()  # set of base ports
        # Single-port slot, disjoint from the windowed range. When None, the
        # scheduler behaves like always (only 9100+ windows). When set (e.g.
        # to 9000), `try_alloc_persistent()` may hand it to one n_envs=1 job.
        self.persistent_port = persistent_port
        self.persistent_in_use = False

    def alloc(self) -> int:
        i = 0
        while True:
            port = self.start + i * self.window_size
            if port not in self.in_use:
                self.in_use.add(port)
                return port
            i += 1

    def try_alloc_persistent(self) -> int | None:
        """Reserve the persistent slot if it's enabled and free; else return
        None. Caller is responsible for confirming liveness / suitability
        first (e.g. n_envs == 1, Unity actually running on this port)."""
        if self.persistent_port is None or self.persistent_in_use:
            return None
        self.persistent_in_use = True
        return self.persistent_port

    def release(self, port: int) -> None:
        if port == self.persistent_port and self.persistent_in_use:
            self.persistent_in_use = False
        else:
            self.in_use.discard(port)
