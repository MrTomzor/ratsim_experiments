"""Unity port window allocator.

Each running training subprocess gets a non-overlapping window of consecutive
ports starting at some `base_port`. We pick window size larger than the largest
expected `n_envs` so windows never overlap. The 10-port gap convention
documented in CLAUDE.md is what we follow.

Allocation is in-memory only — windows are released when the subprocess
finishes. The scheduler is the sole writer.
"""
from __future__ import annotations


class PortAllocator:
    def __init__(self, start: int = 9100, window_size: int = 10):
        self.start = start
        self.window_size = window_size
        self.in_use: set[int] = set()  # set of base ports

    def alloc(self) -> int:
        i = 0
        while True:
            port = self.start + i * self.window_size
            if port not in self.in_use:
                self.in_use.add(port)
                return port
            i += 1

    def release(self, port: int) -> None:
        self.in_use.discard(port)
