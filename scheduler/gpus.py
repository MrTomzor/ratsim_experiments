"""GPU device allocator — the CUDA_VISIBLE_DEVICES counterpart to PortAllocator.

The scheduler dispatches its runs as plain subprocesses inside one SLURM job, so
every child inherits the job's environment. Without this, every GPU method would
read the same CUDA_VISIBLE_DEVICES and pile onto device 0 while the other cards
sat idle — the exact shape of the port collision that PortAllocator exists to
prevent, one resource over.

`ResourceManager` already limits *how many* GPU units are in flight (via
`needs.gpu` against `resources.gpu`). This class answers the other half: *which
device* each of those units is, so the scheduler can hand each child a distinct
one.

Allocation is in-memory only — devices are released when the subprocess
finishes. The scheduler is the sole writer.
"""
from __future__ import annotations

import os


class GpuAllocator:
    """Hands out distinct CUDA device ids from a fixed pool.

    Device ids are kept as **strings, not ints**, and passed through verbatim.
    CUDA_VISIBLE_DEVICES may legitimately contain GPU UUIDs
    ("GPU-a1b2c3d4-...") rather than indices, and some sites configure SLURM
    that way; parsing to int would break there for no benefit, since we only
    ever split the list and re-join it.
    """

    def __init__(self, devices: list[str] | None = None):
        self.devices: list[str] = list(devices or [])
        self.free: list[str] = list(self.devices)

    @property
    def active(self) -> bool:
        """False when there are no GPUs to manage, in which case the scheduler
        leaves CUDA_VISIBLE_DEVICES alone entirely. That is what keeps a
        CPU-only box (default.yaml, a laptop) behaving exactly as before."""
        return bool(self.devices)

    @classmethod
    def from_environment(cls, declared: int) -> "GpuAllocator":
        """Build the pool for a machine config declaring `resources.gpu`.

        Prefers the job's own CUDA_VISIBLE_DEVICES when set. Under SLURM with
        cgroup device isolation the job only *sees* the GPUs it was granted and
        that variable already lists exactly them, so splitting it is both the
        correct set and the correct ids: each entry means the same device to a
        child process as it does to us. Falling back to range(declared) is for
        a plain box with no resource manager in front of it.
        """
        if declared <= 0:
            return cls([])
        raw = os.environ.get("CUDA_VISIBLE_DEVICES")
        if raw is None:
            # Unset: trust the machine config. Note that CUDA treats *empty*
            # differently from *unset* — "" means "no GPUs at all" — so only a
            # missing variable falls through to the declaration.
            return cls([str(i) for i in range(declared)])
        devices = [d.strip() for d in raw.split(",") if d.strip()]
        if len(devices) != declared:
            # Worth shouting about: the machine config and the sbatch disagree.
            # The pool can only ever contain devices that exist, so it reports
            # what is really there; the caller decides how much of it to use
            # (the scheduler takes the smaller of the two — it will not widen a
            # deliberate cap just because --gres over-asked).
            verb = ("only " if len(devices) < declared
                    else "more than declared: ")
            print(f"[scheduler] WARNING: machine config declares gpu="
                  f"{declared} but the allocation granted {verb}"
                  f"{len(devices)} (CUDA_VISIBLE_DEVICES={raw!r}). "
                  f"Fix --gres or resources.gpu so they agree.")
        return cls(devices)

    def alloc(self, n: int) -> list[str] | None:
        """Reserve `n` devices, or None if that many aren't free.

        n=0 returns [] rather than None: "needed nothing, got nothing" is a
        success, and callers distinguish the two by identity.
        """
        if n <= 0:
            return []
        if len(self.free) < n:
            return None
        taken = self.free[:n]
        self.free = self.free[n:]
        return taken

    def release(self, devices: list[str] | None) -> None:
        for d in (devices or []):
            if d in self.devices and d not in self.free:
                self.free.append(d)
        # Keep the pool in its original order so repeated dispatches reuse the
        # lowest-numbered device first. Purely cosmetic, but it makes
        # nvidia-smi output and the scheduler log predictable while debugging.
        self.free.sort(key=self.devices.index)

    def child_env(self, devices: list[str]) -> dict[str, str] | None:
        """Environment for a child that was allocated `devices`.

        Returns None when the caller should just inherit — i.e. there are no
        GPUs under management at all.

        When the allocator IS active, a child that needs no GPU is masked with
        an empty CUDA_VISIBLE_DEVICES rather than left to inherit the job's
        full list. Otherwise a CPU-profile run (PPO with method.device=cpu)
        still opens a context on whichever card it likes and takes memory from
        the dreamer next to it. If a method genuinely needs the GPU, declare
        `needs: {gpu: 1}` in the machine profile — that is what this reads.
        """
        if not self.active:
            return None
        return {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(devices)}

    def __repr__(self) -> str:
        return (f"GpuAllocator(devices={self.devices}, free={self.free})")
