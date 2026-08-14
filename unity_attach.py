"""Shared Unity-port handling for the eval scripts.

Why this exists: `allocate_unity_instances(n_envs=1)` is the right default for
*training* (spawn what you need, reuse :9000 if it happens to be there) and the
wrong one for *eval*, whose entire point is usually to watch the policy in the
Unity Editor. That call spawns a headless build whenever :9000 is not accepting
at that exact instant — so "I started the eval a second before pressing Play"
silently evaluates against an invisible simulator, and since the launcher runs
spawned builds under xvfb there is nothing to see anywhere. Worse, the spawned
build holds :9000, so the Editor then fails to bind it.

So eval ATTACHES by default and waits for a human to press Play; it never
spawns. `--spawn` restores the old behaviour for headless/batch use, and under
SLURM it is selected automatically, because :9000 on a shared node belongs to
somebody else (same rule the launcher itself applies).
"""
from __future__ import annotations

import argparse
import os

from ratsim.unity_launcher import (
    DEFAULT_ATTACH_TIMEOUT,
    PERSISTENT_PORT,
    allocate_unity_instances,
    attach_instance,
)


def add_unity_attach_args(ap: argparse.ArgumentParser) -> None:
    """Add --unity_port / --attach_timeout / --spawn to an eval script."""
    g = ap.add_argument_group("unity")
    g.add_argument("--unity_port", type=int, default=PERSISTENT_PORT,
                   help=f"Port to attach to (default {PERSISTENT_PORT}, the "
                        f"Editor / persistent instance). Ignored with --spawn.")
    g.add_argument("--attach_timeout", type=float, default=None,
                   help=f"Seconds to wait for a Unity to start accepting on "
                        f"--unity_port before giving up. 0 waits forever. "
                        f"Default {DEFAULT_ATTACH_TIMEOUT:.0f} "
                        f"($RATSIM_ATTACH_TIMEOUT overrides).")
    g.add_argument("--spawn", action="store_true",
                   help="Spawn a headless Unity build instead of attaching to "
                        "a running one (needs $RATSIM_UNITY_BIN). Use for "
                        "batch/headless eval where nobody is watching. "
                        "Automatic under SLURM.")


def resolve_unity_port(args: argparse.Namespace, tag: str = "eval") -> int:
    """Return the port the eval should talk to, spawning only if asked.

    `tag` is just the log prefix, so the two eval scripts keep their own.
    """
    under_slurm = any(os.environ.get(v, "").isdigit()
                      for v in ("SLURM_JOB_ID", "SLURM_JOBID"))
    if args.spawn or under_slurm:
        if under_slurm and not args.spawn:
            print(f"[{tag}] under SLURM: spawning a Unity build rather than "
                  f"attaching to :{PERSISTENT_PORT} (a live :{PERSISTENT_PORT} "
                  f"on a shared node is somebody else's simulator).")
        return allocate_unity_instances(n_envs=1)[0].port
    return attach_instance(args.unity_port,
                           timeout_s=args.attach_timeout).port
