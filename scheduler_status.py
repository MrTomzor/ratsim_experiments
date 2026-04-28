#!/usr/bin/env python
"""Show status for a scheduled experiment.

    python scheduler_status.py method_compare
    python scheduler_status.py defs/method_compare.yaml          # tab-completable
    python scheduler_status.py method_compare --watch            # refresh every 2s
    python scheduler_status.py method_compare --watch 5          # refresh every 5s

Same semantics as `python -m scheduler.scheduler status <exp>` plus a watch
mode that re-renders the status in place. Ctrl-C exits watch mode.
"""
import argparse
import time

from scheduler.scheduler import cmd_status


def main():
    p = argparse.ArgumentParser(
        prog="scheduler_status",
        description="Show status of a ratsim experiment.")
    p.add_argument("exp", help="Experiment id (looked up in defs/) or path to a def yaml")
    p.add_argument(
        "--watch", "-w", type=float, nargs="?", const=2.0, default=None,
        metavar="SECONDS",
        help="Refresh in place every SECONDS (default 2). Ctrl-C to exit.")
    args = p.parse_args()

    if args.watch is None:
        cmd_status(args)
        return

    try:
        while True:
            # ANSI: clear screen + cursor home. Works on any modern terminal.
            print("\033[2J\033[H", end="")
            cmd_status(args)
            print(f"\n(watch every {args.watch:g}s — Ctrl-C to exit)")
            time.sleep(args.watch)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
