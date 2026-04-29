#!/usr/bin/env python
"""Run / resume an experiment.

    python scheduler_run.py method_compare
    python scheduler_run.py defs/method_compare.yaml          # tab-completable
    python scheduler_run.py method_compare --machine gpu_example
    python scheduler_run.py method_compare --step-multiplier 0.01

Same semantics as `python -m scheduler.scheduler run <exp>` — this just lets
you skip the `run` subcommand on the CLI.
"""
import argparse
import os

from scheduler.scheduler import cmd_run


def main():
    p = argparse.ArgumentParser(
        prog="scheduler_run",
        description="Run / resume a ratsim experiment.")
    p.add_argument("exp", help="Experiment id (looked up in defs/) or path to a def yaml")
    p.add_argument(
        "--machine", default=os.environ.get("RATSIM_SCHEDULER_MACHINE"),
        help="Machine config: bare name (resolved against scheduler/machines/) or path. "
             "Defaults to scheduler/machines/default.yaml. "
             "Can also be set via $RATSIM_SCHEDULER_MACHINE.")
    p.add_argument(
        "--step-multiplier", type=float, default=None,
        help="Override the def's step_multiplier (e.g. 0.01 for smoke tests).")
    p.add_argument(
        "--restart", action="store_true",
        help="Wipe results/experiments/<exp_id>/ before starting "
             "(equivalent to rm -rf + run). Default behavior is to resume.")
    p.add_argument(
        "--use-port-9000", action="store_true", dest="use_port_9000",
        help="Also consider port 9000 for one n_envs=1 dispatch at a time, "
             "but only when Unity is alive on 9000 (TCP probe at dispatch "
             "time). Useful for manually launching a Unity GUI on 9000 and "
             "watching one training instance live; other dispatches still "
             "go to 9100+ as usual.")
    cmd_run(p.parse_args())


if __name__ == "__main__":
    main()
