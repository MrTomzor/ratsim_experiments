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

from scheduler.scheduler import _csv_list, cmd_run


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
        "--mode", choices=("bfs", "dfs"), default=None,
        help="Override the def's dispatch order for this job. dfs finishes "
             "runs one at a time; bfs advances every run through the early "
             "stages first. Not inferred from anything else.")
    p.add_argument(
        "--methods", type=_csv_list, default=None, metavar="M1,M2",
        help="Run only these of the def's methods (comma-separated). Lets one "
             "def be split across jobs on different partitions — e.g. "
             "`--machine rci --methods ppo` on a CPU node and `--machine "
             "rci_gpu2 --methods dreamer` on a GPU node, both feeding the same "
             "exp_id. Each filter gets its own state file, so the two jobs do "
             "not reap each other's children. submit.sh derives this for you.")
    p.add_argument(
        "--variations", type=_csv_list, default=None, metavar="V1,V2",
        help="Run only these of the def's variations (comma-separated). For "
             "continuing one cell of a ladder def without paying for the "
             "others, e.g. `--variations consec4`. Combines with --methods; "
             "each filter combination gets its own state file, so a job you "
             "start later does not reap a sibling's children. submit.sh "
             "passes this through from its own --variations.")
    p.add_argument(
        "--restart", action="store_true",
        help="Wipe results/experiments/<exp_id>/ before starting "
             "(equivalent to rm -rf + run). Under --methods / --variations, "
             "wipes only that job's runs. Default behavior is to resume.")
    p.add_argument(
        "--use-port-9000", action="store_true", dest="use_port_9000",
        help="Demo mode: forces n_envs=1 on every method (so the run is NOT "
             "comparable with a normal one) and hands "
             "port 9000 to one dispatch at a time, but only when Unity is "
             "alive on 9000 (TCP probe at dispatch time). Useful for manually "
             "launching a Unity GUI on 9000 and watching one training "
             "instance learn live.")
    p.add_argument(
        "--show-console-prints", action="store_true", dest="show_console_prints",
        help="Stream each subprocess's stdout/stderr to this console "
             "(prefixed per run) in addition to writing it to the per-stage "
             "log file. Off by default (logs go to file only).")
    cmd_run(p.parse_args())


if __name__ == "__main__":
    main()
