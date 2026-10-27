"""Integration test of probing/ against REAL Unity instances, several at once.

    RATSIM_UNITY_BIN=/path/to/ForagerSimBuild.x86_64 \
        ~/ratvenv/dreamer_venv/bin/python tests/test_probe_unity.py

Skipped without RATSIM_UNITY_BIN. Explorer episodes only (no checkpoints
needed); each is a full 5000-step episode, so the whole file takes minutes.
The main venv (targets, probes) is $PPO_PYTHON_PATH or ~/ratvenv/venv.

What it establishes, each a property the cluster job depends on:
  1. 3 workers and 1 worker record the same worlds for the same plan.
  2. A second collect on the same dataset is refused by the lock.
  3. SIGKILL of the whole collect (a walltime kill) loses nothing: the
     re-run records only what is missing, keeps what was finished
     byte-for-byte, and seals a dataset that verifies.
  4. targets -> encode -> train_probe -> analyze run on the real data.
  5. Policy episodes from a checkpoint in 3 workers at once, each its own JAX
     process on the one GPU (the cluster's L0/L1 collection), with a
     random-weight size1m checkpoint standing in for a trained one.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from probing.core import dataset  # noqa: E402

MAIN_PY = os.environ.get("PPO_PYTHON_PATH",
                         str(Path.home() / "ratvenv/venv/bin/python"))
DEF = ROOT / "defs" / "memory_orthomaze.yaml"
STOP = ROOT.parent / "ratsim" / "scripts" / "stop_ratsim_headless.sh"


class Env:
    def __init__(self, tmp: Path):
        self.tmp = tmp
        self.exp = tmp / "exp"
        self.exp.mkdir()
        shutil.copy(DEF, self.exp / "experiment.yaml")
        self.vars = {**os.environ, "RATSIM_XVFB": "1",
                     "RATSIM_RUNDIR": str(tmp / "run"),
                     "RATSIM_BOOT_TIMEOUT": "180",
                     "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                     "PYTHONUNBUFFERED": "1"}
        (tmp / "run").mkdir()

    def collect_cmd(self, out, n, workers):
        return [sys.executable, "-u", "-m", "probing.s1_collect.collect", "--out", str(out),
                "--exp_dir", str(self.exp), "--variation", "baseline",
                "--explorer", str(n), "--workers", str(workers),
                "--metaseed", "4242", "--spawn"]

    def run(self, cmd, log, check=True, python=None):
        with open(self.tmp / log, "w") as f:
            p = subprocess.run(cmd if python is None else [python, *cmd],
                               cwd=ROOT, env=self.vars, stdout=f,
                               stderr=subprocess.STDOUT)
        if check and p.returncode:
            raise AssertionError(f"{cmd[:4]} failed ({p.returncode}):\n"
                                 + (self.tmp / log).read_text()[-3000:])
        return p.returncode

    def stop_unity(self):
        subprocess.run([str(STOP), "--all"], env=self.vars,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _episodes(root):
    return {dataset.episode_id(p): dataset.load_episode(p)
            for p in dataset.list_episodes(root)}


def step_parallel_equals_serial(env: Env):
    a, b = env.tmp / "par", env.tmp / "ser"
    t0 = time.time()
    env.run(env.collect_cmd(a, 6, 3), "par.log")
    t_par = time.time() - t0
    t0 = time.time()
    env.run(env.collect_cmd(b, 6, 1), "ser.log")
    t_ser = time.time() - t0
    env.stop_unity()
    ea, eb = _episodes(a), _episodes(b)
    assert sorted(ea) == sorted(eb) == list(range(6))
    assert dataset.verify(a) and dataset.verify(b)
    same_traj = 0
    for i in range(6):
        assert int(ea[i]["world_seed"]) == int(eb[i]["world_seed"]), i
        # The world and spawn come from the seed alone: the reset
        # observation must match whichever worker recorded it.
        for k in ("pose", "lidar", "gps", "compass", "ranges"):
            np.testing.assert_array_equal(ea[i][k][0], eb[i][k][0], err_msg=f"ep{i} {k}")
        same_traj += all(np.array_equal(ea[i][k], eb[i][k])
                         for k in ("pose", "action_0", "action_1"))
    print(f"  parallel {t_par:.0f}s vs serial {t_ser:.0f}s; whole trajectories "
          f"identical in {same_traj}/6 episodes (Unity/explorer determinism)")
    return a


def step_second_collect_is_refused(env: Env):
    out = env.tmp / "lock"
    cmd = env.collect_cmd(out, 2, 1)
    p1 = subprocess.Popen(cmd, cwd=ROOT, env=env.vars,
                          stdout=open(env.tmp / "lock1.log", "w"),
                          stderr=subprocess.STDOUT)
    while not (out / ".collect.lock").exists():
        time.sleep(0.2)
    time.sleep(2)  # p1 holds it now
    rc2 = env.run(cmd, "lock2.log", check=False)
    rc1 = p1.wait()
    env.stop_unity()
    log2 = (env.tmp / "lock2.log").read_text()
    assert rc2 != 0 and "held by another process" in log2, log2[-2000:]
    assert rc1 == 0, (env.tmp / "lock1.log").read_text()[-2000:]
    assert dataset.verify(out)


def step_walltime_kill_then_resume(env: Env, reference: Path):
    out = env.tmp / "killed"
    p = subprocess.Popen(env.collect_cmd(out, 6, 3), cwd=ROOT, env=env.vars,
                         stdout=open(env.tmp / "kill1.log", "w"),
                         stderr=subprocess.STDOUT, start_new_session=True)
    while len(dataset.list_episodes(out)) < 2:
        assert p.poll() is None, (env.tmp / "kill1.log").read_text()[-2000:]
        time.sleep(0.5)
    os.killpg(p.pid, signal.SIGKILL)  # parent and workers, mid-episode
    p.wait()
    env.stop_unity()  # SIGKILL skipped the launcher's atexit cleanup
    kept = {p.name: p.read_bytes() for p in dataset.list_episodes(out)}
    assert 2 <= len(kept) < 6, sorted(kept)
    meta = dataset.load_json(out / "meta.json")
    assert "fingerprint" not in meta, "an incomplete dataset must not be sealed"
    try:
        dataset.verify(out)
    except ValueError:
        pass
    else:
        raise AssertionError("an interrupted dataset verified")
    (dataset.episode_dir(out) / ".ep0005.npz.999.1.tmp").write_bytes(b"torn")

    env.run(env.collect_cmd(out, 6, 3), "kill2.log")
    env.stop_unity()
    log = (env.tmp / "kill2.log").read_text()
    assert f"{6 - len(kept)} to record" in log, log[-2000:]
    dataset.verify(out)
    for name, data in kept.items():  # finished episodes are never rewritten
        assert (dataset.episode_dir(out) / name).read_bytes() == data, name
    assert not list(dataset.episode_dir(out).glob(".*.tmp")), "temp not cleaned"
    ref, got = _episodes(reference), _episodes(out)
    for i in range(6):
        assert int(got[i]["world_seed"]) == int(ref[i]["world_seed"]), i
        np.testing.assert_array_equal(got[i]["pose"][0], ref[i]["pose"][0])


def step_pipeline_on_real_data(env: Env, ds: Path):
    env.run(["-m", "probing.s2_targets.targets", "--dataset", str(ds), "--workers", "6"],
            "targets.log", python=MAIN_PY)
    fake = make_fake_run(env.tmp / "exp", ds)
    states = env.tmp / "states"
    env.run(["-m", "probing.s3_encode.encode", "--dataset", str(ds), "--out", str(states),
             "--exp_dir", str(env.tmp / "exp"), "--runs", f"runs/{fake.name}",
             "--stages", "0", "--random_init", "--platform", "cpu",
             "--batch", "3"], "encode.log", python=sys.executable)
    probes = env.tmp / "probes"
    jobs = [(f"{s.stem}", ["--features", "state", "--states", str(s)])
            for s in sorted(states.glob("*.npz"))]
    jobs += [("obs", ["--features", "obs"]), ("const", ["--features", "const"])]
    for model, feats in jobs:
        for target in ("layout", "coverage", "start"):
            env.run(["-m", "probing.s4_probe.train_probe", "--dataset", str(ds), *feats,
                     "--target", target, "--out", str(probes / model / target),
                     "--epochs", "3"], f"probe_{model}_{target}.log",
                    python=MAIN_PY)
    env.run(["-m", "probing.s5_analysis.analyze", "--probes", str(probes), "--dataset", str(ds)],
            "analyze.log", python=MAIN_PY)
    rows = (probes / "analysis" / "probes.csv").read_text().splitlines()
    assert len(rows) == 1 + 3 * len(jobs), rows
    print("  " + "\n  ".join((env.tmp / "analyze.log").read_text()
                             .split("== layout ==")[1].splitlines()[:6]))


def step_policy_workers_share_the_gpu(env: Env, platform: str):
    out = env.tmp / "policy"
    cmd = env.collect_cmd(out, 0, 3)
    cmd[cmd.index("--explorer") + 1] = "0"
    cmd += ["--policy", "runs/fake__dreamer__seed0:0:3", "--platform", platform]
    env.run(cmd, "policy.log")
    env.stop_unity()
    dataset.verify(out)
    eps = _episodes(out)
    assert sorted(eps) == [0, 1, 2]
    log = (env.tmp / "policy.log").read_text()
    loaded = log.count("loaded fake__dreamer__seed0 stage 0")
    assert loaded == 3, f"each worker loads the checkpoint once, got {loaded}"
    for i, e in eps.items():
        assert str(e["source"]) == "policy" and str(e["policy_run"]) == "fake__dreamer__seed0"
        acts = np.stack([e["action_0"][:-1], e["action_1"][:-1]])
        assert len(np.unique(acts[1])) > 1, f"ep{i}: policy never varied its turn"
    print(f"  3 policy workers on {platform}: "
          + ", ".join(f"ep{i} {len(e['pose']) - 1} steps" for i, e in eps.items()))


def make_fake_run(exp: Path, ds: Path) -> Path:
    """A run dir with a size1m checkpoint built for the dataset's spaces."""
    from probing.core import agent_io
    import elements
    run = exp / "runs" / "fake__dreamer__seed0"
    (run / "checkpoints").mkdir(parents=True)
    (run / "run_config.json").write_text(json.dumps(
        {"size": "size1m", "method_overrides": {"jax.platform": "cpu"}}))
    spaces = dataset.load_json(ds / "spaces.json")
    agent = agent_io.build_agent(agent_io.make_config(run, platform="cpu"),
                                 dataset.spaces_from_json(spaces["obs"]),
                                 dataset.spaces_from_json(spaces["act"]))
    cp = elements.Checkpoint()
    cp.agent = agent
    cp.save(str(run / "checkpoints" / "stage_0"))
    (run / "checkpoints" / "stage_0.done").touch()
    return run


def main() -> int:
    if not os.environ.get("RATSIM_UNITY_BIN"):
        print("SKIP: set RATSIM_UNITY_BIN to run against real Unity")
        return 0
    base = Path(os.environ.get("PROBE_TEST_DIR") or tempfile.mkdtemp(prefix="probe_unity_"))
    base.mkdir(parents=True, exist_ok=True)
    env = Env(Path(tempfile.mkdtemp(dir=base)))
    print(f"work dir {env.tmp}")
    ref = None
    failed = 0
    try:
        for name, fn in [
                ("parallel_equals_serial", lambda: step_parallel_equals_serial(env)),
                ("second_collect_is_refused", lambda: step_second_collect_is_refused(env)),
                ("walltime_kill_then_resume", lambda: step_walltime_kill_then_resume(env, ref)),
                ("pipeline_on_real_data", lambda: step_pipeline_on_real_data(env, ref)),
                ("policy_workers_share_the_gpu",
                 lambda: step_policy_workers_share_the_gpu(
                     env, os.environ.get("PROBE_TEST_PLATFORM", "cuda")))]:
            t0 = time.time()
            try:
                result = fn()
                if name == "parallel_equals_serial":
                    ref = result
                print(f"PASS {name} ({time.time() - t0:.0f}s)")
            except Exception as e:  # noqa: BLE001
                failed += 1
                import traceback
                print(f"FAIL {name}: {type(e).__name__}: {e}")
                traceback.print_exc(limit=4)
                if ref is None:
                    break
    finally:
        env.stop_unity()
    print(f"\n{5 - failed}/5 passed (logs in {env.tmp})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
