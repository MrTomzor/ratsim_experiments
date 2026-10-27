"""Offline probing of DreamerV3's hidden state (Memory Maze protocol).

Pasukonis, Lillicrap & Hafner, "Evaluating Long-Term Memory in 3D Mazes"
(arXiv 2210.13383), section 3.3: a world model is trained as usual, then a
SEPARATE probe network learns to read privileged facts about the environment
out of its frozen state. No gradient reaches the model, so the probe can only
decode what the state already holds.

Here the protocol runs entirely outside training, over checkpoints that
already exist. The subpackages are the steps, in order (README.md has the
inputs and outputs of each):

    core/           dataset.py: episode files, sealing, locks, splits;
                    agent_io.py: a run's agent + stage checkpoint.
    s1_collect/     collect.py: Unity + a policy (agent checkpoint or the
                    ScriptedExplorer in explorer.py) -> one shared dataset of
                    episodes (obs, actions, true pose, raw lidar). Dreamer venv.
    s2_targets/     targets.py: dataset -> probe targets (seen-map crop,
                    geodesic distance to start), by replaying the scans through
                    the same ExplorationTracker the reward uses. Main venv.
                    layouts.py: dataset + the layout-dump Unity build -> the
                    TRUE maze and boxes of every episode, verified against its
                    own lidar and pickups. global_targets.py: layouts ->
                    ground-truth targets (whole map + boxes still there) and the
                    seen / in-view / age masks. Main venv.
    s3_encode/      encode.py: checkpoint x dataset -> the model's state at
                    every step, teacher-forced through observe(). No Unity.
                    Dreamer venv.
    s4_probe/       train_probe.py: state or obs features x targets -> probe
                    metrics. Main venv (torch). The ground-truth targets are
                    cropped around the agent and scored in global_score.py.
    s5_analysis/    analyze.py: metrics -> tables and curves (analyze_global.py
                    for the ground-truth targets). Main venv.
    tools/          inspect_*.py: PNGs of every step (layouts + their
                    verification, targets + masks, predictions vs truth);
                    diagnose.py, dream.py, coverage_arms.py.
    jobs/           the whole pipeline as one cluster job (SLURM / PBS).

Every model sees the same episodes, so differences between models are
differences in what the state holds, not in which trajectories a policy chose.
"""
