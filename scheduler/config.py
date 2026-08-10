"""Machine-config loading + python/script resolution + cross-validation.

The experiment-def schema and loader live one level up in
`ratsim_experiments/experiment_defs.py` so that train.py / train_dreamerv3.py
can import them without going through the scheduler package. This module
adds the scheduler-only pieces:

  * MachineConfig: resources + per-method profiles (declared in
    scheduler/machines/*.yaml). Defaults to default.yaml; overridable via
    --machine or $RATSIM_SCHEDULER_MACHINE.
  * resolve_python / resolve_train_script: figure out which interpreter and
    script to launch for a given method, taking method/profile overrides
    and falling back to environment-variable conventions.
  * validate_against_machine: fail fast on missing profiles, impossible
    resource needs, or unset python env vars.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from experiment_defs import (
    DEFAULT_N_ENVS,
    ExperimentDef,
    MethodSpec,
    StageSpec,
    VariationSpec,
    as_preset_list,
    find_variation,
    load_experiment_def,
    resolve_agent_preset,
    resolve_def_path,
    resolve_stage_world,
    resolve_task_preset,
    snapshot_experiment,
)


# Method-name → env var holding the python interpreter path. The user sets
# these once per machine (e.g. in ~/.bashrc); the scheduler reads them at
# dispatch time so we never hardcode paths in checked-in YAML.
DEFAULT_PYTHON_ENV = {
    "ppo": "PPO_PYTHON_PATH",
    "recurrent_ppo": "PPO_PYTHON_PATH",
    "cnn_ppo": "PPO_PYTHON_PATH",
    "cnn_recurrent_ppo": "PPO_PYTHON_PATH",
    "dreamer": "DREAMER_PYTHON_PATH",
}

DEFAULT_TRAIN_SCRIPT = {
    "ppo": "train.py",
    "recurrent_ppo": "train.py",
    "cnn_ppo": "train.py",
    "cnn_recurrent_ppo": "train.py",
    "dreamer": "train_dreamerv3.py",
}

# train.py needs `method=<name>` on the CLI; train_dreamerv3.py infers.
SCRIPTS_NEEDING_METHOD_ARG = {"train.py"}

# CLI keys the scheduler controls — user method args / common args may not
# override these (would silently break dispatch). `n_envs` is sourced from the
# experiment def (see MethodSpec.n_envs), since vectorization changes what a run
# learns and must therefore stay fixed across machines.
RESERVED_ARGS = {
    "def", "variation", "run_folder", "name",
    "start_stage", "end_stage", "step_multiplier",
    "metaseed", "base_port", "method", "n_envs",
}


# Re-export for convenience so scheduler.py can do `from . import config as cfg`
# and still get the experiment-def types in one place.
__all__ = [
    "ExperimentDef", "MethodSpec", "StageSpec", "VariationSpec",
    "as_preset_list", "find_variation", "load_experiment_def",
    "resolve_agent_preset", "resolve_def_path", "resolve_stage_world",
    "resolve_task_preset", "snapshot_experiment",
    "DEFAULT_PYTHON_ENV", "DEFAULT_TRAIN_SCRIPT",
    "SCRIPTS_NEEDING_METHOD_ARG", "RESERVED_ARGS",
    "MethodProfile", "MachineConfig",
    "resolve_machine_path", "load_machine_config",
    "resolve_python", "resolve_train_script",
    "validate_against_machine",
    "CPU_SLOT_PER_ENV",
]


# Sizing rule of thumb, in cpu_slot units (= threads on the RCI nodes). Below
# roughly this many threads per Unity env, throughput falls off a cliff rather
# than degrading gracefully: the same 4-env PPO run measured 17 fps on 4
# threads, 251 on 8 and 688 on 16. Used only for a startup warning — a machine
# is free to be undersized on purpose.
CPU_SLOT_PER_ENV = 4


@dataclass
class MethodProfile:
    """A method's machine-specific resource requirements + arg overrides.

    Deliberately does NOT carry n_envs: vectorization lives on the experiment
    def (MethodSpec.n_envs) because it changes what a run learns, not just how
    fast it runs. What belongs here is capacity — how much of this box one run
    of this method is allowed to take.

    max_ram_gb is an optional safety net: if the process-tree RSS exceeds it,
    the scheduler SIGTERMs the job and re-dispatches it (relying on per-stage
    .done markers + the method's internal checkpointing for resume). RAM-kills
    don't count toward the consecutive-failure budget. Useful for dreamer,
    which has a known but unfixed memory leak."""
    needs: dict[str, int] = field(default_factory=dict)
    args: dict = field(default_factory=dict)
    max_ram_gb: float | None = None
    python_env: str | None = None
    train_script: str | None = None


@dataclass
class MachineConfig:
    source: Path
    resources: dict[str, int] = field(default_factory=dict)
    method_profiles: dict[str, MethodProfile] = field(default_factory=dict)


def resolve_machine_path(machine_dir: Path, override: str | None) -> Path:
    if override is None:
        return machine_dir / "default.yaml"
    if "/" in override or override.endswith((".yaml", ".yml")):
        return Path(override)
    return machine_dir / f"{override}.yaml"


def load_machine_config(machine_dir: Path, override: str | None = None) -> MachineConfig:
    path = resolve_machine_path(machine_dir, override)
    if not path.exists():
        raise FileNotFoundError(
            f"machine config not found at {path}. Pass --machine <name|path> "
            f"or set $RATSIM_SCHEDULER_MACHINE. See scheduler/machines/default.yaml "
            f"and gpu_example.yaml.")
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    profiles = {}
    for name, profile in (raw.get("method_profiles") or {}).items():
        if "n_envs" in profile:
            # Hard error rather than a silent ignore: a stale `n_envs: 1` here
            # used to be the thing that made a run single-env, so quietly
            # dropping it would change results without saying so.
            raise ValueError(
                f"{path}: method_profiles.{name} sets n_envs — that moved to "
                f"the experiment def. Delete it here and set `n_envs:` on the "
                f"method in the def's `methods:` list (default "
                f"{DEFAULT_N_ENVS}). It changes what the run learns, so it has "
                f"to be the same on every machine.")
        max_ram = profile.get("max_ram_gb")
        profiles[name] = MethodProfile(
            needs=dict(profile.get("needs") or {}),
            args=dict(profile.get("args") or {}),
            max_ram_gb=float(max_ram) if max_ram is not None else None,
            python_env=profile.get("python_env"),
            train_script=profile.get("train_script"),
        )
    return MachineConfig(
        source=path,
        resources=dict(raw.get("resources") or {}),
        method_profiles=profiles,
    )


def resolve_python(method: MethodSpec, profile: MethodProfile) -> str:
    var = method.python_env or profile.python_env or DEFAULT_PYTHON_ENV.get(method.name)
    if var is None:
        raise ValueError(
            f"method '{method.name}': no python_env mapping. Set 'python_env: <VAR>' "
            f"in the experiment def or machine config.")
    val = os.environ.get(var)
    if not val:
        raise EnvironmentError(
            f"env var ${var} (python for method '{method.name}') is unset. "
            f"Add `export {var}=...` to your shell rc.")
    return val


def resolve_train_script(method: MethodSpec, profile: MethodProfile) -> str:
    return (method.train_script or profile.train_script
            or DEFAULT_TRAIN_SCRIPT.get(method.name, "train.py"))


def _configs_declaring(machine_dir: Path, method_name: str) -> list[str]:
    """Names of machine configs in `machine_dir` that have a profile for
    `method_name`. Best-effort: this only ever decorates an error message, so a
    sibling that is itself unparseable is skipped rather than raised over."""
    out = []
    for path in sorted(machine_dir.glob("*.y*ml")):
        try:
            with open(path) as f:
                raw = yaml.safe_load(f) or {}
            if method_name in (raw.get("method_profiles") or {}):
                out.append(path.stem)
        except Exception:
            continue
    return out


def validate_against_machine(exp: ExperimentDef, machine: MachineConfig) -> None:
    """Fail fast on misconfigurations: missing profile, impossible needs,
    unset python env vars, n_envs > port window. Warns (does not fail) when a
    profile grants too few cpu_slots for the def's n_envs."""
    for method in exp.methods:
        if method.name not in machine.method_profiles:
            # A missing profile is usually the wrong machine config rather than
            # an incomplete one — e.g. dreamer against rci.yaml, which is the
            # CPU-node shape and deliberately carries no GPU methods. Point at
            # a sibling that does have it before suggesting the user write one.
            alts = _configs_declaring(machine.source.parent, method.name)
            alts = [a for a in alts if a != machine.source.stem]
            hint = (f" These declare it: {', '.join(alts)} — e.g. "
                    f"--machine {alts[0]}." if alts else
                    f" Add a method_profiles.{method.name} entry.")
            raise ValueError(
                f"method '{method.name}' has no profile in machine config "
                f"'{machine.source}' (it has: "
                f"{', '.join(sorted(machine.method_profiles)) or 'none'})."
                + hint)
        profile = machine.method_profiles[method.name]
        for k, v in profile.needs.items():
            if k not in machine.resources:
                raise ValueError(
                    f"method '{method.name}' needs resource '{k}' which is not "
                    f"declared in machine resources: {list(machine.resources)}")
            if v > machine.resources[k]:
                raise ValueError(
                    f"method '{method.name}' needs {k}={v} but capacity is "
                    f"{machine.resources[k]} — would never dispatch.")
        if method.n_envs > 10:
            # Each dispatched job gets a 10-wide unity port window
            # (PortAllocator window_size=10). n_envs > 10 would overflow into
            # the next job's window.
            raise ValueError(
                f"method '{method.name}': n_envs={method.n_envs} exceeds the "
                f"per-job port window of 10. Bump PortAllocator.window_size "
                f"in scheduler.py or reduce n_envs.")
        # Warn, don't fail: an undersized box still produces valid results, it
        # just produces them slowly, and the user may well know that. Silence
        # would be worse — the failure mode is a run that looks merely slow.
        granted = profile.needs.get("cpu_slot")
        wanted = CPU_SLOT_PER_ENV * method.n_envs
        if granted is not None and granted < wanted:
            print(f"[scheduler] WARNING: method '{method.name}' runs "
                  f"n_envs={method.n_envs} on cpu_slot={granted}. Below "
                  f"~{CPU_SLOT_PER_ENV} slots per env ({wanted} here) "
                  f"throughput drops sharply — raise needs.cpu_slot in "
                  f"{machine.source.name}, or lower n_envs in the def.")
        resolve_python(method, profile)
