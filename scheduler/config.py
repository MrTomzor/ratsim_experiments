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
# override these (would silently break dispatch).
RESERVED_ARGS = {
    "def", "variation", "run_folder", "name",
    "start_stage", "end_stage", "step_multiplier",
    "metaseed", "base_port", "method",
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
]


@dataclass
class MethodProfile:
    """A method's machine-specific resource requirements + arg overrides."""
    needs: dict[str, int] = field(default_factory=dict)
    args: dict = field(default_factory=dict)
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
        profiles[name] = MethodProfile(
            needs=dict(profile.get("needs") or {}),
            args=dict(profile.get("args") or {}),
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


def validate_against_machine(exp: ExperimentDef, machine: MachineConfig) -> None:
    """Fail fast on misconfigurations: missing profile, impossible needs,
    unset python env vars."""
    for method in exp.methods:
        if method.name not in machine.method_profiles:
            raise ValueError(
                f"method '{method.name}' has no profile in machine config "
                f"'{machine.source}'. Add a method_profiles.{method.name} entry.")
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
        resolve_python(method, profile)
