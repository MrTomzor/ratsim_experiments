"""Experiment-def schema + loader.

An experiment def (`defs/<exp_id>.yaml`) is the single source of truth for an
experiment: which agent / task / world to use, the curriculum of stages, the
methods to compare, the variations (per-axis overrides) to test, and the
seed count. The scheduler dispatches one (run, stage) at a time; train.py
consumes the same def to resolve a single (variation, stage)'s configs.

Lives at the top of ratsim_experiments so train.py / train_dreamerv3.py /
scheduler all import from one place.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


# --- Coercion helper -------------------------------------------------------

def as_preset_list(v) -> list[str]:
    """Coerce `string | list | None` → list of preset names."""
    if v is None:
        return []
    if isinstance(v, str):
        return [v]
    if isinstance(v, list):
        return [str(x) for x in v]
    raise TypeError(
        f"expected string or list of preset names, got {type(v).__name__}: {v!r}")


# --- Schema dataclasses ----------------------------------------------------

@dataclass
class StageSpec:
    """One stage of the experiment (shared across all runs).

    `world_preset=None` means "fall back to variation.world_preset →
    experiment-level world_preset" at resolution time.
    """
    steps: int
    world_preset: list[str] | None = None

    def to_dict(self) -> dict:
        d: dict = {"steps": self.steps}
        if self.world_preset is not None:
            d["world_preset"] = list(self.world_preset)
        return d


@dataclass
class VariationSpec:
    """Override bundle on top of the experiment defaults."""
    name: str
    agent_preset: list[str] | None = None
    task_preset: list[str] | None = None
    world_preset: list[str] | None = None
    method_args: dict = field(default_factory=dict)


@dataclass
class MethodSpec:
    name: str
    n_seeds: int
    args: dict = field(default_factory=dict)
    python_env: str | None = None
    train_script: str | None = None


@dataclass
class ExperimentDef:
    exp_id: str
    source: Path
    agent_preset: list[str]
    task_preset: list[str]
    world_preset: list[str]   # default world (used when stages don't specify)
    stages: list[StageSpec]
    methods: list[MethodSpec]
    variations: list[VariationSpec]   # always non-empty (default [{name: baseline}])
    mode: str                 # "bfs" or "dfs"
    step_multiplier: float = 1.0
    common_args: dict = field(default_factory=dict)


# --- Loading ---------------------------------------------------------------

def _expand_stages(raw: dict, exp_world_preset: list[str], src: Path) -> list[StageSpec]:
    """Resolve the short form (`total_steps` + `n_stages`) or long form
    (`stages:` list). Mutually exclusive."""
    has_short = "total_steps" in raw or "n_stages" in raw
    has_long = "stages" in raw
    if has_short and has_long:
        raise ValueError(
            f"{src}: cannot mix `stages:` (long form) with `total_steps:` / "
            f"`n_stages:` (short form). Pick one.")
    if not has_short and not has_long:
        raise ValueError(
            f"{src}: experiment def must specify either `stages:` or "
            f"`total_steps:` + `n_stages:`.")

    if has_short:
        if "total_steps" not in raw or "n_stages" not in raw:
            raise ValueError(
                f"{src}: short form requires both `total_steps:` and `n_stages:`.")
        total = int(raw["total_steps"])
        n = int(raw["n_stages"])
        if n <= 0:
            raise ValueError(f"{src}: n_stages must be > 0, got {n}")
        if total <= 0:
            raise ValueError(f"{src}: total_steps must be > 0, got {total}")
        per = total // n
        if per * n != total:
            print(f"[experiment_def] WARNING: {src.name}: total_steps={total} "
                  f"not divisible by n_stages={n}; truncating each stage to "
                  f"{per} steps (loss = {total - per*n}).")
        return [StageSpec(steps=per, world_preset=None) for _ in range(n)]

    raw_stages = raw["stages"] or []
    if not raw_stages:
        raise ValueError(f"{src}: `stages:` must contain at least one stage")
    out = []
    for i, s in enumerate(raw_stages):
        if "steps" not in s:
            raise ValueError(f"{src}: stage {i} missing `steps:`")
        # accept legacy plural `world_presets:`
        wp = s.get("world_preset", s.get("world_presets"))
        wp_list = as_preset_list(wp) if wp is not None else None
        out.append(StageSpec(steps=int(s["steps"]), world_preset=wp_list))
    return out


def _parse_methods(raw_methods, default_n_seeds: int, src: Path) -> list[MethodSpec]:
    if not raw_methods:
        raise ValueError(f"{src}: at least one method required under `methods:`")
    out: list[MethodSpec] = []
    seen = set()
    for i, m in enumerate(raw_methods):
        if not isinstance(m, dict) or "name" not in m:
            raise ValueError(f"{src}: methods[{i}] must be a dict with 'name'")
        name = m["name"]
        if name in seen:
            raise ValueError(f"{src}: duplicate method '{name}'")
        seen.add(name)
        out.append(MethodSpec(
            name=name,
            n_seeds=int(m.get("n_seeds", default_n_seeds)),
            args=dict(m.get("args") or {}),
            python_env=m.get("python_env"),
            train_script=m.get("train_script"),
        ))
    return out


def _parse_variations(raw, src: Path) -> list[VariationSpec]:
    if raw is None:
        return [VariationSpec(name="baseline")]
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{src}: `variations:` must be a non-empty list (or omitted)")
    out: list[VariationSpec] = []
    seen = set()
    for i, v in enumerate(raw):
        if not isinstance(v, dict) or "name" not in v:
            raise ValueError(f"{src}: variations[{i}] must be a dict with 'name'")
        name = v["name"]
        if name in seen:
            raise ValueError(f"{src}: duplicate variation '{name}'")
        seen.add(name)
        out.append(VariationSpec(
            name=name,
            agent_preset=as_preset_list(v["agent_preset"]) if "agent_preset" in v else None,
            task_preset=as_preset_list(v["task_preset"]) if "task_preset" in v else None,
            world_preset=as_preset_list(v["world_preset"]) if "world_preset" in v else None,
            method_args=dict(v.get("method_args") or {}),
        ))
    return out


def _validate_world_resolves(exp: ExperimentDef) -> None:
    """Every (stage, variation) pair must resolve to a non-empty world_preset."""
    for s_idx, stage in enumerate(exp.stages):
        if stage.world_preset:
            continue
        for var in exp.variations:
            if var.world_preset:
                continue
            if not exp.world_preset:
                raise ValueError(
                    f"{exp.source}: stage {s_idx} under variation '{var.name}' "
                    f"has no world_preset. Set top-level `world_preset:`, "
                    f"per-variation `world_preset:`, or per-stage `world_preset:`.")


def load_experiment_def(path: Path) -> ExperimentDef:
    path = Path(path)
    with open(path) as f:
        raw = yaml.safe_load(f) or {}

    exp_id = path.stem
    agent_preset = as_preset_list(raw.get("agent_preset", "sphereagent_2d_lidar"))
    task_preset = as_preset_list(raw.get("task_preset", "default"))
    world_preset = as_preset_list(raw.get("world_preset"))

    stages = _expand_stages(raw, world_preset, path)
    default_n_seeds = int(raw.get("seeds", 1))
    methods = _parse_methods(raw.get("methods"), default_n_seeds, path)
    variations = _parse_variations(raw.get("variations"), path)

    mode = raw.get("mode", "bfs")
    if mode not in ("bfs", "dfs"):
        raise ValueError(f"{path}: mode must be 'bfs' or 'dfs', got {mode!r}")

    exp = ExperimentDef(
        exp_id=exp_id,
        source=path,
        agent_preset=agent_preset,
        task_preset=task_preset,
        world_preset=world_preset,
        stages=stages,
        methods=methods,
        variations=variations,
        mode=mode,
        step_multiplier=float(raw.get("step_multiplier", 1.0)),
        common_args=dict(raw.get("common_args") or {}),
    )
    _validate_world_resolves(exp)
    return exp


def resolve_def_path(defs_dir: Path, name_or_path: str) -> Path:
    """Bare name → defs_dir/<name>.yaml; anything path-like → as-is."""
    if "/" in name_or_path or name_or_path.endswith((".yaml", ".yml")):
        return Path(name_or_path)
    return defs_dir / f"{name_or_path}.yaml"


# --- Resolution helpers ----------------------------------------------------

def find_variation(exp: ExperimentDef, name: str) -> VariationSpec:
    for v in exp.variations:
        if v.name == name:
            return v
    raise KeyError(
        f"variation '{name}' not found in {exp.source}. "
        f"Available: {[v.name for v in exp.variations]}")


def resolve_agent_preset(exp: ExperimentDef, var: VariationSpec) -> list[str]:
    return list(var.agent_preset if var.agent_preset is not None else exp.agent_preset)


def resolve_task_preset(exp: ExperimentDef, var: VariationSpec) -> list[str]:
    return list(var.task_preset if var.task_preset is not None else exp.task_preset)


def resolve_stage_world(stage: StageSpec, var: VariationSpec,
                        exp: ExperimentDef) -> list[str]:
    """Pick the world_preset list for this (stage, variation) pair.

    Precedence (highest → lowest): per-stage explicit → per-variation → per-exp.
    """
    if stage.world_preset:
        return list(stage.world_preset)
    if var.world_preset:
        return list(var.world_preset)
    return list(exp.world_preset)


# --- Snapshot --------------------------------------------------------------

def snapshot_experiment(exp: ExperimentDef) -> dict:
    """Resolved-and-canonical dict form, suitable for dumping as the
    experiment.yaml snapshot in the results dir."""
    out: dict = {
        "exp_id": exp.exp_id,
        "agent_preset": list(exp.agent_preset),
        "task_preset": list(exp.task_preset),
    }
    if exp.world_preset:
        out["world_preset"] = list(exp.world_preset)
    out["stages"] = [s.to_dict() for s in exp.stages]
    out["methods"] = [
        {"name": m.name, "n_seeds": m.n_seeds,
         **({"args": dict(m.args)} if m.args else {}),
         **({"python_env": m.python_env} if m.python_env else {}),
         **({"train_script": m.train_script} if m.train_script else {})}
        for m in exp.methods
    ]
    out["variations"] = [
        {"name": v.name,
         **({"agent_preset": list(v.agent_preset)} if v.agent_preset is not None else {}),
         **({"task_preset": list(v.task_preset)} if v.task_preset is not None else {}),
         **({"world_preset": list(v.world_preset)} if v.world_preset is not None else {}),
         **({"method_args": dict(v.method_args)} if v.method_args else {})}
        for v in exp.variations
    ]
    out["mode"] = exp.mode
    if exp.step_multiplier != 1.0:
        out["step_multiplier"] = exp.step_multiplier
    if exp.common_args:
        out["common_args"] = dict(exp.common_args)
    return out
