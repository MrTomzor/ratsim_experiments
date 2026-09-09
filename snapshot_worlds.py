"""Overhead pictures of the eval worlds of one or more experiments.

    python snapshot_worlds.py memory_3malls memory_orthomaze                 # ortho
    python snapshot_worlds.py memory_3malls --views ortho,persp
    python snapshot_worlds.py memory_3malls --seeds 1662057957,7 --width 3000
    python snapshot_worlds.py defs/memory_3malls.yaml --seeds 42 --out figs/  # no results dir needed

For each experiment the world config is the one its recorded eval used
(`world_config_file` in `<exp_dir>/trajectories.json`, written by
record_trajectories.py) or, failing that, the def's final-stage world
(`--variation`, `--difficulty`). Seeds default to the manifest's world seeds.
Pictures land in `<exp_dir>/snapshots/seed<seed>_<view>.png` + `.json`
(camera matrix etc.) and are indexed in the manifest under `snapshots`, which
is what `plot_trajectories.py --background <view>` reads. With `--out` they go
to that directory instead (paper "environment" figures, no manifest).

record_trajectories.py --snapshot <views> calls this after recording.
Needs Unity rendering in play mode on --port (default 9000; not -nographics).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from experiment_defs import (  # noqa: E402
    find_variation,
    load_experiment_def,
    resolve_difficulty_overrides,
    resolve_stage_world,
)
from record_trajectories import (  # noqa: E402
    MANIFEST_NAME,
    load_manifest,
    resolve_exp_dir,
    save_manifest,
)

SNAPSHOT_DIRNAME = "snapshots"


def world_config_for_exp(exp_dir: Path, man: dict, variation: str | None,
                         difficulty: float | None) -> tuple[dict, str]:
    """(world config, where it came from) for the experiment's eval world."""
    from ratsim.config_blender import blend_presets
    from ratsim.world_snapshot import load_world_config_file

    wc = man.get("world_config_file")
    if wc and Path(wc).exists() and variation is None and difficulty is None:
        return load_world_config_file(wc), f"world_config_file {wc}"
    def_path = exp_dir / "experiment.yaml"
    if not def_path.exists():
        def_path = exp_dir if exp_dir.suffix in (".yaml", ".yml") else HERE / "defs" / f"{exp_dir.name}.yaml"
    if not def_path.exists():
        raise FileNotFoundError(f"{exp_dir}: no world_config_file in the manifest and no "
                                f"experiment.yaml / defs/{exp_dir.name}.yaml")
    exp = load_experiment_def(def_path)
    var = find_variation(exp, variation or exp.variations[0].name)
    presets = resolve_stage_world(exp.stages[-1], var, exp)
    cfg = blend_presets("world", presets if isinstance(presets, list) else [presets])
    if difficulty is not None:
        cfg.update(resolve_difficulty_overrides(exp, difficulty))
    return cfg, f"def {def_path} (variation {var.name}, final stage, difficulty {difficulty})"


def snapshot_exp(exp_dir: Path, man: dict, seeds: list[int], views: list[str], conn,
                 width: int = 2048, height: int = 0, extra: dict | None = None,
                 out_dir: Path | None = None, variation: str | None = None,
                 difficulty: float | None = None, dry: bool = False) -> dict:
    """Take `views` of each seed; returns {seed: {view: png_path}} and, unless
    `out_dir` is given, records them in `man["snapshots"]`."""
    from ratsim.world_snapshot import fetch_world_snapshot, save_snapshot

    cfg, origin = world_config_for_exp(exp_dir, man, variation, difficulty)
    print(f"  world config: {origin}")
    dest = out_dir or (exp_dir / SNAPSHOT_DIRNAME)
    taken: dict = {}
    for seed in seeds:
        for view in views:
            stem = dest / (f"{exp_dir.name}_seed{seed}_{view}" if out_dir else f"seed{seed}_{view}")
            print(f"  snapshot seed={seed} view={view} → {stem}.png")
            if dry:
                taken.setdefault(int(seed), {})[view] = str(stem.with_suffix(".png"))
                continue
            snap = fetch_world_snapshot(conn, cfg, seed=int(seed), view=view,
                                        width=width, height=height, extra=extra)
            png = save_snapshot(stem, snap)
            m = snap["meta"]
            print(f"    {m['width']}x{m['height']}  world {m['world_width']:g}x{m['world_height']:g}")
            taken.setdefault(int(seed), {})[view] = str(png)
    if out_dir is None:
        snaps = man.setdefault("snapshots", {})
        for seed, per_view in taken.items():
            snaps.setdefault(str(seed), {}).update(per_view)
    return taken


def parse_views(s: str, ap: argparse.ArgumentParser) -> list[str]:
    from ratsim.world_snapshot import VIEWS, canonical_view
    views = [canonical_view(v.strip()) for v in s.split(",") if v.strip()]
    bad = [v for v in views if v not in VIEWS]
    if bad:
        ap.error(f"unknown views {bad}; choose from {VIEWS}")
    return views


def parse_extra(items: list[str], ap: argparse.ArgumentParser) -> dict:
    extra = {}
    for item in items:
        k, _, v = item.partition("=")
        if not _:
            ap.error(f"--set expects KEY=VAL, got {item!r}")
        extra[k] = v
    return extra


def add_snapshot_args(ap: argparse.ArgumentParser) -> None:
    """Shared with record_trajectories.py."""
    ap.add_argument("--width", type=int, default=2048, help="snapshot width in px")
    ap.add_argument("--height", type=int, default=0, help="snapshot height (0 = fit)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="extra world_snapshot/<KEY> override (margin=0.05, show_agent=1, "
                         "background=skybox, fov=30, ...)")
    ap.add_argument("--port", type=int, default=9000, help="Unity port for snapshots")
    ap.add_argument("--agent-preset", default="sphereagent_2d_lidar", dest="agent_preset")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exps", nargs="+", help="experiment ids, result dirs, or def yaml paths")
    ap.add_argument("--views", default="ortho", help="comma-separated: ortho,persp (both straight down)")
    ap.add_argument("--seeds", default=None, help="world seeds (default: manifest world_seeds)")
    ap.add_argument("--variation", default=None, help="def variation (when not using the manifest's config)")
    ap.add_argument("--difficulty", type=float, default=None)
    ap.add_argument("--out", default=None, help="write here instead of <exp_dir>/snapshots (no manifest)")
    ap.add_argument("--local", action="store_true", help="look ids up under results/experiments")
    ap.add_argument("--dry-run", action="store_true", dest="dry_run")
    add_snapshot_args(ap)
    args = ap.parse_args()

    views = parse_views(args.views, ap)
    extra = parse_extra(args.set, ap)
    seeds_arg = [int(s) for s in args.seeds.split(",")] if args.seeds else None

    conn = None
    if not args.dry_run:
        from ratsim.world_snapshot import connect_and_select_scene
        conn = connect_and_select_scene(agent_preset=args.agent_preset, port=args.port)

    for exp in args.exps:
        p = Path(exp)
        if p.suffix in (".yaml", ".yml") and p.exists():
            exp_dir, man = p.resolve(), {"world_seeds": []}
            if args.out is None:
                ap.error("a def yaml needs --out (there is no results dir to write into)")
        else:
            exp_dir = resolve_exp_dir(exp, "local" if args.local else "rci")
            man = load_manifest(exp_dir)
        seeds = seeds_arg or man.get("world_seeds") or []
        print(f"\n=== {exp_dir.name}  seeds={seeds}")
        if not seeds:
            print(f"  [skip] no seeds: pass --seeds or record trajectories first ({MANIFEST_NAME})")
            continue
        snapshot_exp(exp_dir, man, seeds, views, conn, width=args.width, height=args.height,
                     extra=extra, out_dir=Path(args.out) if args.out else None,
                     variation=args.variation, difficulty=args.difficulty, dry=args.dry_run)
        if args.out is None and not args.dry_run:
            save_manifest(exp_dir, man)
            print(f"  manifest: {exp_dir / MANIFEST_NAME}")


if __name__ == "__main__":
    main()
