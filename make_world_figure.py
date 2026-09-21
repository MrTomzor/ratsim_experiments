"""Paper figures of the eval worlds: overhead snapshots in a grid.

    # the 8 paper worlds (paper/results_table.yaml rows), first N eval worlds each
    ~/ratvenv/venv/bin/python make_world_figure.py worlds --seeds-per-world 1
    ~/ratvenv/venv/bin/python make_world_figure.py worlds --seeds-per-world 2
    ~/ratvenv/venv/bin/python make_world_figure.py worlds --seeds-per-world 4     # 8 rows x 4 cols

    # difficulty sweep of an adaptive def, d = linspace(0, 1, K), one tile each
    ~/ratvenv/venv/bin/python make_world_figure.py difficulty --def ortho_wells_adaptive_nohomeprime --n 6
    ~/ratvenv/venv/bin/python make_world_figure.py difficulty --def ortho_wells_adaptive --difficulties 0,0.3,0.6,1 --seeds-per-world 2

    # re-layout from cached tiles only (no Unity needed)
    ~/ratvenv/venv/bin/python make_world_figure.py worlds --seeds-per-world 4 --no-fetch

Tiles are the overhead renders `ratsim.world_snapshot` fetches from Unity
(`WorldSnapshot.cs`; needs the Editor / a gfx build in play mode on --port,
not -nographics). Each (world, seed) tile is fetched once and cached under
--cache (`results/analysis/paper/snapshots/`), so re-layouts are free; a tile
that record_trajectories.py / snapshot_worlds.py already rendered into
`<exp_dir>/snapshots/` is reused too. `--refresh` re-renders, `--no-fetch`
composes from the cache and leaves a "no snapshot" placeholder for anything
missing.

`worlds`: rows and labels come from paper/results_table.yaml (same order and
names as the results table). Seeds default to the eval worlds themselves: the
eval env draws world seeds from `np.random.default_rng(eval_metaseed)`, the
first draw going to the warm-up reset, so eval episode k is draw k+1 and
column k of the figure is the k-th eval world in every row (the same seed in
every row, which is what makes the columns comparable). The world config per
row is the one the recorded eval used (`world_config_file` in the manifest),
else the def's final-stage world (difficulty pinned via the row's
`difficulty:`). Layout: each world is a block of N tiles; blocks wrap at
--cols tiles per row (default 4), so N=1 -> 2x4, N=2 -> 4 rows of 2 worlds,
N=4 -> 8 rows of 1 world (portrait).

Seeds: `--seeds-per-world N` takes eval worlds 1..N, `--seed-start K` shifts that
window to K+1..K+N, `--seeds a,b,c` uses arbitrary seeds. `--shadows 0` renders
without light shadows (tiles cached and named with a `_noshadow` tag).

`difficulty`: one block per difficulty rung of an adaptive def (needs an
`adaptive_difficulty:` block), N seeds per rung. Tiles share ONE metre scale
by default so the world's growth with d is visible (`--scale fill` to fill
each tile instead; `worlds` defaults to fill, `--scale same` flips it);
titles print the interpolated world size / rooms / cue range. World seeds are the same eval seeds as above.

Output: <out>/worlds_<N>seed_<view>[_<name>].{png,pdf} or
<out>/difficulty_<def>_<K>x<N>_<view>[_<name>].{png,pdf}; --out defaults to
`figures_out:` in paper/results_table.yaml, else results/analysis/paper/. Tiles
are downsampled to --tile-px before embedding so the PDF stays small. Needs
matplotlib + pyyaml + Pillow (sb3 venv).
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from record_trajectories import RESULTS_ROOTS, load_manifest  # noqa: E402
from snapshot_worlds import parse_extra, world_config_for_exp  # noqa: E402

DEFAULT_CONFIG = HERE / "paper" / "results_table.yaml"
DEFAULT_CACHE = HERE / "results" / "analysis" / "paper" / "snapshots"
DEFAULT_OUT = HERE / "results" / "analysis" / "paper"
DEFS_DIR = HERE / "defs"
MISSING = "__missing__"

# adaptive-difficulty keys worth printing under a rung's title, in this order
DIFFICULTY_LABEL_KEYS = [
    ("world_bounds/width", "{:g} m"),
    ("maze/n_rooms", "{:g} rooms"),
    ("wells/cue/range", "cue {:g} m"),
]


# ─────────────────────────────────────────────
#  Seeds
# ─────────────────────────────────────────────

def eval_world_seeds(metaseed: int, n: int, skip: int = 1) -> list[int]:
    """The first `n` world seeds an eval with this metaseed runs on.

    Replays `np.random.default_rng(metaseed).integers(0, 2**31 - 1)` (env.py
    reset) and drops the first `skip` draws, which the warm-up reset consumes
    before eval episode 0 — with metaseed 42 that makes eval world 1 the
    1662057957 every trajectories.json manifest records.
    """
    rng = np.random.default_rng(int(metaseed))
    draws = [int(rng.integers(0, 2**31 - 1)) for _ in range(skip + n)]
    return draws[skip:]


# ─────────────────────────────────────────────
#  Config / rows
# ─────────────────────────────────────────────

def load_table_config(path: Path) -> dict:
    import yaml
    with open(path) as f:
        return yaml.safe_load(f) or {}


def table_rows(cfg: dict, only: list[str] | None) -> list[dict]:
    """[{label, exp, group, variation, difficulty, source}] in table order."""
    rows = []
    for g in cfg.get("groups", []):
        for r in g.get("rows", []):
            rows.append({"label": r["label"], "exp": r["exp"], "group": g.get("name", ""),
                         "variation": r.get("variation"), "difficulty": r.get("difficulty"),
                         "source": r.get("source", "rci")})
    if only:
        by = {r["label"]: r for r in rows} | {r["exp"]: r for r in rows}
        bad = [k for k in only if k not in by]
        if bad:
            sys.exit(f"ERROR: unknown rows {bad}; known: {[r['label'] for r in rows]}")
        rows = [by[k] for k in only]
    return rows


def exp_dir_or_def(exp: str, source: str) -> tuple[Path, dict]:
    """(results dir, manifest) for the row, or (def yaml, {}) when nothing was pulled."""
    d = RESULTS_ROOTS[source] / exp
    if d.is_dir():
        return d.resolve(), load_manifest(d)
    p = DEFS_DIR / f"{exp}.yaml"
    if p.exists():
        return p.resolve(), {}
    sys.exit(f"ERROR: {exp}: neither {d} nor {p} exists")


def def_path(name: str) -> Path:
    p = Path(name)
    if p.suffix in (".yaml", ".yml") and p.exists():
        return p.resolve()
    p = DEFS_DIR / f"{name}.yaml"
    if not p.exists():
        sys.exit(f"ERROR: no def {name} ({p})")
    return p


# ─────────────────────────────────────────────
#  Tiles (fetch / cache)
# ─────────────────────────────────────────────

def tile_exists(stem: Path) -> bool:
    return stem.with_suffix(".png").exists() and stem.with_suffix(".json").exists()


def tile_width(stem: Path) -> int | None:
    import json
    try:
        with open(stem.with_suffix(".json")) as f:
            return int(json.load(f).get("width", 0))
    except (OSError, ValueError):
        return None


def existing_manifest_tile(exp_dir: Path, seed: int, view: str, width: int) -> Path | None:
    """A tile snapshot_worlds.py already rendered for this eval world, if it matches."""
    stem = exp_dir / "snapshots" / f"seed{seed}_{view}"
    if exp_dir.is_dir() and tile_exists(stem) and tile_width(stem) == width:
        return stem
    return None


class TileFetcher:
    """Lazy Unity connection: nothing is opened until a tile is actually missing."""

    def __init__(self, port: int, agent_preset: str, width: int, extra: dict,
                 refresh: bool, no_fetch: bool):
        self.port, self.agent_preset, self.width = port, agent_preset, width
        self.extra, self.refresh, self.no_fetch = extra, refresh, no_fetch
        self.conn = None
        self.fetched = 0

    def _connect(self):
        if self.conn is None:
            from ratsim.unity_launcher import attach_instance
            from ratsim.world_snapshot import connect_and_select_scene
            try:
                attach_instance(self.port)       # waits for Play, never spawns
            except RuntimeError as e:
                sys.exit(f"ERROR: {e}\n(no local build has WorldSnapshot.cs yet — press Play in "
                         f"the Editor; --no-fetch composes from the cache alone)")
            self.conn = connect_and_select_scene(agent_preset=self.agent_preset, port=self.port)
            self._scene_key = None
        return self.conn

    def fresh_scene(self, key) -> None:
        """Reload the scene before a world whose config differs from the previous one.

        Unity's WorldLoadingController MERGES each world_config into its param table
        and never clears it, so keys the new config does not mention (a maze's walls,
        the previous world's wells...) leak into the next world. A scene_select reload
        recreates the controller with an empty table; the agent config is resent since
        the scene's AgentLoader is new too.
        """
        from ratsim.config_blender import blend_presets, to_entries_json
        from ratsim.roslike_unity_connector.message_definitions import StringMessage
        conn = self._connect()
        if key == self._scene_key:
            return
        conn.publish(StringMessage(data="Wildfire"), "/sim_control/scene_select")
        conn.send_messages_and_step(enable_physics_step=False)
        conn.read_messages_from_unity()
        agent_config = blend_presets("agents", [self.agent_preset])
        conn.publish(StringMessage(data=to_entries_json(agent_config)), "/sim_control/agent_config")
        conn.send_messages_and_step(enable_physics_step=False)
        conn.read_messages_from_unity()
        self._scene_key = key

    def ensure(self, stem: Path, cfg_fn, seed: int, view: str, reuse: Path | None = None,
               scene_key=None) -> Path | str:
        """Path stem of the tile, fetching it if needed; MISSING with --no-fetch.
        `scene_key` identifies the world config: a change triggers a scene reload."""
        if not self.refresh:
            if tile_exists(stem) and tile_width(stem) == self.width:
                return stem
            if reuse is not None:
                print(f"  reuse {reuse}.png")
                return reuse
        if self.no_fetch:
            print(f"  [missing] {stem}.png (--no-fetch)")
            return MISSING
        from ratsim.world_snapshot import fetch_world_snapshot, save_snapshot
        cfg = cfg_fn()
        self.fresh_scene(scene_key)
        print(f"  snapshot seed={seed} view={view} -> {stem}.png")
        snap = fetch_world_snapshot(self._connect(), cfg, seed=int(seed), view=view,
                                    width=self.width, height=0, extra=self.extra)
        save_snapshot(stem, snap)
        m = snap["meta"]
        print(f"    {m['width']}x{m['height']}  world {m['world_width']:g}x{m['world_height']:g}")
        self.fetched += 1
        return stem


# ─────────────────────────────────────────────
#  Blocks
# ─────────────────────────────────────────────

def world_blocks(args, fetcher: TileFetcher, cache: Path) -> list[dict]:
    cfg = load_table_config(Path(args.config))
    rows = table_rows(cfg, args.rows.split(",") if args.rows else None)
    metaseed = int(args.eval_metaseed if args.eval_metaseed is not None
                   else cfg.get("eval_metaseed", 42))
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else eval_world_seeds(metaseed, args.seeds_per_world, skip=1 + args.seed_start))
    print(f"[world_figure] {len(rows)} worlds x seeds {seeds} (eval_metaseed {metaseed})")
    blocks = []
    for r in rows:
        exp_dir, man = exp_dir_or_def(r["exp"], r["source"])
        if man.get("world_seeds") and args.seeds is None and man["world_seeds"][0] != seeds[0]:
            print(f"  [warn] {r['exp']}: manifest eval world {man['world_seeds'][0]} != "
                  f"figure column 1 ({seeds[0]}) — different eval metaseed?")
        print(f"\n=== {r['label']}  ({r['exp']})")
        cfg_cache: dict = {}

        def cfg_fn(exp_dir=exp_dir, man=man, r=r, cfg_cache=cfg_cache):
            if "cfg" not in cfg_cache:
                wc, origin = world_config_for_exp(exp_dir, man, r["variation"], r["difficulty"])
                print(f"  world config: {origin}")
                cfg_cache["cfg"] = wc
            return cfg_cache["cfg"]

        tiles = []
        for k, seed in enumerate(seeds):
            stem = cache / f"{r['exp']}_seed{seed}_{args.view_tag}"
            reuse = (existing_manifest_tile(exp_dir, seed, args.view, fetcher.width)
                     if args.shadows else None)
            got = fetcher.ensure(stem, cfg_fn, seed, args.view, reuse=reuse, scene_key=r["exp"])
            tiles.append({"stem": got, "label": f"world {k + 1}" if len(seeds) > 1 else ""})
        blocks.append({"title": r["label"], "group": r["group"], "tiles": tiles})
    return blocks


def difficulty_blocks(args, fetcher: TileFetcher, cache: Path) -> list[dict]:
    from experiment_defs import load_experiment_def, resolve_difficulty_overrides
    dp = def_path(args.def_name)
    exp = load_experiment_def(dp)
    if exp.adaptive_difficulty is None:
        sys.exit(f"ERROR: {dp} has no adaptive_difficulty: block")
    if args.difficulties:
        ds = [float(x) for x in args.difficulties.split(",")]
    else:
        ds = [round(x, 4) for x in np.linspace(0.0, 1.0, int(args.n))]
    cfg = load_table_config(Path(args.config)) if Path(args.config).exists() else {}
    metaseed = int(args.eval_metaseed if args.eval_metaseed is not None
                   else cfg.get("eval_metaseed", 42))
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else eval_world_seeds(metaseed, args.seeds_per_world, skip=1 + args.seed_start))
    print(f"[world_figure] {dp.stem}: difficulties {ds} x seeds {seeds}")
    blocks = []
    for d in ds:
        over = resolve_difficulty_overrides(exp, d)
        parts = [fmt.format(over[k]) for k, fmt in DIFFICULTY_LABEL_KEYS if k in over]
        print(f"\n=== d={d:g}  {' · '.join(parts)}")

        def cfg_fn(d=d):
            wc, origin = world_config_for_exp(dp, {}, args.variation, d)
            print(f"  world config: {origin}")
            return wc

        tiles = []
        for k, seed in enumerate(seeds):
            # no '.' in the stem: Path.with_suffix() would take ".40_seed…" for an extension
            stem = cache / f"{dp.stem}_d{int(round(d * 100)):03d}_seed{seed}_{args.view_tag}"
            got = fetcher.ensure(stem, cfg_fn, seed, args.view, scene_key=("d", d))
            tiles.append({"stem": got, "label": f"world {k + 1}" if len(seeds) > 1 else ""})
        blocks.append({"title": f"$d$ = {d:g}", "group": "", "subtitle": " · ".join(parts),
                       "tiles": tiles})
    return blocks


# ─────────────────────────────────────────────
#  Composition
# ─────────────────────────────────────────────

def load_tile(stem: Path, tile_px: int) -> tuple[np.ndarray, dict, float | None]:
    """(image downsampled to <= tile_px, meta, pixels-per-metre or None for persp)."""
    from PIL import Image
    from ratsim.world_snapshot import load_snapshot
    image, meta = load_snapshot(stem)
    if max(image.shape[:2]) > tile_px:
        im = Image.fromarray(image)
        im.thumbnail((tile_px, tile_px), Image.LANCZOS)
        image = np.array(im)
    ppm = None
    if meta.get("projection", meta.get("view")) == "ortho" and meta.get("ortho_size"):
        # ortho_size is the camera half-height in metres; the full-res height covers 2x it
        ppm = float(meta["height"]) / (2.0 * float(meta["ortho_size"]))
        ppm *= image.shape[0] / float(meta["height"])     # after downsampling
    return image, meta, ppm


def draw_scale_bar(ax, metres: float, ppm: float | None, extent: tuple, fontsize: float) -> None:
    if not metres or ppm is None:
        return
    x0, x1, y0, y1 = extent
    span = x1 - x0
    pad = 0.04 * span
    bx, by = x0 + pad, y0 + pad
    ax.plot([bx, bx + metres], [by, by], color="white", lw=4.5, solid_capstyle="butt", zorder=5)
    ax.plot([bx, bx + metres], [by, by], color="black", lw=2.0, solid_capstyle="butt", zorder=6)
    ax.text(bx + metres / 2, by + 0.012 * span, f"{metres:g} m", ha="center", va="bottom",
            fontsize=fontsize, color="black", zorder=6,
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=1.0))


def compose(blocks: list[dict], cols: int, args, title_of: str) -> "plt.Figure":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n_per = max(len(b["tiles"]) for b in blocks)
    if cols % n_per:
        sys.exit(f"ERROR: --cols {cols} must be a multiple of the tiles per block ({n_per})")
    per_row = cols // n_per
    n_rows = math.ceil(len(blocks) / per_row)

    # load everything first: the shared metre scale needs every tile's extent
    loaded: dict = {}
    half = 0.0
    for b in blocks:
        for t in b["tiles"]:
            if t["stem"] is MISSING or t["stem"] in loaded:
                continue
            img, meta, ppm = load_tile(t["stem"], args.tile_px)
            if ppm:
                w_m, h_m = img.shape[1] / ppm, img.shape[0] / ppm
                ext = (-w_m / 2, w_m / 2, -h_m / 2, h_m / 2)
                half = max(half, w_m / 2, h_m / 2)
            else:
                ext = (0, img.shape[1], 0, img.shape[0])
            loaded[t["stem"]] = (img, meta, ppm, ext)
    same_scale = args.scale == "same" and half > 0

    tile_in = args.tile_in
    has_second = any(b.get("subtitle") or (b.get("group") and per_row > 1) for b in blocks)
    has_tile_labels = any(t.get("label") for b in blocks for t in b["tiles"])
    top_pad = (0.44 if has_second else 0.30) + (0.12 if has_tile_labels else 0.0)  # in/row, titles
    left_pad = 0.45 if per_row == 1 and any(b.get("group") for b in blocks) else 0.0  # group labels
    fig_w = cols * tile_in + (per_row - 1) * args.block_gap * tile_in + left_pad + 0.2
    fig_h = n_rows * (tile_in + top_pad) + 0.3
    fig = plt.figure(figsize=(fig_w, fig_h))
    outer = GridSpec(n_rows, per_row, figure=fig,
                     left=(left_pad + 0.1) / fig_w, right=1 - 0.1 / fig_w,
                     top=1 - (top_pad + 0.1) / fig_h, bottom=0.1 / fig_h,
                     # wspace is a fraction of an outer cell (= n_per tiles wide)
                     wspace=args.block_gap / n_per, hspace=top_pad / tile_in)
    fs_title, fs_small = args.fontsize, args.fontsize * 0.78

    block_axes: list[list] = []
    for i, b in enumerate(blocks):
        r, c = divmod(i, per_row)
        inner = outer[r, c].subgridspec(1, n_per, wspace=0.04)
        axes = []
        for j in range(n_per):
            ax = fig.add_subplot(inner[0, j])
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_linewidth(0.4); s.set_color("0.6")
            t = b["tiles"][j] if j < len(b["tiles"]) else {"stem": MISSING, "label": ""}
            if t["stem"] is MISSING:
                ax.set_facecolor("0.94")
                ax.text(0.5, 0.5, "no snapshot", ha="center", va="center", fontsize=fs_small,
                        color="0.4", transform=ax.transAxes)
                ax.set_aspect("equal")
            else:
                img, meta, ppm, ext = loaded[t["stem"]]
                ax.imshow(img, extent=ext, interpolation="lanczos", origin="upper")
                if same_scale and ppm:
                    ax.set_xlim(-half, half); ax.set_ylim(-half, half)
                    ax.set_facecolor(args.bg)
                else:
                    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
                ax.set_aspect("equal")
                lim = (ax.get_xlim() + ax.get_ylim())
                draw_scale_bar(ax, args.scale_bar, ppm, lim, fs_small)
            if t.get("label"):
                ax.text(0.5, -0.02, t["label"], ha="center", va="top", fontsize=fs_small,
                        color="0.35", transform=ax.transAxes)
            axes.append(ax)
        block_axes.append(axes)

    # block titles (span the block) and, for one block per row, rotated group labels
    fig.canvas.draw()
    for b, axes in zip(blocks, block_axes):
        bb = matplotlib.transforms.Bbox.union([ax.get_position() for ax in axes])
        # a grey second line (group name, or the rung's world stats) sits under the title:
        # both anchor at the block top, the title carrying a blank line to make room
        second = b.get("subtitle") or (b.get("group") if per_row > 1 else "")
        title = f"{b['title']}\n" if second else b["title"]
        x = (bb.x0 + bb.x1) / 2
        fig.text(x, bb.y1 + 0.003, title, ha="center", va="bottom",
                 fontsize=fs_title, fontweight="bold", linespacing=1.15)
        if second:
            fig.text(x, bb.y1 + 0.003, second, ha="center", va="bottom",
                     fontsize=fs_small, color="0.4")
    if per_row == 1:
        # one world per row: label each contiguous run of a group once, rotated at the left
        i = 0
        while i < len(blocks):
            g = blocks[i].get("group", "")
            j = i
            while j + 1 < len(blocks) and blocks[j + 1].get("group", "") == g:
                j += 1
            if g:
                top = block_axes[i][0].get_position().y1
                bot = block_axes[j][0].get_position().y0
                x0 = block_axes[i][0].get_position().x0
                fig.text(x0 - 0.62 * left_pad / fig_w, (top + bot) / 2, g, rotation=90,
                         ha="center", va="center", fontsize=fs_title, color="0.25")
                fig.add_artist(matplotlib.lines.Line2D(
                    [x0 - 0.18 * left_pad / fig_w] * 2, [bot, top], color="0.75", lw=1.0))
            i = j + 1
    if title_of:
        fig.suptitle(title_of, fontsize=fs_title + 1)
    return fig


# ─────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────

def add_common(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--seeds-per-world", type=int, default=1, dest="seeds_per_world",
                    help="tiles per world/rung = first N eval worlds (default 1)")
    ap.add_argument("--seeds", default=None, help="explicit world seeds instead (comma-separated)")
    ap.add_argument("--seed-start", type=int, default=0, dest="seed_start",
                    help="skip the first K eval worlds (e.g. 4 with N=4 shows eval worlds 5-8)")
    ap.add_argument("--eval-metaseed", type=int, default=None, dest="eval_metaseed",
                    help="metaseed the eval worlds are drawn from (default: yaml eval_metaseed, 42)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="paper/results_table.yaml")
    ap.add_argument("--view", default="ortho", choices=["ortho", "persp"])
    ap.add_argument("--width", type=int, default=2048, help="render width in px (cache key)")
    ap.add_argument("--shadows", type=int, default=1, choices=[0, 1],
                    help="0 renders without light shadows (flat map look); cached separately")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VAL",
                    help="extra world_snapshot/<KEY> override (margin=0.05, background=skybox, ...)")
    ap.add_argument("--port", type=int, default=9000, help="Unity port (Editor in Play mode)")
    ap.add_argument("--agent-preset", default="sphereagent_2d_lidar", dest="agent_preset")
    ap.add_argument("--cache", default=str(DEFAULT_CACHE), help="tile cache dir")
    ap.add_argument("--refresh", action="store_true", help="re-render tiles even if cached")
    ap.add_argument("--no-fetch", action="store_true", dest="no_fetch",
                    help="never touch Unity; placeholder for missing tiles")
    ap.add_argument("--cols", type=int, default=None, help="tiles per figure row (multiple of N)")
    ap.add_argument("--scale", choices=["fill", "same"], default=None,
                    help="fill: each world fills its tile (scale bar shows size; default for "
                         "worlds); same: one metre scale for every tile so sizes compare "
                         "(default for difficulty)")
    ap.add_argument("--scale-bar", type=float, default=25.0, dest="scale_bar",
                    help="scale bar length in metres (0 = none)")
    ap.add_argument("--tile-in", type=float, default=1.9, dest="tile_in", help="tile size in inches")
    ap.add_argument("--tile-px", type=int, default=900, dest="tile_px",
                    help="downsample tiles to this many px before embedding")
    ap.add_argument("--block-gap", type=float, default=0.18, dest="block_gap",
                    help="gap between world blocks, in tile widths")
    ap.add_argument("--fontsize", type=float, default=9.0)
    ap.add_argument("--bg", default="white", help="axes background behind same-scale tiles")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--title", default="", help="figure suptitle (default none)")
    ap.add_argument("--name", default="", help="suffix for the output file name")
    ap.add_argument("--out", default=None, help="output dir (default: yaml figures_out, else "
                                               "results/analysis/paper)")
    ap.add_argument("--show", action="store_true", help="open the PNG when done")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    w = sub.add_parser("worlds", help="the paper's eval worlds, N seeds each")
    w.add_argument("--rows", default=None, help="subset of table rows (labels or exp ids)")
    add_common(w)
    d = sub.add_parser("difficulty", help="difficulty rungs of an adaptive def")
    d.add_argument("--def", dest="def_name", required=True, help="def name or yaml path")
    d.add_argument("--n", type=int, default=6, help="rungs: d = linspace(0, 1, n)")
    d.add_argument("--difficulties", default=None, help="explicit d values (comma-separated)")
    d.add_argument("--variation", default=None)
    add_common(d)
    args = ap.parse_args()

    if args.seeds_per_world < 1:
        ap.error("--seeds-per-world must be >= 1")
    extra = parse_extra(args.set, ap)
    if not args.shadows:
        extra["shadows"] = 0
    args.view_tag = args.view + ("" if args.shadows else "_noshadow")   # tile cache / file name tag
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    fetcher = TileFetcher(args.port, args.agent_preset, args.width, extra, args.refresh, args.no_fetch)

    if args.scale is None:
        args.scale = "fill" if args.cmd == "worlds" else "same"
    if args.cmd == "worlds":
        blocks = world_blocks(args, fetcher, cache)
        n = len(blocks[0]["tiles"])
        cols = args.cols or (4 if 4 % n == 0 else n)
        stem_name = f"worlds_{n}seed_{args.view_tag}"
    else:
        blocks = difficulty_blocks(args, fetcher, cache)
        n = len(blocks[0]["tiles"])
        k = len(blocks)
        cols = args.cols or (k * n if k * n <= 8 else (8 // n) * n)
        stem_name = f"difficulty_{Path(def_path(args.def_name)).stem}_{k}x{n}_{args.view_tag}"
    if fetcher.fetched:
        print(f"\n[world_figure] rendered {fetcher.fetched} new tile(s) into {cache}")

    fig = compose(blocks, cols, args, args.title)

    cfg = load_table_config(Path(args.config)) if Path(args.config).exists() else {}
    out = Path(args.out or cfg.get("figures_out") or DEFAULT_OUT).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    if args.name:
        stem_name += f"_{args.name}"
    png = out / f"{stem_name}.png"
    pdf = out / f"{stem_name}.pdf"
    fig.savefig(png, dpi=args.dpi)
    fig.savefig(pdf)
    missing = sum(t["stem"] is MISSING for b in blocks for t in b["tiles"])
    print(f"\n[world_figure] -> {png}\n[world_figure] -> {pdf}"
          + (f"\n[world_figure] {missing} tile(s) missing — start Unity (Play) and rerun "
             f"without --no-fetch" if missing else ""))
    if args.show:
        import subprocess
        subprocess.Popen(["xdg-open", str(png)])


if __name__ == "__main__":
    main()
