"""Targets and scores of the ground-truth probes (glayout, gboxes) for train_probe.

The probe predicts a WINDOW x WINDOW crop of the true map (2 m cells), centred
on the agent's cell, axes of the world (no rotation) -- not the whole 100 x 100
map. A full-map probe could not learn at this data scale: from 283 training
mazes, even the current-observation probe stayed at the per-cell prior on the
cells in view (balanced accuracy 0.53 vs 0.53), because each of 10,000 output
units must learn on its own when the lidar is hitting it. Centred on the agent,
an output unit always means the same offset, and the obs probe decodes the
view (0.71 vs 0.51). The paper's Walls grid is 9 x 9 or 15 x 15 tiles from 30M
steps; its Objects target is agent-centric. --window 0 keeps the whole map.
Cells of the crop outside the world are wall / no box and never scored.

The targets are 100 x 100 maps per sample, 202k samples: 8 GB as a dense float
array, and their age maps another 4 GB. Neither is ever built:

  * a target is one of a few distinct maps -- one glayout per episode, one
    box map per number of boxes collected so far -- so IndexedTargets holds
    the maps once plus, per sample, the map's index and the crop's corner;
  * the masks (in view / memory / never seen, age) are needed only to score,
    so they are read one test episode at a time, scored, and dropped. The
    window's predictions are pasted back into the map and only the cells
    inside the window are scored.

Scoring groups at a sample (global_targets): in view = gage == 0; memory =
known & not in view; never seen = not known & not in view.

glayout (headline bal_memory): wall = wall fraction >= 0.5, predicted wall =
p >= 0.5; balanced accuracy = mean of the wall and free recalls, pooled over
the cells of all test samples in the group (chance 0.5 at any class ratio).

gboxes (headline ap_memory): average precision of p over memory cells, where
a positive is a cell holding a box that is still there AND has been observed
(a lidar ray returned off it). Cells whose box was never observed are left out
of the memory group: the agent cannot know about it. AP is exact for float16
scores (histograms over all 65536 bit patterns, ties as one threshold, as in
sklearn's average_precision_score).
  box_hit_rate       per remembered box (observed, still there, out of view):
                     some cell of its 3 x 3 neighbourhood is among the k
                     highest-scored cells of the map, k = cells holding a box
                     now; box_hit_chance is the same for a random ranking
  collected_fa_rate  per collected box's cell (no box left in it): it is still
                     among the k highest-scored cells -- remembering a box the
                     agent already took. Only policy episodes collect boxes.

Everything is also reported per source (explorer / each policy cell), by age
(steps since the cell was last in view, targets.AGE_BINS), and per episode
(for paired bootstraps between models).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from probing.core import dataset
from probing.s2_targets import targets
from probing.s2_targets.targets import AGE_BINS

GLOBAL_TARGETS = ("glayout", "gboxes")
N_F16 = 1 << 15  # non-negative float16 bit patterns, ascending in value


def source_group(ep: dict) -> str:
    """explorer, or the policy's cell (L0_baseline, L1_tbtt, ...)."""
    run = str(ep["policy_run"])
    return run.split("__")[0] if run else str(ep["source"])


# -- targets ------------------------------------------------------------------

def load_global_targets(gt_dir: Path, root: Path, paths: list[Path], fp: str,
                        episodes: list[dict], window: int) -> dict:
    """The small part of the global targets of these episodes (no age maps),
    plus each sample's agent cell (row, col) on the coarse grid."""
    info_path = gt_dir / "gtargets.json"
    if not info_path.exists():
        raise ValueError(f"{gt_dir}: incomplete (no gtargets.json)")
    info = dataset.load_json(info_path)
    if info.get("dataset_fingerprint") != fp:
        raise ValueError(f"{gt_dir}: computed from dataset {str(info.get('dataset_fingerprint'))[:12]}, "
                         f"dataset is now {fp[:12]}; recompute the global targets")
    lay = dataset.load_json(root / "layouts" / "layouts.json")
    if lay.get("fingerprint") != info.get("layouts_fingerprint"):
        raise ValueError(f"{gt_dir}: computed from other layouts than {root / 'layouts'} "
                         f"holds now; recompute the global targets")
    tg = {"t": [], "episode": [], "glayout": [], "gknown_first": [], "box_cell": [],
          "box_collected_t": [], "box_first_hit_t": []}
    for p in paths:
        with np.load(gt_dir / p.name, allow_pickle=False) as f:
            for k in tg:
                if k != "episode":
                    tg[k].append(f[k])
        tg["episode"].append(np.full(len(tg["t"][-1]), dataset.episode_id(p), np.int32))
    out = {k: tg[k] for k in ("box_cell", "box_collected_t", "box_first_hit_t")}
    out.update(t=np.concatenate(tg["t"]), episode=np.concatenate(tg["episode"]),
               glayout=np.stack(tg["glayout"]), gknown_first=np.stack(tg["gknown_first"]),
               dir=gt_dir, names=[p.name for p in paths], info=info,
               window=int(window) or None)
    from probing.s2_targets.global_targets import coarse_rc
    pose = np.concatenate([episodes[e]["pose"][out["t"][out["episode"] == e]]
                           for e in range(len(paths))])
    out["agent_rc"] = coarse_rc(pose[:, :2], dataset.load_json(root / "meta.json"))
    return out


class IndexedTargets:
    """Per-sample targets: distinct maps (M, H, W), and per sample the map's
    index and the corner of its size x size crop. The maps are padded by
    size // 2 cells of `fill`, so the corner of a crop centred on agent cell
    (r, c) is (r, c). Without a window the crop is the whole map."""

    def __init__(self, maps: np.ndarray, index: np.ndarray, agent_rc: np.ndarray,
                 window: int | None, fill: float):
        pad = window // 2 if window else 0
        self.crop = (window, window) if window else maps.shape[1:]
        self.maps = np.pad(maps, ((0, 0), (pad, pad), (pad, pad)),
                           constant_values=fill).astype(np.float16)
        self.index = index.astype(np.int64)
        self.corner = agent_rc if window else np.zeros_like(agent_rc)

    @property
    def n_out(self) -> int:
        return self.crop[0] * self.crop[1]

    @staticmethod
    def _crop(maps, index, corner, ar, ac, rows):
        i, r, c = index[rows], corner[rows, 0], corner[rows, 1]
        return maps[i[:, None, None], r[:, None, None] + ar[None, :, None],
                    c[:, None, None] + ac[None, None, :]].reshape(len(i), -1)

    def __getitem__(self, rows) -> np.ndarray:
        return self._crop(self.maps, self.index, self.corner, np.arange(self.crop[0]),
                          np.arange(self.crop[1]), rows).astype(np.float32)

    def mean(self, rows, chunk: int = 4096) -> np.ndarray:
        total = np.zeros(self.n_out)
        for i in range(0, len(rows), chunk):
            total += self[rows[i:i + chunk]].sum(0)
        return total / len(rows)

    def to(self, device):
        import torch
        maps, index, corner, ar, ac = (torch.from_numpy(a).to(device) for a in
                                       (self.maps, self.index, self.corner,
                                        np.arange(self.crop[0]), np.arange(self.crop[1])))
        crop = self._crop

        class _OnDevice:
            def __getitem__(self, b):
                return crop(maps, index, corner, ar, ac, b).float()
        return _OnDevice()


def alive_at(collected_t: np.ndarray, t) -> np.ndarray:
    """(n, B): box still there at each step of t."""
    return collected_t[None, :] > np.asarray(t)[:, None]


def build_global_targets(kind: str, tg: dict) -> IndexedTargets:
    shape = tg["glayout"].shape[1:]
    n_cells = tg["glayout"][0].size
    if kind == "glayout":  # outside the world: wall
        return IndexedTargets(tg["glayout"], tg["episode"], tg["agent_rc"],
                              tg["window"], fill=1.0)
    maps, index = [], np.zeros(len(tg["t"]), np.int64)
    for e, (cell, coll) in enumerate(zip(tg["box_cell"], tg["box_collected_t"])):
        rows = np.flatnonzero(tg["episode"] == e)
        times = np.unique(coll[coll < np.iinfo(np.int32).max])
        # A sample's map = boxes not yet collected; it changes only at pickups.
        state = np.searchsorted(times, tg["t"][rows], side="right")
        before = np.concatenate([[-1], times])
        index[rows] = len(maps) + state
        for s in range(len(times) + 1):
            m = np.zeros(n_cells, np.float16)
            m[cell[coll > before[s]]] = 1
            maps.append(m.reshape(shape))
    return IndexedTargets(np.stack(maps), index, tg["agent_rc"], tg["window"], fill=0.0)


def uncrop(p: np.ndarray, rc: np.ndarray, window: int | None, shape) -> tuple:
    """Window predictions (n, window^2) -> (n, H*W) on the map, NaN outside the
    window, and the (n, H*W) mask of map cells inside it."""
    if not window:
        return p, np.ones(p.shape, bool)
    n, (h, w), half = len(p), shape, window // 2
    full = np.full((n, h, w), np.nan, np.float32)
    win = p.reshape(n, window, window)
    for i, (r, c) in enumerate(rc):
        r0, c0 = r - half, c - half
        rs, cs = max(r0, 0), max(c0, 0)
        re, ce = min(r0 + window, h), min(c0 + window, w)
        full[i, rs:re, cs:ce] = win[i, rs - r0:re - r0, cs - c0:ce - c0]
    full = full.reshape(n, -1)
    return full, ~np.isnan(full)


def episode_masks(tg: dict, e: int) -> dict:
    """(n, C) masks of episode e's samples, read from its gtargets file."""
    with np.load(tg["dir"] / tg["names"][e], allow_pickle=False) as f:
        gage = f["gage"]
        t = f["t"]
    n = len(t)
    age = gage.reshape(n, -1)
    known = (tg["gknown_first"][e].ravel() >= 0)[None] \
        & (tg["gknown_first"][e].ravel()[None] <= t[:, None])
    inview = age == 0
    return {"t": t, "age": age, "inview": inview,
            "memory": known & ~inview, "never": ~known & ~inview}


# -- accumulators ---------------------------------------------------------------

def _f16_bins(p: np.ndarray) -> np.ndarray:
    return np.asarray(p, np.float16).view(np.uint16).astype(np.int64)


class APHist:
    """Exact average precision of float16 scores in [0, 1]."""

    def __init__(self):
        self.pos = np.zeros(N_F16, np.int64)
        self.neg = np.zeros(N_F16, np.int64)

    def add(self, p: np.ndarray, positive: np.ndarray) -> None:
        b = _f16_bins(p)
        self.pos += np.bincount(b[positive], minlength=N_F16)
        self.neg += np.bincount(b[~positive], minlength=N_F16)

    def ap(self) -> float:
        tp = np.cumsum(self.pos[::-1])  # threshold descending
        fp = np.cumsum(self.neg[::-1])
        if tp[-1] == 0:
            return float("nan")
        gain = self.pos[::-1] > 0
        return float(np.sum(self.pos[::-1][gain] / tp[-1] * tp[gain] / (tp[gain] + fp[gain])))

    @property
    def n_pos(self) -> int:
        return int(self.pos.sum())


class Balanced:
    """Pooled wall/free recalls."""

    def __init__(self):
        self.c = np.zeros(4, np.int64)  # n_wall, hit_wall, n_free, hit_free

    def add(self, hit: np.ndarray, wall: np.ndarray, mask: np.ndarray) -> None:
        self.c += [np.sum(mask & wall), np.sum(mask & wall & hit),
                   np.sum(mask & ~wall), np.sum(mask & ~wall & hit)]

    def value(self) -> float:
        nw, hw, nf, hf = self.c
        return 0.5 * (hw / nw + hf / nf) if nw and nf else float("nan")


def _age_bin(age: np.ndarray) -> np.ndarray:
    return np.digitize(age, AGE_BINS) - 1


def _neighbours(cells: np.ndarray, shape) -> list[np.ndarray]:
    h, w = shape
    out = []
    for c in cells:
        r, q = divmod(int(c), w)
        out.append(np.array([rr * w + qq for rr in range(r - 1, r + 2)
                             for qq in range(q - 1, q + 2)
                             if 0 <= rr < h and 0 <= qq < w]))
    return out


class GlobalScore:
    """Streams test episodes; metrics() gives the JSON-ready result."""

    def __init__(self, kind: str, half: int, shape):
        self.kind, self.half, self.shape = kind, half, shape
        self.groups: dict = {}
        self.per_episode: dict = {}

    def _group(self, name: str) -> dict:
        if name not in self.groups:
            n_age = len(AGE_BINS) - 1
            if self.kind == "glayout":
                self.groups[name] = {k: Balanced() for k in
                                     ("memory", "memory_2nd_half", "inview", "never_seen")}
                self.groups[name]["age"] = [Balanced() for _ in range(n_age)]
            else:
                self.groups[name] = {k: APHist() for k in
                                     ("memory", "inview", "never_seen", "all")}
                self.groups[name]["age"] = [APHist() for _ in range(n_age)]
                self.groups[name]["hits"] = np.zeros(3)   # boxes, hits, chance sum
                self.groups[name]["age_hits"] = np.zeros((n_age, 2))
                self.groups[name]["fa"] = np.zeros(2)     # collected cells, still in top-k
        return self.groups[name]

    def add_episode(self, e: int, source: str, p: np.ndarray, tg: dict) -> None:
        """p (n, window^2): the probe's probabilities for episode e's samples."""
        m = episode_masks(tg, e)
        p, inside = uncrop(p, tg["agent_rc"][tg["episode"] == e], tg["window"], self.shape)
        for k in ("inview", "memory", "never"):  # score only cells in the window
            m[k] = m[k] & inside
        m["inside"] = inside
        gs = [self._group("all"), self._group(source)]
        with np.errstate(invalid="ignore"):  # NaN outside the window; never scored
            if self.kind == "glayout":
                self._add_layout(e, p, m, tg, gs)
            else:
                self._add_boxes(e, p, m, tg, gs)

    def _add_layout(self, e, p, m, tg, gs) -> None:
        wall = np.broadcast_to(tg["glayout"][e].ravel() >= 0.5, p.shape)
        hit = (p >= 0.5) == wall
        late = (m["t"] >= self.half)[:, None]
        abin = _age_bin(m["age"])
        ep = Balanced()
        ep.add(hit, wall, m["memory"])
        self.per_episode[e] = ep.c.tolist()
        for g in gs:
            g["memory"].add(hit, wall, m["memory"])
            g["memory_2nd_half"].add(hit, wall, m["memory"] & late)
            g["inview"].add(hit, wall, m["inview"])
            g["never_seen"].add(hit, wall, m["never"])
            for b, acc in enumerate(g["age"]):
                acc.add(hit, wall, m["memory"] & (abin == b))

    def _add_boxes(self, e, p, m, tg, gs) -> None:
        n, n_cells = p.shape
        cell, coll, first = (tg[k][e] for k in ("box_cell", "box_collected_t", "box_first_hit_t"))
        alive = alive_at(coll, m["t"])                      # (n, B)
        observed = first[None, :] <= m["t"][:, None]
        rows = np.arange(n)[:, None]
        box_now = np.zeros((n, n_cells), bool)
        np.logical_or.at(box_now, (np.broadcast_to(rows, alive.shape)[alive],
                                   np.broadcast_to(cell, alive.shape)[alive]), True)
        known_box = np.zeros((n, n_cells), bool)
        sel = alive & observed
        np.logical_or.at(known_box, (np.broadcast_to(rows, sel.shape)[sel],
                                     np.broadcast_to(cell, sel.shape)[sel]), True)
        # Memory cells with a box nobody has seen yet are not scored.
        unknown_box = box_now & ~known_box
        memory = m["memory"] & ~unknown_box
        abin = _age_bin(m["age"])

        # top-k over the window per sample, k = window cells holding a box now
        inside = m["inside"]
        k_now = (box_now & inside).sum(1)
        ranked = np.where(inside, p, -np.inf).astype(np.float32)
        top = np.zeros((n, n_cells), bool)
        for i in range(n):
            if k_now[i]:
                top[i, np.argpartition(-ranked[i], k_now[i] - 1)[:k_now[i]]] = True
        nbrs = _neighbours(cell, self.shape)
        box_age = m["age"][:, cell]                          # (n, B)
        box_mem = sel & m["memory"][:, cell]
        hits, chance, n_boxes = 0, 0.0, 0
        age_hits = np.zeros((len(AGE_BINS) - 1, 2))
        for i, b in zip(*np.nonzero(box_mem)):
            h = bool(top[i, nbrs[b]].any())
            hits += h
            n_boxes += 1
            chance += 1 - (1 - k_now[i] / inside[i].sum()) ** len(nbrs[b])
            age_hits[_age_bin(box_age[i, b])] += [1, h]
        gone = ~alive & (coll[None, :] <= m["t"][:, None])
        fa_n = fa_hit = 0
        for i, b in zip(*np.nonzero(gone)):
            if inside[i, cell[b]] and not box_now[i, cell[b]]:
                fa_n += 1
                fa_hit += bool(top[i, cell[b]])

        ep_ap = APHist()
        ep_ap.add(p[memory], known_box[memory])
        self.per_episode[e] = {"ap_memory": ep_ap.ap(), "n_pos": ep_ap.n_pos,
                               "boxes": n_boxes, "hits": hits, "chance": chance}
        for g in gs:
            g["memory"].add(p[memory], known_box[memory])
            seen_now = m["inview"] & ~unknown_box
            g["inview"].add(p[seen_now], known_box[seen_now])
            g["never_seen"].add(p[m["never"]], box_now[m["never"]])
            g["all"].add(p[inside], box_now[inside])
            for b, acc in enumerate(g["age"]):
                sel_b = memory & (abin == b)
                acc.add(p[sel_b], known_box[sel_b])
            g["hits"] += [n_boxes, hits, chance]
            g["age_hits"] += age_hits
            g["fa"] += [fa_n, fa_hit]

    def metrics(self) -> dict:
        out = {}
        for name, g in self.groups.items():
            if self.kind == "glayout":
                rec = {"bal_memory": g["memory"].value(),
                       "bal_memory_2nd_half": g["memory_2nd_half"].value(),
                       "bal_inview": g["inview"].value(),
                       "bal_never_seen": g["never_seen"].value(),
                       "n_memory_cells": int(g["memory"].c[[0, 2]].sum()),
                       "memory_wall_share": float(g["memory"].c[0] / max(g["memory"].c[[0, 2]].sum(), 1)),
                       "bal_memory_by_age": [a.value() for a in g["age"]]}
            else:
                boxes, hits, chance = g["hits"]
                rec = {"ap_memory": g["memory"].ap(), "ap_inview": g["inview"].ap(),
                       "ap_never_seen": g["never_seen"].ap(), "ap_all": g["all"].ap(),
                       "n_memory_positives": g["memory"].n_pos,
                       "box_hit_rate": hits / boxes if boxes else float("nan"),
                       "box_hit_chance": chance / boxes if boxes else float("nan"),
                       "n_memory_boxes": int(boxes),
                       "collected_fa_rate": g["fa"][1] / g["fa"][0] if g["fa"][0] else float("nan"),
                       "n_collected_cells": int(g["fa"][0]),
                       "ap_memory_by_age": [a.ap() for a in g["age"]],
                       "box_hit_rate_by_age": [h / n if n else float("nan")
                                               for n, h in g["age_hits"]]}
            if name == "all":
                out.update(rec)
            else:
                out.setdefault("by_source", {})[name] = rec
        out["age_bins"] = AGE_BINS
        out["per_episode"] = {str(e): v for e, v in sorted(self.per_episode.items())}
        return out


def fixed_episodes(test: np.ndarray, groups: dict, per_group: int) -> list[int]:
    """The `per_group` lowest test episode ids of every source."""
    out = []
    for g in sorted(set(groups.values())):
        out += [int(e) for e in test if groups[int(e)] == g][:per_group]
    return sorted(out)
