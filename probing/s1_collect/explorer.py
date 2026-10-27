"""Scripted explorer: the probing dataset's method-independent policy.

The Memory Maze paper recorded its offline dataset with a scripted planner
that navigates to random points ("we intentionally avoid recording data with a
trained agent ... to avoid dataset bias that could favor some methods"). Its
maze layout was known; ours is not exposed by Unity, so the planner works on
the map of what the agent has SEEN -- the same ExplorationTracker grid the
volumetric reward is built on, ray-cast from the true pose.

Goals alternate between the frontier of the unknown (exploration) and random
already-seen free cells at least `min_goal_cells` away (returns to known
places, i.e. the loops that make recall testable). Paths are BFS over seen-
free cells plus every cell the agent has stood in, followed by a pursuit
controller that turns in place for large heading errors.

Walls are NOT inflated by default. The seen map is already conservative: a
lidar return marks the cell it lands in, so walls come out two cells thick
and a 3 m corridor shows ONE free column. Inflating by one cell erased every
corridor -- in pilot episode ep0010 the reachable set dropped from 195 cells
to 1 and the explorer spun in place for 4950 steps. The agent's own cells are
added for the same reason the geodesic target adds them: a return landing in
the cell the agent drives through marks it OCCUPIED for good.

One env step is 0.1 s (measured in the pilot: 1.0 m forward at 10 m/s,
0.149 rad of turn at 1.5 rad/s), so a forward step crosses a whole 1 m cell.

Action format is the gym env's MultiDiscrete exactly (env.py:331-333):
    action_0 in {0: backward, 1: stop, 2: forward}
    action_1 in {0: -1.0, 1: -0.5, 2: 0, 3: +0.5, 4: +1.0} x max angular vel
and +angular_z turns LEFT (counter-clockwise, ROS convention), which is what
Twist2DActuator.cs applies ("ROS frame: +angular_z = left") and the direction
in which gt_pose's yaw grows.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np

UNKNOWN, FREE, OCCUPIED = -1, 0, 100
STOP, FORWARD, BACKWARD = 1, 2, 0
HARD_RIGHT, HALF_RIGHT, STRAIGHT, HALF_LEFT, HARD_LEFT = 0, 1, 2, 3, 4

_NEIGHBOURS = ((-1, 0), (1, 0), (0, -1), (0, 1),
               (-1, -1), (-1, 1), (1, -1), (1, 1))


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def dilate(mask: np.ndarray, r: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(r):
        grown = out.copy()
        grown[1:] |= out[:-1]
        grown[:-1] |= out[1:]
        grown[:, 1:] |= out[:, :-1]
        grown[:, :-1] |= out[:, 1:]
        out = grown
    return out


def bfs(passable: np.ndarray, start: tuple[int, int]):
    """8-connected BFS from `start` over `passable`.

    Returns (dist, parent) as flat arrays over the grid; dist -1 = unreachable.
    Diagonal moves may not cut a blocked corner.
    """
    h, w = passable.shape
    dist = np.full(h * w, -1, np.int32)
    parent = np.full(h * w, -1, np.int32)
    s = start[0] * w + start[1]
    dist[s] = 0
    queue = deque([s])
    flat = passable.ravel()
    while queue:
        cur = queue.popleft()
        r, c = divmod(cur, w)
        for dr, dc in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = nr * w + nc
            if dist[nxt] >= 0 or not flat[nxt]:
                continue
            if dr and dc and not (flat[r * w + nc] and flat[nr * w + c]):
                continue
            dist[nxt] = dist[cur] + 1
            parent[nxt] = cur
            queue.append(nxt)
    return dist, parent


def path_to(parent: np.ndarray, goal: int, w: int) -> list[tuple[int, int]]:
    out = []
    cur = goal
    while cur >= 0:
        out.append(divmod(int(cur), w))
        cur = parent[cur]
    return out[::-1]


class ScriptedExplorer:

    def __init__(self, seed: int = 0, p_frontier: float = 0.6, inflate: int = 0,
                 min_goal_cells: int = 15, replan_every: int = 80,
                 lookahead: int = 4, waypoint_tol: float = 1.2):
        self.rng = np.random.default_rng(seed)
        self.p_frontier = p_frontier
        self.inflate = inflate
        self.min_goal_cells = min_goal_cells
        self.replan_every = replan_every
        self.lookahead = lookahead
        self.waypoint_tol = waypoint_tol
        self.reset()

    def reset(self):
        self.path: list[tuple[int, int]] = []
        self.idx = 0
        self.steps_on_plan = 0
        self.backing_up = 0
        # Positions over consecutive FORWARD commands only: turning in place
        # for ~20 steps is normal (pi rad at 0.149 rad/step) and must not read
        # as stuck. 10 forward steps should cover ~10 m.
        self.history: deque = deque(maxlen=10)
        self.last = (STOP, STRAIGHT)
        self.visited: set[tuple[int, int]] = set()

    # -- planning -------------------------------------------------------------

    def _passable(self, grid: np.ndarray, start):
        free = grid == FREE
        passable = free & ~dilate(grid == OCCUPIED, self.inflate)
        for cell in self.visited | {start}:
            passable[cell] = True  # been there, so it is traversable
        return passable, free

    def plan(self, grid: np.ndarray, start: tuple[int, int],
             goal: tuple[int, int] | None = None) -> bool:
        passable, free = self._passable(grid, start)
        dist, parent = bfs(passable, start)
        h, w = grid.shape
        if goal is None:
            reach = dist.reshape(h, w) >= 0
            unknown = grid == UNKNOWN
            frontier = reach & dilate(unknown, 1) & free
            far = reach & (dist.reshape(h, w) >= self.min_goal_cells)
            pools = [frontier, far] if self.rng.random() < self.p_frontier \
                else [far, frontier]
            cells = next((np.flatnonzero(p) for p in pools if p.any()), None)
            if cells is None:
                return False
            target = int(self.rng.choice(cells))
        else:
            target = goal[0] * w + goal[1]
            if dist[target] < 0:
                return False
        self.path = path_to(parent, target, w)
        self.passable = passable
        self.idx = 0
        self.steps_on_plan = 0
        return len(self.path) > 1

    # -- control --------------------------------------------------------------

    def act(self, x: float, y: float, yaw: float, tracker,
            goal_xy: tuple[float, float] | None = None) -> tuple[int, int]:
        """One action for pose (x, y, yaw) on `tracker` (an ExplorationTracker:
        .grid[row, col], .world_to_cell, .origin_x/.origin_y, .resolution)."""
        if self.last[0] == FORWARD:
            self.history.append((x, y))
        else:
            self.history.clear()
        self.last = self._act(x, y, yaw, tracker, goal_xy)
        return self.last

    def _act(self, x, y, yaw, tracker, goal_xy):
        col, row = tracker.world_to_cell(x, y)
        h, w = tracker.grid.shape
        if not (0 <= row < h and 0 <= col < w):
            return STOP, HARD_LEFT
        self.visited.add((row, col))

        if self.backing_up:
            self.backing_up -= 1
            return BACKWARD, STRAIGHT
        if self._stuck():
            self.backing_up = 3
            self.path = []
            self.history.clear()
            return BACKWARD, STRAIGHT

        goal = None
        if goal_xy is not None:
            gc, gr = tracker.world_to_cell(*goal_xy)
            goal = (gr, gc)
        done = self.idx >= len(self.path) - 1
        if not self.path or done or self.steps_on_plan >= self.replan_every:
            if not self.plan(tracker.grid, (row, col), goal):
                return STOP, HARD_LEFT  # nothing known yet: look around
        self.steps_on_plan += 1

        res = tracker.resolution
        to_xy = lambda rc: (tracker.origin_x + (rc[1] + 0.5) * res,
                            tracker.origin_y + (rc[0] + 0.5) * res)
        # Advance past waypoints already reached.
        while self.idx < len(self.path) - 1:
            px, py = to_xy(self.path[self.idx])
            if math.hypot(px - x, py - y) > self.waypoint_tol:
                break
            self.idx += 1
        # Aim at the furthest waypoint within `lookahead` that a straight line
        # reaches over passable cells: a fixed lookahead cuts the corner of
        # every doorway the path turns through.
        last = min(self.idx + self.lookahead, len(self.path) - 1)
        target = next((k for k in range(last, self.idx, -1)
                       if self._clear(x, y, to_xy(self.path[k]), tracker)),
                      self.idx)
        return self.steer(x, y, yaw, *to_xy(self.path[target]))

    def _clear(self, x, y, txy, tracker, step: float = 0.25) -> bool:
        n = max(1, int(math.hypot(txy[0] - x, txy[1] - y) / step))
        for i in range(1, n + 1):
            c, r = tracker.world_to_cell(x + (txy[0] - x) * i / n,
                                         y + (txy[1] - y) * i / n)
            if not self.passable[r, c]:
                return False
        return True

    @staticmethod
    def steer(x, y, yaw, tx, ty) -> tuple[int, int]:
        err = wrap(math.atan2(ty - y, tx - x) - yaw)
        # Turn in place for anything large: forward + half turn is a 13 m
        # radius at full speed (10 m/s, 0.75 rad/s), wider than a corridor.
        if abs(err) > 0.25:
            return STOP, HARD_LEFT if err > 0 else HARD_RIGHT
        if abs(err) > 0.06:
            return FORWARD, HALF_LEFT if err > 0 else HALF_RIGHT
        return FORWARD, STRAIGHT

    def _stuck(self) -> bool:
        if len(self.history) < self.history.maxlen:
            return False
        (x0, y0), (x1, y1) = self.history[0], self.history[-1]
        return math.hypot(x1 - x0, y1 - y0) < 0.5
