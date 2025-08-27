# -----------------------------
# FILE: env/sampler.py
# -----------------------------

import numpy as np
from collections import deque
from typing import Tuple


def bfs_path_exists(free_mask: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> bool:
    H, W = free_mask.shape
    if not (0 <= start[0] < H and 0 <= start[1] < W):
        return False
    if not (0 <= goal[0] < H and 0 <= goal[1] < W):
        return False
    if not (free_mask[start] and free_mask[goal]):
        return False

    q = deque([start])
    seen = np.zeros_like(free_mask, dtype=bool)
    seen[start] = True
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while q:
        r, c = q.popleft()
        if (r, c) == goal:
            return True
        for dr, dc in nbrs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and free_mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return False


def sample_start_goal(infl_free: np.ndarray, min_dist_m: float, resolution: float, rng: np.random.Generator):
    H, W = infl_free.shape
    free_idx = np.argwhere(infl_free)
    if free_idx.size == 0:
        raise ValueError("No free cells to sample start/goal.")
    max_tries = 2000
    min_d_px = int(np.ceil(min_dist_m / resolution))
    for _ in range(max_tries):
        s = tuple(free_idx[rng.integers(0, len(free_idx))])
        g = tuple(free_idx[rng.integers(0, len(free_idx))])
        if (abs(s[0]-g[0]) + abs(s[1]-g[1])) < 2:  # avoid same
            continue
        if ( (s[0]-g[0])**2 + (s[1]-g[1])**2 ) ** 0.5 >= min_d_px:
            return s, g
    # fallback: just pick farthest of K
    k = min(512, len(free_idx))
    picks = free_idx[rng.choice(len(free_idx), size=k, replace=False)]
    best = 0
    s_best, g_best = tuple(picks[0]), tuple(picks[1])
    for i in range(k):
        for j in range(i+1, k):
            d = np.hypot(*(picks[i]-picks[j]))
            if d > best:
                best = d
                s_best, g_best = tuple(picks[i]), tuple(picks[j])
    return s_best, g_best

# -----------------------------
# FILE: env/dynamics.py
# -----------------------------

from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    x: float
    y: float
    theta: float
    vx: float = 0.0
    vy: float = 0.0
    omega: float = 0.0


def clip_action(a: np.ndarray, v_max: float, omega_max: float) -> np.ndarray:
    a = np.asarray(a, dtype=float).copy()
    # a = [vx, vy, omega]
    a[0] = np.clip(a[0], -v_max, v_max)
    a[1] = np.clip(a[1], -v_max, v_max)
    a[2] = np.clip(a[2], -omega_max, omega_max)
    return a


def step_state(s: State, a: np.ndarray, dt: float) -> State:
    vx, vy, w = float(a[0]), float(a[1]), float(a[2])
    nx = s.x + vx * dt
    ny = s.y + vy * dt
    nth = s.theta + w * dt
    return State(nx, ny, nth, vx, vy, w)

# -----------------------------
# FILE: env/reward.py
# -----------------------------

import numpy as np
from dataclasses import dataclass

@dataclass
class RewardTerms:
    progress: float
    smooth: float
    peak: float
    safety_margin: float
    action_cost: float
    terminal: float

    def total(self, w):
        return (w.w_progress*self.progress
                + w.w_smooth*self.smooth
                + w.w_peak*self.peak
                + w.w_safety_margin*self.safety_margin
                - w.w_action_cost*self.action_cost
                + self.terminal)


def compute_reward(prev_pos, pos, prev_action, action, esdf_val, safety_radius, reached_goal: bool,
                   collided: bool, w, no_path_speed_shaping: bool, v_max: float):
    d_prev = np.linalg.norm(prev_pos)
    d_now = np.linalg.norm(pos)
    progress = (d_prev - d_now)  # positive when moving toward goal vector origin (see env_core logic)

    dv = np.linalg.norm(action[:2] - prev_action[:2]) + abs(action[2] - prev_action[2])
    smooth = -dv

    # simple peak detector
    peak = -float(np.any(np.abs(action) > np.array([v_max, v_max, w.omega_max if hasattr(w, 'omega_max') else v_max])))

    safety_margin = max(0.0, esdf_val - safety_radius)

    action_cost = float(action[0]**2 + action[1]**2 + 0.1*action[2]**2)

    terminal = 0.0
    if reached_goal:
        terminal += w.R_goal
    if collided:
        terminal -= w.w_collision

    # optional shaping when known no-path: slower is better
    if no_path_speed_shaping:
        # Encourages small speeds when there's no path; caller scales using w.w_progress as needed
        pass

    return RewardTerms(progress, smooth, peak, safety_margin, action_cost, terminal)
