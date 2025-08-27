# -----------------------------
# FILE: env/env_core.py
# -----------------------------

import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from .config import EnvConfig
from .map_provider import load_dataset_map, load_generated_map
from .esdf import esdf_from_occupancy
from .sampler import bfs_path_exists, sample_start_goal
from .dynamics import State, clip_action, step_state
from .reward import compute_reward
from .errors import EpisodeClosedError, StepError
from .utils import LOGGER

@dataclass
class Observation:
    state: np.ndarray  # [x,y,theta,vx,vy,omega]
    goal: np.ndarray   # [gx,gy]
    limits: np.ndarray # [v_max, omega_max, safety_radius]
    esdf_local: Optional[np.ndarray] = None
    esdf_global: Optional[np.ndarray] = None

class Env:
    def __init__(self, cfg: EnvConfig, *, rng: Optional[np.random.Generator] = None):
        self.cfg = cfg
        self.rng = rng or np.random.default_rng()
        self._closed = False
        self._terminated = False
        self._truncated = False
        self._step = 0
        self._esdf = None
        self._free = None
        self._infl_free = None
        self._resolution = None
        self._goal_xy = None  # in meters map frame
        self._no_path = False
        self._last_action = np.zeros(3, dtype=float)
        self._state = State(0.0, 0.0, 0.0)

    def reset(self, seed: Optional[int] = None, map_spec: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._closed = False
        self._terminated = False
        self._truncated = False
        self._step = 0
        self._last_action = np.zeros(3, dtype=float)

        # Load map
        if self.cfg.map.source == "dataset":
            md = load_dataset_map(self.cfg.map.yaml_path, override_pgm=self.cfg.map.pgm_path)
        else:
            md = load_generated_map(self.cfg.map.generator_module, self.cfg.map.generator_kwargs)
        self._resolution = md.resolution
        self._free = md.free
        # ESDF
        self._esdf = esdf_from_occupancy(md.occ, md.resolution)
        # Inflated free = esdf >= safety_radius
        self._infl_free = self._esdf >= self.cfg.limits.safety_radius_m

        # Sample start/goal in pixel indices
        s_px, g_px = sample_start_goal(
            self._infl_free,
            self.cfg.map.min_start_goal_m,
            md.resolution,
            self.rng,
        )
        # Pix -> meters (origin at (0,0); use pixel center)
        def px_to_m(p):
            r, c = p
            return np.array([c * md.resolution, r * md.resolution], dtype=float)

        s_xy = px_to_m(s_px)
        g_xy = px_to_m(g_px)
        self._goal_xy = g_xy

        # Initialize state at start (theta random)
        self._state = State(s_xy[0], s_xy[1], float(self.rng.uniform(-np.pi, np.pi)))

        # Reachability check (on inflated free)
        self._no_path = not bfs_path_exists(self._infl_free, s_px, g_px)
        if self._no_path:
            LOGGER.warning("No reachable path on inflated free space; shaping will prefer slower moves.")

        obs = self._make_obs()
        info = {"path_exists": (not self._no_path), "start_px": s_px, "goal_px": g_px}
        return obs, info

    def _make_obs(self) -> Observation:
        st = np.array([self._state.x, self._state.y, self._state.theta,
                       self._state.vx, self._state.vy, self._state.omega], dtype=float)
        lim = np.array([self.cfg.limits.v_max, self.cfg.limits.omega_max, self.cfg.limits.safety_radius_m], dtype=float)
        if self.cfg.obs.mode == "global":
            return Observation(st, self._goal_xy.copy(), lim, esdf_global=self._esdf.copy())
        else:
            # local patch, axis-aligned (no rotation for simplicity)
            H, W = self._free.shape
            pm = self.cfg.obs.patch_meters
            ph, pw = self.cfg.obs.patch_size
            # patch in pixels
            size_px = np.array([pw, ph])  # W,H order
            meters = np.array([pm, pm])
            px_per_m = size_px / meters
            # center at state
            cx = int(round(self._state.x * px_per_m[0]))
            cy = int(round(self._state.y * px_per_m[1]))
            half_w = size_px[0]//2
            half_h = size_px[1]//2
            x0, x1 = cx - half_w, cx + half_w
            y0, y1 = cy - half_h, cy + half_h
            # clamp
            x0c, x1c = max(0, x0), min(W, x1)
            y0c, y1c = max(0, y0), min(H, y1)
            patch = np.zeros((ph, pw), dtype=float)
            patch_y0 = y0c - y0
            patch_x0 = x0c - x0
            patch[patch_y0:patch_y0+(y1c-y0c), patch_x0:patch_x0+(x1c-x0c)] = self._esdf[y0c:y1c, x0c:x1c]
            return Observation(st, self._goal_xy.copy(), lim, esdf_local=patch)

    def _goal_vec(self):
        return self._goal_xy - np.array([self._state.x, self._state.y], dtype=float)

    def step(self, action: np.ndarray):
        if self._terminated or self._truncated:
            raise EpisodeClosedError("Episode already finished. Call reset() before step().",
                                     hint="Avoid calling step after terminated/truncated=True.")
        try:
            a = clip_action(action, self.cfg.limits.v_max, self.cfg.limits.omega_max)
        except Exception as e:
            raise StepError(f"Bad action shape/value: {e}")

        prev_state = self._state
        ns = step_state(prev_state, a, self.cfg.sim.dt)
        self._state = ns
        self._step += 1

        # collision check via ESDF
        # map meters -> pixel indices for sampling esdf value
        r = int(round(ns.y / self._resolution))
        c = int(round(ns.x / self._resolution))
        H, W = self._free.shape
        if 0 <= r < H and 0 <= c < W:
            esdf_val = float(self._esdf[r, c])
        else:
            esdf_val = 0.0  # outside = collision
        collided = esdf_val < self.cfg.limits.safety_radius_m

        # goal check
        goal_vec_prev = self._goal_vec() + np.array([prev_state.x - ns.x, prev_state.y - ns.y])
        goal_vec_now = self._goal_vec()
        reached = np.linalg.norm(goal_vec_now) <= self.cfg.sim.goal_tolerance_m

        terms = compute_reward(goal_vec_prev, goal_vec_now, self._last_action, a,
                               esdf_val, self.cfg.limits.safety_radius_m,
                               reached, collided, self.cfg.reward,
                               self._no_path and self.cfg.reward.no_path_speed_shaping,
                               self.cfg.limits.v_max)
        reward = terms.total(self.cfg.reward)

        terminated = bool(collided or reached)
        truncated = bool(self._step >= self.cfg.sim.max_steps)
        self._terminated = terminated
        self._truncated = truncated
        self._last_action = a

        obs = self._make_obs()
        info = {
            "collided": collided,
            "reached": reached,
            "esdf_here": esdf_val,
            "no_path": self._no_path,
            "step": self._step,
        }
        return obs, float(reward), terminated, truncated, info
