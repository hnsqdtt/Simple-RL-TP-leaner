# env/env_core_v2.py
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .config import EnvConfig
from .session import MapSession, SessionConfig
from .dynamics import State, clip_action, step_state
from .errors import EpisodeClosedError, StepError

@dataclass
class Observation:
    state: np.ndarray
    goal: np.ndarray
    limits: np.ndarray
    esdf_local: Optional[np.ndarray] = None
    esdf_global: Optional[np.ndarray] = None

class EnvV2:
    """固定地图 + 固定任务点训练；只有收敛后才换下一张地图。"""
    def __init__(self, cfg: EnvConfig, sess_cfg: SessionConfig, *, rng: Optional[np.random.Generator]=None):
        self.cfg = cfg
        self.session = MapSession(cfg.map, sess_cfg, safety_radius_m=cfg.limits.safety_radius_m, rng=rng)
        self._closed = False
        self._terminated = False
        self._truncated = False
        self._step = 0
        self._last_action = np.zeros(3, dtype=float)
        self._state = State(0.0, 0.0, 0.0)
        self._goal_xy = np.zeros(2, dtype=float)

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self.session.rng = np.random.default_rng(seed)
        self._closed = False
        self._terminated = False
        self._truncated = False
        self._step = 0
        self._last_action = np.zeros(3, dtype=float)

        # 锁定地图 + 取下一个固定任务
        t = self.session.next_task()
        self._goal_xy = np.array(t.goal_xy, dtype=float)
        # 初始姿态
        hd = t.heading
        if isinstance(hd, str):
            if hd == "random":
                hd = float(self.session.rng.uniform(-np.pi, np.pi))
            else:
                hd = 0.0
        self._state = State(float(t.start_xy[0]), float(t.start_xy[1]), float(hd))

        obs = self._make_obs()
        info = {"path_exists": True, "fixed_task": True}
        return obs, info

    def _make_obs(self) -> Observation:
        st = np.array([self._state.x, self._state.y, self._state.theta,
                       self._state.vx, self._state.vy, self._state.omega], dtype=float)
        lim = np.array([self.cfg.limits.v_max, self.cfg.limits.omega_max, self.cfg.limits.safety_radius_m], dtype=float)
        if self.cfg.obs.mode == "global":
            return Observation(st, self._goal_xy.copy(), lim, esdf_global=self.session.esdf.copy())
        # 局部 patch（与 V1 相同的轴对齐实现）
        H, W = self.session.md.free.shape
        pm = self.cfg.obs.patch_meters
        ph, pw = self.cfg.obs.patch_size
        size_px = np.array([pw, ph])
        meters = np.array([pm, pm])
        px_per_m = size_px / meters
        cx = int(round(self._state.x * px_per_m[0]))
        cy = int(round(self._state.y * px_per_m[1]))
        half_w = size_px[0]//2
        half_h = size_px[1]//2
        x0, x1 = cx - half_w, cx + half_w
        y0, y1 = cy - half_h, cy + half_h
        x0c, x1c = max(0, x0), min(W, x1)
        y0c, y1c = max(0, y0), min(H, y1)
        patch = np.zeros((ph, pw), dtype=float)
        patch_y0 = y0c - y0
        patch_x0 = x0c - x0
        patch[patch_y0:patch_y0+(y1c-y0c), patch_x0:patch_x0+(x1c-x0c)] = self.session.esdf[y0c:y1c, x0c:x1c]
        return Observation(st, self._goal_xy.copy(), lim, esdf_local=patch)

    def step(self, action: np.ndarray):
        if self._terminated or self._truncated:
            raise EpisodeClosedError("Episode finished; call reset().")
        try:
            a = clip_action(action, self.cfg.limits.v_max, self.cfg.limits.omega_max)
        except Exception as e:
            raise StepError(f"Bad action: {e}")

        prev = self._state
        ns = step_state(prev, a, self.cfg.sim.dt)
        self._state = ns
        self._step += 1

        # 碰撞
        r = int(round(ns.y / self.session.md.resolution))
        c = int(round(ns.x / self.session.md.resolution))
        H, W = self.session.md.free.shape
        if 0 <= r < H and 0 <= c < W:
            esdf_val = float(self.session.esdf[r, c])
        else:
            esdf_val = 0.0
        collided = esdf_val < self.cfg.limits.safety_radius_m
        # 到达
        reached = (np.linalg.norm(self._goal_xy - np.array([ns.x, ns.y])) <= self.cfg.sim.goal_tolerance_m)

        # 奖励与 V1 一致
        from .reward import compute_reward
        progress_prev = self._goal_xy - np.array([prev.x, prev.y])
        progress_now  = self._goal_xy - np.array([ns.x, ns.y])
        terms = compute_reward(progress_prev, progress_now, self._last_action, a,
                               esdf_val, self.cfg.limits.safety_radius_m,
                               reached, collided, self.cfg.reward,
                               False, self.cfg.limits.v_max)
        reward = terms.total(self.cfg.reward)
        self._last_action = a

        terminated = bool(collided or reached)
        truncated = bool(self._step >= self.cfg.sim.max_steps)
        self._terminated, self._truncated = terminated, truncated

        obs = self._make_obs()
        info = {"collided": collided, "reached": reached, "step": self._step}
        return obs, float(reward), terminated, truncated, info

    # ---- 会话控制 API ----
    def report_episode(self, success: bool):
        """在每个 episode 结束后调用；若为 auto 策略且达成收敛，则自动切图。"""
        self.session.tracker.update(success)
        if self.session.sess_cfg.advance_mode == "auto" and self.session.tracker.should_advance():
            from env.utils import LOGGER
            LOGGER.info("Auto advance triggered: switching to next generated map.")
            self.session.advance_map()
            return True
        return False

    def advance_map(self):
        """手动切换地图（例如你在训练脚本中判断收敛后调用）。"""
        self.session.advance_map()
