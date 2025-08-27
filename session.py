# env/session.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from .map_provider import load_dataset_map, load_generated_map, MapData
from .esdf import esdf_from_occupancy
from .sampler import bfs_path_exists, sample_start_goal
from .errors import ConfigError

@dataclass
class TaskSpec:
    start_xy: Tuple[float, float]  # meters
    goal_xy: Tuple[float, float]   # meters
    heading: float = 0.0           # rad

@dataclass
class AutoCriteria:
    window: int = 100
    success_rate: float = 0.9
    min_episodes: int = 50

@dataclass
class SessionConfig:
    # 任务点配置
    tasks_mode: str = "fixed"            # "fixed" | "sample_once"
    tasks_unit: str = "meters"           # "meters" | "pixels"
    tasks: Optional[List[List[float]]] = None  # [[sx,sy,gx,gy,[heading]]...]
    initial_heading: str = "zero"        # "zero" | "random" | numeric str
    # 地图切换策略
    advance_mode: str = "manual"         # "manual" | "auto"
    auto: AutoCriteria = field(default_factory=AutoCriteria)
    # 每张地图上采样任务（当 tasks_mode==sample_once 时）
    sample_k: int = 16
    min_start_goal_m: float = 15.0

class ConvergenceTracker:
    def __init__(self, crit: AutoCriteria):
        self.crit = crit
        self.hist: List[bool] = []

    def update(self, success: bool):
        self.hist.append(success)
        if len(self.hist) > self.crit.window:
            self.hist = self.hist[-self.crit.window:]

    def should_advance(self) -> bool:
        if len(self.hist) < self.crit.min_episodes:
            return False
        w = self.hist[-self.crit.window:]
        if not w:
            return False
        rate = sum(w) / len(w)
        return rate >= self.crit.success_rate

class MapSession:
    """锁定当前地图的生命周期与任务序列。
    - dataset: 只加载一次；advance_map() 对 dataset 无操作（或重置计数）。
    - generator: 首次加载并锁定；当 advance_map() 被触发时才重新调用生成器生成新地图。
    - 任务点：来自 config 固定表或在当前地图上一次性采样 sample_k 个并锁定。
    """
    def __init__(self, map_cfg, sess_cfg: SessionConfig, safety_radius_m: float, rng: Optional[np.random.Generator] = None):
        self.map_cfg = map_cfg
        self.sess_cfg = sess_cfg
        self.safety_radius_m = float(safety_radius_m)
        self.rng = rng or np.random.default_rng()

        self.md: Optional[MapData] = None
        self.esdf: Optional[np.ndarray] = None
        self.infl_free: Optional[np.ndarray] = None
        self.tasks: List[TaskSpec] = []
        self.task_idx: int = 0
        self.tracker = ConvergenceTracker(sess_cfg.auto)

    # --- 地图装载/切换 ---
    def _load_map(self):
        if self.map_cfg.source == "dataset":
            self.md = load_dataset_map(self.map_cfg.yaml_path, override_pgm=self.map_cfg.pgm_path)
        else:
            self.md = load_generated_map(self.map_cfg.generator_module, self.map_cfg.generator_kwargs)
        self.esdf = esdf_from_occupancy(self.md.occ, self.md.resolution)
        self.infl_free = self.esdf >= self.safety_radius_m

    def ensure_map(self):
        if self.md is None:
            self._load_map()

    def advance_map(self):
        if self.map_cfg.source == "generator":
            self._load_map()  # 重新生成
        # dataset 模式可选择重置统计
        self.task_idx = 0
        self.tasks.clear()
        self.tracker = ConvergenceTracker(self.sess_cfg.auto)
        self._prepare_tasks()

    # --- 任务准备 ---
    def _prepare_tasks(self):
        assert self.md is not None and self.esdf is not None and self.infl_free is not None
        if self.sess_cfg.tasks_mode == "fixed":
            if not self.sess_cfg.tasks:
                raise ConfigError("tasks_mode=fixed 但未提供 tasks")
            self.tasks = []
            for row in self.sess_cfg.tasks:
                if len(row) < 4:
                    raise ConfigError("每个 task 至少包含 [sx,sy,gx,gy]")
                sx, sy, gx, gy = row[:4]
                hd = float(row[4]) if len(row) >= 5 else 0.0
                if self.sess_cfg.tasks_unit == "pixels":
                    r2m = self.md.resolution
                    # pixels -> meters: x=col*res, y=row*res
                    sx, sy, gx, gy = (sx*r2m, sy*r2m, gx*r2m, gy*r2m)
                self.tasks.append(TaskSpec((sx, sy), (gx, gy), hd))
        elif self.sess_cfg.tasks_mode == "sample_once":
            # 在当前地图上一次性采样 K 个任务并锁定
            K = int(self.sess_cfg.sample_k)
            rng = self.rng
            for _ in range(K):
                s_px, g_px = sample_start_goal(self.infl_free, self.sess_cfg.min_start_goal_m, self.md.resolution, rng)
                sx, sy = s_px[1]*self.md.resolution, s_px[0]*self.md.resolution
                gx, gy = g_px[1]*self.md.resolution, g_px[0]*self.md.resolution
                self.tasks.append(TaskSpec((sx, sy), (gx, gy), 0.0))
        else:
            raise ConfigError("未知 tasks_mode: %s" % self.sess_cfg.tasks_mode)
        # 可达性预检
        ok = 0
        for t in self.tasks:
            s_px = (int(round(t.start_xy[1]/self.md.resolution)), int(round(t.start_xy[0]/self.md.resolution)))
            g_px = (int(round(t.goal_xy[1]/self.md.resolution)), int(round(t.goal_xy[0]/self.md.resolution)))
            if bfs_path_exists(self.infl_free, s_px, g_px):
                ok += 1
        if ok == 0:
            raise ConfigError("任务集在膨胀自由区内全部不可达，请检查 tasks 或 safety_radius_m/min_start_goal_m")

    def ensure_tasks(self):
        if not self.tasks:
            self._prepare_tasks()

    def next_task(self) -> TaskSpec:
        self.ensure_map()
        self.ensure_tasks()
        t = self.tasks[self.task_idx % len(self.tasks)]
        self.task_idx += 1
        return t
