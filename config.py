# -----------------------------
# FILE: env/config.py
# -----------------------------

from dataclasses import dataclass
from typing import Optional, Tuple, Literal, Dict, Any

from .utils import read_json, require

ObsMode = Literal["local", "global"]

@dataclass
class MapConfig:
    source: Literal["dataset", "generator"] = "dataset"
    width: Optional[int] = None
    height: Optional[int] = None
    resolution: Optional[float] = None
    min_start_goal_m: float = 15.0
    yaml_path: Optional[str] = None  # when source=dataset
    pgm_path: Optional[str] = None   # optional override
    generator_module: Optional[str] = None  # when source=generator (e.g., "random_map_generator_connected_v3_hr")
    generator_kwargs: Optional[Dict[str, Any]] = None

@dataclass
class LimitsConfig:
    v_max: float = 1.2
    omega_max: float = 1.2
    safety_radius_m: float = 0.35

@dataclass
class SimConfig:
    dt: float = 0.05
    max_steps: int = 1200
    goal_tolerance_m: float = 0.25

@dataclass
class RewardConfig:
    w_progress: float = 1.0
    w_smooth: float = 0.1
    w_peak: float = 0.05
    w_safety_margin: float = 0.5
    w_collision: float = 50.0
    w_action_cost: float = 0.01
    no_path_speed_shaping: bool = True
    R_goal: float = 50.0
    R_timeout: float = 5.0

@dataclass
class ObsConfig:
    mode: ObsMode = "local"
    patch_size: Tuple[int, int] = (96, 96)
    patch_meters: float = 9.6

@dataclass
class EnvConfig:
    map: MapConfig
    limits: LimitsConfig
    sim: SimConfig
    reward: RewardConfig
    obs: ObsConfig

    @staticmethod
    def from_json(path: str) -> "EnvConfig":
        raw = read_json(path)
        # minimal validation
        require("map" in raw and isinstance(raw["map"], dict), "Missing 'map' in config")
        require("limits" in raw and isinstance(raw["limits"], dict), "Missing 'limits' in config")
        require("sim" in raw and isinstance(raw["sim"], dict), "Missing 'sim' in config")
        require("reward" in raw and isinstance(raw["reward"], dict), "Missing 'reward' in config")
        require("obs" in raw and isinstance(raw["obs"], dict), "Missing 'obs' in config")

        m = raw["map"]
        mc = MapConfig(**m)
        lc = LimitsConfig(**raw["limits"])
        sc = SimConfig(**raw["sim"])
        rc = RewardConfig(**raw["reward"])
        oc = ObsConfig(**raw["obs"])
        return EnvConfig(mc, lc, sc, rc, oc)
