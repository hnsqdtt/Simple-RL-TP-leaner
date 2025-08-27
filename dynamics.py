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
