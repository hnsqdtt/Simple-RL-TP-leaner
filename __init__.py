# =============================
# YSC-Quad EnvEngine (Windows-first, pure Python/numpy)
# Version: 0.1.0
# =============================
#
# This is a minimal yet practical environment engine scaffold tailored for
# your quadruped path-planning RL project. It provides:
# - Config loading & validation (JSON)
# - Map readers (Cartographer .pgm/.yaml) + optional Generator adapter
# - ESDF (Felzenszwalb 1D transform; two-pass) in meters
# - Start/goal sampling with BFS reachability check
# - Simple nonholonomic-like planar dynamics (vx, vy, omega)
# - Reward shaping (progress, smoothness, safety margin, action cost)
# - Gym-like API (reset/step)
# - Robust error system with codes, friendly messages, and suggestions
# - Demo runner (random policy) and simple trajectory visualizer
#
# Directory layout encoded in one file for convenience. Save each section
# to the indicated path if you split into files.
#
# Python 3.9+
# Requires: numpy, matplotlib (only for visualize_trajectory.py)
# Optional: PyYAML (if absent, a tiny fallback YAML reader is used)
#
# -----------------------------
# FILE: env/__init__.py
# -----------------------------

__all__ = [
    "errors",
    "config",
    "utils",
    "map_provider",
    "esdf",
    "sampler",
    "dynamics",
    "reward",
    "env_core",
]

__version__ = "0.1.0"

# -----------------------------