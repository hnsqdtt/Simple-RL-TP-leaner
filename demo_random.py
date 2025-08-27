# -----------------------------
# FILE: demo_random.py
# -----------------------------

"""
Minimal smoke test (no ROS). Reads env_config.json from CWD, runs random policy,
saves trajectory to trajectory.npy and prints diagnostics.

Usage:
  python demo_random.py
"""

import json
import numpy as np
from pathlib import Path

from env.config import EnvConfig
from env.env_core import Env
from env.utils import LOGGER


def main():
    cfg = EnvConfig.from_json("env_config.json")
    env = Env(cfg)
    obs, info = env.reset(seed=42)
    LOGGER.info(f"path_exists={info['path_exists']} start_px={info['start_px']} goal_px={info['goal_px']}")

    traj = []
    for t in range(cfg.sim.max_steps):
        # random policy within limits
        a = np.array([
            np.random.uniform(-cfg.limits.v_max, cfg.limits.v_max),
            np.random.uniform(-cfg.limits.v_max, cfg.limits.v_max),
            np.random.uniform(-cfg.limits.omega_max, cfg.limits.omega_max),
        ], dtype=float)
        obs, r, term, trunc, info = env.step(a)
        traj.append([float(obs.state[0]), float(obs.state[1])])
        if term or trunc:
            LOGGER.info(f"ended at t={t} term={term} trunc={trunc} collided={info['collided']} reached={info['reached']}")
            break
    traj = np.array(traj, dtype=float)
    np.save("trajectory.npy", traj)
    LOGGER.info("Saved trajectory.npy")

if __name__ == "__main__":
    main()
