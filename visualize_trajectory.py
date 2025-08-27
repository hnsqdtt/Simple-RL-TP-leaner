# -----------------------------
# FILE: visualize_trajectory.py
# -----------------------------

"""
Plot a saved trajectory (trajectory.npy) on the map loaded from env_config.json.

Usage:
  python visualize_trajectory.py
"""

import numpy as np
import matplotlib.pyplot as plt

from env.config import EnvConfig
from env.map_provider import load_dataset_map, load_generated_map


def main():
    cfg = EnvConfig.from_json("env_config.json")
    if cfg.map.source == "dataset":
        md = load_dataset_map(cfg.map.yaml_path, override_pgm=cfg.map.pgm_path)
    else:
        md = load_generated_map(cfg.map.generator_module, cfg.map.generator_kwargs)
    traj = np.load("trajectory.npy")  # [N,2] in meters (x,y)

    H, W = md.free.shape
    extent = [0, W*md.resolution, H*md.resolution, 0]  # x_min,x_max,y_min,y_max for imshow

    # background: obstacle=black, free=white
    bg = np.where(md.occ, 0.0, 1.0)

    plt.figure()
    plt.imshow(bg, cmap="gray", extent=extent)
    plt.plot(traj[:,0], traj[:,1], linewidth=2)
    plt.scatter([traj[0,0], traj[-1,0]], [traj[0,1], traj[-1,1]], marker="o")
    plt.title("Trajectory on Map")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

# -----------------------------