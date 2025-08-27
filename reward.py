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

def compute_reward(prev_pos, pos, prev_action, action, esdf_val, safety_radius,
                   reached_goal: bool, collided: bool, w,
                   no_path_speed_shaping: bool, v_max: float):
    # 越靠近目标越好
    d_prev = np.linalg.norm(prev_pos)
    d_now  = np.linalg.norm(pos)
    progress = (d_prev - d_now)

    # 平滑性惩罚（相邻动作差）
    dv = np.linalg.norm(action[:2] - prev_action[:2]) + abs(action[2] - prev_action[2])
    smooth = -dv

    # 峰值惩罚（有无明显的超限趋势；角速度上限未知则用 v_max 近似）
    omega_max_guess = v_max
    peak = -float(np.any(np.abs(action) > np.array([v_max, v_max, omega_max_guess])))

    # 碰撞距离冗余（esdf - 安全半径）
    safety_margin = max(0.0, float(esdf_val) - float(safety_radius))

    # 动作能耗
    action_cost = float(action[0]**2 + action[1]**2 + 0.1*action[2]**2)

    # 终止项
    terminal = 0.0
    if reached_goal: terminal += w.R_goal
    if collided:     terminal -= w.w_collision

    # 无可达路时的速度整形（可选）
    if no_path_speed_shaping:
        pass

    return RewardTerms(progress, smooth, peak, safety_margin, action_cost, terminal)
