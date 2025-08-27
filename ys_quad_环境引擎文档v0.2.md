# YSC-Quad 环境引擎文档（改进版 v0.2，Windows 优先）

> 目的：统一你当前仓库里的环境引擎实现（纯 Python / numpy），澄清**目录结构**、**模块职责**、**接口契约**，并指出与代码现状的差异和 TODO。本文兼容你现有 demo 与生成器脚本。

---

## 0. 目标与特性
- 单/会话两种环境：`Env`（每回合随机任务）与 `EnvV2 + Session`（同图固定若干任务，收敛后再切图）。
- 两类地图来源：**数据集**（`.pgm/.yaml`）与**随机生成器**（模块 `generate(**kwargs)`）。
- ESDF 碰撞判定（米单位）、BFS 可达性预检、简化动力学（vx, vy, ω）。
- Gym 风格 API：`reset() -> (obs, info)`、`step(a) -> (obs, r, terminated, truncated, info)`。
- Windows 优先，无 ROS 依赖；仅 `matplotlib` 用于可视化。

---

## 1) 推荐的目录结构（标准布局）
> 你可以直接采用此布局；如用现有平铺的 `.py` 文件，只需把对应文件移入 `env/` 目录并修正导入路径。

```
project_root/
├─ env/                                 # 环境引擎主包
│  ├─ __init__.py
│  ├─ config.py                         # EnvConfig 等数据类 & JSON 装载/校验
│  ├─ errors.py                         # 统一异常 & 错误码
│  ├─ utils.py                          # 日志、read_json、简单校验 require()
│  ├─ map_provider.py                   # 读取 .pgm/.yaml；加载生成器模块
│  ├─ esdf.py                           # ← 需补充：esdf_from_occupancy()
│  ├─ sampler.py                        # BFS 可达性、起终点采样
│  ├─ dynamics.py                       # 状态结构体、动作裁剪、状态积分
│  ├─ reward.py                         # 奖励项和合成
│  ├─ env_core.py                       # Env：每回合随机起终点
│  ├─ session.py                        # 任务会话、收敛追踪、切图策略
│  └─ env_core_v2.py                    # EnvV2：会话版（固定任务循环）
│
├─ random_map_generator_connected_v3.py # 生成器 A：高连通图（raw-only 接口）
├─ random_map_generator_rounded_v4_fixed.py # 生成器 B：圆角多边形障碍（raw-only）
│
├─ demo_random.py                       # 冒烟测试（随机策略）
└─ visualize_trajectory.py              # 轨迹绘制
```

> **与当前仓库的差异**：你现有代码里，`dynamics` 与 `reward` 的实现临时放在了 `sampler.py` 末尾；`esdf.py` 未落盘（但被导入），详见第 5 节 TODO。

---

## 2) 快速开始（dataset 模式）
1. 准备地图：把 `ys_my.yaml` 与 `ys_my.pgm` 放到 `./map/`。
2. 写 `env_config.json`：
```json
{
  "map": {
    "source": "dataset",
    "yaml_path": "./map/ys_my.yaml",
    "pgm_path": "./map/ys_my.pgm",
    "min_start_goal_m": 15.0
  },
  "limits": {"v_max": 1.2, "omega_max": 1.2, "safety_radius_m": 0.35},
  "sim": {"dt": 0.05, "max_steps": 1200, "goal_tolerance_m": 0.25},
  "reward": {
    "w_progress": 1.0, "w_smooth": 0.1, "w_peak": 0.05,
    "w_safety_margin": 0.5, "w_collision": 50.0, "w_action_cost": 0.01,
    "no_path_speed_shaping": true, "R_goal": 50.0, "R_timeout": 5.0
  },
  "obs": {"mode": "local", "patch_size": [96, 96], "patch_meters": 9.6}
}
```
3. 运行：
```bash
python demo_random.py
python visualize_trajectory.py
```

> 如需会话训练，额外提供 `session_config.json`（如 `sample_once` 采样 64 个任务、成功率阈值 0.9 等），再在你的训练脚本中使用 `EnvV2`。

---

## 3) 观测、动作与奖励（接口契约）
- **动作** `a=[v_x, v_y, ω]`，单位 `m/s, m/s, rad/s`，内部自动裁剪到 `v_max/omega_max`。
- **观测**（默认局部 patch）：
  - `state=[x,y,theta,vx,vy,omega]`
  - `goal=[g_x,g_y]`
  - `limits=[v_max, omega_max, safety_radius]`
  - `esdf_local`（HxW）或 `esdf_global`（HxW）
- **奖励**（默认实现，可自定义）：
  - `progress`（接近目标的距离减小量）
  - `smooth`（动作平滑惩罚）
  - `peak`（超阈峰值惩罚）
  - `safety_margin`（ESDF 与安全半径差值）
  - `action_cost`（速度/角速度代价）
  - 终止项：到达奖励、碰撞惩罚；步数截断为 `truncated`。

---

## 4) 模块一览（职责与关键 API）
- **env/config.py**：`EnvConfig.from_json(path)` 读入并校验配置；包含 `MapConfig/LimitsConfig/SimConfig/RewardConfig/ObsConfig`。
- **env/map_provider.py**：
  - `load_dataset_map(yaml_path, override_pgm=None)`：解析 ROS map_server 语义（`negate/occupied_thresh/free_thresh`），输出 `MapData{occ, free, resolution, origin}`。
  - `load_generated_map(module_name, kwargs=None)`：导入 `<module>.generate(**kwargs)`；允许：
    - 返回文件：`{"yaml_path":..., "pgm_path":...}`（随后用同一阈值解析），或
    - 返回内存：`{"free":bool(HxW), "resolution":float}`（True 表示自由）。
- **env/esdf.py**（建议新增）：`esdf_from_occupancy(occ: bool(HxW), resolution: float) -> float(HxW)`（米单位）。实现建议用 Felzenszwalb 两趟 1D 距离变换。
- **env/sampler.py**：
  - `bfs_path_exists(free_mask, start_px, goal_px)`：8 联通 BFS
  - `sample_start_goal(inflated_free, min_dist_m, resolution, rng)`：起终点像素采样（含距离门槛与兜底远距挑选）。
- **env/dynamics.py**：`State` 数据类；`clip_action()`；`step_state()`（欧拉积分）。
- **env/reward.py**：`compute_reward()` 返回 `RewardTerms` 并合成。
- **env/env_core.py**：
  - `reset()`：载图→ESDF→基于 **膨胀自由区** 采样起终点→可达性预检；若无路可走，内部标记 `no_path=True` 以便奖励整形。
  - `step(a)`：裁剪/积分→ESDF 碰撞→到达判据→奖励→终止/截断。
- **env/session.py**：
  - `MapSession`：锁定地图与任务集、`advance_map()`、`next_task()`；`ConvergenceTracker` 用滑窗成功率判收敛。
  - `SessionConfig`：`tasks_mode=fixed|sample_once`、`advance_mode=manual|auto`、`auto.success_rate/window/min_episodes`。
- **env/env_core_v2.py**：会话版环境；通过 `session.next_task()` 切换固定任务，`report_episode(success)` 按策略自动切换下一张生成器地图。
- **工具**：
  - `demo_random.py`：读 `env_config.json` 冒烟，随机策略生成 `trajectory.npy`。
  - `visualize_trajectory.py`：把 `trajectory.npy` 画到地图上，坐标用米并与像素对齐。
- **生成器脚本**：
  - `random_map_generator_connected_v3.py`：图状障碍 + 余裕圆二次连通保障；**raw-only 接口**。
  - `random_map_generator_rounded_v4_fixed.py`：圆角三角障碍 + 噪点修复；支持外侧噪点、arena 掩膜导出；**raw-only 接口**。

---

## 5) 与现状的差异 & TODO（重要）
1) **缺失 `env/esdf.py` 文件**：
   - 现有代码在多个位置从 `.esdf` 导入 `esdf_from_occupancy`；请新建 `env/esdf.py` 并实现：
   ```python
   import numpy as np
   def esdf_from_occupancy(occ: np.ndarray, resolution: float) -> np.ndarray:
       """简约实现：对 occ (True=占据/未知) 做两趟 1D 距离变换，返回米单位 ESDF。"""
       # 这里可放 Felzenszwalb 1D 二次距离变换；若赶时间，可先用欧氏距离的近似替代。
       ...
   ```
   - 注意：ESDF 中自由区值应 >=0，越大越安全；碰撞判据：`ESDF(x,y) < safety_radius_m`。

2) **`sampler.py` 中临时包含 `dynamics`/`reward`**：
   - 为保持与本文档一致、便于维护，建议把它们拆到独立的 `env/dynamics.py` 与 `env/reward.py`；拆分后请更新 `import` 路径。

3) **生成器接口统一为 raw-only**：
   - 你的两个生成器都已经按约定返回 `{"yaml_path", "pgm_path"}` 或 `{"free", "resolution"}`；继续保持 **只导出 raw 版本**，避免接口侧歧义。

4) **像素/米坐标统一**：
   - 规范：像素 `(row, col)`；米 `(x, y)`；转换：`x = col * resolution, y = row * resolution`。绘图时设置 `extent=[0, W*res, H*res, 0]` 以对齐。

---

## 6) 训练建议（简要）
- **算法**：SAC+CNN（推荐）> TD3+CNN > PPO+CNN（并行充分时）。
- **课程学习**：大安全半径/低速上限 → 逐步收紧 → 加入窄通道与外侧噪点地图。
- **预训练**：用生成器+BFS/跟踪生成专家数据，先做 BC，再用 SAC 微调。
- **指标**：成功率、最短路近似长度、`min ESDF`、碰撞次数、动作裁剪比、曲率等。

---

## 7) 常见问题（FAQ）
- **PGM 魔数错误**：确保是二进制 `P5`；否则解析失败。
- **未知区域如何处理**：按安全保守策略视为“占据”。
- **任务不可达**：会话/采样阶段均在**膨胀自由区**上做 BFS 预检；若全部不可达，请调大外形尺度或降低安全半径、放宽 `min_start_goal_m`。

---

## 8) 版本记录
- v0.2（本文）：给出标准目录、补齐 `esdf.py` 约定、明确生成器 raw-only、拆分 `dynamics/reward` 建议。
- v0.1：初版（Env / EnvV2 / Session / 错误码 / 中文文档）。

---

> 如需把这份文档导出为 Markdown/PDF 供团队协作，请直接在本页导出或复制。

