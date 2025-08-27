# YSC-Quad 环境引擎文档（重写版）

> 本文档覆盖当前画布中的最新代码（`env/`、`demo_random.py`、`visualize_trajectory.py`）与新增的会话机制（EnvV2 + Session）。适用于 **Windows 优先、纯 Python/numpy** 的研究/训练环境。

---

## 0. 目标与特性
- 统一的**环境引擎（Env）**：随机采样起终点、简单动力学、ESDF 判碰、奖励整合、Gym 风格 API。
- **会话版环境（EnvV2）**：锁定一张地图与一组任务点，直到收敛再切换下一张（生成器或数据集）。
- **两类地图来源**：数据集（`.pgm/.yaml`）与随机生成器（module.generate）。
- **健壮报错系统**：结构化错误码、清晰 hint；随附《ERROR_CODES.md》。
- **工具链**：随机策略冒烟（`demo_random.py`）与轨迹可视化（`visualize_trajectory.py`）。

---

## 1. 安装与依赖
- Python 3.9+
- 必需：`numpy`
- 可视化：`matplotlib`（仅 `visualize_trajectory.py` 使用）
- 可选：`PyYAML`（无则使用内置的精简 YAML 解析器）

```bash
pip install numpy matplotlib pyyaml
```

---

## 2. 目录结构
```
env/
  __init__.py        # 导出模块
  errors.py          # 统一异常与错误码
  utils.py           # 日志、JSON 读取、校验
  config.py          # 配置数据类与校验（EnvConfig）
  map_provider.py    # 数据集/生成器地图读取
  esdf.py            # Felzenszwalb 1D 两趟变换，输出米单位 ESDF
  sampler.py         # 起终点采样、BFS 可达性
  dynamics.py        # 简化平面动力学 + 动作裁剪
  reward.py          # 奖励项组合
  env_core.py        # Env：逐回合随机起终点

# 若使用会话机制（固定任务直至收敛再切图）
env/session.py       # 会话：锁图、任务集、收敛跟踪（success-rate 窗口）
env/env_core_v2.py   # EnvV2：固定任务循环 + 自动/手动切图

# 工具
demo_random.py       # 随机策略冒烟测试（生成 trajectory.npy）
visualize_trajectory.py  # 轨迹在地图上绘制
```

---

## 3. 快速开始
### 3.1 仅用一张固定地图 + 每回合随机任务（Env）
1. 准备 `./map/ys_my.yaml` 与对应 `.pgm`。
2. 写一个 `env_config.json`：
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
> Env 会在 **同一张地图** 上，每个 episode 的 `reset()` **随机**起点/终点与初始朝向。

### 3.2 同图多任务收敛后再切图（EnvV2 + Session）
1. 写 `session_config.json`（可与 `env_config.json` 并存）：
```json
{
  "tasks_mode": "sample_once",              
  "sample_k": 64,
  "min_start_goal_m": 15.0,
  "initial_heading": "random",              
  "tasks_unit": "meters",
  "advance_mode": "auto",
  "auto": {"window": 100, "success_rate": 0.9, "min_episodes": 50}
}
```
2. 训练脚本（伪代码）：
```python
from env.config import EnvConfig
from env.session import SessionConfig
from env.env_core_v2 import EnvV2
import json

cfg = EnvConfig.from_json("env_config.json")
sess = SessionConfig(**json.load(open("session_config.json", "r", encoding="utf-8")))
env = EnvV2(cfg, sess)

for ep in range(MAX_EPISODES):
    obs, info = env.reset()
    done = False
    while not done:
        action = policy(obs)  # 你的模型
        obs, r, term, trunc, info = env.step(action)
        done = term or trunc
    success = bool(info.get("reached", False))
    env.report_episode(success)  # auto: 达标则内部切到下一张图（仅 generator 模式）
```
> `tasks_mode = sample_once`：**每张图首次加载时随机采样 K 个任务并锁定**；之后按顺序循环这些任务直到收敛。
> 若要完全固定任务点，改用 `tasks_mode=fixed` 并在 `tasks` 列出 `[sx,sy,gx,gy,heading?]`（支持 `pixels` 或 `meters`）。

---

## 4. 配置说明
### 4.1 `env_config.json`
- `map.source`: `dataset` | `generator`
- `map.yaml_path/pgm_path`: 数据集路径（dataset）
- `map.generator_module/generator_kwargs`: 生成器模块名与参数（generator）
- `map.min_start_goal_m`: 采样起终点最小直线距离（米）
- `limits.v_max/omega_max/safety_radius_m`: 最大线/角速度与碰撞半径（米）
- `sim.dt/max_steps/goal_tolerance_m`: 步长、最大步数、到达判据（米）
- `reward`: 进展/平滑/峰值/安全余量/动作代价/终止项权重；`no_path_speed_shaping` 为“无通路时偏爱低速”的开关（示例未启具体形状）
- `obs.mode`: `local` | `global`; `patch_size`（HxW 像素）、`patch_meters`（窗口覆盖米数）

### 4.2 `session_config.json`（会话）
- `tasks_mode`: `fixed` | `sample_once`
- `tasks_unit`: `pixels` | `meters`
- `tasks`: 固定任务列表 `[[sx,sy,gx,gy,heading?], ...]`
- `initial_heading`: `zero` | `random` | `<数值>`（当 `tasks` 未给 heading 时的缺省策略）
- `advance_mode`: `manual` | `auto`（何时切换下一张图）
- `auto.window/success_rate/min_episodes`: 收敛判据滑窗大小、成功率阈值、最小回合数
- `sample_k`: `sample_once` 模式下每图采样任务数量
- `min_start_goal_m`: 与 `env_config.map.min_start_goal_m` 同义，用于当前会话的采样门槛

> `dataset` 模式下，`advance_map()` 仅重置任务循环（不会更换地图）。`generator` 模式下会调用生成器重新产图。

---

## 5. 地图与 ESDF
- 数据集地图：解析 `.yaml`（ROS map_server 语义），PGM 必须为 **二进制 P5**。未知区域视为障碍以保守安全。
- 生成器地图：约定 `module.generate(**kwargs)` 返回：
  - `{"yaml_path": ..., "pgm_path": ...}` **或** `{"free": HxW bool, "resolution": mpp}`。
- ESDF：对障碍布尔图（True=障碍/未知）做两趟 1D 二次距离变换，输出 **米**；碰撞判据为 `ESDF(x,y) < safety_radius_m`。

---

## 6. 观测与动作（模型接口契约）
- 动作 `a = [v_x, v_y, ω]`，单位 `m/s, m/s, rad/s`，内部会裁剪到 `v_max/omega_max`。
- 观测：
  - `obs.state = [x, y, theta, vx, vy, omega]`
  - `obs.goal = [g_x, g_y]`
  - `obs.limits = [v_max, omega_max, safety_radius]`
  - `obs.esdf_local`（HxW）或 `obs.esdf_global`（H×W）
- 常见做法：MLP 编码 `state+goal+limits`，CNN 编码 ESDF patch，二者 concat 后输入 Actor/Critic。

---

## 7. 奖励（默认实现）
- `progress`：靠近目标的距离减少量（按前后“目标向量”范数差）。
- `smooth`：动作变化惩罚（Δv/Δω）。
- `peak`：超阈峰值惩罚（简化实现）。
- `safety_margin`：ESDF 与安全半径的差值（>0 更安全）。
- `action_cost`：速度二范数与角速度二次代价。
- 终止项：达到奖励 `R_goal`、碰撞惩罚 `w_collision`。
> 你可在 `reward.py` 拓展曲率/抖动/能耗模型、或者实现 `no_path_speed_shaping` 的具体函数形状。

---

## 8. 会话（Session）与收敛
- `MapSession` 负责：加载/锁定地图、预计算 ESDF、在 **膨胀自由区** 上生成（或解析）任务集，并提供 `next_task()` 轮换任务。
- `ConvergenceTracker`：滑窗统计成功率（到达视为成功）。
- `EnvV2.report_episode(success)`：上报结果，若 `advance_mode=auto` 且达标则调用 `advance_map()`；`manual` 模式下请在外部根据你自己的指标手动调用。

---

## 9. 视觉化与坐标
- 栅格存储：像素 `(row, col)`；动力学/观测：米 `(x, y)`。
- 转换：`x = col * resolution`，`y = row * resolution`。
- `visualize_trajectory.py` 背景 `occ->黑/白`，轨迹使用米坐标，`extent=[0, W*res, H*res, 0]` 与像素对齐。

---

## 10. 错误码与排障
- 详见画布内 **ERROR_CODES.md**。核心代码会抛出：`CFG_BAD / MAP_LOAD / GEN_MISSING / STEP_BAD / EPI_CLOSED` 等。
- 任务不合法（长度<4、坐标越界、全部不可达）默认归入 `CFG_BAD`；如需细分，可自行在 `errors.py` 增加 `TaskError(code="TASK_BAD")` 并在 `session.py` 中抛出该异常。

---

## 11. 训练建议（简要）
- 算法优先级：**SAC+CNN**（首选） > **TD3+CNN** > **PPO+CNN**（若能强并行）。
- 课程学习：从大安全半径/低速上限开始 → 逐步收紧 → 引入窄通道/外侧噪点地图。
- 预训练：用生成器+BFS/跟踪生成专家数据，先做 BC（行为克隆）后接 SAC 微调。
- 指标：`success rate`、`path length`、`min ESDF`、`collision count`、`clip ratio`、`curvature`。

---

## 12. 常见问答（FAQ）
- **Q：Env 是否支持“随机起点/终点/初始朝向”？**
  A：支持。`Env.reset()` 每回合随机；`EnvV2` 的 `sample_once` 是“每图一次性采样并固定”。
- **Q：会话版如何手动切图？**
  A：`env.advance_map()`；`dataset` 模式仅重置任务循环，`generator` 模式会重新产图。
- **Q：PGM 报魔数错误**（P2/P5 等）？
  A：必须是二进制 P5；否则抛 `MAP_LOAD`。
- **Q：ESDF 开销大？**
  A：每图计算一次即可；必要时离线缓存或形态学近似。

---

## 13. 变更与兼容
- v0.1.0：提供 Env（随机起终点）与全套工具；新增 **EnvV2 + Session** 固定任务机制；完善错误码与中文文档。
- 向下兼容：保留 `env_core.Env` 不变；新增内容互不影响原有用法。

---

## 14. 单元测试建议（自备）
- **ESDF 正确性**：已知形状距离的数值对比。
- **采样/可达性**：`min_start_goal_m` 与 BFS 判定。
- **终止逻辑**：碰撞立即终止、到达给奖、步数截断。
- **会话循环**：`sample_once K` 个任务轮换、收敛阈值触发自动切图。

---

> 文档完。欢迎在此文档上直接批注/修改，我们保持与画布代码同步更新。

