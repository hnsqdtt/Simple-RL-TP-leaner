
# SAC + CNN 模型说明与对接指南（开箱即跑版）

> 适用场景：你的 YunShenChu 机器狗**环境引擎 v0.2**。本文档解释我给你的 **SAC（Soft Actor-Critic）+ CNN** 模型如何与环境对接、训练、调参，并附上**环境引擎文件结构图**，方便项目对外介绍/写报告时直接引用。

---

## 0. 一句话概览（你可以对外如此介绍）
- **输入**：ESDF 局部栅格（单通道图像） + 低维向量（位姿、目标、相对目标、指向角等）。  
- **策略**：高斯策略经 `tanh` 压缩到 \[-1,1]，再按物理上限 `[v_max, v_max, ω_max]` 缩放到真实动作。  
- **价值**：双 Q 网络（减少高估），目标网络软更新。  
- **探索**：最大化“回报 + 熵”，并用**自动温度 α**把策略熵拉到目标值（一般为 `-|A|`）。  
- **效果**：在随机/真实地图上**又快又稳**到达目标，同时保持碰撞安全与运动平滑。

---

## 1. 数据契约（Env ↔ 模型）

### 1.1 Observation 约定
环境 `reset()/step()` 返回的 `obs` 推荐包含这些键：

| 键名 | 形状/类型 | 含义 |
|---|---|---|
| `state` | `[x, y, θ, vx, vy, ω]`（缺项可为0） | 机器人当前位姿与速度 |
| `goal` | `[gx, gy]` | 目标在地图坐标系的位置（米） |
| `limits` | `[v_max, v_max, ω_max]` | 线速度/角速度上限（米/秒，弧度/秒） |
| `esdf_local` | `H×W` float32（米） | 以机器人为中心的 ESDF patch（无符号，米） |
| `reached` | bool | 是否到达（info 字段即可） |
| `collided` | bool | 是否碰撞（info 字段即可） |
| `no_path` | bool | 是否当前任务不可达（可选，用于奖励 shaping） |

> 说明：ESDF 在进入 CNN 之前会做归一化（剪裁到 `[0,1]`）。如果暂时没有 ESDF，可用零矩阵占位，流程仍可跑通。

### 1.2 Action 约定
- 模型输出 `a_raw ∈ [-1,1]^3`，经缩放得到物理动作：  
  `a = a_raw * [v_max, v_max, ω_max]`。  
- 环境内部仍可再次裁剪，但**建议**以 `limits` 缩放为准，避免在裁剪处产生不可导断点。

---

## 2. 模型结构（SAC + CNN）

```
ESDF patch (1×H×W) ─┐
                     ├─ CNNEncoder(Conv×3 + AdaptiveAvgPool + FC→256) ┐
low-dim vec (state,goal,g_rel,bearing,limits) ── VecEncoder(MLP→128) ─┤
                                                                      ├─ concat → trunk MLP(256)
Actor head: μ, logσ → reparameterize → tanh → scale by limits → a, logπ
Critic head (×2): concat(vec, action) + CNN → Q1(s,a), Q2(s,a)
```

- **Actor**：输出 `μ, logσ`；采样时 `a = tanh(μ + σ·ε)`，并做 **tanh 修正** 的对数概率 `logπ`。  
- **Critic**：**双 Q**（两套独立参数）。图像经 CNN，向量与动作拼接经 MLP，再融合输出 `Q(s,a)`。  
- **目标网络**：拷贝 Critic，按 `τ=5e-3` **软更新**。

### 2.1 损失函数
- **Critic**：
  \[y = r + γ · (1-done) · (min(Q1′,Q2′)(s′,a′) − α·logπ(a′|s′))]  
  最小化 \[(Q1−y)^2 + (Q2−y)^2]。  
- **Actor**：最小化 \[E_s (α·logπ(a|s) − min(Q1,Q2)(s,a))]。  
- **温度 α**（自动调参）：  
  最小化 \[E_s (−α·(logπ(a|s) + H_target))]，`H_target = −|A|`。

### 2.2 关键超参（默认）
| 名称 | 数值 | 说明 |
|---|---|---|
| 学习率 | `3e-4` | Actor/Critic/α 共用 |
| 折扣 γ | `0.99` | |
| 软更新 τ | `5e-3` | |
| Batch | `256` | |
| 经验池 | `1e6` | |
| 预热探索 | `3000` 步 | 均匀随机动作（已按 limits 缩放） |
| 目标熵 | `-|A|=-3` | A 维度为 3 |

---

## 3. 训练脚本与文件说明（rl_sac/）

- `rl_sac/sac_train.py`：训练主循环（支持 `--config` / `--session` / `--use-env-v2`）。  
- `rl_sac/sac_models.py`：CNN+MLP 的 **Actor/Critic** 实现（双 Q、tanh 修正、自动温度 α）。  
- `rl_sac/sac_buffer.py`：最简 **ReplayBuffer**。  
- `rl_sac/sac_utils.py`：`parse_obs()`（从 obs 中提取/组合输入张量）、`evaluate()`（无噪声评估）、存档工具。

### 3.1 关键接口：`parse_obs()`
- ESDF：`esdf_local`（米）→ 裁剪/缩放到 `[0,1]`，变成 `1×H×W`。  
- 向量：拼接 `[state, goal, goal - pos, bearing]`（可按需增改）。  
- 动作上限：`limits = [v_max, v_max, ω_max]`（直接用于 actor 缩放）。

---

## 4. 环境引擎文件结构图（v0.2 推荐）

```
project_root/
├─ env/                          # 环境引擎包
│  ├─ __init__.py
│  ├─ env_core.py                # 基础 Env（每回合随机任务）
│  ├─ env_core_v2.py             # EnvV2 + Session（固定任务/自动切图）
│  ├─ config.py                  # EnvConfig（from_json）
│  ├─ session.py                 # SessionConfig（from_json）
│  ├─ map_provider.py            # 地图加载/生成（dataset/generator/raw接口）
│  ├─ esdf.py                    # ← ESDF（Felzenszwalb EDT, 米）
│  ├─ dynamics.py                # ← 状态积分/动作裁剪
│  ├─ reward.py                  # ← 奖励项合成
│  ├─ utils.py
│  └─ errors.py
├─ rl_sac/                       # 强化学习（SAC）
│  ├─ sac_train.py               # 训练脚本（--config/--session）
│  ├─ sac_models.py              # 模型结构（Actor/Critic）
│  ├─ sac_buffer.py              # 经验回放
│  └─ sac_utils.py               # 观测解析/评估/IO
├─ map/
│  ├─ ys_my.yaml
│  └─ ys_my.pgm
├─ env_config.json               # 数据集地图配置（dataset 模式）
├─ env_config_generator.json     # 生成器地图配置（generator 模式）
├─ session_config.json           # EnvV2 会话配置（自动切图等）
├─ demo_random.py                # 随机策略冒烟测试（你已有）
├─ visualize_trajectory.py       # 可视化脚本（你已有）
├─ random_map_generator_connected_v3.py   # 随机地图生成脚本
├─ random_map_generator_rounded_v4_fixed.py
└─ runs/                         # 训练输出（ckpt、日志等）
```

> 如果你的仓库仍是**扁平结构**（`env_core.py` 等在根目录），只需把 `rl_sac/` 放到根目录，并保证 `config.py/session.py` 等可被 `import` 到即可。

---

## 5. 快速开始

### 5.1 安装（WSL Ubuntu）
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# 先用 CPU 版更稳；有 CUDA 再换官网命令
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install numpy tqdm
```

### 5.2 冒烟测试（随机策略）
```bash
python demo_random.py
python visualize_trajectory.py
```

### 5.3 训练（SAC + CNN）
```bash
# Env（每回合随机任务）
python -m rl_sac.sac_train --config env_config.json

# EnvV2 + Session（固定任务，达标自动切图）
python -m rl_sac.sac_train --use-env-v2 --config env_config_generator.json --session session_config.json
```

---

## 6. 调参建议（从稳到快）
- **先稳**：`v_max=0.6~0.8`、`omega_max=1.0~1.2`；`patch_size=96`；课程从“障碍稀/通道宽”到“障碍密/窄通道”。  
- **再快**：增大 `patch_meters`（例如 9.6 m）、提升上限、增大 `sample_k`。  
- **卡住时**：把 `start_random_steps` 提到 `10k`、适当提高奖励各项尺度（别太小）以让 Q 值有梯度。

---

## 7. 常见问题（FAQ）
**Q: ESDF 归一化有必要吗？**  
A: 有。我们用 `esdf/(3*safety_radius)` 剪裁到 `[0,1]`，能稳定 CNN 的数值范围。

**Q: limits 必须来自 obs 吗？**  
A: 是。这样 Actor 的缩放就与当前环境上限一致，避免训练/执行不一致。

**Q: 只有 `x,y,θ`，没有速度怎么办？**  
A: `parse_obs()` 会自动补0，训练可正常进行。

**Q: 想加“速度观测/障碍掩码/多通道ESDF”怎么办？**  
A: 在 `sac_utils.parse_obs()` 里扩展向量特征或将新通道 `np.stack` 在 ESDF 上，改 `CNNEncoder(in_ch=...)` 即可。

---

## 8. 版本/依赖说明
- Python 3.8+；PyTorch（CPU 或 CUDA）；`numpy`、`tqdm`。  
- 运行设备：Windows + WSL Ubuntu（优先），或原生 Linux 均可。

---

（完）
