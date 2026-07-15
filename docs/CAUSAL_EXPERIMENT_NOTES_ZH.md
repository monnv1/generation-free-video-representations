# Cosmos 因果去噪实验记录

## 背景与目的

目标是验证一个核心假设：在 Cosmos Video2World 的 DiT 中，中间 sigma 或短链去噪得到的 future latent / hidden states，是否比 `t=0` 或“直接沿用上一帧 latent”更接近真实未来帧 latent。

如果成立，说明 Cosmos 的中间去噪状态可能携带对动作有用的动态推演信息；如果不成立，后续基于这个方向的 WM4A 改造风险很高。

## 输入素材

- 视频：`/data/repos/cosmos-predict2.5/assets/base/robot_pouring.mp4`
- Prompt：`/data/repos/cosmos-predict2.5/assets/base/robot_pouring.txt`
- 视频信息：`1280x704`, `16 fps`, `93` 帧，约 `5.81s`
- 模型：`/data/repos/starVLA/playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World`

## 代码位置

实验代码独立放在：

```bash
/data/repos/cosmos_causal_probe
```

主要文件：

- `cosmos_causal_probe.py`：主实验脚本，包含 A1 因果短链和 A0 sanity check。
- `run_eval.sh`：单组评测入口。
- `run_history_sigma_steps_sweep.sh`：历史帧数 + sigma + 去噪步数 sweep。
- `aggregate_history_sweep.py`：聚合多个历史帧长度的 sweep 输出。
- `README.md`：运行说明。

## 可用环境

已确认 starVLA conda 环境可运行实验：

```bash
/data/miniconda3/envs/starVLA/bin/python
```

依赖状态：

- `torch 2.6.0+cu124`
- `diffusers 0.37.1`
- `transformers 4.57.0`
- CUDA 可用
- `PIL / cv2 / matplotlib / av` 可用

## Cosmos / VAE 关键事实

Cosmos Video2World 接口不硬编码历史帧数。可以传 1 帧 image，也可以传多帧 video。实际条件信息由输入视频帧数决定。

VAE temporal downsample factor 是 4：

```text
T_lat = (num_frames - 1) // 4 + 1
```

所以条件像素帧数和条件 latent slot 的关系是：

```text
1-4 帧   -> 1 个条件 latent slot
5-8 帧   -> 2 个条件 latent slots
9-12 帧  -> 3 个条件 latent slots
13-16 帧 -> 4 个条件 latent slots
```

之前 starVLA DOM Cosmos 配置里的 5 帧条件输入是人为设置：`history_k=4 + 当前帧 = 5`，不是 Cosmos 原生强制要求。

## 实验 A1：因果短链去噪

A1 是主实验，真正测试因果预测能力。

输入构造：

```text
历史条件帧 -> condition latent slots
未来 slot  -> 随机噪声初始化，不使用真实未来 latent
```

例如默认设置：

```text
frame 0..4 作为条件帧
frame 5..8 作为未来目标
```

在 `1280x704` 下，9 帧会得到：

```text
latent shape = [1, 16, 3, 88, 160]
condition latent slots = 0, 1
target future latent slot = 2
```

A1 对每个 `(sigma_start, K)`：

1. 初始化 future slot：

```python
x_future = randn_like(target_future_latent) * sigma_start
```

2. 跑 K 步短链去噪：

```text
sigma_start -> ... -> sigma_min=0.002
```

3. 每一步遵循 Diffusers Cosmos2VideoToWorldPipeline 的实际 preconditioning：

```python
current_t = sigma / (sigma + 1)
c_in = 1 - current_t      # 1 / (sigma + 1)
c_skip = 1 - current_t
c_out = -current_t
```

注意：这里不是标准 EDM 的 `1 / sqrt(sigma^2 + 1)`，而是 Cosmos pipeline 代码里的实际逻辑。

4. 条件 slot 使用 clean condition latent 覆盖，不乘 `c_in`；future slot 使用当前 noisy latent。

5. `K=1` 的语义是：

```text
1 次 DiT forward + 1 次 Euler update -> x_final
```

不是“只 forward 一次看 hidden state”。

## 实验 A0：noised true future sanity check

A0 不是因果测试，只是 sanity check。

A0 把真实未来 latent 加噪后喂给 DiT：

```python
x_sigma = (1 - sigma) * target_future_latent + sigma * eps
```

然后看 DiT 是否能还原真实未来 latent。

A0 成立只说明模型能从“带噪的真实未来”中恢复未来，不说明它能只靠历史帧预测未来。

## 指标定义

### 主指标：A1 净收益

```text
cos_x_final_vs_baseline_diff
= cos(x_final, target_future_latent)
  - cos(previous_condition_latent, target_future_latent)
```

其中：

- `x_final`：A1 短链 K 步 Euler update 后生成的 future latent。
- `target_future_latent`：真实未来帧对应的 latent，作为正确答案。
- `previous_condition_latent`：最后一个条件 latent slot，表示“直接沿用上一帧/上一条件 latent”的 baseline。

判读：

```text
> 0: Cosmos 因果短链优于直接沿用上一 latent。
< 0: Cosmos 因果短链不如直接沿用上一 latent。
```

### A1 其他指标

- `latent_cos_x_final`：`cos(x_final, target_future_latent)`。
- `latent_mse_x_final`：`MSE(x_final, target_future_latent)`。
- `latent_cos_pred_x0`：最后一次 DiT forward 预测的 clean latent 和 target 的 cosine。
- `latent_mse_pred_x0`：对应 MSE。
- `hidden_cos`：A1 hidden future-slot mean-pool 和 clean-reference hidden 的 cosine。
- `initial_noise_cos`：初始噪声 future slot 和 target 的 cosine。

### A0 指标

- `noisy_input_cos`：带噪真实未来和 target 的 cosine。
- `latent_cos_pred_x0`：DiT 从带噪真实未来中还原出的 clean estimate 和 target 的 cosine。
- `cos_pred_x0_vs_baseline_diff`：A0 pred_x0 相对 baseline 的提升。

## 已完成实验 1：默认 sigma + K sweep

输出目录：

```bash
/data/repos/cosmos_causal_probe/outputs/robot_pouring
```

配置：

```text
cond_frames = 5
future_pixel_frames = 4
sigma_start = 0.002, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0
K = 1, 3, 5
```

A1 baseline：

```text
prev_latent_baseline_cos = 0.8893
```

A1 gain heatmap：

```text
K/sigma  0.002   0.1    0.2    0.3    0.5    0.8    1.0
K=1     -0.890  -0.787 -0.778 -0.813 -0.815 -0.763 -0.725
K=3     -0.887  -0.805 -0.776 -0.786 -0.794 -0.751 -0.734
K=5     -0.888  -0.810 -0.789 -0.782 -0.795 -0.802 -0.750
```

最佳点：

```text
sigma=1.0, K=1
x_final cos = 0.1642
gain vs baseline = -0.7251
```

结论：A1 全部远低于 baseline，没有发现中间 sigma 或短链带来的正收益。

A0 sanity check：

```text
sigma=0.002 pred_x0 cos = 0.99999
sigma=0.1   pred_x0 cos = 0.99531
sigma=0.2   pred_x0 cos = 0.98925
sigma=0.3   pred_x0 cos = 0.98357
sigma=0.5   pred_x0 cos = 0.97179
sigma=0.8   pred_x0 cos = 0.87825
sigma=1.0   pred_x0 cos = 0.14809
```

A0 说明 DiT 能还原“带噪真实未来”，但这不是因果预测。

## 已完成实验 2：提高去噪步数

输出目录：

```bash
/data/repos/cosmos_causal_probe/outputs/robot_pouring_steps_sweep
```

配置：

```text
cond_frames = 5
future_pixel_frames = 4
sigma_start = 0.002, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0
K = 1, 3, 5, 10, 20, 35
```

A1 gain heatmap：

```text
K/sigma  0.002   0.1    0.2    0.3    0.5    0.8    1.0
1       -0.890  -0.787 -0.778 -0.813 -0.815 -0.763 -0.725
3       -0.887  -0.805 -0.776 -0.786 -0.794 -0.751 -0.734
5       -0.888  -0.810 -0.789 -0.782 -0.795 -0.802 -0.750
10      -0.889  -0.815 -0.804 -0.807 -0.820 -0.806 -0.765
20      -0.891  -0.810 -0.796 -0.812 -0.818 -0.762 -0.787
35      -0.892  -0.814 -0.794 -0.809 -0.823 -0.813 -0.768
```

最佳点仍然是：

```text
sigma=1.0, K=1
x_final cos = 0.1642
gain vs baseline = -0.7251
```

结论：提高到 35 步仍然没有带来正收益，且多数情况下更差。

## 已完成实验 3：历史帧 + Sigma + 去噪步数组合 sweep

输出目录：

```bash
/data/repos/cosmos_causal_probe/outputs/history_sigma_steps_sweep
```

配置：

```text
cond_frames = 1, 5, 9, 13
future_pixel_frames = 4
sigma_start = 0.002, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0, 2.0, 5.0
K = 1, 3, 5, 10, 20, 35
```

聚合输出：

- `aggregate_report.txt`：总览。
- `best_a1_by_history.csv`：每个历史长度下 A1 最佳组合。
- `combined_a1_results.csv`：全部 240 个 A1 组合。
- `combined_a0_results.csv`：全部 40 个 A0 sanity 组合。
- `best_gain_by_history.png`：不同历史长度下的最佳净收益图。
- `best_xfinal_vs_baseline_by_history.png`：不同历史长度下最佳 x_final 和 baseline 对比图。

实际完成规模：

```text
runs = 4
A1 rows = 240
A0 rows = 40
```

每个历史长度下的 A1 最佳点：

```text
cond_frames  cond_latent  best_sigma  best_K  baseline_cos  x_final_cos  gain_vs_baseline  pred_x0_cos  hidden_cos
1            1            0.8         20      0.8280        0.2326       -0.5954           0.2359       0.7043
5            2            2.0         1       0.8893        0.1943       -0.6950           0.1952       0.4593
9            3            2.0         1       0.9386        0.1103       -0.8283           0.1106       0.4993
13           4            1.0         1       0.9647        0.0722       -0.8925           0.0726       0.5837
```

全局最佳点：

```text
cond_frames = 1
cond_latent_frames = 1
sigma_start = 0.8
K = 20
x_final_cos = 0.2326
baseline_cos = 0.8280
gain_vs_baseline = -0.5954
```

Top 12 A1 组合全部来自 `cond_frames=1`，但仍全部是负收益：

```text
cond  sigma  K   x_final_cos  gain
1     0.8    20  0.2326       -0.5954
1     1.0    10  0.2165       -0.6115
1     1.0    3   0.2103       -0.6177
1     0.5    5   0.2084       -0.6196
1     1.0    35  0.2075       -0.6205
1     5.0    20  0.2071       -0.6210
1     2.0    20  0.2070       -0.6210
1     0.8    35  0.2066       -0.6214
1     0.8    3   0.2056       -0.6225
1     1.0    5   0.1994       -0.6286
1     5.0    35  0.1985       -0.6295
1     2.0    10  0.1976       -0.6305
```

A1 的重要现象：

1. 增加历史帧没有改善因果短链结果，反而 baseline 更高、A1 净收益更负。
2. `cond_frames=1` 的 baseline 最低，所以它的负收益看起来最小；这不是因为 Cosmos 预测好了，而是因为“上一 latent baseline”更弱。
3. `cond_frames=5/9/13` 下，历史越多，最后一个条件 latent 与目标 latent 的 cosine 越高：`0.8893 -> 0.9386 -> 0.9647`。这说明该视频相邻 latent 很相似，直接沿用历史末端已经是很强 baseline。
4. 在所有 240 个 `(history, sigma, K)` 组合里，没有任何一个 A1 组合超过 baseline。
5. 去噪步数从 1 到 35 没有形成稳定正向趋势；最佳 K 也不稳定，不能支持“多步短链带来动态推演收益”。

A0 sanity 在历史 sweep 中仍然成立。每个历史长度下，`sigma=0.002` 的 A0 pred_x0 基本等于真实 target：

```text
cond_frames  baseline_cos  A0_best_sigma  A0_pred_x0_cos  A0_gain
1            0.8280        0.002          1.0000          +0.1720
5            0.8893        0.002          1.0000          +0.1107
9            0.9386        0.002          1.0000          +0.0614
13           0.9647        0.002          1.0000          +0.0353
```

这再次说明：如果真实未来 latent 已经在输入里，DiT/预处理链路可以识别并保持它；但从纯噪声 future slot 出发，只靠历史条件没有产生超过 baseline 的未来 latent。

## 当前结论

基于已经完成的三个实验：

1. A0 成立：DiT 能从“带噪真实未来 latent”中还原真实未来。
2. A1 不成立：只给历史帧，从纯噪声 future slot 出发，短链去噪没有生成比上一条件 latent 更接近真实未来的结果。
3. 提高去噪步数到 35 没有改善。
4. 增加历史帧到 `1/5/9/13` 并扩大 sigma 到 `5.0` 后，A1 仍然没有任何正收益。
5. 当前证据不支持“中间 sigma / 短链去噪 hidden states 比 t=0 更接近真实未来 latent”的核心因果假设。
6. 对这个 robot_pouring 视频，最强的简单方法仍是直接沿用最后一个条件 latent；Cosmos 因果短链在当前实现和指标下明显不如该 baseline。

## 新增实验脚本：CFG + Negative Prompt + Multi-Slot 协同去噪

为排除两个 confound，已扩展实验代码：

1. 原生 Cosmos 常用 CFG，默认 `guidance_scale=7.0`，且未传 negative prompt 时走空字符串 negative prompt。
2. 原生生成不是单 future slot，而是多个 future latent slots 一起去噪，future slots 之间可以通过 self-attention 互相约束。

新增主脚本参数：

```text
--negative-prompt        negative prompt 文本或文件路径，空字符串表示空 negative prompt
--guidance-scales        逗号分隔，例如 1,3,5,7
--future-latent-slots    一次协同去噪的未来 latent slot 数 M
```

新增 sweep 入口：

```bash
/data/repos/cosmos_causal_probe/run_cfg_sequence_sweep.sh
```

默认组合：

```text
cond_frames = 1,5,9
future_latent_slots = 1,2,4
negative_prompt_modes = empty,quality
guidance_scales = 1,3,5,7
sigma_start = 0.5,0.8,1.0,2.0
K = 1,3,5,10,20
```

其中 `negative_prompt_modes=quality` 使用：

```bash
/data/repos/cosmos_causal_probe/negative_prompt_quality.txt
```

`future_latent_slots=M` 时，脚本自动设置：

```text
future_pixel_frames = M * 4
```

因为 Cosmos VAE temporal downsample factor 是 4。A1 会把 M 个 future slots 全部从噪声初始化，并同时送入 DiT 协同去噪。输出指标同时包含：

```text
latent_cos_x_final                         # 第一个 future slot
cos_x_final_vs_baseline_diff               # 第一个 future slot 相对 baseline 的净收益
sequence_cos_x_final                       # M 个 future slots 整段 flatten 后的 cosine
sequence_cos_x_final_vs_baseline_diff      # 整段 future sequence 相对 repeat-last-latent baseline 的净收益
```

48GB 显存下的 full-res `1280x704` 保守建议：

```text
优先跑：future_latent_slots = 1,2
可以试：future_latent_slots = 4
高风险 OOM：future_latent_slots >= 8
```

如果要试 `future_latent_slots=8`，建议先降到 `640x352` 看趋势。

