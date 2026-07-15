# Cosmos Causal Sigma Probe

这个目录包含 WM4A/Cosmos 因果验证实验的独立脚本，不修改 `starVLA` 或 `cosmos-predict2.5` 仓库。

## 文件

- `cosmos_causal_probe.py`: 主评测脚本。
- `run_eval.sh`: 默认评测入口。
- `outputs/`: 运行后生成 CSV、JSON 和图。

## 默认输入

- 视频: `/data/repos/cosmos-predict2.5/assets/base/robot_pouring.mp4`
- Prompt: `/data/repos/cosmos-predict2.5/assets/base/robot_pouring.txt`
- 模型: `/data/repos/starVLA/playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World`

默认视频是 `1280x704`, `16 fps`, `93` 帧。脚本默认取第 `0..8` 帧：`0..4` 做条件帧，`5..8` 形成一个未来 latent slot。

## 关键实现约定

Cosmos VAE temporal downsample factor 是 4，因此：

```text
9 pixel frames -> T_lat = (9 - 1) // 4 + 1 = 3
condition pixel frames 0..4 -> condition latent slots 0..1
future pixel frames 5..8 -> target latent slot 2
```

当前本地 Cosmos-Predict2 模型在 `1280x704` 下的 latent shape 约为：

```text
[1, 16, 3, 88, 160]
```

这和 DOM 训练配置的 `576x320 -> [1,16,T,40,72]` 不同；这里按提供的视频原生分辨率跑。

脚本按 Diffusers `Cosmos2VideoToWorldPipeline` 的实际逻辑做 preconditioning：

```python
current_t = sigma / (sigma + 1)
c_in = 1 - current_t      # 1 / (sigma + 1)
c_skip = 1 - current_t
c_out = -current_t
```

注意这不是标准 EDM 的 `1 / sqrt(sigma^2 + 1)`。

条件 latent slot 会被 clean condition latent 覆盖，不乘 `c_in`；future slot 使用当前 noisy latent 并按当前 sigma 做 preconditioning。条件 slot 的 timestep 使用 `sigma_conditioning=0.0001`。

## 实验内容

### A1: 因果短链去噪

未来 latent slot 不使用真实未来 latent 初始化，而是：

```python
x_future = randn_like(target_future) * sigma_start
```

对每个 `sigma_start` 和 `K` 跑短链：

```text
sigma_start -> ... -> sigma_min=0.002
```

`K=1` 的语义是：

```text
1 次 DiT forward + 1 次 Euler update -> x_final
```

不是“只 forward 一次看 hidden state”。

默认消融：

```text
sigma_start = 0.002,0.1,0.2,0.3,0.5,0.8,1.0
K = 1,3,5
```

主判据：

```text
cos_x_final_vs_baseline_diff
  = cos(x_final, target_future_latent)
  - cos(previous_condition_latent, target_future_latent)
```

这个差值表示相对“直接沿用上一条件 latent slot”的净收益。

### A0: noised true future sanity check

把真实未来 latent 加噪后喂给 DiT，只跑单次 forward：

```python
x_sigma = (1 - sigma) * target_future + sigma * eps
```

A0 不是因果预测实验，只用于确认 DiT 在带噪真未来输入下能否还原目标 latent。

## 运行

需要一个包含 `torch`, `transformers`, `diffusers`, `Pillow`, `opencv-python`, `matplotlib` 的环境。当前机器默认 `/data/miniconda3/bin/python` 缺少 `torch/diffusers`，所以通常需要显式指定环境：

```bash
cd /data/repos/cosmos_causal_probe
PYTHON_BIN=/path/to/env/bin/python ./run_eval.sh
```

常用覆盖项：

```bash
PYTHON_BIN=/path/to/env/bin/python \
CUDA_VISIBLE_DEVICES=0 \
OUTPUT_DIR=/data/repos/cosmos_causal_probe/outputs/robot_pouring_test \
./run_eval.sh \
  --sigmas 0.002,0.1,0.2,0.3,0.5,0.8,1.0 \
  --denoise-steps 1,3,5 \
  --frame-start 0
```

如果显存不够，可以先降分辨率做工程 sanity check：

```bash
PYTHON_BIN=/path/to/env/bin/python ./run_eval.sh --width 576 --height 320
```

降分辨率结果不能直接和原生 `1280x704` 的结论混用。

## 输出

运行后输出到 `OUTPUT_DIR`：

- `a1_results.csv`: A1 主实验表。
- `a0_results.csv`: A0 sanity check 表。
- `summary.json`: 配置和 latent shape。
- `a1_gain_heatmap.png`: `sigma_start x K` 的净收益热力图。
- `a1_x_final_curves.png`: A1 `x_final` cosine 曲线。
- `a0_sanity_curve.png`: A0 曲线。

重点先看：

```text
a1_results.csv -> cos_x_final_vs_baseline_diff
a1_gain_heatmap.png
```

如果 A1 在中间 sigma、K=3 或 K=5 上出现稳定正峰值，说明短链动态推演相对上一 latent baseline 有净收益。若 A0 有峰值但 A1 没有，说明模型主要受益于“看到了带噪真未来”，不支持因果预测假设。

## 历史帧 + Sigma + 去噪步数组合 Sweep

新增脚本：

- `run_history_sigma_steps_sweep.sh`: 按多个历史帧长度重复运行主评测。
- `aggregate_history_sweep.py`: 合并每个历史帧长度的输出，生成总表和最佳组合报告。

默认 sweep：

```text
cond_frames = 1,5,9,13
sigma_start = 0.002,0.05,0.1,0.2,0.3,0.5,0.8,1.0,2.0,5.0
K = 1,3,5,10,20,35
```

这里 `cond_frames` 是条件像素帧数，包含当前帧。由于 VAE temporal downsample factor 是 4，`1,5,9,13` 分别对应 `1,2,3,4` 个条件 latent slots。

运行：

```bash
cd /data/repos/cosmos_causal_probe
PYTHON_BIN=/data/miniconda3/envs/starVLA/bin/python \
OUTPUT_ROOT=/data/repos/cosmos_causal_probe/outputs/history_sigma_steps_sweep \
./run_history_sigma_steps_sweep.sh
```

可覆盖 sweep 网格：

```bash
PYTHON_BIN=/data/miniconda3/envs/starVLA/bin/python \
COND_FRAMES_LIST=1,5,9,13,17 \
SIGMAS=0.002,0.05,0.1,0.2,0.3,0.5,0.8,1.0,2.0,5.0,10.0 \
DENOISE_STEPS=1,3,5,10,20,35,50 \
OUTPUT_ROOT=/data/repos/cosmos_causal_probe/outputs/history_sigma_steps_sweep_big \
./run_history_sigma_steps_sweep.sh
```

每个历史帧长度会输出到：

```text
${OUTPUT_ROOT}/cond_${cond_frames}_future_${FUTURE_PIXEL_FRAMES}/
```

聚合输出在 `OUTPUT_ROOT` 根目录：

- `combined_a1_results.csv`: 全部 A1 组合。
- `combined_a0_results.csv`: 全部 A0 sanity check。
- `best_a1_by_history.csv`: 每个历史帧长度下的最佳 A1 组合。
- `aggregate_report.txt`: 文本摘要。
- `best_gain_by_history.png`: 历史帧长度 vs 最佳净收益。
- `best_xfinal_vs_baseline_by_history.png`: 最佳 x_final cosine 和 baseline 对比。

## CFG + Negative Prompt + Multi-Slot Sweep

新增入口：

```bash
/data/repos/cosmos_causal_probe/run_cfg_sequence_sweep.sh
```

它会组合以下变量：

```text
condition pixel frames
future latent slots
negative prompt mode
guidance scale
sigma_start
denoise steps K
```

默认保守网格：

```text
cond_frames = 1,5,9
future_latent_slots = 1,2,4
negative_prompt_modes = empty,quality
guidance_scales = 1,3,5,7
sigma_start = 0.5,0.8,1.0,2.0
K = 1,3,5,10,20
```

运行：

```bash
cd /data/repos/cosmos_causal_probe
PYTHON_BIN=/data/miniconda3/envs/starVLA/bin/python \
OUTPUT_ROOT=/data/repos/cosmos_causal_probe/outputs/cfg_sequence_sweep \
./run_cfg_sequence_sweep.sh
```

只跑空 negative prompt 的窄版：

```bash
PYTHON_BIN=/data/miniconda3/envs/starVLA/bin/python \
COND_FRAMES_LIST=1,5 \
FUTURE_LATENT_SLOTS_LIST=1,2 \
NEGATIVE_PROMPT_MODES=empty \
GUIDANCE_SCALES=1,3,5,7 \
SIGMAS=0.8,1.0,2.0 \
DENOISE_STEPS=1,3,5,10 \
OUTPUT_ROOT=/data/repos/cosmos_causal_probe/outputs/cfg_sequence_sweep_small \
./run_cfg_sequence_sweep.sh
```

`negative_prompt_modes=quality` 使用仓库里的：

```bash
/data/repos/cosmos_causal_probe/negative_prompt_quality.txt
```

也可以传自己的 negative prompt 文件路径：

```bash
NEGATIVE_PROMPT_MODES=/path/to/negative_prompt.txt ./run_cfg_sequence_sweep.sh
```

### Future Latent Slots

`future_latent_slots=M` 表示一次把 M 个未来 latent slots 从噪声初始化，并让它们一起进入 DiT self-attention 协同去噪。脚本会自动设置：

```text
future_pixel_frames = M * 4
```

这是因为 Cosmos VAE temporal downsample factor 是 4。输出里会同时记录：

```text
latent_cos_x_final                         # 第一个 future slot
cos_x_final_vs_baseline_diff               # 第一个 future slot 相对 baseline 的净收益
sequence_cos_x_final                       # 全部 M 个 future slots flatten 后的序列指标
sequence_cos_x_final_vs_baseline_diff      # 全序列相对 repeat-last-latent baseline 的净收益
```

### 48GB 显存建议

在 full resolution `1280x704` 下，latent 空间大小是 `88x160`，patch size 是 `1x2x2`，所以每个 latent slot 大约有：

```text
(88 / 2) * (160 / 2) = 3520 tokens
```

总 token 数约等于：

```text
(cond_latent_slots + future_latent_slots) * 3520
```

CFG 会多跑一次 negative branch，主要增加计算时间，峰值显存通常不会严格翻倍，因为这里是顺序跑 conditional/unconditional forward。

保守建议：

```text
48GB full-res 起步：future_latent_slots = 1,2
可以尝试：future_latent_slots = 4
高风险 OOM：future_latent_slots >= 8
```

如果要试 `future_latent_slots=8`，建议先降分辨率，例如 `WIDTH=640 HEIGHT=352`，确认趋势后再回到 full-res。

## Saving Latents For Later Metrics

默认不保存大 tensor。需要保存当前/历史/真实未来/去噪未来 latent 时，加：

```bash
--save-latents
```

单次运行示例：

```bash
PYTHON_BIN=/data/miniconda3/envs/starVLA/bin/python \
./run_eval.sh \
  --cond-frames 4 \
  --future-pixel-frames 28 \
  --future-latent-slots 7 \
  --guidance-scales 1,7 \
  --sigmas 1.0,2.0 \
  --denoise-steps 3,5,10 \
  --save-latents
```

使用 sweep 脚本时，用环境变量打开：

```bash
SAVE_LATENTS=1 SAVE_LATENT_DTYPE=bfloat16 ./run_cfg_sequence_sweep.sh
```

保存位置：

```text
output_dir/latents/reference_latents.pt
output_dir/latents/a1/cfg_<cfg>_sigma_<sigma>_K_<K>_slots_<M>.pt
```

`reference_latents.pt` 包含：

```text
all_latents
history_latents
current_latent
true_future_latents
cond_latent_frames
target_latent_idx
target_latent_end
future_latent_slots
frame_indices
condition_pixel_frame_indices
future_pixel_frame_indices
latent_shape
saved_dtype
```

每个 A1 latent 文件包含：

```text
x_final_future_latents      # K 步 Euler update 后的去噪 future latents
pred_x0_future_latents      # 最后一次 DiT forward 的 clean estimate
initial_noise_future_latents
true_future_latents
current_latent
history_latents
guidance_scale
sigma_start
denoise_steps
sigma_schedule
cond_latent_frames
target_latent_idx
target_latent_end
future_latent_slots
```

同时 `a1_results.csv` 里会新增：

```text
latents_file
reference_latents_file
```

后续计算 delta / velocity / motion-mask 指标时，可以按 CSV 行加载对应的 `latents_file`。

