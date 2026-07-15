# 实验结果图汇总

> 整理时间: 2026-05-31
> 来源: `/data/repos/` 下四个实验仓库 + `/data/datasets/` 诊断探针预览 + `/data/` 下额外文件
> 总计: **549 个文件**, **43 MB**

---

## 扫描范围

| 来源 | 路径 | 是否有实验结果图 | 数量 |
|------|------|:----------------:|------|
| cosmos_causal_probe | `/data/repos/cosmos_causal_probe` | 是 | 70 PNG |
| DOMINO | `/data/repos/DOMINO` | 否 (仅仿真资产+mp4) | 0 |
| domino_cosmos_denoise_probe | `/data/repos/domino_cosmos_denoise_probe` | 是 | 21 PNG + 2 HTML |
| world_model_probe | `/data/repos/world_model_probe` | 否 (纯源码,无预生成图) | 0 |
| 诊断探针预览 | `/data/datasets/DOMINO/diagnostic_probe_labels/` | 是 | 416 JPG/PNG |
| 额外发现 (仓库外) | `/data/video_probe_adjust_bottle/`, `/data/*.html` | 是 | 36 JPG + 2 HTML |
| DOMINO_new | `/data/datasets/DOMINO_new/` | 是 | 1 PNG |

---

## 1. cosmos_causal_probe — 因果探针实验 (70张)

研究 Cosmos 视频世界模型中 latent slot 的因果干预效应。

### 实验条件

| 目录 | 实验说明 |
|------|----------|
| `decode_nocfg/` | 无CFG解码, sanity曲线+gain热力图+最终帧质量曲线 |
| `cfg_test/` | CFG测试, 同上三件套 |
| `robot_pouring/` | 机器人倒水任务 |
| `robot_pouring_steps_sweep/` | 机器人倒水 — 步数扫描 |
| `history_sigma_steps_sweep/` | 历史帧数+噪声sigma扫描 — 4组条件 (cond_1/5/9/13, future_4) |
| `cfg_sequence_sweep_small/` | CFG序列扫描 — 4组slot配置 (1_empty/1_quality/7_empty/8_10) |
| `save_latents_test/` | **最丰富**: 解码帧对比、潜空间热力图、slot差分分析、spatial delta |

### 关键图类型

- **`a0_sanity_curve.png`** — 探针校准曲线 (验证探针是否学到有效信息)
- **`a1_gain_heatmap.png`** — 因果干预增益热力图 (slot × step 的干预效果)
- **`a1_x_final_curves.png`** — 最终帧预测质量 vs baseline
- **`decoded_*.png`** — VAE解码视频帧: 当前帧/预测未来/真实未来/差异图
- **`latent_heatmap_*.png`** — 潜空间激活热力图叠加 (top 1%/5%/10% 激活)
- **`heatmap_slot*_minus_slot*.png`** — Slot间差分热力图
- **`best_gain_by_history.png`** — 不同历史帧数下的最佳干预增益
- **`*_delta_spatial.png`** — 预测与真实帧的空间像素差

---

## 2. domino_cosmos_denoise_probe — DOMINO+Cosmos联合去噪探针 (23个)

研究对 Cosmos latent 进行 denoise 后对 DOMINO 机器人操控任务的影响。

### 实验: `domino_cosmos_uv_probe_3s_native_stop26_l8_24_s10_26`

**顶层图:**
- `contact_frame_examples_three_tasks.png` — 三任务接触帧示例总览
- `baseline_raw_mae_future_uv_example.png` — Baseline 原始 MAE UV 示例
- `PixPin_2026-05-27_14-00-40.png` — 实验截图 (6MB)

**交互式3D图 (浏览器打开):**
- `denoise_layer_horizon_3d.html` — 去噪层 × 时间视野 3D 可视化
- `denoise_layer_horizon_3d.before_short_steps.html` — 同上 (调参前版本)

**Baseline MAE 帧示例 (9张):** `baseline_mae_examples/`
- 未去噪时三个任务的预测帧 vs 真实帧 MAE 对比
- 任务: adjust_bottle (3张) / move_playingcard_away (3张) / click_bell (3张)

**Denoise 后 MAE 帧示例 (9张):** `denoise_step018_layer8_mae_examples/`
- step=18, layer=8 去噪后的同任务对比
- 与 baseline 一一对应, 可直接对比去噪前后的 MAE 变化

### 命名解析

例: `baseline_raw_mae_example_01_adjust_bottle_ep000036_t005.png`
- `baseline` / `denoise_step018_layer8` — 实验条件
- `adjust_bottle` — 任务名
- `ep000036_t005` — episode 36, timestep 5

---

## 3. diagnostic_probe_previews — 诊断探针可视化 (417张)

来源: `/data/datasets/DOMINO/diagnostic_probe_labels/` + `/data/datasets/DOMINO_new/`

DOMINO 数据集的诊断探针标注预览图, 包括多种探针方法和大量任务的帧级可视化。

### 探针方法

| 目录 | 探针方法 | 任务数 | 预览数 |
|------|----------|:------:|:------:|
| `grounding_dino_tiny_smoke_h1_2_4_8_15/` | GroundingDINO-tiny (smoke) | 32 | 193 |
| `grounding_dino_tiny_h1_2_4_8_15/` | GroundingDINO-tiny (h1_2_4_8_15) | 18 | 113 |
| `grounding_dino_tiny_primary_fix_smoke/` | GroundingDINO-tiny (primary fix) | 1 | 6 |
| `sam2_hiera_tiny_red_block_preview/` | SAM2 Hiera-tiny | 1 | 8 |
| `hybrid_dino_sam2_hammer_preview_ep0/` | Hybrid DINO+SAM2 | 1 | 8 |
| `hybrid_dino_sam2_hammer_preview_ep0_ep1/` | Hybrid DINO+SAM2 (ep0+ep1) | 1 | 16 |
| `hybrid_single_target_red_block_hammer_preview_ep0_ep1/` | Hybrid single-target | 1 | 16 |
| `sam2_test_previews/` | SAM2 test | 1 | 21 |
| `sam2_test_previews_auto/` | SAM2 test auto | 1 | 6 |
| `sam2_test_previews_auto_all/` | SAM2 test auto all | 1 | 5 |
| `DOMINO_new/` | UV overlay (first frame) | 1 | 1 |

### 覆盖的任务 (代表性)
`beat_block_hammer`, `click_bell`, `click_alarmclock`, `move_playingcard_away`, `move_can_pot`, `move_stapler_pad`, `handover_block`, `place_can_basket`, `place_fan`, `shake_bottle`, `rotate_qrcode`, `stamp_seal`, `dump_bin_bigbin`, `grab_roller`, `hanging_mug`, `place_a2b_left/right`, `place_bread_basket/skillet`, `place_phone_stand`, `press_stapler`, `scan_object` 等 30+ 个 DOMINO 操控任务

### 文件类型
- **`preview_contact_sheet.jpg`** — 任务预览 contact sheet 总览
- **`episode_000000_frame_*.jpg`** — 单个 episode 的关键帧截图 (每episode 5-21帧)
- **`first_frame_uv_overlay.png`** — 首帧 UV 坐标叠加可视化

---

## 4. 额外发现 — 仓库外的图 (38个)

### video_probe_adjust_bottle (36张 JPG)
来源: `/data/video_probe_adjust_bottle/`
- `contact_success_vs_oob.jpg` — 接触成功率 vs OOB 图表
- `ep01~ep27_0~4.jpg` — 7个episode的帧序列 (每episode 5帧, 共35张)

### 架构/流程图 (2个 HTML)
- `cosmos2_architecture.html` — Cosmos2 模型架构图
- `wm_dynamic_latent_pipeline.html` — 世界模型动态潜变量流水线图

---

## 5. 无图的仓库

### DOMINO
`/data/repos/DOMINO/` 内含 12,941 个图像文件，但全部是仿真资产:
- `assets/background_texture/` — 11,000张背景纹理
- `assets/objects/` — 1,885张物体纹理/部件渲染
- `assets/embodiments/` — 机器人mesh贴图
- `eval_result/` — 只有 `.mp4` 视频和 `.json`/`.txt` 指标, 无PNG图表
- `assets/static/` — demo用GIF动图

### world_model_probe
`/data/repos/world_model_probe/` 是纯源码仓库。`reports/` 目录只有 `.jsonl`/`.npz`/`.json` 数据。图需要在运行时由 `world_model_probe/diagnostics/report.py` 生成。

本次扫描还检查了 `/data/checkpoints/world_model_probe/` 下的 wandb run 目录 — 其 `tmp/` 和 `files/` 子目录均为空或仅含日志, 无可视化图片。

---

## 目录结构总览

```
experiment_figures/
├── README.md                          # 本文件
├── cosmos_causal_probe/               # 70 PNG — 因果探针实验
│   ├── decode_nocfg/                  # 无CFG解码 (3图 + frames/)
│   ├── cfg_test/                      # CFG测试 (3图)
│   ├── robot_pouring/                 # 倒水任务 (3图)
│   ├── robot_pouring_steps_sweep/     # 步数扫描 (3图)
│   ├── history_sigma_steps_sweep/     # 历史+噪声扫描 (14图, 含4组条件)
│   ├── cfg_sequence_sweep_small/      # CFG序列扫描 (14图, 含4组条件)
│   └── save_latents_test/            # 潜空间可视化 (28图, 最丰富)
├── domino_cosmos_denoise_probe/       # 23个 — DOMINO去噪探针
│   ├── contact_frame_examples_three_tasks.png
│   └── domino_cosmos_uv_probe_3s_native_stop26_l8_24_s10_26/
│       ├── baseline_mae_examples/     # Baseline MAE (9图)
│       ├── denoise_step018_layer8_mae_examples/  # 去噪后MAE (9图)
│       ├── denoise_layer_horizon_3d.html          # 交互式3D图
│       └── ...                        # 概要图+截图
├── diagnostic_probe_previews/         # 417个 — 诊断探针标注预览
│   ├── grounding_dino_tiny_smoke_h1_2_4_8_15/    # 193张, 32任务
│   ├── grounding_dino_tiny_h1_2_4_8_15/          # 113张, 18任务
│   ├── grounding_dino_tiny_primary_fix_smoke/     # 6张
│   ├── sam2_hiera_tiny_red_block_preview/         # 8张 (含contact sheet)
│   ├── hybrid_dino_sam2_hammer_preview_ep0*/      # 8+16张
│   ├── hybrid_single_target_red_block_hammer_preview_ep0_ep1/  # 16张
│   ├── sam2_test_previews*/                       # 21+6+5张
│   └── first_frame_uv_overlay.png
└── external/                          # 38个 — 仓库外发现
    ├── cosmos2_architecture.html
    ├── wm_dynamic_latent_pipeline.html
    └── video_probe_adjust_bottle/     # 36张 — adjust_bottle视频探针
```
