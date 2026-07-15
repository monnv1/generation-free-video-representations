# Cosmos 去噪实验回顾

整理日期：2026-07-15

## 一句话结论

这批工作的主线是：先确认 frozen Cosmos 的 raw hidden tokens 是否含有可读的动力学信息，再直接检验从纯噪声 future slots 开始的短链去噪是否能形成更好的未来表征，最后在 DOMINO 上用未来物体 UV probe 做下游验证。

目前最稳妥的结论是：**raw Cosmos 表征里有少量但纠缠、对对象运动很弱的动态信息；测试时加入短链或中间去噪没有显示出稳定收益，在更严格的原生调度实验里反而显著破坏了未来 UV 的可读性。** 因此后续更合理的方向不是测试时生成未来，而是冻结世界模型，用对象级运动监督蒸馏一个面向控制的 dynamics latent。

## 1. Frozen world-model feature probe（2026-05-15 至 05-20）

位置：

- `/data/repos/world_model_probe`
- `/data/checkpoints/world_model_probe`
- `/data/repos/world_model_probe/reports/object_motion_diagnostics`

这条路线没有运行生成去噪链，而是把 DOM 历史帧和 instruction 输入 frozen Cosmos，缓存最后层 full tokens，再训练 cross-attention readout 预测：

- object position delta
- object velocity delta
- robot arm position delta
- horizons `[1, 5, 10, 20]`，对应 25 FPS 下的 0.04、0.2、0.4、0.8 秒

最清楚的 chunk0 strong-readout 诊断（1326 个 eval samples）：

| target | probe L2 | persistence L2 | 改善 |
|---|---:|---:|---:|
| obj_pos | 0.0361 | 0.0363 | 0.72% |
| obj_vel | 0.1030 | 0.1043 | 1.17% |
| arm_pos | 0.0606 | 0.0830 | 27.0% |

关键解释：

- Cosmos token 对机械臂自身运动明显更可读。
- 对象运动几乎只比预测零变化的 persistence baseline 好一点。
- 大运动样本上存在明显的 near-zero prediction：obj_pos 的预测/真实运动范数比只有 0.179，obj_vel 只有 0.083。
- 时间对齐检查没有发现 off-by-one；弱结果不能简单归因于 cache/label 错位。
- 后续实验逐步从 future delta 降级到 current-state、再到 current obj_pos，说明当时正在区分“当前状态可解码”和“未来动态可预测”。最终 current obj-pos run 没有 baseline，不能据此声称有效提升。

主要证据：

- `/data/repos/world_model_probe/reports/object_motion_diagnostics/object_motion_diagnostics_eval.md`
- `/data/checkpoints/world_model_probe/cosmos_dom_chunk0_fulltok_s30_delta_strongreadout_e125_probe`
- `/data/checkpoints/world_model_probe/cosmos_dom_chunk198_fulltok_s30_delta_strongreadout_e125_probe`

## 2. Cosmos causal denoise probe（2026-05-21 至 05-22）

位置：

- `/data/repos/cosmos_causal_probe`
- `/data/exp.md`

目标是直接检验：只给历史帧，把未知 future latent slots 初始化为纯噪声后，Cosmos 的短链去噪结果是否比“复制最后一个历史 latent”更接近真实未来 latent。

实验严格区分：

- A1：纯噪声 future slot，只依赖历史条件，是真正的因果未来测试。
- A0：把真实未来 latent 加噪再恢复，只是实现链路 sanity check，存在真实未来信息，不能证明因果预测。

实现还原了 Cosmos pipeline 的实际 preconditioning：

```text
current_t = sigma / (sigma + 1)
c_in = c_skip = 1 / (sigma + 1)
c_out = -sigma / (sigma + 1)
```

主指标：

```text
gain = cos(x_final, true_future_latent)
     - cos(last_condition_latent, true_future_latent)
```

### 已完成 sweep

1. 默认 sigma × K：最佳 `x_final_cos=0.1642`，baseline `0.8893`，gain `-0.7251`。
2. K 扩大到 35：最佳点不变，没有正收益。
3. history `[1,5,9,13]` × sigma `0.002..5.0` × K `[1,3,5,10,20,35]`：240 个 A1 组合，无一超过 baseline。

历史帧 sweep 的最佳结果：

| cond frames | best x_final cosine | baseline cosine | gain |
|---:|---:|---:|---:|
| 1 | 0.2326 | 0.8280 | -0.5954 |
| 5 | 0.1943 | 0.8893 | -0.6950 |
| 9 | 0.1103 | 0.9386 | -0.8283 |
| 13 | 0.0722 | 0.9647 | -0.8925 |

4. CFG、quality negative prompt、多 future-slot 扩展也实际运行过，不只是写了脚本：

- CFG=7 + quality negative prompt 的总体最好结果为 `x_final=0.4182`、gain `-0.4099`。有改善，但仍远低于 baseline。
- 10 future slots 协同去噪：first-slot gain `-0.7368`，sequence gain `-0.7815`。
- 8-slot save-latents 实验：first-slot gain `-0.7088`，sequence gain `-0.8050`。

A0 在低 sigma 下接近完美（cosine 约 1.0），说明代码能从“带噪真未来”恢复目标。A1 持续失败说明问题不是简单的 VAE、hook 或 preconditioning 完全失效，而是这种短链生成没有把历史条件转化成超过 persistence 的未知未来。

限制：这部分核心实验只用了 `robot_pouring.mp4`、单 seed 和 latent cosine，不能外推为“Cosmos 普遍没有生成能力”。准确说法应是：**在当前单视频、当前初始化和指标下，没有观察到因果短链优于复制上一 latent。**

主要证据：

- `/data/repos/cosmos_causal_probe/outputs/history_sigma_steps_sweep/aggregate_report.txt`
- `/data/repos/cosmos_causal_probe/outputs/cfg_sequence_sweep_small/aggregate_report.txt`
- `/data/repos/cosmos_causal_probe/outputs/save_latents_test/aggregate_report.txt`

## 3. DOMINO + Cosmos denoise probe（2026-05-25 至 05-27）

位置：`/data/repos/domino_cosmos_denoise_probe`

统一设计：

- frozen Cosmos-Predict2-2B，不微调、不训练 action head、不做 closed-loop control。
- 输入只有 5 帧历史；future pixels 不输入 Cosmos，只作为 probe labels。
- 每轮 3600 slices，按 episode 做 70/15/15 split，每任务最多 1200 slices。
- 比较 raw/no-denoise hidden 与不同去噪状态、不同 transformer layer 的 hidden readout。

### 3.1 初版 multi-target（2026-05-25）

运行：`domino_cosmos_denoise_probe_20260525_144759`

- tasks：grab_roller、move_playingcard_away、click_bell
- horizons：`[1,2,4,8,15]`
- sources：raw + 8 个 tau
- layers：`[6,14,27]`
- targets：XYZ、UV、depth、velocity、contact、success、time-to-contact

结果：24/24 个 denoise future 组合的综合 loss 都比同层 raw baseline 差。最好的 denoise 仍然是负收益：layer 14、tau 0.2，loss 0.8846，较 raw 差 18.49%。不过它的 XYZ RMSE 为 0.1089 m，比 raw 好 7.42%，说明个别子指标存在局部改善，但不足以支持总体假设。

### 3.2 第一版 UV-only（2026-05-26）

运行：`domino_cosmos_uv_probe_20260526_005100`

同一批短 horizon 数据，只训练 masked UV probe。旧 tau grid 中 13/24 个 denoise 组合的 future UV MSE 优于 raw。最佳 layer 27、tau 0.6：

- valid-weighted MSE：0.02894，较 raw 改善 14.03%
- valid-weighted MAE：38.32 px，较 raw 改善 4.61%

这是唯一出现较明确正信号的批次，但只能解释为“旧实现、短 horizon、特定 tau/layer 的局部 UV readout 增益”，不能视为普遍结论。

### 3.3 Native scheduler + per-future-step UV（2026-05-26 至 05-27）

实验升级为：

- tasks：adjust_bottle、move_playingcard_away、click_bell
- horizons：`[8,16,24,36,45]`，约 0.53 至 3 秒
- 35-step native scheduler
- layers：`[8,12,16,20,24]`
- 每个 horizon 映射到对应 future latent slot，并单独训练 probe
- early capture steps：`[1,3,5,7]`
- late capture steps：`[10,14,18,22,26]`

结果非常一致：

- early run：100/100 个 denoise layer/horizon 组合都输给 raw。
- late run：125/125 个 denoise layer/horizon 组合都输给 raw。

跨 early/late 两轮合并看，每个 horizon 的最佳 raw MAE 与最佳 denoise MAE：

| horizon | best raw | best denoise | denoise 更差 |
|---:|---:|---:|---:|
| 8 | 21.35 px | 39.60 px | 85.5% |
| 16 | 19.62 px | 37.68 px | 92.0% |
| 24 | 17.07 px | 36.45 px | 113.6% |
| 36 | 14.25 px | 29.88 px | 109.7% |
| 45 | 13.64 px | 27.38 px | 100.7% |

注意：MAE 随 horizon 变小不是远期预测更容易，而是 UV valid mask 越来越稀，只剩较容易且仍可见的样本。

## 4. 需要保留的实验限制

1. `domino_cosmos_denoise_probe` 和 `cosmos_causal_probe` 都不是 Git 仓库，只能靠 mtime、保存配置和结果还原历史。
2. DOMINO probe 当前代码已经从 tau API 改成 native capture steps；`configs/default.yaml` 仍保留旧 `tau_values`，但当前 extractor 不读取它。旧实验无法从当前源码严格复现。
3. UV evaluator 存在聚合 bug：batch 内按 valid points 平均后，跨 batch 却按 batch rows 加权。第一版 UV-only 有单独的 valid-weighted 重算文件；两轮 native 没有，因此绝对 MAE 数值有偏差风险。raw 全面优于 denoise 的方向很强，但正式报告前仍应修复后重算。
4. 新三任务数据中 `click_bell` 的 UV valid mask 在所有 split/horizon 都是 0，因此 UV 结论实际只来自 adjust_bottle 和 move_playingcard_away。
5. horizon 45 的 test valid 只有 109/483，长 horizon 统计较稀。
6. summarizer 直接在 test set 上从大量组合中挑 best，没有多 seed、置信区间或独立 final test，所有 best 数值应视作探索性结果。
7. causal probe 只覆盖单个 robot_pouring 视频和单 seed。

## 5. 研究脉络与后续方向

这批实验支持的研究叙事不是“Cosmos 完全没有动态信息”，而是：

1. clean/raw Cosmos tokens 含有少量可监督读出的状态和动力学信息，尤其机械臂自身运动。
2. 对象未来运动信号弱、纠缠且容易退化为近零预测。
3. 从纯噪声 future slot 做短链去噪，没有超过强 persistence baseline。
4. 在更严格的 DOMINO native-step UV probe 中，denoise hidden 的可读性全面差于 raw hidden。
5. 因此更有希望的路线是保留 frozen backbone，通过 object pose、velocity、future pose、relative pose、time-to-contact 等监督蒸馏 control-oriented dynamics latent，并在策略训练中验证它是否带来实际动作收益；推理时不生成未来。

该后续设计已整理在：

- `/data/wm_dynamic_latent_pipeline.html`
- `/data/cosmos2_architecture.html`

## 6. 产物索引

- 总实验图归档：`/data/experiment_figures/README.md`
- 归档规模：549 个文件、43 MB
- causal probe：70 张 PNG，包括 gain heatmap、A0 curve、decode 对比和 latent spatial heatmap
- DOMINO denoise probe：raw/denoise MAE 示例、两个交互式 layer × horizon × step HTML
- 诊断标签预览：417 张 GroundingDINO、SAM2 和 hybrid probe 图片

最值得先看的文件：

1. `/data/exp.md`
2. `/data/repos/world_model_probe/reports/object_motion_diagnostics/object_motion_diagnostics_eval.md`
3. `/data/repos/cosmos_causal_probe/outputs/history_sigma_steps_sweep/aggregate_report.txt`
4. `/data/repos/cosmos_causal_probe/outputs/cfg_sequence_sweep_small/aggregate_report.txt`
5. `/data/repos/domino_cosmos_denoise_probe/results/domino_cosmos_uv_probe_3s_native_stop26_l8_24_s10_26/denoise_layer_horizon_3d.html`
6. `/data/experiment_figures/README.md`
