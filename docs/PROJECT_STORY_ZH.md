# 项目叙事与简历表述

## 项目定位

**视频扩散模型的生成式表征复用与低延迟多模态推理**

这个项目研究的不是如何更快地生成一段完整视频，而是：视频扩散模型已经
学习到的空间和时序表征，能否绕过完整像素生成，直接用于下游决策；如果
模型计算仍然较慢，能否通过异步 action chunk 将其移出控制关键路径。

## 技术主线

```text
RGB history + language + robot state
                  |
       pretrained Cosmos-Predict2
                  |
    VAE latent / Transformer hidden
                  |
       lightweight policy adapter
                  |
    async action chunk + RTC handoff
```

1. **表征诊断**：计算相邻 latent slot 的通道 L2 差分，将高变化位置映射回
   图像，检查 latent 是否保留空间拓扑和局部运动敏感性。
2. **生成路径消融**：比较 clean/raw representation、短链去噪和完整原生
   scheduler 的 hidden-state 可读性，并以 persistence latent 作为强基线。
3. **闭环系统验证**：将 Cosmos 表征接入动作模型，使用异步请求、action
   chunk 缓冲和 RTC overlap 隐藏推理等待。

## 当前最可靠证据

- 单个已保存 frame pair 上，latent delta 与 RGB change 的 Spearman 为
  `0.350`；top-5% spatial cells 重合率 `64.3%`，随机期望为 `5%`。这是
  single-pair pilot，不证明 VAE 线性、因果预测或语义解耦。
- DOMINO `adjust_bottle` 单任务配置完成 100 episodes，成功率 `23/100`，
  Manipulation Score 和 Route Completion 均为 `30.86`。
- clean-latent + async/RTC 的模型推理 p50 为 `92.2 ms`、单步延迟 p95 为
  `124.7 ms`；同步三步 CFG 去噪配置分别为 `447.7 ms` 和 `764.8 ms`。
  两个延迟配置同时改变了异步模式和去噪方式，只能作为系统级对比。
- 因果 probe 的 240 个 history/sigma/K 组合全部未超过 repeat-last latent；
  更严格的 native scheduler UV probe 中，225 个 denoise layer/horizon 组合
  全部弱于 raw representation。

## 简历 Bullet

- 针对视频扩散模型多步生成难以进入实时决策链路的问题，探索无需显式
  生成视频的 Cosmos-Predict2 表征复用方案，将时序 latent 接入动作模型，
  并实现异步 action chunk 与 RTC handoff。
- 设计 latent motion locality probe，通过相邻 VAE latent 的通道聚合差分和
  pixel-space 回映射诊断运动敏感性；single-pair pilot 中 Spearman 为
  `0.350`，top-5% 变化区域重合率 `64.3%`（随机期望 `5%`）。
- 在 DOMINO `adjust_bottle` 单任务上完成 100-episode 闭环评测，取得
  `23%` 成功率；clean-latent + async/RTC 的模型推理 p50 为 `92.2 ms`，
  相比同步三步 CFG 去噪的 `447.7 ms` 降低 `79.4%`，并结合数百组 probe
  观察到 short-chain denoise 未稳定提升下游表征可读性。

在扩展 locality probe 到多视频并报告均值和方差前，第二条必须保留
“single-pair pilot”，不能暗示普遍性。

## 60 秒回答

完整视频世界模型通常需要多步去噪，生成视频的延迟和显存开销很难进入
高频决策链路。我的问题是，预训练视频扩散模型学到的时序表征能否不经过
完整视频生成，直接服务下游模型。

我先做了 representation probe：计算相邻帧 VAE latent 的时间差分并映射回
图像，定性和单样本定量结果都显示高变化 latent 区域与主要视觉变化区域
存在空间对应。这不能证明 latent 是线性的，但说明它保留了一定的空间结构
和动态敏感性。随后我把 Cosmos 表征接入动作模型，并用异步 action chunk
和 RTC 解耦模型计算与动作执行。在 DOMINO adjust_bottle 的单任务 100 次
评测中，最佳配置成功 23 次；clean-latent + async/RTC 的模型推理 p50 为
92.2 ms，而同步三步 CFG 去噪配置为 447.7 ms。结合数百组离线 probe，我的
判断是生成质量不等于决策表征质量，直接复用 clean latent 并隐藏推理等待，
比盲目截断去噪链更值得继续研究。

## 不能越界的表述

- 说“空间对应和运动敏感性”，不说“证明 VAE 线性”。
- 说“系统级 latency hiding”，不说“降低了模型 FLOPs”。
- 说“单任务 pilot”，不说“方法已泛化”。
- 说“当前三步配置没有收益”，不说“去噪普遍无效”。
- 说“单任务 23/100，高于部分公开 VLA 基线”，不说“达到 SOTA”；公开
  DOMINO 表中 PUMA 和 OpenVLA-OFT 的同任务结果更高。

## 最小补实验

1. 扩展 latent locality 到 10 至 20 个 frame pairs，加入时间错配和 spatial
   shuffle，对 Spearman/top-k overlap 报告均值、标准差。
2. 固定同一 checkpoint 和 episode，只切换 sync/async，各跑 3 至 5 个
   episodes；报告 control-loop p50/p95、blocking ratio 和 stale steps，不用
   小样本成功率声称策略提升。

这两项都不需要训练，也不需要运行完整 35-step 视频生成。
