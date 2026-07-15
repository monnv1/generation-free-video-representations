# World Model Dynamics Probe

这个目录是独立的 probing 工程，用来验证视频生成 world-model backbone 的 DiT latent 是否包含可解码的动态/物理信息。它不修改 `/data/repos/starVLA`，但默认直接复用 starVLA 原生的 world-model wrapper 来抽取 latent。

## 放置路径

- 代码：`/data/repos/world_model_probe`
- 配置：`/data/repos/world_model_probe/configs`
- 启动脚本：`/data/repos/world_model_probe/scripts`
- DOM 数据只读输入：`/data/datasets/DOM`
- latent cache 输出：`/data/datasets/DOM_latents/world_model_probe/{backbone}`
- probe checkpoint 输出：`/data/checkpoints/world_model_probe/{run_name}`

## 核心流程

1. `cache_latents.sh`
   - 通过 `data.adapter` 读取数据样本；adapter 只需要给 backbone 提供历史帧 `frames` 和语义文本 `semantic`。
   - 默认 DOM adapter 读取 DOM chunk-000 的 `observation.images.opst_cam` 视频和 parquet 标签。
   - 通过 `starVLA.model.modules.world_model.get_world_model(config)` 加载并冻结 Cosmos/Wan backbone。
   - 对每个采样帧抽取 DiT hidden tokens。
   - 保存为 `.pt` latent 文件，并生成 `index_train.jsonl` / `index_eval.jsonl`。

2. `train_probe.sh`
   - 不再跑 Cosmos/Wan，只读取 cached latent。
   - 训练轻量 cross-attention probe。
   - 输出 `best.pt`、`last.pt`、metrics 到 `/data/checkpoints/world_model_probe/{run_name}`。

3. `eval_probe.sh`
   - 读取 checkpoint 和 eval latent index。
   - 输出整体 loss/MAE 和按 horizon 拆分的 MAE。

## Probe 架构

输入是 frozen backbone 的 token：

```text
hidden_states[-1] -> [B, N, D]
```

probe 结构：

```text
LayerNorm(D)
Linear(D -> query_dim)
learnable queries [num_queries, query_dim]
repeat num_readout_layers:
  query self-attention
  cross-attention: Q=queries, KV=DiT tokens
  FFN
obj_feat = obj queries mean
arm_feat = arm queries mean
shared = all queries mean
obj heads: [obj_feat, shared] -> obj_pos / obj_vel
arm head:  [arm_feat, shared] -> arm_pos
```

Cosmos/Wan 默认使用 stronger readout：

```yaml
probe:
  query_dim: 768
  num_queries: 64
  num_obj_queries: 32
  num_heads: 12
  num_readout_layers: 4
  readout_ffn_dim: 3072
  head_hidden_dim: 1024
  dropout: 0.05
```

在 Cosmos token dim 为 2048 时，probe 参数量约 44.2M；保存 `best.pt`
或 `last.pt` 时还包含 AdamW optimizer state，单个 checkpoint 预计约
0.5GB。

默认预测目标现在只启用目标物体当前位置：

- `obj_pos`: `observation.environment_state[..., 0:3]`

当前 Cosmos 默认做 current-state obj-pos probe，不预测未来 horizon：

```yaml
targets:
  mode: absolute
  horizons: [0]
  target_keys: [obj_pos]
```

这表示 cached latent 对齐同一采样点 `t` 的 `obj_pos(t)`。输入帧仍由
`clip_frame_indices(frame_index, input_frames, frame_stride)` 构造，最后一帧就是当前帧 `t`。

默认 DOM adapter 在 cache latent 前会复用 `dynamic-vla` 原生的
`utils.instruction_generator.InstructionGenerator`，把 DOM `tasks.jsonl`
里保存的结构化 JSON task metadata 转成自然语言 instruction，再传给
Cosmos/Wan 文本编码器。旧版本地 DOM 导出里 `task` 可能写成
`long_horizon`，DOM adapter 会在调用 dynamic-vla 生成器前兼容成
`long-horizon`。cached `.pt` 的 metadata 会同时保留 raw `task`、
转换后的 `instruction` 和实际传入 backbone 的 `prompt`。
如果从旧的 JSON prompt cache 切换到自然语言 prompt，需要重新生成
latent：运行 cache 脚本时加 `--overwrite`，或者改 `paths.latent_root` /
`backbone.name` 使用新的 cache 目录。

## 数据适配器接口

cache 阶段已经和 DOM 文件结构解耦。backbone 不再知道视频路径、parquet
路径、episode/frame index，也不直接读 Lerobot 数据；它只接收：

```text
frames:   list[PIL.Image.Image]  # 历史帧
semantic: str                    # 传给 backbone 的任务语义文本
```

默认配置使用：

```yaml
data:
  adapter: world_model_probe.data.dom:iter_probe_samples
```

adapter 是一个 Python callable，签名固定为：

```python
def iter_probe_samples(cfg: dict, split: str) -> Iterable[ProbeSample]:
    ...
```

它需要 yield `world_model_probe.data.base.ProbeSample`：

```python
from world_model_probe.data.base import BackboneInput, ProbeSample

yield ProbeSample(
    sample_id="episode_000000_frame_000000",
    backbone_input=BackboneInput(
        frames=history_frames,
        semantic=instruction_text,
        metadata={"episode_index": 0, "frame_index": 0},
    ),
    targets={
        "obj_pos": obj_pos_targets,  # [num_horizons, dim]
        "obj_vel": obj_vel_targets,
        "arm_pos": arm_pos_targets,
    },
    valid=valid_mask,               # [num_horizons]
    metadata=extra_metadata,
)
```

因此换数据集时，不需要改 Cosmos/Wan backbone 或 probe；只要新增一个
adapter，把该数据集自己的视频/图像/标注格式转换成“历史帧 + 语义文本 +
probe targets”。如果继续使用 `targets.mode: delta`，训练阶段仍需要能从
metadata 里的 `parquet_path` 和 `frame_index` 读到当前时刻状态；非 DOM
数据集更建议在 adapter 里直接产出已经需要的 target，并把
`targets.mode` 设为 `absolute`，或者同步扩展 `LatentProbeDataset` 的
delta 逻辑。

当前评测口径是：对一个时间点 `t` 的 cached latent，probe 直接输出当前状态，并和 parquet 里同一行 `t` 构造出的标签对比。

```text
latent(t) -> probe -> prediction(t)
target    -> parquet[t]
```

默认训练损失使用 `training.loss_type: l2`，即当前目标物体位置向量的欧氏距离：

```text
loss = || pred_obj_pos(t) - obj_pos(t) ||_2
```

保留排查的对齐问题：

1. 仿真 state 和渲染帧是否同一时刻。这里不是假设 state 错，而是视频帧 `t` 和环境 state `t` 可能存在 off-by-one、action step/render step 不一致、frame skip 或 chunk 截取偏移；这些都会让 current-state probe 看起来不稳定。
2. 标签坐标系和图像可观测性是否一致。仿真 `obj_pos` 可能是 world frame 或 robot base frame 坐标，但单目 RGB latent 看到的是 image-space appearance；深度、遮挡、相机投影、物体被夹爪挡住时，world position 仍然真实，但 RGB 里不一定能直接读出。
3. TODO: 补充第三个保留检查项。

单独 eval 会额外写出逐样本 CSV，默认路径是：

```text
/data/checkpoints/world_model_probe/{run_name}/eval_{split}_predictions.csv
```

CSV 第一列是递增 `index`，后面只保留目标物体位置的真实值和预测值：

```text
obj_pos_true, obj_pos_pred
```

每个单元格保存一个 JSON 数组。eval 的 `csv/*` 指标会从这张 CSV 重新读回后计算，包括 MAE、RMSE、L2 mean/median/min/max/var。

## 和 starVLA 原生实现的关系

Cosmos/Wan 默认配置使用：

```yaml
backbone:
  adapter: world_model_probe.backbones.starvla_native:StarVLACosmosPredict2BackboneAdapter
```

或：

```yaml
backbone:
  adapter: world_model_probe.backbones.starvla_native:StarVLAWan22BackboneAdapter
```

这个 adapter 直接调用 `/data/repos/starVLA/starVLA/model/modules/world_model/__init__.py` 里的 `get_world_model(config)`。也就是说，它和 `/data/repos/starVLA/starVLA/model/framework/WM4A` 里的 `CosmoPredict2GR00T`、`WanGR00T`、`WanOFT` 等框架类使用同一个底层 world-model wrapper，并且同样走：

```text
backbone.build_inputs(images, instructions)
backbone(**inputs, output_hidden_states=True, return_dict=True)
outputs.hidden_states[-1]
```

没有直接调用 `WM4A` 框架类本身，是因为那些类还包含 starVLA 的动作头、训练/推理包装和任务损失；这里 probing 只需要冻结视频世界模型并缓存 `hidden_states[-1]`。这样能保持 latent 抽取路径和 starVLA 原生路径一致，同时不把 probe 代码耦合进 starVLA 的动作模型。

`world_model_probe.backbones.diffusers_world` 仍保留为备用 adapter，但 Cosmos/Wan 的默认 yaml 不走它。

## 如何启用

先确认只读数据存在：

```bash
ls /data/datasets/DOM/data/chunk-000
ls /data/datasets/DOM/videos/chunk-000/observation.images.opst_cam
```

### 1. Cosmos 版本

配置文件：

```text
/data/repos/world_model_probe/configs/cosmos_probe.yaml
```

执行：

```bash
/data/repos/world_model_probe/scripts/cache_then_train.sh \
  /data/repos/world_model_probe/configs/cosmos_probe.yaml all

/data/repos/world_model_probe/scripts/eval_probe.sh \
  /data/repos/world_model_probe/configs/cosmos_probe.yaml
```

如果想分开执行，也可以分别运行 `cache_latents.sh` 和 `train_probe.sh`。

一键脚本也支持把 cache 参数和 train 参数分开传。`--` 前面的参数传给 `cache_latents.sh`，`--` 后面的参数传给 `train_probe.sh`：

```bash
/data/repos/world_model_probe/scripts/cache_then_train.sh \
  /data/repos/world_model_probe/configs/cosmos_probe.yaml all \
  --overwrite -- \
  --override logging.wandb.mode=offline
```

默认 Cosmos model id 已经指向 starVLA 本地权重：

```yaml
backbone:
  model_id: /data/repos/starVLA/playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World
  local_files_only: true
```

这个目录当前存在，`cache_latents.sh` 也默认设置 HF offline 环境变量，所以不会从 Hugging Face 静默下载。

### 2. Wan2.2 版本

配置文件：

```text
/data/repos/world_model_probe/configs/wan_probe.yaml
```

执行：

```bash
/data/repos/world_model_probe/scripts/cache_latents.sh \
  /data/repos/world_model_probe/configs/wan_probe.yaml all

/data/repos/world_model_probe/scripts/train_probe.sh \
  /data/repos/world_model_probe/configs/wan_probe.yaml

/data/repos/world_model_probe/scripts/eval_probe.sh \
  /data/repos/world_model_probe/configs/wan_probe.yaml
```

默认 Wan model id 也写成本地路径：

```yaml
backbone:
  model_id: /data/repos/starVLA/playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers
  local_files_only: true
```

我检查时这个 Wan 目录还不存在；在下载或放好 Wan 权重前运行 Wan cache 会直接报缺本地目录/文件，不会去 Hugging Face 下载。

### 3. 快速检查脚本链路

`dry_run` 不使用真实 world model，只产生 deterministic fake tokens。它只能检查数据读取、cache、train、eval 是否通，不代表实验结果。

```bash
/data/repos/world_model_probe/scripts/cache_latents.sh \
  /data/repos/world_model_probe/configs/dry_run_probe.yaml all --overwrite

/data/repos/world_model_probe/scripts/train_probe.sh \
  /data/repos/world_model_probe/configs/dry_run_probe.yaml

/data/repos/world_model_probe/scripts/eval_probe.sh \
  /data/repos/world_model_probe/configs/dry_run_probe.yaml
```

## 常用配置项

切 backbone：

```yaml
backbone:
  name: cosmos
  adapter: world_model_probe.backbones.starvla_native:StarVLACosmosPredict2BackboneAdapter
```

或：

```yaml
backbone:
  name: wan
  adapter: world_model_probe.backbones.starvla_native:StarVLAWan22BackboneAdapter
```

改回未来预测步长时可以设置非零 horizon；当前状态 probe 使用：

```yaml
targets:
  mode: absolute
  horizons: [0]
```

改 train/test split：

```yaml
data:
  splits:
    train: [0, 800]
    eval: [800, 1000]
```

当前 Cosmos 默认配置使用 chunk198 的 current-state full-token cache，不再做 `max_tokens=512` 截断：

```yaml
project:
  run_name: cosmos_dom_chunk198_fulltok_s30_current_obj_pos_strongreadout_e125_probe

data:
  chunk_id: 198
  sample_stride: 30

targets:
  mode: absolute
  horizons: [0]
  target_keys: [obj_pos]

backbone:
  name: cosmos_chunk198_current_fulltok_s30
  max_tokens: null
```

这个配置覆盖 chunk198 的 1000 个 episode，只是每个 episode 抽更少时间点。按当前单路 `opst_cam` 的 `320x240` 预处理和 8 帧输入估算，每条 Cosmos latent 约 9.4 MiB，总量约 59GiB / 63GB：

```text
train: 5121 samples
eval:  1326 samples
total: 6447 samples
```

训练默认策略：

```yaml
training:
  batch_size: 32
  eval_batch_size: 64
  epochs: 125
  lr: 1.0e-4
  weight_decay: 1.0e-4
  max_grad_norm: 1.0
  amp: true
  scheduler:
    type: cosine
    warmup_steps: 500
    min_lr_ratio: 0.05
  ema_beta: 0.98
  loss_type: l2
  log_every: 50
  eval_every: 500
```

按当前 `s30` 数据量计算：

```text
steps_per_epoch: 161
total optimizer steps: 20125
```

学习率不是固定值：前 500 step 从接近 0 warmup 到 `1e-4`，之后 cosine decay 到 `5e-6`。WandB 会同时记录 raw mini-batch 指标和 `train/*_ema` 平滑指标；raw train 曲线会比较抖，判断趋势时优先看 `train/loss_ema` 和 `eval/loss`。

checkpoint 保存频率由 `training.eval_every` 控制。当前 `eval_every=500`，所以训练中会在 step 500、1000、1500 ... 20000 左右执行 eval，并保存：

```text
last.pt
best.pt              # 只有 eval loss 变好时更新
best_metrics.json    # 只有 best.pt 更新时更新
```

训练结束后还会额外保存：

```text
last.pt
best.pt              # 如果最终 eval loss 更好
final_metrics.json
```

如果想接近每个 epoch 保存一次，可以覆盖：

```bash
--override training.eval_every=160
```

如果要进一步减少 cache 体积：

```yaml
data:
  sample_stride: 50
  max_samples_per_episode: 20

backbone:
  max_tokens: 256
  cache_dtype: float16
```

改 probe 大小：

```yaml
probe:
  query_dim: 512
  num_queries: 16
  num_obj_queries: 8
  num_heads: 8
```

## WandB

Cosmos/Wan 的默认 yaml 已经打开 wandb online logging：

```yaml
logging:
  wandb:
    enabled: true
    entity: weisb25-tsinghua-university
    project: world_model_probe_DOMINO
    name: "{run_name}"
    group: "{backbone}"
    mode: online
```

`mode: online` 需要先在 `starVLA` 环境登录一次：

```bash
conda activate starVLA
wandb login
```

如果不想上传，直接关掉：

```bash
/data/repos/world_model_probe/scripts/train_probe.sh \
  /data/repos/world_model_probe/configs/cosmos_probe.yaml \
  --override logging.wandb.enabled=false
```

如果想先本地记录、之后再同步：

```bash
/data/repos/world_model_probe/scripts/train_probe.sh \
  /data/repos/world_model_probe/configs/cosmos_probe.yaml \
  --override logging.wandb.mode=offline
```

offline run 会写到 checkpoint 目录下的 `wandb/` 子目录，之后可用 `wandb sync` 上传。

## 输出文件

cache 后：

```text
/data/datasets/DOM_latents/world_model_probe/cosmos_chunk198_current_fulltok_s30/
  index_train.jsonl
  index_eval.jsonl
  meta_train.json
  meta_eval.json
  train/*.pt
  eval/*.pt
```

训练后：

```text
/data/checkpoints/world_model_probe/cosmos_dom_chunk198_fulltok_s30_current_obj_pos_strongreadout_e125_probe/
  config.yaml
  best.pt
  last.pt
  best_metrics.json
  final_metrics.json
  eval_eval.json
  eval_eval_predictions.csv
```

## 注意事项

- `cache_latents.sh` 是唯一会跑 Cosmos/Wan heavy backbone 的阶段。
- `train_probe.sh` 和 `eval_probe.sh` 只读 latent cache，适合反复调 probe 超参。
- 当前实现只使用 `opst_cam`，所以结论对应单路外视角 latent。
- `probe.backbone_dim: auto` 会从 cache 的 token shape 自动推断维度，方便 Cosmos/Wan 一键切换。
- Cosmos 的 `backbone.video_size` 会传给 starVLA 的 `framework.obs_image_size`；当前 starVLA Wan wrapper 里图像尺寸固定为 480x832，所以 Wan 的 `video_size` 配置只作为记录保留。
- 如果 Wan 报 `seq_len exceeds max_seq_len`，优先降低 `data.input_frames`；当前 starVLA Wan wrapper 内部固定预处理尺寸，`backbone.video_size` 不会改变 Wan 的实际尺寸。
- 如果显存不够，Cosmos 先调小 `backbone.video_size`、`data.input_frames`、`backbone.max_tokens`；Wan 先调小 `data.input_frames`、`backbone.max_tokens`。
