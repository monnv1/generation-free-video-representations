# Object Motion Probe Diagnostics

- Checkpoint: `/data/checkpoints/world_model_probe/cosmos_dom_chunk0_fulltok_s30_delta_strongreadout_e125_probe/best.pt`
- Split: `eval`
- Samples: `1326`
- Target mode: `delta`
- Horizons: `[1, 5, 10, 20]`

## Overall Delta Prediction

| target | probe L2 | persistence L2 | improve % | pred norm | gt norm | pred/gt norm |
| --- | --- | --- | --- | --- | --- | --- |
| obj_pos | 0.0361 | 0.0363 | 0.7205 | 0.0103 | 0.0363 | 0.2843 |
| obj_vel | 0.1030 | 0.1043 | 1.1662 | 0.0112 | 0.1043 | 0.1074 |
| arm_pos | 0.0606 | 0.0830 | 27.0157 | 0.0554 | 0.0830 | 0.6672 |

## Motion Bucket Evaluation

Buckets are computed over sample-horizon pairs. Persistence L2 is the error of predicting zero delta.

### Bucketed by ||Delta obj_pos||, evaluating obj_pos

Buckets: near-static <= `0.0100`, large >= `0.1699`.

| bucket | count | bucket motion norm | gt norm | pred norm | pred/gt | probe L2 | persist L2 | L2/persist | zero-like pred rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| near-static | 2168.0000 | 0.0038 | 0.0038 | 0.0040 | 1.0488 | 0.0064 | 0.0038 | 1.6808 | 0.0139 |
| medium | 2822.0000 | 0.0379 | 0.0379 | 0.0114 | 0.3013 | 0.0378 | 0.0379 | 0.9975 | 0.2041 |
| large | 314.0000 | 0.2461 | 0.2461 | 0.0440 | 0.1786 | 0.2246 | 0.2461 | 0.9123 | 0.4522 |

### Bucketed by ||Delta obj_vel||, evaluating obj_vel

Buckets: near-static <= `0.0100`, large >= `0.6157`.

| bucket | count | bucket motion norm | gt norm | pred norm | pred/gt | probe L2 | persist L2 | L2/persist | zero-like pred rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| near-static | 2844.0000 | 0.0031 | 0.0031 | 0.0053 | 1.7382 | 0.0074 | 0.0031 | 2.4290 | 0.0970 |
| medium | 2214.0000 | 0.1372 | 0.1372 | 0.0109 | 0.0798 | 0.1349 | 0.1372 | 0.9831 | 0.6518 |
| large | 246.0000 | 0.9779 | 0.9779 | 0.0814 | 0.0832 | 0.9220 | 0.9779 | 0.9429 | 0.8252 |

## Object Motion Blindness Check

- `obj_pos` large bucket: pred/gt norm `0.179`, L2/persistence `0.912`, zero-like rate `0.452` -> strong evidence of near-zero prediction on large-motion samples.
- `obj_vel` large bucket: pred/gt norm `0.083`, L2/persistence `0.943`, zero-like rate `0.825` -> strong evidence of near-zero prediction on large-motion samples.

Largest object-position-motion examples:

| sample_id | episode | frame | horizon | gt obj_pos norm | pred obj_pos norm | L2 | pred/gt |
| --- | --- | --- | --- | --- | --- | --- | --- |
| episode_000927_frame_000180 | 927.0000 | 180.0000 | 20.0000 | 0.6025 | 0.0042 | 0.6018 | 0.0070 |
| episode_000806_frame_000180 | 806.0000 | 180.0000 | 20.0000 | 0.5323 | 0.1050 | 0.4483 | 0.1973 |
| episode_000989_frame_000180 | 989.0000 | 180.0000 | 20.0000 | 0.4894 | 0.1458 | 0.4992 | 0.2978 |
| episode_000924_frame_000180 | 924.0000 | 180.0000 | 20.0000 | 0.4842 | 0.0161 | 0.4770 | 0.0332 |
| episode_000991_frame_000180 | 991.0000 | 180.0000 | 20.0000 | 0.4626 | 0.0767 | 0.4452 | 0.1658 |

## R2, Direction Cosine, Sign Accuracy

| target | horizon | count | R2 | cos mean | cos median | sign acc |
| --- | --- | --- | --- | --- | --- | --- |
| obj_pos | all | 5304.0000 | 0.1126 | 0.0933 | 0.0910 | 0.5427 |
| obj_pos | 1.0000 | 1326.0000 | -0.0265 | -0.0533 | -0.0055 | 0.4893 |
| obj_pos | 5.0000 | 1326.0000 | 0.0938 | 0.0897 | 0.0927 | 0.5200 |
| obj_pos | 10.0000 | 1326.0000 | 0.1161 | 0.1676 | 0.1787 | 0.5734 |
| obj_pos | 20.0000 | 1326.0000 | 0.1143 | 0.1695 | 0.1751 | 0.5859 |
| obj_vel | all | 5304.0000 | 0.0643 | 0.0858 | 0.0843 | 0.5336 |
| obj_vel | 1.0000 | 1326.0000 | -0.0017 | 0.0319 | 0.0187 | 0.5432 |
| obj_vel | 5.0000 | 1326.0000 | 0.0139 | 0.1138 | 0.1686 | 0.5344 |
| obj_vel | 10.0000 | 1326.0000 | 0.0390 | 0.1130 | 0.1523 | 0.5469 |
| obj_vel | 20.0000 | 1326.0000 | 0.1178 | 0.0844 | 0.1239 | 0.5107 |
| arm_pos | all | 5304.0000 | 0.3504 | 0.6265 | 0.8390 | 0.7203 |

## Persistence Baseline Strength

The table reports true object delta norm distributions. A high near-static fraction means persistence is intrinsically strong.

| target | horizon | count | mean | median | p75 | p90 | p95 | p99 | near-static frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| obj_pos | 1.0000 | 1326.0000 | 0.0055 | 0.0016 | 0.0075 | 0.0157 | 0.0218 | 0.0517 | 0.8039 |
| obj_pos | 5.0000 | 1326.0000 | 0.0237 | 0.0082 | 0.0149 | 0.0796 | 0.1098 | 0.1852 | 0.6388 |
| obj_pos | 10.0000 | 1326.0000 | 0.0422 | 0.0164 | 0.0301 | 0.1423 | 0.1778 | 0.2680 | 0.1252 |
| obj_pos | 20.0000 | 1326.0000 | 0.0738 | 0.0324 | 0.0720 | 0.2237 | 0.2713 | 0.3780 | 0.0671 |
| obj_vel | 1.0000 | 1326.0000 | 0.0417 | 0.0008 | 0.0226 | 0.0888 | 0.3890 | 0.3924 | 0.6169 |
| obj_vel | 5.0000 | 1326.0000 | 0.0963 | 0.0045 | 0.0360 | 0.2721 | 0.5731 | 1.2765 | 0.5400 |
| obj_vel | 10.0000 | 1326.0000 | 0.1196 | 0.0080 | 0.0608 | 0.4099 | 0.6243 | 1.2415 | 0.5535 |
| obj_vel | 20.0000 | 1326.0000 | 0.1595 | 0.0185 | 0.1418 | 0.5501 | 0.8315 | 1.3216 | 0.4344 |

## Time Alignment Check

- `input_frames=8`, `frame_stride=1`: cache uses contiguous history ending at current frame `t`.
- History span: `0.280s`; sample stride: `1.200s`.
- FPS: `25`; horizons `[1, 5, 10, 20]` correspond to `[0.04, 0.2, 0.4, 0.8]` seconds after `t`.
- Adapter code constructs `frame_indices = clip_frame_indices(frame_index, input_frames, frame_stride)`, so latent current time is the last historical frame `t`, not the sparse sample index.
- Targets are built from parquet rows `t + horizon`; no off-by-one was found in the cache/index metadata check below.

| sample_id | t | history frames | target frames |
| --- | --- | --- | --- |
| episode_000800_frame_000000 | 0.000 | [0, 0, 0, 0, 0, 0, 0, 0] | [1, 5, 10, 20] |
| episode_000800_frame_000030 | 30.000 | [23, 24, 25, 26, 27, 28, 29, 30] | [31, 35, 40, 50] |
| episode_000800_frame_000060 | 60.000 | [53, 54, 55, 56, 57, 58, 59, 60] | [61, 65, 70, 80] |
