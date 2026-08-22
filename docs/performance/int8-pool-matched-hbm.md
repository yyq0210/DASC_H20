---
title: "独立 INT8 checkpoint pool matched-HBM 实验"
description: "说明旧独立 INT8 pool 的 matched-HBM serving 配置、脚本和解释边界。"
---

# 独立 INT8 checkpoint pool matched-HBM 实验

该实验使用 SGLang 的独立 INT8 mamba checkpoint pool。它测量 INT8 payload 与该 pool 的 TP-split layout 共同产生的容量和性能，不是纯粹的位宽消融。

## 解释边界

独立 INT8 pool 每个 rank 存 16 heads；dense/ragged head-aware pool 每个 rank 存 32 heads。除约 1.97× 量化压缩外，独立 pool 还获得约 2× layout 优势。因此：

- 可以报告实际 BPS、HBM、slot、hit、TTFT 和吞吐。
- 不能把相对 dense 的全部提升解释为 INT8 量化本身。
- 与 DASC 做同布局公平比较时，应使用 [公平 INT8 实验](int8fair-matched-hbm.md)。

## 固定配置

| 项目 | 值 |
|---|---|
| 模型 | `Kimi-Linear-48B-A3B-Instruct` |
| GPU / TP | 2 × H20，TP=2 |
| Workload | ShareGPT V3 N40，184 requests |
| Budgets | S=48/96/144 dense-equivalent slots |
| Seeds | 42, 123, 7 |
| Dense reference BPS | 20,971,520 |
| INT8 offline BPS estimate | 5,324,800 |
| 并发 / 输出 | 16 / 64 token |
| Prefill graph | `tc_piecewise` |

## 脚本

| 路径 | 作用 |
|---|---|
| `experiments/scripts/run_equal_hbm_int8.sh` | live calibration、matched-HBM slot 计算和多预算编排 |
| `experiments/scripts/bench_arena_shared_prefix.sh` | `int8` 模式下启用独立 INT8 pool |
| `experiments/scripts/plot_equal_hbm.py` | 与 dense/padded/ragged 结果一起聚合 |

`bench_arena_shared_prefix.sh` 在 `int8` 模式下使用：

```text
--enable-int8-mamba-checkpoint
--int8-mamba-ckpt-size <slots>
```

## Dry-run

```bash
S_LIST="48 96 144" \
  bash experiments/scripts/run_equal_hbm_int8.sh --dry-run
```

dry-run 使用离线 BPS。正式运行会先以 48 slots 启动一次 calibration，并用结果替换离线值。

## 最小复现

```bash
NGROUPS=40 \
S_LIST=96 \
SEEDS=42,123,7 \
BACKEND_FLAGS="--linear-attn-backend triton --cuda-graph-backend-prefill tc_piecewise" \
LOGDIR="$PWD/experiments/results/int8_pool_s96" \
  bash experiments/scripts/run_equal_hbm_int8.sh
```

## 完整复现

```bash
NGROUPS=40 \
S_LIST="48 96 144" \
SEEDS=42,123,7 \
MAX_CONC=16 \
OUTPUT_LEN=64 \
BACKEND_FLAGS="--linear-attn-backend triton --cuda-graph-backend-prefill tc_piecewise" \
LOGDIR="$PWD/experiments/results/int8_pool_full" \
  bash experiments/scripts/run_equal_hbm_int8.sh
```

## 输出与聚合

正式结果命名为：

```text
result_kda_int8_N<groups>_w0_ckpt<slots>_recon0_seed<seed>.json
```

关键字段：

- `_bytes_per_slot`：INT8 payload、scale 和脚本可解析的 checkpoint state BPS。
- `_ckpt_slots` / `_ckpt_hbm_mb`：实际 pool 配置。
- `_capacity_x`：相对 dense head-aware BPS 的总容量比，包含 layout 差异。
- `_token_hit_rate`、`mean_ttft_ms`、`output_throughput`：性能指标。

聚合命令：

```bash
python experiments/scripts/plot_equal_hbm.py \
  --logdir "$PWD/experiments/results/int8_pool_full"
```

若要与 matched-HBM dense/ragged 同表比较，把正式 JSON 复制到一个只包含正式结果的后处理目录。不要混入 calibration JSON。

## 验收

1. live `_bytes_per_slot` 大于 0，不能回退到未验证的离线 BPS。
2. `slots = floor(S * dense_bps / int8_bps)`。
3. `int8_bps * slots <= S * dense_bps`，脚本允许的相对 drift 上限为 5%。
4. 三个正式 seed 完整且输入 token 数一致。
5. 报告必须明确 16-head TP-split 与 32-head replicated layout 的差异。
