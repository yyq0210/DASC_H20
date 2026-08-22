---
title: "同布局公平 INT8 matched-HBM 实验"
description: "在 head-aware pool 的相同 TP layout 下复现纯 INT8 checkpoint 量化对照。"
---

# 同布局公平 INT8 matched-HBM 实验

该实验使用与 dense/ragged 相同的 head-aware checkpoint pool，并设置 `Wmax=0` 保留全部 state unit，只把持久化 checkpoint 从 BF16 量化为 INT8。它隔离位宽变化，不引入独立 INT8 pool 的 TP layout 优势。

## 论文映射

- 附录：`INT8 state-checkpoint quantization and DASC composition`。
- 固定-HBM 表中的 `INT8` 行：compression 1.97×、63 slots、0.625 GiB checkpoint budget。
- 主入口：`experiments/scripts/run_int8_fair_equal_hbm.sh`。

量化只作用于缓存中的 temporal recurrent state。active state 和后续 recurrence 保持 BF16；Conv1D window state 保持原始 dtype。每个 `(layer, head, key-channel)` 使用一个 BF16 scale，INT8 payload 使用对称区间 `[-127,127]`。

## 固定配置

| 项目 | 论文固定-HBM点 | N40 扩展 sweep 默认值 |
|---|---|---|
| Workload | N200，1,051 requests | N40，184 requests |
| Dense-equivalent S | 32 | 48/96/144 |
| Seeds | 42,123,7,555,31337 | 42,123,7 |
| Dense BPS | 20,971,520 | 20,971,520 |
| INT8-fair BPS estimate | 10,649,600 | 10,649,600 |
| Compression | 约 1.97× | 约 1.97× |
| Pool layout | head-aware，32 heads/rank | 相同 |
| Wmax | 0 | 0 |

公共模型为 Kimi-Linear-48B、TP=2、2×H20、并发16、输出64 token、LRU、`tc_piecewise` prefill graph。

## 脚本

| 路径 | 作用 |
|---|---|
| `experiments/scripts/run_int8_fair_equal_hbm.sh` | calibration、slot 计算和多预算编排 |
| `experiments/scripts/bench_arena_shared_prefix.sh` | `int8fair` 模式与 server 生命周期 |
| `experiments/scripts/plot_equal_hbm.py` | matched-HBM 汇总 |
| `experiments/scripts/plot_wmax_sweep.py` | 把 int8fair 作为 W-independent baseline 加入 G23 图 |

`int8fair` 模式的核心环境变量为：

```text
SGLANG_FORCE_HEAD_AWARE_WMAX=0
SGLANG_STATE_QUANT_MODE=int8
SGLANG_HEAD_AWARE_RAGGED=1
```

## Dry-run

```bash
S_LIST="48 96 144" \
  bash experiments/scripts/run_int8_fair_equal_hbm.sh --dry-run
```

## 复现论文 N200、S=32 点

先确保 N200 workload 存在：

```bash
python experiments/scripts/build_arena_shared_prefix.py \
  --n-groups 200 \
  --out /tmp/arena_shared_prefix_N200.jsonl
```

运行：

```bash
NGROUPS=200 \
S_LIST=32 \
SEEDS=42,123,7,555,31337 \
MAX_CONC=16 \
OUTPUT_LEN=64 \
BACKEND_FLAGS="--linear-attn-backend triton --cuda-graph-backend-prefill tc_piecewise" \
LOGDIR="$PWD/experiments/results/int8fair_n200_s32" \
  bash experiments/scripts/run_int8_fair_equal_hbm.sh
```

论文目标值：

| Arm | Compression | Slots | Hit rate | TTFT ms | Throughput tok/s |
|---|---:|---:|---:|---:|---:|
| dense BF16 | 1.00× | 32 | 0.08±0.01 | 432.7±1.7 | 785±3 |
| INT8 fair | 1.97× | 63 | 0.23±0.01 | 423.3±5.6 | 767±6 |

该入口只运行 INT8-fair arm。dense 与各 W 的 DASC NR/WR 对照来自 Wmax fixed-HBM 实验，合并时必须使用相同 N200 workload、S=32 和五个 seeds。

## N40 多预算扩展

```bash
NGROUPS=40 \
S_LIST="48 96 144" \
SEEDS=42,123,7 \
BACKEND_FLAGS="--linear-attn-backend triton --cuda-graph-backend-prefill tc_piecewise" \
LOGDIR="$PWD/experiments/results/int8fair_n40_full" \
  bash experiments/scripts/run_int8_fair_equal_hbm.sh
```

N40 扩展结果不能替代论文 N200、S=32 表。

## 输出与合并

正式结果命名为：

```text
result_kda_int8fair_N<groups>_w0_ckpt<slots>_recon0_seed<seed>.json
```

要生成论文固定-HBM综合图，将以下正式 JSON 放入同一后处理目录：

- N200/S32 dense 五 seed。
- N200/S32 int8fair 五 seed。
- N200/S32 每个 W 的 NR/WR 五 seed。

然后运行：

```bash
python experiments/scripts/plot_wmax_sweep.py \
  --logdir "$PWD/experiments/results/wmax_n200_s32_combined" \
  --slots 32
```

## 验收

1. live BPS 对应约 1.97× compression，不能出现独立 INT8 pool 的约 3.9× layout 增益。
2. N200/S32 的 slot 应接近 63，persistent checkpoint HBM 不超过 0.625 GiB。
3. 五个 seed 与 dense、NR、WR 完全配对。
4. `Wmax=0`，所有 state unit 保留；若结果出现 decay-aware drop，则实验配置错误。
5. 报告中区分 checkpoint HBM 与 full-attention KV、模型权重、active-state 和 graph memory。
