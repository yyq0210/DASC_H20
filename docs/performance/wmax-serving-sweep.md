---
title: "DASC Wmax serving sweep"
description: "复现固定 slot 的 suffix-refresh 开销和固定 HBM 的容量性能曲线。"
---

# DASC Wmax serving sweep

该实验只运行 KDA。它把两种效应分开：固定 slot 时测 WR 相对 NR 的纯 reconstruction 开销；固定 HBM 时测压缩增加 slot 后的 hit rate、TTFT 和吞吐收益。

## 论文映射

- 正文：`DASC-WR: accuracy--latency trade-off`。
- 附录：`Fixed-slot suffix-refresh cost`。
- 附录：`Fixed-HBM KDA serving` 表中的 dense、NR 和 WR 行。
- 主入口：`experiments/scripts/run_wmax_sweep.sh`。

## 实验组

| 组 | 固定量 | Arm | 回答的问题 |
|---|---|---|---|
| G1 | checkpoint slots | 每个 W 的 NR/WR | suffix refresh 增加多少 TTFT |
| G2 | checkpoint HBM | dense + 每个 W 的 NR | 压缩容量本身带来多少收益 |
| G3 | checkpoint HBM | 每个 W 的 WR | 加入 refresh 后的净收益 |

默认 W 集合为 16、64、128、512、1024。NR 对应 `RECON=0`，WR 对应 `RECON=1`。

## 固定配置

| 项目 | 值 |
|---|---|
| 模型 | `Kimi-Linear-48B-A3B-Instruct` |
| GPU / TP | 2 × H20，TP=2 |
| Workload | ShareGPT V3 N200，200 groups，1,051 requests |
| 并发 / 输出 | 16 / 64 token |
| Seeds | 42, 123, 7, 555, 31337 |
| Dense BPS | 20,971,520 |
| G23 dense-equivalent slots | 32 |
| G23 persistent checkpoint HBM | 0.625 GiB |
| Layout | ragged |
| Eviction | LRU |
| Prefill graph | `tc_piecewise` |

论文 fixed-HBM 的代表性 slot 数为：

| Arm | Compression | Slots |
|---|---:|---:|
| dense | 1.00× | 32 |
| W16 | 2.79× | 89 |
| W64 | 8.91× | 285 |
| W128 | 28.1× | 899 |
| W512 | 581× | 18,591 |

W512/W1024 会产生数万 slot。此时 slot metadata 可能不再可忽略，dry-run 的警告不能当作通过。

## 脚本

| 路径 | 作用 |
|---|---|
| `experiments/scripts/run_wmax_sweep.sh` | pilot、G1/G23 编排、manifest、预算检查 |
| `experiments/scripts/compute_bps_wmax.py` | 从真实 Kimi 权重构造计划并计算每个 W 的 BPS |
| `experiments/scripts/bench_arena_shared_prefix.sh` | 运行单个 NR/WR arm |
| `experiments/scripts/plot_wmax_sweep.py` | 配对 seed、计算 ΔTTFT、检查预算并绘图 |

## 先计算 BPS

默认输出目录：

```bash
python experiments/scripts/compute_bps_wmax.py
```

若使用自定义 `LOGDIR`，必须显式把 BPS 文件写到该目录。当前 orchestrator 在 BPS 文件缺失时调用计算脚本，但没有向计算脚本转发自定义输出路径。

```bash
WMAX_LOGDIR="$PWD/experiments/results/wmax_sweep"
mkdir -p "$WMAX_LOGDIR"
python experiments/scripts/compute_bps_wmax.py \
  --out "$WMAX_LOGDIR/bps_wmax_table.json"
```

## Dry-run

```bash
NGROUPS=200 \
SLOTS=32 \
W_LIST="16 64 128 512 1024" \
LOGDIR="$PWD/experiments/results/wmax_sweep" \
  bash experiments/scripts/run_wmax_sweep.sh --dry-run
```

检查：

- W16/64/128 没有全部饱和，否则 hit-rate 曲线失去区分度。
- 每个 G23 arm 使用 floor slot，不能超过 `SLOTS * dense_bps`。
- W512/W1024 的 `max CKPT_W > 10000` 警告必须在报告中保留。

## Pilot

```bash
PILOT_N_LIST=200 \
PILOT_S_LIST="24 32 48" \
PILOT_W_LIST="16 64 128" \
PILOT_SEEDS=42 \
BACKEND_FLAGS="--linear-attn-backend triton --cuda-graph-backend-prefill tc_piecewise" \
LOGDIR="$PWD/experiments/results/wmax_pilot" \
  bash experiments/scripts/run_wmax_sweep.sh --pilot
```

选择满足以下条件的最大 S：W128 hit < 0.95 且 W16 hit < 0.9。论文固定-HBM设置使用 N200、S=32。

## 固定-HBM full sweep

先生成自定义目录的 BPS 文件，然后运行：

```bash
WMAX_LOGDIR="$PWD/experiments/results/wmax_fixed_hbm_s32"
mkdir -p "$WMAX_LOGDIR"
python experiments/scripts/compute_bps_wmax.py \
  --out "$WMAX_LOGDIR/bps_wmax_table.json"

NGROUPS=200 \
SLOTS=32 \
W_LIST="16 64 128 512 1024" \
SEEDS=42,123,7,555,31337 \
MAX_CONC=16 \
OUTPUT_LEN=64 \
BACKEND_FLAGS="--linear-attn-backend triton --cuda-graph-backend-prefill tc_piecewise" \
LOGDIR="$WMAX_LOGDIR" \
  bash experiments/scripts/run_wmax_sweep.sh --full
```

该命令运行 21 个正式 launch：G1 10 个、G23 dense 1 个、G23 NR/WR 10 个。

## 复现论文 fixed-slot=899 表

`run_wmax_sweep.sh` 用同一个 `SLOTS` 同时控制 G1 slot 和 G23 dense-equivalent budget。默认 `SLOTS=32` 能复现论文固定-HBM表，但它的 G1 是 fixed-slot=32；论文附录 fixed-slot 表固定为 899 slots。

因此 fixed-slot=899 应单独运行，不要把 `SLOTS=899` 传给整套 full sweep。对每个 W 分别运行 NR 和 WR：

```bash
NGROUPS=200 \
MEASURE_SEEDS=42,123,7,555,31337 \
CKPT_OVERRIDE=899 \
WMAX_DASC=128 \
RECON=0 \
BACKEND_FLAGS="--linear-attn-backend triton --cuda-graph-backend-prefill tc_piecewise" \
LOGDIR="$PWD/experiments/results/wmax_fixed_slots_899" \
  bash experiments/scripts/bench_arena_shared_prefix.sh kda ragged
```

把 `WMAX_DASC` 依次改为 16/64/128/512/1024，并对每个 W 再运行一次 `RECON=1`。共 10 个 launch。NR 与 WR 必须使用完全相同的 seed 集合。

## 输出

| 文件 | 内容 |
|---|---|
| `bps_wmax_table.json` | BPS、compression、各预算 slot 表 |
| `manifest_*.json` | 每个 arm 的状态与 seed JSON 数 |
| `orchestrator_wmax_*.log` | 运行进度与 budget slack |
| `result_kda_*_seed*.json` | TTFT、throughput、hit、recon counter、peak memory |
| `fig_wmax_*.pdf/png` | 安装 matplotlib 时生成的曲线 |

## 验收

1. 每个 W 的 NR/WR seed 集合完全匹配，`plot_wmax_sweep.py` 的 paired completeness 全部为 `OK`。
2. G23 每个 arm 的 `bps * slots <= budget`，相对 drift 小于 2%。
3. WR 中 `_recon_n_hits > 0`，NR 中为 0；`_replay_tokens_mean` 与 W 一致或受真实 prefix 长度上限约束。
4. G1 比较使用同一 slot 数；G23 比较使用同一 persistent checkpoint HBM。
5. W128 是经过 end-task accuracy 验证的边界；W512/W1024 只作为 scaling study，不外推质量结论。
6. fixed-slot=899 的论文目标 ΔTTFT 从 W16 的约 61.8 ms 增长到 W1024 的约 393.3 ms。
