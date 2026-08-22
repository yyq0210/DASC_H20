---
title: "DASC matched-HBM serving 实验"
description: "复现 dense、padded 和 ragged 在相同 recurrent-state HBM 下的 serving 性能。"
---

# DASC matched-HBM serving 实验

该实验固定 recurrent-state checkpoint pool 的 HBM 字节数，让压缩 arm 用更小的 bytes/slot 换取更多 slot。它验证论文中的容量 → hit rate → TTFT/吞吐链路，而不是比较不同显存预算。

## 论文映射

- 正文：`Matched-HBM serving`，S=96 主表。
- 附录：`Additional matched-HBM serving budgets`，S=48 和 S=144。
- 主入口：`experiments/scripts/run_equal_hbm.sh`。

论文 S=96 目标值：

| 模型 | Arm | State HBM GiB | Slots | Hit rate | TTFT ms | Throughput tok/s |
|---|---|---:|---:|---:|---:|---:|
| KDA | dense | 1.88 | 96 | 0.5887±0.0211 | 248.8±24.6 | 1061.8±28.4 |
| KDA | padded | 1.88 | 161 | 0.7585±0.0530 | 182.6±36.0 | 1240.2±83.4 |
| KDA | ragged | 1.87 | 267 | 0.9729±0.0051 | 82.9±15.5 | 1504.7±40.0 |
| GDN | dense | 3.38 | 96 | 0.6045±0.0286 | 499.5±26.1 | 690.0±17.4 |
| GDN | padded | 3.38 | 99 | 0.6027±0.0320 | 499.5±27.3 | 688.5±20.2 |
| GDN | ragged | 3.35 | 128 | 0.7174±0.0307 | 426.3±17.1 | 758.4±23.4 |

## 对照设计

| Arm | `Wmax` | Layout | Reconstruction | 作用 |
|---|---:|---|---|---|
| dense | 0 | 完整 BF16 | 无 | 基准 bytes/slot 与性能 |
| padded | KDA 16 / GDN 64 | 每层 pad 到最大保留宽度 | NR | 分离选择收益与 ragged packing 收益 |
| ragged | KDA 16 / GDN 64 | 跨层 flat-pack | NR | DASC 完整存储路径 |

预算定义：

```text
budget_bytes = S * dense_bytes_per_slot
arm_slots = floor(budget_bytes / live_arm_bytes_per_slot)
```

脚本先对 padded/ragged 各启动一次服务，读取 live bytes/slot，再计算正式 slot 数。不要用 dry-run 中的离线 BPS 代替 live calibration。

## 固定配置

| 配置 | KDA | GDN |
|---|---|---|
| 模型 | `Kimi-Linear-48B-A3B-Instruct` | `Qwen3-Next-80B-A3B-Instruct-NVFP4` |
| TP | 2 | 2 |
| DASC `Wmax` | 16 | 64 |
| dense BPS | 20,971,520 | 37,748,736 |
| `MEM_FRAC` | 0.85 | 0.70 |
| attention flags | `--linear-attn-backend triton` | `--attention-backend triton` |
| prefill graph | `tc_piecewise` | `tc_piecewise` |

公共配置为 N40、184 请求、并发 16、输出 64 token、`lru`、seeds 42/123/7、`--chunked-prefill-size 2048`、`--disable-overlap-schedule`。

## 脚本

| 路径 | 作用 |
|---|---|
| `experiments/scripts/run_equal_hbm.sh` | calibration、预算计算、三 arm 编排、失败检查 |
| `experiments/scripts/bench_arena_shared_prefix.sh` | server 生命周期、warm/measure、hit 统计、JSON 注入 |
| `experiments/scripts/build_arena_shared_prefix.py` | 生成 N40 ShareGPT 多轮共享前缀 workload |
| `experiments/scripts/plot_equal_hbm.py` | 按 backend/mode/S 聚合 mean±SD，检查预算与输入 token |

## Dry-run

```bash
BACKENDS="kda gdn" S_LIST="48 96 144" \
  bash experiments/scripts/run_equal_hbm.sh --dry-run
```

通过条件：输出 `EQ-GATE-0: PASS`，dense slots 小于 184，且每个模型内部 `dense <= padded <= ragged`。

## 最小复现：KDA、S=96

```bash
BACKENDS=kda \
S_LIST=96 \
SEEDS=42,123,7 \
NGROUPS=40 \
MAX_CONC=16 \
OUTPUT_LEN=64 \
BACKEND_FLAGS="--linear-attn-backend triton --cuda-graph-backend-prefill tc_piecewise" \
LOGDIR="$PWD/experiments/results/matched_hbm_kda_s96" \
  bash experiments/scripts/run_equal_hbm.sh
```

该命令包含两个 calibration launch 和三个正式 arm。不要把 calibration JSON 当作正式 S=96 结果。

## 完整复现

KDA：

```bash
BACKENDS=kda \
S_LIST="48 96 144" \
SEEDS=42,123,7 \
NGROUPS=40 \
BACKEND_FLAGS="--linear-attn-backend triton --cuda-graph-backend-prefill tc_piecewise" \
LOGDIR="$PWD/experiments/results/matched_hbm_kda_full" \
  bash experiments/scripts/run_equal_hbm.sh
```

GDN：

```bash
BACKENDS=gdn \
S_LIST="48 96 144" \
SEEDS=42,123,7 \
NGROUPS=40 \
BACKEND_FLAGS="--attention-backend triton --cuda-graph-backend-prefill tc_piecewise" \
LOGDIR="$PWD/experiments/results/matched_hbm_gdn_full" \
  bash experiments/scripts/run_equal_hbm.sh
```

不要把两个模型放进同一次带 `BACKEND_FLAGS` 的调用，否则其中一个模型会收到错误的 backend flag。

## 单 arm 诊断

已知 slot 数后，可直接运行一个 arm：

```bash
NGROUPS=40 \
MEASURE_SEEDS=42,123,7 \
CKPT_OVERRIDE=267 \
WMAX_DASC=16 \
RECON=0 \
BACKEND_FLAGS="--linear-attn-backend triton --cuda-graph-backend-prefill tc_piecewise" \
LOGDIR="$PWD/experiments/results/matched_hbm_kda_ragged_diag" \
  bash experiments/scripts/bench_arena_shared_prefix.sh kda ragged
```

该命令适合验证单个配置，不替代 live BPS calibration。

## 输出

| 文件 | 内容 |
|---|---|
| `orchestrator_equal_hbm_*.log` | 编排进度、live BPS、slot、预算交叉检查 |
| `server_<label>.log` | server args、pool 容量、prefill/decode 日志 |
| `bench_<label>_warm.log` | warm pass |
| `bench_<label>_measure_s<seed>.log` | 每个 seed 的 benchmark 文本输出 |
| `result_<label>_seed<seed>.json` | 指标、server_info、BPS、slot、hit、seed |
| `equal_hbm.pdf/png` | 安装 matplotlib 时生成的聚合图 |

`plot_equal_hbm.py` 会读取目录中所有 `result_*_seed*.json`。calibration 文件与正式文件共用相同命名规则，可能被错误聚合为额外预算行。正式报告应只选择目标 slot 的 JSON，或把 calibration 与正式结果放在不同的后处理输入目录。

## 验收

1. 每个正式 arm 有 3 个 seed JSON，seed 集合为 `{42,123,7}`。
2. 所有正式结果的 `total_input_tokens` 相同；当前 N40 应为 234,443。
3. `_bytes_per_slot * _ckpt_slots <= S * dense_bps`，相对 slack 小于 2%。
4. 结果 JSON 的 prefill backend 为 `tc_piecewise`。
5. KDA S=96 的 live slot 应接近 96/161/267，ragged compression 约 2.79×。
6. 三 seed 均值相对论文均值建议控制在 5% 内；性能受驱动、时钟、依赖版本影响，不要求逐 seed 完全相同。

当前机器的 KDA S=96 复现结果为：

| Arm | Hit rate | TTFT ms | Throughput tok/s |
|---|---:|---:|---:|
| dense | 0.5851±0.0301 | 253.4±19.9 | 1061.9±23.9 |
| padded | 0.7835±0.0571 | 176.6±36.5 | 1280.5±96.6 |
| ragged | 0.9729±0.0051 | 86.1±22.0 | 1513.7±46.7 |

ragged 相对 dense：hit rate +66.3%、TTFT -66.0%、throughput +42.5%。
