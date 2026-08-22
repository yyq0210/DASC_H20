---
title: "DASC 性能实验复现索引"
description: "DASC 论文性能实验的配置、入口脚本、运行顺序和验收口径。"
---

# DASC 性能实验复现索引

本目录按论文中的性能实验族拆分复现说明。每篇文档都列出实验目的、固定变量、入口脚本、最小复现命令、完整命令、输出文件和验收条件。

## 实验清单

| 实验 | 论文问题 | 主入口 | 文档 | 当前验证状态 |
|---|---|---|---|---|
| Matched-HBM serving | 相同 recurrent-state HBM 下，压缩是否通过更多 slot 提升 hit rate、TTFT 和吞吐 | `experiments/scripts/run_equal_hbm.sh` | [matched-HBM serving](matched-hbm-serving.md) | KDA、S=96、三 seed 已复现 |
| Wmax serving sweep | 压缩容量收益与 suffix refresh 开销如何随 `Wmax` 变化 | `experiments/scripts/run_wmax_sweep.sh` | [Wmax sweep](wmax-serving-sweep.md) | dry-run 已验证，GPU 全量 sweep 未重跑 |
| 独立 INT8 pool | 旧 INT8 pool 在 matched-HBM 下的容量与性能，包含 TP layout 差异 | `experiments/scripts/run_equal_hbm_int8.sh` | [独立 INT8 pool](int8-pool-matched-hbm.md) | 未重跑 |
| 同布局公平 INT8 | 在相同 head-aware TP layout 下隔离纯 INT8 量化收益 | `experiments/scripts/run_int8_fair_equal_hbm.sh` | [公平 INT8](int8fair-matched-hbm.md) | 未重跑 |

RULER 准确率实验已单独整理到 [DASC RULER 复现实验](../ruler/README.md)。LoCoMo、五个推理数据集和机制分析不在本目录范围内。

## 公共实验环境

论文性能实验以单机 NVIDIA CUDA 环境运行。当前复现机器配置如下：

| 项目 | 配置 |
|---|---|
| GPU | 2 × NVIDIA H20-3e，单卡 143,771 MiB |
| 并行 | TP=2，单个模型占用两张卡 |
| CUDA / PyTorch | CUDA 12.9，PyTorch 2.11.0 |
| Python / Triton | Python 3.12.13，Triton 3.6.0 |
| SGLang commit | `f05a0725a7e85cabc0d0149c9ebe16d6218e933b` |
| KDA 模型 | `Kimi-Linear-48B-A3B-Instruct` |
| GDN 模型 | `Qwen3-Next-80B-A3B-Instruct-NVFP4` |
| 请求并发 / 输出长度 | `MAX_CONC=16`，`OUTPUT_LEN=64` |
| eviction policy | `lru` |

模型目录默认从 `SGLANG_REPO` 指向的独立 checkout 查找；若权重位于其他位置，通过 `MODEL=/absolute/path` 覆盖。

## 公共 workload

`experiments/scripts/build_arena_shared_prefix.py` 从本地 ShareGPT V3 多轮对话构造严格嵌套的共享前缀。默认输出到 `/tmp`：

```bash
python experiments/scripts/build_arena_shared_prefix.py \
  --n-groups 40 \
  --out /tmp/arena_shared_prefix_N40.jsonl
```

本复现包同时保存了固定副本：`datasets/serving/arena_shared_prefix_N40.jsonl` 和 `datasets/serving/arena_shared_prefix_N200.jsonl`，正式复查时优先使用这些输入，避免源数据或构造逻辑变化。

当前 N40 workload 包含 40 个 conversation group、184 个请求和约 234,443 个输入 token。Wmax 实验使用 N200 workload，包含 200 个 group、1,051 个请求。

运行前保留以下文件并记录哈希：

```bash
sha256sum \
  /tmp/arena_shared_prefix_N40.jsonl \
  /tmp/arena_shared_prefix_N40.stats.json
```

不要把 `/tmp/arena_shared_prefix_N40.jsonl` 与 N200 workload 混用。结果 JSON 中的 `total_input_tokens` 必须在所有 arm 和 seed 间一致。

## 必须锁定 prefill CUDA Graph

当前 runtime 会把未显式指定的 KDA prefill CUDA Graph 解析为 `disabled`。论文性能运行使用 `tc_piecewise`。若不锁定，hit rate 与容量仍可能正确，但 TTFT 和吞吐无法复现。

KDA 使用：

```bash
BACKEND_FLAGS="--linear-attn-backend triton --cuda-graph-backend-prefill tc_piecewise"
```

GDN 使用：

```bash
BACKEND_FLAGS="--attention-backend triton --cuda-graph-backend-prefill tc_piecewise"
```

一个 `BACKEND_FLAGS` 不能同时表达 KDA 和 GDN 的默认 backend，因此完整 matched-HBM sweep 必须按模型拆成两次调用。每个结果 JSON 都应满足：

```text
server_info.cuda_graph_config.prefill.backend == "tc_piecewise"
```

## 公共脚本链路

```text
run_*.sh
  ├─ build_arena_shared_prefix.py     生成 workload（缺失时）
  ├─ bench_arena_shared_prefix.sh     启动 server、warm、测量、写 JSON
  └─ plot_equal_hbm.py / plot_wmax_sweep.py
                                      聚合 mean±sample-SD 并绘图
```

`bench_arena_shared_prefix.sh` 每次只启动一个 server，并在退出时终止它启动的 PID。测量采用一次 warm pass，随后按 seed 在同一 warm cache 上运行，不在 seed 之间 flush。

## 推荐运行顺序

1. 对所有 shell 脚本执行 `bash -n`，对 Python 辅助脚本执行 `python -m py_compile`。
2. 运行每个入口的 `--dry-run`，检查 slot、HBM、饱和区间和预算不变量。
3. 先运行 KDA matched-HBM 的 S=96 最小集合。
4. 再扩展到 S=48/96/144 和 GDN。
5. 运行 Wmax pilot，确认 N200、S=32 仍处于可区分的 hit-rate 区间。
6. 最后运行 Wmax full、独立 INT8 和公平 INT8。

## 统一结果验收

每个正式 arm 至少检查：

- seed 集合完整，不能用 calibration 或 pilot 输出补正式 seed。
- `_bytes_per_slot * _ckpt_slots` 不超过目标预算；matched-HBM 相对误差应小于 2%，INT8 脚本允许 5%。
- 所有正式 arm 的 `total_input_tokens` 完全相同。
- `_token_hit_rate` 使用每个 seed 的聚合 token 比率，再对 seed 求均值与 sample SD。
- `_recon_n_hits` 与 `_replay_tokens_*` 在 WR arm 中存在且与配置一致。
- 服务日志没有 OOM、server restart、CUDA error 或 replay-window eviction。
- 结果目录只包含同一个实验协议的 JSON；聚合脚本会按 glob 读取，混入 calibration、pilot 或旧结果会污染均值。
