---
title: 论文数据点复现：KDA RULER-S1 16k
description: 按 N=30 和三轮 replay 正式协议复现 dense 与 DASC-NR Wmax=16
---

# 论文数据点复现：KDA RULER-S1 16k

## 结论

2026-08-22 在 2 × H20 上完成一次新的论文口径复现。论文 KDA RULER 表中 S1-16k 的 dense 与 DASC-NR Wmax=16 均为 1.00；本次两臂的 warm-up、replay 和 overall accuracy 也均为 1.0000，数据点精确复现。

## 配置

| 项目 | 值 |
| --- | --- |
| 模型 | Kimi-Linear-48B-A3B-Instruct |
| SGLang commit | f05a0725a7e85cabc0d0149c9ebe16d6218e933b |
| GPU / 并行 | 2 × NVIDIA H20-3e / TP=2 |
| task / length | RULER S1 / 16k |
| arms | dense、DASC-NR Wmax=16 |
| 独立样本 | N=30，两个 arms 使用相同生成协议 |
| 请求轮次 | 1 次 cache-population warm-up + 3 次 replay |
| 并发 | client parallel=8，max running=8 |
| checkpoint slots | 200 |
| DASC | ragged=1、re-prefill=0、route A、BF16 |
| 输出 | temperature=0，max-new-tokens=64 |
| prefill graph | tc_piecewise |

每臂共 30 × 4 = 120 个请求，两臂合计 240 个请求。

## 复现命令

```bash
cd /path/to/DASC_H20
BACKEND=kda WMAX_LIST=16 RULER_TASKS=s1 LENGTHS=16k \
  NUM_SAMPLES=30 NUM_TRIES=4 PARALLEL=8 MAXNEW=64 \
  CKPT_SIZE=200 PORT=30019 \
  LOGDIR="$PWD/experiments/results/ruler_paper_kda_s1_16k_20260822" \
  BACKEND_FLAGS="--linear-attn-backend triton --cuda-graph-backend-prefill tc_piecewise" \
  bash experiments/scripts/run_ruler_suite_smallctx_redo.sh
```

## 论文值与实测值

| arm | 论文 S1-16k | warm-up | replay（正式口径） | overall | 差值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| dense | 1.00 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| DASC-NR Wmax=16 | 1.00 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |

原始 token 计数为：

| arm | cached tokens | new tokens | pooled token hit |
| --- | ---: | ---: | ---: |
| dense | 2,095,657 | 390,323 | 0.842990 |
| DASC-NR Wmax=16 | 2,104,247 | 381,733 | 0.846446 |

token hit 是本次链路校验的附加数据，不是该论文准确率单元格的验收指标。

## 输出与验收

- 汇总：`experiments/results/ruler_paper_kda_s1_16k_20260822/acc_results.jsonl`
- 原始生成：`out_dense_s1_16k.jsonl`、`out_w16_s1_16k.jsonl`，各 120 行。
- 服务日志：`server_smallctx_dense.log`、`server_smallctx_w16.log`。
- 客户端日志：`bench_dense_s1_16k.log`、`bench_w16_s1_16k.log`。
- 两个服务日志均确认 prefill backend 为 `tc_piecewise`，且无 OOM、CUDA error 或 traceback。
- DASC 每 slot recurrent-state storage 为约 7.48/7.57 MB（两个 TP rank），dense 为 20 MB，压缩约 2.77–2.81×。

本次只复现论文表中的一个成对数据点，不代表 13 个任务和五档 Wmax 的完整矩阵。
