---
title: RULER 4k/8k/16k 完整 Sweep
description: KDA 和 GDN 的 13 任务、五档 Wmax 正式准确率实验配置
---

# RULER 4k/8k/16k 完整 Sweep

对应 `experiments/scripts/run_ruler_suite_smallctx_redo.sh`，是论文 RULER 主结果的主要入口。

## 默认配置

| 项目 | KDA | GDN |
| --- | --- | --- |
| 模型 | Kimi-Linear-48B-A3B-Instruct | Qwen3-Next-80B-A3B-Instruct-NVFP4 |
| backend | linear-attn triton | attention triton |
| mem fraction | 0.85 | 0.90 |
| TP / checkpoint slots | 2 / 200 | 2 / 200 |
| client parallel / max running | 8 / 8 | 8 / 8 |

共享配置：13 个任务，长度 4k/8k/16k，N=30，4 tries，Wmax=16/64/128/512/1024，ragged=1，默认 no-reconstruction，temperature=0，max-new-tokens=64。

每个服务 arm 执行 13 × 3 × 30 × 4 = 4,680 个请求。完整 backend sweep 包含 dense 加五档 Wmax，共 6 次服务启动。

## 命令

```bash
cd /path/to/DASC_H20
BACKEND=kda bash experiments/scripts/run_ruler_suite_smallctx_redo.sh
BACKEND=gdn bash experiments/scripts/run_ruler_suite_smallctx_redo.sh
```

需要 suffix refresh 时显式设置 `REPREFILL=1`。脚本支持用 `LOGDIR=/absolute/path` 隔离结果；正式运行前不要用较小参数写入同一目录，否则 resume 逻辑会把同名 arm 当作已完成。

已完成的正式单点命令与结果见 [KDA RULER-S1 16k](paper-kda-s1-16k.md)。

## 输出

KDA 写入 `idea1_accuracy_logs/ruler_suite_kda_smallctx_redo/`，GDN 写入 `idea1_accuracy_logs/ruler_suite_gdn_norecon_smallctx_redo/`。核心文件为 `acc_results.jsonl`、`summary_smallctx.txt`、每 arm 的输出/bench 日志和每个服务 arm 的 server 日志。

## 注意事项

- 未设置 `LOGDIR` 时使用上述默认目录；正式复现建议显式指定新目录。
- 启动前会执行 `pkill -f "launch_server.*$PORT"`；共享机器务必使用独占端口。
- 结果按 arm 名 resume-skip，不校验 N、tries 或其他配置是否变化。
- max-new-tokens=64 曾导致部分 GDN CWE/MK3 等任务截断；这些任务应以 [GDN 定点复跑](gdn-cwe-mk3-rerun.md) 的 256-token 结果为准。
- 论文验收应核对 replay accuracy，而不是把 cache-population warm-up 混入平均值。
