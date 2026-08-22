---
title: GDN CWE 与 MK3 定点复跑
description: 使用 256 输出 token 修复 GDN 聚合与多键任务截断的定点实验
---

# GDN CWE 与 MK3 定点复跑

对应 `experiments/scripts/run_gdn_cwe_mk3_rerun.sh`。该实验只复跑容易被 64-token 上限截断的 CWE 和 MK3，并同时比较 no-reconstruction 与 reconstruction。

## 配置

| 项目 | 值 |
| --- | --- |
| 模型 | Qwen3-Next-80B-A3B-Instruct-NVFP4 |
| GPU / backend | TP=2 / Triton attention |
| 长度 | 4k、8k、16k |
| 任务 | cwe、mk3 |
| N / tries | 30 / 4 |
| 输出上限 | 256 tokens |
| Wmax | 16、64、128、512、1024 |
| arms | dense + 5 norecon + 5 recon |
| checkpoint slots | 200 |
| ragged | DASC arms 为 1 |

共 11 次服务启动；每次 arm 有 2 × 3 × 30 × 4 = 720 个请求。recon arms 把 max-running-requests 从 4 提高到 16。

## 命令与输出

```bash
cd /path/to/DASC_H20
bash experiments/scripts/run_gdn_cwe_mk3_rerun.sh
```

结果固定写入 `idea1_accuracy_logs/ruler_gdn_cwe_mk3_rerun/`，包括 `acc_results.jsonl`、`summary.txt`、orchestrator 日志以及每 arm 的输出、bench 和 server 日志。

## 注意事项

- 脚本硬编码了仓库路径、模型、端口和 LOGDIR，不支持常规环境变量覆盖。
- 它不在启动前按端口清理旧服务；启动前需自行确认 31007 未占用。
- resume 只看 arm 名；改变配置后应换新目录或清理相应的旧记录。
- 正式表应使用 256-token 复跑结果替代曾被截断的 CWE/MK3 行。
