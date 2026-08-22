#!/usr/bin/env python3
"""Joint per-request RULER accuracy + TTFT probe (Figure 3 upgrade).

Unlike the cross-workload Figure 3(left) — where TTFT comes from ShareGPT serving
and accuracy from separate RULER runs — this probe sends each RULER 16k request
through the **normal serving path** with **streaming HTTP** so that BOTH accuracy
AND TTFT are measured on the SAME request.  Cache runs normally (no forced hit);
each instance is repeated ``numtries`` times so try 1 is a miss and tries 2..N
are hits (assuming the checkpoint pool is oversized, which the driver guarantees
with ``ckpt_size >> distinct instances``).

Tasks (4 representative RULER subtasks):
  s1  — NIAH single retrieval (single needle, repeat haystack)
  mk1 — multi-key NIAH (4 keys, essay haystack, 1 queried)
  vt  — variable tracking (4-hop chain in noise haystack)
  cwe — common words extraction (top-10 frequent words)

Scoring: ``ruler_gen.score_recall`` (RULER string_match_part).

Usage:
  python experiments/scripts/bench_ruler_joint.py --host 127.0.0.1 --port 30000 \\
      --arm-label dense --n 50 --numtries 4 \\
      --output experiments/results/ruler_joint/result_dense_joint.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import requests

# Reuse the OFFICIAL RULER material from datasets/ruler/ruler_gen.py.
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_REPO, "datasets", "ruler"))
import ruler_gen  # noqa: E402

TASKS = ["s1", "mk1", "vt", "cwe"]
# tokens_to_generate from ruler_gen.TASKS[task_type]["tokens_to_generate"]
# s1/mk1 -> niah -> 128; vt -> variable_tracking -> 30; cwe -> common_words_extraction -> 120
MAX_TOKENS = {"s1": 128, "mk1": 128, "vt": 30, "cwe": 120}
RULER_LENGTH = "16k"


# --------------------------------------------------------------------------- #
# Streaming HTTP                                                              #
# --------------------------------------------------------------------------- #
def send_streaming(
    host: str,
    port: int,
    prompt: str,
    max_new_tokens: int,
    temperature: float = 0.0,
    timeout: int = 600,
) -> Tuple[Optional[float], str, int]:
    """Send one streaming /generate request.

    Returns ``(ttft_ms, full_text, cached_tokens)``.
    ``ttft_ms`` is the time from POST to the first chunk carrying text output.
    ``cached_tokens`` is extracted from the final ``meta_info`` if present.
    """
    url = f"http://{host}:{port}/generate"
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        },
        "stream": True,
    }
    st = time.perf_counter()
    ttft_ms: Optional[float] = None
    full_text = ""
    cached_tokens = 0
    resp = requests.post(url, json=payload, stream=True, timeout=timeout)
    resp.raise_for_status()
    for raw in resp.iter_lines():
        if not raw:
            continue
        line = raw.decode("utf-8", errors="replace")
        if line.startswith("data:"):
            line = line[len("data:") :].strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        # The native /generate SSE API returns cumulative output in ``text``.
        # Older experiment branches used a delta-style ``text_output`` field,
        # so keep that as a compatibility fallback.
        if isinstance(data.get("text"), str):
            current_text = data["text"]
            if current_text and ttft_ms is None:
                ttft_ms = (time.perf_counter() - st) * 1000.0
            full_text = current_text
        elif isinstance(data.get("text_output"), str):
            delta_text = data["text_output"]
            if delta_text and ttft_ms is None:
                ttft_ms = (time.perf_counter() - st) * 1000.0
            full_text += delta_text
        meta = data.get("meta_info")
        if isinstance(meta, dict) and meta.get("cached_tokens") is not None:
            cached_tokens = meta["cached_tokens"]
    if ttft_ms is None:
        ttft_ms = (time.perf_counter() - st) * 1000.0
    return ttft_ms, full_text, cached_tokens


# --------------------------------------------------------------------------- #
# Instance generation                                                         #
# --------------------------------------------------------------------------- #
def generate_all_samples(
    n: int, base_seed: int = 42
) -> Dict[str, List[Tuple[str, list]]]:
    """Build ``n`` RULER 16k instances per subtask.

    Returns ``{task: [(prompt, gold_list), ...]}``.
    """
    samples: Dict[str, List[Tuple[str, list]]] = {}
    for task in TASKS:
        samples[task] = ruler_gen.build_ruler_samples(task, RULER_LENGTH, n, base_seed)
    return samples


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def run_arm(args) -> Dict:
    n = args.n
    numtries = args.numtries
    base_seed = args.base_seed

    print(
        f"[joint] generating {len(TASKS)} tasks x {n} instances x {RULER_LENGTH} "
        f"(base_seed={base_seed}) ..."
    )
    samples = generate_all_samples(n, base_seed)
    for task in TASKS:
        chars = len(samples[task][0][0])
        print(
            f"  {task}: {n} instances, ~{chars} chars/instance, "
            f"gold={len(samples[task][0][1])}"
        )

    records: List[dict] = []
    # Interleave instances across tasks so that no single task monopolises the
    # cache pool ordering.  Within an instance, tries are strictly sequential
    # (1->2->...->numtries) so try 1 misses and populates cache; 2..N hit.
    total = len(TASKS) * n * numtries
    done = 0
    for task in TASKS:
        for i in range(n):
            prompt, gold = samples[task][i]
            for try_idx in range(1, numtries + 1):
                ttft_ms, text, cached_tokens = send_streaming(
                    args.host,
                    args.port,
                    prompt,
                    MAX_TOKENS[task],
                    temperature=args.temperature,
                )
                acc = ruler_gen.score_recall(text, gold)
                rec = {
                    "task": task,
                    "instance_idx": i,
                    "try_idx": try_idx,
                    "ttft_ms": round(ttft_ms, 2),
                    "accuracy": acc,
                    "cached_tokens": cached_tokens,
                    "prompt_chars": len(prompt),
                    "output_text": text,
                    "gold": [str(g) for g in gold],
                }
                records.append(rec)
                done += 1
                if done % 50 == 0 or done == total:
                    print(
                        f"[joint] {done}/{total}  last: task={task} try={try_idx} "
                        f"ttft={ttft_ms:.0f}ms acc={acc:.2f} cached={cached_tokens}"
                    )

    summary = summarize(records)
    result = {
        "arm": args.arm_label,
        "config": {
            "length": RULER_LENGTH,
            "n_per_task": n,
            "numtries": numtries,
            "tasks": list(TASKS),
            "max_tokens": dict(MAX_TOKENS),
            "base_seed": base_seed,
            "temperature": args.temperature,
        },
        "records": records,
        "summary": summary,
    }
    return result


def summarize(records: List[dict]) -> Dict:
    """Per-task + overall summary."""
    by_task: Dict[str, List[dict]] = {}
    for r in records:
        by_task.setdefault(r["task"], []).append(r)

    per_task: Dict[str, dict] = {}
    for task, recs in sorted(by_task.items()):
        try1 = [r for r in recs if r["try_idx"] == 1]
        hit_recs = [r for r in recs if r["try_idx"] > 1]
        accs = [r["accuracy"] for r in recs]
        ttft1 = [r["ttft_ms"] for r in try1]
        ttfthit = [r["ttft_ms"] for r in hit_recs]
        cached_nonzero = sum(
            1 for r in recs if r["cached_tokens"] and r["cached_tokens"] > 0
        )
        per_task[task] = {
            "n_instances": len(try1),
            "numtries": max(r["try_idx"] for r in recs),
            "acc_mean": _mean(accs),
            "acc_std": _std(accs),
            "ttft_try1_mean_ms": _mean(ttft1),
            "ttft_try1_std_ms": _std(ttft1),
            "ttft_hit_mean_ms": _mean(ttfthit),
            "ttft_hit_std_ms": _std(ttfthit),
            "hit_rate_from_cached": cached_nonzero / len(recs) if recs else 0.0,
            "hit_rate_from_try": len(hit_recs) / len(recs) if recs else 0.0,
        }

    all_accs = [r["accuracy"] for r in records]
    all_ttfts = [r["ttft_ms"] for r in records]
    return {
        "per_task": per_task,
        "overall_acc_mean": _mean(all_accs),
        "overall_acc_std": _std(all_accs),
        "overall_ttft_mean_ms": _mean(all_ttfts),
        "overall_ttft_std_ms": _std(all_ttfts),
        "total_requests": len(records),
        "tasks": sorted(by_task.keys()),
    }


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _std(xs):
    import math

    xs = [x for x in xs if x is not None]
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", "30000")))
    ap.add_argument("--arm-label", default=os.environ.get("ARM_LABEL", "dense"))
    ap.add_argument(
        "--n",
        type=int,
        default=int(os.environ.get("N", "50")),
        help="instances per subtask",
    )
    ap.add_argument(
        "--numtries",
        type=int,
        default=int(os.environ.get("NUMTRIES", "4")),
        help="repeats per instance (1 miss + (N-1) hits)",
    )
    ap.add_argument("--base-seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument(
        "--output",
        default=os.environ.get("OUTPUT", "experiments/results/ruler_joint/result_dense_joint.json"),
    )
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    result = run_arm(args)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    s = result["summary"]
    print(f"\n[joint] arm={args.arm_label} -> {args.output}")
    print(
        f"  overall: acc={s['overall_acc_mean']:.4f}+-{s['overall_acc_std']:.4f}  "
        f"ttft={s['overall_ttft_mean_ms']:.1f}+-{s['overall_ttft_std_ms']:.1f}ms  "
        f"n={s['total_requests']}"
    )
    for task, ts in sorted(s["per_task"].items()):
        print(
            f"  {task}: acc={ts['acc_mean']:.4f}  "
            f"ttft_miss={ts['ttft_try1_mean_ms']:.1f}ms  "
            f"ttft_hit={ts['ttft_hit_mean_ms']:.1f}ms  "
            f"hit={ts['hit_rate_from_try']:.3f}"
        )


if __name__ == "__main__":
    main()
