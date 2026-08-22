#!/usr/bin/env python3
"""Build a REAL multi-turn shared-prefix serving workload to feed `bench_serving
--dataset-name custom` (replaces the SYNTHETIC generated-shared-prefix load).

Sources:
  --source sharegpt : local ShareGPT V3 JSON (real multi-turn, ungated; DEFAULT).
  --source arena    : HF `lmsys/chatbot_arena_conversations` (GATED -> needs HF_TOKEN).

NOTE (2026-07-29): `lmsys/chatbot_arena_conversations` turned out to be gated and no
HF token is available on this box; user chose ShareGPT V3 (cached locally, real
multi-turn human-AI conversations, deeper turns than Arena). Data source is reported
honestly in the stats/caption -- NOT relabeled as Arena.

Mechanism (docs/EXPERIMENT_SPEC_lmsys_serving.md):
  A real multi-turn conversation naturally reuses a shared prefix: round k's input =
  (rounds 1..k-1 history) + new user turn. Round k's content is a strict character-
  prefix of round k+1's content -> radix tokenizes to a nested shared prefix ->
  cross-request prefix hits, with NO single global hot prefix (each conversation has
  a distinct prefix) -- exactly the profile DASC needs.

We flatten each conversation into one custom-jsonl line per round k>=emit_from_round:
    {"conversations":[{"role":"user","content": <serialized msgs[0..this user]>},
                      {"role":"assistant","content": "OK."}]}
The custom loader (benchmark/datasets/custom.py) reads only conversations[0] as the
prompt and random.shuffles all lines -- the shuffle interleaves groups (realistic
arrival) but does NOT break character-identical prefixes, so reuse is preserved.

Two-pass to keep it cheap: pass-1 filters/sorts candidates by a char/4 length approx
(no tokenizer over the whole corpus); pass-2 exact-tokenizes ONLY the selected N
groups for stats + selection reporting.

GATE-0 (workload health): run this alone, read the printed distribution. If
qualifying groups < N or prefix median is too short, STOP and report the boundary.

Usage:
  python experiments/scripts/build_arena_shared_prefix.py --n-groups 40
  python experiments/scripts/build_arena_shared_prefix.py --n-groups 100
"""

import argparse
import glob
import json
import os
import statistics
import sys
from typing import List, Optional


def log(*a):
    print(*a, file=sys.stderr, flush=True)


def pct(xs: List[float], q: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def load_tokenizer(path: str):
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        log(f"[tok] loaded {path}")
        return tok
    except Exception as e:  # noqa
        log(f"[tok] FAILED to load {path} ({e}); using ~len/4 estimate")
        return None


def ntok(tok, text: str) -> int:
    if tok is None:
        return max(1, len(text) // 4)
    return len(tok.encode(text, add_special_tokens=False))


ROLE_LABEL = {
    "user": "User",
    "human": "User",
    "assistant": "Assistant",
    "gpt": "Assistant",
    "bard": "Assistant",
    "bing": "Assistant",
    "system": "System",
}
USER_ROLES = ("user", "human")


def norm_role(r: str) -> str:
    return str(r).lower()


def is_user(role: str) -> bool:
    return norm_role(role) in USER_ROLES


def serialize(msgs) -> str:
    """Deterministic canonical serialization. Nested by construction: the text for
    msgs[0..i] is a strict character-prefix of msgs[0..j] for i<j."""
    parts = []
    for m in msgs:
        parts.append(
            f"{ROLE_LABEL.get(norm_role(m['role']), 'User')}: {m['content']}\n\n"
        )
    return "".join(parts)


def looks_english(text: str) -> bool:
    if not text:
        return False
    sample = text[:400]
    letters = sum(c.isascii() and c.isalpha() for c in sample)
    return letters >= 0.45 * max(1, len(sample))


# ---------------------------------------------------------------------------
# sources: yield normalized conversations as list of {"role","content"} dicts
# ---------------------------------------------------------------------------
def iter_sharegpt(path: str):
    log(f"[load] sharegpt json {path} ...")
    data = json.load(open(path))
    log(f"[load] {len(data)} raw conversations")
    for d in data:
        convs = d.get("conversations", [])
        msgs = []
        for m in convs:
            role = m.get("from", m.get("role", ""))
            content = m.get("value", m.get("content", ""))
            if content is None:
                content = ""
            msgs.append({"role": role, "content": content})
        yield msgs, None  # sharegpt has no reliable language tag


def iter_arena(dataset: str, conv_field: str):
    from datasets import load_dataset

    log(f"[load] {dataset} (train) ...")
    ds = load_dataset(dataset, split="train")
    log(f"[load] {len(ds)} rows")
    for row in ds:
        msgs = [
            {"role": m.get("role", ""), "content": m.get("content", "") or ""}
            for m in row.get(conv_field, [])
        ]
        yield msgs, row.get("language", None)


def emit_rows_for_conv(msgs, tok, emit_from_round: int):
    """Exact-tokenize: (content, prefix_tok, content_tok) for each round k>=emit_from_round."""
    rows = []
    round_idx = 0
    for i, m in enumerate(msgs):
        if not is_user(m["role"]):
            continue
        round_idx += 1
        if round_idx < emit_from_round:
            continue
        content = serialize(msgs[: i + 1])
        prefix_text = serialize(msgs[:i])
        rows.append((content, ntok(tok, prefix_text), ntok(tok, content)))
    return rows


def default_sharegpt_path() -> Optional[str]:
    pats = [
        os.path.expanduser(
            "~/.cache/huggingface/hub/datasets--anon8231489123--"
            "ShareGPT_Vicuna_unfiltered/snapshots/*/ShareGPT_V3_*.json"
        ),
    ]
    for p in pats:
        hits = glob.glob(p)
        if hits:
            return hits[0]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-groups", type=int, default=40)
    ap.add_argument("--source", choices=["sharegpt", "arena"], default="sharegpt")
    ap.add_argument(
        "--sharegpt-json",
        default=None,
        help="local ShareGPT V3 json (auto-detected from HF cache if unset)",
    )
    ap.add_argument("--min-msgs", type=int, default=4, help=">=4 => >=2 rounds")
    ap.add_argument("--emit-from-round", type=int, default=2)
    ap.add_argument(
        "--language",
        default="English",
        help="arena: exact tag; sharegpt: ASCII heuristic",
    )
    ap.add_argument(
        "--tokenizer", default=os.path.join(os.getcwd(), "Kimi-Linear-48B-A3B-Instruct")
    )
    ap.add_argument("--dataset", default="lmsys/chatbot_arena_conversations")
    ap.add_argument("--conv-field", default="conversation_a")
    ap.add_argument(
        "--target-prefix-tok",
        type=int,
        default=2048,
        help="selection target; groups need approx max-prefix >= min-group-prefix-tok",
    )
    ap.add_argument(
        "--min-group-prefix-tok",
        type=int,
        default=None,
        help="min approx max-prefix per selected group (default 0.75*target)",
    )
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.min_group_prefix_tok is None:
        args.min_group_prefix_tok = int(0.75 * args.target_prefix_tok)

    out = args.out or f"/tmp/arena_shared_prefix_N{args.n_groups}.jsonl"
    stats_path = out.replace(".jsonl", "") + ".stats.json"

    tok = load_tokenizer(args.tokenizer)

    if args.source == "sharegpt":
        sp = args.sharegpt_json or default_sharegpt_path()
        if not sp or not os.path.isfile(sp):
            log("[fatal] no ShareGPT json found; pass --sharegpt-json PATH")
            sys.exit(2)
        src = iter_sharegpt(sp)
        src_name = f"sharegpt_v3:{os.path.basename(sp)}"
    else:
        src = iter_arena(args.dataset, args.conv_field)
        src_name = f"{args.dataset}:{args.conv_field}"

    # --- pass 1: cheap char-based candidate filter/rank (no tokenizer) ---
    APPROX = 4  # chars per token
    n_seen = n_lang = n_multiturn = 0
    candidates = []  # (approx_maxprefix, msgs)
    for msgs, lang in src:
        n_seen += 1
        if not msgs or len(msgs) < args.min_msgs:
            continue
        # language filter
        if args.source == "arena":
            if args.language and lang != args.language:
                continue
        else:  # sharegpt heuristic on first user turn
            first_user = next((m["content"] for m in msgs if is_user(m["role"])), "")
            if args.language and not looks_english(first_user):
                continue
        n_lang += 1
        n_user = sum(1 for m in msgs if is_user(m["role"]))
        if n_user < 2:
            continue
        n_multiturn += 1
        # approx of the LARGEST *shared prefix* in this group = history before the
        # final user turn (excludes the trailing question). Ranking on this (not the
        # full conversation) rejects convs whose bulk sits in the last turn -> keeps
        # only groups with a genuinely long reusable prefix.
        last_user = max(i for i, m in enumerate(msgs) if is_user(m["role"]))
        approx_prefix = len(serialize(msgs[:last_user])) // APPROX
        if approx_prefix < args.min_group_prefix_tok:
            continue
        candidates.append((approx_prefix, msgs))

    log(
        f"[pass1] seen={n_seen} lang_ok={n_lang} multiturn={n_multiturn} "
        f"candidates(approx_maxprefix>={args.min_group_prefix_tok})={len(candidates)}"
    )

    if not candidates:
        log("[GATE-0] NO qualifying candidate groups -> workload boundary. STOP.")
        json.dump(
            {"source": src_name, "qualifying_candidates": 0}, open(stats_path, "w")
        )
        sys.exit(3)

    # rank by approx length desc, take up to N
    candidates.sort(key=lambda x: x[0], reverse=True)
    picked = candidates[: args.n_groups]
    if len(picked) < args.n_groups:
        log(f"[WARN] only {len(picked)} candidate groups (< requested {args.n_groups})")

    # --- pass 2: exact tokenize ONLY selected groups; emit jsonl ---
    all_prefix_tok, all_content_tok, reuse_counts, group_maxpref = [], [], [], []
    n_rows = 0
    with open(out, "w", encoding="utf-8") as f:
        for _, msgs in picked:
            rows = emit_rows_for_conv(msgs, tok, args.emit_from_round)
            if not rows:
                continue
            reuse_counts.append(len(rows))
            group_maxpref.append(max(pt for _, pt, _ in rows))
            for content, pt, ct in rows:
                all_prefix_tok.append(pt)
                all_content_tok.append(ct)
                f.write(
                    json.dumps(
                        {
                            "conversations": [
                                {"role": "user", "content": content},
                                {"role": "assistant", "content": "OK."},
                            ]
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n_rows += 1

    distinct_prefix_est = n_rows  # each nested row is a distinct prefix checkpoint

    def dist(xs):
        xs = [float(x) for x in xs]
        return {
            "mean": round(statistics.fmean(xs), 1) if xs else 0,
            "median": round(statistics.median(xs), 1) if xs else 0,
            "p90": round(pct(xs, 0.90), 1),
            "p99": round(pct(xs, 0.99), 1),
            "min": round(min(xs), 1) if xs else 0,
            "max": round(max(xs), 1) if xs else 0,
        }

    stats = {
        "source": src_name,
        "tokenizer": args.tokenizer,
        "language_filter": args.language,
        "seen": n_seen,
        "lang_ok": n_lang,
        "multiturn": n_multiturn,
        "candidates": len(candidates),
        "selected_groups": len(reuse_counts),
        "total_requests": n_rows,
        "target_prefix_tok": args.target_prefix_tok,
        "min_group_prefix_tok": args.min_group_prefix_tok,
        "reuse_per_group": dist(reuse_counts),
        "prefix_tok": dist(all_prefix_tok),
        "content_tok": dist(all_content_tok),
        "group_max_prefix_tok": dist(group_maxpref),
        "distinct_prefix_checkpoints_est": distinct_prefix_est,
        "dense_pool_saturation_0.7x": round(0.7 * distinct_prefix_est),
        "out": out,
    }
    json.dump(stats, open(stats_path, "w"), indent=2)

    # --- GATE-0 summary ---
    log("")
    log("=" * 68)
    log(f"GATE-0 WORKLOAD HEALTH  ({src_name})")
    log("=" * 68)
    log(f"  seen / lang_ok / multiturn : {n_seen} / {n_lang} / {n_multiturn}")
    log(
        f"  candidate groups           : {len(candidates)}  (selected {len(reuse_counts)}/{args.n_groups})"
    )
    log(f"  total requests emitted     : {n_rows}")
    log(
        f"  reuse per group            : mean={stats['reuse_per_group']['mean']} "
        f"median={stats['reuse_per_group']['median']} "
        f"min={stats['reuse_per_group']['min']} max={stats['reuse_per_group']['max']}"
    )
    log(
        f"  PREFIX tok (shared, exact) : mean={stats['prefix_tok']['mean']} "
        f"median={stats['prefix_tok']['median']} p90={stats['prefix_tok']['p90']} "
        f"max={stats['prefix_tok']['max']}"
    )
    log(
        f"  content tok (full req)     : mean={stats['content_tok']['mean']} "
        f"median={stats['content_tok']['median']} p90={stats['content_tok']['p90']}"
    )
    log(
        f"  group max-prefix tok       : median={stats['group_max_prefix_tok']['median']} "
        f"min={stats['group_max_prefix_tok']['min']}"
    )
    log(
        f"  distinct ckpt est          : {distinct_prefix_est}  "
        f"(dense saturate ~0.7x = {stats['dense_pool_saturation_0.7x']})"
    )
    log("-" * 68)
    ok_groups = len(reuse_counts) >= args.n_groups
    ok_prefix = stats["prefix_tok"]["median"] >= 0.5 * args.target_prefix_tok
    ok_reuse = stats["reuse_per_group"]["median"] > 1
    verdict = "PASS" if (ok_groups and ok_prefix and ok_reuse) else "MARGINAL/FAIL"
    log(
        f"  GATE-0: groups>={args.n_groups}:{ok_groups}  "
        f"prefix_median>={0.5*args.target_prefix_tok:.0f}:{ok_prefix}  "
        f"reuse>1:{ok_reuse}  => {verdict}"
    )
    log("=" * 68)
    log(f"[out]   {out}")
    log(f"[stats] {stats_path}")
    print(out)


if __name__ == "__main__":
    main()
