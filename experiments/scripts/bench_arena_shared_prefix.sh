#!/usr/bin/env bash
# REAL-TRACE serving A/B (replaces the SYNTHETIC gsp workload): dense vs DASC-padded
# vs DASC-ragged on a REAL multi-turn shared-prefix workload built from ShareGPT V3
# (experiments/scripts/build_arena_shared_prefix.py). One BACKEND x MODE arm per launch.
#
#   Usage: bash experiments/scripts/bench_arena_shared_prefix.sh <kda|gdn> <dense|padded|ragged>
#   Env:   NGROUPS=40|100  (selects /tmp/arena_shared_prefix_N${NGROUPS}.jsonl)
#
# WHY real-trace (not gsp): the synthetic generated-shared-prefix load manufactures
# N equally-hot distinct long prefixes; reviewers flagged the DASC gain as "synthetic
# only". Here each request's input = (prior turns of a real conversation) + new user
# turn, so round k's content is a strict char-prefix of round k+1's -> genuine nested
# shared-prefix reuse, no single global hot prefix. If the chain still holds here it is
# real-trace evidence; if it is null we report that honestly (spec section 6).
#
# Three arms (ALL NORECON: hits free, accuracy-neutral on hybrid models; SAME lru
# eviction -> isolates the pure capacity effect, not Idea4):
#   dense  : W_max=0 -> full-column checkpoint (big bytes/slot). SMALL pool (< distinct
#            prefixes) -> saturates -> thrash-miss on the reuse tail.
#   padded : route A, W_max=16(KDA)/64(GDN), RAGGED=0 (cross-layer max-pad). BIG pool.
#   ragged : same but RAGGED=1 (per-layer widths, tighter bytes/slot -> more slots/HBM).
# HBM note: dense pool is sized to SATURATE (0.7x distinct); DASC pools HOLD-ALL at a
# FRACTION of dense HBM (bytes/slot is ~capacity_x smaller) -> DASC dominates at <= dense
# HBM. Each arm's measured ckpt bytes/slot + HBM are recorded so the table shows parity.
# For an EXACT equal-HBM pass, override DENSE_CKPT / DASC_CKPT.
set -euo pipefail

export no_proxy='127.0.0.1,localhost' NO_PROXY='127.0.0.1,localhost'
# Respect caller-provided proxy variables; do not inject site-specific defaults.
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK="${SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK:-1}"

# Explicitly clear head-aware env that must NOT leak from the parent shell (the
# w_max-sweep isolates one knob per launch; random_drop / quant would confound).
unset SGLANG_HEAD_AWARE_RANDOM_DROP SGLANG_HEAD_AWARE_RANDOM_SEED SGLANG_STATE_QUANT_MODE SGLANG_STATE_QUANT_GROUP_SIZE

BACKEND="${1:?usage: $0 <kda|gdn> <dense|padded|ragged|int8>}"
MODE="${2:?usage: $0 <kda|gdn> <dense|padded|ragged|int8>}"
case "$BACKEND" in kda|gdn) ;; *) echo "backend must be kda|gdn"; exit 2;; esac
case "$MODE" in dense|padded|ragged|int8|int8fair) ;; *) echo "mode must be dense|padded|ragged|int8|int8fair"; exit 2;; esac

NGROUPS="${NGROUPS:-40}"
DATASET_PATH="${DATASET_PATH:-/tmp/arena_shared_prefix_N${NGROUPS}.jsonl}"
STATS_PATH="${DATASET_PATH%.jsonl}.stats.json"
[ -f "$DATASET_PATH" ] || { echo "[fatal] dataset $DATASET_PATH missing; run build_arena_shared_prefix.py --n-groups $NGROUPS"; exit 3; }

# per-backend model / attention backend / wmax / mem
if [ "$BACKEND" = kda ]; then
  MODEL="${MODEL:-${SGLANG_REPO:-$PWD}/Kimi-Linear-48B-A3B-Instruct}"
  BACKEND_FLAGS="${BACKEND_FLAGS:---linear-attn-backend triton}"
  WMAX_DASC="${WMAX_DASC:-16}"
  MEM_FRAC="${MEM_FRAC:-0.85}"
else
  MODEL="${MODEL:-${SGLANG_REPO:-$PWD}/Qwen3-Next-80B-A3B-Instruct-NVFP4}"
  BACKEND_FLAGS="${BACKEND_FLAGS:---attention-backend triton}"
  WMAX_DASC="${WMAX_DASC:-64}"
  MEM_FRAC="${MEM_FRAC:-0.7}"
fi

PORT="${PORT:-31007}"; HOST=127.0.0.1
TP="${TP:-2}"
MAX_CONC="${MAX_CONC:-16}"
OUTPUT_LEN="${OUTPUT_LEN:-64}"

# distinct prefix checkpoints (each nested row = one checkpoint) drives pool sizing.
read -r DISTINCT NUM_PROMPTS < <(python - "$STATS_PATH" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(d.get("distinct_prefix_checkpoints_est", d.get("total_requests",256)),
      d.get("total_requests",256))
PY
)
NUM_PROMPTS="${NUM_PROMPTS_OVERRIDE:-$NUM_PROMPTS}"
# dense: SATURATE (>conc in-flight, <distinct so it evicts+misses). DASC: HOLD-ALL.
DENSE_CKPT_DEFAULT=$(python - "$DISTINCT" "$MAX_CONC" <<'PY'
import sys; dist,conc=int(sys.argv[1]),int(sys.argv[2])
print(max(conc+16, round(0.7*dist)))
PY
)
DASC_CKPT_DEFAULT=$(python - "$DISTINCT" <<'PY'
import sys; dist=int(sys.argv[1]); print(round(1.3*dist)+16)
PY
)

if [ "$MODE" = dense ]; then
  CKPT="${DENSE_CKPT:-$DENSE_CKPT_DEFAULT}"; WMAX=0; RAGGED=0
  declare -a MODE_ENV=(SGLANG_FORCE_HEAD_AWARE_WMAX=0)
elif [ "$MODE" = int8 ]; then
  CKPT="${CKPT_OVERRIDE:-$DENSE_CKPT_DEFAULT}"; WMAX=0; RAGGED=0
  declare -a MODE_ENV=()
elif [ "$MODE" = int8fair ]; then
  # FAIR INT8: head-aware pool (32 heads/rank, SAME TP layout as dense/ragged)
  # + SGLANG_STATE_QUANT_MODE=int8 (uniform INT8 quantization, W=0 all-global).
  # Unlike the "int8" mode (which uses the separate INT8 pool storing 16 heads/rank
  # TP-split -> 2x layout advantage), this uses the SAME head-aware pool as
  # dense/ragged, only swapping BF16->INT8 precision. Fair equal-HBM comparison.
  CKPT="${CKPT_OVERRIDE:-$DENSE_CKPT_DEFAULT}"; WMAX=0; RAGGED=1
  declare -a MODE_ENV=(SGLANG_FORCE_HEAD_AWARE_WMAX=0
                       SGLANG_STATE_QUANT_MODE=int8
                       SGLANG_HEAD_AWARE_RAGGED=1)
else
  CKPT="${DASC_CKPT:-$DASC_CKPT_DEFAULT}"; WMAX="$WMAX_DASC"
  RAGGED=$([ "$MODE" = ragged ] && echo 1 || echo 0)
  declare -a MODE_ENV=(SGLANG_FORCE_HEAD_AWARE_WMAX="$WMAX"
                       SGLANG_ENABLE_HEAD_AWARE_REPREFILL="${RECON:-0}"
                       SGLANG_HEAD_AWARE_RAGGED="$RAGGED")
fi

# EQUAL-HBM: force an EXACT per-arm slot count (padded/ragged have different
# bytes/slot so a single DASC_CKPT can't set both). Unset -> unchanged.
CKPT="${CKPT_OVERRIDE:-$CKPT}"

LOGDIR="${LOGDIR:-$PWD/experiments/results/arena_ab}"
mkdir -p "$LOGDIR"
LABEL="${BACKEND}_${MODE}_N${NGROUPS}_w${WMAX}_ckpt${CKPT}_recon${RECON:-0}"
SRVLOG="$LOGDIR/server_${LABEL}.log"
RESJSON="$LOGDIR/result_${LABEL}.json"

if [ "$MODE" = int8 ]; then
  CHECKPOINT_FLAGS=(--enable-int8-mamba-checkpoint --int8-mamba-ckpt-size "$CKPT")
else
  CHECKPOINT_FLAGS=(--enable-head-aware-mamba-checkpoint --head-aware-route A
                    --head-aware-mamba-ckpt-size "$CKPT")
fi

COMMON=(--model-path "$MODEL" --trust-remote-code --tp "$TP"
        --host "$HOST" --port "$PORT" --mem-fraction-static "$MEM_FRAC"
        --mamba-radix-cache-strategy extra_buffer
        $BACKEND_FLAGS
        --chunked-prefill-size 2048
        --disable-overlap-schedule
        --enable-metrics
        --max-running-requests "$(( MAX_CONC + 16 ))"
        "${CHECKPOINT_FLAGS[@]}")

echo "[launch] backend=$BACKEND mode=$MODE N=$NGROUPS distinct=$DISTINCT ckpt=$CKPT wmax=$WMAX ragged=$RAGGED reqs=$NUM_PROMPTS -> $SRVLOG"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
  env "${MODE_ENV[@]}" SGLANG_MAMBA_EVICT_POLICY="${EVICT:-lru}" \
  python -m sglang.launch_server "${COMMON[@]}" > "$SRVLOG" 2>&1 &
SRVPID=$!
trap 'kill $SRVPID 2>/dev/null || true; wait $SRVPID 2>/dev/null || true' EXIT

echo "[wait] server pid=$SRVPID warming up..."
for i in $(seq 1 360); do
  if curl -s "http://$HOST:$PORT/health_generate" >/dev/null 2>&1; then
    echo "[ready] after ${i}0s"; break
  fi
  if ! kill -0 $SRVPID 2>/dev/null; then
    echo "[fatal] server died; tail:"; tail -80 "$SRVLOG"; exit 1
  fi
  sleep 10
done
curl -s "http://$HOST:$PORT/health_generate" >/dev/null 2>&1 || { echo "[fatal] server never became ready"; tail -80 "$SRVLOG"; exit 1; }

# Extract ALL TP ranks' bytes/slot and capacity, then average.
# Fix: tail -1 picked one TP randomly (race condition on log order), causing NR/WR
# to report different bps despite having the same plan (plan is W-only dependent).
read -r BYTES CAP HBM_MB < <(python3 - "$SRVLOG" "$CKPT" <<'PY'
import re, sys
log_path, ckpt = sys.argv[1], int(sys.argv[2])
bps_vals, cap_vals = [], []
GB = 1 << 30
DENSE_BPS = 20971520  # KDA dense; used as fallback for cap ratio
for line in open(log_path):
    low = line.lower()
    if "x capacity" in low:
        m = re.search(r"bytes/slot=(\d+)", line)
        if m: bps_vals.append(int(m.group(1)))
        m = re.search(r"([\d.]+)x capacity", line)
        if m: cap_vals.append(float(m.group(1)))
    elif "int8 mamba checkpoint pool:" in low:
        # Format: "int8 mamba checkpoint pool: N slots, XGB (qdata A + scale B + conv C)"
        m = re.search(r"(\d+) slots, [\d.]+GB \(qdata ([\d.]+) \+ scale ([\d.]+)", line)
        if m:
            slots = int(m.group(1))
            qdata_gb = float(m.group(2))
            scale_gb = float(m.group(3))
            # bytes_per_slot = (qdata + scale) GB / slots (state only, no conv)
            bps = int((qdata_gb + scale_gb) * GB / slots)
            bps_vals.append(bps)
            cap_vals.append(DENSE_BPS / max(1, bps))
bps = sum(bps_vals) // len(bps_vals) if bps_vals else 0
cap = sum(cap_vals) / len(cap_vals) if cap_vals else 0.0
hbm = ckpt * bps / 1048576
print(bps, f"{cap:.2f}", f"{hbm:.1f}")
PY
)
CAPLINE=$(grep -iE "x capacity" "$SRVLOG" | head -1 || true)
NTP=$(grep -ciE "x capacity" "$SRVLOG" 2>/dev/null || echo 0)
echo "[capacity] backend=$BACKEND mode=$MODE cap=${CAP:-NA}x bytes/slot=${BYTES:-NA} slots=$CKPT ckpt_HBM=${HBM_MB}MB (avg of ${NTP} TP ranks) ($CAPLINE)"

# custom dataset: each jsonl line's conversations[0].content = the full nested prefix
# text; the loader shuffles + reads only that -> real cross-request prefix reuse.
run_pass () {  # $1=pass $2=resjson-or-empty $3=benchlog $4=seed-or-empty
  python -m sglang.benchmark.serving \
    --backend sglang --host "$HOST" --port "$PORT" --model "$MODEL" \
    --dataset-name custom --dataset-path "$DATASET_PATH" \
    --sharegpt-output-len "$OUTPUT_LEN" --apply-chat-template \
    --num-prompts "$NUM_PROMPTS" --max-concurrency "$MAX_CONC" \
    $([ -n "$2" ] && echo --output-file "$2") \
    $([ -n "${4:-}" ] && echo --seed "$4") \
    2>&1 | tee "$3"
}

# Compute token hit-rate over the server-log window since $2 and inject the
# capacity/hit/seed/peak-mem/recon metadata into the per-pass result JSON $1.
finalize_pass () {  # $1=resjson $2=mark $3=seed-or-empty
  local rj="$1" mk="$2" seed="${3:-}"
  local cached new hit
  cached=$(tail -n +"$mk" "$SRVLOG" | grep -Eo "#cached-token: [0-9]+" | awk '{s+=$2} END{print s+0}')
  new=$(tail -n +"$mk" "$SRVLOG"    | grep -Eo "#new-token: [0-9]+"    | awk '{s+=$2} END{print s+0}')
  hit=$(awk -v c="$cached" -v n="$new" 'BEGIN{printf (c+n>0)?"%.4f":"NA",(c+n>0)?c/(c+n):0}')
  echo "[hit] backend=$BACKEND mode=$MODE seed=${seed:-NA} cached=$cached new=$new token_hit_rate=$hit"
  # Extract peak memory (GiB, rank-0 post-reset) + Route-A recon counters from
  # /server_info. Gated: NA if the server doesn't expose them (older build / the
  # reset endpoint wasn't called). peak_allocated is the measure-window peak
  # AFTER /reset_peak_memory (excludes model-load / warm-up spike).
  local peak_mem recon_n recon_mean recon_p95 recon_max
  IFS=' ' read -r peak_mem recon_n recon_mean recon_p95 recon_max <<EOF
$(curl -s --max-time 15 "http://$HOST:$PORT/server_info" 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
    mu=d['internal_states'][0]['memory_usage']
    rc=d['internal_states'][0].get('recon_counters',{}) or {}
    print(mu.get('peak_allocated','NA'), rc.get('n_recon_hits','NA'), rc.get('replay_tokens_mean','NA'), rc.get('replay_tokens_p95','NA'), rc.get('replay_tokens_max','NA'))
except Exception:
    print('NA NA NA NA NA')
" 2>/dev/null)
EOF
  python - "$rj" "$BACKEND" "$MODE" "$NGROUPS" "$WMAX" "$RAGGED" "$hit" "$cached" "$new" "${CAP:-0}" "${BYTES:-0}" "$CKPT" "$HBM_MB" "$DISTINCT" "${seed:-NA}" "${peak_mem:-NA}" "${recon_n:-NA}" "${recon_mean:-NA}" "${recon_p95:-NA}" "${recon_max:-NA}" <<'PY'
import json,sys
(p,backend,mode,ng,wmax,ragged,hit,cached,new,cap,b,ckpt,hbm,distinct,seed,
 peak_mem,recon_n,recon_mean,recon_p95,recon_max)=sys.argv[1:21]
def _f(x): return None if x=="NA" else float(x)
def _i(x): return None if x=="NA" else int(x)
d=json.load(open(p))
d.update(_backend=backend,_dasc_mode=mode,_ngroups=int(ng),_wmax=int(wmax),
         _ragged=int(ragged),_source="sharegpt_v3_shared_prefix",
         _token_hit_rate=None if hit=="NA" else float(hit),
         _cached_tokens=int(cached),_new_tokens=int(new),
         _capacity_x=float(cap),_bytes_per_slot=int(b),_ckpt_slots=int(ckpt),
         _ckpt_hbm_mb=float(hbm),_distinct_prefix_est=int(distinct),
         _seed=None if seed=="NA" else int(seed),
         _peak_mem_gb=_f(peak_mem),          # rank-0 measure-window peak (GiB)
         _recon_n_hits=_i(recon_n),           # Route-A reconstruction hits this pass
         _replay_tokens_mean=_f(recon_mean),  # actual replay=min(W_max,P) per hit
         _replay_tokens_p95=_f(recon_p95),
         _replay_tokens_max=_i(recon_max),
         _recon=(mode!="dense" and int(ckpt)!=0 and _i(recon_n) is not None and _i(recon_n)>0))
json.dump(d,open(p,"w"))
print("[result] %s/%s N%s seed=%s: hit=%s ttft_mean=%.1f p50=%.1f p95=%.1f p99=%.1f thr=%.1f e2e=%.1f in_tok=%d ckpt=%s HBM=%.1fMB cap=%sx peak=%s recon_hits=%s replay_mean=%s"%(
  backend,mode,ng,seed,hit,d["mean_ttft_ms"],d["median_ttft_ms"],
  d.get("p95_ttft_ms",d.get("p99_ttft_ms",0)),d["p99_ttft_ms"],
  d["output_throughput"],d["mean_e2e_latency_ms"],d["total_input_tokens"],ckpt,float(hbm),cap,
  peak_mem,recon_n,recon_mean))
PY
}

echo "[flush]"; curl -s -X POST "http://$HOST:$PORT/flush_cache" >/dev/null 2>&1 || true; sleep 2
echo "[warm] backend=$BACKEND mode=$MODE"
run_pass warm "" "$LOGDIR/bench_${LABEL}_warm.log" ""

if [ -z "${MEASURE_SEEDS:-}" ]; then
  # ORIGINAL single-measure path (arena_ab behavior byte-identical).
  curl -s --max-time 15 -X POST "http://$HOST:$PORT/reset_peak_memory" >/dev/null 2>&1 || true
  MARK=$(wc -l < "$SRVLOG")
  echo "[measure] backend=$BACKEND mode=$MODE"
  run_pass measure "$RESJSON" "$LOGDIR/bench_${LABEL}_measure.log" ""
  finalize_pass "$RESJSON" "$MARK" ""
  echo "[done] backend=$BACKEND mode=$MODE -> $RESJSON"
else
  # EQUAL-HBM: repeat measure over each seed on the warm pool (no re-flush ->
  # steady-state; same request set reshuffled -> clean scheduling/eviction variance).
  IFS=',' read -ra SEEDS <<< "$MEASURE_SEEDS"
  for SEED in "${SEEDS[@]}"; do
    SEED="$(echo "$SEED" | tr -d ' ')"
    [ -z "$SEED" ] && continue
    SRJSON="${RESJSON%.json}_seed${SEED}.json"
    curl -s --max-time 15 -X POST "http://$HOST:$PORT/reset_peak_memory" >/dev/null 2>&1 || true
    MARK=$(wc -l < "$SRVLOG")
    echo "[measure] backend=$BACKEND mode=$MODE seed=$SEED"
    run_pass "measure_s${SEED}" "$SRJSON" "$LOGDIR/bench_${LABEL}_measure_s${SEED}.log" "$SEED"
    finalize_pass "$SRJSON" "$MARK" "$SEED"
  done
  echo "[done] backend=$BACKEND mode=$MODE (seeds=$MEASURE_SEEDS) -> ${RESJSON%.json}_seed*.json"
fi
