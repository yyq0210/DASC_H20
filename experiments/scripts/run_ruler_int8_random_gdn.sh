#!/usr/bin/env bash
# RULER INT8 + 5-seed random drop + GDN rerun (MAXNEW=256 fixes truncation).
# All 13 tasks x 3 lengths (4k/8k/16k) x 4 tries.
#
# Phase 1: KDA INT8 (1 launch)
#   W=0 all-global + SGLANG_STATE_QUANT_MODE=int8 + RAGGED=1
#   Tests uniform INT8 quantization (FP32->INT8, 4x) accuracy on all RULER tasks.
#
# Phase 2: KDA 5-seed random drop (5 W x 5 seeds = 25 launches)
#   SGLANG_HEAD_AWARE_RANDOM_DROP=1 with 5 seeds at each W.
#   Same #local per layer (same compression) but randomized positions.
#   Tests whether decay-aware selection beats random.
#
# Phase 3: GDN rerun with MAXNEW=256 (dense + 5 W = 6 launches)
#   Fixes the MAXNEW=64 truncation that broke CWE/MK3/VT/FWE/QA.
#   Qwen3-Next-80B NVFP4, --attention-backend triton, mem-frac 0.90.
#
# Usage:  bash experiments/scripts/run_ruler_int8_random_gdn.sh
# Env:    RUN_KDA_INT8=1 RUN_KDA_RANDOM=1 RUN_GDN=1  (all default 1)
set -uo pipefail
cd "$(dirname "$0")/../.."

export no_proxy='127.0.0.1,localhost' NO_PROXY='127.0.0.1,localhost'
# Respect caller-provided proxy variables; do not inject site-specific defaults.
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

# Prevent env leakage from parent shell
unset SGLANG_STATE_QUANT_MODE SGLANG_HEAD_AWARE_RANDOM_DROP \
      SGLANG_HEAD_AWARE_RANDOM_SEED SGLANG_ENABLE_HEAD_AWARE_REPREFILL \
      SGLANG_HEAD_AWARE_RAGGED SGLANG_FORCE_HEAD_AWARE_WMAX

# ---- shared config ----
RULER_TASKS="${RULER_TASKS:-s1 s2 s3 mk1 mk2 mk3 mq mv cwe fwe vt qa_h qa_s}"
LENGTHS="${LENGTHS:-4k 8k 16k}"
NUM_SAMPLES="${NUM_SAMPLES:-30}"
NUM_TRIES="${NUM_TRIES:-4}"
TEMP="${TEMP:-0.0}"
MAXNEW="${MAXNEW:-256}"
WMAX_LIST="${WMAX_LIST:-16 64 128 512 1024}"
SEEDS="${SEEDS:-42 123 7 555 31337}"
FREE_GB="${FREE_GB:-40}"
CKPT_SIZE="${CKPT_SIZE:-200}"
PORT="${PORT:-31007}"; HOST=127.0.0.1
TP="${TP:-2}"

# ---- KDA config ----
KDA_MODEL="${KDA_MODEL:-${SGLANG_REPO:-$PWD}/Kimi-Linear-48B-A3B-Instruct}"
KDA_BACKEND_FLAGS="--linear-attn-backend triton"
KDA_MEM_FRAC="0.85"
KDA_PARALLEL=8

# ---- GDN config ----
GDN_MODEL="${GDN_MODEL:-${SGLANG_REPO:-$PWD}/Qwen3-Next-80B-A3B-Instruct-NVFP4}"
GDN_BACKEND_FLAGS="--attention-backend triton"
GDN_MEM_FRAC="0.90"
GDN_PARALLEL=4
GDN_MAX_RUNNING=4

# ---- dirs ----
KDA_LOGDIR="$PWD/experiments/results/ruler_int8_random"
GDN_LOGDIR="$PWD/experiments/results/ruler_gdn_rerun_256"
mkdir -p "$KDA_LOGDIR" "$GDN_LOGDIR"
ORCH="$KDA_LOGDIR/orchestrator_$(date +%Y%m%d_%H%M%S).log"

RUN_KDA_INT8="${RUN_KDA_INT8:-1}"
RUN_KDA_RANDOM="${RUN_KDA_RANDOM:-1}"
RUN_GDN_INT8="${RUN_GDN_INT8:-1}"
RUN_GDN_RANDOM="${RUN_GDN_RANDOM:-1}"
RUN_GDN="${RUN_GDN:-1}"

log(){ echo "[$(date +%H:%M:%S)] $*" | tee -a "$ORCH"; }

wait_gpu_free(){
  local need=$(( FREE_GB * 1024 ))
  for _ in $(seq 1 120); do
    local minfree
    minfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
    [ -z "$minfree" ] && { sleep 5; continue; }
    if [ "$minfree" -ge "$need" ]; then
      log "  gpu free ok (min ${minfree}MiB >= ${need}MiB)"; return 0
    fi
    sleep 10
  done
  log "  WARN: gpu never freed; proceeding anyway"; return 0
}

# run_arm <backend> <tag> <wmax> <logdir> [extra env KEY=VAL ...]
run_arm () {
  local BK="$1" TAG="$2" WMAX="$3" LOGDIR="$4"
  shift 4
  local -a EXTRA_ENV=("$@")

  local MODEL MEM_FRAC BACKEND_FLAGS PARALLEL
  local -a MAX_RUNNING_FLAG=()
  if [ "$BK" = kda ]; then
    MODEL="$KDA_MODEL"; MEM_FRAC="$KDA_MEM_FRAC"
    BACKEND_FLAGS="$KDA_BACKEND_FLAGS"; PARALLEL="$KDA_PARALLEL"
  else
    MODEL="$GDN_MODEL"; MEM_FRAC="$GDN_MEM_FRAC"
    BACKEND_FLAGS="$GDN_BACKEND_FLAGS"; PARALLEL="$GDN_PARALLEL"
    MAX_RUNNING_FLAG=(--max-running-requests "$GDN_MAX_RUNNING")
  fi

  local RESULT_FILE="$LOGDIR/acc_results.jsonl"
  local SUMMARY="$LOGDIR/summary.txt"
  touch "$SUMMARY"
  local SRVLOG="$LOGDIR/server_${TAG}.log"

  local -a MODE_ENV=("SGLANG_FORCE_HEAD_AWARE_WMAX=$WMAX")
  MODE_ENV+=("${EXTRA_ENV[@]}")

  log "===== [$TAG] $BK wmax=$WMAX ====="
  wait_gpu_free
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
    env "${MODE_ENV[@]}" SGLANG_MAMBA_EVICT_POLICY="lru" \
    python -m sglang.launch_server \
      --model-path "$MODEL" --trust-remote-code --tp "$TP" \
      --host "$HOST" --port "$PORT" --mem-fraction-static "$MEM_FRAC" \
      --mamba-radix-cache-strategy extra_buffer \
      $BACKEND_FLAGS --chunked-prefill-size 2048 \
      --disable-overlap-schedule --enable-metrics \
      "${MAX_RUNNING_FLAG[@]}" \
      --enable-head-aware-mamba-checkpoint --head-aware-route A \
      --head-aware-mamba-ckpt-size "$CKPT_SIZE" > "$SRVLOG" 2>&1 &
  local SRVPID=$!
  trap 'kill $SRVPID 2>/dev/null || true; wait $SRVPID 2>/dev/null || true' RETURN

  local ok=0
  for i in $(seq 1 600); do
    if curl -s "http://$HOST:$PORT/health_generate" >/dev/null 2>&1; then
      log "[ready] $TAG after ${i}0s"; ok=1; break
    fi
    if ! kill -0 $SRVPID 2>/dev/null; then
      log "[fatal] $TAG server died; tail:"; tail -60 "$SRVLOG" | tee -a "$ORCH"; break
    fi
    sleep 10
  done
  [ "$ok" = 1 ] || { log "[skip] $TAG did not come up"; return 1; }

  grep -iE "head-aware mamba checkpoint pool|x capacity|ragged-diag" "$SRVLOG" | tail -4 || true

  for TASK in $RULER_TASKS; do
    for LEN in $LENGTHS; do
      local NS="$NUM_SAMPLES"
      local ARM="${TAG}_${TASK}_${LEN}"
      # resume-skip
      if [ -f "$RESULT_FILE" ] && grep -q "\"arm\": \"${ARM}\"" "$RESULT_FILE" 2>/dev/null; then
        log "[skip-done] $ARM"; continue
      fi
      local MARK; MARK=$(wc -l < "$SRVLOG")
      log "[bench] $ARM nsamp=$NS tries=$NUM_TRIES"
      { python "$PWD/experiments/scripts/bench_idea1_accuracy.py" \
          --task ruler --arm "$ARM" --wmax "$WMAX" \
          --ruler-tasks "$TASK" --ruler-length "$LEN" \
          --ruler-num-samples "$NS" \
          --num-tries "$NUM_TRIES" --parallel "$PARALLEL" \
          --temperature "$TEMP" --top-p 1.0 --top-k -1 \
          --presence-penalty 0.0 --max-new-tokens "$MAXNEW" \
          --host "$HOST" --port "$PORT" \
          --result-file "$RESULT_FILE" \
          --output-file "$LOGDIR/out_${ARM}.jsonl" \
          2>&1 | tee "$LOGDIR/bench_${ARM}.log"; } \
        || { log "[bench-fail] $ARM — skipping"; }

      local CACHED NEW
      CACHED=$(tail -n +"$MARK" "$SRVLOG" | grep -Eo "#cached-token: [0-9]+" | awk '{s+=$2} END{print s+0}')
      NEW=$(tail -n +"$MARK" "$SRVLOG" | grep -Eo "#new-token: [0-9]+" | awk '{s+=$2} END{print s+0}')
      log "[radix] $ARM cached=$CACHED new=$NEW"
      echo "${ARM} cached=$CACHED new=$NEW" >> "$SUMMARY"
    done
  done

  log "[shutdown] $TAG"
  kill $SRVPID 2>/dev/null || true
  wait $SRVPID 2>/dev/null || true
  sleep 5
}

log "=== RULER INT8 + random + GDN rerun START ==="
log "tasks='$RULER_TASKS' lengths='$LENGTHS' nsamp=$NUM_SAMPLES tries=$NUM_TRIES maxnew=$MAXNEW"
log "wmax='$WMAX_LIST' seeds='$SEEDS'"
nw=$(echo $WMAX_LIST | wc -w); ns=$(echo $SEEDS | wc -w)
log "launches: KDA-INT8=1 + KDA-random=$((nw*ns)) + GDN-INT8=1 + GDN-random=$((nw*ns)) + GDN-sweep=$((nw+1)) = $((1 + nw*ns + 1 + nw*ns + nw + 1))"

# ---- Phase 1: KDA INT8 (1 launch) ----
if [ "$RUN_KDA_INT8" = 1 ]; then
  log "--- Phase 1: KDA INT8 quantization ---"
  run_arm kda "int8" 0 "$KDA_LOGDIR" \
    "SGLANG_STATE_QUANT_MODE=int8" \
    "SGLANG_HEAD_AWARE_RAGGED=1"
fi

# ---- Phase 2: KDA 5-seed random (25 launches) ----
if [ "$RUN_KDA_RANDOM" = 1 ]; then
  log "--- Phase 2: KDA 5-seed random drop ---"
  for W in $WMAX_LIST; do
    for SEED in $SEEDS; do
      run_arm kda "rand_w${W}_s${SEED}" "$W" "$KDA_LOGDIR" \
        "SGLANG_HEAD_AWARE_RAGGED=1" \
        "SGLANG_ENABLE_HEAD_AWARE_REPREFILL=0" \
        "SGLANG_HEAD_AWARE_RANDOM_DROP=1" \
        "SGLANG_HEAD_AWARE_RANDOM_SEED=$SEED"
    done
  done
fi

# ---- Phase 3: GDN INT8 (1 launch) ----
if [ "$RUN_GDN_INT8" = 1 ]; then
  log "--- Phase 3: GDN INT8 quantization ---"
  run_arm gdn "int8" 0 "$GDN_LOGDIR" \
    "SGLANG_STATE_QUANT_MODE=int8" \
    "SGLANG_HEAD_AWARE_RAGGED=1"
fi

# ---- Phase 4: GDN 5-seed random (25 launches) ----
if [ "$RUN_GDN_RANDOM" = 1 ]; then
  log "--- Phase 4: GDN 5-seed random drop ---"
  for W in $WMAX_LIST; do
    for SEED in $SEEDS; do
      run_arm gdn "rand_w${W}_s${SEED}" "$W" "$GDN_LOGDIR" \
        "SGLANG_HEAD_AWARE_RAGGED=1" \
        "SGLANG_ENABLE_HEAD_AWARE_REPREFILL=0" \
        "SGLANG_HEAD_AWARE_RANDOM_DROP=1" \
        "SGLANG_HEAD_AWARE_RANDOM_SEED=$SEED"
    done
  done
fi

# ---- Phase 5: GDN rerun MAXNEW=256 (6 launches) ----
if [ "$RUN_GDN" = 1 ]; then
  log "--- Phase 5: GDN rerun (MAXNEW=256, dense + W sweep) ---"
  run_arm gdn "dense" 0 "$GDN_LOGDIR"
  for W in $WMAX_LIST; do
    run_arm gdn "w${W}" "$W" "$GDN_LOGDIR" \
      "SGLANG_HEAD_AWARE_RAGGED=1" \
      "SGLANG_ENABLE_HEAD_AWARE_REPREFILL=0"
  done
fi

log "=== ALL DONE ==="

# ---- Summary ----
log "=== SUMMARY ==="
for LOGDIR in "$KDA_LOGDIR" "$GDN_LOGDIR"; do
  RF="$LOGDIR/acc_results.jsonl"
  SF="$LOGDIR/summary.txt"
  [ -f "$RF" ] || continue
  log "--- $(basename "$LOGDIR") ---"
  python - "$RF" "$SF" <<'PY'
import json, sys, os
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
hit = {}
if os.path.exists(sys.argv[2]):
    for l in open(sys.argv[2]):
        m = l.split()
        if len(m) >= 3:
            arm = m[0]; c = int(m[1].split("=")[1]); n = int(m[2].split("=")[1])
            hit[arm] = c/(c+n) if (c+n)>0 else 0.0
print(f"{'arm':>40} {'overall':>8} {'warmup':>8} {'replay':>8} {'tok_hit':>8} {'n_q':>5}")
for r in sorted(rows, key=lambda x: x["arm"]):
    if r.get("task") != "ruler": continue
    arm = r["arm"]
    wa = r.get("warmup_accuracy", "NA")
    ra = r.get("replay_accuracy", "NA")
    wa_s = f"{wa:.4f}" if isinstance(wa, float) else "  NA"
    ra_s = f"{ra:.4f}" if isinstance(ra, float) else "  NA"
    print(f"{arm:>40} {r['overall_accuracy']:8.4f} {wa_s:>8} {ra_s:>8} "
          f"{hit.get(arm, float('nan')):8.3f} {r['num_questions']:5d}")
PY
done
