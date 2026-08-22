#!/usr/bin/env bash
# GDN CWE+MK3 rerun: dense + DASC 5W x {recon, norecon} x {cwe, mk3} x {4k,8k,16k}
# MAXNEW=256 (fixes truncation from prior MAXNEW=64 runs).
# Qwen3-Next-80B-A3B-Instruct-NVFP4, --attention-backend triton, mem-frac 0.90.
#
# Usage:  bash experiments/scripts/run_gdn_cwe_mk3_rerun.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

export no_proxy='127.0.0.1,localhost' NO_PROXY='127.0.0.1,localhost'
# Respect caller-provided proxy variables; do not inject site-specific defaults.
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

unset SGLANG_STATE_QUANT_MODE SGLANG_HEAD_AWARE_RANDOM_DROP \
      SGLANG_HEAD_AWARE_RANDOM_SEED SGLANG_ENABLE_HEAD_AWARE_REPREFILL \
      SGLANG_HEAD_AWARE_RAGGED SGLANG_FORCE_HEAD_AWARE_WMAX

# ---- config ----
MODEL="${MODEL:-${SGLANG_REPO:-$PWD}/Qwen3-Next-80B-A3B-Instruct-NVFP4}"
BACKEND_FLAGS="--attention-backend triton"
MEM_FRAC="0.90"
PARALLEL=4
MAX_RUNNING=4
RULER_TASKS="cwe mk3"
LENGTHS="4k 8k 16k"
NUM_SAMPLES=30
NUM_TRIES=4
TEMP=0.0
MAXNEW=256
WMAX_LIST="16 64 128 512 1024"
CKPT_SIZE=200
FREE_GB=40
PORT=31007; HOST=127.0.0.1
TP=2

LOGDIR="$PWD/experiments/results/ruler_gdn_cwe_mk3_rerun"
mkdir -p "$LOGDIR"
RESULT_FILE="$LOGDIR/acc_results.jsonl"
SUMMARY="$LOGDIR/summary.txt"
touch "$SUMMARY"
ORCH="$LOGDIR/orchestrator_$(date +%Y%m%d_%H%M%S).log"

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

# run_arm <tag> <wmax> [extra env KEY=VAL ...]
run_arm () {
  local TAG="$1" WMAX="$2"
  shift 2
  local -a EXTRA_ENV=("$@")
  local SRVLOG="$LOGDIR/server_${TAG}.log"

  local -a MODE_ENV=("SGLANG_FORCE_HEAD_AWARE_WMAX=$WMAX")
  MODE_ENV+=("${EXTRA_ENV[@]}")

  # recon path needs more req pool slots for reconstruction batches
  local MR="$MAX_RUNNING"
  [[ "$TAG" == recon_* ]] && MR=16

  log "===== [$TAG] launch wmax=$WMAX max_running=$MR ====="
  wait_gpu_free
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
    env "${MODE_ENV[@]}" SGLANG_MAMBA_EVICT_POLICY="lru" \
    python -m sglang.launch_server \
      --model-path "$MODEL" --trust-remote-code --tp "$TP" \
      --host "$HOST" --port "$PORT" --mem-fraction-static "$MEM_FRAC" \
      --mamba-radix-cache-strategy extra_buffer \
      $BACKEND_FLAGS --chunked-prefill-size 2048 \
      --disable-overlap-schedule --enable-metrics \
      --enforce-disable-flashinfer-allreduce-fusion \
      --max-running-requests "$MR" \
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
      local ARM="${TAG}_${TASK}_${LEN}"
      if [ -f "$RESULT_FILE" ] && grep -q "\"arm\": \"${ARM}\"" "$RESULT_FILE" 2>/dev/null; then
        log "[skip-done] $ARM"; continue
      fi
      local MARK; MARK=$(wc -l < "$SRVLOG")
      log "[bench] $ARM nsamp=$NUM_SAMPLES tries=$NUM_TRIES"
      { python "$PWD/experiments/scripts/bench_idea1_accuracy.py" \
          --task ruler --arm "$ARM" --wmax "$WMAX" \
          --ruler-tasks "$TASK" --ruler-length "$LEN" \
          --ruler-num-samples "$NUM_SAMPLES" \
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
      echo "$ARM cached=$CACHED new=$NEW" >> "$SUMMARY"
    done
  done

  log "[shutdown] $TAG"
  kill $SRVPID 2>/dev/null || true
  wait $SRVPID 2>/dev/null || true
  sleep 5
}

log "=== GDN CWE+MK3 rerun START ==="
log "tasks='$RULER_TASKS' lengths='$LENGTHS' nsamp=$NUM_SAMPLES tries=$NUM_TRIES maxnew=$MAXNEW"
log "wmax='$WMAX_LIST' modes: dense + {norecon, recon} x 5W"
nw=$(echo $WMAX_LIST | wc -w)
log "launches: 1 (dense) + $nw x 2 (norecon+recon) = $((1 + nw * 2))"

# ---- dense baseline (W=0, no RAGGED) ----
log "--- dense baseline ---"
run_arm "dense" 0

# ---- DASC norecon: 5W ----
log "--- DASC norecon (RAGGED=1, REPREFILL=0) ---"
for W in $WMAX_LIST; do
  run_arm "norecon_w${W}" "$W" \
    "SGLANG_HEAD_AWARE_RAGGED=1" \
    "SGLANG_ENABLE_HEAD_AWARE_REPREFILL=0"
done

# ---- DASC recon: 5W ----
log "--- DASC recon (RAGGED=1, REPREFILL=1) ---"
for W in $WMAX_LIST; do
  run_arm "recon_w${W}" "$W" \
    "SGLANG_HEAD_AWARE_RAGGED=1" \
    "SGLANG_ENABLE_HEAD_AWARE_REPREFILL=1"
done

log "=== ALL DONE ==="

# ---- Summary ----
log "=== SUMMARY ==="
python - "$RESULT_FILE" "$SUMMARY" <<'PY'
import json, sys, os
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
hit = {}
if os.path.exists(sys.argv[2]):
    for l in open(sys.argv[2]):
        m = l.split()
        if len(m) >= 3:
            arm = m[0]; c = int(m[1].split("=")[1]); n = int(m[2].split("=")[1])
            hit[arm] = c/(c+n) if (c+n)>0 else 0.0
print(f"{'arm':>30} {'overall':>8} {'warmup':>8} {'replay':>8} {'tok_hit':>8} {'n_q':>5}")
for r in sorted(rows, key=lambda x: x["arm"]):
    if r.get("task") != "ruler": continue
    arm = r["arm"]
    wa = r.get("warmup_accuracy", "NA")
    ra = r.get("replay_accuracy", "NA")
    wa_s = f"{wa:.4f}" if isinstance(wa, float) else "  NA"
    ra_s = f"{ra:.4f}" if isinstance(ra, float) else "  NA"
    print(f"{arm:>30} {r['overall_accuracy']:8.4f} {wa_s:>8} {ra_s:>8} "
          f"{hit.get(arm, float('nan')):8.3f} {r['num_questions']:5d}")
PY
