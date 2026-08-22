#!/usr/bin/env bash
# RULER suite at 4k/8k/16k — REDO to capture cache hit data for ALL tasks.
# The original run lost non-QA cache hit data (server logs overwritten on restart,
# summary.txt truncated). This redo uses a separate LOGDIR so resume-skip doesn't
# fire, and does NOT truncate summary on start (resume-safe).
#
# Key differences from longctx:
#   - LENGTHS="4k 8k 16k"
#   - Separate LOGDIR (*_smallctx_redo) so existing arms don't get skipped
#   - MAX_RUNNING=8 / PARALLEL=8 (small KV, can handle more)
#   - No summary truncation (resume-safe: >> instead of : >)
#   - Server log prefix: server_smallctx_
#
# Usage:
#   BACKEND=kda bash experiments/scripts/run_ruler_suite_smallctx_redo.sh
#   BACKEND=gdn bash experiments/scripts/run_ruler_suite_smallctx_redo.sh
set -uo pipefail
cd "$(dirname "$0")/../.."

export no_proxy='127.0.0.1,localhost' NO_PROXY='127.0.0.1,localhost'
# Respect caller-provided proxy variables; do not inject site-specific defaults.
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

BACKEND="${BACKEND:-kda}"     # kda (Kimi-Linear) | gdn (Qwen3-Next)
if [ "$BACKEND" = kda ]; then
  MODEL="${MODEL:-${SGLANG_REPO:-$PWD}/Kimi-Linear-48B-A3B-Instruct}"
  BACKEND_FLAGS="${BACKEND_FLAGS:---linear-attn-backend triton}"
  CKPT_SIZE="${CKPT_SIZE:-200}"
  MEM_FRAC="${MEM_FRAC:-0.85}"
  MAX_RUNNING="${MAX_RUNNING:-8}"
  LOGDIR="${LOGDIR:-$PWD/experiments/results/ruler_suite_kda_smallctx_redo}"
elif [ "$BACKEND" = gdn ]; then
  MODEL="${MODEL:-${SGLANG_REPO:-$PWD}/Qwen3-Next-80B-A3B-Instruct-NVFP4}"
  BACKEND_FLAGS="${BACKEND_FLAGS:---attention-backend triton}"
  CKPT_SIZE="${CKPT_SIZE:-200}"
  MEM_FRAC="${MEM_FRAC:-0.90}"   # NVFP4 uses ~104GB total, 0.85 leaves negative rest
  MAX_RUNNING="${MAX_RUNNING:-8}"
  LOGDIR="${LOGDIR:-$PWD/experiments/results/ruler_suite_gdn_norecon_smallctx_redo}"
else
  echo "BACKEND must be kda|gdn"; exit 2
fi

PORT="${PORT:-31007}"; HOST=127.0.0.1
TP="${TP:-2}"
RAGGED="${RAGGED:-1}"
REPREFILL="${REPREFILL:-0}"   # 0=NORECON (drop local, free)
WMAX_LIST="${WMAX_LIST:-16 64 128 512 1024}"
RULER_TASKS="${RULER_TASKS:-mk1 mk2 mk3 s1 s2 s3 mq mv cwe fwe vt qa_h qa_s}"
LENGTHS="${LENGTHS:-4k 8k 16k}"
NUM_SAMPLES="${NUM_SAMPLES:-30}"
NUM_TRIES="${NUM_TRIES:-4}"
PARALLEL="${PARALLEL:-8}"
TEMP="${TEMP:-0.0}"
MAXNEW="${MAXNEW:-64}"
if [ "$REPREFILL" = "1" ]; then RECON_TAG=recon; else RECON_TAG=norecon; fi

# Separate LOGDIR: existing acc_results.jsonl won't trigger resume-skip
mkdir -p "$LOGDIR"
RESULT_FILE="$LOGDIR/acc_results.jsonl"
SUMMARY="$LOGDIR/summary_smallctx.txt"
# Do NOT truncate — resume-safe (if script restarts, prior summary rows survive)

log(){ echo "[$(date +%H:%M:%S)] $*"; }

echo "[cfg] ruler-suite-smallctx-redo backend=$BACKEND model=$(basename "$MODEL") tp=$TP ckpt=$CKPT_SIZE"
echo "[cfg]   mem_frac=$MEM_FRAC max_running=$MAX_RUNNING ragged=$RAGGED reprefill=$REPREFILL"
echo "[cfg]   wmax='$WMAX_LIST' tasks='$RULER_TASKS' lengths='$LENGTHS'"
echo "[cfg]   nsamp=$NUM_SAMPLES tries=$NUM_TRIES parallel=$PARALLEL"
echo "[cfg]   logdir=$LOGDIR  result_file=$RESULT_FILE"

wait_gpu_free(){
  local need=40960
  for _ in $(seq 1 120); do
    local minfree
    minfree=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | sort -n | head -1)
    [ -z "$minfree" ] && { sleep 5; continue; }
    [ "$minfree" -ge "$need" ] && { log "  gpu free ok (min ${minfree}MiB)"; return 0; }
    sleep 10
  done
  log "  WARN: gpu never freed; proceeding anyway"; return 0
}

run_server_arm () {   # $1 = arm-tag (dense|w16...), $2 = wmax (0 for dense)
  local TAG="$1" WMAX="$2"
  local SRVLOG="$LOGDIR/server_smallctx_${TAG}.log"
  local -a MODE_ENV=()
  if [ "$WMAX" = "0" ]; then
    MODE_ENV+=(SGLANG_FORCE_HEAD_AWARE_WMAX=0)
  else
    MODE_ENV+=(SGLANG_FORCE_HEAD_AWARE_WMAX="$WMAX"
               SGLANG_ENABLE_HEAD_AWARE_REPREFILL="$REPREFILL"
               SGLANG_HEAD_AWARE_RAGGED="$RAGGED")
  fi

  echo ""
  echo "===== [$TAG] launch (backend=$BACKEND wmax=$WMAX ragged=$RAGGED) -> $SRVLOG ====="
  pkill -f "launch_server.*$PORT" 2>/dev/null || true
  sleep 5
  wait_gpu_free

  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" env "${MODE_ENV[@]}" \
    python -m sglang.launch_server \
      --model-path "$MODEL" --trust-remote-code --tp "$TP" \
      --host "$HOST" --port "$PORT" --mem-fraction-static "$MEM_FRAC" \
      --mamba-radix-cache-strategy extra_buffer \
      $BACKEND_FLAGS --chunked-prefill-size 2048 \
      --disable-overlap-schedule --enable-metrics \
      --max-running-requests "$MAX_RUNNING" \
      --enable-head-aware-mamba-checkpoint --head-aware-route A \
      --head-aware-mamba-ckpt-size "$CKPT_SIZE" > "$SRVLOG" 2>&1 &
  local SRVPID=$!

  local ok=0
  for i in $(seq 1 600); do
    if curl -s "http://$HOST:$PORT/health_generate" >/dev/null 2>&1; then
      echo "[ready] after ${i}0s"; ok=1; break; fi
    if ! kill -0 $SRVPID 2>/dev/null; then
      echo "[fatal] $TAG server died; tail:"; tail -60 "$SRVLOG"; break; fi
    sleep 10
  done
  if [ "$ok" = 0 ]; then
    echo "[skip] $TAG did not come up"
    kill $SRVPID 2>/dev/null || true; wait $SRVPID 2>/dev/null || true
    return 1
  fi

  echo "[capacity] head-aware pool init:"
  grep -iE "head-aware mamba checkpoint pool|x capacity|ragged-diag" "$SRVLOG" | tail -4 || true

  for TASK in $RULER_TASKS; do
    for LEN in $LENGTHS; do
      local NS="$NUM_SAMPLES"
      local ARM="${TAG}_${TASK}_${LEN}"
      # resume: skip if this arm already has a result row in THIS redo's result file
      if [ -f "$RESULT_FILE" ] && grep -q "\"arm\": \"${ARM}\"" "$RESULT_FILE" 2>/dev/null; then
        echo "[skip-done] $ARM already in $RESULT_FILE"; continue; fi
      local MARK; MARK=$(wc -l < "$SRVLOG")
      echo "[bench] $ARM nsamp=$NS tries=$NUM_TRIES"
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
        || { echo "[bench-fail] $ARM — skipping" | tee -a "$LOGDIR/bench_${ARM}.log"; }

      local CACHED NEW
      CACHED=$(tail -n +"$MARK" "$SRVLOG" | grep -Eo "#cached-token: [0-9]+" | awk '{s+=$2} END{print s+0}')
      NEW=$(tail -n +"$MARK" "$SRVLOG"    | grep -Eo "#new-token: [0-9]+"    | awk '{s+=$2} END{print s+0}')
      echo "[radix] $ARM cached=$CACHED new=$NEW"
      echo "${ARM} cached=$CACHED new=$NEW" >> "$SUMMARY"
    done
  done

  echo "[shutdown] $TAG"; kill $SRVPID 2>/dev/null || true; wait $SRVPID 2>/dev/null || true; sleep 5
}

run_server_arm dense 0
for W in $WMAX_LIST; do
  run_server_arm "w${W}" "$W"
done

echo ""
echo "===== [ruler-suite-smallctx-redo $BACKEND] SUMMARY (recall + cache hit per task x length x wmax) ====="
python - "$RESULT_FILE" "$SUMMARY" <<'PY'
import json, sys, os
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by={r["arm"]: r for r in rows if r.get("task")=="ruler"}
hit={}
if os.path.exists(sys.argv[2]):
    for l in open(sys.argv[2]):
        m=l.split()
        if len(m)>=3:
            arm=m[0]; c=int(m[1].split("=")[1]); n=int(m[2].split("=")[1])
            hit[arm]= c/(c+n) if (c+n)>0 else 0.0

from collections import defaultdict
by_len = defaultdict(dict)
for arm, r in sorted(by.items()):
    parts = arm.rsplit("_",2)
    if len(parts)==3:
        wmax, task, length = parts
        by_len[length][f"{wmax}_{task}"] = r

for length in sorted(by_len.keys()):
    print(f"\n--- {length} ---")
    print(f"{'arm':>28} {'recall':>8} {'tok_hit':>8} {'n_q':>5}")
    for arm in sorted(by_len[length]):
        r = by_len[length][arm]
        acc = r['overall_accuracy']
        h = hit.get(f"{arm}_{length}", hit.get(arm, float('nan')))
        print(f"{arm:>28} {acc:8.4f} {h:8.3f} {r['num_questions']:5d}")
PY
echo "[done] ruler-suite-smallctx-redo $BACKEND -> $RESULT_FILE"
