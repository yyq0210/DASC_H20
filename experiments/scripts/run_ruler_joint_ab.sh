#!/usr/bin/env bash
# Joint RULER-16k accuracy + TTFT A/B (Figure 3 upgrade).
#
# Sends each RULER 16k request through the normal serving path with streaming
# HTTP so accuracy AND TTFT are measured on the SAME request.  9 arms:
#
#   dense      W=0  recon=off   (full bf16 baseline, head-aware env forced to 0)
#   NR_{16,64,128,512}  W=w  recon=off  RAGGED=1  (no-recon, hits are free)
#   WR_{16,64,128,512}  W=w  recon=on   RAGGED=1  (re-prefill on hit)
#
# Each instance repeated numtries=4 times -> try 1 miss, tries 2-4 hit (~75%).
# ckpt_size=2000 >> 200 distinct instances -> no eviction -> clean hit/miss.
#
# Usage:
#   bash experiments/scripts/run_ruler_joint_ab.sh              # full 9-arm run
#   bash experiments/scripts/run_ruler_joint_ab.sh --dry-run    # print config table
#   ARMS="dense NR_16" bash experiments/scripts/run_ruler_joint_ab.sh   # subset
set -euo pipefail

# ---- proxy / env ----
export no_proxy='127.0.0.1,localhost' NO_PROXY='127.0.0.1,localhost'
# Respect caller-provided proxy variables; do not inject site-specific defaults.
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1

# ---- config ----
MODEL="${MODEL:-${SGLANG_REPO:-$PWD}/Kimi-Linear-48B-A3B-Instruct}"
TP="${TP:-2}"
CKPT_SIZE="${CKPT_SIZE:-2000}"
N="${N:-50}"
NUMTRIES="${NUMTRIES:-4}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-30000}"
MEM_FRAC="${MEM_FRAC:-0.80}"
LOGDIR="${LOGDIR:-$PWD/experiments/results/ruler_joint}"
ALL_ARMS=("dense" "NR_16" "NR_64" "NR_128" "NR_512"
          "WR_16" "WR_64" "WR_128" "WR_512")
ARMS=(${ARMS:-${ALL_ARMS[@]}})

# ---- arm config resolver ----
# Parses arm label "dense" | "NR_W" | "WR_W" into (wmax, recon) and sets
# the corresponding env vars + server flag string.
arm_env () {  # $1 = arm label
  local arm="$1"
  case "$arm" in
    dense)
      echo "SGLANG_FORCE_HEAD_AWARE_WMAX=0 SGLANG_ENABLE_HEAD_AWARE_REPREFILL=0 SGLANG_HEAD_AWARE_RAGGED=0"
      ;;
    NR_*) echo "SGLANG_FORCE_HEAD_AWARE_WMAX=${arm#NR_} SGLANG_ENABLE_HEAD_AWARE_REPREFILL=0 SGLANG_HEAD_AWARE_RAGGED=1"
      ;;
    WR_*) echo "SGLANG_FORCE_HEAD_AWARE_WMAX=${arm#WR_} SGLANG_ENABLE_HEAD_AWARE_REPREFILL=1 SGLANG_HEAD_AWARE_RAGGED=1"
      ;;
    *) echo "ERROR"; return 1 ;;
  esac
}

arm_wmax () {
  case "$1" in dense) echo 0;; NR_*|WR_*) echo "${1#*_}";; *) echo "?";; esac
}
arm_recon () {
  case "$1" in WR_*) echo 1;; *) echo 0;; esac
}

# ---- dry-run ----
if [[ "${1:-}" == "--dry-run" ]]; then
  echo "=== RULER Joint A/B dry-run (9 arms) ==="
  printf "%-10s %5s %5s  %-45s %-50s\n" "ARM" "W" "recon" "ENV" "SERVER_EXTRA"
  printf '%.0s-' {1..120}; echo
  for arm in "${ALL_ARMS[@]}"; do
    local_env="$(arm_env "$arm")"
    local_w="$(arm_wmax "$arm")"
    local_r="$(arm_recon "$arm")"
    if [[ "$arm" == dense ]]; then
      extra="(none)"
    else
      extra="--enable-head-aware-mamba-checkpoint --head-aware-route A --head-aware-mamba-ckpt-size $CKPT_SIZE"
    fi
    printf "%-10s %5s %5s  %-45s %-50s\n" "$arm" "$local_w" "$local_r" "$local_env" "$extra"
  done
  echo ""
  echo "MODEL=$MODEL TP=$TP CKPT=$CKPT_SIZE N=$N NUMTRIES=$NUMTRIES"
  echo "Total requests/arm: $(( 4 * N * NUMTRIES ))  Total arms: ${#ALL_ARMS[@]}"
  exit 0
fi

mkdir -p "$LOGDIR"

# ---- common server flags ----
COMMON_FLAGS=(--model-path "$MODEL" --trust-remote-code --tp "$TP"
  --host "$HOST" --port "$PORT" --mem-fraction-static "$MEM_FRAC"
  --mamba-radix-cache-strategy extra_buffer
  --linear-attn-backend triton
  --chunked-prefill-size 2048
  --disable-overlap-schedule
  --enable-metrics)

# ---- per-arm launch ----
run_arm () {
  local arm="$1"
  local env_str
  env_str="$(arm_env "$arm")" || { echo "[skip] bad arm label: $arm"; return 1; }

  local SRVLOG="$LOGDIR/server_${arm}.log"
  local RESJSON="$LOGDIR/result_${arm}_joint.json"

  echo ""
  echo "===== [joint/$arm] wmax=$(arm_wmax "$arm") recon=$(arm_recon "$arm") -> $SRVLOG ====="

  # Build the head-aware flag set: dense has W=0 so head-aware flags are
  # still added (the pool is allocated but unused since WMAX=0 forces full
  # columns); this keeps the server binary path identical across arms.
  local -a HA_FLAGS=()
  if [[ "$arm" != "dense" ]]; then
    HA_FLAGS+=(--enable-head-aware-mamba-checkpoint --head-aware-route A
               --head-aware-mamba-ckpt-size "$CKPT_SIZE")
  fi

  # Export env vars for this arm.
  env $env_str \
    CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}" \
    python -m sglang.launch_server \
      "${COMMON_FLAGS[@]}" "${HA_FLAGS[@]}" > "$SRVLOG" 2>&1 &
  local SRVPID=$!

  # Wait for readiness
  local ok=0
  for i in $(seq 1 360); do
    if curl -s "http://$HOST:$PORT/health_generate" >/dev/null 2>&1; then
      echo "[ready] $arm after ${i}0s"; ok=1; break
    fi
    if ! kill -0 $SRVPID 2>/dev/null; then
      echo "[fatal] $arm server died; tail:"; tail -60 "$SRVLOG"; break
    fi
    sleep 10
  done
  if [[ "$ok" != 1 ]]; then
    echo "[skip] $arm did not come up"
    kill $SRVPID 2>/dev/null || true; wait $SRVPID 2>/dev/null || true
    return 1
  fi

  # Capacity line (for sanity)
  grep -iE "head-aware mamba checkpoint pool|x capacity|ragged-diag" "$SRVLOG" | tail -4 || true

  # Flush cache before measurement
  curl -s -X POST "http://$HOST:$PORT/flush_cache" >/dev/null 2>&1 || true
  sleep 2

  # Run probe
  echo "[bench] $arm tasks=s1,mk1,vt,cwe len=16k n=$N numtries=$NUMTRIES"
  python "$PWD/experiments/scripts/bench_ruler_joint.py" \
    --host "$HOST" --port "$PORT" \
    --arm-label "$arm" --n "$N" --numtries "$NUMTRIES" \
    --output "$RESJSON" \
    2>&1 | tee "$LOGDIR/probe_${arm}.log"

  echo "[shutdown] $arm"
  kill $SRVPID 2>/dev/null || true
  wait $SRVPID 2>/dev/null || true
  sleep 5
}

# ---- main loop ----
FAILED=()
for arm in "${ARMS[@]}"; do
  if ! run_arm "$arm"; then
    echo "[fail] arm=$arm — continuing to next arm"
    FAILED+=("$arm")
  fi
done

# ---- summary ----
echo ""
echo "===== [joint] SUMMARY (overall acc + ttft) ====="
python - "$LOGDIR" <<'PY'
import json, sys, os, glob
logdir = sys.argv[1]
files = sorted(glob.glob(os.path.join(logdir, "result_*_joint.json")))
if not files:
    print("  (no result files found)"); sys.exit(0)
print(f"{'arm':<10} {'acc':>8} {'ttft_ms':>10} {'ttft_miss':>10} {'ttft_hit':>10} {'hit_rate':>9} {'n':>6}")
rows = []
for f in files:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    arm = d.get("arm", os.path.basename(f))
    s = d.get("summary", {})
    overall_acc = s.get("overall_acc_mean")
    overall_ttft = s.get("overall_ttft_mean_ms")
    # average per-task miss/hit ttft
    pt = s.get("per_task", {})
    miss_vals = [v["ttft_try1_mean_ms"] for v in pt.values() if v.get("ttft_try1_mean_ms") is not None]
    hit_vals  = [v["ttft_hit_mean_ms"]  for v in pt.values() if v.get("ttft_hit_mean_ms") is not None]
    miss_avg = sum(miss_vals)/len(miss_vals) if miss_vals else None
    hit_avg  = sum(hit_vals)/len(hit_vals)  if hit_vals else None
    hit_rate = s.get("per_task", {})
    hrs = [v.get("hit_rate_from_try",0) for v in hit_rate.values()]
    hr_avg = sum(hrs)/len(hrs) if hrs else None
    n = s.get("total_requests", 0)
    print(f"{arm:<10} {overall_acc:>8.4f} {overall_ttft:>10.1f} "
          f"{miss_avg or 0:>10.1f} {hit_avg or 0:>10.1f} {hr_avg or 0:>9.3f} {n:>6}")
    rows.append({"arm": arm, "acc": overall_acc, "ttft": overall_ttft,
                 "miss": miss_avg, "hit": hit_avg, "hit_rate": hr_avg, "n": n})
# cross-check: input tokens identical across arms (same RULER instances)
if rows:
    # re-read for total prompt chars per arm (should be identical)
    chars = {}
    for f in files:
        try:
            d = json.load(open(f))
            arm = d.get("arm","?")
            tot = sum(r["prompt_chars"] for r in d.get("records",[]))
            chars[arm] = tot
        except Exception:
            pass
    vals = sorted(set(chars.values()))
    print(f"\n[cross-check] total prompt_chars per arm: {vals} "
          f"({'OK' if len(vals)==1 else 'WARN'})")
print(f"\n[done] {len(rows)} arms completed -> {logdir}/")
PY

if [[ ${#FAILED[@]} -gt 0 ]]; then
  echo "[warn] failed arms: ${FAILED[*]}"
fi
