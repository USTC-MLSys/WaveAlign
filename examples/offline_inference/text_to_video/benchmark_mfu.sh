#!/bin/bash
# ============================================================================
# Benchmark Attention Kernel MFU — PARALLEL across 8 GPUs (4 × TP=2)
# 8 tasks = 4 modes × 2 thresholds → 2 batches of 4 parallel jobs
# ============================================================================
set -uo pipefail

export LD_LIBRARY_PATH=/home/zijian/anaconda3/envs/xattn/lib:${LD_LIBRARY_PATH:-}
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11

PYTHON=/data1/dhelix_shared/daizijian/vllm-omni/.venv/bin/python
export VLLM_LOGGING_LEVEL=WARNING
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error

CACHE_ROOT=${CACHE_ROOT:-/data1/dhelix_shared/daizijian/tmp/runtime_cache}
export TMPDIR=${TMPDIR:-$CACHE_ROOT/tmp}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$CACHE_ROOT/xdg_cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$CACHE_ROOT/triton_cache}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor_cache}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$CACHE_ROOT/vllm_cache}
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"

# ── Configuration ──
MODEL="/data1/dhelix_shared/daizijian/Wan2.2-T2V-A14B-Diffusers"
PROMPT="Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
HEIGHT=1280
WIDTH=720
NUM_FRAMES=241
STEPS=2
TP=2
SPARSE_MODE=xattn
GUIDANCE_SCALE=4.0
GUIDANCE_SCALE_HIGH=3.0
FLOW_SHIFT=12.0
FPS=16

MODES=(no_reorder nnz_reorder wave_1d_reorder fast_greedy)
THRESHOLDS=(0.7 0.9)

# GPU pairs: 4 groups of 2
GPU_PAIRS=("0,1" "2,3" "4,5" "6,7")
NUM_SLOTS=${#GPU_PAIRS[@]}

LOG_DIR="mfu_logs"
OUT_DIR="mfu_output"
mkdir -p "$LOG_DIR" "$OUT_DIR"

# Build task list: (mode, threshold)
TASKS=()
for THR in "${THRESHOLDS[@]}"; do
  for MODE in "${MODES[@]}"; do
    TASKS+=("$MODE $THR")
  done
done
TOTAL=${#TASKS[@]}

echo "============================================="
echo " Attention Kernel MFU Benchmark (PARALLEL)"
echo " Model:      $MODEL"
echo " Resolution: ${WIDTH}x${HEIGHT}, ${NUM_FRAMES} frames"
echo " Thresholds: ${THRESHOLDS[*]}"
echo " Modes:      ${MODES[*]}"
echo " TP:         $TP × $NUM_SLOTS parallel slots"
echo " Tasks:      $TOTAL total"
echo "============================================="

CUR=0
while [ $CUR -lt $TOTAL ]; do
  PIDS=()
  LOGS_THIS_BATCH=()

  for ((slot=0; slot < NUM_SLOTS && CUR < TOTAL; slot++, CUR++)); do
    GPUS=${GPU_PAIRS[$slot]}
    TASK=${TASKS[$CUR]}
    MODE=$(echo "$TASK" | cut -d' ' -f1)
    THR=$(echo "$TASK" | cut -d' ' -f2)

    LOG="$LOG_DIR/mfu_${MODE}_thr${THR}.log"
    OUT="$OUT_DIR/mfu_${MODE}_thr${THR}.mp4"

    if [ -f "$LOG" ]; then
      echo "[SKIP] $LOG exists"
      continue
    fi

    echo "[GPU $GPUS] $MODE thr=$THR → $LOG"

    TUNE_MODE=0 \
    TORCH_COMPILE_DISABLE=1 \
    SPARSE_MODE=$SPARSE_MODE \
    VLLM_THRESHOLD=$THR \
    VLLM_MODE=$MODE \
    CUDA_VISIBLE_DEVICES=$GPUS \
    $PYTHON text_to_video.py \
      --model "$MODEL" \
      --prompt "$PROMPT" \
      --height $HEIGHT --width $WIDTH --num-frames $NUM_FRAMES \
      --guidance-scale $GUIDANCE_SCALE --guidance-scale-high $GUIDANCE_SCALE_HIGH \
      --flow-shift $FLOW_SHIFT \
      --num-inference-steps $STEPS --tensor-parallel-size $TP \
      --fps $FPS --output "$OUT" \
      > "$LOG" 2>&1 &

    PIDS+=("$!")
    LOGS_THIS_BATCH+=("$LOG")
  done

  # Wait for this batch
  echo "  ⏳ Waiting for batch (${#PIDS[@]} jobs)..."
  for PID in "${PIDS[@]}"; do
    wait "$PID"
  done
  echo "  ✓ Batch done."

  # Print quick summary for each log in this batch
  for L in "${LOGS_THIS_BATCH[@]}"; do
    echo "  ── $(basename $L) ──"
    grep -E "^\[MFU\] MFU:|^\[MFU\] Achieved|^\[MY_LOG\].*Pure Attn" "$L" 2>/dev/null | head -6 || true
  done
  echo ""
done

echo ""
echo "============================================="
echo " All $TOTAL benchmarks complete!"
echo " Run: python parse_mfu_logs.py"
echo "============================================="
