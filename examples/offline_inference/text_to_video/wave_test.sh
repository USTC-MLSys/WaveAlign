#!/bin/bash
# End-to-end test: wave_reorder vs nnz_reorder
# Runs both modes across all thresholds and frame counts for fair comparison
export LD_LIBRARY_PATH=/home/zijian/anaconda3/envs/xattn/lib:$LD_LIBRARY_PATH

THRESHOLDS=(0.8 0.9 0.95 0.99)
MODES=(wave_reorder nnz_reorder)
FRAMES_LIST=(161 241 321 401 481 521 561 601 641 681 721 801 881)
HEIGHT=1280
WIDTH=720
OUTDIR=output_wave

mkdir -p $OUTDIR output_mp4

GPU_IDS=(0 1 2 3 4 5 6 7)
GPU_COUNT=${#GPU_IDS[@]}
PAIR_COUNT=$((GPU_COUNT / 2))

TASKS=()
for FRAMES in "${FRAMES_LIST[@]}"; do
  for THR in "${THRESHOLDS[@]}"; do
    for MODE in "${MODES[@]}"; do
      LOG="$OUTDIR/vllm_${MODE}_thr${THR}_f${FRAMES}.log"
      if [ -f "$LOG" ]; then
        echo "[SKIP] $LOG exists"
        continue
      fi
      TASKS+=("$MODE $THR $FRAMES")
    done
  done
done

TOTAL_TASKS=${#TASKS[@]}
echo "[wave_test] Total tasks: $TOTAL_TASKS"
CUR=0

while [ $CUR -lt $TOTAL_TASKS ]; do
  JOBS=()
  for ((i=0; i<$PAIR_COUNT && $CUR<$TOTAL_TASKS; i++, CUR++)); do
    GPU0=${GPU_IDS[$((i*2))]}
    GPU1=${GPU_IDS[$((i*2+1))]}
    GPUS="$GPU0,$GPU1"
    TASK=${TASKS[$CUR]}
    set -- $TASK
    MODE=$1; THR=$2; FRAMES=$3
    OUT="output_mp4/vllm_${MODE}_thr${THR}_f${FRAMES}.mp4"
    LOG="$OUTDIR/vllm_${MODE}_thr${THR}_f${FRAMES}.log"
    if [ -f "$LOG" ]; then
      echo "[SKIP] $LOG exists"
      continue
    fi
    echo "[GPU $GPUS] $MODE thr=$THR frames=$FRAMES -> $LOG"
    TORCH_COMPILE_DISABLE=1 VLLM_THRESHOLD=$THR VLLM_MODE=$MODE CUDA_VISIBLE_DEVICES=$GPUS python text_to_video.py \
      --model /data1/dhelix_shared/daizijian/Wan2.2-T2V-A14B-Diffusers \
      --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
      --height $HEIGHT --width $WIDTH --num-frames $FRAMES \
      --guidance-scale 4.0 --guidance-scale-high 3.0 --flow-shift 12.0 \
      --num-inference-steps 2 --tensor-parallel-size 2 --fps 16 --output "$OUT" \
      > "$LOG" 2>&1 &
    JOBS+=("$!")
  done
  for PID in "${JOBS[@]}"; do
    wait $PID
  done
done
echo "All wave_test tasks finished."
