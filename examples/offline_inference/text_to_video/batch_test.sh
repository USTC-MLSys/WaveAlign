#!/bin/bash
# 设置CUDA和GCC环境变量，确保运行时环境和编译时一致
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:/home/zijian/anaconda3/envs/xattn/lib:$LD_LIBRARY_PATH
export CC=/usr/bin/gcc-11
export CXX=/usr/bin/g++-11


# 配置参数
# THRESHOLDS=(0.8 0.9 0.95 0.99)
# MODES=(no_reorder nnz_reorder wave_reorder bcg_reorder opt_reorder fast_greedy numba_reorder wave_1d_reorder wave_2d_reorder hybrid_reorder)
# FRAMES_LIST=(161 241 321 401 481 521 561 601 641 681 721 801 881)
THRESHOLDS=(0.7)
MODES=(no_reorder)
FRAMES_LIST=(161 241 321 401 481 521 561 601 641 681)
HEIGHT=1280
WIDTH=720
STEPS=2
SPARSE_MODE=sparge
TUNE_MODE=0
SPARGE_TUNE_PV=${SPARGE_TUNE_PV:-0}
GUIDANCE_SCALE=4.0
GUIDANCE_SCALE_HIGH=3.0
TP=2  
SPARGE_STATE_DIR=${SPARGE_STATE_DIR:-/data1/dhelix_shared/daizijian/SpargeAttn/evaluate/models_dict/result}
SPARGE_AUTOTUNE_LOG_LEVEL=${SPARGE_AUTOTUNE_LOG_LEVEL:-0}
SPARGE_AUTOTUNE_PRINT_LEVEL=${SPARGE_AUTOTUNE_PRINT_LEVEL:-1}
SPARGE_AUTOTUNE_SIM_GRANULARITY=${SPARGE_AUTOTUNE_SIM_GRANULARITY:-4}
VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-WARNING}
VLLM_RPC_BASE_PATH=${VLLM_RPC_BASE_PATH:-/tmp}
CACHE_ROOT=${CACHE_ROOT:-/data1/dhelix_shared/daizijian/tmp/runtime_cache}
export TMPDIR=${TMPDIR:-$CACHE_ROOT/tmp}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$CACHE_ROOT/xdg_cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$CACHE_ROOT/triton_cache}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$CACHE_ROOT/torchinductor_cache}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$CACHE_ROOT/vllm_cache}
export VLLM_LOGGING_LEVEL
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TRANSFORMERS_VERBOSITY=error

mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$TRITON_CACHE_DIR" "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"

# Keep IPC path short and always existing; fallback to /tmp if custom path is invalid.
if [[ ! -d "$VLLM_RPC_BASE_PATH" ]]; then
  mkdir -p "$VLLM_RPC_BASE_PATH" 2>/dev/null || true
fi
if [[ ! -d "$VLLM_RPC_BASE_PATH" ]]; then
  echo "[WARN] invalid VLLM_RPC_BASE_PATH=$VLLM_RPC_BASE_PATH, fallback to /tmp"
  VLLM_RPC_BASE_PATH=/tmp
fi

# Tune mode constraints: disable CFG and force single denoising step.
if [[ "$TUNE_MODE" == "1" ]]; then
  if [[ "$STEPS" != "1" ]]; then
    echo "[INFO] TUNE_MODE=1: force STEPS from ${STEPS} to 1"
  fi
  STEPS=1
  GUIDANCE_SCALE=1.0
  GUIDANCE_SCALE_HIGH=1.0
fi

EAGER_ARGS=()
if [[ "$TUNE_MODE" == "1" ]]; then
  EAGER_ARGS+=(--enforce-eager)
fi

GPU_IDS=(0 1 2 3 4 5 6 7)
GPU_COUNT=${#GPU_IDS[@]}

# 检查GPU数量是否足够
if (( GPU_COUNT < TP )); then
  echo "[ERROR] GPU_IDS 数量不足以支持TP=${TP}，当前仅${GPU_COUNT}张卡。"
  exit 1
fi

# tp=2时，要求GPU数量为2的倍数
if (( TP == 2 )) && (( GPU_COUNT % 2 != 0 )); then
  echo "[ERROR] GPU_IDS 数量为奇数（${GPU_COUNT}），每组需要两张卡，请补齐或删去一张。"
  exit 1
fi

GROUP_COUNT=$((GPU_COUNT / TP))


# 组装任务列表
TASKS=()
for FRAMES in "${FRAMES_LIST[@]}"; do
  for THR in "${THRESHOLDS[@]}"; do
    for MODE in "${MODES[@]}"; do
      LOG="output_${SPARSE_MODE}_tp${TP}_dy_spar/vllm_${MODE}_thr${THR}_f${FRAMES}.log"
      if [ -f "$LOG" ]; then
        echo "[SKIP] $LOG 已存在，跳过该任务。"
        continue
      fi
      TASKS+=("$MODE $THR $FRAMES")
    done
  done
done

TOTAL_TASKS=${#TASKS[@]}
CUR=0

while [ $CUR -lt $TOTAL_TASKS ]; do
  JOBS=()
  for ((i=0; i<$GROUP_COUNT && $CUR<$TOTAL_TASKS; i++, CUR++)); do
    # 分配GPU
    if (( TP == 2 )); then
      GPU0=${GPU_IDS[$((i*2))]}
      GPU1=${GPU_IDS[$((i*2+1))]}
      GPUS="$GPU0,$GPU1"
    else
      GPU0=${GPU_IDS[$i]}
      GPUS="$GPU0"
    fi
    TASK=${TASKS[$CUR]}
    set -- $TASK
    MODE=$1; THR=$2; FRAMES=$3
    OUT="output_mp4/vllm_${MODE}_thr${THR}_f${FRAMES}.mp4"
    LOG="output_${SPARSE_MODE}_tp${TP}_dy_spar/vllm_${MODE}_thr${THR}_f${FRAMES}.log"
    FRAME_STATE_FILE="${SPARGE_STATE_DIR}/wan_saved_state_dict_frame${FRAMES}.pt"
    # FRAME_STATE_FILE="${SPARGE_STATE_DIR}/wan_saved_state_dict_frame561.pt"
    if [ -f "$LOG" ]; then
      echo "[SKIP] $LOG 已存在，跳过该任务。"
      continue
    fi
    echo "[GPU $GPUS] $MODE thr=$THR frames=$FRAMES -> $LOG"
    if [[ "$SPARSE_MODE" == "sparge" ]]; then
      TUNE_MODE=$TUNE_MODE TORCH_COMPILE_DISABLE=1 SPARSE_MODE=$SPARSE_MODE SPARGE_FRAME=$FRAMES SPARGE_STATE_PATH=$FRAME_STATE_FILE SPARGE_AUTOTUNE_LOG_LEVEL=$SPARGE_AUTOTUNE_LOG_LEVEL SPARGE_AUTOTUNE_PRINT_LEVEL=$SPARGE_AUTOTUNE_PRINT_LEVEL SPARGE_AUTOTUNE_SIM_GRANULARITY=$SPARGE_AUTOTUNE_SIM_GRANULARITY SPARGE_TUNE_PV=$SPARGE_TUNE_PV VLLM_RPC_BASE_PATH=$VLLM_RPC_BASE_PATH VLLM_THRESHOLD=$THR VLLM_MODE=$MODE CUDA_VISIBLE_DEVICES=$GPUS python text_to_video.py \
        --model /data1/dhelix_shared/daizijian/models/Wan2.1-D \
        --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
        --height $HEIGHT --width $WIDTH --num-frames $FRAMES \
        --guidance-scale $GUIDANCE_SCALE --guidance-scale-high $GUIDANCE_SCALE_HIGH --flow-shift 12.0 \
        --num-inference-steps $STEPS --tensor-parallel-size $TP --fps 16 --output "$OUT" \
        "${EAGER_ARGS[@]}" \
        > "$LOG" 2>&1 &
    else
      TUNE_MODE=$TUNE_MODE TORCH_COMPILE_DISABLE=1 SPARSE_MODE=$SPARSE_MODE SPARGE_AUTOTUNE_LOG_LEVEL=$SPARGE_AUTOTUNE_LOG_LEVEL SPARGE_AUTOTUNE_PRINT_LEVEL=$SPARGE_AUTOTUNE_PRINT_LEVEL SPARGE_AUTOTUNE_SIM_GRANULARITY=$SPARGE_AUTOTUNE_SIM_GRANULARITY SPARGE_TUNE_PV=$SPARGE_TUNE_PV VLLM_RPC_BASE_PATH=$VLLM_RPC_BASE_PATH VLLM_THRESHOLD=$THR VLLM_MODE=$MODE CUDA_VISIBLE_DEVICES=$GPUS python text_to_video.py \
        --model /data1/dhelix_shared/daizijian/models/Wan2.1-D \
        --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
        --height $HEIGHT --width $WIDTH --num-frames $FRAMES \
        --guidance-scale $GUIDANCE_SCALE --guidance-scale-high $GUIDANCE_SCALE_HIGH --flow-shift 12.0 \
        --num-inference-steps $STEPS --tensor-parallel-size $TP --fps 16 --output "$OUT" \
        "${EAGER_ARGS[@]}" \
        > "$LOG" 2>&1 &
    fi
    JOBS+=("$!")
  done
  for PID in "${JOBS[@]}"; do
    wait $PID
  done
done
echo "All processes finished."