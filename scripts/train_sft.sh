#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Omni-7B}"
MODEL_TYPE="${MODEL_TYPE:-qwen2_5_omni}"
DATASET_PATH="${DATASET_PATH:-${ROOT_DIR}/data/train_sft.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/sft}"
TRAIN_TYPE="${TRAIN_TYPE:-full}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
LEARNING_RATE="${LEARNING_RATE:-1e-4}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-1}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-1}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
SAVE_STEPS="${SAVE_STEPS:-200}"
EVAL_STEPS="${EVAL_STEPS:-200}"
LOGGING_STEPS="${LOGGING_STEPS:-10}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-3}"
FREEZE_VIT="${FREEZE_VIT:-false}"
FREEZE_ALIGNER="${FREEZE_ALIGNER:-false}"
DEEPSPEED_STAGE="${DEEPSPEED_STAGE:-}"

cmd=(
  swift sft
  --model "$MODEL_PATH"
  --model_type "$MODEL_TYPE"
  --dataset "$DATASET_PATH"
  --train_type "$TRAIN_TYPE"
  --output_dir "$OUTPUT_DIR"
  --torch_dtype "$TORCH_DTYPE"
  --learning_rate "$LEARNING_RATE"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --freeze_vit "$FREEZE_VIT"
  --freeze_aligner "$FREEZE_ALIGNER"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --save_strategy steps
  --eval_strategy steps
  --save_steps "$SAVE_STEPS"
  --eval_steps "$EVAL_STEPS"
  --save_total_limit "$SAVE_TOTAL_LIMIT"
  --logging_steps "$LOGGING_STEPS"
)

if [[ -n "$DEEPSPEED_STAGE" ]]; then
  cmd+=(--deepspeed "$DEEPSPEED_STAGE")
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "${cmd[@]}"
