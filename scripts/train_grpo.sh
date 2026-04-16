#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Omni-7B}"
DATASET_PATH="${DATASET_PATH:-${ROOT_DIR}/data/train_grpo.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/outputs/grpo}"
PLUGIN_PATH="${PLUGIN_PATH:-${ROOT_DIR}/src/dual_axis_rm/rewards.py}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
TORCH_DTYPE="${TORCH_DTYPE:-bfloat16}"
LEARNING_RATE="${LEARNING_RATE:-1e-6}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-2}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
SAVE_STEPS="${SAVE_STEPS:-300}"
EVAL_STEPS="${EVAL_STEPS:-300}"
LOGGING_STEPS="${LOGGING_STEPS:-5}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-5}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-512}"
NUM_GENERATIONS="${NUM_GENERATIONS:-2}"
TEMPERATURE="${TEMPERATURE:-1.0}"
NUM_ITERATIONS="${NUM_ITERATIONS:-1}"
BETA="${BETA:-0.01}"
ASYNC_GENERATE="${ASYNC_GENERATE:-false}"
USE_VLLM="${USE_VLLM:-false}"
REPORT_TO="${REPORT_TO:-none}"
DEEPSPEED_STAGE="${DEEPSPEED_STAGE:-zero3_offload}"

cmd=(
  swift rlhf
  --rlhf_type grpo
  --model "$MODEL_PATH"
  --external_plugins "$PLUGIN_PATH"
  --reward_funcs dual_axis_score_acc dual_axis_format_acc
  --use_vllm "$USE_VLLM"
  --train_type full
  --torch_dtype "$TORCH_DTYPE"
  --dataset "$DATASET_PATH"
  --max_completion_length "$MAX_COMPLETION_LENGTH"
  --num_train_epochs "$NUM_TRAIN_EPOCHS"
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
  --per_device_eval_batch_size "$PER_DEVICE_EVAL_BATCH_SIZE"
  --learning_rate "$LEARNING_RATE"
  --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
  --save_strategy steps
  --eval_strategy steps
  --eval_steps "$EVAL_STEPS"
  --save_steps "$SAVE_STEPS"
  --save_total_limit "$SAVE_TOTAL_LIMIT"
  --logging_steps "$LOGGING_STEPS"
  --output_dir "$OUTPUT_DIR"
  --warmup_ratio 0.01
  --dataloader_num_workers 1
  --num_generations "$NUM_GENERATIONS"
  --temperature "$TEMPERATURE"
  --log_completions true
  --num_iterations "$NUM_ITERATIONS"
  --async_generate "$ASYNC_GENERATE"
  --beta "$BETA"
  --report_to "$REPORT_TO"
)

if [[ -n "$DEEPSPEED_STAGE" ]]; then
  cmd+=(--deepspeed "$DEEPSPEED_STAGE")
fi

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" NPROC_PER_NODE="$NPROC_PER_NODE" "${cmd[@]}"
