#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Omni-7B}"
VAL_DATASET="${VAL_DATASET:-${ROOT_DIR}/data/val.jsonl}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
TEMPERATURE="${TEMPERATURE:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
INFER_BACKEND="${INFER_BACKEND:-pt}"

CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" swift infer \
  --model "$MODEL_PATH" \
  --infer_backend "$INFER_BACKEND" \
  --temperature "$TEMPERATURE" \
  --val_dataset "$VAL_DATASET" \
  --max_new_tokens "$MAX_NEW_TOKENS"
