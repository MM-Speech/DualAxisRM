<div align="center">
  <p><strong>Dual-Axis Generative Reward Model Toward Semantic and Turn-taking Robustness in Interactive Spoken Dialogue Models</strong></p>
  <p>Accepted by ACL 2026 Main Conference</p>
  <p>
    <a href="https://dualrm.github.io/">Project Page</a> |
    <span>Paper</span> |
    <span>Model</span> |
    <span>Dataset</span>
  </p>
</div>

This repository releases the training code for DualAxisRM. The current open-source version focuses on the core training path used in the paper:

- Stage 1 / Stage 2 supervised fine-tuning with audio-conditioned judge prompts
- Stage 3 GRPO refinement with score and format rewards
- Dataset formatting utilities for building `ms-swift`-ready JSONL files
- Clean inference entrypoints for local evaluation

Weights, full datasets, and the final model release are not included yet and will be released later.

License: Apache-2.0

## Overview

DualAxisRM is a generative reward model for interactive spoken dialogue evaluation. It decomposes reward modeling into two complementary axes:

- `Response Relevance`: whether the reply is logically consistent and topically appropriate
- `Interactional Fluency`: whether turn-taking is natural, including long pauses and extended overlap

The model generates explicit chain-of-thought style analyses for both axes and then outputs a unified binary reward:

- `0`: poor interaction
- `1`: strong interaction

## What Is Released

This repo intentionally keeps only the parts that are necessary for a clean code release:

- `src/dual_axis_rm/prompts.py`: final prompt template and sample builders
- `src/dual_axis_rm/rewards.py`: GRPO reward plugins for score accuracy and output format
- `tools/build_dataset.py`: convert simple source JSONL into `ms-swift` training files
- `scripts/train_sft.sh`: reusable SFT launcher for Stage 1 or Stage 2
- `scripts/train_grpo.sh`: GRPO launcher for Stage 3
- `scripts/infer.sh`: local inference helper
- `examples/data/source.example.jsonl`: minimal source-data example

The following are intentionally excluded from this repo:

- model weights and checkpoints
- private or large-scale datasets
- raw audio assets
- temporary annotation scripts
- hard-coded local paths, API keys, and experiment artifacts
- vendored copies of third-party frameworks such as `transformers` or `ms-swift`

## Repository Layout

```text
DualAxisRM/
├── examples/
│   └── data/
├── scripts/
├── src/
│   └── dual_axis_rm/
└── tools/
```

## Installation

Install a CUDA-matched PyTorch stack first, then:

```bash
pip install -r requirements.txt
pip install -e .
```

This codebase expects `ms-swift` for training and inference.

## Data Format

The repo uses a simple source JSONL format and converts it into `ms-swift` JSONL.

Each input line in `examples/data/source.example.jsonl` follows this schema:

```json
{
  "audio": "relative/or/absolute/path/to/dialogue.wav",
  "overall_score": 1,
  "response_think": "The response stays coherent and answers the previous turn directly.",
  "fluency_think": "Turn-taking is natural, with no harmful overlap or long silence."
}
```

Build SFT data:

```bash
python tools/build_dataset.py \
  --input examples/data/source.example.jsonl \
  --output data/train_sft.jsonl \
  --mode sft
```

Build GRPO data:

```bash
python tools/build_dataset.py \
  --input examples/data/source.example.jsonl \
  --output data/train_grpo.jsonl \
  --mode grpo
```

## Training

### Stage 1 / Stage 2: SFT

Use the same launcher for perception-oriented SFT data or CoT-style SFT data.

```bash
MODEL_PATH=Qwen/Qwen2.5-Omni-7B \
DATASET_PATH=data/train_sft.jsonl \
OUTPUT_DIR=outputs/sft \
bash scripts/train_sft.sh
```

### Stage 3: GRPO

```bash
MODEL_PATH=outputs/sft/checkpoint-xxx \
DATASET_PATH=data/train_grpo.jsonl \
OUTPUT_DIR=outputs/grpo \
bash scripts/train_grpo.sh
```

The GRPO script loads reward plugins from `src/dual_axis_rm/rewards.py`.

## Inference

```bash
MODEL_PATH=outputs/grpo/checkpoint-xxx \
VAL_DATASET=data/val.jsonl \
bash scripts/infer.sh
```

## Notes

- The current release is training-code only.
- Model weights and dataset release details will be updated on the project page.
