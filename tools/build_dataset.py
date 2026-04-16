#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterator

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dual_axis_rm.prompts import build_grpo_record, build_sft_record, format_assistant_response


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ms-swift JSONL files for DualAxisRM.")
    parser.add_argument("--input", required=True, help="Source JSONL file.")
    parser.add_argument("--output", required=True, help="Output JSONL file.")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["sft", "grpo"],
        help="Whether to create SFT or GRPO training records.",
    )
    parser.add_argument(
        "--audio-root",
        default="",
        help="Optional directory prepended to relative audio paths.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> Iterator[Dict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} of {path}") from exc


def resolve_audio_path(audio_root: Path, audio: str) -> str:
    audio_path = Path(audio)
    if audio_path.is_absolute() or not audio_root:
        return str(audio_path)
    return str(audio_root / audio_path)


def build_record(raw: Dict, mode: str, audio_root: Path) -> Dict:
    audio = raw["audio"]
    score = int(raw["overall_score"])
    audio_path = resolve_audio_path(audio_root, audio)

    if mode == "grpo":
        return build_grpo_record(audio_path=audio_path, solution=score)

    assistant_response = raw.get("assistant_response")
    if assistant_response is None:
        assistant_response = format_assistant_response(
            response_think=raw["response_think"],
            fluency_think=raw["fluency_think"],
            overall_score=score,
        )
    return build_sft_record(
        audio_path=audio_path,
        assistant_response=assistant_response,
        solution=score,
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    audio_root = Path(args.audio_root) if args.audio_root else Path()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as handle:
        for raw in load_jsonl(input_path):
            record = build_record(raw, args.mode, audio_root)
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


if __name__ == "__main__":
    main()
