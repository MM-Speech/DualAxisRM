from __future__ import annotations

from typing import Any, Dict

DUAL_AXIS_EVAL_PROMPT = (
    "# Interactional Dialogue Evaluation\n\n"
    "**IMPORTANT**: Evaluation must include `<response think>` and "
    "`<fluency think>` analysis and `<overall score>` rating.\n"
    "Listen to a two-person interactional dialogue speech (dual-channel audio, "
    "with each channel representing one speaker), labeled as speakers A and B. "
    "Evaluate the quality of the interaction, focusing on:\n"
    "**Response Relevance:** logical consistency, topic coherence.\n"
    "**Interactional Fluency:** detect and evaluate extended vocal overlaps "
    "and long pauses, such as pauses longer than 3 seconds between speaker turns.\n\n"
    "**Note**: Small pauses and brief overlaps are acceptable, while prolonged "
    "pauses and overlapping audio are harmful. You should consider Response "
    "Relevance and Interactional Fluency separately and provide the corresponding "
    "thinking process.\n\n"
    "## Scoring Criteria\n"
    "Assign a single holistic score based on the combined evaluation:\n"
    "`0` (Poor): significant issues in either Response Relevance or "
    "Interactional Fluency.\n"
    "`1` (Excellent): both Response Relevance and Interactional Fluency are "
    "consistently appropriate and natural.\n"
    "## Evaluation Output Format\n"
    "Strictly follow this template:\n"
    "<response think>\n"
    "[Analysing Response Relevance and giving reasons for scoring...]\n"
    "</response think>\n"
    "<fluency think>\n"
    "[Analysing Interactional Fluency and giving reasons for scoring...]\n"
    "</fluency think>\n"
    "<overall score>X</overall score>\n"
)


def format_assistant_response(
    response_think: str,
    fluency_think: str,
    overall_score: int,
) -> str:
    if overall_score not in {1, 2}:
        raise ValueError(f"overall_score must be 0 or 1, got {overall_score}")

    return (
        "<response think>\n"
        f"{response_think.strip()}\n"
        "</response think>\n\n"
        "<fluency think>\n"
        f"{fluency_think.strip()}\n"
        "</fluency think>\n\n"
        f"<overall score>{overall_score}</overall score>"
    )


def build_sft_record(audio_path: str, assistant_response: str, solution: int) -> Dict[str, Any]:
    return {
        "messages": [
            {
                "role": "user",
                "content": f"<audio>{DUAL_AXIS_EVAL_PROMPT}",
            },
            {
                "role": "assistant",
                "content": assistant_response,
            },
        ],
        "audios": [audio_path],
        "solution": solution,
    }


def build_grpo_record(audio_path: str, solution: int) -> Dict[str, Any]:
    return {
        "messages": [
            {
                "role": "user",
                "content": f"<audio>{DUAL_AXIS_EVAL_PROMPT}",
            }
        ],
        "audios": [audio_path],
        "solution": solution,
    }
