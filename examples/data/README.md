# Example Data

`source.example.jsonl` is a minimal source file for `tools/build_dataset.py`.

Required fields:

- `audio`: path to the dialogue audio file
- `overall_score`: binary reward label, either `1` or `2`

For `--mode sft`, provide either:

- `assistant_response`

or:

- `response_think`
- `fluency_think`

For `--mode grpo`, only `audio` and `overall_score` are required.
