import json

from scripts.measure_prompt_vram import (
    build_prompt_for_token_length,
    parse_prompt_lengths,
    write_results,
)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


def test_parse_prompt_lengths_accepts_csv_and_range():
    assert parse_prompt_lengths("8, 16,32") == [8, 16, 32]
    assert parse_prompt_lengths("8:24:8") == [8, 16, 24]


def test_build_prompt_for_token_length_hits_requested_length():
    tokenizer = FakeTokenizer()

    prompt, actual_tokens = build_prompt_for_token_length(
        tokenizer=tokenizer,
        target_tokens=7,
        seed_text="Describe the image carefully.",
    )

    assert actual_tokens == 7
    assert tokenizer.encode(prompt, add_special_tokens=False) == [
        "Describe",
        "the",
        "image",
        "carefully.",
        "Describe",
        "the",
        "image",
    ]


def test_write_results_creates_csv_json_and_png(tmp_path):
    rows = [
        {
            "target_text_tokens": 8,
            "actual_text_tokens": 8,
            "input_tokens": 42,
            "repeat": 1,
            "baseline_allocated_mb": 100.0,
            "peak_allocated_mb": 120.0,
            "delta_peak_mb": 20.0,
            "current_allocated_mb": 101.0,
        },
        {
            "target_text_tokens": 16,
            "actual_text_tokens": 16,
            "input_tokens": 50,
            "repeat": 1,
            "baseline_allocated_mb": 100.0,
            "peak_allocated_mb": 132.0,
            "delta_peak_mb": 32.0,
            "current_allocated_mb": 101.0,
        },
    ]

    output_paths = write_results(rows, tmp_path, metric="delta_peak_mb")

    assert output_paths["csv"].exists()
    assert output_paths["json"].exists()
    assert output_paths["plot"].exists()
    saved = json.loads(output_paths["json"].read_text(encoding="utf-8"))
    assert saved["rows"][1]["delta_peak_mb"] == 32.0
