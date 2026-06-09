from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.build_ai2d_grpo import build_ai2d_grpo_dataset


def _image_bytes(label):
    return b"\x89PNG\r\n\x1a\n" + label.encode("ascii")


def _write_cauldron_ai2d(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_build_ai2d_grpo_dataset_keeps_clean_four_choice_rows(tmp_path):
    source = tmp_path / "0000.parquet"
    output = tmp_path / "ai2d_grpo.parquet"
    _write_cauldron_ai2d(
        source,
        [
            {
                "images": [{"bytes": _image_bytes("good-a"), "path": "a.png"}],
                "texts": [
                    {
                        "user": (
                            "<image>\nQuestion: Which object is shown?\n"
                            "Choices:\nA. cat\nB. dog\nC. bird\nD. fish\n"
                            "Answer with the letter."
                        ),
                        "assistant": "Answer: A",
                        "source": "AI2D",
                    }
                ],
            },
            {
                "images": [{"bytes": _image_bytes("bad-answer"), "path": "e.png"}],
                "texts": [
                    {
                        "user": "Question: choose.\nChoices:\nA. a\nB. b\nC. c\nD. d",
                        "assistant": "Answer: E",
                        "source": "AI2D",
                    }
                ],
            },
            {
                "images": [],
                "texts": [
                    {
                        "user": "Question: choose.\nChoices:\nA. a\nB. b\nC. c\nD. d",
                        "assistant": "B",
                        "source": "AI2D",
                    }
                ],
            },
            {
                "images": [{"bytes": _image_bytes("good-c"), "path": "c.png"}],
                "texts": [
                    {
                        "user": (
                            "Question: Which label is correct?\n"
                            "Choices:\nA. one\nB. two\nC. three\nD. four\n"
                            "Answer:"
                        ),
                        "assistant": "(C)",
                        "source": "AI2D",
                    }
                ],
            },
        ],
    )

    summary = build_ai2d_grpo_dataset(source=source, output=output)

    table = pq.read_table(output)
    rows = table.to_pylist()
    assert summary["kept"] == 2
    assert summary["skipped"]["invalid_answer"] == 1
    assert summary["skipped"]["missing_image"] == 1
    assert [row["answer"] for row in rows] == ["A", "C"]
    assert {row["task_type"] for row in rows} == {"multiple_choice"}
    assert {row["source"] for row in rows} == {"ai2d"}
    assert all(isinstance(row["image_bytes"], bytes) for row in rows)
    assert all("<image>" not in row["question"] for row in rows)
    assert all("Answer with the letter" not in row["question"] for row in rows)
    assert all(not row["question"].rstrip().endswith("Answer:") for row in rows)


def test_build_ai2d_grpo_dataset_reports_skips_after_last_valid_row(tmp_path):
    source = tmp_path / "0000.parquet"
    output = tmp_path / "ai2d_grpo.parquet"
    _write_cauldron_ai2d(
        source,
        [
            {
                "images": [{"bytes": _image_bytes("good"), "path": "a.png"}],
                "texts": [
                    {
                        "user": "Question: choose.\nChoices:\nA. a\nB. b\nC. c\nD. d",
                        "assistant": "A",
                        "source": "AI2D",
                    }
                ],
            },
            {
                "images": [{"bytes": _image_bytes("bad"), "path": "e.png"}],
                "texts": [
                    {
                        "user": "Question: choose.\nChoices:\nA. a\nB. b\nC. c\nD. d",
                        "assistant": "E",
                        "source": "AI2D",
                    }
                ],
            },
        ],
    )

    summary = build_ai2d_grpo_dataset(source=source, output=output)

    assert summary["kept"] == 1
    assert summary["skipped"]["invalid_answer"] == 1
