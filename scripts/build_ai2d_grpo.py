#!/usr/bin/env python3
"""Clean The Cauldron AI2D parquet into mini-VLM GRPO training format."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SOURCE = Path("data/the_cauldron_parquet/ai2d/train/0000.parquet")
DEFAULT_OUTPUT = Path("data/ai2d_grpo_mmstar.parquet")

LETTER_RE = re.compile(
    r"^\s*(?:answer\s*:\s*)?\(?([ABCD])\)?(?:[.)\]\s]|$)",
    re.IGNORECASE,
)
CHOICE_LINE_RE = re.compile(r"^\s*([ABCD])\s*[.)]\s+\S+", re.IGNORECASE)
ANSWER_INSTRUCTION_RE = re.compile(
    r"^\s*(?:answer\s*(?:with\s+the\s+letter)?|reply\s+with.*|choose\s+the\s+correct\s+option)\s*[:.]?\s*$",
    re.IGNORECASE,
)


def normalize_letter_answer(answer: Any) -> str | None:
    match = LETTER_RE.match(str(answer))
    return match.group(1).upper() if match else None


def extract_image_bytes(images: Any) -> bytes | None:
    if not images:
        return None
    image = images[0]
    if not isinstance(image, dict):
        return None
    raw = image.get("bytes")
    if raw is None:
        return None
    return bytes(raw)


def clean_question(user_prompt: Any) -> str:
    text = str(user_prompt).replace("<image>", "")
    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if ANSWER_INSTRUCTION_RE.match(line):
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines).strip()


def has_four_choices(question: str) -> bool:
    labels = set()
    for line in question.splitlines():
        match = CHOICE_LINE_RE.match(line)
        if match:
            labels.add(match.group(1).upper())
    return labels == {"A", "B", "C", "D"}


def _iter_clean_rows(source: Path):
    parquet_file = pq.ParquetFile(source)
    skipped = Counter()
    row_offset = 0
    for batch in parquet_file.iter_batches(columns=["images", "texts"], batch_size=1000):
        for batch_row_index, row in enumerate(batch.to_pylist()):
            row_index = row_offset + batch_row_index
            image_bytes = extract_image_bytes(row.get("images"))
            if image_bytes is None:
                skipped["missing_image"] += len(row.get("texts") or [None])
                continue

            texts = row.get("texts") or []
            if not texts:
                skipped["missing_text"] += 1
                continue

            for text_index, item in enumerate(texts):
                if not isinstance(item, dict):
                    skipped["invalid_text"] += 1
                    continue

                answer = normalize_letter_answer(item.get("assistant", ""))
                if answer is None:
                    skipped["invalid_answer"] += 1
                    continue

                question = clean_question(item.get("user", ""))
                if not question:
                    skipped["missing_question"] += 1
                    continue
                if not has_four_choices(question):
                    skipped["not_four_choice"] += 1
                    continue

                yield {
                    "question": question,
                    "answer": answer,
                    "image_bytes": image_bytes,
                    "source": "ai2d",
                    "task_type": "multiple_choice",
                    "row_index": row_index,
                    "text_index": text_index,
                }, skipped
        row_offset += batch.num_rows


def build_ai2d_grpo_dataset(*, source: Path, output: Path) -> dict[str, Any]:
    if not source.is_file():
        raise FileNotFoundError(f"AI2D source parquet does not exist: {source}")

    rows = []
    skipped = Counter()
    for clean_row, current_skipped in _iter_clean_rows(source):
        rows.append(clean_row)
        skipped = current_skipped

    if not rows:
        raise RuntimeError(f"No valid AI2D GRPO rows were built from {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output)

    answer_counts = Counter(row["answer"] for row in rows)
    return {
        "source": str(source),
        "output": str(output),
        "kept": len(rows),
        "skipped": dict(skipped),
        "answer_counts": dict(sorted(answer_counts.items())),
    }


def format_summary(summary: dict[str, Any]) -> str:
    return json.dumps(summary, ensure_ascii=True, indent=2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_ai2d_grpo_dataset(source=args.source, output=args.output)
    print(format_summary(summary))


if __name__ == "__main__":
    main()
