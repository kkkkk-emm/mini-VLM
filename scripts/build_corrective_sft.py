#!/usr/bin/env python3
"""Build a corrective SFT dataset from selected The Cauldron subsets."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq


CAULDRON_CONFIGS = ("ai2d", "scienceqa", "vsr", "chartqa")
LETTER_LABELS = ("A", "B", "C", "D")
YES_NO_LABELS = ("Yes", "No")

LETTER_RE = re.compile(
    r"^\s*(?:answer\s*:\s*)?\(?([ABCD])\)?(?:[.)。])?\s*$",
    re.IGNORECASE,
)
YES_NO_RE = re.compile(r"^\s*(yes|no)(?:[.!。])?\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class CauldronCandidate:
    parquet_path: Path
    row_index: int
    text_index: int
    source: str
    task_type: str
    answer: str


@dataclass(frozen=True)
class ReplayCandidate:
    parquet_path: Path
    row_index: int


def normalize_letter_answer(answer: str) -> str | None:
    match = LETTER_RE.match(str(answer))
    return match.group(1).upper() if match else None


def normalize_yes_no_answer(answer: str) -> str | None:
    match = YES_NO_RE.match(str(answer))
    if not match:
        return None
    return match.group(1).capitalize()


def normalize_chart_answer(answer: str) -> str | None:
    cleaned = str(answer).strip()
    if not cleaned:
        return None
    if cleaned.endswith((".", "。")):
        cleaned = cleaned[:-1].strip()
    if not cleaned:
        return None
    if len(cleaned.split()) > 6:
        return None
    return cleaned


def ensure_image_prompt(user_prompt: str) -> str:
    cleaned = str(user_prompt).replace("<image>", "").strip()
    return "<image>\n" + cleaned


def build_conversation(user_prompt: str, assistant_answer: str) -> str:
    return json.dumps(
        [
            {"role": "user", "content": ensure_image_prompt(user_prompt)},
            {"role": "assistant", "content": assistant_answer},
        ],
        ensure_ascii=False,
    )


def _iter_cauldron_parquets(cauldron_dir: Path, config: str) -> Iterable[Path]:
    train_dir = cauldron_dir / config / "train"
    return sorted(train_dir.glob("*.parquet"))


def _classify_cauldron_item(config: str, assistant: str) -> tuple[str, str] | None:
    if config in {"ai2d", "scienceqa"}:
        answer = normalize_letter_answer(assistant)
        return ("multiple_choice", answer) if answer else None
    if config == "vsr":
        answer = normalize_yes_no_answer(assistant)
        return ("yes_no", answer) if answer else None
    if config == "chartqa":
        answer = normalize_yes_no_answer(assistant)
        if answer:
            return "yes_no", answer
        answer = normalize_chart_answer(assistant)
        return ("chart_short", answer) if answer else None
    return None


def scan_cauldron_candidates(cauldron_dir: Path) -> list[CauldronCandidate]:
    candidates: list[CauldronCandidate] = []
    for config in CAULDRON_CONFIGS:
        for parquet_path in _iter_cauldron_parquets(cauldron_dir, config):
            parquet_file = pq.ParquetFile(parquet_path)
            row_offset = 0
            for batch in parquet_file.iter_batches(columns=["texts"], batch_size=1000):
                texts_column = batch.column(0).to_pylist()
                for batch_row_index, texts in enumerate(texts_column):
                    row_index = row_offset + batch_row_index
                    for text_index, item in enumerate(texts or []):
                        if not isinstance(item, dict):
                            continue
                        classified = _classify_cauldron_item(
                            config,
                            str(item.get("assistant", "")),
                        )
                        if classified is None:
                            continue
                        task_type, answer = classified
                        candidates.append(
                            CauldronCandidate(
                                parquet_path=parquet_path,
                                row_index=row_index,
                                text_index=text_index,
                                source=config,
                                task_type=task_type,
                                answer=answer,
                            )
                        )
                row_offset += batch.num_rows
    return candidates


def _target_counts(total: int, replay_ratio: float) -> dict[str, int]:
    replay = int(round(total * replay_ratio))
    remaining = total - replay
    multiple_choice = int(round(remaining * 0.5))
    yes_no = int(round(remaining * (5 / 18)))
    chart_short = remaining - multiple_choice - yes_no
    return {
        "multiple_choice": multiple_choice,
        "yes_no": yes_no,
        "chart_short": chart_short,
        "replay": replay,
    }


def _balanced_label_counts(labels: tuple[str, ...], total: int) -> dict[str, int]:
    base = total // len(labels)
    remainder = total % len(labels)
    return {
        label: base + (1 if index < remainder else 0)
        for index, label in enumerate(labels)
    }


def _sample_pool(pool: list[Any], count: int, rng: random.Random) -> list[Any]:
    if count <= 0:
        return []
    if not pool:
        raise ValueError("Cannot sample from an empty candidate pool.")
    if len(pool) >= count:
        return rng.sample(pool, count)
    return [rng.choice(pool) for _ in range(count)]


def select_cauldron_candidates(
    candidates: list[CauldronCandidate],
    *,
    counts: dict[str, int],
    rng: random.Random,
) -> list[CauldronCandidate]:
    by_task: dict[str, list[CauldronCandidate]] = defaultdict(list)
    by_task_label: dict[tuple[str, str], list[CauldronCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_task[candidate.task_type].append(candidate)
        by_task_label[(candidate.task_type, candidate.answer)].append(candidate)

    selected: list[CauldronCandidate] = []
    for label, count in _balanced_label_counts(
        LETTER_LABELS,
        counts["multiple_choice"],
    ).items():
        selected.extend(_sample_pool(by_task_label[("multiple_choice", label)], count, rng))
    for label, count in _balanced_label_counts(YES_NO_LABELS, counts["yes_no"]).items():
        selected.extend(_sample_pool(by_task_label[("yes_no", label)], count, rng))
    selected.extend(_sample_pool(by_task["chart_short"], counts["chart_short"], rng))
    return selected


def _extract_image_bytes(images: Any) -> bytes | None:
    if not images:
        return None
    image = images[0]
    if not isinstance(image, dict):
        return None
    raw = image.get("bytes")
    if raw is None:
        return None
    return bytes(raw)


def materialize_cauldron_rows(
    selected: list[CauldronCandidate],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[Path, list[CauldronCandidate]] = defaultdict(list)
    for candidate in selected:
        grouped[candidate.parquet_path].append(candidate)

    for parquet_path, candidates in grouped.items():
        parquet_file = pq.ParquetFile(parquet_path)
        by_group: dict[int, list[tuple[CauldronCandidate, int]]] = defaultdict(list)
        for candidate in candidates:
            group_index, local_index = _row_group_for_index(
                parquet_file,
                candidate.row_index,
            )
            by_group[group_index].append((candidate, local_index))

        for group_index, indexed_candidates in by_group.items():
            table = parquet_file.read_row_group(group_index, columns=["images", "texts"])
            for candidate, local_index in indexed_candidates:
                row = table.slice(local_index, 1).to_pylist()[0]
                image_bytes = _extract_image_bytes(row.get("images"))
                texts = row.get("texts") or []
                if image_bytes is None or candidate.text_index >= len(texts):
                    continue
                item = texts[candidate.text_index]
                rows.append(
                    {
                        "conversations": build_conversation(
                            str(item.get("user", "")),
                            candidate.answer,
                        ),
                        "image_bytes": image_bytes,
                        "source": candidate.source,
                        "task_type": candidate.task_type,
                        "answer": candidate.answer,
                    }
                )
    return rows


def _normalize_replay_conversation(conversations: str) -> tuple[str, str] | None:
    try:
        messages = json.loads(conversations)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(messages, list) or not messages:
        return None

    normalized: list[dict[str, str]] = []
    first_user_index: int | None = None
    first_assistant = ""
    for message in messages:
        if not isinstance(message, dict):
            return None
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            return None
        if role == "user":
            content = content.replace("<image>", "").strip()
            if first_user_index is None:
                first_user_index = len(normalized)
        if role == "assistant" and not first_assistant:
            first_assistant = content.strip()
        normalized.append({"role": role, "content": content})

    if first_user_index is None or not first_assistant:
        return None
    normalized[first_user_index]["content"] = (
        "<image>\n" + normalized[first_user_index]["content"]
    )
    return json.dumps(normalized, ensure_ascii=False), first_assistant


def _row_group_for_index(parquet_file: pq.ParquetFile, row_index: int) -> tuple[int, int]:
    offset = 0
    for group_index in range(parquet_file.metadata.num_row_groups):
        group_rows = parquet_file.metadata.row_group(group_index).num_rows
        if offset <= row_index < offset + group_rows:
            return group_index, row_index - offset
        offset += group_rows
    raise IndexError(f"Row index {row_index} is outside parquet row range.")


def sample_replay_candidates(
    replay_source: Path,
    *,
    count: int,
    rng: random.Random,
) -> list[ReplayCandidate]:
    if count <= 0:
        return []
    parquet_file = pq.ParquetFile(replay_source)
    row_count = parquet_file.metadata.num_rows
    if row_count <= 0:
        raise ValueError(f"Replay source is empty: {replay_source}")

    if row_count < count:
        indices = [rng.randrange(row_count) for _ in range(count)]
        return [ReplayCandidate(replay_source, index) for index in indices]

    group_offsets: list[tuple[int, int, int]] = []
    offset = 0
    for group_index in range(parquet_file.metadata.num_row_groups):
        group_rows = parquet_file.metadata.row_group(group_index).num_rows
        group_offsets.append((group_index, offset, group_rows))
        offset += group_rows

    rng.shuffle(group_offsets)
    selected: list[int] = []
    remaining = count
    for _, group_offset, group_rows in group_offsets:
        if remaining <= 0:
            break
        take = min(remaining, group_rows)
        selected.extend(group_offset + local for local in rng.sample(range(group_rows), take))
        remaining -= take
    return [ReplayCandidate(replay_source, index) for index in selected]


def materialize_replay_rows(selected: list[ReplayCandidate]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    grouped: dict[Path, list[int]] = defaultdict(list)
    for candidate in selected:
        grouped[candidate.parquet_path].append(candidate.row_index)

    for parquet_path, row_indices in grouped.items():
        parquet_file = pq.ParquetFile(parquet_path)
        by_group: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for row_index in row_indices:
            group_index, local_index = _row_group_for_index(parquet_file, row_index)
            by_group[group_index].append((row_index, local_index))

        for group_index, indexed_rows in by_group.items():
            table = parquet_file.read_row_group(
                group_index,
                columns=["conversations", "image_bytes"],
            )
            for _, local_index in indexed_rows:
                row = table.slice(local_index, 1).to_pylist()[0]
                image_bytes = row.get("image_bytes")
                normalized = _normalize_replay_conversation(row.get("conversations"))
                if image_bytes is None or normalized is None:
                    continue
                conversation, answer = normalized
                rows.append(
                    {
                        "conversations": conversation,
                        "image_bytes": bytes(image_bytes),
                        "source": "sft_replay",
                        "task_type": "replay",
                        "answer": answer,
                    }
                )
    return rows


def build_corrective_sft(
    *,
    cauldron_dir: Path,
    replay_source: Path,
    output: Path,
    total: int = 30000,
    replay_ratio: float = 0.10,
    seed: int = 42,
) -> dict[str, Any]:
    if total <= 0:
        raise ValueError("--total must be positive")
    if not 0 <= replay_ratio < 1:
        raise ValueError("--replay-ratio must be in [0, 1)")

    rng = random.Random(seed)
    counts = _target_counts(total, replay_ratio)
    cauldron_candidates = scan_cauldron_candidates(cauldron_dir)
    selected_cauldron = select_cauldron_candidates(
        cauldron_candidates,
        counts=counts,
        rng=rng,
    )
    selected_replay = sample_replay_candidates(
        replay_source,
        count=counts["replay"],
        rng=rng,
    )

    rows = materialize_cauldron_rows(selected_cauldron)
    rows.extend(materialize_replay_rows(selected_replay))
    if len(rows) != total:
        raise RuntimeError(f"Expected {total} rows, built {len(rows)} rows.")
    rng.shuffle(rows)

    output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output)

    answer_counter = Counter(row["answer"] for row in rows)
    return {
        "output": str(output),
        "total": len(rows),
        "target_counts": counts,
        "task_type_counts": dict(Counter(row["task_type"] for row in rows)),
        "answer_counts_top": dict(answer_counter.most_common(40)),
        "source_counts": dict(Counter(row["source"] for row in rows)),
    }


def format_summary(summary: dict[str, Any]) -> str:
    return json.dumps(summary, ensure_ascii=True, indent=2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cauldron-dir", type=Path, default=Path("data/the_cauldron_parquet"))
    parser.add_argument("--replay-source", type=Path, default=Path("data/sft_i2t.parquet"))
    parser.add_argument("--output", type=Path, default=Path("data/corrective_sft_30k.parquet"))
    parser.add_argument("--total", type=int, default=30000)
    parser.add_argument("--replay-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_corrective_sft(
        cauldron_dir=args.cauldron_dir,
        replay_source=args.replay_source,
        output=args.output,
        total=args.total,
        replay_ratio=args.replay_ratio,
        seed=args.seed,
    )
    print(format_summary(summary))


if __name__ == "__main__":
    main()
