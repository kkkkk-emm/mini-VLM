from __future__ import annotations

import glob
import io
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset

from data.datasets import SkippedSample
from data.processors import get_image_processor, get_image_string


GRPO_BENCHMARKS = ("mmstar", "mme", "pope")
DEFAULT_GRPO_SOURCES = {
    "mmstar": "data/MMStar/mmstar.parquet",
    "mme": "data/MME/data/*.parquet",
    "pope": "data/POPE/data/*.parquet",
}


def build_grpo_prompt(benchmark: str, question: str) -> str:
    question = str(question).rstrip()
    if benchmark == "mmstar":
        return (
            question
            + "\nChoose the correct option."
            + "\nReply with exactly one uppercase letter: A, B, C, or D."
            + "\nDo not provide an explanation."
            + "\nAnswer:"
        )
    if benchmark in {"mme", "pope"}:
        return (
            question
            + "\nReply with exactly Yes or No."
            + "\nDo not provide an explanation."
            + "\nAnswer:"
        )
    raise ValueError(f"Unsupported GRPO benchmark: {benchmark}")


def benchmark_task_type(benchmark: str) -> str:
    if benchmark == "mmstar":
        return "multiple_choice"
    if benchmark in {"mme", "pope"}:
        return "yes_no"
    raise ValueError(f"Unsupported GRPO benchmark: {benchmark}")


def resolve_grpo_dataset_files(benchmark: str, source: str | None) -> list[str]:
    if benchmark not in GRPO_BENCHMARKS:
        raise ValueError(f"Unsupported GRPO benchmark: {benchmark}")
    source = source or DEFAULT_GRPO_SOURCES[benchmark]
    source_path = Path(source)
    if source_path.is_dir():
        canonical_data_dir = source_path / "data"
        search_dir = canonical_data_dir if canonical_data_dir.is_dir() else source_path
        files = sorted(str(path) for path in search_dir.rglob("*.parquet"))
    elif source_path.is_file():
        files = [str(source_path)]
    else:
        files = sorted(glob.glob(source))
    if not files:
        raise FileNotFoundError(f"No parquet files found for {benchmark}: {source}")
    return files


def _image_from_value(value: Any) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(value))).convert("RGB")
    if isinstance(value, dict):
        raw = value.get("bytes")
        if isinstance(raw, (bytes, bytearray, memoryview)):
            return Image.open(io.BytesIO(bytes(raw))).convert("RGB")
        path = value.get("path")
        if path:
            return Image.open(path).convert("RGB")
    return None


@dataclass
class GRPOSampleProcessor:
    tokenizer: Any
    image_processor: Any
    cfg: SimpleNamespace
    benchmark: str

    def process(self, row: dict[str, Any]):
        question = row.get("question")
        answer = row.get("answer")
        if not isinstance(question, str) or answer is None:
            return SkippedSample("missing_question_or_answer")

        image = _image_from_value(row.get("image", row.get("image_bytes")))
        if image is None:
            return SkippedSample("missing_image")

        try:
            processed_image, grid = self.image_processor(image)
        except Exception:
            return SkippedSample("invalid_image")
        if not hasattr(self.tokenizer, "global_image_token") and grid != (1, 1):
            processed_image = processed_image[1:]

        image_string = get_image_string(
            self.tokenizer,
            [grid],
            self.cfg.mp_image_token_length,
        )
        prompt = image_string + build_grpo_prompt(self.benchmark, question)
        encoded = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        input_ids = encoded["input_ids"]
        if input_ids and isinstance(input_ids[0], list):
            input_ids = input_ids[0]
        attention_mask = encoded.get("attention_mask")
        if attention_mask is None:
            attention_mask = [1] * len(input_ids)
        elif attention_mask and isinstance(attention_mask[0], list):
            attention_mask = attention_mask[0]

        expected_image_tokens = processed_image.size(0) * self.cfg.mp_image_token_length
        if input_ids.count(self.tokenizer.image_token_id) != expected_image_tokens:
            return SkippedSample("image_token_mismatch")

        return {
            "input_ids": torch.tensor([input_ids], dtype=torch.long),
            "attention_mask": torch.tensor([attention_mask], dtype=torch.long),
            "images": processed_image,
            "answer": str(answer),
            "question": question,
            "task_type": benchmark_task_type(self.benchmark),
            "benchmark": self.benchmark,
            "sample_id": str(row.get("index", row.get("id", row.get("question_id", "")))),
        }


class _ProcessedGRPOMapDataset(Dataset):
    def __init__(self, dataset, processor: GRPOSampleProcessor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.processor.process(self.dataset[index])


class _ProcessedGRPOIterableDataset(IterableDataset):
    def __init__(self, dataset, processor: GRPOSampleProcessor):
        self.dataset = dataset
        self.processor = processor

    def __iter__(self):
        for row in self.dataset:
            yield self.processor.process(row)


class GRPODataCollator:
    def __call__(self, samples):
        skipped = {}
        valid = []
        for sample in samples:
            if isinstance(sample, SkippedSample):
                skipped[sample.reason] = skipped.get(sample.reason, 0) + 1
            else:
                valid.append(sample)
        if not valid:
            return {
                "empty": True,
                "skipped_counts": skipped,
            }
        if len(valid) != 1:
            raise ValueError("GRPO first version requires prompt_batch_size=1")
        output = dict(valid[0])
        output["empty"] = False
        output["skipped_counts"] = skipped
        return output


def build_grpo_data_loader(args, model, *, shuffle_seed: int):
    from datasets import load_dataset
    from torch.utils.data import DataLoader

    files = resolve_grpo_dataset_files(args.benchmark, args.dataset_source)
    raw_dataset = load_dataset(
        "parquet",
        data_files={args.dataset_split: files},
        split=args.dataset_split,
        streaming=args.stream_dataset,
    )
    if args.stream_dataset:
        raw_dataset = raw_dataset.shuffle(
            seed=shuffle_seed,
            buffer_size=args.shuffle_buffer_size,
        )
    else:
        raw_dataset = raw_dataset.shuffle(seed=shuffle_seed)

    processor = GRPOSampleProcessor(
        tokenizer=model.tokenizer,
        image_processor=get_image_processor(
            model.cfg.max_img_size,
            model.cfg.vit_img_size,
            model.cfg.resize_to_max_side_len,
        ),
        cfg=SimpleNamespace(mp_image_token_length=model.cfg.mp_image_token_length),
        benchmark=args.benchmark,
    )
    dataset = (
        _ProcessedGRPOIterableDataset(raw_dataset, processor)
        if args.stream_dataset
        else _ProcessedGRPOMapDataset(raw_dataset, processor)
    )
    return DataLoader(
        dataset,
        batch_size=args.prompt_batch_size,
        num_workers=args.num_workers,
        collate_fn=GRPODataCollator(),
        pin_memory=torch.cuda.is_available(),
    ), files
