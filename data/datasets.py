import io
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset, IterableDataset

from data.processors import get_image_string


@dataclass(frozen=True)
class SkippedSample:
    reason: str


class ConversationSampleProcessor:
    """Convert one parquet row into decoder inputs and assistant-only labels."""

    def __init__(self, tokenizer, image_processor, cfg, stage: str):
        if stage not in {"pretrain", "sft"}:
            raise ValueError("stage must be either 'pretrain' or 'sft'")
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.cfg = cfg
        self.stage = stage

    def process(self, row: dict[str, Any]):
        try:
            messages = json.loads(row["conversations"])
        except (KeyError, TypeError, json.JSONDecodeError):
            return SkippedSample("invalid_json")

        if not isinstance(messages, list) or not messages:
            return SkippedSample("invalid_messages")

        user_placeholder_count = sum(
            message.get("content", "").count("<image>")
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        )
        if self.stage == "pretrain" and user_placeholder_count != 1:
            return SkippedSample("image_placeholder_count")
        if self.stage == "sft" and user_placeholder_count > 1:
            return SkippedSample("image_placeholder_count")

        images = []
        image_string = None
        if user_placeholder_count == 1:
            processed = self._process_image(row.get("image_bytes"))
            if isinstance(processed, SkippedSample):
                return processed
            image_tensor, grid = processed
            if not hasattr(self.tokenizer, "global_image_token") and grid != (1, 1):
                image_tensor = image_tensor[1:]
            images.append(image_tensor)
            image_string = get_image_string(
                self.tokenizer,
                [grid],
                self.cfg.mp_image_token_length,
            )

        token_ids = []
        labels = []
        replaced_user_image = False
        for message in messages:
            if not isinstance(message, dict):
                return SkippedSample("invalid_messages")
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"} or not isinstance(content, str):
                return SkippedSample("invalid_messages")
            if role == "assistant":
                content = content.replace("<image>", "图片")
            elif role == "user" and image_string is not None and not replaced_user_image:
                content = content.replace("<image>", image_string, 1)
                replaced_user_image = True

            header_ids = self._encode(f"<|im_start|>{role}\n")
            body_ids = self._encode(content)
            ending_ids = self._encode("<|im_end|>\n")
            token_ids.extend(header_ids + body_ids + ending_ids)
            if role == "assistant":
                labels.extend([-100] * len(header_ids) + body_ids + ending_ids)
            else:
                labels.extend([-100] * (len(header_ids) + len(body_ids) + len(ending_ids)))

        max_length = getattr(self.cfg, "max_sample_length", getattr(self.cfg, "lm_max_length", 4096))
        if len(token_ids) - 1 > max_length:
            return SkippedSample("overlength")
        if len(token_ids) < 2:
            return SkippedSample("empty_tokens")

        if images:
            expected_image_tokens = sum(image.size(0) for image in images) * self.cfg.mp_image_token_length
            if token_ids.count(self.tokenizer.image_token_id) != expected_image_tokens:
                return SkippedSample("image_token_mismatch")

        return {
            "input_ids": token_ids[:-1],
            "target_ids": labels[1:],
            "attention_mask": [1] * (len(token_ids) - 1),
            "images": images,
        }

    def _process_image(self, image_bytes):
        if not isinstance(image_bytes, (bytes, bytearray, memoryview)):
            return SkippedSample("missing_image")
        try:
            with Image.open(io.BytesIO(bytes(image_bytes))) as image:
                image = image.convert("RGB")
                return self.image_processor(image)
        except Exception:
            return SkippedSample("invalid_image")

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text, add_special_tokens=False)


class _ProcessedMapDataset(Dataset):
    def __init__(self, dataset, processor: ConversationSampleProcessor):
        self.dataset = dataset
        self.processor = processor

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.processor.process(self.dataset[index])


class _ProcessedIterableDataset(IterableDataset):
    def __init__(self, dataset, processor: ConversationSampleProcessor):
        self.dataset = dataset
        self.processor = processor

    def __iter__(self):
        for row in self.dataset:
            yield self.processor.process(row)


class VLMDataCollator:
    def __init__(self, tokenizer):
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, samples):
        skipped_counts = Counter(
            sample.reason for sample in samples if isinstance(sample, SkippedSample)
        )
        valid_samples = [
            sample for sample in samples if not isinstance(sample, SkippedSample)
        ]
        if not valid_samples:
            return {
                "empty": True,
                "skipped_counts": dict(skipped_counts),
            }

        max_length = max(len(sample["input_ids"]) for sample in valid_samples)
        input_ids = []
        target_ids = []
        attention_masks = []
        images = []
        for sample in valid_samples:
            padding = max_length - len(sample["input_ids"])
            input_ids.append(sample["input_ids"] + [self.pad_token_id] * padding)
            target_ids.append(sample["target_ids"] + [-100] * padding)
            attention_masks.append(sample["attention_mask"] + [0] * padding)
            images.extend(sample["images"])

        return {
            "empty": False,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "images": images,
            "skipped_counts": dict(skipped_counts),
        }


def load_stage_datasets(
    source: str,
    *,
    split: str,
    streaming: bool,
    val_size: int,
    shuffle_buffer_size: int,
    seed: int,
    processor: ConversationSampleProcessor,
    dataset_name: Optional[str] = None,
):
    """Load, split and wrap a local parquet file or Hub dataset."""

    from datasets import load_dataset

    source_path = Path(source)
    if source.endswith(".parquet") or source_path.is_file():
        raw_dataset = load_dataset(
            "parquet",
            data_files={split: source},
            split=split,
            streaming=streaming,
        )
    else:
        raw_dataset = load_dataset(
            source,
            dataset_name,
            split=split,
            streaming=streaming,
        )

    if streaming:
        validation = raw_dataset.take(val_size)
        training = raw_dataset.skip(val_size).shuffle(
            seed=seed,
            buffer_size=shuffle_buffer_size,
        )
        return (
            _ProcessedIterableDataset(training, processor),
            _ProcessedIterableDataset(validation, processor),
        )

    validation_size = min(val_size, len(raw_dataset))
    validation = raw_dataset.select(range(validation_size))
    training = raw_dataset.select(range(validation_size, len(raw_dataset))).shuffle(seed=seed)
    return (
        _ProcessedMapDataset(training, processor),
        _ProcessedMapDataset(validation, processor),
    )
