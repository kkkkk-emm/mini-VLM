import io
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

from data.datasets import (
    ConversationSampleProcessor,
    SkippedSample,
    VLMDataCollator,
)


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2
    image_token = "<|image|>"
    image_token_id = 3
    global_image_token = "<|global_image|>"

    def __init__(self):
        self._special_tokens = {
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            self.image_token: self.image_token_id,
            self.global_image_token: 4,
        }

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        ids = []
        while text:
            for token, token_id in self._special_tokens.items():
                if text.startswith(token):
                    ids.append(token_id)
                    text = text[len(token):]
                    break
            else:
                ids.append(10 + ord(text[0]))
                text = text[1:]
        return ids


class FakeImageProcessor:
    def __call__(self, image):
        self.last_size = image.size
        return torch.ones(1, 3, 2, 2), (1, 1)


def make_image_bytes():
    image = Image.new("RGB", (2, 2), color="white")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def make_row(messages, image_bytes=None):
    return {
        "conversations": json.dumps(messages, ensure_ascii=False),
        "image_bytes": image_bytes,
    }


class ConversationSampleProcessorTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.cfg = SimpleNamespace(mp_image_token_length=2, max_sample_length=256)

    def processor(self, stage):
        return ConversationSampleProcessor(
            tokenizer=self.tokenizer,
            image_processor=FakeImageProcessor(),
            cfg=self.cfg,
            stage=stage,
        )

    def test_sft_keeps_text_only_samples_and_supervises_all_assistant_replies(self):
        sample = self.processor("sft").process(
            make_row(
                [
                    {"role": "user", "content": "q1"},
                    {"role": "assistant", "content": "a1", "reasoning_content": "ignore"},
                    {"role": "user", "content": "q2"},
                    {"role": "assistant", "content": "a2"},
                ]
            )
        )

        self.assertNotIsInstance(sample, SkippedSample)
        self.assertEqual(sample["images"], [])
        supervised = [
            token_id
            for token_id, target_id in zip(sample["input_ids"], sample["target_ids"])
            if target_id != -100
        ]
        self.assertIn(10 + ord("a"), supervised)
        self.assertEqual(supervised.count(self.tokenizer.eos_token_id), 2)

    def test_pretrain_replaces_user_image_and_sanitizes_assistant_image_literals(self):
        sample = self.processor("pretrain").process(
            make_row(
                [
                    {"role": "user", "content": "<image>describe"},
                    {"role": "assistant", "content": "literal <image> text"},
                ],
                image_bytes=make_image_bytes(),
            )
        )

        self.assertNotIsInstance(sample, SkippedSample)
        self.assertEqual(len(sample["images"]), 1)
        self.assertEqual(sample["input_ids"].count(self.tokenizer.image_token_id), 2)
        supervised_targets = [
            target_id for target_id in sample["target_ids"] if target_id != -100
        ]
        self.assertNotIn(self.tokenizer.image_token_id, supervised_targets)

    def test_pretrain_skips_text_only_samples(self):
        sample = self.processor("pretrain").process(
            make_row(
                [
                    {"role": "user", "content": "describe"},
                    {"role": "assistant", "content": "answer"},
                ]
            )
        )

        self.assertEqual(sample.reason, "image_placeholder_count")

    def test_sft_skips_samples_over_max_length(self):
        self.cfg.max_sample_length = 4
        sample = self.processor("sft").process(
            make_row(
                [
                    {"role": "user", "content": "too long"},
                    {"role": "assistant", "content": "answer"},
                ]
            )
        )

        self.assertEqual(sample.reason, "overlength")

    def test_collator_filters_skipped_samples_and_pads_valid_samples(self):
        processor = self.processor("sft")
        valid = processor.process(
            make_row(
                [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "answer"},
                ]
            )
        )
        batch = VLMDataCollator(self.tokenizer)([
            valid,
            SkippedSample("invalid_json"),
        ])

        self.assertFalse(batch["empty"])
        self.assertEqual(batch["input_ids"].shape[0], 1)
        self.assertEqual(batch["skipped_counts"], {"invalid_json": 1})


@unittest.skipUnless(importlib.util.find_spec("datasets"), "datasets is not installed")
class DatasetLoadingTests(unittest.TestCase):
    def test_local_parquet_supports_streaming_and_map_modes(self):
        import pyarrow as pa
        import pyarrow.parquet as pq

        from data.datasets import load_stage_datasets

        with tempfile.TemporaryDirectory() as temp_dir:
            parquet_path = Path(temp_dir) / "tiny.parquet"
            pq.write_table(
                pa.table(
                    {
                        "conversations": [
                            json.dumps([{"role": "user", "content": "val"}]),
                            json.dumps([{"role": "user", "content": "train"}]),
                        ],
                        "image_bytes": [None, None],
                    }
                ),
                parquet_path,
            )

            class PassThrough:
                def process(self, row):
                    return row["conversations"]

            for streaming in (True, False):
                train_dataset, val_dataset = load_stage_datasets(
                    str(parquet_path),
                    split="train",
                    streaming=streaming,
                    val_size=1,
                    shuffle_buffer_size=4,
                    seed=42,
                    processor=PassThrough(),
                )
                self.assertEqual(len(list(val_dataset)), 1)
                self.assertEqual(len(list(train_dataset)), 1)


if __name__ == "__main__":
    unittest.main()
