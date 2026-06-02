import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys

import torch
import torch.nn as nn

from models.config import VLMConfig
from models.language_model import LanguageModel
from training.trainer import (
    CheckpointManager,
    SwanLabLogger,
    configure_trainable_parameters,
    require_sft_source,
    restore_training_state,
    resolve_precision,
    validate_training_args,
)


class TinyVLM(nn.Module):
    def __init__(self, use_moe=False):
        super().__init__()
        cfg = VLMConfig(
            lm_hidden_dim=8,
            lm_inter_dim=16,
            lm_n_heads=2,
            lm_n_kv_heads=1,
            lm_n_blocks=1,
            lm_vocab_size=32,
            lm_num_experts=2,
            lm_num_experts_per_tok=1,
            lm_moe_inter_dim=8,
            lm_use_moe=use_moe,
        )
        self.cfg = cfg
        self.vision_encoder = nn.Linear(2, 2)
        self.projector = nn.Linear(2, 2)
        self.decoder = LanguageModel(cfg)


class FakeTokenizer:
    def save_pretrained(self, path):
        Path(path, "tokenizer.txt").write_text("tokenizer", encoding="utf-8")


class TrainingRuntimeTests(unittest.TestCase):
    def test_dense_pretrain_only_trains_projector(self):
        model = TinyVLM()

        configure_trainable_parameters(model, "pretrain")

        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("projector.") for name in trainable))

    def test_moe_pretrain_trains_projector_router_and_experts(self):
        model = TinyVLM(use_moe=True)

        configure_trainable_parameters(model, "pretrain")

        trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
        self.assertTrue(any(".gate." in name for name in trainable))
        self.assertTrue(any(".experts." in name for name in trainable))
        self.assertTrue(all(
            name.startswith("projector.") or ".gate." in name or ".experts." in name
            for name in trainable
        ))

    def test_sft_freezes_vision_and_trains_projector_and_decoder(self):
        model = TinyVLM()

        configure_trainable_parameters(model, "sft")

        self.assertTrue(all(not parameter.requires_grad for parameter in model.vision_encoder.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.projector.parameters()))
        self.assertTrue(all(parameter.requires_grad for parameter in model.decoder.parameters()))

    def test_precision_auto_uses_fp32_without_cuda(self):
        precision = resolve_precision("auto", device=torch.device("cpu"))

        self.assertEqual(precision.name, "fp32")
        self.assertEqual(precision.dtype, torch.float32)
        self.assertFalse(precision.use_grad_scaler)

    def test_sft_requires_checkpoint_or_resume(self):
        with self.assertRaisesRegex(ValueError, "checkpoint"):
            require_sft_source(checkpoint=None, resume=None)

    def test_training_intervals_must_be_positive(self):
        args = SimpleNamespace(
            gradient_accumulation_steps=8,
            stats_log_interval=0,
            eval_interval=500,
            checkpoint_interval=1000,
            max_steps=10,
        )

        with self.assertRaisesRegex(ValueError, "stats_log_interval"):
            validate_training_args(args)

    def test_checkpoint_manager_keeps_latest_three_step_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(Path(temp_dir), keep_latest=3)
            for step in range(1, 5):
                manager.save(
                    model=nn.Linear(1, 1),
                    tokenizer=FakeTokenizer(),
                    optimizer=torch.optim.AdamW(nn.Linear(1, 1).parameters()),
                    scheduler=None,
                    scaler=None,
                    step=step,
                    stage="pretrain",
                    config=SimpleNamespace(),
                )

            checkpoints = sorted(path.name for path in Path(temp_dir).glob("step-*"))
            self.assertEqual(checkpoints, ["step-2", "step-3", "step-4"])

    def test_checkpoint_manager_saves_best_separately(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = CheckpointManager(Path(temp_dir), keep_latest=3)
            manager.save(
                model=nn.Linear(1, 1),
                tokenizer=FakeTokenizer(),
                optimizer=torch.optim.AdamW(nn.Linear(1, 1).parameters()),
                scheduler=None,
                scaler=None,
                step=5,
                stage="sft",
                config=SimpleNamespace(),
                is_best=True,
            )

            self.assertTrue(Path(temp_dir, "best", "model.safetensors").is_file())
            self.assertEqual(
                torch.load(Path(temp_dir, "best", "training_state.pt"), weights_only=False)["global_step"],
                5,
            )

    def test_checkpoint_restores_optimizer_scheduler_and_best_loss(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model = nn.Linear(1, 1)
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.25)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 0.5)
            optimizer.zero_grad()
            model(torch.ones(1, 1)).sum().backward()
            optimizer.step()
            scheduler.step()
            manager = CheckpointManager(Path(temp_dir))
            checkpoint = manager.save(
                model=model,
                tokenizer=FakeTokenizer(),
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=None,
                step=7,
                stage="pretrain",
                config=SimpleNamespace(),
                best_val_loss=0.125,
            )
            restored_model = nn.Linear(1, 1)
            restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1.0)
            restored_scheduler = torch.optim.lr_scheduler.LambdaLR(restored_optimizer, lambda _: 0.5)

            state = restore_training_state(
                checkpoint,
                optimizer=restored_optimizer,
                scheduler=restored_scheduler,
                scaler=None,
            )

            self.assertEqual(state["global_step"], 7)
            self.assertEqual(state["best_val_loss"], 0.125)
            self.assertEqual(restored_optimizer.param_groups[0]["lr"], optimizer.param_groups[0]["lr"])

    def test_swanlab_logger_records_scalar_metrics(self):
        recorded = []

        class FakeRun:
            def finish(self):
                recorded.append("finished")

        fake_swanlab = SimpleNamespace(
            init=lambda **kwargs: FakeRun(),
            log=lambda metrics, step: recorded.append((metrics, step)),
        )
        with patch.dict(sys.modules, {"swanlab": fake_swanlab}):
            logger = SwanLabLogger(
                enabled=True,
                project="mini-VLM",
                workspace=None,
                mode="offline",
                stage="pretrain",
                config={},
            )
            logger.log({"train/loss": 1.0}, step=3)
            logger.finish()

        self.assertEqual(recorded, [({"train/loss": 1.0}, 3), "finished"])


if __name__ == "__main__":
    unittest.main()
