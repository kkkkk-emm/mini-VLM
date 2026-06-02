import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in this Python environment")
class LanguageModelRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global torch, nn, VLMConfig, LanguageModel, LanguageModelMLP, LanguageModelMoE

        import torch
        import torch.nn as nn
        from config import VLMConfig
        from language_model import LanguageModel, LanguageModelMLP, LanguageModelMoE

    def tiny_config(self, **overrides):
        values = {
            "lm_hidden_dim": 8,
            "lm_inter_dim": 16,
            "lm_n_heads": 2,
            "lm_n_kv_heads": 1,
            "lm_n_blocks": 2,
            "lm_vocab_size": 32,
            "lm_max_position_embeddings": 16,
            "lm_num_experts": 4,
            "lm_num_experts_per_tok": 2,
            "lm_moe_inter_dim": 8,
        }
        values.update(overrides)
        return VLMConfig(**values)

    def test_dense_is_default_and_moe_replaces_every_decoder_ffn(self):
        dense_model = LanguageModel(self.tiny_config())
        moe_model = LanguageModel(self.tiny_config(lm_use_moe=True))

        self.assertTrue(all(isinstance(block.mlp, LanguageModelMLP) for block in dense_model.blocks))
        self.assertTrue(all(isinstance(block.mlp, LanguageModelMoE) for block in moe_model.blocks))

    def test_dense_and_moe_decoder_return_same_hidden_shape(self):
        x = torch.randn(2, 3, 8)

        dense_output, dense_cache, dense_aux_loss = LanguageModel(self.tiny_config())(x)
        moe_output, moe_cache, moe_aux_loss = LanguageModel(self.tiny_config(lm_use_moe=True))(x)

        self.assertEqual(dense_output.shape, moe_output.shape)
        self.assertEqual(len(dense_cache), 2)
        self.assertEqual(len(moe_cache), 2)
        self.assertEqual(dense_aux_loss.ndim, 0)
        self.assertEqual(moe_aux_loss.ndim, 0)

    def test_moe_uses_normalized_router_probabilities_as_expert_weights(self):
        class ScaleExpert(nn.Module):
            def __init__(self, scale):
                super().__init__()
                self.scale = scale

            def forward(self, x):
                return x * self.scale

        moe = LanguageModelMoE(
            self.tiny_config(
                lm_use_moe=True,
                lm_num_experts=2,
                lm_num_experts_per_tok=2,
            )
        )
        moe.experts = nn.ModuleList([ScaleExpert(1.0), ScaleExpert(3.0)])
        nn.init.zeros_(moe.gate.weight)
        x = torch.ones(1, 1, 8)

        output, _ = moe(x)

        self.assertTrue(torch.allclose(output, x * 2.0))

    def test_router_auxiliary_loss_ignores_masked_tokens(self):
        moe = LanguageModelMoE(
            self.tiny_config(
                lm_hidden_dim=2,
                lm_inter_dim=4,
                lm_n_heads=1,
                lm_n_kv_heads=1,
                lm_num_experts=2,
                lm_num_experts_per_tok=1,
                lm_moe_inter_dim=2,
            )
        )
        with torch.no_grad():
            moe.gate.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        x = torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]])

        _, unmasked_aux_loss = moe(x)
        _, masked_aux_loss = moe(x, attention_mask=torch.tensor([[1, 0]]))

        self.assertGreater(masked_aux_loss.item(), unmasked_aux_loss.item())

    def test_unselected_experts_remain_connected_to_backward_graph(self):
        moe = LanguageModelMoE(
            self.tiny_config(
                lm_use_moe=True,
                lm_num_experts=4,
                lm_num_experts_per_tok=1,
            )
        )
        with torch.no_grad():
            moe.gate.weight.zero_()
            moe.gate.weight[0].fill_(1.0)
        x = torch.ones(1, 1, 8, requires_grad=True)

        output, aux_loss = moe(x)
        (output.sum() + aux_loss).backward()

        for expert in moe.experts:
            self.assertTrue(all(parameter.grad is not None for parameter in expert.parameters()))

    def test_invalid_top_k_is_rejected(self):
        with self.assertRaises(ValueError):
            self.tiny_config(lm_num_experts=2, lm_num_experts_per_tok=3)


if __name__ == "__main__":
    unittest.main()
