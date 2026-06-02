import unittest
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
if str(MODELS) not in sys.path:
    sys.path.insert(0, str(MODELS))

from config import VLMConfig


def read_text(relative_path):
    return (ROOT / relative_path).read_text(encoding="utf-8")


class MoESourceContractTests(unittest.TestCase):
    def test_config_exposes_dense_default_and_moe_parameters(self):
        source = read_text("models/config.py")

        self.assertIn("lm_use_moe: bool = False", source)
        self.assertIn("lm_num_experts: int = 8", source)
        self.assertIn("lm_num_experts_per_tok: int = 2", source)
        self.assertIn("lm_moe_inter_dim: int = 1280", source)
        self.assertIn("lm_norm_topk_prob: bool = True", source)
        self.assertIn("lm_router_aux_loss_coef: float = 0.01", source)
        self.assertIn("def __post_init__(self):", source)

    def test_config_rejects_more_selected_experts_than_available_experts(self):
        with self.assertRaises(ValueError):
            VLMConfig(lm_num_experts=2, lm_num_experts_per_tok=3)

    def test_decoder_selects_moe_and_returns_auxiliary_loss(self):
        source = read_text("models/language_model.py")

        self.assertIn(
            "self.mlp = LanguageModelMoE(cfg) if cfg.lm_use_moe else LanguageModelMLP(cfg)",
            source,
        )
        self.assertIn("top_k_weight, top_k_indices = torch.topk(scores", source)
        self.assertIn("def forward(self, x, attention_mask=None):", source)
        self.assertIn("return x, block_kv_cache, router_aux_loss", source)

    def test_vlm_adds_auxiliary_loss_and_uses_cache_contract(self):
        source = read_text("models/vision_language_model.py")

        self.assertIn("hidden_status, _, router_aux_loss = self.decoder(", source)
        self.assertIn("loss = ce_loss + router_aux_loss", source)
        self.assertIn("block_kv_cache=None", source)
        self.assertIn("block_kv_cache=block_kv_cache", source)
        self.assertIn("next_output = next_output[:, -1, :]", source)
        self.assertNotIn("self.decoder.cfg", source)

    def test_tensor_conditions_are_explicit(self):
        lm_source = read_text("models/language_model.py")
        vlm_source = read_text("models/vision_language_model.py")

        self.assertNotIn("if attention_mask:", lm_source)
        self.assertNotIn("if additive_attn_mask:", lm_source)
        self.assertNotIn("if images_tensors:", vlm_source)
        self.assertNotIn("if target_ids:", vlm_source)
        self.assertNotIn("if attention_mask:", vlm_source)

    def test_image_string_and_cli_generation_blockers_are_fixed(self):
        processors_source = read_text("data/processors.py")
        generate_source = read_text("generate.py")

        self.assertIn("for idx, (nh, nw) in enumerate(splitted_image_counts):", processors_source)
        self.assertIn("tokenizer.batch_decode(", generate_source)
        self.assertIn("splitted_image_ratio != (1, 1)", generate_source)


if __name__ == "__main__":
    unittest.main()
