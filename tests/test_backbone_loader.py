import unittest

import torch

from models.backbone_loader import _copy_language_weights, _copy_vision_weights
from models.config import VLMConfig
from models.language_model import LanguageModel, LanguageModelMoE
from models.vision_transformer import ViT


class BackboneLoaderTests(unittest.TestCase):
    def test_vision_qkv_weights_are_concatenated_in_query_key_value_order(self):
        cfg = VLMConfig(
            vit_hidden_dim=2,
            vit_inter_dim=4,
            vit_img_size=2,
            vit_patch_size=1,
            vit_n_heads=1,
            vit_n_blocks=1,
        )
        vision = ViT(cfg)
        state = {
            "vision_model.embeddings.patch_embedding.weight": torch.ones_like(vision.patch_embedding.conv.weight),
            "vision_model.embeddings.patch_embedding.bias": torch.ones_like(vision.patch_embedding.conv.bias),
            "vision_model.embeddings.position_embedding.weight": torch.ones(4, 2),
            "vision_model.encoder.layers.0.layer_norm1.weight": torch.ones(2),
            "vision_model.encoder.layers.0.layer_norm1.bias": torch.zeros(2),
            "vision_model.encoder.layers.0.self_attn.q_proj.weight": torch.full((2, 2), 1.0),
            "vision_model.encoder.layers.0.self_attn.q_proj.bias": torch.full((2,), 1.0),
            "vision_model.encoder.layers.0.self_attn.k_proj.weight": torch.full((2, 2), 2.0),
            "vision_model.encoder.layers.0.self_attn.k_proj.bias": torch.full((2,), 2.0),
            "vision_model.encoder.layers.0.self_attn.v_proj.weight": torch.full((2, 2), 3.0),
            "vision_model.encoder.layers.0.self_attn.v_proj.bias": torch.full((2,), 3.0),
            "vision_model.encoder.layers.0.self_attn.out_proj.weight": torch.ones(2, 2),
            "vision_model.encoder.layers.0.self_attn.out_proj.bias": torch.zeros(2),
            "vision_model.encoder.layers.0.layer_norm2.weight": torch.ones(2),
            "vision_model.encoder.layers.0.layer_norm2.bias": torch.zeros(2),
            "vision_model.encoder.layers.0.mlp.fc1.weight": torch.ones(4, 2),
            "vision_model.encoder.layers.0.mlp.fc1.bias": torch.zeros(4),
            "vision_model.encoder.layers.0.mlp.fc2.weight": torch.ones(2, 4),
            "vision_model.encoder.layers.0.mlp.fc2.bias": torch.zeros(2),
            "vision_model.post_layernorm.weight": torch.ones(2),
            "vision_model.post_layernorm.bias": torch.zeros(2),
        }

        _copy_vision_weights(vision, state)

        self.assertEqual(vision.patch_embedding.position_embedding.shape, (1, 4, 2))
        self.assertTrue(torch.equal(vision.blocks[0].attn.qkv.weight[:, 0], torch.tensor([1., 1., 2., 2., 3., 3.])))

    def test_moe_language_loader_skips_dense_ffn_weights(self):
        cfg = VLMConfig(
            lm_hidden_dim=4,
            lm_inter_dim=8,
            lm_n_heads=1,
            lm_n_kv_heads=1,
            lm_n_blocks=1,
            lm_base_vocab_size=6,
            lm_vocab_size=8,
            lm_num_experts=2,
            lm_num_experts_per_tok=1,
            lm_moe_inter_dim=4,
            lm_use_moe=True,
        )
        decoder = LanguageModel(cfg)
        expert_weight = decoder.blocks[0].mlp.experts[0].gate_proj.weight.detach().clone()
        state = self._language_state(decoder, cfg.lm_base_vocab_size)

        _copy_language_weights(decoder, state, base_vocab_size=cfg.lm_base_vocab_size)

        self.assertIsInstance(decoder.blocks[0].mlp, LanguageModelMoE)
        self.assertTrue(torch.equal(decoder.blocks[0].mlp.experts[0].gate_proj.weight, expert_weight))
        self.assertTrue(torch.equal(decoder.token_embedding.weight[:6], state["model.embed_tokens.weight"]))

    def test_dense_language_loader_copies_ffn_and_preserves_extra_embeddings(self):
        cfg = VLMConfig(
            lm_hidden_dim=4,
            lm_inter_dim=8,
            lm_n_heads=1,
            lm_n_kv_heads=1,
            lm_n_blocks=1,
            lm_base_vocab_size=6,
            lm_vocab_size=8,
        )
        decoder = LanguageModel(cfg)
        extra_embeddings = decoder.token_embedding.weight[6:].detach().clone()
        state = self._language_state(decoder, cfg.lm_base_vocab_size)

        _copy_language_weights(decoder, state, base_vocab_size=cfg.lm_base_vocab_size)

        self.assertTrue(torch.equal(decoder.blocks[0].mlp.gate_proj.weight, torch.ones(8, 4)))
        self.assertTrue(torch.equal(decoder.token_embedding.weight[6:], extra_embeddings))

    @staticmethod
    def _language_state(decoder, base_vocab_size):
        block = decoder.blocks[0]
        return {
            "model.embed_tokens.weight": torch.ones(base_vocab_size, decoder.token_embedding.embedding_dim),
            "model.layers.0.self_attn.q_proj.weight": torch.ones_like(block.attn.q_proj.weight),
            "model.layers.0.self_attn.k_proj.weight": torch.ones_like(block.attn.k_proj.weight),
            "model.layers.0.self_attn.v_proj.weight": torch.ones_like(block.attn.v_proj.weight),
            "model.layers.0.self_attn.o_proj.weight": torch.ones_like(block.attn.o_proj.weight),
            "model.layers.0.input_layernorm.weight": torch.ones_like(block.norm1.weight),
            "model.layers.0.post_attention_layernorm.weight": torch.ones_like(block.norm2.weight),
            "model.layers.0.mlp.gate_proj.weight": torch.ones(8, 4),
            "model.layers.0.mlp.up_proj.weight": torch.ones(8, 4),
            "model.layers.0.mlp.down_proj.weight": torch.ones(4, 8),
            "model.norm.weight": torch.ones_like(decoder.norm.weight),
        }


if __name__ == "__main__":
    unittest.main()
