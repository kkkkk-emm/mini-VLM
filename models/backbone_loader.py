import gc

import torch

try:
    from .language_model import LanguageModelMLP
except ImportError:
    from language_model import LanguageModelMLP


def _copy_parameter(target, source):
    target.copy_(source.to(device=target.device, dtype=target.dtype))


@torch.no_grad()
def _copy_vision_weights(vision_encoder, state_dict):
    if "embeddings.patch_embedding.weight" in state_dict:
        key_prefix = ""
    elif "vision_model.embeddings.patch_embedding.weight" in state_dict:
        key_prefix = "vision_model."
    else:
        raise KeyError("Could not find SigLIP vision weights in state_dict")

    patch_embedding = vision_encoder.patch_embedding
    _copy_parameter(
        patch_embedding.conv.weight,
        state_dict[f"{key_prefix}embeddings.patch_embedding.weight"],
    )
    _copy_parameter(
        patch_embedding.conv.bias,
        state_dict[f"{key_prefix}embeddings.patch_embedding.bias"],
    )
    position_embedding = state_dict[f"{key_prefix}embeddings.position_embedding.weight"]
    _copy_parameter(patch_embedding.position_embedding, position_embedding.unsqueeze(0))

    for index, block in enumerate(vision_encoder.blocks):
        prefix = f"{key_prefix}encoder.layers.{index}"
        _copy_parameter(block.ln1.weight, state_dict[f"{prefix}.layer_norm1.weight"])
        _copy_parameter(block.ln1.bias, state_dict[f"{prefix}.layer_norm1.bias"])
        _copy_parameter(
            block.attn.qkv.weight,
            torch.cat(
                [
                    state_dict[f"{prefix}.self_attn.q_proj.weight"],
                    state_dict[f"{prefix}.self_attn.k_proj.weight"],
                    state_dict[f"{prefix}.self_attn.v_proj.weight"],
                ],
                dim=0,
            ),
        )
        _copy_parameter(
            block.attn.qkv.bias,
            torch.cat(
                [
                    state_dict[f"{prefix}.self_attn.q_proj.bias"],
                    state_dict[f"{prefix}.self_attn.k_proj.bias"],
                    state_dict[f"{prefix}.self_attn.v_proj.bias"],
                ],
                dim=0,
            ),
        )
        _copy_parameter(block.attn.out.weight, state_dict[f"{prefix}.self_attn.out_proj.weight"])
        _copy_parameter(block.attn.out.bias, state_dict[f"{prefix}.self_attn.out_proj.bias"])
        _copy_parameter(block.ln2.weight, state_dict[f"{prefix}.layer_norm2.weight"])
        _copy_parameter(block.ln2.bias, state_dict[f"{prefix}.layer_norm2.bias"])
        _copy_parameter(block.mlp.fc1.weight, state_dict[f"{prefix}.mlp.fc1.weight"])
        _copy_parameter(block.mlp.fc1.bias, state_dict[f"{prefix}.mlp.fc1.bias"])
        _copy_parameter(block.mlp.fc2.weight, state_dict[f"{prefix}.mlp.fc2.weight"])
        _copy_parameter(block.mlp.fc2.bias, state_dict[f"{prefix}.mlp.fc2.bias"])

    _copy_parameter(vision_encoder.norm.weight, state_dict[f"{key_prefix}post_layernorm.weight"])
    _copy_parameter(vision_encoder.norm.bias, state_dict[f"{key_prefix}post_layernorm.bias"])


@torch.no_grad()
def _copy_language_weights(decoder, state_dict, *, base_vocab_size: int):
    source_embeddings = state_dict["model.embed_tokens.weight"]
    if source_embeddings.size(0) != base_vocab_size:
        raise ValueError(
            f"Expected {base_vocab_size} base vocabulary rows, got {source_embeddings.size(0)}"
        )
    _copy_parameter(decoder.token_embedding.weight[:base_vocab_size], source_embeddings)

    for index, block in enumerate(decoder.blocks):
        prefix = f"model.layers.{index}"
        _copy_parameter(block.attn.q_proj.weight, state_dict[f"{prefix}.self_attn.q_proj.weight"])
        _copy_parameter(block.attn.k_proj.weight, state_dict[f"{prefix}.self_attn.k_proj.weight"])
        _copy_parameter(block.attn.v_proj.weight, state_dict[f"{prefix}.self_attn.v_proj.weight"])
        _copy_parameter(block.attn.o_proj.weight, state_dict[f"{prefix}.self_attn.o_proj.weight"])
        _copy_parameter(block.norm1.weight, state_dict[f"{prefix}.input_layernorm.weight"])
        _copy_parameter(block.norm2.weight, state_dict[f"{prefix}.post_attention_layernorm.weight"])
        if isinstance(block.mlp, LanguageModelMLP):
            _copy_parameter(block.mlp.gate_proj.weight, state_dict[f"{prefix}.mlp.gate_proj.weight"])
            _copy_parameter(block.mlp.up_proj.weight, state_dict[f"{prefix}.mlp.up_proj.weight"])
            _copy_parameter(block.mlp.down_proj.weight, state_dict[f"{prefix}.mlp.down_proj.weight"])

    _copy_parameter(decoder.norm.weight, state_dict["model.norm.weight"])


def load_hf_backbones(model):
    """Load official SigLIP2 and SmolLM2 weights into the custom architecture."""

    from transformers import AutoModelForCausalLM, SiglipVisionModel

    vision_model = SiglipVisionModel.from_pretrained(model.cfg.vit_model_type)
    _copy_vision_weights(model.vision_encoder, vision_model.state_dict())
    del vision_model
    gc.collect()

    language_model = AutoModelForCausalLM.from_pretrained(model.cfg.lm_model_type)
    _copy_language_weights(
        model.decoder,
        language_model.state_dict(),
        base_vocab_size=model.cfg.lm_base_vocab_size,
    )
    del language_model
    gc.collect()
    return model
