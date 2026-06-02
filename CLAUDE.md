# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a minimal PyTorch implementation of a Vision-Language Model (VLM). It combines a Vision Transformer (ViT) image encoder, a modality projector, and a decoder-only language model for image-to-text generation.

## Common Commands

- **Run inference:** `python generate.py --image <path> --prompt "<text>"`
  - Load from a local checkpoint: `python generate.py --checkpoint <path> --image <path>`
  - Load from HuggingFace: `python generate.py --hf_model lusxvr/nanoVLM-230M-8k --image <path>`
  - Measure VRAM: add `--measure_vram`
- **Run a model component test:** Each file under `models/` has a `__main__` block for smoke testing (e.g., `python models/language_model.py`).
- **There is no formal test suite, build system, or linting configuration.**

## Inferred Dependencies

The code imports the following packages (no `requirements.txt` exists):

- `torch`, `torchvision`
- `transformers`, `huggingface_hub`
- `safetensors`
- `einops`
- `Pillow`

## Architecture

### High-level Flow

`VisionLanguageModel` (`models/vision_language_model.py`) orchestrates three components:

1. **Vision Encoder** (`models/vision_transformer.py`): A standard ViT that encodes image patches into feature vectors. Default backbone is `google/siglip2-base-patch16-512`.
2. **Modality Projector** (`models/modality_projector.py`): Uses a **pixel-shuffle** downsampling step followed by a linear layer to map ViT features into the language model's embedding space and reduce sequence length.
3. **Language Model** (`models/language_model.py`): A decoder-only transformer with RMSNorm, Rotary Position Embeddings (RoPE), Grouped Query Attention (GQA), and optional Mixture-of-Experts (MoE) layers. Default backbone is `HuggingFaceTB/SmolLM2-360M-Instruct`.

During the forward pass, special image placeholder tokens in the text sequence are replaced by the projected image embeddings, and the decoder generates text autoregressively.

### Key Design Patterns

- **KV-Cache:** The language model's `forward` accepts a `block_kv_cache` list and a `start_pos`, enabling efficient autoregressive generation. The `generate` method in `VisionLanguageModel` performs a prefill phase followed by token-by-token decoding.
- **Image Token Replacement:** Image tokens (`<|image|>`) exist in the tokenizer vocabulary. During `forward`, the embedding lookup produces placeholder vectors that are overwritten by vision features via `_replace_img_tokens_with_embd`.
- **Dynamic Image Resizing & Splitting:** `data/custom_transforms.py` resizes images to be patch-divisible and optionally splits them into a global patch plus local grid patches (`GlobalAndSplitImages`). `data/processors.py` maps grid dimensions to strings like `<row_1_col_1>` using extra tokenizer tokens.
- **Config as a Single Source of Truth:** `models/config.py` defines `VLMConfig` and `TrainConfig` dataclasses. All hyperparameters (hidden dims, patch sizes, image token counts, extra token dictionaries) live there.

### Loading and Saving

- `VisionLanguageModel.from_pretrained(repo)` loads a `config.json` and `model.safetensors` from a local path or HuggingFace Hub.
- The tokenizer is lazily cached in `data/processors.py` (`TOKENIZER_CACHE`) to avoid repeated loads.

## File Guide

- `models/config.py` — All hyperparameters and training configuration.
- `models/vision_language_model.py` — Main model class, forward pass, and generation logic.
- `models/vision_transformer.py` — ViT encoder.
- `models/language_model.py` — Decoder-only LLM with GQA, RoPE, and optional MoE.
- `models/modality_projector.py` — Pixel-shuffle projector.
- `data/processors.py` — Tokenizer and image processor factories.
- `data/custom_transforms.py` — Dynamic resize and image splitting transforms.
- `generate.py` — CLI inference script.

## Notes

- The codebase mixes Chinese and English comments.
- `data/datasets.py` is currently near-empty; dataset definitions may need to be added for training.
- There are unused/accidental imports in some files (e.g., `from turtle import forward`, `from tkinter import NO`, `from email import message`) that do not affect runtime.
