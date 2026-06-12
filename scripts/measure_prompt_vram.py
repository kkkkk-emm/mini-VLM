#!/usr/bin/env python3
"""Measure CUDA VRAM usage across prompt lengths and plot a line chart."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.processors import get_image_processor, get_image_string
from models.vision_language_model import VisionLanguageModel


DEFAULT_IMAGE = Path("data/eval_images/image-01-golden-dog-balloons.jpg")
DEFAULT_OUTPUT_DIR = Path("results/prompt_vram")
DEFAULT_SEED_TEXT = "Describe the image carefully and answer the visual question."


def parse_prompt_lengths(value: str) -> list[int]:
    """Parse prompt lengths from CSV, or inclusive range syntax start:end:step."""

    value = value.strip()
    if not value:
        raise ValueError("prompt lengths cannot be empty")

    if "," in value:
        lengths = [int(part.strip()) for part in value.split(",") if part.strip()]
    elif ":" in value:
        parts = [int(part.strip()) for part in value.split(":")]
        if len(parts) not in {2, 3}:
            raise ValueError("range syntax must be start:end or start:end:step")
        start, end = parts[0], parts[1]
        step = parts[2] if len(parts) == 3 else 1
        if step <= 0:
            raise ValueError("range step must be positive")
        lengths = list(range(start, end + 1, step))
    else:
        lengths = [int(value)]

    if not lengths or any(length <= 0 for length in lengths):
        raise ValueError("prompt lengths must be positive integers")
    return lengths


def _tokenize_text(tokenizer: Any, text: str) -> list[Any]:
    if hasattr(tokenizer, "encode"):
        return list(tokenizer.encode(text, add_special_tokens=False))
    encoded = tokenizer(text, add_special_tokens=False)
    input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
    return list(input_ids)


def _decode_tokens(tokenizer: Any, token_ids: list[Any]) -> str:
    if hasattr(tokenizer, "decode"):
        return tokenizer.decode(token_ids, skip_special_tokens=True)
    return " ".join(str(token_id) for token_id in token_ids)


def build_prompt_for_token_length(
    *,
    tokenizer: Any,
    target_tokens: int,
    seed_text: str,
) -> tuple[str, int]:
    """Create a text prompt whose tokenizer length is at least close to target_tokens."""

    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    seed_ids = _tokenize_text(tokenizer, seed_text)
    if not seed_ids:
        raise ValueError("seed_text must produce at least one token")

    repeats = math.ceil(target_tokens / len(seed_ids))
    target_ids = (seed_ids * repeats)[:target_tokens]
    prompt = _decode_tokens(tokenizer, target_ids).strip()
    actual_tokens = len(_tokenize_text(tokenizer, prompt))

    # Some tokenizers do not round-trip decode -> encode exactly. Keep extending
    # with seed text until the requested scale is reached, and record actual size.
    guard = 0
    while actual_tokens < target_tokens and guard < 32:
        prompt = f"{prompt} {seed_text}".strip()
        actual_tokens = len(_tokenize_text(tokenizer, prompt))
        guard += 1
    return prompt, actual_tokens


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure mini-VLM CUDA VRAM usage for different prompt lengths."
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--checkpoint",
        type=str,
        help="Complete local VLM checkpoint directory containing config.json and model.safetensors.",
    )
    source_group.add_argument(
        "--hf-model",
        "--hf_model",
        dest="hf_model",
        type=str,
        help="Hugging Face model repo or local exported VLM directory.",
    )
    parser.add_argument("--image", type=str, default=str(DEFAULT_IMAGE), help="Input image path.")
    parser.add_argument(
        "--prompt-lengths",
        type=str,
        default="8,16,32,64,128,256,512",
        help="Target text token lengths. Use CSV like 8,16,32 or inclusive range like 8:512:8.",
    )
    parser.add_argument(
        "--seed-text",
        type=str,
        default=DEFAULT_SEED_TEXT,
        help="Text repeated/truncated to synthesize prompts of different token lengths.",
    )
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--greedy", action="store_true", help="Use greedy decoding.")
    parser.add_argument("--repeats", type=int, default=1, help="Measured repeats per prompt length.")
    parser.add_argument("--warmup", type=int, default=1, help="Unmeasured warmup generations.")
    parser.add_argument(
        "--metric",
        choices=["delta_peak_mb", "peak_allocated_mb"],
        default="delta_peak_mb",
        help="Metric used for the plotted line.",
    )
    args = parser.parse_args(argv)

    try:
        args.prompt_lengths = parse_prompt_lengths(args.prompt_lengths)
    except ValueError as error:
        parser.error(str(error))
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.top_k < 0:
        parser.error("--top-k must be non-negative")
    if not 0.0 < args.top_p <= 1.0:
        parser.error("--top-p must be in the range (0, 1]")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if args.repeats <= 0:
        parser.error("--repeats must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if not args.seed_text.strip():
        parser.error("--seed-text cannot be empty")
    return args


def prepare_vlm_inputs(
    *,
    model: VisionLanguageModel,
    image: Image.Image,
    prompt: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, tuple[int, int]]:
    tokenizer = model.tokenizer
    resize_to_max_side_len = getattr(model.cfg, "resize_to_max_side_len", False)
    image_processor = get_image_processor(
        model.cfg.max_img_size,
        model.cfg.vit_img_size,
        resize_to_max_side_len,
    )
    processed_image, split_ratio = image_processor(image.convert("RGB"))
    if not hasattr(tokenizer, "global_image_token") and split_ratio != (1, 1):
        processed_image = processed_image[1:]

    image_string = get_image_string(tokenizer, [split_ratio], model.cfg.mp_image_token_length)
    messages = [{"role": "user", "content": image_string + prompt}]
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )

    input_ids = encoded["input_ids"]
    if input_ids and isinstance(input_ids[0], int):
        input_ids = [input_ids]
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None and attention_mask and isinstance(attention_mask[0], int):
        attention_mask = [attention_mask]

    tokens = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention = (
        torch.tensor(attention_mask, dtype=torch.long, device=device)
        if attention_mask is not None
        else torch.ones_like(tokens, dtype=torch.long, device=device)
    )
    return tokens, attention, processed_image.to(device), tokens.size(1), split_ratio


def run_generation(
    *,
    model: VisionLanguageModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    images: torch.Tensor,
    args: argparse.Namespace,
) -> torch.Tensor:
    with torch.inference_mode():
        return model.generate(
            input_ids=input_ids,
            images=images,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            greedy=args.greedy,
        )


def measure_one(
    *,
    model: VisionLanguageModel,
    image: Image.Image,
    prompt: str,
    target_text_tokens: int,
    actual_text_tokens: int,
    repeat_index: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    input_ids, attention_mask, images, input_tokens, split_ratio = prepare_vlm_inputs(
        model=model,
        image=image,
        prompt=prompt,
        device=device,
    )

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    _ = run_generation(
        model=model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        images=images,
        args=args,
    )
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    current = torch.cuda.memory_allocated(device)

    return {
        "target_text_tokens": target_text_tokens,
        "actual_text_tokens": actual_text_tokens,
        "input_tokens": input_tokens,
        "repeat": repeat_index,
        "image_grid_h": split_ratio[0],
        "image_grid_w": split_ratio[1],
        "max_new_tokens": args.max_new_tokens,
        "baseline_allocated_mb": baseline / 1024**2,
        "peak_allocated_mb": peak / 1024**2,
        "delta_peak_mb": (peak - baseline) / 1024**2,
        "current_allocated_mb": current / 1024**2,
    }


def write_results(rows: list[dict[str, Any]], output_dir: Path | str, metric: str) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "prompt_vram.csv"
    json_path = output_dir / "prompt_vram.json"
    plot_path = output_dir / "prompt_vram.png"

    fieldnames = [
        "target_text_tokens",
        "actual_text_tokens",
        "input_tokens",
        "repeat",
        "image_grid_h",
        "image_grid_w",
        "max_new_tokens",
        "baseline_allocated_mb",
        "peak_allocated_mb",
        "delta_peak_mb",
        "current_allocated_mb",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        grouped[int(row["target_text_tokens"])].append(float(row[metric]))
    xs = sorted(grouped)
    ys = [sum(grouped[x]) / len(grouped[x]) for x in xs]

    payload = {
        "metric": metric,
        "summary": [
            {"target_text_tokens": x, f"mean_{metric}": y, "repeats": len(grouped[x])}
            for x, y in zip(xs, ys)
        ],
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o", linewidth=2)
    plt.xlabel("Target text prompt length (tokens)")
    ylabel = "Peak VRAM delta (MB)" if metric == "delta_peak_mb" else "Peak allocated VRAM (MB)"
    plt.ylabel(ylabel)
    plt.title("mini-VLM VRAM vs Prompt Length")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=180)
    plt.close()

    return {"csv": csv_path, "json": json_path, "plot": plot_path}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VRAM measurement. Run this script on a CUDA GPU machine.")

    device = torch.device("cuda")
    source = args.checkpoint if args.checkpoint else args.hf_model
    print(f"Using device: {device}")
    print(f"Loading model from: {source}")
    model = VisionLanguageModel.from_pretrained(source).to(device)
    model.eval()

    image = Image.open(args.image).convert("RGB")
    prompts = [
        (length, *build_prompt_for_token_length(tokenizer=model.tokenizer, target_tokens=length, seed_text=args.seed_text))
        for length in args.prompt_lengths
    ]

    if args.warmup:
        warmup_length, warmup_prompt, warmup_actual = prompts[0]
        print(f"Running {args.warmup} warmup generation(s) at target_text_tokens={warmup_length}")
        for _ in range(args.warmup):
            input_ids, attention_mask, images, _, _ = prepare_vlm_inputs(
                model=model,
                image=image,
                prompt=warmup_prompt,
                device=device,
            )
            _ = run_generation(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                args=args,
            )
            torch.cuda.synchronize(device)
        print(f"Warmup prompt actual_text_tokens={warmup_actual}")

    rows = []
    for target_text_tokens, prompt, actual_text_tokens in prompts:
        for repeat_index in range(1, args.repeats + 1):
            row = measure_one(
                model=model,
                image=image,
                prompt=prompt,
                target_text_tokens=target_text_tokens,
                actual_text_tokens=actual_text_tokens,
                repeat_index=repeat_index,
                args=args,
                device=device,
            )
            rows.append(row)
            print(
                "target_text_tokens={target} actual_text_tokens={actual} input_tokens={input_len} "
                "repeat={repeat} delta_peak_mb={delta:.2f} peak_allocated_mb={peak:.2f}".format(
                    target=row["target_text_tokens"],
                    actual=row["actual_text_tokens"],
                    input_len=row["input_tokens"],
                    repeat=row["repeat"],
                    delta=row["delta_peak_mb"],
                    peak=row["peak_allocated_mb"],
                )
            )

    output_paths = write_results(rows, args.output_dir, metric=args.metric)
    print(f"CSV: {output_paths['csv']}")
    print(f"JSON: {output_paths['json']}")
    print(f"Plot: {output_paths['plot']}")


if __name__ == "__main__":
    main()
