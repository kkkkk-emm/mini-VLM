#!/usr/bin/env python3
"""
Evaluate the custom mini-VLM model on MMStar.

Outputs:
- predictions_<mode>.csv
- predictions_<mode>.xlsx
- summary_<mode>.json

The official score uses the strict answer extraction patterns from the
MMStar evaluator.

An optional forced-choice diagnostic can additionally measure whether the
model prefers the correct option when restricted to A/B/C/D. The diagnostic
score is useful for debugging, but it is NOT the official MMStar score.
"""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Image as HFImage
from datasets import load_dataset
from tqdm.auto import tqdm

from data.processors import get_image_processor, get_image_string
from models.config import VLMConfig
from models.vision_language_model import VisionLanguageModel


PROMPT_SUFFIXES = {
    "original": (
        "\nAnswer with only the option letter: A, B, C, or D."
    ),
    "strict": (
        "\nChoose the correct option."
        "\nReply with exactly one uppercase letter: A, B, C, or D."
        "\nDo not provide an explanation."
        "\nAnswer:"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate mini-VLM on MMStar"
    )

    source_group = parser.add_mutually_exclusive_group(required=True)

    source_group.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Load a complete trained VLM checkpoint directory.",
    )

    source_group.add_argument(
        "--init-from-backbones",
        action="store_true",
        help=(
            "Initialize the VLM from the original SigLIP2 and SmolLM2 "
            "backbones without loading a VLM checkpoint. "
            "The projector remains randomly initialized."
        ),
    )

    parser.add_argument(
        "--vision-backbone",
        type=str,
        default="./google/siglip2-base-patch16-512",
        help=(
            "Local SigLIP2 directory used with --init-from-backbones."
        ),
    )

    parser.add_argument(
        "--language-backbone",
        type=str,
        default="./HuggingFaceTB/SmolLM2-360M-Instruct",
        help=(
            "Local SmolLM2 directory used with --init-from-backbones."
        ),
    )

    parser.add_argument(
        "--dataset-path",
        type=str,
        default="data/MMStar/mmstar.parquet",
        help="Local MMStar parquet file.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/mmstar",
        help="Directory used to save evaluation outputs.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help=(
            "Maximum number of generated tokens for each question. "
            "Use 32 for normal evaluation."
        ),
    )

    parser.add_argument(
        "--prompt-style",
        choices=tuple(PROMPT_SUFFIXES),
        default="strict",
        help=(
            "Prompt suffix used during evaluation. "
            "'strict' encourages a single-letter answer. "
            "'original' reproduces the previous prompt."
        ),
    )

    parser.add_argument(
        "--forced-choice-diagnostic",
        action="store_true",
        help=(
            "Additionally score A/B/C/D by conditional log-probability. "
            "This diagnostic is slower and is not the official MMStar score."
        ),
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Only evaluate the first N samples. "
            "Useful for debugging."
        ),
    )

    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help=(
            "Save intermediate results every N newly evaluated samples."
        ),
    )

    parser.add_argument(
        "--text-only",
        action="store_true",
        help=(
            "Remove images and evaluate the same VLM in text-only mode."
        ),
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Ignore existing CSV results and restart evaluation."
        ),
    )

    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
        help="Autocast precision used during evaluation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "Random seed. Generation is greedy, but a seed improves "
            "reproducibility."
        ),
    )

    args = parser.parse_args()

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    if args.save_every <= 0:
        parser.error("--save-every must be positive")

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive when provided")

    return args


def strict_choice(prediction: str) -> str:
    """
    Extract A/B/C/D according to the strict patterns accepted by the
    official MMStar evaluator.

    Accepted examples:
    - A
    - (A)
    - option A
    - the answer is A

    The function intentionally does not search for a letter inside a
    free-form explanation, because that would inflate the official score.
    """
    pred = str(prediction).lower().strip().replace("\n", " ")

    if not pred:
        return ""

    if pred[0] in "abcd":
        return pred[0].upper()

    if (
        len(pred) > 1
        and pred[0] == "("
        and pred[1] in "abcd"
    ):
        return pred[1].upper()

    if (
        pred.startswith("option ")
        and len(pred) > 7
        and pred[7] in "abcd"
    ):
        return pred[7].upper()

    if (
        pred.startswith("the answer is ")
        and len(pred) > 14
        and pred[14] in "abcd"
    ):
        return pred[14].upper()

    return ""


def get_source_benchmark(meta_info: Any) -> str:
    """
    Extract the original source benchmark from MMStar metadata.
    """
    if isinstance(meta_info, dict):
        return str(meta_info.get("source", ""))

    if isinstance(meta_info, str):
        try:
            parsed = json.loads(meta_info)
        except json.JSONDecodeError:
            return ""

        if isinstance(parsed, dict):
            return str(parsed.get("source", ""))

    return ""


def resolve_autocast_dtype(
    requested: str,
    device: torch.device,
) -> torch.dtype | None:
    """
    Select autocast dtype.

    On a CUDA GPU:
    - auto: bf16 when supported, otherwise fp16
    - bf16: explicitly use bf16
    - fp16: explicitly use fp16
    - fp32: disable autocast

    On CPU:
    - use fp32 for maximum compatibility
    """
    if requested == "fp32":
        return None

    if device.type != "cuda":
        print(
            f"Warning: precision={requested!r} is ignored on "
            f"{device.type}; evaluation will use fp32."
        )
        return None

    if requested == "auto":
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError(
                "The selected GPU does not support bf16. "
                "Use --precision fp16 or --precision auto."
            )
        return torch.bfloat16

    if requested == "fp16":
        return torch.float16

    raise ValueError(f"Unsupported precision: {requested}")


def autocast_context(
    device: torch.device,
    autocast_dtype: torch.dtype | None,
):
    """
    Return a CUDA autocast context when mixed precision is enabled.
    """
    if autocast_dtype is None:
        return nullcontext()

    return torch.autocast(
        device_type=device.type,
        dtype=autocast_dtype,
    )


def normalize_bool_column(
    frame: pd.DataFrame,
    column: str,
) -> None:
    """
    Normalize boolean columns after loading CSV files during resume.
    """
    if column not in frame.columns:
        return

    frame[column] = frame[column].map(
        lambda value: (
            value
            if isinstance(value, bool)
            else str(value).strip().lower() == "true"
        )
    )


def build_group_summary(
    frame: pd.DataFrame,
    *,
    value_column: str,
    group_columns: list[str],
) -> list[dict[str, Any]]:
    """
    Calculate count, correct count and accuracy by category.
    """
    if frame.empty:
        return []

    summary = (
        frame.groupby(
            group_columns,
            dropna=False,
        )[value_column]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(
            columns={
                "sum": "correct_count",
                "mean": "accuracy",
            }
        )
    )

    return summary.to_dict(orient="records")


def save_results(
    records: list[dict[str, Any]],
    output_dir: Path,
    mode: str,
    evaluation_config: dict[str, Any],
) -> None:
    """
    Save detailed predictions and summary statistics.
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = pd.DataFrame(records)

    if frame.empty:
        return

    frame = (
        frame.sort_values("index")
        .reset_index(drop=True)
    )

    normalize_bool_column(
        frame,
        "correct",
    )

    normalize_bool_column(
        frame,
        "diagnostic_correct",
    )

    csv_path = output_dir / f"predictions_{mode}.csv"
    xlsx_path = output_dir / f"predictions_{mode}.xlsx"
    summary_path = output_dir / f"summary_{mode}.json"

    frame.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    frame.to_excel(
        xlsx_path,
        index=False,
    )

    parsed_mask = (
        frame["parsed_choice"]
        .fillna("")
        != ""
    )

    parsed_frame = frame[parsed_mask]

    letter_only_mask = (
        frame["prediction"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.fullmatch(
            r"\(?[ABCDabcd]\)?[\.\:：]?"
        )
        .fillna(False)
    )

    summary: dict[str, Any] = {
        "mode": mode,
        "num_samples": int(len(frame)),
        "evaluation_config": evaluation_config,
        "official_accuracy": float(
            frame["correct"].mean()
        ),
        "unparsed_predictions": int(
            (~parsed_mask).sum()
        ),
        "parse_rate": float(
            parsed_mask.mean()
        ),
        "parsed_only_accuracy": (
            float(
                parsed_frame["correct"].mean()
            )
            if len(parsed_frame)
            else 0.0
        ),
        "letter_only_rate": float(
            letter_only_mask.mean()
        ),
        "choice_distribution": {
            choice: int(
                (
                    parsed_frame["parsed_choice"]
                    == choice
                ).sum()
            )
            for choice in "ABCD"
        },
        "category": build_group_summary(
            frame,
            value_column="correct",
            group_columns=["category"],
        ),
        "l2_category": build_group_summary(
            frame,
            value_column="correct",
            group_columns=[
                "category",
                "l2_category",
            ],
        ),
    }

    if "diagnostic_choice" in frame.columns:
        diagnostic_mask = (
            frame["diagnostic_choice"]
            .fillna("")
            != ""
        )

        diagnostic_frame = frame[
            diagnostic_mask
        ]

        summary["forced_choice_diagnostic"] = {
            "num_scored": int(
                diagnostic_mask.sum()
            ),
            "accuracy": (
                float(
                    diagnostic_frame[
                        "diagnostic_correct"
                    ].mean()
                )
                if len(diagnostic_frame)
                else 0.0
            ),
            "choice_distribution": {
                choice: int(
                    (
                        diagnostic_frame[
                            "diagnostic_choice"
                        ]
                        == choice
                    ).sum()
                )
                for choice in "ABCD"
            },
            "category": build_group_summary(
                diagnostic_frame,
                value_column="diagnostic_correct",
                group_columns=["category"],
            ),
            "l2_category": build_group_summary(
                diagnostic_frame,
                value_column="diagnostic_correct",
                group_columns=[
                    "category",
                    "l2_category",
                ],
            ),
        }

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def prepare_inputs(
    sample: dict[str, Any],
    model: VisionLanguageModel,
    image_processor,
    device: torch.device,
    text_only: bool,
    prompt_style: str,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
]:
    """
    Build the model input for one MMStar sample.
    """
    tokenizer = model.tokenizer

    image_string = ""
    images = None

    if not text_only:
        image = sample["image"].convert("RGB")

        processed_image, split_ratio = (
            image_processor(image)
        )

        # Keep this behavior consistent with generate.py
        # and the training pipeline.
        if (
            not hasattr(
                tokenizer,
                "global_image_token",
            )
            and split_ratio != (1, 1)
        ):
            processed_image = processed_image[1:]

        image_string = get_image_string(
            tokenizer,
            [split_ratio],
            model.cfg.mp_image_token_length,
        )

        images = processed_image.to(device)

    prompt = (
        image_string
        + str(sample["question"]).rstrip()
        + PROMPT_SUFFIXES[prompt_style]
    )

    messages = [
        {
            "role": "user",
            "content": prompt,
        }
    ]

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    input_ids = (
        encoded["input_ids"]
        .to(device)
    )

    attention_mask = encoded.get(
        "attention_mask"
    )

    if attention_mask is not None:
        attention_mask = attention_mask.to(
            device
        )

    return (
        input_ids,
        attention_mask,
        images,
    )


def generate_prediction(
    model: VisionLanguageModel,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    images: torch.Tensor | None,
    max_new_tokens: int,
) -> str:
    """
    Generate one free-form answer.

    This result is used to compute the official MMStar accuracy.
    """
    generated_ids = model.generate(
        input_ids=input_ids,
        images=images,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        greedy=True,
    )

    prediction = (
        model.tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )[0]
    )

    return prediction.strip()


def score_candidate(
    model: VisionLanguageModel,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    images: torch.Tensor | None,
    candidate: str,
) -> float:
    """
    Compute:

        log P(candidate | prompt, image)

    The implementation supports candidates containing one or multiple
    tokenizer tokens. A/B/C/D are normally single-token strings, but the
    generic implementation avoids silently relying on that assumption.
    """
    candidate_ids = (
        model.tokenizer.encode(
            candidate,
            add_special_tokens=False,
        )
    )

    if not candidate_ids:
        raise ValueError(
            f"Candidate {candidate!r} produced no token IDs."
        )

    candidate_tensor = torch.tensor(
        [candidate_ids],
        dtype=torch.long,
        device=input_ids.device,
    )

    full_input_ids = torch.cat(
        [
            input_ids,
            candidate_tensor,
        ],
        dim=1,
    )

    full_attention_mask = attention_mask

    if attention_mask is not None:
        candidate_mask = torch.ones(
            (
                attention_mask.size(0),
                len(candidate_ids),
            ),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        full_attention_mask = torch.cat(
            [
                attention_mask,
                candidate_mask,
            ],
            dim=1,
        )

    logits, _ = model(
        input_ids=full_input_ids,
        images=images,
        attention_mask=full_attention_mask,
    )

    log_probs = F.log_softmax(
        logits.float(),
        dim=-1,
    )

    prompt_length = input_ids.size(1)

    score = 0.0

    for offset, token_id in enumerate(
        candidate_ids
    ):
        prediction_position = (
            prompt_length
            - 1
            + offset
        )

        score += float(
            log_probs[
                0,
                prediction_position,
                token_id,
            ].item()
        )

    return score


def infer_forced_choice(
    model: VisionLanguageModel,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    images: torch.Tensor | None,
) -> str:
    """
    Diagnostic only.

    Restrict the model to A/B/C/D and select the option with the highest
    conditional probability. Do not treat this result as an official
    MMStar leaderboard score.
    """
    scores = {
        choice: score_candidate(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            candidate=choice,
        )
        for choice in "ABCD"
    }

    return max(
        scores,
        key=scores.get,
    )


def infer_one(
    sample: dict[str, Any],
    model: VisionLanguageModel,
    image_processor,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
    text_only: bool,
    prompt_style: str,
    max_new_tokens: int,
    forced_choice_diagnostic: bool,
) -> tuple[str, str]:
    """
    Evaluate one MMStar sample.
    """
    (
        input_ids,
        attention_mask,
        images,
    ) = prepare_inputs(
        sample=sample,
        model=model,
        image_processor=image_processor,
        device=device,
        text_only=text_only,
        prompt_style=prompt_style,
    )

    with autocast_context(
        device,
        autocast_dtype,
    ):
        prediction = generate_prediction(
            model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            max_new_tokens=max_new_tokens,
        )

        diagnostic_choice = ""

        if forced_choice_diagnostic:
            diagnostic_choice = (
                infer_forced_choice(
                    model,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=images,
                )
            )

    return (
        prediction,
        diagnostic_choice,
    )


def load_model(
    args: argparse.Namespace,
    device: torch.device,
) -> VisionLanguageModel:
    """
    Load either a trained checkpoint or the original backbones.
    """
    if args.init_from_backbones:
        print(
            "model_source : original local backbones"
        )
        print(
            f"vision       : {args.vision_backbone}"
        )
        print(
            f"language     : {args.language_backbone}"
        )
        print(
            "projector    : randomly initialized"
        )

        cfg = VLMConfig(
            vit_model_type=args.vision_backbone,
            lm_model_type=args.language_backbone,
            lm_tokenizer=args.language_backbone,
            vlm_load_backbone_weights=True,
        )

        model = (
            VisionLanguageModel
            .from_hf_backbones(cfg)
        )

    else:
        print(
            "model_source : checkpoint"
        )
        print(
            f"checkpoint   : {args.checkpoint}"
        )

        model = (
            VisionLanguageModel
            .from_pretrained(
                args.checkpoint
            )
        )

    return model.to(device)


def main() -> None:
    args = parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            args.seed
        )

    base_mode = (
        "text_only"
        if args.text_only
        else "multimodal"
    )

    mode = (
        f"{base_mode}_with_forced_choice"
        if args.forced_choice_diagnostic
        else base_mode
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_path = (
        output_dir
        / f"predictions_{mode}.csv"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    autocast_dtype = resolve_autocast_dtype(
        args.precision,
        device,
    )

    display_precision = (
        str(autocast_dtype)
        .removeprefix("torch.")
        if autocast_dtype is not None
        else "fp32"
    )

    print("=" * 80)
    print(
        f"device       : {device}"
    )
    print(
        f"precision    : {display_precision}"
    )
    print(
        f"mode         : {mode}"
    )
    print(
        f"output_dir   : {output_dir}"
    )
    print(
        f"dataset      : {args.dataset_path}"
    )
    print(
        f"prompt_style : {args.prompt_style}"
    )
    print(
        f"max_tokens   : {args.max_new_tokens}"
    )

    model = load_model(
        args,
        device,
    )

    model.eval()

    print("=" * 80)

    image_processor = get_image_processor(
        model.cfg.max_img_size,
        model.cfg.vit_img_size,
        getattr(
            model.cfg,
            "resize_to_max_side_len",
            False,
        ),
    )

    dataset = load_dataset(
        "parquet",
        data_files={
            "val": args.dataset_path,
        },
        split="val",
    )

    dataset = dataset.cast_column(
        "image",
        HFImage(),
    )

    if args.limit is not None:
        dataset = dataset.select(
            range(
                min(
                    args.limit,
                    len(dataset),
                )
            )
        )

    records: list[dict[str, Any]] = []
    completed_indices: set[int] = set()

    if (
        csv_path.exists()
        and not args.no_resume
    ):
        previous = pd.read_csv(
            csv_path,
            keep_default_na=False,
        )

        records = previous.to_dict(
            orient="records"
        )

        completed_indices = {
            int(index)
            for index in previous[
                "index"
            ].tolist()
        }

        print(
            "Resume enabled: loaded "
            f"{len(completed_indices)} "
            "completed samples."
        )

    evaluation_config = {
        "checkpoint": args.checkpoint,
        "init_from_backbones": bool(
            args.init_from_backbones
        ),
        "vision_backbone": (
            args.vision_backbone
            if args.init_from_backbones
            else None
        ),
        "language_backbone": (
            args.language_backbone
            if args.init_from_backbones
            else None
        ),
        "dataset_path": args.dataset_path,
        "text_only": bool(
            args.text_only
        ),
        "prompt_style": args.prompt_style,
        "prompt_suffix": (
            PROMPT_SUFFIXES[
                args.prompt_style
            ]
        ),
        "max_new_tokens": int(
            args.max_new_tokens
        ),
        "forced_choice_diagnostic": bool(
            args.forced_choice_diagnostic
        ),
        "precision": args.precision,
        "seed": int(args.seed),
    }

    newly_evaluated = 0

    try:
        with torch.inference_mode():
            for sample in tqdm(
                dataset,
                desc=f"MMStar/{mode}",
            ):
                index = int(
                    sample["index"]
                )

                if index in completed_indices:
                    continue

                (
                    prediction,
                    diagnostic_choice,
                ) = infer_one(
                    sample=sample,
                    model=model,
                    image_processor=(
                        image_processor
                    ),
                    device=device,
                    autocast_dtype=(
                        autocast_dtype
                    ),
                    text_only=args.text_only,
                    prompt_style=(
                        args.prompt_style
                    ),
                    max_new_tokens=(
                        args.max_new_tokens
                    ),
                    forced_choice_diagnostic=(
                        args.forced_choice_diagnostic
                    ),
                )

                parsed_choice = strict_choice(
                    prediction
                )

                answer = str(
                    sample["answer"]
                ).strip().upper()

                meta_info = sample[
                    "meta_info"
                ]

                record: dict[str, Any] = {
                    "index": index,
                    "question": (
                        sample["question"]
                    ),
                    "answer": answer,
                    "category": (
                        sample["category"]
                    ),
                    "l2_category": (
                        sample["l2_category"]
                    ),
                    "bench": (
                        get_source_benchmark(
                            meta_info
                        )
                    ),
                    "meta_info": json.dumps(
                        meta_info,
                        ensure_ascii=False,
                    ),
                    "prediction": prediction,
                    "parsed_choice": (
                        parsed_choice
                    ),
                    "correct": (
                        parsed_choice
                        == answer
                    ),
                }

                if args.forced_choice_diagnostic:
                    record[
                        "diagnostic_choice"
                    ] = diagnostic_choice

                    record[
                        "diagnostic_correct"
                    ] = (
                        diagnostic_choice
                        == answer
                    )

                records.append(
                    record
                )

                completed_indices.add(
                    index
                )

                newly_evaluated += 1

                if (
                    newly_evaluated
                    % args.save_every
                    == 0
                ):
                    save_results(
                        records=records,
                        output_dir=(
                            output_dir
                        ),
                        mode=mode,
                        evaluation_config=(
                            evaluation_config
                        ),
                    )

    except KeyboardInterrupt:
        print(
            "\nInterrupted. "
            "Saving completed samples "
            "before exit."
        )

        save_results(
            records=records,
            output_dir=output_dir,
            mode=mode,
            evaluation_config=(
                evaluation_config
            ),
        )

        raise

    save_results(
        records=records,
        output_dir=output_dir,
        mode=mode,
        evaluation_config=(
            evaluation_config
        ),
    )

    summary_path = (
        output_dir
        / f"summary_{mode}.json"
    )

    print(
        "\nEvaluation finished."
    )

    print(
        summary_path.read_text(
            encoding="utf-8"
        )
    )


if __name__ == "__main__":
    main()