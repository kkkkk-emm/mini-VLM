#!/usr/bin/env python3
"""Evaluate mini-VLM on MMStar, MME and POPE."""

from __future__ import annotations

import argparse
import glob
import json
import random
import re
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


BENCHMARKS = ("mmstar", "mme", "pope")
DEFAULT_DATASET_PATTERNS = {
    "mmstar": "data/MMStar/mmstar.parquet",
    "mme": "data/MME/data/*.parquet",
    "pope": "data/POPE/data/*.parquet",
}
MME_PERCEPTION_CATEGORIES = {
    "existence",
    "count",
    "position",
    "color",
    "posters",
    "celebrity",
    "scene",
    "landmark",
    "artwork",
    "OCR",
}
MME_COGNITION_CATEGORIES = {
    "commonsense_reasoning",
    "numerical_calculation",
    "text_translation",
    "code_reasoning",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate mini-VLM on MMStar, MME and POPE",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--checkpoint",
        help="Complete trained VLM checkpoint directory.",
    )
    source_group.add_argument(
        "--init-from-backbones",
        action="store_true",
        help="Initialize from local SigLIP2 and SmolLM2 backbones.",
    )
    parser.add_argument(
        "--benchmark",
        choices=(*BENCHMARKS, "all"),
        default="all",
        help="Benchmark to evaluate. 'all' evaluates all three sequentially.",
    )
    parser.add_argument(
        "--vision-backbone",
        default="./google/siglip2-base-patch16-512",
    )
    parser.add_argument(
        "--language-backbone",
        default="./HuggingFaceTB/SmolLM2-360M-Instruct",
    )
    parser.add_argument(
        "--mmstar-path",
        default=None,
        help="MMStar parquet file or glob. Defaults to data/MMStar/mmstar.parquet.",
    )
    parser.add_argument(
        "--mme-path",
        default=None,
        help="MME parquet file, directory or glob. Defaults to data/MME/data/*.parquet.",
    )
    parser.add_argument(
        "--pope-path",
        default=None,
        help="POPE parquet file, directory or glob. Defaults to data/POPE/data/*.parquet.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/evaluation",
        help="Output root. Each benchmark writes into its own subdirectory.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument(
        "--prompt-style",
        choices=("strict", "original"),
        default="strict",
    )
    parser.add_argument(
        "--forced-choice-diagnostic",
        action="store_true",
        help="Also score A/B/C/D log probabilities for MMStar only.",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")
    if args.save_every <= 0:
        parser.error("--save-every must be positive")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive when provided")
    return args


def resolve_dataset_files(benchmark: str, override: str | None = None) -> list[str]:
    if benchmark not in BENCHMARKS:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    source = Path(override) if override else Path(DEFAULT_DATASET_PATTERNS[benchmark])
    if source.is_dir():
        canonical_data_dir = source / "data"
        search_dir = canonical_data_dir if canonical_data_dir.is_dir() else source
        files = sorted(str(path) for path in search_dir.rglob("*.parquet"))
    elif source.is_file():
        files = [str(source)]
    else:
        files = sorted(glob.glob(str(source)))
    if not files:
        raise FileNotFoundError(f"No parquet files found for {benchmark}: {source}")
    return files


def load_benchmark_dataset(benchmark: str, override: str | None = None):
    files = resolve_dataset_files(benchmark, override)
    dataset = load_dataset(
        "parquet",
        data_files={"test": files},
        split="test",
    )
    return dataset.cast_column("image", HFImage()), files


def parse_choice(prediction: str) -> str:
    prediction = str(prediction).strip()
    match = re.match(
        r"^(?:\(([A-Da-d])\)|([A-Da-d])|option\s+([A-Da-d])|the answer is\s+([A-Da-d]))(?:$|\b|[\s.,:;])",
        prediction,
        flags=re.IGNORECASE,
    )
    if not match:
        return ""
    return next(group for group in match.groups() if group).upper()


def parse_yes_no(prediction: str) -> str:
    """Match the MME/lmms-eval yes/no answer processor."""

    prediction = str(prediction).lower().strip().replace(".", "")
    if prediction in {"yes", "no"}:
        return prediction
    prefix = prediction[:4]
    if "yes" in prefix:
        return "yes"
    if "no" in prefix:
        return "no"
    return ""


def parse_pope_answer(prediction: str) -> str:
    """Match the official POPE evaluator's first-sentence no/not rule."""

    first_sentence = str(prediction).split(".", 1)[0].replace(",", "")
    words = first_sentence.split()
    return "no" if any(word in {"No", "no", "not"} for word in words) else "yes"


def build_prompt(benchmark: str, question: str, prompt_style: str) -> str:
    question = str(question).rstrip()
    if benchmark == "mmstar":
        if prompt_style == "strict":
            suffix = (
                "\nChoose the correct option."
                "\nReply with exactly one uppercase letter: A, B, C, or D."
                "\nDo not provide an explanation."
                "\nAnswer:"
            )
        else:
            suffix = "\nAnswer with only the option letter: A, B, C, or D."
    else:
        if prompt_style == "strict":
            suffix = (
                "\nReply with exactly Yes or No."
                "\nDo not provide an explanation."
                "\nAnswer:"
            )
        else:
            suffix = "\nPlease answer yes or no."
    return question + suffix


def get_source_benchmark(meta_info: Any) -> str:
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


def normalize_bool_column(frame: pd.DataFrame, column: str) -> None:
    if column not in frame.columns:
        return
    frame[column] = frame[column].map(
        lambda value: value
        if isinstance(value, bool)
        else str(value).strip().lower() == "true"
    )


def build_group_summary(
    frame: pd.DataFrame,
    *,
    value_column: str,
    group_columns: list[str],
) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    summary = (
        frame.groupby(group_columns, dropna=False)[value_column]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"sum": "correct_count", "mean": "accuracy"})
    )
    return summary.to_dict(orient="records")


def _binary_metrics(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "count": 0,
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "yes_ratio": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }
    answer = frame["answer"].astype(str).str.lower()
    prediction = frame["parsed_answer"].astype(str).str.lower()
    tp = int(((answer == "yes") & (prediction == "yes")).sum())
    tn = int(((answer == "no") & (prediction == "no")).sum())
    fp = int(((answer == "no") & (prediction != "no")).sum())
    fn = int(((answer == "yes") & (prediction != "yes")).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "count": int(len(frame)),
        "accuracy": (tp + tn) / len(frame),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "yes_ratio": float((prediction == "yes").mean()),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def _parse_rate(frame: pd.DataFrame) -> float:
    if frame.empty or "parsed_answer" not in frame:
        return 0.0
    return float((frame["parsed_answer"].fillna("") != "").mean())


def _summarize_mmstar(frame: pd.DataFrame) -> dict[str, Any]:
    parsed_mask = frame["parsed_answer"].fillna("") != ""
    summary = {
        "official_accuracy": float(frame["correct"].mean()) if len(frame) else 0.0,
        "parse_rate": _parse_rate(frame),
        "unparsed_predictions": int((~parsed_mask).sum()),
        "choice_distribution": {
            choice: int((frame["parsed_answer"] == choice).sum())
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
            group_columns=["category", "l2_category"],
        ),
    }
    if "diagnostic_choice" in frame:
        diagnostic_mask = frame["diagnostic_choice"].fillna("") != ""
        diagnostic_frame = frame[diagnostic_mask].copy()
        normalize_bool_column(diagnostic_frame, "diagnostic_correct")
        summary["forced_choice_diagnostic"] = {
            "num_scored": int(len(diagnostic_frame)),
            "accuracy": float(diagnostic_frame["diagnostic_correct"].mean())
            if len(diagnostic_frame)
            else 0.0,
            "choice_distribution": {
                choice: int((diagnostic_frame["diagnostic_choice"] == choice).sum())
                for choice in "ABCD"
            },
            "category": build_group_summary(
                diagnostic_frame,
                value_column="diagnostic_correct",
                group_columns=["category"],
            ),
        }
    return summary


def _summarize_mme(frame: pd.DataFrame) -> dict[str, Any]:
    category_scores = []
    for category, category_frame in frame.groupby("category", dropna=False):
        grouped = category_frame.groupby("question_id")["correct"]
        paired = grouped.agg(["count", "all"])
        paired = paired[paired["count"] >= 2]
        accuracy = float(category_frame["correct"].mean() * 100)
        accuracy_plus = float(paired["all"].mean() * 100) if len(paired) else 0.0
        category_scores.append(
            {
                "category": category,
                "count": int(len(category_frame)),
                "pair_count": int(len(paired)),
                "accuracy": accuracy,
                "accuracy_plus": accuracy_plus,
                "score": accuracy + accuracy_plus,
            }
        )
    perception_score = sum(
        item["score"]
        for item in category_scores
        if item["category"] in MME_PERCEPTION_CATEGORIES
    )
    cognition_score = sum(
        item["score"]
        for item in category_scores
        if item["category"] in MME_COGNITION_CATEGORIES
    )
    return {
        "overall_accuracy": float(frame["correct"].mean()) if len(frame) else 0.0,
        "parse_rate": _parse_rate(frame),
        "perception_score": perception_score,
        "cognition_score": cognition_score,
        "total_score": perception_score + cognition_score,
        "category": category_scores,
    }


def _summarize_pope(frame: pd.DataFrame) -> dict[str, Any]:
    categories = []
    for category, category_frame in frame.groupby("category", dropna=False):
        categories.append({"category": category, **_binary_metrics(category_frame)})
    return {
        "overall": _binary_metrics(frame),
        "parse_rate": _parse_rate(frame),
        "category": categories,
    }


def summarize_records(
    benchmark: str,
    records: list[dict[str, Any]],
    *,
    mode: str,
    evaluation_config: dict[str, Any],
) -> dict[str, Any]:
    frame = pd.DataFrame(records)
    normalize_bool_column(frame, "correct")
    summary = {
        "benchmark": benchmark,
        "mode": mode,
        "num_samples": int(len(frame)),
        "evaluation_config": evaluation_config,
    }
    if benchmark == "mmstar":
        summary.update(_summarize_mmstar(frame))
    elif benchmark == "mme":
        summary.update(_summarize_mme(frame))
    elif benchmark == "pope":
        summary.update(_summarize_pope(frame))
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")
    return summary


def save_results(
    benchmark: str,
    records: list[dict[str, Any]],
    output_dir: Path,
    mode: str,
    evaluation_config: dict[str, Any],
) -> None:
    if not records:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(records).sort_values("index").reset_index(drop=True)
    normalize_bool_column(frame, "correct")
    normalize_bool_column(frame, "diagnostic_correct")
    frame.to_csv(
        output_dir / f"predictions_{mode}.csv",
        index=False,
        encoding="utf-8-sig",
    )
    try:
        frame.to_excel(output_dir / f"predictions_{mode}.xlsx", index=False)
    except ImportError:
        print("Warning: openpyxl is not installed; skipped XLSX output.")
    summary = summarize_records(
        benchmark,
        frame.to_dict(orient="records"),
        mode=mode,
        evaluation_config=evaluation_config,
    )
    (output_dir / f"summary_{mode}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def resolve_autocast_dtype(requested: str, device: torch.device) -> torch.dtype | None:
    if requested == "fp32" or device.type != "cuda":
        return None
    if requested == "auto":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if requested == "bf16":
        if not torch.cuda.is_bf16_supported():
            raise ValueError("The selected GPU does not support bf16.")
        return torch.bfloat16
    if requested == "fp16":
        return torch.float16
    raise ValueError(f"Unsupported precision: {requested}")


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def prepare_inputs(
    sample: dict[str, Any],
    *,
    benchmark: str,
    model: VisionLanguageModel,
    image_processor,
    device: torch.device,
    text_only: bool,
    prompt_style: str,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    image_string = ""
    images = None
    if not text_only:
        image = sample["image"].convert("RGB")
        processed_image, split_ratio = image_processor(image)
        if not hasattr(model.tokenizer, "global_image_token") and split_ratio != (1, 1):
            processed_image = processed_image[1:]
        image_string = get_image_string(
            model.tokenizer,
            [split_ratio],
            model.cfg.mp_image_token_length,
        )
        images = processed_image.to(device)
    prompt = image_string + build_prompt(benchmark, sample["question"], prompt_style)
    encoded = model.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    return input_ids, attention_mask, images


def generate_prediction(
    model: VisionLanguageModel,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    images: torch.Tensor | None,
    max_new_tokens: int,
) -> str:
    generated_ids = model.generate(
        input_ids=input_ids,
        images=images,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        greedy=True,
    )
    return model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


def score_candidate(
    model: VisionLanguageModel,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    images: torch.Tensor | None,
    candidate: str,
) -> float:
    candidate_ids = model.tokenizer.encode(candidate, add_special_tokens=False)
    if not candidate_ids:
        raise ValueError(f"Candidate {candidate!r} produced no token IDs.")
    candidate_tensor = torch.tensor(
        [candidate_ids],
        dtype=torch.long,
        device=input_ids.device,
    )
    full_input_ids = torch.cat([input_ids, candidate_tensor], dim=1)
    full_attention_mask = attention_mask
    if attention_mask is not None:
        candidate_mask = torch.ones(
            (attention_mask.size(0), len(candidate_ids)),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        full_attention_mask = torch.cat([attention_mask, candidate_mask], dim=1)
    logits, _ = model(
        input_ids=full_input_ids,
        images=images,
        attention_mask=full_attention_mask,
    )
    log_probs = F.log_softmax(logits.float(), dim=-1)
    prompt_length = input_ids.size(1)
    return sum(
        float(log_probs[0, prompt_length - 1 + offset, token_id].item())
        for offset, token_id in enumerate(candidate_ids)
    )


def infer_forced_choice(
    model: VisionLanguageModel,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None,
    images: torch.Tensor | None,
) -> str:
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
    return max(scores, key=scores.get)


def load_model(args: argparse.Namespace, device: torch.device) -> VisionLanguageModel:
    if args.init_from_backbones:
        cfg = VLMConfig(
            vit_model_type=args.vision_backbone,
            lm_model_type=args.language_backbone,
            lm_tokenizer=args.language_backbone,
            vlm_load_backbone_weights=True,
        )
        model = VisionLanguageModel.from_hf_backbones(cfg)
    else:
        model = VisionLanguageModel.from_pretrained(args.checkpoint)
    return model.to(device)


def _answer_and_parser(benchmark: str, sample: dict[str, Any]):
    if benchmark == "mmstar":
        return str(sample["answer"]).strip().upper(), parse_choice
    if benchmark == "pope":
        return str(sample["answer"]).strip().lower(), parse_pope_answer
    return str(sample["answer"]).strip().lower(), parse_yes_no


def build_record(
    benchmark: str,
    sample: dict[str, Any],
    *,
    index: int,
    prediction: str,
    diagnostic_choice: str = "",
) -> dict[str, Any]:
    answer, parser = _answer_and_parser(benchmark, sample)
    parsed_answer = parser(prediction)
    record = {
        "index": index,
        "sample_id": str(sample.get("index", sample.get("id", index))),
        "question": sample["question"],
        "answer": answer,
        "category": sample.get("category", ""),
        "prediction": prediction,
        "parsed_answer": parsed_answer,
        "correct": parsed_answer == answer,
    }
    if benchmark == "mmstar":
        meta_info = sample.get("meta_info", {})
        record.update(
            {
                "l2_category": sample.get("l2_category", ""),
                "bench": get_source_benchmark(meta_info),
                "meta_info": json.dumps(meta_info, ensure_ascii=False),
            }
        )
        if diagnostic_choice:
            record["diagnostic_choice"] = diagnostic_choice
            record["diagnostic_correct"] = diagnostic_choice == answer
    elif benchmark == "mme":
        record["question_id"] = str(sample["question_id"])
    else:
        record.update(
            {
                "question_id": str(sample["question_id"]),
                "image_source": str(sample.get("image_source", "")),
            }
        )
    return record


def _load_previous_records(csv_path: Path) -> tuple[list[dict[str, Any]], set[int]]:
    if not csv_path.exists():
        return [], set()
    previous = pd.read_csv(csv_path, keep_default_na=False)
    if "parsed_answer" not in previous and "parsed_choice" in previous:
        previous["parsed_answer"] = previous["parsed_choice"]
    return previous.to_dict(orient="records"), {int(value) for value in previous["index"]}


def run_benchmark(
    benchmark: str,
    *,
    args: argparse.Namespace,
    model: VisionLanguageModel,
    image_processor,
    device: torch.device,
    autocast_dtype: torch.dtype | None,
) -> None:
    override = getattr(args, f"{benchmark}_path")
    dataset, dataset_files = load_benchmark_dataset(benchmark, override)
    if args.limit is not None:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    mode = "text_only" if args.text_only else "multimodal"
    if benchmark == "mmstar" and args.forced_choice_diagnostic:
        mode += "_with_forced_choice"
    output_dir = Path(args.output_dir) / benchmark
    csv_path = output_dir / f"predictions_{mode}.csv"
    records, completed_indices = (
        ([], set()) if args.no_resume else _load_previous_records(csv_path)
    )
    evaluation_config = {
        "checkpoint": args.checkpoint,
        "init_from_backbones": bool(args.init_from_backbones),
        "dataset_files": dataset_files,
        "text_only": bool(args.text_only),
        "prompt_style": args.prompt_style,
        "max_new_tokens": int(args.max_new_tokens),
        "forced_choice_diagnostic": bool(
            benchmark == "mmstar" and args.forced_choice_diagnostic
        ),
        "precision": args.precision,
        "seed": int(args.seed),
    }

    newly_evaluated = 0
    try:
        with torch.inference_mode():
            for index, sample in enumerate(tqdm(dataset, desc=f"{benchmark}/{mode}")):
                if index in completed_indices:
                    continue
                input_ids, attention_mask, images = prepare_inputs(
                    sample,
                    benchmark=benchmark,
                    model=model,
                    image_processor=image_processor,
                    device=device,
                    text_only=args.text_only,
                    prompt_style=args.prompt_style,
                )
                with autocast_context(device, autocast_dtype):
                    prediction = generate_prediction(
                        model,
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        images=images,
                        max_new_tokens=args.max_new_tokens,
                    )
                    diagnostic_choice = ""
                    if benchmark == "mmstar" and args.forced_choice_diagnostic:
                        diagnostic_choice = infer_forced_choice(
                            model,
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            images=images,
                        )
                records.append(
                    build_record(
                        benchmark,
                        sample,
                        index=index,
                        prediction=prediction,
                        diagnostic_choice=diagnostic_choice,
                    )
                )
                completed_indices.add(index)
                newly_evaluated += 1
                if newly_evaluated % args.save_every == 0:
                    save_results(
                        benchmark,
                        records,
                        output_dir,
                        mode,
                        evaluation_config,
                    )
    except KeyboardInterrupt:
        save_results(benchmark, records, output_dir, mode, evaluation_config)
        raise

    save_results(benchmark, records, output_dir, mode, evaluation_config)
    summary_path = output_dir / f"summary_{mode}.json"
    print(f"\n{benchmark.upper()} evaluation finished: {summary_path}")
    print(summary_path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = resolve_autocast_dtype(args.precision, device)
    model = load_model(args, device)
    model.eval()
    image_processor = get_image_processor(
        model.cfg.max_img_size,
        model.cfg.vit_img_size,
        getattr(model.cfg, "resize_to_max_side_len", False),
    )
    benchmarks = BENCHMARKS if args.benchmark == "all" else (args.benchmark,)
    for benchmark in benchmarks:
        run_benchmark(
            benchmark,
            args=args,
            model=model,
            image_processor=image_processor,
            device=device,
            autocast_dtype=autocast_dtype,
        )


if __name__ == "__main__":
    main()
