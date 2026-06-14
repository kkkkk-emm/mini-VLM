#!/usr/bin/env python3
"""Extract one GRPO failure case each from MME and POPE by live inference."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from PIL import Image
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.processors import get_image_processor
from evaluate import (
    autocast_context,
    generate_prediction,
    load_benchmark_dataset,
    load_model,
    parse_pope_answer,
    parse_yes_no,
    prepare_inputs,
    resolve_autocast_dtype,
)


InferFn = Callable[[Mapping[str, Any], str, int], str]


@dataclass
class FailureCase:
    benchmark: str
    index: int
    sample_id: str
    question_id: str
    category: str
    question: str
    answer: str
    prediction: str
    parsed_answer: str
    analysis: str
    image: Image.Image = field(repr=False)
    image_source: str = ""
    image_path: str = ""

    def to_record(self) -> dict[str, Any]:
        return {
            "benchmark": self.benchmark,
            "index": self.index,
            "sample_id": self.sample_id,
            "question_id": self.question_id,
            "category": self.category,
            "image_source": self.image_source,
            "image_path": self.image_path,
            "question": self.question,
            "answer": self.answer,
            "prediction": self.prediction,
            "parsed_answer": self.parsed_answer,
            "analysis": self.analysis,
        }


class ModelInferencer:
    def __init__(
        self,
        *,
        model,
        image_processor,
        device: torch.device,
        autocast_dtype: torch.dtype | None,
        prompt_style: str,
    ) -> None:
        self.model = model
        self.image_processor = image_processor
        self.device = device
        self.autocast_dtype = autocast_dtype
        self.prompt_style = prompt_style

    def __call__(self, sample: Mapping[str, Any], benchmark: str, max_new_tokens: int) -> str:
        input_ids, attention_mask, images = prepare_inputs(
            dict(sample),
            benchmark=benchmark,
            model=self.model,
            image_processor=self.image_processor,
            device=self.device,
            text_only=False,
            prompt_style=self.prompt_style,
        )
        with autocast_context(self.device, self.autocast_dtype):
            return generate_prediction(
                self.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                images=images,
                max_new_tokens=max_new_tokens,
            )


def _as_pil_image(sample: Mapping[str, Any]) -> Image.Image:
    image = sample["image"]
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Expected PIL image, got {type(image).__name__}")


def _sample_id(sample: Mapping[str, Any], index: int) -> str:
    return str(sample.get("index", sample.get("id", index)))


def _base_case(
    sample: Mapping[str, Any],
    *,
    benchmark: str,
    index: int,
    answer: str,
    prediction: str,
    parsed_answer: str,
    analysis: str,
) -> FailureCase:
    return FailureCase(
        benchmark=benchmark,
        index=index,
        sample_id=_sample_id(sample, index),
        question_id=str(sample.get("question_id", "")),
        category=str(sample.get("category", "")),
        question=str(sample["question"]),
        answer=answer,
        prediction=prediction,
        parsed_answer=parsed_answer,
        analysis=analysis,
        image=_as_pil_image(sample),
        image_source=str(sample.get("image_source", "")),
    )


def find_mme_failure(
    dataset: Iterable[Mapping[str, Any]],
    *,
    infer: InferFn,
    categories: set[str],
    max_new_tokens: int,
) -> FailureCase | None:
    for index, sample in enumerate(dataset):
        category = str(sample.get("category", ""))
        if category not in categories:
            continue
        answer = str(sample["answer"]).strip().lower()
        prediction = infer(sample, "mme", max_new_tokens)
        parsed_answer = parse_yes_no(prediction)
        if parsed_answer != answer:
            analysis = (
                f"该 MME 失败案例来自 {category} 类。模型把答案解析为 {parsed_answer or '无法解析'}，"
                "说明 GRPO 后模型在计数或位置等空间细节上仍有不足，容易忽略局部视觉证据。"
            )
            return _base_case(
                sample,
                benchmark="mme",
                index=index,
                answer=answer,
                prediction=prediction,
                parsed_answer=parsed_answer,
                analysis=analysis,
            )
    return None


def find_pope_failure(
    dataset: Iterable[Mapping[str, Any]],
    *,
    infer: InferFn,
    max_new_tokens: int,
) -> FailureCase | None:
    for index, sample in enumerate(dataset):
        answer = str(sample["answer"]).strip().lower()
        if answer != "no":
            continue
        prediction = infer(sample, "pope", max_new_tokens)
        parsed_answer = parse_pope_answer(prediction)
        if parsed_answer == "yes":
            analysis = (
                "该 POPE 失败案例中目标对象实际不存在，但模型仍回答 yes。"
                "这说明模型可能受到对象共现或问题文本的语言先验影响，同时对图像中的反证性视觉证据不足够敏感。"
            )
            return _base_case(
                sample,
                benchmark="pope",
                index=index,
                answer=answer,
                prediction=prediction,
                parsed_answer=parsed_answer,
                analysis=analysis,
            )
    return None


def write_failure_outputs(cases: Sequence[FailureCase | None], output_dir: Path) -> dict[str, str]:
    if any(case is None for case in cases):
        missing = [str(i) for i, case in enumerate(cases) if case is None]
        raise ValueError(f"Missing failure case at position(s): {', '.join(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    concrete_cases = [case for case in cases if case is not None]
    for case in concrete_cases:
        filename = f"{case.benchmark}_failure.png"
        case.image_path = filename
        case.image.save(output_dir / filename)

    payload = {"cases": [case.to_record() for case in concrete_cases]}
    json_path = output_dir / "failure_cases.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    markdown_path = output_dir / "failure_cases.md"
    markdown_path.write_text(_render_markdown(concrete_cases), encoding="utf-8")
    return {"json": str(json_path), "markdown": str(markdown_path)}


def _render_markdown(cases: Sequence[FailureCase]) -> str:
    lines = ["# GRPO 失败案例", ""]
    for case in cases:
        label = "MME failure" if case.benchmark == "mme" else "POPE failure"
        title = "MME 失败案例" if case.benchmark == "mme" else "POPE 失败案例"
        lines.extend(
            [
                f"## {title}",
                "",
                f"![{label}]({case.image_path})",
                "",
                f"- 数据集：{case.benchmark.upper()}",
                f"- 类别：{case.category}",
                f"- 样本 ID：{case.sample_id}",
                f"- 问题 ID：{case.question_id}",
                f"- 问题：{case.question}",
                f"- 正确答案：{case.answer}",
                f"- 模型错误回答：{case.prediction}",
                f"- 解析后的回答：{case.parsed_answer}",
                f"- 分析：{case.analysis}",
                "",
            ]
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one live-inference GRPO failure case each from MME and POPE.",
    )
    parser.add_argument("--checkpoint", required=True, help="GRPO checkpoint directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("figures/grpo_failure_cases"))
    parser.add_argument("--mme-path", default=None)
    parser.add_argument("--pope-path", default=None)
    parser.add_argument("--mme-categories", nargs="+", default=["count", "position"])
    parser.add_argument("--mme-max-new-tokens", type=int, default=4)
    parser.add_argument("--pope-max-new-tokens", type=int, default=2)
    parser.add_argument(
        "--prompt-style",
        choices=("strict", "original", "none"),
        default="strict",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16", "fp16", "fp32"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.mme_max_new_tokens <= 0:
        parser.error("--mme-max-new-tokens must be positive")
    if args.pope_max_new_tokens <= 0:
        parser.error("--pope-max-new-tokens must be positive")
    return args


def _validate_checkpoint(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Checkpoint must be a directory: {path}")


def _load_live_inferencer(args: argparse.Namespace) -> ModelInferencer:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    autocast_dtype = resolve_autocast_dtype(args.precision, device)
    model_args = argparse.Namespace(
        checkpoint=str(args.checkpoint),
        init_from_backbones=False,
        vision_backbone="./google/siglip2-base-patch16-512",
        language_backbone="./HuggingFaceTB/SmolLM2-360M-Instruct",
    )
    model = load_model(model_args, device)
    model.eval()
    image_processor = get_image_processor(
        model.cfg.max_img_size,
        model.cfg.vit_img_size,
        getattr(model.cfg, "resize_to_max_side_len", False),
    )
    return ModelInferencer(
        model=model,
        image_processor=image_processor,
        device=device,
        autocast_dtype=autocast_dtype,
        prompt_style=args.prompt_style,
    )


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.checkpoint = Path(args.checkpoint)
    _validate_checkpoint(args.checkpoint)

    inferencer = _load_live_inferencer(args)
    mme_dataset, _ = load_benchmark_dataset("mme", args.mme_path)
    pope_dataset, _ = load_benchmark_dataset("pope", args.pope_path)

    mme_case = find_mme_failure(
        tqdm(mme_dataset, desc="scan MME count/position failures"),
        infer=inferencer,
        categories=set(args.mme_categories),
        max_new_tokens=args.mme_max_new_tokens,
    )
    if mme_case is None:
        raise RuntimeError(
            "No MME failure found in the requested categories: "
            + ", ".join(args.mme_categories)
        )

    pope_case = find_pope_failure(
        tqdm(pope_dataset, desc="scan POPE no-to-yes failures"),
        infer=inferencer,
        max_new_tokens=args.pope_max_new_tokens,
    )
    if pope_case is None:
        raise RuntimeError("No POPE no-to-yes false positive failure found.")

    outputs = write_failure_outputs([mme_case, pope_case], args.output_dir)
    print(f"Wrote {outputs['json']}")
    print(f"Wrote {outputs['markdown']}")


if __name__ == "__main__":
    main()
