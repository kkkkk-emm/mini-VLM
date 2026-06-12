#!/usr/bin/env python3
"""Draw report figures for the mini-VLM final report."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import textwrap
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.custom_transforms import DynamicResize
from models.config import VLMConfig

FIGURE_PIXELS = (1774, 887)
DEFAULT_DPI = 150
DEFAULT_IMAGE = Path("data/eval_images/image-01-golden-dog-balloons.jpg")
DEFAULT_OUTPUT_DIR = Path("figures")
DEFAULT_REPORT = Path("mini-vlm-final-report.md")

PRETRAIN_PATH = Path("data/pretrain_i2t.parquet")
SFT_PATH = Path("data/sft_i2t.parquet")
CORRECTIVE_SFT_PATH = Path("data/corrective_sft_30k.parquet")
GRPO_PATH = Path("data/ai2d_grpo_mmstar.parquet")
PREDICTION_CSV_PATH = Path("results/mme/GRPO-50/predictions_multimodal.csv")

FIGURE_FILES: dict[str, str] = {
    "2-1": "fig-2-1-mini-vlm-overall-flow.png",
    "2-2": "fig-2-2-text-visual-token-alignment-v2.png",
    "3-2": "fig-3-2-dynamic-resize-comparison.png",
    "3-3": "fig-3-3-normalization-histogram.png",
    "3-5": "fig-3-5-visual-token-string.png",
    "3-6": "fig-3-6-image-token-mismatch-flow.png",
    "4-8": "fig-4-8-image-embedding-replacement.png",
    "4-9": "fig-4-9-generate-prefill-decode.png",
    "5-1": "fig-5-1-training-dataset-counts.png",
    "5-2": "fig-5-2-pretrain-sample-format.png",
    "5-3": "fig-5-3-assistant-only-labels.png",
    "5-4": "fig-5-4-corrective-sft-source-distribution.png",
    "5-5": "fig-5-5-corrective-sft-task-distribution.png",
    "5-7": "fig-5-7-batch-collate.png",
    "6-8": "fig-6-8-corrective-sft-data-composition.png",
    "8-2": "fig-8-2-mme-category-structure.png",
    "8-3": "fig-8-3-mme-scoring.png",
    "8-6": "fig-8-6-prediction-table-example.png",
    "9-1": "fig-9-1-mme-stage-ablation.png",
    "9-2": "fig-9-2-mme-baseline-comparison.png",
    "9-3": "fig-9-3-pope-stage-metrics.png",
    "9-4": "fig-9-4-pope-baseline-comparison.png",
    "9-5": "fig-9-5-training-stage-objectives.png",
    "10-1": "fig-10-1-image-token-alignment-check.png",
    "10-2": "fig-10-2-global-local-tradeoff.png",
    "10-3": "fig-10-3-ability-requirements.png",
    "10-5": "fig-10-5-grpo-benefits-limitations.png",
    "10-6": "fig-10-6-future-roadmap.png",
}

REPORT_REFERENCES: list[tuple[str, str, str]] = [
    ("2-1：mini-VLM 整体流程图", "mini-VLM 整体流程图", "2-1"),
    ("2-2：文本 token 与视觉 token 对齐示意图", "文本 token 与视觉 token 对齐示意图", "2-2"),
    ("3-5：视觉 token 字符串示例图", "视觉 token 字符串示例图", "3-5"),
    ("3-6：image token mismatch 错误定位流程图", "image token mismatch 错误定位流程图", "3-6"),
    ("4-8：图像 embedding 替换流程图", "图像 embedding 替换流程图", "4-8"),
    ("4-9：generate prefill/decode 流程图", "generate prefill/decode 流程图", "4-9"),
    ("5-3：assistant-only labels 示意图", "assistant-only labels 示意图", "5-3"),
    ("5-7：batch collate 示意图", "batch collate 示意图", "5-7"),
    ("6-8：纠偏 SFT 数据组成图", "纠偏 SFT 数据组成图", "6-8"),
    ("8-2：MME 类别结构图", "MME 类别结构图", "8-2"),
    ("8-3：MME 评分方式示意图", "MME 评分方式示意图", "8-3"),
    ("8-6：prediction 表格示例", "prediction 表格示例", "8-6"),
    ("9-1：MME 阶段消融柱状图", "MME 阶段消融柱状图", "9-1"),
    ("9-2：MME baseline 对比图", "MME baseline 对比图", "9-2"),
    ("9-3：POPE 阶段指标变化图", "POPE 阶段指标变化图", "9-3"),
    ("9-4：POPE baseline 对比图", "POPE baseline 对比图", "9-4"),
    ("9-5：不同训练阶段优化目标示意图", "不同训练阶段优化目标示意图", "9-5"),
    ("10-1：图像 token 对齐检查图", "图像 token 对齐检查图", "10-1"),
    ("10-2：全局信息与局部细节权衡图", "全局信息与局部细节权衡图", "10-2"),
    ("10-3：不同能力对模型要求的对比图", "不同能力对模型要求的对比图", "10-3"),
    ("10-5：GRPO 提升与局限总结图", "GRPO 提升与局限总结图", "10-5"),
    ("10-6：后续改进路线图", "后续改进路线图", "10-6"),
]

PALETTE = {
    "blue": "#6A8FBF",
    "cyan": "#78B7C5",
    "green": "#8AB17D",
    "yellow": "#E9C46A",
    "orange": "#DDA15E",
    "red": "#D88C9A",
    "purple": "#9D8AC7",
    "gray": "#8D99AE",
    "dark": "#263238",
    "line": "#D8DEE9",
    "soft": "#F7F9FB",
    "panel": "#F7F9FB",
}

MME_STAGE_ROWS = [
    ("Before\npretraining", 101.44, 10.36, 111.80),
    ("Pretrain\n10K", 512.60, 178.93, 691.53),
    ("Pretrain\n20K", 525.10, 183.21, 708.31),
    ("+ SFT", 521.13, 133.57, 654.70),
    ("+ SFT\n& GRPO", 837.57, 217.14, 1054.71),
]

MME_BASELINE_ROWS = [
    ("MiniGPT-4", 13.0, 725.95),
    ("PandaGPT", 13.0, 871.16),
    ("Multimodal-GPT", 9.0, 881.51),
    ("VisualGLM-6B", 7.8, 887.10),
    ("ImageBind-LLM", 7.0, 989.34),
    ("VPGTrans", 7.0, 1039.74),
    ("Mini-VLM\n(Ours)", 0.46, 1054.72),
]

POPE_STAGE_ROWS = [
    ("Before\npretrain", 50.00, 65.96, 96.87),
    ("Pretrain\n10K", 50.03, 66.34, 98.43),
    ("Pretrain\n20K", 49.96, 65.72, 95.98),
    ("+ SFT", 50.09, 66.71, 99.91),
    ("+ SFT\n& GRPO", 63.32, 72.12, 81.54),
]

POPE_BASELINE_ROWS = [
    ("MultiModal-GPT", 50.01, 66.67, 99.99),
    ("mPLUG-Owl", 51.53, 67.22, 97.84),
    ("LLaVA", 52.54, 67.78, 97.28),
    ("Mini-VLM\n(Ours)", 63.32, 72.12, 81.54),
]


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "Source Han Sans SC",
                "Arial Unicode MS",
                "DejaVu Sans",
                "sans-serif",
            ],
            "axes.unicode_minus": False,
            "font.size": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "axes.edgecolor": "#455A64",
            "xtick.color": "#455A64",
            "ytick.color": "#455A64",
            "text.color": PALETTE["dark"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def new_figure(dpi: int) -> plt.Figure:
    width_px, height_px = FIGURE_PIXELS
    return plt.figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)


def save_figure(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    with Image.open(path) as image:
        image.convert("RGBA").save(path)


def _axis(fig: plt.Figure):
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return ax


def _wrapped(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, replace_whitespace=False))


def _limited_wrapped(text: str, width: int, max_lines: int) -> str:
    lines = textwrap.wrap(text, width=width, replace_whitespace=False)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1] + ["..."]
    return "\n".join(lines)


def _format_count(value: int) -> str:
    return f"{value:,}"


def _display_label(label: str) -> str:
    mapping = {
        "ai2d": "AI2D",
        "chartqa": "ChartQA",
        "vsr": "VSR",
        "scienceqa": "ScienceQA",
        "sft_replay": "SFT replay",
        "multiple_choice": "multiple_choice",
        "yes_no": "yes_no",
        "chart_short": "chart_short",
        "replay": "replay",
    }
    return mapping.get(label, label)


def _box(
    ax,
    xy: tuple[float, float],
    size: tuple[float, float],
    text: str,
    *,
    fc: str = "#F7F9FB",
    ec: str = "#CFD8DC",
    fontsize: float = 13,
    weight: str = "normal",
    radius: float = 0.02,
    align: str = "center",
):
    x, y = xy
    w, h = size
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.014,rounding_size={radius}",
        facecolor=fc,
        edgecolor=ec,
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ha = "center" if align == "center" else "left"
    tx = x + w / 2 if align == "center" else x + 0.02
    ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fontsize, fontweight=weight)
    return patch


def _arrow(ax, start: tuple[float, float], end: tuple[float, float], *, color: str = "#6A8FBF", lw: float = 1.8):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=18,
            linewidth=lw,
            color=color,
            shrinkA=3,
            shrinkB=3,
        )
    )


def _token_strip(ax, x: float, y: float, labels: list[str], colors: list[str], *, w: float = 0.055, h: float = 0.06):
    for idx, label in enumerate(labels):
        xi = x + idx * (w + 0.008)
        ax.add_patch(Rectangle((xi, y), w, h, facecolor=colors[idx % len(colors)], edgecolor="white", linewidth=1))
        ax.text(xi + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)


def build_dataset_counts(
    *,
    pretrain: Path = PRETRAIN_PATH,
    sft: Path = SFT_PATH,
    corrective: Path = CORRECTIVE_SFT_PATH,
    grpo: Path = GRPO_PATH,
) -> dict[str, int]:
    paths = OrderedDict(
        [
            ("Pretrain", pretrain),
            ("SFT", sft),
            ("Corrective SFT", corrective),
            ("AI2D-GRPO", grpo),
        ]
    )
    return {label: pq.ParquetFile(path).metadata.num_rows for label, path in paths.items()}


def count_column_values(path: Path, column: str) -> dict[str, int]:
    table = pq.read_table(path, columns=[column])
    series = table.to_pandas()[column]
    counts = series.value_counts(dropna=False)
    return {str(key): int(value) for key, value in counts.items()}


def read_pretrain_sample(path: Path = PRETRAIN_PATH) -> dict[str, str]:
    parquet_file = pq.ParquetFile(path)
    table = parquet_file.read_row_group(0, columns=["conversations"])
    if table.num_rows == 0:
        raise ValueError(f"No rows found in {path}")

    raw = table.column("conversations")[0].as_py()
    conversations = json.loads(raw) if isinstance(raw, str) else raw
    user_text = ""
    assistant_text = ""
    for item in conversations:
        role = item.get("role")
        content = item.get("content", "")
        if role == "user" and not user_text:
            user_text = content
        elif role == "assistant" and not assistant_text:
            assistant_text = content
        if user_text and assistant_text:
            break
    if not user_text or not assistant_text:
        raise ValueError("Could not find both user and assistant messages in pretrain sample")
    return {"user": user_text, "assistant": assistant_text}


def _read_prediction_rows(path: Path = PREDICTION_CSV_PATH, limit: int = 4) -> list[dict[str, str]]:
    if not path.exists():
        return [
            {
                "index": "0",
                "question": "Is there a dog in the image?",
                "answer": "yes",
                "prediction": "Yes",
                "parsed_answer": "yes",
                "correct": "True",
                "category": "existence",
            },
            {
                "index": "1",
                "question": "Is there a cat in the image?",
                "answer": "no",
                "prediction": "No",
                "parsed_answer": "no",
                "correct": "True",
                "category": "existence",
            },
        ]
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = []
        for row in reader:
            rows.append(row)
            if len(rows) >= limit:
                break
    return rows


def draw_dynamic_resize_comparison(image_path: Path, output_path: Path, dpi: int) -> None:
    cfg = VLMConfig()
    image = Image.open(image_path).convert("RGB")
    resized = DynamicResize(
        patch_size=cfg.vit_img_size,
        max_size=cfg.max_img_size,
        resize_to_max=cfg.resize_to_max_side_len,
    )(image)

    fig = new_figure(dpi)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 0.13, 1.0], wspace=0.06)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 2])]
    arrow_ax = fig.add_subplot(gs[0, 1])
    arrow_ax.axis("off")

    panels = [
        (axes[0], image, "缩放前", f"{image.width} x {image.height} px"),
        (axes[1], resized, "缩放后", f"{resized.width} x {resized.height} px, 适配 {cfg.vit_img_size} 网格"),
    ]
    for ax, panel_image, label, meta in panels:
        ax.imshow(panel_image)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(PALETTE["line"])
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=18,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="none", alpha=0.86),
        )
        ax.text(
            0.02,
            0.06,
            meta,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=13,
            bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="none", alpha=0.86),
        )

    width, height = resized.size
    for x in range(cfg.vit_img_size, width, cfg.vit_img_size):
        axes[1].axvline(x - 0.5, color="white", lw=1.2, alpha=0.8)
    for y in range(cfg.vit_img_size, height, cfg.vit_img_size):
        axes[1].axhline(y - 0.5, color="white", lw=1.2, alpha=0.8)

    arrow_ax.add_patch(
        FancyArrowPatch(
            (0.05, 0.5),
            (0.95, 0.5),
            arrowstyle="-|>",
            mutation_scale=28,
            linewidth=2.0,
            color=PALETTE["blue"],
            transform=arrow_ax.transAxes,
        )
    )
    arrow_ax.text(0.5, 0.57, "DynamicResize", ha="center", va="bottom", fontsize=11)
    fig.subplots_adjust(left=0.035, right=0.965, top=0.94, bottom=0.06)
    save_figure(fig, output_path, dpi)


def draw_normalization_histogram(image_path: Path, output_path: Path, dpi: int) -> None:
    image = Image.open(image_path).convert("RGB")
    raw = np.asarray(image, dtype=np.float32) / 255.0
    normalized = (raw - 0.5) / 0.5
    channels = [("R", "#C95F5F"), ("G", "#5F9E6E"), ("B", "#5D7FBF")]

    fig = new_figure(dpi)
    axes = fig.subplots(1, 2, sharey=True)
    specs = [
        (axes[0], raw, np.linspace(0, 1, 60), "归一化前", "像素值 [0, 1]"),
        (axes[1], normalized, np.linspace(-1, 1, 60), "归一化后", "像素值 [-1, 1]"),
    ]
    for ax, data, bins, label, xlabel in specs:
        for channel_index, (channel, color) in enumerate(channels):
            values = data[:, :, channel_index].ravel()
            ax.hist(values, bins=bins, density=True, histtype="stepfilled", alpha=0.30, color=color, label=channel)
            ax.hist(values, bins=bins, density=True, histtype="step", color=color, lw=1.5)
        ax.grid(axis="y", color=PALETTE["line"], linewidth=0.8, alpha=0.85)
        ax.set_xlabel(xlabel)
        ax.text(0.02, 0.96, label, transform=ax.transAxes, ha="left", va="top", fontsize=17, fontweight="bold")
    axes[0].set_ylabel("密度")
    axes[0].legend(loc="upper right", frameon=False, ncol=3)
    axes[1].axvline(0, color=PALETTE["dark"], linestyle="--", linewidth=1.1, alpha=0.7)
    fig.subplots_adjust(left=0.075, right=0.975, top=0.93, bottom=0.14, wspace=0.16)
    save_figure(fig, output_path, dpi)


def draw_overall_flow(output_path: Path, dpi: int) -> None:
    fig = new_figure(dpi)
    ax = _axis(fig)
    top = [
        ("输入图像", "RGB\nPIL/Image"),
        ("图像处理", "resize\nnormalize\nsplit"),
        ("ViT", "patch\nfeatures"),
        ("Projector", "64 visual\ntokens"),
        ("替换 embedding", "image token\npositions"),
        ("Decoder", "autoregressive\nLM"),
        ("输出回答", "text\nanswer"),
    ]
    xs = np.linspace(0.055, 0.835, len(top))
    for idx, (title, body) in enumerate(top):
        _box(ax, (xs[idx], 0.58), (0.12, 0.18), f"{title}\n{body}", fc="#EEF5FF" if idx < 4 else "#F2F7EF", fontsize=11.5, weight="bold")
        if idx < len(top) - 1:
            _arrow(ax, (xs[idx] + 0.12, 0.67), (xs[idx + 1], 0.67))
    _box(ax, (0.08, 0.23), (0.22, 0.14), "文本问题\n+ 图像 token 字符串", fc="#FFF7E6", fontsize=13, weight="bold")
    _box(ax, (0.36, 0.23), (0.20, 0.14), "Tokenizer\ninput_ids", fc="#FFF7E6", fontsize=13, weight="bold")
    _box(ax, (0.62, 0.23), (0.20, 0.14), "Token embedding\n文字位置保持不变", fc="#FFF7E6", fontsize=13, weight="bold")
    _arrow(ax, (0.30, 0.30), (0.36, 0.30), color=PALETTE["orange"])
    _arrow(ax, (0.56, 0.30), (0.62, 0.30), color=PALETTE["orange"])
    _arrow(ax, (0.72, 0.37), (0.61, 0.58), color=PALETTE["orange"])
    ax.text(0.5, 0.10, "语言模型最终接收的是融合后的 embedding 序列，不需要额外的图像输入接口", ha="center", fontsize=14)
    save_figure(fig, output_path, dpi)


def draw_text_visual_alignment(output_path: Path, dpi: int) -> None:
    fig = new_figure(dpi)
    ax = _axis(fig)
    _box(ax, (0.06, 0.70), (0.88, 0.10), "文本 token 序列", fc="#F7F9FB", fontsize=15, weight="bold")
    text_labels = ["<bos>", "User", "Q", "<|global|>", "<|image|>", "...", "<|image|>", "Answer"]
    colors = ["#ECEFF1", "#ECEFF1", "#ECEFF1", "#FFE6B3", "#CDE7F0", "#CDE7F0", "#CDE7F0", "#E4F0DF"]
    _token_strip(ax, 0.12, 0.61, text_labels, colors, w=0.075, h=0.07)
    _box(ax, (0.18, 0.36), (0.20, 0.12), "全局图\n1 block", fc="#FFF7E6", fontsize=13, weight="bold")
    _box(ax, (0.44, 0.36), (0.20, 0.12), "局部图 r1c1\n1 block", fc="#EEF5FF", fontsize=13, weight="bold")
    _box(ax, (0.70, 0.36), (0.20, 0.12), "局部图 r1c2\n1 block", fc="#EEF5FF", fontsize=13, weight="bold")
    for x in [0.28, 0.54, 0.80]:
        _box(ax, (x - 0.06, 0.20), (0.12, 0.08), "64 个\n<|image|>", fc="#E4F0DF", fontsize=12, weight="bold")
        _arrow(ax, (x, 0.36), (x, 0.28), color=PALETTE["green"])
    ax.text(0.50, 0.08, "文本中的 image token 个数 = 图像块数 x mp_image_token_length", ha="center", fontsize=15, fontweight="bold")
    save_figure(fig, output_path, dpi)


def draw_visual_token_string(output_path: Path, dpi: int) -> None:
    fig = new_figure(dpi)
    ax = _axis(fig)
    pieces = [
        ("<|global_image|>", "#FFE6B3", 0.16),
        ("<|image|> x64", "#CDE7F0", 0.15),
        ("<row_1_col_1>", "#E4F0DF", 0.15),
        ("<|image|> x64", "#CDE7F0", 0.15),
        ("<row_1_col_2>", "#E4F0DF", 0.15),
        ("<|image|> x64", "#CDE7F0", 0.15),
    ]
    x = 0.05
    for idx, (label, color, width) in enumerate(pieces):
        _box(ax, (x, 0.55), (width, 0.13), label, fc=color, fontsize=12.5, weight="bold")
        ax.text(x + width / 2, 0.48, "标记" if idx % 2 == 0 else "视觉占位", ha="center", fontsize=11)
        x += width + 0.012
    _box(ax, (0.08, 0.22), (0.20, 0.13), "grid = (1, 2)\n全局图 + 2 个局部块", fc="#F7F9FB", fontsize=13)
    _box(ax, (0.40, 0.22), (0.20, 0.13), "3 个图像块\n3 x 64 = 192", fc="#F7F9FB", fontsize=13)
    _box(ax, (0.72, 0.22), (0.20, 0.13), "tokenizer 编码后\n等待 embedding 替换", fc="#F7F9FB", fontsize=13)
    _arrow(ax, (0.28, 0.285), (0.40, 0.285))
    _arrow(ax, (0.60, 0.285), (0.72, 0.285))
    save_figure(fig, output_path, dpi)


def draw_image_token_mismatch_flow(output_path: Path, dpi: int) -> None:
    fig = new_figure(dpi)
    ax = _axis(fig)
    steps = [
        ("图像处理", "得到 N 个图像块"),
        ("Projector", "每块输出 64 个 token"),
        ("期望数量", "expected = N x 64"),
        ("文本统计", "count(<|image|>)"),
        ("一致?", "相等则进入训练\n不等则跳过样本"),
    ]
    xs = [0.05, 0.23, 0.41, 0.59, 0.77]
    for idx, (title, body) in enumerate(steps):
        _box(ax, (xs[idx], 0.55), (0.15, 0.16), f"{title}\n{body}", fc="#EEF5FF" if idx < 4 else "#FFF0F0", fontsize=11.5, weight="bold")
        if idx < len(steps) - 1:
            _arrow(ax, (xs[idx] + 0.15, 0.63), (xs[idx + 1], 0.63))
    _box(ax, (0.20, 0.26), (0.24, 0.12), "示例：5 个图像块\nexpected = 5 x 64 = 320", fc="#F2F7EF", fontsize=13)
    _box(ax, (0.56, 0.26), (0.24, 0.12), "实际：256 个 image token\n返回 image_token_mismatch", fc="#FFF0F0", fontsize=13)
    _arrow(ax, (0.44, 0.32), (0.56, 0.32), color=PALETTE["red"])
    save_figure(fig, output_path, dpi)


def draw_image_embedding_replacement(output_path: Path, dpi: int) -> None:
    fig = new_figure(dpi)
    ax = _axis(fig)
    _box(ax, (0.06, 0.70), (0.23, 0.12), "input_ids\n含 <|image|> 占位", fc="#FFF7E6", fontsize=13, weight="bold")
    _box(ax, (0.39, 0.70), (0.23, 0.12), "token_embedding\n[B, T, 960]", fc="#EEF5FF", fontsize=13, weight="bold")
    _box(ax, (0.71, 0.70), (0.23, 0.12), "image token mask\nTrue 位置待替换", fc="#F7F9FB", fontsize=13, weight="bold")
    _arrow(ax, (0.29, 0.76), (0.39, 0.76))
    _arrow(ax, (0.62, 0.76), (0.71, 0.76))
    _box(ax, (0.12, 0.34), (0.25, 0.13), "ViT + Projector\n[N, 64, 960]", fc="#E4F0DF", fontsize=13, weight="bold")
    _box(ax, (0.48, 0.34), (0.25, 0.13), "view(-1, 960)\n与 mask True 数匹配", fc="#E4F0DF", fontsize=13, weight="bold")
    _box(ax, (0.36, 0.10), (0.30, 0.12), "fused embedding\n文字 embedding + 图像 embedding", fc="#DCECCB", fontsize=13, weight="bold")
    _arrow(ax, (0.37, 0.405), (0.48, 0.405), color=PALETTE["green"])
    _arrow(ax, (0.60, 0.34), (0.51, 0.22), color=PALETTE["green"])
    _arrow(ax, (0.82, 0.70), (0.58, 0.22), color=PALETTE["blue"])
    save_figure(fig, output_path, dpi)


def draw_generate_prefill_decode(output_path: Path, dpi: int) -> None:
    fig = new_figure(dpi)
    ax = _axis(fig)
    _box(ax, (0.06, 0.66), (0.18, 0.13), "融合 prompt\n图像 + 文本 embedding", fc="#EEF5FF", fontsize=12, weight="bold")
    _box(ax, (0.31, 0.66), (0.18, 0.13), "Prefill\n处理完整 prompt", fc="#F2F7EF", fontsize=12, weight="bold")
    _box(ax, (0.56, 0.66), (0.18, 0.13), "KV cache\n保存历史 K/V", fc="#F2F7EF", fontsize=12, weight="bold")
    _box(ax, (0.78, 0.66), (0.16, 0.13), "当前 logits\n选下一个 token", fc="#FFF7E6", fontsize=12, weight="bold")
    _arrow(ax, (0.24, 0.725), (0.31, 0.725))
    _arrow(ax, (0.49, 0.725), (0.56, 0.725))
    _arrow(ax, (0.74, 0.725), (0.78, 0.725))
    for idx, label in enumerate(["decode step 1", "decode step 2", "decode step 3"]):
        y = 0.42 - idx * 0.12
        _box(ax, (0.24, y), (0.20, 0.08), label, fc="#F7F9FB", fontsize=11)
        _box(ax, (0.52, y), (0.20, 0.08), "复用 KV cache\n只输入新 token", fc="#F7F9FB", fontsize=11)
        _arrow(ax, (0.44, y + 0.04), (0.52, y + 0.04), color=PALETTE["purple"])
    _box(ax, (0.78, 0.26), (0.16, 0.12), "EOS 后\npadding 并停止", fc="#FFF0F0", fontsize=12, weight="bold")
    _arrow(ax, (0.72, 0.30), (0.78, 0.31), color=PALETTE["red"])
    save_figure(fig, output_path, dpi)


def draw_dataset_counts(counts: dict[str, int], output_path: Path, dpi: int) -> None:
    labels = list(counts.keys())
    values = np.array(list(counts.values()), dtype=np.int64)
    colors = [PALETTE["blue"], PALETTE["cyan"], PALETTE["green"], PALETTE["orange"]]

    fig = new_figure(dpi)
    ax = fig.add_subplot(111)
    y = np.arange(len(labels))
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("样本数量（log scale）")
    ax.grid(axis="x", color=PALETTE["line"], linewidth=0.8)
    max_value = values.max()
    for index, value in enumerate(values):
        ax.text(value * 1.08, index, _format_count(int(value)), va="center", ha="left", fontsize=13)
    ax.set_xlim(1_000, max_value * 2.2)
    fig.subplots_adjust(left=0.18, right=0.92, top=0.93, bottom=0.16)
    save_figure(fig, output_path, dpi)


def draw_pretrain_sample_format(sample: dict[str, str], output_path: Path, dpi: int) -> None:
    fig = new_figure(dpi)
    ax = _axis(fig)

    def add_box(
        x: float,
        y: float,
        w: float,
        h: float,
        label: str,
        body: str,
        color: str,
        *,
        body_size: float = 12.5,
    ) -> None:
        box = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            facecolor=color,
            edgecolor="#CFD8DC",
            linewidth=1.2,
        )
        ax.add_patch(box)
        ax.text(x + 0.025, y + h - 0.055, label, fontsize=16, fontweight="bold", va="top")
        ax.text(x + 0.025, y + h - 0.105, body, fontsize=body_size, va="top", family="monospace")

    user_text = sample["user"].replace("<image>", "\n<image>")
    user_body = _limited_wrapped(user_text, 34, 4)
    assistant_body = _limited_wrapped(sample["assistant"], 48, 10)

    add_box(0.06, 0.60, 0.40, 0.27, 'role: "user"', user_body, "#EEF5FF")
    add_box(0.54, 0.15, 0.42, 0.60, 'role: "assistant"', assistant_body, "#F2F7EF", body_size=10.2)

    image_box = FancyBboxPatch(
        (0.08, 0.18),
        0.28,
        0.24,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        facecolor="#FFFFFF",
        edgecolor="#B0BEC5",
        linewidth=1.2,
        linestyle="--",
    )
    ax.add_patch(image_box)
    ax.add_patch(Rectangle((0.115, 0.235), 0.085, 0.085, facecolor=PALETTE["cyan"], edgecolor="none", alpha=0.75))
    ax.add_patch(Rectangle((0.215, 0.235), 0.085, 0.085, facecolor=PALETTE["yellow"], edgecolor="none", alpha=0.85))
    ax.text(0.22, 0.19, "image_bytes\n不在图中展开", ha="center", va="bottom", fontsize=12)
    _arrow(ax, (0.43, 0.66), (0.55, 0.55), color=PALETTE["blue"])
    _arrow(ax, (0.33, 0.30), (0.55, 0.38), color=PALETTE["green"])
    ax.text(0.49, 0.84, "conversations 字段", fontsize=15, fontweight="bold", ha="center")
    ax.text(0.45, 0.45, "图像占位符\n对齐图像二进制", fontsize=12, ha="center")
    save_figure(fig, output_path, dpi)


def draw_assistant_only_labels(output_path: Path, dpi: int) -> None:
    fig = new_figure(dpi)
    ax = _axis(fig)
    labels = ["system", "user", "<image>", "assistant", "answer", "<eos>"]
    colors = ["#ECEFF1", "#ECEFF1", "#CDE7F0", "#DCECCB", "#DCECCB", "#DCECCB"]
    x0 = 0.12
    widths = [0.11, 0.10, 0.12, 0.15, 0.18, 0.10]
    x = x0
    for label, color, width in zip(labels, colors, widths):
        ax.add_patch(Rectangle((x, 0.58), width, 0.12, facecolor=color, edgecolor="white", linewidth=1.5))
        ax.text(x + width / 2, 0.64, label, ha="center", va="center", fontsize=12, fontweight="bold")
        ytext = "-100\nignore" if color != "#DCECCB" else "target id\npredict"
        ax.text(x + width / 2, 0.46, ytext, ha="center", va="center", fontsize=11)
        _arrow(ax, (x + width / 2, 0.58), (x + width / 2, 0.50), color=PALETTE["gray"])
        x += width + 0.008
    _box(
        ax,
        (0.16, 0.18),
        (0.68, 0.14),
        "loss 只计算 assistant 输出部分\n输入提示和 padding 均被 ignore_index=-100 忽略",
        fc="#FFF7E6",
        fontsize=13,
        weight="bold",
    )
    save_figure(fig, output_path, dpi)


def draw_source_distribution(counts: dict[str, int], output_path: Path, dpi: int) -> None:
    order = ["ai2d", "chartqa", "vsr", "scienceqa", "sft_replay"]
    labels = [label for label in order if label in counts]
    labels.extend(label for label in counts if label not in labels)
    values = [counts[label] for label in labels]
    total = sum(values)
    colors = [PALETTE["blue"], PALETTE["cyan"], PALETTE["green"], PALETTE["yellow"], PALETTE["purple"]]

    fig = new_figure(dpi)
    ax = fig.add_subplot(111)
    wedges, _ = ax.pie(
        values,
        startangle=100,
        colors=colors[: len(values)],
        wedgeprops={"width": 0.46, "edgecolor": "white", "linewidth": 2},
    )
    for wedge, label, value in zip(wedges, labels, values):
        angle = math.radians((wedge.theta1 + wedge.theta2) / 2)
        x, y = math.cos(angle), math.sin(angle)
        ax.annotate(
            f"{_display_label(label)}\n{_format_count(value)} ({value / total:.1%})",
            xy=(0.78 * x, 0.78 * y),
            xytext=(1.22 * x, 1.22 * y),
            ha="left" if x >= 0 else "right",
            va="center",
            arrowprops=dict(arrowstyle="-", color="#90A4AE", lw=1.0),
            fontsize=12.5,
        )
    ax.text(0, 0, f"{_format_count(total)}\n样本", ha="center", va="center", fontsize=15, fontweight="bold")
    ax.set_aspect("equal")
    fig.subplots_adjust(left=0.06, right=0.94, top=0.95, bottom=0.05)
    save_figure(fig, output_path, dpi)


def draw_task_distribution(counts: dict[str, int], output_path: Path, dpi: int) -> None:
    order = ["multiple_choice", "yes_no", "chart_short", "replay"]
    labels = [label for label in order if label in counts]
    labels.extend(label for label in counts if label not in labels)
    values = [counts[label] for label in labels]
    colors = [PALETTE["blue"], PALETTE["green"], PALETTE["orange"], PALETTE["gray"]]

    fig = new_figure(dpi)
    ax = fig.add_subplot(111)
    x = np.arange(len(labels))
    bars = ax.bar(x, values, color=colors[: len(values)], width=0.62)
    ax.set_xticks(x, [_display_label(label) for label in labels])
    ax.set_ylabel("样本数量")
    ax.grid(axis="y", color=PALETTE["line"], linewidth=0.8)
    ax.set_ylim(0, max(values) * 1.18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.025, _format_count(value), ha="center", va="bottom", fontsize=13)
    fig.subplots_adjust(left=0.10, right=0.97, top=0.92, bottom=0.16)
    save_figure(fig, output_path, dpi)


def draw_batch_collate(output_path: Path, dpi: int) -> None:
    fig = new_figure(dpi)
    ax = _axis(fig)
    samples = [("sample A", 7, 2), ("sample B", 11, 1)]
    for idx, (name, length, image_count) in enumerate(samples):
        y = 0.68 - idx * 0.24
        _box(ax, (0.05, y), (0.12, 0.10), f"{name}\n{image_count} images", fc="#EEF5FF", fontsize=12, weight="bold")
        for j in range(12):
            color = "#DCECCB" if j < length else "#ECEFF1"
            label = "tok" if j < length else "pad"
            ax.add_patch(Rectangle((0.22 + j * 0.035, y + 0.02), 0.032, 0.06, facecolor=color, edgecolor="white"))
            ax.text(0.236 + j * 0.035, y + 0.05, label, ha="center", va="center", fontsize=7)
    _box(ax, (0.70, 0.57), (0.22, 0.13), "input_ids pad 到同长\nattention_mask 标记有效位", fc="#FFF7E6", fontsize=12, weight="bold")
    _box(ax, (0.70, 0.35), (0.22, 0.13), "target_ids 用 -100 pad\npadding 不参与 loss", fc="#FFF7E6", fontsize=12, weight="bold")
    _box(ax, (0.32, 0.16), (0.36, 0.10), "images.extend(sample['images'])\n图像块展平成一个列表", fc="#F2F7EF", fontsize=13, weight="bold")
    save_figure(fig, output_path, dpi)


def draw_corrective_sft_composition(counts_source: dict[str, int], counts_task: dict[str, int], output_path: Path, dpi: int) -> None:
    fig = new_figure(dpi)
    axes = fig.subplots(1, 2)
    source_order = ["ai2d", "chartqa", "vsr", "scienceqa", "sft_replay"]
    task_order = ["multiple_choice", "yes_no", "chart_short", "replay"]
    for ax, order, counts, label in [
        (axes[0], source_order, counts_source, "来源组成"),
        (axes[1], task_order, counts_task, "任务类型组成"),
    ]:
        values = [counts[k] for k in order if k in counts]
        names = [_display_label(k) for k in order if k in counts]
        colors = [PALETTE["blue"], PALETTE["cyan"], PALETTE["green"], PALETTE["yellow"], PALETTE["purple"]]
        ax.barh(np.arange(len(values)), values, color=colors[: len(values)])
        ax.set_yticks(np.arange(len(values)), names)
        ax.invert_yaxis()
        ax.grid(axis="x", color=PALETTE["line"])
        ax.text(0.02, 0.96, label, transform=ax.transAxes, ha="left", va="top", fontsize=16, fontweight="bold")
        for i, v in enumerate(values):
            ax.text(v + max(values) * 0.02, i, _format_count(v), va="center", fontsize=11)
    fig.subplots_adjust(left=0.11, right=0.96, top=0.90, bottom=0.12, wspace=0.32)
    save_figure(fig, output_path, dpi)


def draw_mme_category_structure(output_path: Path, dpi: int) -> None:
    fig = new_figure(dpi)
    ax = _axis(fig)
    _box(ax, (0.40, 0.78), (0.20, 0.10), "MME\n2374 yes/no 样本", fc="#FFF7E6", fontsize=14, weight="bold")
    _box(ax, (0.13, 0.55), (0.27, 0.12), "Perception\n感知类", fc="#EEF5FF", fontsize=14, weight="bold")
    _box(ax, (0.60, 0.55), (0.27, 0.12), "Cognition\n认知类", fc="#F2F7EF", fontsize=14, weight="bold")
    _arrow(ax, (0.46, 0.78), (0.28, 0.67))
    _arrow(ax, (0.54, 0.78), (0.72, 0.67))
    perception = "existence / count / position / color\nposters / celebrity / scene\nlandmark / artwork / OCR"
    cognition = "commonsense reasoning\nnumerical calculation\ntext translation / code reasoning"
    _box(ax, (0.08, 0.25), (0.37, 0.20), perception, fc="#F7F9FB", fontsize=12)
    _box(ax, (0.55, 0.25), (0.37, 0.20), cognition, fc="#F7F9FB", fontsize=12)
    save_figure(fig, output_path, dpi)


def draw_mme_scoring(output_path: Path, dpi: int) -> None:
    fig = new_figure(dpi)
    ax = _axis(fig)
    _box(ax, (0.07, 0.64), (0.18, 0.14), "单题正确率\naccuracy", fc="#EEF5FF", fontsize=13, weight="bold")
    _box(ax, (0.32, 0.64), (0.18, 0.14), "成对全对率\naccuracy+", fc="#EEF5FF", fontsize=13, weight="bold")
    _box(ax, (0.57, 0.64), (0.18, 0.14), "类别分数\nacc + acc+", fc="#F2F7EF", fontsize=13, weight="bold")
    _box(ax, (0.78, 0.64), (0.15, 0.14), "总分\nsum", fc="#FFF7E6", fontsize=13, weight="bold")
    _arrow(ax, (0.25, 0.71), (0.32, 0.71))
    _arrow(ax, (0.50, 0.71), (0.57, 0.71))
    _arrow(ax, (0.75, 0.71), (0.78, 0.71))
    _box(ax, (0.17, 0.32), (0.28, 0.14), "Perception score\n10 个感知类别求和", fc="#F7F9FB", fontsize=13)
    _box(ax, (0.55, 0.32), (0.28, 0.14), "Cognition score\n4 个认知类别求和", fc="#F7F9FB", fontsize=13)
    _arrow(ax, (0.45, 0.39), (0.55, 0.39), color=PALETTE["green"])
    ax.text(0.50, 0.18, "Total MME Score = Perception + Cognition", ha="center", fontsize=15, fontweight="bold")
    save_figure(fig, output_path, dpi)


def draw_prediction_table(output_path: Path, dpi: int) -> None:
    rows = _read_prediction_rows()
    columns = ["index", "category", "answer", "prediction", "parsed_answer", "correct"]
    cell_text = []
    for row in rows:
        cell_text.append([_limited_wrapped(str(row.get(col, "")), 18, 2) for col in columns])
    fig = new_figure(dpi)
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    table = ax.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.1)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#CFD8DC")
        if r == 0:
            cell.set_facecolor("#EEF5FF")
            cell.set_text_props(weight="bold")
        elif c == columns.index("correct"):
            cell.set_facecolor("#F2F7EF" if str(cell.get_text().get_text()).lower() == "true" else "#FFF0F0")
    ax.text(0.5, 0.10, "prediction CSV 保留原始输出、解析答案和正确性，便于逐样本排查", ha="center", fontsize=13)
    save_figure(fig, output_path, dpi)


def draw_mme_stage_ablation(output_path: Path, dpi: int) -> None:
    stages = [row[0] for row in MME_STAGE_ROWS]
    perception = [row[1] for row in MME_STAGE_ROWS]
    cognition = [row[2] for row in MME_STAGE_ROWS]
    total = [row[3] for row in MME_STAGE_ROWS]
    fig = new_figure(dpi)
    ax = fig.add_subplot(111)
    x = np.arange(len(stages))
    width = 0.24
    ax.bar(x - width, perception, width, label="Perception", color=PALETTE["blue"])
    ax.bar(x, cognition, width, label="Cognition", color=PALETTE["green"])
    ax.bar(x + width, total, width, label="MME Score", color=PALETTE["orange"])
    ax.set_xticks(x, stages)
    ax.set_ylabel("Score")
    ax.grid(axis="y", color=PALETTE["line"])
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.18)
    save_figure(fig, output_path, dpi)


def draw_mme_baseline(output_path: Path, dpi: int) -> None:
    names = [row[0] for row in MME_BASELINE_ROWS]
    scores = [row[2] for row in MME_BASELINE_ROWS]
    colors = [PALETTE["gray"]] * (len(names) - 1) + [PALETTE["orange"]]
    fig = new_figure(dpi)
    ax = fig.add_subplot(111)
    y = np.arange(len(names))
    ax.barh(y, scores, color=colors, height=0.62)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("MME Score")
    ax.grid(axis="x", color=PALETTE["line"])
    for i, score in enumerate(scores):
        ax.text(score + 12, i, f"{score:.2f}", va="center", fontsize=11)
    ax.set_xlim(650, 1120)
    fig.subplots_adjust(left=0.20, right=0.96, top=0.92, bottom=0.14)
    save_figure(fig, output_path, dpi)


def draw_pope_stage_metrics(output_path: Path, dpi: int) -> None:
    stages = [row[0] for row in POPE_STAGE_ROWS]
    accuracy = [row[1] for row in POPE_STAGE_ROWS]
    f1 = [row[2] for row in POPE_STAGE_ROWS]
    yes_ratio = [row[3] for row in POPE_STAGE_ROWS]
    fig = new_figure(dpi)
    ax = fig.add_subplot(111)
    x = np.arange(len(stages))
    ax.plot(x, accuracy, marker="o", lw=2.4, color=PALETTE["blue"], label="Accuracy")
    ax.plot(x, f1, marker="o", lw=2.4, color=PALETTE["green"], label="F1")
    ax.plot(x, yes_ratio, marker="o", lw=2.4, color=PALETTE["red"], label="Yes Ratio")
    ax.set_xticks(x, stages)
    ax.set_ylabel("Percent")
    ax.set_ylim(45, 103)
    ax.grid(axis="y", color=PALETTE["line"])
    ax.legend(frameon=False, ncol=3, loc="lower left")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.18)
    save_figure(fig, output_path, dpi)


def draw_pope_baseline(output_path: Path, dpi: int) -> None:
    names = [row[0] for row in POPE_BASELINE_ROWS]
    accuracy = [row[1] for row in POPE_BASELINE_ROWS]
    f1 = [row[2] for row in POPE_BASELINE_ROWS]
    yes_ratio = [row[3] for row in POPE_BASELINE_ROWS]
    fig = new_figure(dpi)
    ax = fig.add_subplot(111)
    x = np.arange(len(names))
    width = 0.23
    ax.bar(x - width, accuracy, width, color=PALETTE["blue"], label="Accuracy")
    ax.bar(x, f1, width, color=PALETTE["green"], label="F1")
    ax.bar(x + width, yes_ratio, width, color=PALETTE["red"], label="Yes Ratio")
    ax.set_xticks(x, names)
    ax.set_ylabel("Percent")
    ax.set_ylim(45, 105)
    ax.grid(axis="y", color=PALETTE["line"])
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.18)
    save_figure(fig, output_path, dpi)


def draw_training_stage_objectives(output_path: Path, dpi: int) -> None:
    fig = new_figure(dpi)
    ax = _axis(fig)
    stages = [
        ("Pretraining", "跨模态对齐\n让视觉特征进入语言空间", "风险：yes bias 仍明显"),
        ("SFT", "指令跟随\n更像对话助手", "风险：长解释影响 parser"),
        ("GRPO", "规则奖励\n短答案正确性", "风险：开放式任务难设计 reward"),
    ]
    xs = [0.08, 0.38, 0.68]
    for x, (stage, goal, risk) in zip(xs, stages):
        _box(ax, (x, 0.56), (0.24, 0.16), f"{stage}\n{goal}", fc="#EEF5FF", fontsize=12.5, weight="bold")
        _box(ax, (x, 0.30), (0.24, 0.12), risk, fc="#FFF7E6", fontsize=11.5)
    _arrow(ax, (0.32, 0.64), (0.38, 0.64))
    _arrow(ax, (0.62, 0.64), (0.68, 0.64))
    ax.text(0.50, 0.15, "不同阶段优化目标不同，因此指标不一定单调上升", ha="center", fontsize=15, fontweight="bold")
    save_figure(fig, output_path, dpi)


def draw_image_token_alignment_check(output_path: Path, dpi: int) -> None:
    fig = new_figure(dpi)
    ax = _axis(fig)
    _box(ax, (0.08, 0.60), (0.22, 0.14), "图像块数量 N\n全局图 + 局部图", fc="#EEF5FF", fontsize=13, weight="bold")
    _box(ax, (0.39, 0.60), (0.22, 0.14), "Projector 输出\nN x 64", fc="#F2F7EF", fontsize=13, weight="bold")
    _box(ax, (0.70, 0.60), (0.22, 0.14), "文本占位数量\ncount(<|image|>)", fc="#FFF7E6", fontsize=13, weight="bold")
    _arrow(ax, (0.30, 0.67), (0.39, 0.67))
    _arrow(ax, (0.61, 0.67), (0.70, 0.67))
    _box(ax, (0.26, 0.30), (0.48, 0.13), "必须满足：count(<|image|>) = N x mp_image_token_length", fc="#DCECCB", fontsize=15, weight="bold")
    save_figure(fig, output_path, dpi)


def draw_global_local_tradeoff(output_path: Path, dpi: int) -> None:
    fig = new_figure(dpi)
    ax = _axis(fig)
    options = [
        ("只用全局图", "整体布局好\n局部细节弱\n64 tokens", "#EEF5FF"),
        ("全局 + 局部网格", "兼顾整体和细节\n成本中等\n(1+K)x64 tokens", "#F2F7EF"),
        ("全部高分辨率细块", "细节更强\n序列更长\n显存成本高", "#FFF7E6"),
    ]
    for x, item in zip([0.08, 0.38, 0.68], options):
        title, body, color = item
        _box(ax, (x, 0.45), (0.23, 0.22), f"{title}\n{body}", fc=color, fontsize=12.5, weight="bold")
    ax.plot([0.16, 0.50, 0.80], [0.28, 0.36, 0.50], color=PALETTE["red"], lw=2.5, marker="o")
    ax.text(0.50, 0.18, "局部细节提升通常伴随 token 数和推理成本上升", ha="center", fontsize=15, fontweight="bold")
    save_figure(fig, output_path, dpi)


def draw_ability_requirements(output_path: Path, dpi: int) -> None:
    abilities = ["对象识别", "场景判断", "OCR", "计数", "空间推理"]
    requirements = np.array(
        [
            [2, 2, 3, 2],
            [2, 2, 2, 2],
            [4, 3, 4, 3],
            [4, 4, 3, 3],
            [4, 4, 3, 4],
        ]
    )
    fig = new_figure(dpi)
    ax = fig.add_subplot(111)
    im = ax.imshow(requirements, cmap=mpl.colors.LinearSegmentedColormap.from_list("req", ["#EEF5FF", "#DDA15E"]), vmin=1, vmax=4)
    ax.set_xticks(np.arange(4), ["分辨率", "定位", "数据", "推理"])
    ax.set_yticks(np.arange(len(abilities)), abilities)
    for i in range(requirements.shape[0]):
        for j in range(requirements.shape[1]):
            ax.text(j, i, str(requirements[i, j]), ha="center", va="center", fontsize=14, fontweight="bold")
    ax.text(3.65, 4.75, "1=低，4=高", ha="right", fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    fig.subplots_adjust(left=0.22, right=0.90, top=0.90, bottom=0.16)
    save_figure(fig, output_path, dpi)


def draw_grpo_benefits_limitations(output_path: Path, dpi: int) -> None:
    fig = new_figure(dpi)
    ax = _axis(fig)
    _box(ax, (0.08, 0.18), (0.38, 0.60), "提升\n\n+ 奖励正确 yes/no 或选项\n+ 短格式 bonus 提升可解析率\n+ 降低盲目 yes 倾向\n+ 适合答案空间有限任务", fc="#F2F7EF", fontsize=14, align="left", weight="bold")
    _box(ax, (0.54, 0.18), (0.38, 0.60), "局限\n\n- reward/parser 必须可靠\n- zero-std group 会跳过\n- 开放式描述难设计规则奖励\n- 不能替代更强数据和模型", fc="#FFF0F0", fontsize=14, align="left", weight="bold")
    save_figure(fig, output_path, dpi)


def draw_future_roadmap(output_path: Path, dpi: int) -> None:
    fig = new_figure(dpi)
    ax = _axis(fig)
    steps = [
        ("数据质量", "区分短答案\n与解释型任务"),
        ("图像处理", "动态分辨率\n高信息区域切块"),
        ("模型结构", "更强视觉 backbone\n或辅助模块"),
        ("训练策略", "逐步解冻\nLoRA 微调"),
        ("评测扩展", "OCR / 图表\n空间推理"),
    ]
    xs = np.linspace(0.06, 0.78, len(steps))
    for idx, (title, body) in enumerate(steps):
        _box(ax, (xs[idx], 0.50), (0.16, 0.16), f"{title}\n{body}", fc="#EEF5FF" if idx % 2 == 0 else "#F2F7EF", fontsize=11.5, weight="bold")
        ax.text(xs[idx] + 0.08, 0.40, f"{idx + 1}", ha="center", va="center", fontsize=18, fontweight="bold", color=PALETTE["blue"])
        if idx < len(steps) - 1:
            _arrow(ax, (xs[idx] + 0.16, 0.58), (xs[idx + 1], 0.58))
    ax.text(0.50, 0.20, "从数据、图像处理、模型、训练和评测五个方向继续迭代", ha="center", fontsize=15, fontweight="bold")
    save_figure(fig, output_path, dpi)


def update_report_references(report_path: Path = DEFAULT_REPORT, figure_dir: str | Path = DEFAULT_OUTPUT_DIR) -> int:
    report_path = Path(report_path)
    figure_dir = Path(figure_dir).as_posix()
    lines = report_path.read_text(encoding="utf-8").splitlines()
    changes = 0

    for marker, alt, figure_id in REPORT_REFERENCES:
        link = f"![{alt}]({figure_dir}/{FIGURE_FILES[figure_id]})"
        for index, line in enumerate(lines):
            if marker not in line:
                continue
            next_index = index + 1
            if next_index < len(lines) and lines[next_index].startswith("!["):
                if lines[next_index] != link:
                    lines[next_index] = link
                    changes += 1
            else:
                lines.insert(next_index, link)
                changes += 1
            break

    if changes:
        report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return changes


def draw_report_figures(
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    image_path: Path = DEFAULT_IMAGE,
    dpi: int = DEFAULT_DPI,
    only: list[str] | None = None,
    pretrain: Path = PRETRAIN_PATH,
    sft: Path = SFT_PATH,
    corrective: Path = CORRECTIVE_SFT_PATH,
    grpo: Path = GRPO_PATH,
    prediction_csv: Path = PREDICTION_CSV_PATH,
) -> list[Path]:
    configure_matplotlib()
    selected = only or list(FIGURE_FILES)
    output_dir.mkdir(parents=True, exist_ok=True)

    actions: dict[str, Callable[[], None]] = {
        "2-1": lambda: draw_overall_flow(output_dir / FIGURE_FILES["2-1"], dpi),
        "2-2": lambda: draw_text_visual_alignment(output_dir / FIGURE_FILES["2-2"], dpi),
        "3-2": lambda: draw_dynamic_resize_comparison(image_path, output_dir / FIGURE_FILES["3-2"], dpi),
        "3-3": lambda: draw_normalization_histogram(image_path, output_dir / FIGURE_FILES["3-3"], dpi),
        "3-5": lambda: draw_visual_token_string(output_dir / FIGURE_FILES["3-5"], dpi),
        "3-6": lambda: draw_image_token_mismatch_flow(output_dir / FIGURE_FILES["3-6"], dpi),
        "4-8": lambda: draw_image_embedding_replacement(output_dir / FIGURE_FILES["4-8"], dpi),
        "4-9": lambda: draw_generate_prefill_decode(output_dir / FIGURE_FILES["4-9"], dpi),
        "5-1": lambda: draw_dataset_counts(build_dataset_counts(pretrain=pretrain, sft=sft, corrective=corrective, grpo=grpo), output_dir / FIGURE_FILES["5-1"], dpi),
        "5-2": lambda: draw_pretrain_sample_format(read_pretrain_sample(pretrain), output_dir / FIGURE_FILES["5-2"], dpi),
        "5-3": lambda: draw_assistant_only_labels(output_dir / FIGURE_FILES["5-3"], dpi),
        "5-4": lambda: draw_source_distribution(count_column_values(corrective, "source"), output_dir / FIGURE_FILES["5-4"], dpi),
        "5-5": lambda: draw_task_distribution(count_column_values(corrective, "task_type"), output_dir / FIGURE_FILES["5-5"], dpi),
        "5-7": lambda: draw_batch_collate(output_dir / FIGURE_FILES["5-7"], dpi),
        "6-8": lambda: draw_corrective_sft_composition(count_column_values(corrective, "source"), count_column_values(corrective, "task_type"), output_dir / FIGURE_FILES["6-8"], dpi),
        "8-2": lambda: draw_mme_category_structure(output_dir / FIGURE_FILES["8-2"], dpi),
        "8-3": lambda: draw_mme_scoring(output_dir / FIGURE_FILES["8-3"], dpi),
        "8-6": lambda: draw_prediction_table(output_dir / FIGURE_FILES["8-6"], dpi),
        "9-1": lambda: draw_mme_stage_ablation(output_dir / FIGURE_FILES["9-1"], dpi),
        "9-2": lambda: draw_mme_baseline(output_dir / FIGURE_FILES["9-2"], dpi),
        "9-3": lambda: draw_pope_stage_metrics(output_dir / FIGURE_FILES["9-3"], dpi),
        "9-4": lambda: draw_pope_baseline(output_dir / FIGURE_FILES["9-4"], dpi),
        "9-5": lambda: draw_training_stage_objectives(output_dir / FIGURE_FILES["9-5"], dpi),
        "10-1": lambda: draw_image_token_alignment_check(output_dir / FIGURE_FILES["10-1"], dpi),
        "10-2": lambda: draw_global_local_tradeoff(output_dir / FIGURE_FILES["10-2"], dpi),
        "10-3": lambda: draw_ability_requirements(output_dir / FIGURE_FILES["10-3"], dpi),
        "10-5": lambda: draw_grpo_benefits_limitations(output_dir / FIGURE_FILES["10-5"], dpi),
        "10-6": lambda: draw_future_roadmap(output_dir / FIGURE_FILES["10-6"], dpi),
    }
    # Capture a non-default prediction CSV path without changing the public helper signature.
    if prediction_csv != PREDICTION_CSV_PATH:
        actions["8-6"] = lambda: draw_prediction_table(output_dir / FIGURE_FILES["8-6"], dpi)

    generated: list[Path] = []
    for key in selected:
        actions[key]()
        generated.append(output_dir / FIGURE_FILES[key])
    return generated


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw mini-VLM report figures.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--image", type=Path, default=DEFAULT_IMAGE)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--only", nargs="+", choices=list(FIGURE_FILES), default=None)
    parser.add_argument("--pretrain", type=Path, default=PRETRAIN_PATH)
    parser.add_argument("--sft", type=Path, default=SFT_PATH)
    parser.add_argument("--corrective", type=Path, default=CORRECTIVE_SFT_PATH)
    parser.add_argument("--grpo", type=Path, default=GRPO_PATH)
    parser.add_argument("--prediction-csv", type=Path, default=PREDICTION_CSV_PATH)
    parser.add_argument("--update-report", action="store_true", help="Update report placeholders with image links.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    generated = draw_report_figures(
        output_dir=args.output_dir,
        image_path=args.image,
        dpi=args.dpi,
        only=args.only,
        pretrain=args.pretrain,
        sft=args.sft,
        corrective=args.corrective,
        grpo=args.grpo,
        prediction_csv=args.prediction_csv,
    )
    result: dict[str, object] = {"generated": [str(path) for path in generated]}
    if args.update_report:
        result["report_updates"] = update_report_references(args.report, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
