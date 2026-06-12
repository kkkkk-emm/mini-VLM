import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from scripts.draw_report_figures import (
    FIGURE_FILES,
    build_dataset_counts,
    count_column_values,
    read_pretrain_sample,
    update_report_references,
)


NEW_FIGURE_IDS = [
    "2-1",
    "2-2",
    "3-5",
    "3-6",
    "4-8",
    "4-9",
    "5-3",
    "5-7",
    "6-8",
    "8-2",
    "8-3",
    "8-6",
    "9-1",
    "9-2",
    "9-3",
    "9-4",
    "9-5",
    "10-1",
    "10-2",
    "10-3",
    "10-5",
    "10-6",
]


def _write_parquet(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_dataset_counts_use_parquet_metadata(tmp_path):
    pretrain = tmp_path / "pretrain.parquet"
    sft = tmp_path / "sft.parquet"
    corrective = tmp_path / "corrective.parquet"
    grpo = tmp_path / "grpo.parquet"
    _write_parquet(pretrain, [{"conversations": "[]", "image_bytes": b"a"} for _ in range(3)])
    _write_parquet(sft, [{"conversations": "[]", "image_bytes": b"b"} for _ in range(5)])
    _write_parquet(
        corrective,
        [
            {
                "conversations": "[]",
                "image_bytes": b"c",
                "source": "ai2d",
                "task_type": "multiple_choice",
                "answer": "A",
            }
            for _ in range(2)
        ],
    )
    _write_parquet(grpo, [{"question": "Q", "answer": "A", "image_bytes": b"d"}])

    counts = build_dataset_counts(
        pretrain=pretrain,
        sft=sft,
        corrective=corrective,
        grpo=grpo,
    )

    assert counts == {
        "Pretrain": 3,
        "SFT": 5,
        "Corrective SFT": 2,
        "AI2D-GRPO": 1,
    }


def test_count_column_values_and_read_pretrain_sample(tmp_path):
    corrective = tmp_path / "corrective.parquet"
    pretrain = tmp_path / "pretrain.parquet"
    conversations = [
        {
            "role": "user",
            "content": "Could you describe this image?<image>",
        },
        {
            "role": "assistant",
            "content": "A small dog is running through a park with colorful balloons.",
        },
    ]
    _write_parquet(
        corrective,
        [
            {"source": "ai2d", "task_type": "multiple_choice"},
            {"source": "ai2d", "task_type": "multiple_choice"},
            {"source": "vsr", "task_type": "yes_no"},
        ],
    )
    _write_parquet(pretrain, [{"conversations": json.dumps(conversations), "image_bytes": b"x"}])

    assert count_column_values(corrective, "source") == {"ai2d": 2, "vsr": 1}
    assert count_column_values(corrective, "task_type") == {"multiple_choice": 2, "yes_no": 1}

    sample = read_pretrain_sample(pretrain)
    assert sample["user"].endswith("<image>")
    assert sample["assistant"].startswith("A small dog")


def test_cli_generates_selected_figures(tmp_path):
    image = tmp_path / "input.png"
    output_dir = tmp_path / "figures"
    Image.new("RGB", (640, 640), color=(80, 120, 160)).save(image)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/draw_report_figures.py",
            "--only",
            "3-2",
            "3-3",
            "--image",
            str(image),
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    for filename in [
        "fig-3-2-dynamic-resize-comparison.png",
        "fig-3-3-normalization-histogram.png",
    ]:
        path = output_dir / filename
        assert path.is_file()
        with Image.open(path) as generated:
            assert generated.mode == "RGBA"
            assert generated.size == (1774, 887)


def test_registry_includes_requested_report_figures():
    for figure_id in NEW_FIGURE_IDS:
        assert figure_id in FIGURE_FILES

    assert len(FIGURE_FILES) == 28
    assert FIGURE_FILES["2-2"] == "fig-2-2-text-visual-token-alignment-v2.png"
    assert FIGURE_FILES["3-5"] == "fig-3-5-visual-token-string.png"
    assert FIGURE_FILES["3-6"] == "fig-3-6-image-token-mismatch-flow.png"
    assert FIGURE_FILES["9-3"] == "fig-9-3-pope-stage-metrics.png"


def test_cli_generates_new_selected_figures(tmp_path):
    output_dir = tmp_path / "figures"

    result = subprocess.run(
        [
            sys.executable,
            "scripts/draw_report_figures.py",
            "--only",
            "2-1",
            "9-1",
            "10-6",
            "--output-dir",
            str(output_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    generated_names = {path.name for path in output_dir.iterdir()}
    assert generated_names == {
        "fig-2-1-mini-vlm-overall-flow.png",
        "fig-9-1-mme-stage-ablation.png",
        "fig-10-6-future-roadmap.png",
    }
    for path in output_dir.iterdir():
        with Image.open(path) as generated:
            assert generated.mode == "RGBA"
            assert generated.size == (1774, 887)


def test_update_report_references_inserts_and_is_idempotent(tmp_path):
    report = tmp_path / "report.md"
    report.write_text(
        "\n".join(
            [
                "【图片占位 2-1：mini-VLM 整体流程图】  ",
                "建议画成横向流程图。",
                "",
                "【图 2-2：文本 token 与视觉 token 对齐示意图】  ",
                "![文本 token 与视觉 token 对齐示意图](figures/old.png)",
                "",
                "【图片占位 9-3：POPE 阶段指标变化图】  ",
                "建议把 Accuracy、F1 和 Yes Ratio 放在同一页。",
            ]
        ),
        encoding="utf-8",
    )

    first_count = update_report_references(report)
    second_count = update_report_references(report)
    text = report.read_text(encoding="utf-8")

    assert first_count == 3
    assert second_count == 0
    assert "![mini-VLM 整体流程图](figures/fig-2-1-mini-vlm-overall-flow.png)" in text
    assert "![文本 token 与视觉 token 对齐示意图](figures/fig-2-2-text-visual-token-alignment-v2.png)" in text
    assert "![POPE 阶段指标变化图](figures/fig-9-3-pope-stage-metrics.png)" in text
    assert "figures/old.png" not in text
    assert text.count("fig-2-1-mini-vlm-overall-flow.png") == 1
