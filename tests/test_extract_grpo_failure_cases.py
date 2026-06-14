import json
import subprocess
import sys
from pathlib import Path

from PIL import Image

from scripts.extract_grpo_failure_cases import (
    find_mme_failure,
    find_pope_failure,
    write_failure_outputs,
)


def _sample(
    *,
    question: str,
    answer: str,
    category: str,
    image_color: tuple[int, int, int] = (120, 80, 40),
    **extra,
):
    return {
        "question": question,
        "answer": answer,
        "category": category,
        "question_id": extra.pop("question_id", "q-1"),
        "image": Image.new("RGB", (16, 12), color=image_color),
        **extra,
    }


def test_find_mme_failure_uses_only_count_or_position_candidates():
    dataset = [
        _sample(
            question="Is there a cat in the image?",
            answer="no",
            category="existence",
            question_id="existence/1.jpg",
        ),
        _sample(
            question="Are there two bottles in the image?",
            answer="no",
            category="count",
            question_id="count/2.jpg",
        ),
    ]

    def infer(_sample, _benchmark, _max_new_tokens):
        return "Yes, there are"

    failure = find_mme_failure(
        dataset,
        infer=infer,
        categories={"count", "position"},
        max_new_tokens=4,
    )

    assert failure is not None
    assert failure.benchmark == "mme"
    assert failure.category == "count"
    assert failure.question_id == "count/2.jpg"
    assert failure.answer == "no"
    assert failure.prediction == "Yes, there are"
    assert failure.parsed_answer == "yes"
    assert "空间细节" in failure.analysis


def test_find_pope_failure_uses_only_no_to_yes_false_positive():
    dataset = [
        _sample(
            question="Is there a car in the image?",
            answer="yes",
            category="adversarial",
            question_id="1",
            image_source="COCO_val2014_000000000001",
        ),
        _sample(
            question="Is there a dining table in the image?",
            answer="no",
            category="adversarial",
            question_id="2",
            image_source="COCO_val2014_000000000002",
        ),
    ]

    def infer(sample, _benchmark, _max_new_tokens):
        return "No." if sample["answer"] == "yes" else "Yes,"

    failure = find_pope_failure(dataset, infer=infer, max_new_tokens=2)

    assert failure is not None
    assert failure.benchmark == "pope"
    assert failure.question_id == "2"
    assert failure.answer == "no"
    assert failure.prediction == "Yes,"
    assert failure.parsed_answer == "yes"
    assert "语言先验" in failure.analysis
    assert "视觉证据不足" in failure.analysis


def test_write_failure_outputs_saves_images_json_and_markdown(tmp_path):
    cases = [
        find_mme_failure(
            [
                _sample(
                    question="Are there two bottles in the image?",
                    answer="no",
                    category="count",
                    question_id="count/2.jpg",
                )
            ],
            infer=lambda _sample, _benchmark, _max_new_tokens: "Yes, there are",
            categories={"count", "position"},
            max_new_tokens=4,
        ),
        find_pope_failure(
            [
                _sample(
                    question="Is there a dining table in the image?",
                    answer="no",
                    category="adversarial",
                    question_id="2",
                    image_source="COCO_val2014_000000000002",
                    image_color=(10, 20, 30),
                )
            ],
            infer=lambda _sample, _benchmark, _max_new_tokens: "Yes,",
            max_new_tokens=2,
        ),
    ]

    output = write_failure_outputs(cases, tmp_path)

    assert Path(output["json"]).is_file()
    assert Path(output["markdown"]).is_file()
    assert (tmp_path / "mme_failure.png").is_file()
    assert (tmp_path / "pope_failure.png").is_file()

    payload = json.loads((tmp_path / "failure_cases.json").read_text(encoding="utf-8"))
    assert [case["benchmark"] for case in payload["cases"]] == ["mme", "pope"]
    assert payload["cases"][0]["image_path"] == "mme_failure.png"
    assert payload["cases"][1]["image_path"] == "pope_failure.png"
    assert payload["cases"][0]["question"] == "Are there two bottles in the image?"
    assert payload["cases"][1]["answer"] == "no"
    assert payload["cases"][1]["prediction"] == "Yes,"

    markdown = (tmp_path / "failure_cases.md").read_text(encoding="utf-8")
    assert "![MME failure](mme_failure.png)" in markdown
    assert "正确答案：no" in markdown
    assert "模型错误回答：Yes," in markdown


def test_cli_help_runs_when_script_is_executed_directly():
    result = subprocess.run(
        [sys.executable, "scripts/extract_grpo_failure_cases.py", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--checkpoint" in result.stdout
    assert "--mme-categories" in result.stdout
