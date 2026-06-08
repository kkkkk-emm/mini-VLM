import pytest
import torch

from training.grpo_rewards import RuleRewardResult
from training.grpo_trainer import SampledGroupStats


def _result(parsed_answer, *, correct=False, parseable=True):
    return RuleRewardResult(
        reward=1.0 if correct else -1.0,
        parsed_answer=parsed_answer,
        normalized_answer=parsed_answer,
        is_parseable=parseable,
        is_correct=correct,
        format_bonus=0.0,
        verbosity_penalty=0.0,
    )


def test_sampled_group_stats_classifies_zero_std_groups():
    stats = SampledGroupStats()

    stats.update([_result("yes", correct=True)], reward_std=torch.tensor(0.0), zero_std=True)
    stats.update([_result("no", correct=False)], reward_std=torch.tensor(0.0), zero_std=True)
    stats.update([_result("", parseable=False)], reward_std=torch.tensor(0.0), zero_std=True)
    stats.update(
        [_result("yes", correct=True), _result("", parseable=False)],
        reward_std=torch.tensor(0.0),
        zero_std=True,
    )
    stats.update([_result("yes", correct=True), _result("no", correct=False)], reward_std=torch.tensor(0.5), zero_std=False)

    metrics = stats.metrics(prefix="train")

    assert metrics["train/sampled_groups"] == 5
    assert metrics["train/zero_std_groups"] == 4
    assert metrics["train/zero_std_rate"] == pytest.approx(0.8)
    assert metrics["train/zero_std_all_correct"] == 1
    assert metrics["train/zero_std_all_wrong"] == 1
    assert metrics["train/zero_std_all_unparseable"] == 1
    assert metrics["train/zero_std_other"] == 1


def test_sampled_group_stats_reports_completion_level_rates():
    stats = SampledGroupStats()

    stats.update(
        [
            _result("yes", correct=True),
            _result("no", correct=False),
            _result("", parseable=False),
            _result("A", correct=True),
        ],
        reward_std=torch.tensor(1.0),
        zero_std=False,
    )

    metrics = stats.metrics(prefix="train")

    assert metrics["train/correct_rate"] == pytest.approx(0.5)
    assert metrics["train/parseable_rate"] == pytest.approx(0.75)
    assert metrics["train/yes_ratio"] == pytest.approx(0.25)
    assert metrics["train/no_ratio"] == pytest.approx(0.25)
