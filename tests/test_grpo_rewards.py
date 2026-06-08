import pytest

from training.grpo_rewards import (
    RuleRewardConfig,
    normalize_choice_answer,
    normalize_yes_no_answer,
    score_rule_reward,
)


def test_normalize_choice_answer_prefers_final_explicit_option():
    text = "The image contains a suitcase, but the final answer is option B."

    assert normalize_choice_answer(text) == "B"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Yes", "yes"),
        ("no.", "no"),
        ("The answer is yes because it is visible.", "yes"),
        ("not enough evidence", "no"),
    ],
)
def test_normalize_yes_no_answer(text, expected):
    assert normalize_yes_no_answer(text) == expected


def test_score_rule_reward_rewards_short_correct_choice():
    result = score_rule_reward(
        completion="A",
        answer="A",
        task_type="multiple_choice",
        config=RuleRewardConfig(),
    )

    assert result.parsed_answer == "A"
    assert result.is_correct is True
    assert result.reward == pytest.approx(1.2)
    assert result.format_bonus == pytest.approx(0.2)
    assert result.verbosity_penalty == pytest.approx(0.0)


def test_score_rule_reward_penalizes_final_wrong_choice_even_with_description():
    result = score_rule_reward(
        completion="The relationship looks plausible, but the final answer is B.",
        answer="A",
        task_type="multiple_choice",
        config=RuleRewardConfig(),
    )

    assert result.parsed_answer == "B"
    assert result.is_correct is False
    assert result.reward < -1.0
    assert result.verbosity_penalty > 0.0


def test_score_rule_reward_penalizes_unparseable_answer():
    result = score_rule_reward(
        completion="I cannot tell from the image.",
        answer="yes",
        task_type="yes_no",
        config=RuleRewardConfig(),
    )

    assert result.parsed_answer == ""
    assert result.is_parseable is False
    assert result.reward == pytest.approx(-0.5)
