from __future__ import annotations

import re
from dataclasses import dataclass


CHOICE_TASKS = {"multiple_choice", "choice", "mmstar"}
YES_NO_TASKS = {"yes_no", "binary", "mme", "pope"}

_CHOICE_EXPLICIT_RE = re.compile(
    r"(?:final\s+)?(?:answer|option|choice)\s*(?:is|:)?\s*\(?([ABCD])\)?",
    flags=re.IGNORECASE,
)
_CHOICE_SHORT_RE = re.compile(
    r"^\s*(?:answer\s*:\s*)?\(?([ABCD])\)?(?:[.)\]\s]|$)",
    flags=re.IGNORECASE,
)
_YES_NO_EXPLICIT_RE = re.compile(
    r"(?:final\s+)?answer\s*(?:is|:)?\s*(yes|no)\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class RuleRewardConfig:
    correct_reward: float = 1.0
    incorrect_reward: float = -1.0
    unparseable_reward: float = -0.5
    short_format_bonus: float = 0.2
    verbosity_penalty_max: float = 0.3
    short_token_limit: int = 3
    verbosity_token_threshold: int = 8


@dataclass(frozen=True)
class RuleRewardResult:
    reward: float
    parsed_answer: str
    normalized_answer: str
    is_parseable: bool
    is_correct: bool
    format_bonus: float
    verbosity_penalty: float


def normalize_choice_answer(text: str) -> str:
    cleaned = str(text).strip()
    if not cleaned:
        return ""

    explicit_matches = _CHOICE_EXPLICIT_RE.findall(cleaned)
    if explicit_matches:
        return explicit_matches[-1].upper()

    short_match = _CHOICE_SHORT_RE.match(cleaned)
    if short_match:
        return short_match.group(1).upper()

    parenthesized = re.findall(r"\(([ABCD])\)", cleaned, flags=re.IGNORECASE)
    if parenthesized:
        return parenthesized[-1].upper()

    return ""


def normalize_yes_no_answer(text: str) -> str:
    cleaned = str(text).strip().lower()
    if not cleaned:
        return ""

    explicit_matches = _YES_NO_EXPLICIT_RE.findall(cleaned)
    if explicit_matches:
        return explicit_matches[-1].lower()

    words = re.findall(r"[a-z']+", cleaned)
    if not words:
        return ""
    if words[0] in {"yes", "yeah", "yep"}:
        return "yes"
    if words[0] in {"no", "nope", "not"}:
        return "no"

    prefix = words[:4]
    if "not" in prefix or "no" in prefix:
        return "no"
    if "yes" in prefix:
        return "yes"
    return ""


def normalize_reference_answer(answer: str, task_type: str) -> str:
    if task_type in CHOICE_TASKS:
        normalized = normalize_choice_answer(answer)
        return normalized or str(answer).strip().upper()
    if task_type in YES_NO_TASKS:
        normalized = normalize_yes_no_answer(answer)
        return normalized or str(answer).strip().lower()
    raise ValueError(f"Unsupported GRPO reward task_type: {task_type}")


def normalize_completion_answer(completion: str, task_type: str) -> str:
    if task_type in CHOICE_TASKS:
        return normalize_choice_answer(completion)
    if task_type in YES_NO_TASKS:
        return normalize_yes_no_answer(completion)
    raise ValueError(f"Unsupported GRPO reward task_type: {task_type}")


def _word_count(text: str) -> int:
    return len(re.findall(r"\S+", str(text).strip()))


def _has_short_format(completion: str, parsed_answer: str, task_type: str, config: RuleRewardConfig) -> bool:
    if not parsed_answer:
        return False
    if _word_count(completion) > config.short_token_limit:
        return False

    compact = str(completion).strip().lower().rstrip(".。!！")
    parsed = parsed_answer.lower()
    if task_type in CHOICE_TASKS:
        return compact in {
            parsed,
            f"({parsed})",
            f"answer: {parsed}",
            f"option {parsed}",
            f"choice {parsed}",
        }
    return compact in {parsed, f"answer: {parsed}"}


def _verbosity_penalty(completion: str, config: RuleRewardConfig) -> float:
    words = _word_count(completion)
    if words <= config.verbosity_token_threshold:
        return 0.0
    excess = words - config.verbosity_token_threshold
    scale = max(config.verbosity_token_threshold, 1)
    return min(config.verbosity_penalty_max, config.verbosity_penalty_max * excess / scale)


def score_rule_reward(
    *,
    completion: str,
    answer: str,
    task_type: str,
    config: RuleRewardConfig | None = None,
) -> RuleRewardResult:
    config = config or RuleRewardConfig()
    parsed_answer = normalize_completion_answer(completion, task_type)
    normalized_answer = normalize_reference_answer(answer, task_type)
    is_parseable = bool(parsed_answer)
    is_correct = is_parseable and parsed_answer == normalized_answer

    if not is_parseable:
        return RuleRewardResult(
            reward=config.unparseable_reward,
            parsed_answer="",
            normalized_answer=normalized_answer,
            is_parseable=False,
            is_correct=False,
            format_bonus=0.0,
            verbosity_penalty=0.0,
        )

    reward = config.correct_reward if is_correct else config.incorrect_reward
    format_bonus = (
        config.short_format_bonus
        if _has_short_format(completion, parsed_answer, task_type, config)
        else 0.0
    )
    verbosity_penalty = _verbosity_penalty(completion, config)
    reward = reward + format_bonus - verbosity_penalty

    return RuleRewardResult(
        reward=reward,
        parsed_answer=parsed_answer,
        normalized_answer=normalized_answer,
        is_parseable=True,
        is_correct=is_correct,
        format_bonus=format_bonus,
        verbosity_penalty=verbosity_penalty,
    )
