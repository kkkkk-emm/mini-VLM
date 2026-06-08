import torch

from training.grpo_utils import (
    build_completion_mask,
    compute_group_advantages,
    gather_completion_log_probs,
    sequence_log_probs,
)


def test_compute_group_advantages_standardizes_rewards():
    rewards = torch.tensor([1.0, 2.0, 3.0])

    advantages, std, should_skip = compute_group_advantages(rewards, eps=1e-6)

    assert should_skip is False
    assert std.item() == torch.std(rewards, unbiased=False).item()
    assert torch.allclose(advantages.mean(), torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(advantages.std(unbiased=False), torch.tensor(1.0), atol=1e-6)


def test_compute_group_advantages_marks_zero_std_group_for_skip():
    rewards = torch.tensor([0.5, 0.5, 0.5, 0.5])

    advantages, std, should_skip = compute_group_advantages(rewards, eps=1e-6)

    assert should_skip is True
    assert std.item() == 0.0
    assert torch.equal(advantages, torch.zeros_like(rewards))


def test_build_completion_mask_includes_eos_and_excludes_after_eos_and_pad():
    completion_ids = torch.tensor(
        [
            [7, 2, 0, 0],
            [8, 9, 2, 0],
            [4, 5, 6, 0],
        ]
    )

    mask = build_completion_mask(
        completion_ids,
        eos_token_id=2,
        pad_token_id=0,
    )

    expected = torch.tensor(
        [
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ]
    )
    assert torch.equal(mask, expected)


def test_gather_completion_log_probs_uses_logits_that_predict_completion_tokens():
    logits = torch.full((1, 5, 4), -10.0)
    logits[0, 2, 1] = 10.0
    logits[0, 3, 3] = 10.0
    completion_ids = torch.tensor([[1, 3]])

    token_log_probs = gather_completion_log_probs(
        logits=logits,
        completion_ids=completion_ids,
        prompt_length=3,
    )

    assert token_log_probs.shape == (1, 2)
    assert torch.all(token_log_probs > -1e-4)


def test_sequence_log_probs_averages_only_masked_completion_tokens():
    token_log_probs = torch.tensor(
        [
            [-1.0, -3.0, -99.0],
            [-2.0, -4.0, -6.0],
        ]
    )
    completion_mask = torch.tensor(
        [
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )

    result = sequence_log_probs(token_log_probs, completion_mask)

    assert torch.allclose(result, torch.tensor([-2.0, -2.0]))
