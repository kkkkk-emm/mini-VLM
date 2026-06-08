from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_group_advantages(
    rewards: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    rewards = rewards.float()
    std = rewards.std(unbiased=False)
    if std.item() < eps:
        return torch.zeros_like(rewards), std, True
    advantages = (rewards - rewards.mean()) / (std + eps)
    return advantages, std, False


def build_completion_mask(
    completion_ids: torch.Tensor,
    *,
    eos_token_id: int | None,
    pad_token_id: int | None,
) -> torch.Tensor:
    mask = torch.ones_like(completion_ids, dtype=torch.float32)
    if pad_token_id is not None:
        mask = mask * (completion_ids != pad_token_id).float()

    if eos_token_id is None:
        return mask

    seq_len = completion_ids.size(1)
    positions = torch.arange(seq_len, device=completion_ids.device).unsqueeze(0)
    eos_positions = torch.where(
        completion_ids == eos_token_id,
        positions.expand_as(completion_ids),
        torch.full_like(completion_ids, seq_len),
    )
    first_eos = eos_positions.min(dim=1, keepdim=True).values
    eos_mask = (positions <= first_eos).float()
    return mask * eos_mask


def gather_completion_log_probs(
    *,
    logits: torch.Tensor,
    completion_ids: torch.Tensor,
    prompt_length: int,
) -> torch.Tensor:
    if prompt_length <= 0:
        raise ValueError("prompt_length must be positive")
    completion_length = completion_ids.size(1)
    start = prompt_length - 1
    end = start + completion_length
    if logits.size(1) < end:
        raise ValueError(
            f"logits sequence length {logits.size(1)} is shorter than required {end}"
        )
    completion_logits = logits[:, start:end, :]
    log_probs = F.log_softmax(completion_logits.float(), dim=-1)
    return log_probs.gather(dim=-1, index=completion_ids.unsqueeze(-1)).squeeze(-1)


def sequence_log_probs(
    completion_token_log_probs: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    denominator = completion_mask.sum(dim=1).clamp_min(1.0)
    return (completion_token_log_probs * completion_mask).sum(dim=1) / denominator
