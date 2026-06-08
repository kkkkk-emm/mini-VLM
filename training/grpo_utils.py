from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_group_advantages(
    rewards: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """计算群体相对优势（standardized advantages）。

    给定一个长度为 G 的奖励向量，返回标准化的 advantage、奖励的标准差和一个布尔值
    指示该组是否应被跳过（当标准差小于 eps 时认为无差异需要跳过）。

    返回值： (advantages, std, should_skip)
    """
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
    """构建 completion 部分的掩码，用于指示哪些 token 应计入序列概率。

    规则：
    - 如果提供 `pad_token_id`，则排除 pad token；
    - 如果提供 `eos_token_id`，则在每序列第一个 eos 之后的 token 不计入。

    参数:
        completion_ids: 形状 [G, L] 的生成 token id 张量。
        eos_token_id: 可选的 eos id。
        pad_token_id: 可选的 pad id。

    返回:
        FloatTensor 掩码，形状与 `completion_ids` 相同，1 表示计入，0 表示忽略。
    """
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
    """从模型输出的 logits 中收集 completion 部分每个 token 的对数概率。

    参数:
        logits: 模型前向得到的 logits，形状 [B, T_full, V]
        completion_ids: 生成的 completion ids，形状 [G, L]
        prompt_length: prompt 的 token 长度（用于在 logits 中定位 completion 的起始位置）。

    返回:
        对数概率张量，形状为 [G, L]，对应每个生成 token 的 log-prob。
    """
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
    """将 token 级的 log-prob 按掩码平均为序列级的 log-prob。

    参数:
        completion_token_log_probs: 形状 [G, L] 的 token log-prob。
        completion_mask: 形状 [G, L] 的 mask（1 表示计入）。

    返回:
        形状 [G] 的序列级 log-prob（对每序列有效 token 的平均）。
    """
    denominator = completion_mask.sum(dim=1).clamp_min(1.0)
    return (completion_token_log_probs * completion_mask).sum(dim=1) / denominator
