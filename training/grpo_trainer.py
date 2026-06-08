from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm.auto import tqdm

from models.vision_language_model import VisionLanguageModel
from training.grpo_data import GRPO_BENCHMARKS, build_grpo_data_loader
from training.grpo_rewards import RuleRewardConfig, score_rule_reward
from training.grpo_utils import (
    build_completion_mask,
    compute_group_advantages,
    gather_completion_log_probs,
    sequence_log_probs,
)
from training.trainer import (
    CheckpointManager,
    SwanLabLogger,
    _autocast_context,
    _skipped_metrics,
    build_optimizer,
    build_scheduler,
    configure_trainable_parameters,
    print_trainability_report,
    read_training_state,
    resolve_precision,
    restore_training_state,
    select_device,
)


@dataclass
class SampledGroupStats:
    sampled_groups: int = 0
    zero_std_groups: int = 0
    zero_std_all_correct: int = 0
    zero_std_all_wrong: int = 0
    zero_std_all_unparseable: int = 0
    zero_std_other: int = 0
    sampled_completions: int = 0
    correct_completions: int = 0
    parseable_completions: int = 0
    yes_completions: int = 0
    no_completions: int = 0

    def update(self, reward_results, *, reward_std, zero_std: bool):
        self.sampled_groups += 1
        self.sampled_completions += len(reward_results)
        self.correct_completions += sum(1 for result in reward_results if result.is_correct)
        self.parseable_completions += sum(1 for result in reward_results if result.is_parseable)
        self.yes_completions += sum(1 for result in reward_results if result.parsed_answer == "yes")
        self.no_completions += sum(1 for result in reward_results if result.parsed_answer == "no")

        if not zero_std:
            return

        self.zero_std_groups += 1
        if reward_results and all(result.is_correct for result in reward_results):
            self.zero_std_all_correct += 1
        elif reward_results and all(result.is_parseable for result in reward_results) and not any(
            result.is_correct for result in reward_results
        ):
            self.zero_std_all_wrong += 1
        elif reward_results and not any(result.is_parseable for result in reward_results):
            self.zero_std_all_unparseable += 1
        else:
            self.zero_std_other += 1

    def metrics(self, *, prefix: str):
        completion_count = max(self.sampled_completions, 1)
        sampled_groups = max(self.sampled_groups, 1)
        return {
            f"{prefix}/sampled_groups": self.sampled_groups,
            f"{prefix}/zero_std_groups": self.zero_std_groups,
            f"{prefix}/zero_std_rate": self.zero_std_groups / sampled_groups,
            f"{prefix}/zero_std_all_correct": self.zero_std_all_correct,
            f"{prefix}/zero_std_all_wrong": self.zero_std_all_wrong,
            f"{prefix}/zero_std_all_unparseable": self.zero_std_all_unparseable,
            f"{prefix}/zero_std_other": self.zero_std_other,
            f"{prefix}/correct_rate": self.correct_completions / completion_count,
            f"{prefix}/parseable_rate": self.parseable_completions / completion_count,
            f"{prefix}/yes_ratio": self.yes_completions / completion_count,
            f"{prefix}/no_ratio": self.no_completions / completion_count,
        }


def _validate_grpo_args(args):
    """验证 GRPO 训练所需的命令行参数并在不合法时抛出异常。

    参数:
        args: argparse.Namespace，包含训练运行时的所有参数。

    该函数会检查参数范围、互斥性以及给定路径是否存在，并在发现问题时
    抛出 `ValueError` 或 `FileNotFoundError`。
    """
    # 参数合法性检查
    if args.prompt_batch_size != 1:
        raise ValueError("GRPO first version requires --prompt-batch-size 1")
    if args.num_generations < 2:
        raise ValueError("num_generations must be at least 2 for group-relative rewards")
    for name in (
        "max_new_tokens",
        "gradient_accumulation_steps",
        "stats_log_interval",
        "checkpoint_interval",
        "max_steps",
        "zero_std_max_consecutive",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if args.top_k < 0:
        raise ValueError("top_k must be non-negative")
    if not 0.0 < args.top_p <= 1.0:
        raise ValueError("top_p must be in the range (0, 1]")
    if args.checkpoint is None and args.resume is None:
        raise ValueError("GRPO requires --checkpoint from SFT or --resume from GRPO")
    if args.checkpoint is not None and not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {args.checkpoint}")
    if args.resume is not None and not Path(args.resume).exists():
        raise FileNotFoundError(f"Resume path does not exist: {args.resume}")


def _set_grpo_module_modes(model, *, decoder_training: bool):
    """设置模型各子模块的训练/推理模式。

    规则：视觉编码器与 projector 固定为评估模式（不更新），
    decoder 根据 `decoder_training` 参数决定是否进入训练模式。

    参数:
        model: VisionLanguageModel 实例。
        decoder_training: bool，是否将 decoder 设为训练模式。
    """
    model.vision_encoder.eval()
    model.projector.eval()
    model.decoder.train(decoder_training)


def _move_grpo_batch(batch, device):
    """将数据批次移动到指定设备（CPU/GPU）。

    仅会将 `input_ids`、`attention_mask` 和 `images`（若存在）迁移到目标设备，
    其余字段保持不变并原样返回。

    参数:
        batch: 包含张量与元数据的字典。
        device: torch.device 目标设备。

    返回:
        一个新的 batch 字典，其中上述字段已移动到 `device`。
    """
    return {
        **batch,
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "images": batch["images"].to(device) if batch["images"] is not None else None,
    }


def _repeat_images(images, repeats: int):
    """重复图像张量以匹配生成样本数（用于同一 prompt 的多次采样）。

    参数:
        images: 图像张量，形状应为 [N, C, H, W] 或者为 None。
        repeats: int，沿第 0 维重复的次数（通常为 `num_generations`）。

    返回:
        重复后的图像张量或 None（当输入为 None 时）。

    抛出:
        ValueError: 当输入张量维度不为 4 时。
    """
    if images is None:
        return None
    if images.ndim != 4:
        raise ValueError(f"Expected images with shape [N, C, H, W], got {tuple(images.shape)}")
    return images.repeat((repeats, 1, 1, 1))


def _sample_group(model, batch, args):
    """对单个 prompt 进行多次自回归采样，返回生成结果与奖励。

    流程：
    1. 将 prompt 和 attention mask 在 batch 维度重复 `num_generations` 次；
    2. 将图像按相同次数重复（若存在）；
    3. 以评估模式（decoder 禁用梯度）调用 `model.generate` 多次采样；
    4. 解码生成文本并用规则奖励函数打分，返回生成 ids、解码文本、打分详情、奖励张量与重复后的 images。

    参数:
        model: VisionLanguageModel 实例（包含 tokenizer 与 generate 方法）。
        batch: 单个训练样本的字典。
        args: 解析后的训练运行参数。

    返回:
        generated_ids: Tensor，生成的 token id，形状为 [num_generations, seq_len]
        completions: List[str]，解码后的文本完成项。
        reward_results: List[RuleRewardResult]，每个完成项的打分与解析信息。
        rewards: Tensor，形状为 [num_generations] 的奖励值。
        images: 重复后的图像张量或 None。
    """
    prompt_ids = batch["input_ids"].repeat(args.num_generations, 1)
    attention_mask = batch["attention_mask"].repeat(args.num_generations, 1)
    images = _repeat_images(batch["images"], args.num_generations)

    _set_grpo_module_modes(model, decoder_training=False)
    generated_ids = model.generate(
        input_ids=prompt_ids,
        images=images,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        greedy=False,
    )
    generated_ids = generated_ids.clone() # 将 inference tensor 转换为普通 tensor。
    _set_grpo_module_modes(model, decoder_training=True)

    completions = model.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    reward_results = [
        score_rule_reward(
            completion=completion.strip(),
            answer=batch["answer"],
            task_type=batch["task_type"],
            config=RuleRewardConfig(
                correct_reward=args.reward_correct,
                incorrect_reward=args.reward_incorrect,
                unparseable_reward=args.reward_unparseable,
                short_format_bonus=args.reward_short_format_bonus,
                verbosity_penalty_max=args.reward_verbosity_penalty_max,
                short_token_limit=args.reward_short_token_limit,
                verbosity_token_threshold=args.reward_verbosity_token_threshold,
            ),
        )
        for completion in completions
    ]
    rewards = torch.tensor(
        [result.reward for result in reward_results],
        dtype=torch.float32,
        device=batch["input_ids"].device,
    )
    return generated_ids, completions, reward_results, rewards, images


def _compute_grpo_loss(model, batch, generated_ids, images, advantages):
    """计算 GRPO 损失：基于生成序列的对数概率与群体优势值（advantages）。

    步骤：
    1. 将 prompt 重复以与生成序列对齐，构造完整输入（prompt + completion）；
    2. 使用模型前向得到 logits；
    3. 收集 completion 部分的 token 级 log-prob，并合成序列级 log-prob；
    4. 计算 loss = - mean(advantages * seq_log_probs)，用于策略梯度更新。

    参数:
        model: VisionLanguageModel 实例。
        batch: 原始 batch（用于提取 prompt）。
        generated_ids: Tensor，生成的 completion token ids，形状 [G, L]
        images: 重复后的 images 张量或 None。
        advantages: Tensor，与生成数一致的优势值向量，形状 [G]

    返回:
        loss: 标量 Tensor，可用于反向传播。
        completion_mask: Tensor，标识 completion token 的 mask。
        seq_log_probs.detach(): Tensor，序列级 log-prob（已 detach，用于记录）。
    """
    prompt_ids = batch["input_ids"].repeat(generated_ids.size(0), 1)
    prompt_attention_mask = batch["attention_mask"].repeat(generated_ids.size(0), 1)
    completion_mask = build_completion_mask(
        generated_ids,
        eos_token_id=model.tokenizer.eos_token_id,
        pad_token_id=model.tokenizer.pad_token_id,
    ).to(generated_ids.device)
    full_input_ids = torch.cat([prompt_ids, generated_ids], dim=1)
    full_attention_mask = torch.cat(
        [prompt_attention_mask, completion_mask.to(prompt_attention_mask.dtype)],
        dim=1,
    )
    logits, _ = model(
        input_ids=full_input_ids,
        images=images,
        attention_mask=full_attention_mask,
    )
    token_log_probs = gather_completion_log_probs(
        logits=logits,
        completion_ids=generated_ids,
        prompt_length=prompt_ids.size(1),
    )
    seq_log_probs = sequence_log_probs(token_log_probs, completion_mask)
    loss = -(advantages.detach() * seq_log_probs).mean()
    return loss, completion_mask, seq_log_probs.detach()


def _format_reward_debug(reward_results):
    """为调试打印格式化 reward 结果，返回解析答案与正确性简要字符串。"""
    parsed = [result.parsed_answer or "<unparsed>" for result in reward_results]
    correctness = ["1" if result.is_correct else "0" for result in reward_results]
    return f"parsed={parsed}, correct={correctness}"


def run_grpo_training(args):
    """GRPO 训练主流程。

    主要功能：参数校验、随机种子设定、模型加载与参数配置、优化器/调度器构建、数据加载、
    以及训练循环（包括多次采样、奖励计算、优势估计、策略梯度更新与检查点保存）。

    参数:
        args: argparse.Namespace，包含所有训练相关参数（见 `build_parser`）。
    """
    _validate_grpo_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device()
    precision = resolve_precision(args.precision, device=device)
    resume_state = read_training_state(args.resume) if args.resume else None
    if resume_state is not None and resume_state["stage"] != "grpo":
        raise ValueError(f"Cannot resume {resume_state['stage']} checkpoint as grpo")
    global_step = resume_state["global_step"] if resume_state is not None else 0

    source = args.resume or args.checkpoint
    model = VisionLanguageModel.from_pretrained(source).to(device)
    configure_trainable_parameters(model, "grpo")
    _set_grpo_module_modes(model, decoder_training=True)
    print_trainability_report(model)
    print(f"  vision encoder training mode: {model.vision_encoder.training}")
    print(f"  projector training mode: {model.projector.training}")
    print(f"  decoder training mode: {model.decoder.training}")

    optimizer = build_optimizer(model, args)
    if not optimizer.param_groups:
        raise RuntimeError("No trainable parameters found for GRPO")
    scheduler = build_scheduler(
        optimizer,
        max_steps=args.max_steps,
        warmup_ratio=args.warmup_ratio,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=precision.use_grad_scaler)
    if args.resume:
        restore_training_state(
            args.resume,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
        )

    train_loader, dataset_files = build_grpo_data_loader(
        args,
        model,
        shuffle_seed=args.seed + global_step,
    )
    checkpoint_manager = CheckpointManager(
        Path(args.output_dir),
        keep_latest=args.checkpoint_keep_latest,
    )
    logger = SwanLabLogger(
        enabled=args.swanlab_enabled,
        project=args.swanlab_project,
        workspace=args.swanlab_workspace,
        mode=args.swanlab_mode,
        stage="grpo",
        config={
            **vars(args),
            "dataset_files": dataset_files,
        },
    )

    skipped = Counter()
    sampled_group_stats = SampledGroupStats()
    consecutive_zero_std_groups = 0
    accumulated_groups = 0
    accumulated_loss = 0.0
    accumulated_reward_mean = 0.0
    accumulated_completion_tokens = 0.0
    log_started_at = time.perf_counter()

    progress = tqdm(total=args.max_steps, initial=global_step, desc="grpo", unit="step")
    optimizer.zero_grad(set_to_none=True)
    try:
        while global_step < args.max_steps:
            produced_batch = False
            produced_valid_sample = False
            for batch in train_loader:
                produced_batch = True
                skipped.update(batch["skipped_counts"])
                if batch["empty"]:
                    continue
                produced_valid_sample = True
                batch = _move_grpo_batch(batch, device)

                generated_ids, completions, reward_results, rewards, images = _sample_group(
                    model,
                    batch,
                    args,
                )
                advantages, reward_std, should_skip = compute_group_advantages(
                    rewards,
                    eps=args.reward_std_eps,
                )
                sampled_group_stats.update(
                    reward_results,
                    reward_std=reward_std,
                    zero_std=should_skip,
                )
                if should_skip:
                    consecutive_zero_std_groups += 1
                    if consecutive_zero_std_groups > args.zero_std_max_consecutive:
                        raise RuntimeError(
                            "Too many consecutive zero-std GRPO groups. "
                            f"Last rewards={rewards.detach().cpu().tolist()}, "
                            f"{_format_reward_debug(reward_results)}"
                        )
                    continue
                consecutive_zero_std_groups = 0

                with _autocast_context(device, precision):
                    loss, completion_mask, _ = _compute_grpo_loss(
                        model,
                        batch,
                        generated_ids,
                        images,
                        advantages,
                    )
                    scaled_loss = loss / args.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()

                accumulated_groups += 1
                accumulated_loss += loss.detach().float().item()
                accumulated_reward_mean += rewards.detach().float().mean().item()
                accumulated_completion_tokens += float(completion_mask.sum().item())
                if accumulated_groups < args.gradient_accumulation_steps:
                    continue

                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    args.max_grad_norm,
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                progress.update(1)

                train_loss = accumulated_loss / accumulated_groups
                reward_mean = accumulated_reward_mean / accumulated_groups
                metrics = {
                    "train/loss": train_loss,
                    "train/reward_mean": reward_mean,
                    "train/reward_std_last_group": reward_std.detach().float().item(),
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": float(grad_norm),
                    **sampled_group_stats.metrics(prefix="train"),
                    "train/skipped": sum(skipped.values()),
                    **_skipped_metrics("train", skipped),
                }
                if global_step % args.stats_log_interval == 0:
                    elapsed = max(time.perf_counter() - log_started_at, 1e-9)
                    metrics["train/completion_tokens_per_second"] = (
                        accumulated_completion_tokens / elapsed
                    )
                    progress.set_postfix(
                        loss=f"{train_loss:.4f}",
                        reward=f"{reward_mean:.3f}",
                        std=f"{metrics['train/reward_std_last_group']:.3f}",
                        grad_norm=f"{float(grad_norm):.3f}",
                        zero_std=sampled_group_stats.zero_std_groups,
                    )
                    log_started_at = time.perf_counter()
                logger.log(metrics, step=global_step)

                accumulated_groups = 0
                accumulated_loss = 0.0
                accumulated_reward_mean = 0.0
                accumulated_completion_tokens = 0.0

                if global_step % args.checkpoint_interval == 0:
                    checkpoint_manager.save(
                        model=model,
                        tokenizer=model.tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        step=global_step,
                        stage="grpo",
                        config=model.cfg,
                        train_config=args,
                    )
                if global_step >= args.max_steps:
                    break
            if not produced_batch:
                raise RuntimeError("GRPO dataset did not produce any samples")
            if not produced_valid_sample:
                raise RuntimeError("GRPO dataset did not produce any valid samples")
    finally:
        progress.close()
        logger.finish()

    checkpoint_manager.save(
        model=model,
        tokenizer=model.tokenizer,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        step=global_step,
        stage="grpo",
        config=model.cfg,
        train_config=args,
    )


def build_parser():
    """构建并返回命令行参数解析器。

    返回:
        argparse.ArgumentParser 已配置好 GRPO 训练所需的参数及默认值。
    """
    defaults = RuleRewardConfig()
    parser = argparse.ArgumentParser(description="mini-VLM rule-reward GRPO training")
    parser.add_argument("--checkpoint", default=None, help="SFT checkpoint directory")
    parser.add_argument("--resume", default=None, help="GRPO checkpoint directory to resume")
    parser.add_argument("--output-dir", default=str(Path("checkpoints") / "grpo"))
    parser.add_argument("--benchmark", choices=GRPO_BENCHMARKS, default="mmstar")
    parser.add_argument("--dataset-source", default=None)
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--stream-dataset", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--shuffle-buffer-size", type=int, default=10000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prompt-batch-size", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--lr-language-backbone", type=float, default=5e-6)
    parser.add_argument("--lr-mp", type=float, default=0.0)
    parser.add_argument("--lr-vision-backbone", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--reward-std-eps", type=float, default=1e-6)
    parser.add_argument("--zero-std-max-consecutive", type=int, default=50)
    parser.add_argument("--reward-correct", type=float, default=defaults.correct_reward)
    parser.add_argument("--reward-incorrect", type=float, default=defaults.incorrect_reward)
    parser.add_argument("--reward-unparseable", type=float, default=defaults.unparseable_reward)
    parser.add_argument(
        "--reward-short-format-bonus",
        type=float,
        default=defaults.short_format_bonus,
    )
    parser.add_argument(
        "--reward-verbosity-penalty-max",
        type=float,
        default=defaults.verbosity_penalty_max,
    )
    parser.add_argument(
        "--reward-short-token-limit",
        type=int,
        default=defaults.short_token_limit,
    )
    parser.add_argument(
        "--reward-verbosity-token-threshold",
        type=int,
        default=defaults.verbosity_token_threshold,
    )
    parser.add_argument("--stats-log-interval", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--checkpoint-keep-latest", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default="auto")
    parser.add_argument("--swanlab-enabled", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--no-swanlab", dest="swanlab_enabled", action="store_false")
    parser.add_argument("--swanlab-project", default="mini-VLM")
    parser.add_argument("--swanlab-workspace", default=None)
    parser.add_argument("--swanlab-mode", default="cloud")
    return parser
