from __future__ import annotations

import argparse
import random
import time
from collections import Counter
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


def _validate_grpo_args(args):
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
    model.vision_encoder.eval()
    model.projector.eval()
    model.decoder.train(decoder_training)


def _move_grpo_batch(batch, device):
    return {
        **batch,
        "input_ids": batch["input_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "images": batch["images"].to(device) if batch["images"] is not None else None,
    }


def _repeat_images(images, repeats: int):
    if images is None:
        return None
    if images.ndim != 4:
        raise ValueError(f"Expected images with shape [N, C, H, W], got {tuple(images.shape)}")
    return images.repeat((repeats, 1, 1, 1))


def _sample_group(model, batch, args):
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
    parsed = [result.parsed_answer or "<unparsed>" for result in reward_results]
    correctness = ["1" if result.is_correct else "0" for result in reward_results]
    return f"parsed={parsed}, correct={correctness}"


def run_grpo_training(args):
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
    zero_std_groups = 0
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
                if should_skip:
                    zero_std_groups += 1
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
                    "train/zero_std_groups": zero_std_groups,
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
                        zero_std=zero_std_groups,
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
