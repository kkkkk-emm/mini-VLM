import argparse
import json
import math
import random
import shutil
import time
from collections import Counter
from contextlib import nullcontext
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import torch
from safetensors.torch import save_model
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.datasets import ConversationSampleProcessor, VLMDataCollator, load_stage_datasets
from data.processors import get_image_processor
from models.config import TrainConfig, VLMConfig
from models.vision_language_model import VisionLanguageModel


@dataclass(frozen=True)
class Precision:
    name: str
    dtype: torch.dtype
    use_grad_scaler: bool


def resolve_precision(requested: str, *, device: torch.device) -> Precision:
    """解析并返回数值精度配置。

    当 `requested` 为 "auto" 时，会根据设备能力（CUDA 是否支持 bfloat16）选择合适精度。
    返回 `Precision`，包含名字、对应的 torch.dtype 以及是否使用 grad scaler。
    """
    if requested == "auto":
        if device.type == "cuda":
            requested = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
        else:
            requested = "fp32"
    if requested == "bf16":
        return Precision("bf16", torch.bfloat16, False)
    if requested == "fp16":
        return Precision("fp16", torch.float16, device.type == "cuda")
    if requested == "fp32":
        return Precision("fp32", torch.float32, False)
    raise ValueError("precision must be one of: auto, bf16, fp16, fp32")


def select_device() -> torch.device:
    """选择可用设备，优先 CUDA，其次 MPS，最后 CPU。"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_trainable_parameters(model, stage: str):
    """基于训练阶段配置模型中哪些参数需要参与优化。

    - 'pretrain': 通常解冻 projector 与 MoE/MLP（若配置使用 MoE），用于大规模预训练。
    - 'sft': 解冻 projector 与 decoder，用于监督微调。
    - 'grpo': 仅解冻 decoder，用于策略梯度基于生成结果的优化。

    参数:
        model: 要修改的模型实例。
        stage: 训练阶段标识字符串。
    """
    if stage not in {"pretrain", "sft", "grpo"}:
        raise ValueError("stage must be one of: pretrain, sft, grpo")
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.projector.parameters():
        parameter.requires_grad = True
        
    if stage == "sft" or stage == "grpo":
        for parameter in model.decoder.parameters():
            parameter.requires_grad = True
    elif model.cfg.lm_use_moe:
        for block in model.decoder.blocks:
            for parameter in block.mlp.parameters():
                parameter.requires_grad = True


def parameter_trainability_report(model):
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    trainable_percentage = (
        100.0 * trainable_parameters / total_parameters if total_parameters else 0.0
    )

    def module_frozen(module):
        parameters = list(module.parameters())
        return bool(parameters) and all(not parameter.requires_grad for parameter in parameters)

    def module_trainable(module):
        return any(parameter.requires_grad for parameter in module.parameters())

    return {
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "trainable_percentage": trainable_percentage,
        "vision_encoder_frozen": module_frozen(model.vision_encoder),
        "projector_frozen": module_frozen(model.projector),
        "decoder_trainable": module_trainable(model.decoder),
    }


def print_trainability_report(model):
    """打印模型参数可训练性报告并返回该报告字典。

    该报告包含参数总数、可训练参数数目以及各主要子模块是否被冻结等信息。
    """
    report = parameter_trainability_report(model)
    print("Parameter trainability:")
    print(f"  total parameters: {report['total_parameters']:,}")
    print(f"  trainable parameters: {report['trainable_parameters']:,}")
    print(f"  trainable percentage: {report['trainable_percentage']:.4f}%")
    print(f"  vision encoder frozen: {report['vision_encoder_frozen']}")
    print(f"  projector frozen: {report['projector_frozen']}")
    print(f"  decoder trainable: {report['decoder_trainable']}")
    return report


def require_sft_source(*, checkpoint: Optional[str], resume: Optional[str]):
    if checkpoint is None and resume is None:
        raise ValueError("SFT requires --checkpoint from pretrain or --resume from an SFT checkpoint")


def validate_training_args(args):
    for name in (
        "gradient_accumulation_steps",
        "stats_log_interval",
        "eval_interval",
        "checkpoint_interval",
        "max_steps",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")


def _skipped_metrics(prefix, skipped):
    """将 `skipped` 统计字典转换为带前缀的 metrics 字典，便于日志记录。

    例如传入 prefix='train' 将产生键 'train/skipped/<reason>'。
    """
    return {
        f"{prefix}/skipped/{reason}": count
        for reason, count in skipped.items()
    }


def _config_dict(config):
    if config is None:
        return {}
    if is_dataclass(config):
        return asdict(config)
    if isinstance(config, dict):
        return dict(config)
    return vars(config)


class CheckpointManager:
    """检查点管理器：负责保存模型、tokenizer 以及训练状态，并清理旧检查点。"""
    def __init__(self, output_dir: Path, *, keep_latest: int = 3):
        self.output_dir = Path(output_dir)
        self.keep_latest = keep_latest
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        *,
        model,
        tokenizer,
        optimizer,
        scheduler,
        scaler,
        step: int,
        stage: str,
        config,
        train_config=None,
        is_best: bool = False,
        best_val_loss: float = float("inf"),
    ):
        checkpoint_dir = self.output_dir / f"step-{step}"
        self._save_directory(
            checkpoint_dir,
            model=model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            step=step,
            stage=stage,
            config=config,
            train_config=train_config,
            best_val_loss=best_val_loss,
        )
        if is_best:
            self._save_directory(
                self.output_dir / "best",
                model=model,
                tokenizer=tokenizer,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                step=step,
                stage=stage,
                config=config,
                train_config=train_config,
                best_val_loss=best_val_loss,
            )
        self._remove_old_checkpoints()
        return checkpoint_dir

    def _save_directory(
        self,
        checkpoint_dir,
        *,
        model,
        tokenizer,
        optimizer,
        scheduler,
        scaler,
        step,
        stage,
        config,
        train_config,
        best_val_loss,
    ):
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_model(model, str(checkpoint_dir / "model.safetensors"))
        tokenizer.save_pretrained(str(checkpoint_dir))
        (checkpoint_dir / "config.json").write_text(
            json.dumps(_config_dict(config), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if train_config is not None:
            (checkpoint_dir / "train_config.json").write_text(
                json.dumps(_config_dict(train_config), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        state = {
            "global_step": step,
            "stage": stage,
            "best_val_loss": best_val_loss,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
        }
        torch.save(state, checkpoint_dir / "training_state.pt")

    def _remove_old_checkpoints(self):
        checkpoints = sorted(
            self.output_dir.glob("step-*"),
            key=lambda path: int(path.name.removeprefix("step-")),
        )
        for checkpoint in checkpoints[:-self.keep_latest]:
            resolved_output = self.output_dir.resolve()
            resolved_checkpoint = checkpoint.resolve()
            if resolved_checkpoint.parent != resolved_output:
                raise ValueError(f"Refusing to remove checkpoint outside {resolved_output}")
            shutil.rmtree(resolved_checkpoint)


def read_training_state(checkpoint: str):
    """从指定检查点目录加载训练状态文件 `training_state.pt` 并返回其内容。

    参数:
        checkpoint: 检查点目录路径字符串。

    返回:
        已加载的训练状态字典（用于恢复训练）。
    """
    return torch.load(
        Path(checkpoint) / "training_state.pt",
        map_location="cpu",
        weights_only=False,
    )


def restore_training_state(checkpoint, *, optimizer, scheduler, scaler):
    """恢复检查点中的优化器、调度器和 scaler 状态。

    参数:
        checkpoint: 检查点目录路径。
        optimizer: 优化器实例，会调用 `load_state_dict` 恢复状态。
        scheduler: 可选的学习率调度器实例。
        scaler: AMP scaler（可选）。

    返回:
        加载的完整训练状态字典。
    """
    state = read_training_state(checkpoint)
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state["scheduler"] is not None:
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state["scaler"] is not None:
        scaler.load_state_dict(state["scaler"])
    return state


def build_optimizer(model, cfg: TrainConfig):
    """根据模型中各子模块是否可训练及给定学习率构建 AdamW 优化器。

    会为 projector、decoder 与 vision_encoder（若对应参数可训练）分别创建参数组。
    返回 `torch.optim.AdamW` 实例。
    """
    groups = []
    for module, learning_rate in (
        (model.projector, cfg.lr_mp),
        (model.decoder, cfg.lr_language_backbone),
        (model.vision_encoder, cfg.lr_vision_backbone),
    ):
        parameters = [parameter for parameter in module.parameters() if parameter.requires_grad]
        if parameters:
            groups.append({"params": parameters, "lr": learning_rate})
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def build_scheduler(optimizer, *, max_steps: int, warmup_ratio: float):
    """构建学习率调度器：线性 warmup + cosine decay（使用 LambdaLR）。"""
    warmup_steps = int(max_steps * warmup_ratio)

    def lr_lambda(step):
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class SwanLabLogger:
    def __init__(self, *, enabled, project, workspace, mode, stage, config):
        """封装 swanlab 的简单日志接口。

        当 `enabled` 为 True 时尝试导入并初始化 `swanlab`，否则成为空实现。
        """
        self.enabled = enabled
        self.run = None
        if not enabled:
            return
        try:
            import swanlab
        except ImportError as error:
            raise RuntimeError("SwanLab logging is enabled but swanlab is not installed") from error
        self.swanlab = swanlab
        self.run = swanlab.init(
            project=project,
            workspace=workspace,
            experiment_name=f"{stage}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            mode=mode,
            config=_config_dict(config),
        )

    def log(self, metrics, *, step):
        if self.enabled:
            self.swanlab.log(metrics, step=step)

    def finish(self):
        if self.run is not None and hasattr(self.run, "finish"):
            self.run.finish()


def _log_step_metrics(logger, metrics, *, step, validation=None):
    combined_metrics = dict(metrics)
    if validation is not None:
        combined_metrics.update(
            {
                "val/loss": validation["loss"],
                "val/skipped": sum(validation["skipped"].values()),
                **_skipped_metrics("val", validation["skipped"]),
            }
        )
    logger.log(combined_metrics, step=step)


def _move_batch(batch, device):
    return {
        "input_ids": batch["input_ids"].to(device),
        "target_ids": batch["target_ids"].to(device),
        "attention_mask": batch["attention_mask"].to(device),
        "images": batch["images"],
    }


def _autocast_context(device, precision):
    """根据 precision 返回合适的自动混合精度上下文管理器。

    当使用 fp32 时返回空上下文（不启用 autocast），否则返回 `torch.autocast`。
    """
    if precision.name == "fp32":
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=precision.dtype)


@torch.no_grad()
def evaluate(model, data_loader, *, device, precision, max_batches):
    was_training = model.training
    model.eval()
    losses = []
    skipped = Counter()
    for batch_index, batch in enumerate(data_loader):
        if batch_index >= max_batches:
            break
        skipped.update(batch["skipped_counts"])
        if batch["empty"]:
            continue
        moved_batch = _move_batch(batch, device)
        with _autocast_context(device, precision):
            _, loss = model(**moved_batch)
        losses.append(loss.detach().float().item())
    if was_training:
        model.train()
    return {
        "loss": sum(losses) / len(losses) if losses else float("inf"),
        "skipped": dict(skipped),
    }


def _build_data_loaders(args, cfg, model, *, stage, shuffle_seed):
    processor = ConversationSampleProcessor(
        tokenizer=model.tokenizer,
        image_processor=get_image_processor(
            model.cfg.max_img_size,
            model.cfg.vit_img_size,
            model.cfg.resize_to_max_side_len,
        ),
        cfg=SimpleNamespace(
            mp_image_token_length=model.cfg.mp_image_token_length,
            max_sample_length=args.max_sample_length,
        ),
        stage=stage,
    )
    train_dataset, val_dataset = load_stage_datasets(
        args.dataset_source,
        split=args.dataset_split,
        streaming=args.stream_dataset,
        val_size=args.val_size,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=shuffle_seed,
        processor=processor,
        dataset_name=args.dataset_name,
    )
    collator = VLMDataCollator(model.tokenizer)
    loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "collate_fn": collator,
        "pin_memory": torch.cuda.is_available(),
    }
    return DataLoader(train_dataset, **loader_args), DataLoader(val_dataset, **loader_args)


def _validate_architecture_overrides(args, cfg):
    for argument_name, config_name in (
        ("lm_use_moe", "lm_use_moe"),
        ("lm_num_experts", "lm_num_experts"),
        ("lm_num_experts_per_tok", "lm_num_experts_per_tok"),
        ("lm_moe_inter_dim", "lm_moe_inter_dim"),
    ):
        argument = getattr(args, argument_name)
        if argument is not None and argument != getattr(cfg, config_name):
            raise ValueError(f"--{argument_name.replace('_', '-')} does not match checkpoint config")


def _apply_architecture_overrides(args, cfg):
    for argument_name in (
        "lm_use_moe",
        "lm_num_experts",
        "lm_num_experts_per_tok",
        "lm_moe_inter_dim",
    ):
        argument = getattr(args, argument_name)
        if argument is not None:
            setattr(cfg, argument_name, argument)
    cfg.__post_init__()


def _load_model(args, *, stage):
    if stage == "sft":
        require_sft_source(checkpoint=args.checkpoint, resume=args.resume)
    checkpoint = args.resume or args.checkpoint
    if checkpoint is not None:
        if stage == "sft" and args.checkpoint is not None:
            state = read_training_state(args.checkpoint)
            if state["stage"] != "pretrain":
                raise ValueError("SFT --checkpoint must point to a pretrain checkpoint")
        model = VisionLanguageModel.from_pretrained(checkpoint)
        _validate_architecture_overrides(args, model.cfg)
        return model
    cfg = VLMConfig()
    _apply_architecture_overrides(args, cfg)
    return VisionLanguageModel.from_hf_backbones(cfg)


def run_training(args, *, stage: str):
    validate_training_args(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device()
    precision = resolve_precision(args.precision, device=device)
    resume_state = read_training_state(args.resume) if args.resume else None
    if resume_state is not None and resume_state["stage"] != stage:
        raise ValueError(f"Cannot resume {resume_state['stage']} checkpoint as {stage}")
    global_step = resume_state["global_step"] if resume_state is not None else 0

    model = _load_model(args, stage=stage).to(device)
    configure_trainable_parameters(model, stage)
    optimizer = build_optimizer(model, args)
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

    train_loader, val_loader = _build_data_loaders(
        args,
        args,
        model,
        stage=stage,
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
        stage=stage,
        config=args,
    )
    skipped = Counter()
    best_val_loss = (
        resume_state.get("best_val_loss", float("inf"))
        if resume_state is not None
        else float("inf")
    )
    progress = tqdm(total=args.max_steps, initial=global_step, desc=stage, unit="step")
    optimizer.zero_grad(set_to_none=True)
    accumulated_batches = 0
    accumulated_loss = 0.0
    tokens_since_log = 0
    log_started_at = time.perf_counter()

    try:
        while global_step < args.max_steps:
            produced_batch = False
            produced_valid_batch = False
            for batch in train_loader:
                produced_batch = True
                skipped.update(batch["skipped_counts"])
                if batch["empty"]:
                    continue
                produced_valid_batch = True
                moved_batch = _move_batch(batch, device)
                tokens_since_log += int(moved_batch["attention_mask"].sum().item())
                with _autocast_context(device, precision):
                    _, loss = model(**moved_batch)
                    scaled_loss = loss / args.gradient_accumulation_steps
                scaler.scale(scaled_loss).backward()
                accumulated_batches += 1
                accumulated_loss += loss.detach().float().item()
                if accumulated_batches < args.gradient_accumulation_steps:
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

                train_loss = accumulated_loss / accumulated_batches
                accumulated_batches = 0
                accumulated_loss = 0.0
                metrics = {
                    "train/loss": train_loss,
                    "train/lr": optimizer.param_groups[0]["lr"],
                    "train/grad_norm": float(grad_norm),
                    "train/skipped": sum(skipped.values()),
                    **_skipped_metrics("train", skipped),
                }
                if global_step % args.stats_log_interval == 0:
                    elapsed = max(time.perf_counter() - log_started_at, 1e-9)
                    metrics["train/tokens_per_second"] = tokens_since_log / elapsed
                    tokens_since_log = 0
                    log_started_at = time.perf_counter()
                    progress.set_postfix(
                        loss=f"{train_loss:.4f}",
                        lr=f"{optimizer.param_groups[0]['lr']:.2e}",
                        grad_norm=f"{float(grad_norm):.3f}",
                        tok_s=f"{metrics['train/tokens_per_second']:.1f}",
                        skipped=sum(skipped.values()),
                    )
                is_best = False
                validation = None
                if global_step % args.eval_interval == 0:
                    validation = evaluate(
                        model,
                        val_loader,
                        device=device,
                        precision=precision,
                        max_batches=args.max_eval_batches,
                    )
                    is_best = validation["loss"] < best_val_loss
                    best_val_loss = min(best_val_loss, validation["loss"])

                _log_step_metrics(
                    logger,
                    metrics,
                    validation=validation,
                    step=global_step,
                )

                if is_best or global_step % args.checkpoint_interval == 0:
                    checkpoint_manager.save(
                        model=model,
                        tokenizer=model.tokenizer,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        step=global_step,
                        stage=stage,
                        config=model.cfg,
                        train_config=args,
                        is_best=is_best,
                        best_val_loss=best_val_loss,
                    )
                if global_step >= args.max_steps:
                    break
            if not produced_batch:
                raise RuntimeError("Training dataset did not produce any samples")
            if not produced_valid_batch:
                raise RuntimeError("Training dataset did not produce any valid samples")
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
        stage=stage,
        config=model.cfg,
        train_config=args,
        best_val_loss=best_val_loss,
    )


def build_parser(stage: str):
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description=f"mini-VLM {stage} training")
    default_source = (
        defaults.pretrain_dataset_path if stage == "pretrain" else defaults.sft_dataset_path
    )
    default_steps = (
        defaults.pretrain_max_training_steps
        if stage == "pretrain"
        else defaults.sft_max_training_steps
    )
    parser.add_argument("--dataset-source", default=default_source)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--dataset-split", default=defaults.dataset_split)
    parser.add_argument("--stream-dataset", action=argparse.BooleanOptionalAction, default=defaults.stream_dataset)
    parser.add_argument("--shuffle-buffer-size", type=int, default=defaults.shuffle_buffer_size)
    parser.add_argument("--val-size", type=int, default=defaults.val_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=defaults.gradient_accumulation_steps)
    parser.add_argument("--max-grad-norm", type=float, default=defaults.max_grad_norm)
    parser.add_argument("--max-sample-length", type=int, default=defaults.max_sample_length)
    parser.add_argument("--max-steps", type=int, default=default_steps)
    parser.add_argument("--lr-mp", type=float, default=defaults.lr_mp)
    parser.add_argument("--lr-language-backbone", type=float, default=defaults.lr_language_backbone)
    parser.add_argument("--lr-vision-backbone", type=float, default=defaults.lr_vision_backbone)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--warmup-ratio", type=float, default=defaults.warmup_ratio)
    parser.add_argument("--eval-interval", type=int, default=defaults.eval_interval)
    parser.add_argument("--stats-log-interval", type=int, default=defaults.stats_log_interval)
    parser.add_argument("--max-eval-batches", type=int, default=defaults.max_eval_batches)
    parser.add_argument("--checkpoint-interval", type=int, default=defaults.checkpoint_interval)
    parser.add_argument("--checkpoint-keep-latest", type=int, default=defaults.checkpoint_keep_latest)
    parser.add_argument("--output-dir", default=str(Path(defaults.checkpoint_path) / stage))
    if stage == "sft":
        parser.add_argument("--checkpoint", default=None)
    else:
        parser.set_defaults(checkpoint=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--precision", choices=("auto", "bf16", "fp16", "fp32"), default=defaults.precision)
    parser.add_argument("--swanlab-enabled", action=argparse.BooleanOptionalAction, default=defaults.swanlab_enabled)
    parser.add_argument("--no-swanlab", dest="swanlab_enabled", action="store_false")
    parser.add_argument("--swanlab-project", default=defaults.swanlab_project)
    parser.add_argument("--swanlab-workspace", default=defaults.swanlab_workspace)
    parser.add_argument("--swanlab-mode", default=defaults.swanlab_mode)
    parser.add_argument("--lm-use-moe", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--lm-num-experts", type=int, default=None)
    parser.add_argument("--lm-num-experts-per-tok", type=int, default=None)
    parser.add_argument("--lm-moe-inter-dim", type=int, default=None)
    return parser
