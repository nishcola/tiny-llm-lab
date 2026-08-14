"""Small, reproducible training and validation loops."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random

import torch
from torch import nn

from tiny_llm_lab.config import ExperimentConfig
from tiny_llm_lab.data import TextDataset
from tiny_llm_lab.model import DecoderOnlyTransformer
from tiny_llm_lab.tokenizer import Tokenizer
from tiny_llm_lab.training.checkpoint import (
    TimelineRun,
    append_timeline_metric,
    create_timeline_run,
    save_checkpoint,
    save_timeline_checkpoint,
)


@dataclass(frozen=True)
class TrainingResult:
    step: int
    validation_loss: float


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available in this PyTorch installation")
    return torch.device(requested)


@torch.no_grad()
def evaluate(
    model: DecoderOnlyTransformer,
    dataset: TextDataset,
    config: ExperimentConfig,
    device: torch.device,
    generator: torch.Generator,
) -> float:
    was_training = model.training
    model.eval()
    losses: list[float] = []
    for _ in range(config.training.eval_batches):
        inputs, targets = dataset.get_batch(
            "validation",
            config.training.batch_size,
            config.model.context_length,
            device,
            generator,
        )
        loss = model(inputs, targets).loss
        if loss is None:
            raise RuntimeError("Model did not return a validation loss")
        losses.append(float(loss.item()))
    model.train(was_training)
    return sum(losses) / len(losses)


def train_model(
    model: DecoderOnlyTransformer,
    dataset: TextDataset,
    tokenizer: Tokenizer,
    config: ExperimentConfig,
    *,
    device: torch.device | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    start_step: int = 0,
    run: TimelineRun | None = None,
) -> TrainingResult:
    training = config.training
    if start_step >= training.max_steps:
        raise ValueError("max_steps must be greater than the checkpoint step")
    if start_step == 0:
        seed_everything(training.seed)
    selected_device = device or select_device(training.device)
    model.to(selected_device)
    model.train()
    active_optimizer = optimizer or torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    active_run = run or create_timeline_run(
        training.output_dir,
        config=config,
        tokenizer=tokenizer,
        dataset_metadata=dataset.metadata,
    )
    train_generator = torch.Generator().manual_seed(training.seed + start_step)
    eval_generator = torch.Generator().manual_seed(training.seed + 1)
    latest_validation_loss = math.nan

    for completed_step in range(start_step + 1, training.max_steps + 1):
        active_optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0.0
        for _ in range(training.gradient_accumulation_steps):
            inputs, targets = dataset.get_batch(
                "train",
                training.batch_size,
                config.model.context_length,
                selected_device,
                train_generator,
            )
            loss = model(inputs, targets).loss
            if loss is None or not torch.isfinite(loss):
                raise RuntimeError("Training produced a missing or non-finite loss")
            accumulated_loss += float(loss.item())
            (loss / training.gradient_accumulation_steps).backward()
        nn.utils.clip_grad_norm_(model.parameters(), training.max_grad_norm)
        active_optimizer.step()

        should_evaluate = completed_step % training.eval_interval == 0 or completed_step == training.max_steps
        if should_evaluate:
            latest_validation_loss = evaluate(
                model,
                dataset,
                config,
                selected_device,
                eval_generator,
            )
            mean_training_loss = accumulated_loss / training.gradient_accumulation_steps
            append_timeline_metric(
                active_run,
                step=completed_step,
                training_loss=mean_training_loss,
                validation_loss=latest_validation_loss,
            )
            print(
                f"step {completed_step}/{training.max_steps} "
                f"train_loss={mean_training_loss:.4f} val_loss={latest_validation_loss:.4f}"
            )

        should_checkpoint = (
            completed_step % training.checkpoint_interval == 0 or completed_step == training.max_steps
        )
        if should_checkpoint:
            if not math.isfinite(latest_validation_loss):
                latest_validation_loss = evaluate(
                    model,
                    dataset,
                    config,
                    selected_device,
                    eval_generator,
                )
                append_timeline_metric(
                    active_run,
                    step=completed_step,
                    training_loss=accumulated_loss / training.gradient_accumulation_steps,
                    validation_loss=latest_validation_loss,
                )
            checkpoint_arguments = {
                "model": model,
                "optimizer": active_optimizer,
                "tokenizer": tokenizer,
                "config": config,
                "step": completed_step,
                "validation_loss": latest_validation_loss,
                "dataset_metadata": dataset.metadata,
            }
            save_checkpoint(
                active_run.path / "resume" / "latest.pt",
                **checkpoint_arguments,
            )
            save_timeline_checkpoint(
                active_run,
                model=model,
                tokenizer=tokenizer,
                config=config,
                step=completed_step,
                validation_loss=latest_validation_loss,
                dataset_metadata=dataset.metadata,
            )

    return TrainingResult(step=training.max_steps, validation_loss=latest_validation_loss)
