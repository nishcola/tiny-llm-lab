"""Command-line entry points for data download and model training."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
from pathlib import Path
from typing import Sequence
from urllib.request import urlopen

import torch

from tiny_llm_lab.config import ExperimentConfig, load_config
from tiny_llm_lab.data import DatasetMetadata, TextDataset
from tiny_llm_lab.model import DecoderOnlyTransformer
from tiny_llm_lab.tokenizer import BytePairTokenizer, Tokenizer
from tiny_llm_lab.training.checkpoint import load_checkpoint
from tiny_llm_lab.training.trainer import select_device, train_model


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/"
    "master/data/tinyshakespeare/input.txt"
)


def download_corpus(source_url: str, output: str | Path) -> DatasetMetadata:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(source_url, timeout=30) as response:
        content = response.read()
    content.decode("utf-8")
    output_path.write_bytes(content)
    return DatasetMetadata(
        source=source_url,
        byte_count=len(content),
        sha256=hashlib.sha256(content).hexdigest(),
    )


def _prepare_experiment(config: ExperimentConfig) -> tuple[
    ExperimentConfig, Tokenizer, TextDataset
]:
    text = config.data.path.read_bytes().decode("utf-8")
    tokenizer = BytePairTokenizer.train(text, config.tokenizer.vocabulary_size)
    effective_config = replace(
        config,
        model=replace(config.model, vocabulary_size=tokenizer.vocabulary_size),
    )
    dataset = TextDataset.from_file(
        effective_config.data.path,
        tokenizer,
        train_fraction=effective_config.data.train_fraction,
        source=effective_config.data.source_url,
    )
    return effective_config, tokenizer, dataset


def _run_train(config_path: Path, resume_path: Path | None) -> int:
    config, tokenizer, dataset = _prepare_experiment(load_config(config_path))
    device = select_device(config.training.device)
    model = DecoderOnlyTransformer(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    start_step = 0
    if resume_path is not None:
        loaded = load_checkpoint(resume_path, model, optimizer, map_location=device)
        if loaded.tokenizer != tokenizer:
            raise ValueError("Checkpoint tokenizer does not match the configured corpus")
        if loaded.dataset_metadata.sha256 != dataset.metadata.sha256:
            raise ValueError("Checkpoint dataset digest does not match the configured corpus")
        start_step = loaded.step
        print(f"resumed {resume_path} at step {start_step}")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"device={device.type} parameters={parameter_count:,} vocabulary={tokenizer.vocabulary_size}")
    result = train_model(
        model,
        dataset,
        tokenizer,
        config,
        device=device,
        optimizer=optimizer,
        start_step=start_step,
    )
    print(f"training complete at step {result.step}; validation_loss={result.validation_loss:.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tiny-llm", description="Train a tiny decoder-only transformer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    download_parser = subparsers.add_parser("download-data", help="Download a UTF-8 text corpus")
    download_parser.add_argument("--output", type=Path, required=True)
    download_parser.add_argument("--url", default=TINY_SHAKESPEARE_URL)

    train_parser = subparsers.add_parser("train", help="Train or resume a tiny transformer")
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--resume", type=Path)
    return parser


def main(arguments: Sequence[str] | None = None) -> int:
    parsed = build_parser().parse_args(arguments)
    if parsed.command == "download-data":
        metadata = download_corpus(parsed.url, parsed.output)
        print(
            f"downloaded {metadata.byte_count} bytes to {parsed.output}; "
            f"sha256={metadata.sha256}"
        )
        return 0
    return _run_train(parsed.config, parsed.resume)


if __name__ == "__main__":
    raise SystemExit(main())
