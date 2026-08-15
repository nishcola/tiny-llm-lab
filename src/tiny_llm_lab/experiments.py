"""Fixed, reproducible controlled experiments for the Tiny LLM lab."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import torch

from tiny_llm_lab.config import DataConfig, ExperimentConfig, ModelConfig, TokenizerConfig, TrainingConfig
from tiny_llm_lab.data import DatasetMetadata, TextDataset
from tiny_llm_lab.inference import generate
from tiny_llm_lab.model import DecoderOnlyTransformer
from tiny_llm_lab.tokenizer import BytePairTokenizer, CharacterTokenizer, Tokenizer
from tiny_llm_lab.training.checkpoint import discover_timeline_run
from tiny_llm_lab.training.trainer import seed_everything, select_device, train_model


MILESTONE_10_SEEDS = (1337, 2027)


@dataclass(frozen=True)
class ExperimentCondition:
    study: str
    name: str
    tokenizer_kind: str
    config: ExperimentConfig
    hypothesis: str


@dataclass(frozen=True)
class PreparedCorpus:
    raw_train_text: str
    raw_validation_text: str
    tokenizer: Tokenizer
    dataset: TextDataset


def milestone_10_conditions() -> tuple[ExperimentCondition, ...]:
    """Return the deliberately small, approved Milestone 10 comparison matrix."""
    baseline = ExperimentConfig(
        model=ModelConfig(
            context_length=128,
            embedding_dim=192,
            num_layers=4,
            num_heads=4,
            mlp_dim=768,
            dropout=0.1,
        ),
        data=DataConfig(path=Path("data/tiny_shakespeare.txt"), train_fraction=0.9),
        tokenizer=TokenizerConfig(vocabulary_size=320),
        training=TrainingConfig(
            device="auto",
            batch_size=16,
            gradient_accumulation_steps=2,
            max_steps=2000,
            learning_rate=3e-4,
            weight_decay=0.1,
            max_grad_norm=1.0,
            eval_interval=100,
            eval_batches=20,
            checkpoint_interval=2000,
            output_dir=Path("checkpoints/experiments/milestone-10"),
            max_timeline_checkpoints=2,
        ),
    )
    def condition(study: str, name: str, tokenizer_kind: str, hypothesis: str, **model: object) -> ExperimentCondition:
        return ExperimentCondition(study, name, tokenizer_kind, replace(baseline, model=replace(baseline.model, **model)), hypothesis)

    return (
        condition("tokenization", "character", "character", "Character and BPE tokenization differ in sequence compression and normalized validation entropy."),
        condition("tokenization", "byte-bpe-320", "byte_pair", "Character and BPE tokenization differ in sequence compression and normalized validation entropy."),
        condition("position", "learned", "byte_pair", "Learned and fixed sinusoidal position information may optimize differently on this bounded corpus."),
        condition("position", "sinusoidal", "byte_pair", "Learned and fixed sinusoidal position information may optimize differently on this bounded corpus.", position_encoding="sinusoidal"),
        condition("attention-heads", "2-heads", "byte_pair", "Attention factorization may affect optimization while trainable parameter count remains fixed.", num_heads=2),
        condition("attention-heads", "4-heads", "byte_pair", "Attention factorization may affect optimization while trainable parameter count remains fixed.", num_heads=4),
        condition("attention-heads", "8-heads", "byte_pair", "Attention factorization may affect optimization while trainable parameter count remains fixed.", num_heads=8),
    )


def prepare_corpus(path: str | Path, *, tokenizer_kind: str, train_fraction: float) -> PreparedCorpus:
    """Split raw UTF-8 text before fitting a tokenizer, avoiding validation leakage."""
    raw_bytes = Path(path).read_bytes()
    split_index = _utf8_split_index(raw_bytes, train_fraction)
    raw_train = raw_bytes[:split_index].decode("utf-8")
    raw_validation = raw_bytes[split_index:].decode("utf-8")
    if tokenizer_kind == "character":
        tokenizer: Tokenizer = CharacterTokenizer.from_text(raw_train)
        unseen = set(raw_validation) - set(raw_train)
        if unseen:
            raise ValueError("validation characters are missing from the train partition")
    elif tokenizer_kind == "byte_pair":
        tokenizer = BytePairTokenizer.train(raw_train, vocabulary_size=320)
    else:
        raise ValueError("tokenizer_kind must be 'character' or 'byte_pair'")
    dataset = TextDataset(
        train_tokens=torch.tensor(tokenizer.encode(raw_train), dtype=torch.long),
        validation_tokens=torch.tensor(tokenizer.encode(raw_validation), dtype=torch.long),
        metadata=DatasetMetadata(source=str(path), byte_count=len(raw_bytes), sha256=hashlib.sha256(raw_bytes).hexdigest()),
    )
    return PreparedCorpus(raw_train, raw_validation, tokenizer, dataset)


def _utf8_split_index(raw_bytes: bytes, train_fraction: float) -> int:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    index = int(len(raw_bytes) * train_fraction)
    while index > 0 and (raw_bytes[index] & 0b1100_0000) == 0b1000_0000:
        index -= 1
    if index == 0 or index == len(raw_bytes):
        raise ValueError("corpus is too short for the selected train fraction")
    return index


def summarize_results(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate the deliberately small set of seed replicas descriptively."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault((str(record["study"]), str(record["condition"])), []).append(record)
    summaries: list[dict[str, Any]] = []
    for (study, condition), group in sorted(grouped.items()):
        losses = [float(item["validation_loss"]) for item in group]
        bits_per_byte = [float(item["bits_per_byte"]) for item in group]
        times = [float(item["training_seconds"]) for item in group]
        summaries.append(
            {
                "study": study,
                "condition": condition,
                "runs": len(group),
                "validation_loss_mean": sum(losses) / len(losses),
                "validation_loss_range": [min(losses), max(losses)],
                "bits_per_byte_mean": sum(bits_per_byte) / len(bits_per_byte),
                "bits_per_byte_range": [min(bits_per_byte), max(bits_per_byte)],
                "parameter_count": int(group[0]["parameter_count"]),
                "training_seconds_mean": sum(times) / len(times),
            }
        )
    return summaries


def run_milestone_10(
    corpus_path: str | Path,
    *,
    output_dir: str | Path = Path("checkpoints/experiments/milestone-10"),
    device: str = "auto",
) -> Path:
    """Execute the fixed suite and save a self-contained, descriptive result artifact."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for condition in milestone_10_conditions():
        prepared = prepare_corpus(
            corpus_path, tokenizer_kind=condition.tokenizer_kind, train_fraction=condition.config.data.train_fraction
        )
        effective_config = replace(
            condition.config,
            model=replace(condition.config.model, vocabulary_size=prepared.tokenizer.vocabulary_size),
            data=replace(condition.config.data, path=Path(corpus_path)),
        )
        for seed in MILESTONE_10_SEEDS:
            run_name = f"{condition.study}-{condition.name}-seed-{seed}"
            config = replace(
                effective_config,
                training=replace(effective_config.training, seed=seed, device=device, output_dir=output, run_name=run_name),
            )
            selected_device = select_device(device)
            seed_everything(seed)
            if selected_device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(selected_device)
                torch.cuda.synchronize(selected_device)
            model = DecoderOnlyTransformer(config.model).to(selected_device)
            start = perf_counter()
            result = train_model(model, prepared.dataset, prepared.tokenizer, config, device=selected_device)
            if selected_device.type == "cuda":
                torch.cuda.synchronize(selected_device)
            seconds = perf_counter() - start
            timeline = discover_timeline_run(output / "runs" / run_name)
            samples = _samples(model, prepared.tokenizer, selected_device)
            validation_bytes = len(prepared.raw_validation_text.encode("utf-8"))
            record = {
                "study": condition.study,
                "condition": condition.name,
                "hypothesis": condition.hypothesis,
                "seed": seed,
                "config": config.to_dict(),
                "corpus": {
                    "sha256": prepared.dataset.metadata.sha256,
                    "train_bytes": len(prepared.raw_train_text.encode("utf-8")),
                    "validation_bytes": validation_bytes,
                },
                "tokenizer": {
                    "kind": condition.tokenizer_kind,
                    "vocabulary_size": prepared.tokenizer.vocabulary_size,
                    "train_tokens": len(prepared.dataset.train_tokens),
                    "validation_tokens": len(prepared.dataset.validation_tokens),
                    "bytes_per_token": len(prepared.raw_train_text.encode("utf-8")) / len(prepared.dataset.train_tokens),
                },
                "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
                "training_seconds": seconds,
                "peak_gpu_memory_mib": (
                    torch.cuda.max_memory_allocated(selected_device) / (1024 * 1024)
                    if selected_device.type == "cuda"
                    else None
                ),
                "validation_loss": result.validation_loss,
                "bits_per_byte": result.validation_loss * len(prepared.dataset.validation_tokens) / validation_bytes / torch.log(torch.tensor(2.0)).item(),
                "metrics": [
                    {"step": metric.step, "training_loss": metric.training_loss, "validation_loss": metric.validation_loss}
                    for metric in timeline.metrics
                ],
                "final_checkpoint": str((timeline.path / "timeline" / f"step_{result.step:06d}.pt").resolve()),
                "samples": samples,
            }
            records.append(record)
            _write_json(output / "records.json", {"version": 1, "runs": records})
    results = {"version": 1, "suite": "milestone-10", "runs": records, "summary": summarize_results(records)}
    _write_json(output / "results.json", results)
    report = output / "milestone-10-results.md"
    report.write_text(_markdown_report(results), encoding="utf-8")
    return output


def _samples(model: DecoderOnlyTransformer, tokenizer: Tokenizer, device: torch.device) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for prompt, seed in (("ROMEO:", 9001), ("To be, or not to be,", 9002)):
        generator = torch.Generator(device=device).manual_seed(seed)
        ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        generated = generate(model, ids, max_new_tokens=256, temperature=0.8, top_k=40, generator=generator)
        text = tokenizer.decode(generated.token_ids[0].tolist())
        samples.append({"prompt": prompt, "seed": seed, "continuation": text[len(prompt) : ][:160], "generated_tokens": generated.token_ids.shape[1] - ids.shape[1]})
    return samples


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _markdown_report(results: dict[str, Any]) -> str:
    lines = ["# Milestone 10 Results", "", "Two-seed descriptive observations; no significance or causal claims are made.", "", "| Study | Condition | Validation loss (mean [range]) | Approx. bits per byte (mean [range]) | Parameters | Mean training time |", "| --- | --- | --- | --- | ---: | ---: |"]
    for item in results["summary"]:
        low, high = item["validation_loss_range"]
        bpb_low, bpb_high = item["bits_per_byte_range"]
        lines.append(f"| {item['study']} | {item['condition']} | {item['validation_loss_mean']:.4f} [{low:.4f}, {high:.4f}] | {item['bits_per_byte_mean']:.4f} [{bpb_low:.4f}, {bpb_high:.4f}] | {item['parameter_count']:,} | {item['training_seconds_mean']:.1f}s |")
    return "\n".join(lines) + "\n"
