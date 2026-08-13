from pathlib import Path

import torch

from tiny_llm_lab.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from tiny_llm_lab.data import TextDataset
from tiny_llm_lab.model import DecoderOnlyTransformer
from tiny_llm_lab.tokenizer import CharacterTokenizer
from tiny_llm_lab.training.trainer import train_model


def training_fixture(tmp_path: Path, max_steps: int = 1) -> tuple[
    ExperimentConfig, CharacterTokenizer, TextDataset, DecoderOnlyTransformer
]:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("abcd" * 80, encoding="utf-8")
    tokenizer = CharacterTokenizer.from_text(corpus_path.read_text(encoding="utf-8"))
    config = ExperimentConfig(
        model=ModelConfig(
            vocabulary_size=tokenizer.vocabulary_size,
            context_length=8,
            embedding_dim=12,
            num_layers=1,
            num_heads=3,
            mlp_dim=24,
            dropout=0.0,
        ),
        data=DataConfig(path=corpus_path, train_fraction=0.8),
        training=TrainingConfig(
            device="cpu",
            seed=7,
            batch_size=2,
            max_steps=max_steps,
            eval_interval=1,
            eval_batches=1,
            checkpoint_interval=1,
            output_dir=tmp_path / "checkpoints",
        ),
    )
    dataset = TextDataset.from_file(corpus_path, tokenizer, train_fraction=0.8)
    return config, tokenizer, dataset, DecoderOnlyTransformer(config.model)


def test_one_training_step_has_finite_loss_and_updates_parameters(tmp_path: Path) -> None:
    config, tokenizer, dataset, model = training_fixture(tmp_path)
    before = next(model.parameters()).detach().clone()

    result = train_model(model, dataset, tokenizer, config, device=torch.device("cpu"))

    assert result.step == 1
    assert torch.isfinite(torch.tensor(result.validation_loss))
    assert not torch.equal(next(model.parameters()).detach(), before)
    assert (config.training.output_dir / "latest.pt").is_file()
    assert (config.training.output_dir / "step_000001.pt").is_file()


def test_training_resume_starts_after_saved_step(tmp_path: Path) -> None:
    config, tokenizer, dataset, model = training_fixture(tmp_path, max_steps=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.learning_rate)

    result = train_model(
        model,
        dataset,
        tokenizer,
        config,
        device=torch.device("cpu"),
        optimizer=optimizer,
        start_step=1,
    )

    optimizer_steps = {int(state["step"].item()) for state in optimizer.state.values()}
    assert result.step == 2
    assert optimizer_steps == {1}

