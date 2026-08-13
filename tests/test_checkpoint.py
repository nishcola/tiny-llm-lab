from pathlib import Path

import pytest
import torch

from tiny_llm_lab.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from tiny_llm_lab.data import DatasetMetadata
from tiny_llm_lab.model import DecoderOnlyTransformer
from tiny_llm_lab.tokenizer import CharacterTokenizer
from tiny_llm_lab.training.checkpoint import load_checkpoint, save_checkpoint


def checkpoint_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        model=ModelConfig(
            vocabulary_size=4,
            context_length=6,
            embedding_dim=12,
            num_layers=1,
            num_heads=3,
            mlp_dim=24,
            dropout=0.0,
        ),
        data=DataConfig(path=tmp_path / "corpus.txt", train_fraction=0.8),
        training=TrainingConfig(max_steps=2, output_dir=tmp_path / "checkpoints"),
    )


def test_checkpoint_round_trip_preserves_predictions_and_state(tmp_path: Path) -> None:
    torch.manual_seed(11)
    config = checkpoint_config(tmp_path)
    tokenizer = CharacterTokenizer.from_text("abcd")
    model = DecoderOnlyTransformer(config.model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.tensor([[0, 1, 2, 3]])
    loss = model(inputs, inputs).loss
    assert loss is not None
    loss.backward()
    optimizer.step()
    model.eval()
    expected_logits = model(inputs).logits.detach().clone()
    checkpoint_path = tmp_path / "checkpoint.pt"
    metadata = DatasetMetadata(source="fixture", byte_count=4, sha256="abc123")

    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        config=config,
        step=7,
        validation_loss=1.25,
        dataset_metadata=metadata,
    )
    restored_model = DecoderOnlyTransformer(config.model)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    loaded = load_checkpoint(checkpoint_path, restored_model, restored_optimizer)
    restored_model.eval()

    torch.testing.assert_close(restored_model(inputs).logits, expected_logits)
    assert loaded.step == 7
    assert loaded.validation_loss == 1.25
    assert loaded.tokenizer.vocabulary == tokenizer.vocabulary
    assert loaded.config == config
    assert loaded.dataset_metadata == metadata
    assert len(restored_optimizer.state) > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_checkpoint_loads_cuda_rng_state_when_mapped_to_cuda(tmp_path: Path) -> None:
    config = checkpoint_config(tmp_path)
    tokenizer = CharacterTokenizer.from_text("abcd")
    model = DecoderOnlyTransformer(config.model).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.tensor([[0, 1, 2, 3]], device="cuda")
    loss = model(inputs, inputs).loss
    assert loss is not None
    loss.backward()
    optimizer.step()
    checkpoint_path = tmp_path / "cuda-checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        tokenizer=tokenizer,
        config=config,
        step=1,
        validation_loss=float(loss.item()),
        dataset_metadata=DatasetMetadata(source="fixture", byte_count=4, sha256="abc123"),
    )
    restored_model = DecoderOnlyTransformer(config.model).cuda()

    load_checkpoint(checkpoint_path, restored_model, map_location="cuda")

    restored_model.eval()
    torch.testing.assert_close(restored_model(inputs).logits, model.eval()(inputs).logits)
