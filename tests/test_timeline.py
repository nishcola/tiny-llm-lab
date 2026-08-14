from pathlib import Path
import hashlib
import json

import pytest
import torch

from tiny_llm_lab.config import DataConfig, ExperimentConfig, ModelConfig, TrainingConfig
from tiny_llm_lab.data import DatasetMetadata
from tiny_llm_lab.model import DecoderOnlyTransformer
from tiny_llm_lab.tokenizer import CharacterTokenizer
from tiny_llm_lab.app import explorer
from tiny_llm_lab.app import streamlit_page
from tiny_llm_lab.app.explorer import TimelineCheckpointCache
from tiny_llm_lab.app.streamlit_page import EmbeddingExplorerCache, render_timeline_explorer
from tiny_llm_lab.training import checkpoint
from tiny_llm_lab.training.checkpoint import (
    create_timeline_run,
    discover_timeline_run,
    load_timeline_checkpoint,
    save_timeline_checkpoint,
)


def test_timeline_retention_has_a_bounded_default() -> None:
    training = TrainingConfig()

    assert training.max_timeline_checkpoints == 12


def test_checkpoint_module_exposes_run_creation_api() -> None:
    assert hasattr(checkpoint, "create_timeline_run")


def timeline_fixture(tmp_path: Path, *, maximum: int = 3) -> tuple[ExperimentConfig, CharacterTokenizer, DatasetMetadata]:
    tokenizer = CharacterTokenizer.from_text("abcd")
    config = ExperimentConfig(
        model=ModelConfig(vocabulary_size=4, context_length=4, embedding_dim=12, num_layers=1, num_heads=3, mlp_dim=24),
        data=DataConfig(path=tmp_path / "corpus.txt"),
        training=TrainingConfig(output_dir=tmp_path / "checkpoints", run_name="timeline", max_timeline_checkpoints=maximum),
    )
    return config, tokenizer, DatasetMetadata(source="fixture", byte_count=4, sha256="digest")


def test_timeline_checkpoint_is_discoverable_and_restores_inference_weights(tmp_path: Path) -> None:
    config, tokenizer, metadata = timeline_fixture(tmp_path)
    model = DecoderOnlyTransformer(config.model).eval()
    inputs = torch.tensor([[0, 1, 2]])
    expected_logits = model(inputs).logits.detach().clone()
    run = create_timeline_run(config.training.output_dir, config=config, tokenizer=tokenizer, dataset_metadata=metadata)

    saved = save_timeline_checkpoint(
        run, model=model, tokenizer=tokenizer, config=config, step=5, validation_loss=1.25, dataset_metadata=metadata
    )
    discovered = discover_timeline_run(run.path)
    loaded = load_timeline_checkpoint(discovered, discovered.checkpoints[0])

    assert saved.path == Path("timeline/step_000005.pt")
    assert discovered.error is None
    assert discovered.checkpoints[0].validation_loss == 1.25
    assert discovered.checkpoints[0].sha256
    assert loaded.tokenizer == tokenizer
    torch.testing.assert_close(loaded.model(inputs).logits, expected_logits)


def test_timeline_retention_preserves_first_and_latest_checkpoints(tmp_path: Path) -> None:
    config, tokenizer, metadata = timeline_fixture(tmp_path, maximum=3)
    model = DecoderOnlyTransformer(config.model)
    run = create_timeline_run(config.training.output_dir, config=config, tokenizer=tokenizer, dataset_metadata=metadata)

    for step in range(1, 6):
        save_timeline_checkpoint(
            run, model=model, tokenizer=tokenizer, config=config, step=step, validation_loss=float(step), dataset_metadata=metadata
        )

    discovered = discover_timeline_run(run.path)
    assert len(discovered.checkpoints) == 3
    assert discovered.checkpoints[0].step == 1
    assert discovered.checkpoints[-1].step == 5


def test_discovery_marks_missing_checkpoint_unavailable_without_hiding_others(tmp_path: Path) -> None:
    config, tokenizer, metadata = timeline_fixture(tmp_path)
    model = DecoderOnlyTransformer(config.model)
    run = create_timeline_run(config.training.output_dir, config=config, tokenizer=tokenizer, dataset_metadata=metadata)
    for step in (1, 2):
        save_timeline_checkpoint(
            run, model=model, tokenizer=tokenizer, config=config, step=step, validation_loss=1.0, dataset_metadata=metadata
        )
    (run.path / "timeline" / "step_000001.pt").unlink()

    discovered = discover_timeline_run(run.path)

    assert [record.available for record in discovered.checkpoints] == [False, True]
    assert "missing" in (discovered.checkpoints[0].error or "")


def test_loader_rejects_a_corrupted_timeline_checkpoint(tmp_path: Path) -> None:
    config, tokenizer, metadata = timeline_fixture(tmp_path)
    run = create_timeline_run(config.training.output_dir, config=config, tokenizer=tokenizer, dataset_metadata=metadata)
    save_timeline_checkpoint(
        run,
        model=DecoderOnlyTransformer(config.model),
        tokenizer=tokenizer,
        config=config,
        step=1,
        validation_loss=1.0,
        dataset_metadata=metadata,
    )
    checkpoint_path = run.path / "timeline" / "step_000001.pt"
    checkpoint_path.write_bytes(b"corrupted")
    discovered = discover_timeline_run(run.path)

    assert discovered.checkpoints[0].available is False
    with pytest.raises(ValueError, match="size|checksum|unavailable"):
        load_timeline_checkpoint(discovered, discovered.checkpoints[0])


def test_loader_rejects_checkpoint_with_incompatible_embedded_model_config(tmp_path: Path) -> None:
    config, tokenizer, metadata = timeline_fixture(tmp_path)
    run = create_timeline_run(config.training.output_dir, config=config, tokenizer=tokenizer, dataset_metadata=metadata)
    save_timeline_checkpoint(
        run, model=DecoderOnlyTransformer(config.model), tokenizer=tokenizer, config=config, step=1, validation_loss=1.0, dataset_metadata=metadata
    )
    checkpoint_path = run.path / "timeline" / "step_000001.pt"
    payload = torch.load(checkpoint_path, weights_only=False)
    payload["config"]["model"]["dropout"] = 0.2
    torch.save(payload, checkpoint_path)
    index_path = run.path / "timeline" / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    index["checkpoints"][0]["byte_count"] = checkpoint_path.stat().st_size
    index["checkpoints"][0]["sha256"] = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    index_path.write_text(json.dumps(index), encoding="utf-8")
    discovered = discover_timeline_run(run.path)

    with pytest.raises(ValueError, match="model configuration"):
        load_timeline_checkpoint(discovered, discovered.checkpoints[0])


def test_timeline_configuration_requires_room_for_first_and_latest() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        TrainingConfig(max_timeline_checkpoints=1)


def test_explorer_exposes_a_bounded_timeline_checkpoint_cache() -> None:
    assert hasattr(explorer, "TimelineCheckpointCache")


def test_timeline_cache_reuses_recent_checkpoint_and_evicts_oldest(tmp_path: Path) -> None:
    config, tokenizer, metadata = timeline_fixture(tmp_path, maximum=3)
    model = DecoderOnlyTransformer(config.model)
    run = create_timeline_run(config.training.output_dir, config=config, tokenizer=tokenizer, dataset_metadata=metadata)
    for step in (1, 2, 3):
        save_timeline_checkpoint(
            run, model=model, tokenizer=tokenizer, config=config, step=step, validation_loss=1.0, dataset_metadata=metadata
        )
    discovered = discover_timeline_run(run.path)
    cache = TimelineCheckpointCache(discovered, torch.device("cpu"), max_entries=2)

    first = cache.load(discovered.checkpoints[0])
    assert cache.load(discovered.checkpoints[0]) is first
    cache.load(discovered.checkpoints[1])
    cache.load(discovered.checkpoints[2])

    assert cache.cached_steps == (2, 3)


def test_streamlit_page_exposes_timeline_renderer() -> None:
    assert hasattr(streamlit_page, "render_timeline_explorer")


class FakeTimelinePage:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.slider_options: list[int] = []
        self.loss_plots: list[object] = []

    def title(self, value: str) -> None: pass
    def caption(self, value: str) -> None: pass
    def info(self, value: str) -> None: pass
    def error(self, value: str) -> None: self.errors.append(value)
    def select_slider(self, label: str, *, options: list[int], value: int, **kwargs: object) -> int:
        self.slider_options = options
        return value
    def subheader(self, value: str) -> None: pass
    def line_chart(self, value: object, **kwargs: object) -> None: self.loss_plots.append(value)
    def text_area(self, label: str, *, value: str) -> str: return ""
    def number_input(self, label: str, **kwargs: object) -> object: return kwargs["value"]


def test_timeline_renderer_offers_steps_and_plots_available_loss_history(tmp_path: Path) -> None:
    config, tokenizer, metadata = timeline_fixture(tmp_path)
    run = create_timeline_run(config.training.output_dir, config=config, tokenizer=tokenizer, dataset_metadata=metadata)
    model = DecoderOnlyTransformer(config.model)
    save_timeline_checkpoint(
        run, model=model, tokenizer=tokenizer, config=config, step=1, validation_loss=1.0, dataset_metadata=metadata
    )
    checkpoint.append_timeline_metric(run, step=1, training_loss=1.2, validation_loss=1.0)
    discovered = discover_timeline_run(run.path)
    page = FakeTimelinePage()

    render_timeline_explorer(page, discovered, TimelineCheckpointCache(discovered, torch.device("cpu")))

    assert page.slider_options == [1]
    assert len(page.loss_plots) == 1
    assert page.errors == ["Enter a prompt to inspect its tokenization and predictions."]


class FakeTimelineEmbeddingPage:
    def __init__(self) -> None:
        self.errors: list[str] = []
        self.figures: list[object] = []

    def title(self, value: str) -> None: pass
    def caption(self, value: str) -> None: pass
    def info(self, value: str) -> None: pass
    def error(self, value: str) -> None: self.errors.append(value)
    def select_slider(self, label: str, *, options: list[int], value: int, **kwargs: object) -> int: return value
    def subheader(self, value: str) -> None: pass
    def line_chart(self, value: object, **kwargs: object) -> None: pass
    def text_area(self, label: str, *, value: str) -> str: return ""
    def number_input(self, label: str, **kwargs: object) -> object: return kwargs["value"]
    def text_input(self, label: str, *, value: str) -> str: return "a"
    def selectbox(self, label: str, options: tuple[int, ...], **kwargs: object) -> int: return options[0]
    def dataframe(self, value: object, **kwargs: object) -> None: pass
    def plotly_chart(self, figure: object, **kwargs: object) -> None: self.figures.append(figure)


def test_timeline_renderer_caches_embedding_projection_by_verified_checkpoint_checksum(tmp_path: Path) -> None:
    config, tokenizer, metadata = timeline_fixture(tmp_path)
    run = create_timeline_run(config.training.output_dir, config=config, tokenizer=tokenizer, dataset_metadata=metadata)
    save_timeline_checkpoint(
        run,
        model=DecoderOnlyTransformer(config.model),
        tokenizer=tokenizer,
        config=config,
        step=1,
        validation_loss=1.0,
        dataset_metadata=metadata,
    )
    discovered = discover_timeline_run(run.path)
    page = FakeTimelineEmbeddingPage()
    cache_directory = tmp_path / "embedding-cache"

    render_timeline_explorer(
        page,
        discovered,
        TimelineCheckpointCache(discovered, torch.device("cpu")),
        embedding_cache=EmbeddingExplorerCache(cache_directory),
    )

    assert (cache_directory / f"{discovered.checkpoints[0].sha256}.json").is_file()
    assert len(page.figures) == 1
