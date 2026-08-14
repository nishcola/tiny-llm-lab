from hashlib import sha256
from pathlib import Path

from tiny_llm_lab.cli import _prepare_experiment, download_corpus, main
from tiny_llm_lab.config import load_config
from tiny_llm_lab.tokenizer import BytePairTokenizer


def test_download_corpus_writes_bytes_and_reports_digest(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    output = tmp_path / "nested" / "corpus.txt"
    content = b"small local corpus\n"
    source.write_bytes(content)

    metadata = download_corpus(source.as_uri(), output)

    assert output.read_bytes() == content
    assert metadata.source == source.as_uri()
    assert metadata.byte_count == len(content)
    assert metadata.sha256 == sha256(content).hexdigest()


def test_train_cli_smoke_run_writes_checkpoint(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("To be, or not to be.\n" * 30, encoding="utf-8")
    checkpoint_dir = tmp_path / "checkpoints"
    config_path = tmp_path / "smoke.toml"
    config_path.write_text(
        f"""
[model]
context_length = 8
embedding_dim = 12
num_layers = 1
num_heads = 3
mlp_dim = 24
dropout = 0.0

[data]
path = "{corpus_path.as_posix()}"
train_fraction = 0.8

[training]
device = "cpu"
seed = 5
batch_size = 2
gradient_accumulation_steps = 1
max_steps = 1
learning_rate = 0.001
weight_decay = 0.0
max_grad_norm = 1.0
eval_interval = 1
eval_batches = 1
checkpoint_interval = 1
output_dir = "{checkpoint_dir.as_posix()}"
run_name = "smoke"
""".strip(),
        encoding="utf-8",
    )

    exit_code = main(["train", "--config", str(config_path)])

    assert exit_code == 0
    assert (checkpoint_dir / "runs" / "smoke" / "resume" / "latest.pt").is_file()
    assert (checkpoint_dir / "runs" / "smoke" / "timeline" / "step_000001.pt").is_file()


def test_prepare_experiment_trains_configured_bpe_and_derives_model_vocabulary(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text("banana bandana " * 20, encoding="utf-8")
    config_path = tmp_path / "bpe.toml"
    config_path.write_text(
        f"""
[model]
context_length = 8
embedding_dim = 12
num_layers = 1
num_heads = 3
mlp_dim = 24

[data]
path = "{corpus_path.as_posix()}"

[tokenizer]
vocabulary_size = 260
""".strip(),
        encoding="utf-8",
    )

    effective_config, tokenizer, dataset = _prepare_experiment(load_config(config_path))

    assert isinstance(tokenizer, BytePairTokenizer)
    assert tokenizer.vocabulary_size == 260
    assert effective_config.model.vocabulary_size == tokenizer.vocabulary_size
    assert tokenizer.decode(dataset.train_tokens.tolist()).startswith("banana")
