import torch
from torch import nn

from tiny_llm_lab.app.explorer import ExplorerSession
from tiny_llm_lab.app.streamlit_page import (
    EmbeddingExplorerCache,
    render_embedding_explorer,
    render_explorer,
)
from tiny_llm_lab.tokenizer import CharacterTokenizer


class EmbeddingFixture(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embeddings = nn.Embedding(4, 2)
        with torch.no_grad():
            self.token_embeddings.weight.copy_(
                torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]])
            )


class FakeEmbeddingPage:
    def __init__(self) -> None:
        self.captions: list[str] = []
        self.tables: list[list[dict[str, object]]] = []
        self.figures: list[object] = []
        self.infos: list[str] = []
        self.errors: list[str] = []

    def title(self, value: str) -> None: pass
    def subheader(self, value: str) -> None: pass
    def caption(self, value: str) -> None: self.captions.append(value)
    def text_area(self, label: str, *, value: str) -> str: return value
    def text_input(self, label: str, *, value: str) -> str: return value
    def number_input(self, label: str, **kwargs: object) -> object: return kwargs["value"]
    def selectbox(self, label: str, options: tuple[int, ...], **kwargs: object) -> int: return options[0]
    def dataframe(self, value: list[dict[str, object]], **kwargs: object) -> None: self.tables.append(value)
    def plotly_chart(self, figure: object, **kwargs: object) -> None: self.figures.append(figure)
    def info(self, value: str) -> None: self.infos.append(value)
    def error(self, value: str) -> None: self.errors.append(value)


def test_embedding_explorer_renders_searchable_pca_plot_and_cosine_neighbors(tmp_path) -> None:
    page = FakeEmbeddingPage()
    session = ExplorerSession(
        EmbeddingFixture(), CharacterTokenizer.from_text("abcd"), torch.device("cpu")
    )

    render_embedding_explorer(
        page,
        session,
        EmbeddingExplorerCache(tmp_path),
        checkpoint_digest="checkpoint-digest",
        default_search="a",
    )

    assert any("PCA" in caption and "cosine similarity" in caption for caption in page.captions)
    assert len(page.figures) == 1
    assert len(page.figures[0].data) == 3
    assert page.tables[0][0]["Cosine similarity"] == "0.9939"


def test_embedding_explorer_remains_available_when_no_prediction_prompt_is_entered(tmp_path) -> None:
    page = FakeEmbeddingPage()
    session = ExplorerSession(
        EmbeddingFixture(), CharacterTokenizer.from_text("abcd"), torch.device("cpu")
    )

    render_explorer(
        page,
        session,
        embedding_cache=EmbeddingExplorerCache(tmp_path),
        checkpoint_digest="checkpoint-digest",
    )

    assert page.errors == ["Enter a prompt to inspect its tokenization and predictions."]
    assert len(page.figures) == 1
