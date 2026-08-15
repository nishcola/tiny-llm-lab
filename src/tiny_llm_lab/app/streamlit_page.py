"""Streamlit entry point for the next-token explorer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from tiny_llm_lab.embedding_analysis import (
    CheckpointDigestCache,
    EmbeddingProjection,
    EmbeddingProjectionCache,
    extract_token_embeddings,
    nearest_neighbors,
    plot_token_ids,
    search_token_labels,
)
from tiny_llm_lab.app.activation_explorer import (
    MLP_REPRESENTATION,
    PromptActivationResult,
    inspect_mlp_activation,
    scan_mlp_activation,
)
from tiny_llm_lab.app.explorer import (
    AttentionView,
    ExplorerSession,
    TimelineCheckpointCache,
    inspect_prompt,
)
from tiny_llm_lab.app.formatting import format_token
from tiny_llm_lab.training.checkpoint import (
    TimelineRun,
    discover_timeline_run,
    load_inference_checkpoint,
)
from tiny_llm_lab.training.trainer import select_device


class EmbeddingExplorerCache:
    """Memory and disk projection cache scoped to one Streamlit explorer session."""

    def __init__(self, directory: str | Path) -> None:
        self.disk_cache = EmbeddingProjectionCache(directory)
        self._projections: dict[tuple[str, tuple[str, ...]], EmbeddingProjection] = {}

    def projection(
        self, session: ExplorerSession, checkpoint_digest: str, token_labels: tuple[str, ...]
    ) -> EmbeddingProjection:
        key = (checkpoint_digest, token_labels)
        if key not in self._projections:
            cached = self.disk_cache.load(checkpoint_digest, token_labels)
            if cached is None:
                cached = self.disk_cache.load_or_compute(
                    checkpoint_digest, token_labels, extract_token_embeddings(session.model)
                )
            self._projections[key] = cached
        return self._projections[key]


def render_explorer(
    page: Any,
    session: ExplorerSession,
    *,
    default_prompt: str = "",
    embedding_cache: EmbeddingExplorerCache | None = None,
    checkpoint_digest: str | None = None,
) -> None:
    """Render the page against Streamlit or a compatible test double."""
    page.title("Next-Token Explorer")
    page.caption("Inspect tokenization and the trained model's next-token distribution.")
    prompt = page.text_area("Prompt", value=default_prompt)
    temperature = float(page.number_input("Temperature", min_value=0.01, value=1.0, step=0.1))
    display_count = int(
        page.number_input(
            "Top-k display count",
            min_value=1,
            max_value=session.tokenizer.vocabulary_size,
            value=min(10, session.tokenizer.vocabulary_size),
            step=1,
        )
    )
    if not prompt:
        page.error("Enter a prompt to inspect its tokenization and predictions.")
        if embedding_cache is not None and checkpoint_digest is not None:
            render_embedding_explorer(page, session, embedding_cache, checkpoint_digest=checkpoint_digest)
        return
    page.subheader("Attention Explorer")
    layer_index = int(
        page.selectbox(
            "Transformer layer",
            options=range(session.model.config.num_layers),
            format_func=lambda value: f"Layer {value}",
        )
    )
    head_index = int(
        page.selectbox(
            "Attention head",
            options=range(session.model.config.num_heads),
            format_func=lambda value: f"Head {value}",
        )
    )
    try:
        inspection = inspect_prompt(
            session,
            prompt,
            temperature=temperature,
            display_count=display_count,
            layer_index=layer_index,
            head_index=head_index,
        )
    except ValueError as error:
        page.error(str(error))
        return
    result = inspection.next_token
    page.subheader("Prompt tokens")
    page.dataframe(
        [{"Token ID": row.token_id, "Token text": row.text} for row in result.tokens],
        hide_index=True,
        width="stretch",
    )
    page.subheader("Top next-token predictions")
    page.dataframe(
        [
            {"Token": row.text, "Probability": f"{row.probability:.2%}"}
            for row in result.predictions
        ],
        hide_index=True,
        width="stretch",
    )
    if inspection.attention.was_truncated:
        page.info(
            f"Showing attention for the first {len(inspection.attention.view.token_labels)} "
            "tokens to keep the matrix readable."
        )
    render_attention_heatmap(page, inspection.attention.view)
    if hasattr(session.model.config, "mlp_dim"):
        render_activation_explorer(page, session, prompt)
    if embedding_cache is not None and checkpoint_digest is not None:
        render_embedding_explorer(page, session, embedding_cache, checkpoint_digest=checkpoint_digest)


def render_activation_explorer(page: Any, session: ExplorerSession, prompt: str) -> None:
    """Render one selected post-GELU MLP unit over the prompt tokens."""
    page.subheader("MLP Activation Explorer")
    page.caption(
        f"Showing: {MLP_REPRESENTATION}. This is exploratory analysis: a high activation "
        "does not, by itself, establish a human-readable meaning for a unit."
    )
    layer_index = int(
        page.selectbox(
            "MLP layer",
            options=range(session.model.config.num_layers),
            format_func=lambda value: f"Layer {value}",
        )
    )
    unit_index = int(
        page.selectbox(
            "MLP hidden unit",
            options=range(session.model.config.mlp_dim),
            format_func=lambda value: f"Unit {value}",
        )
    )
    try:
        result = inspect_mlp_activation(
            session, prompt, layer_index=layer_index, unit_index=unit_index
        )
    except ValueError as error:
        page.error(str(error))
        return
    render_activation_plot(page, result)
    if session.config is None:
        page.info("The checkpoint does not include a configured training corpus to scan.")
        return
    if not page.button("Scan training corpus"):
        return
    try:
        scan = scan_mlp_activation(
            session,
            session.config.data.path,
            train_fraction=session.config.data.train_fraction,
            layer_index=layer_index,
            unit_index=unit_index,
        )
    except ValueError as error:
        page.error(str(error))
        return
    page.subheader("Strongest positive activations in the training corpus")
    page.caption(
        f"Scanned {scan.scanned_tokens:,} training tokens from {scan.source}; results are examples, "
        "not evidence of a definitive unit meaning."
    )
    page.dataframe(
        [
            {
                "Token position": match.token_position,
                "Token": match.token,
                "Activation": f"{match.value:.6f}",
                "Text snippet": match.snippet,
            }
            for match in scan.matches
        ],
        hide_index=True,
        width="stretch",
    )


def render_activation_plot(page: Any, result: PromptActivationResult) -> None:
    """Render signed unit values with magnitude available in hover text."""
    import plotly.graph_objects as go

    figure = go.Figure(
        go.Bar(
            x=[token.position for token in result.tokens],
            y=[token.value for token in result.tokens],
            customdata=[(token.token, token.magnitude) for token in result.tokens],
            hovertemplate=(
                "Position: %{x}<br>Token: %{customdata[0]}<br>"
                "Activation: %{y:.6f}<br>Magnitude: %{customdata[1]:.6f}<extra></extra>"
            ),
            marker_color="#8da0cb",
        )
    )
    figure.update_layout(
        height=360,
        margin={"l": 45, "r": 25, "t": 25, "b": 50},
        xaxis_title="Input token position",
        yaxis_title="Post-GELU activation value",
    )
    page.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def render_embedding_explorer(
    page: Any,
    session: ExplorerSession,
    cache: EmbeddingExplorerCache,
    *,
    checkpoint_digest: str,
    default_search: str = "",
) -> None:
    """Render a searchable PCA projection and original-space cosine neighbors."""
    page.subheader("Embedding Explorer")
    page.caption(
        "Coordinates use centered PCA of learned input token embeddings. "
        "Neighbor scores are cosine similarity in the original embedding space."
    )
    token_labels = tuple(
        format_token(session.tokenizer, token_id)
        for token_id in range(session.tokenizer.vocabulary_size)
    )
    try:
        projection = cache.projection(session, checkpoint_digest, token_labels)
    except (OSError, RuntimeError, ValueError) as error:
        page.info(f"Could not prepare embedding projection: {error}")
        return
    query = page.text_input("Search for a token", value=default_search)
    matches = search_token_labels(token_labels, query)
    selected_token_id: int | None = None
    if query and not matches:
        page.info("No token labels match that search.")
    elif matches:
        selected_token_id = int(
            page.selectbox(
                "Matching token",
                options=matches,
                format_func=lambda token_id: f"{token_id} · {token_labels[token_id]}",
            )
        )
    neighbors = (
        nearest_neighbors(session.model.token_embeddings.weight, selected_token_id)
        if selected_token_id is not None
        else ()
    )
    _render_embedding_scatter(page, projection, selected_token_id, neighbors)
    if selected_token_id is not None:
        page.dataframe(
            [
                {
                    "Token ID": neighbor.token_id,
                    "Token": token_labels[neighbor.token_id],
                    "Cosine similarity": f"{neighbor.cosine_similarity:.4f}",
                }
                for neighbor in neighbors
            ],
            hide_index=True,
            width="stretch",
        )


def _render_embedding_scatter(
    page: Any,
    projection: EmbeddingProjection,
    selected_token_id: int | None,
    neighbors: tuple[Any, ...],
) -> None:
    import plotly.graph_objects as go

    highlighted = (() if selected_token_id is None else (selected_token_id,)) + tuple(
        neighbor.token_id for neighbor in neighbors
    )
    point_ids = plot_token_ids(len(projection.token_labels), include=highlighted)
    coordinates = projection.coordinate_tensor()
    figure = go.Figure(
        go.Scattergl(
            x=coordinates[list(point_ids), 0].tolist(),
            y=coordinates[list(point_ids), 1].tolist(),
            mode="markers",
            marker={"color": "#8da0cb", "size": 6, "opacity": 0.65},
            customdata=[(token_id, projection.token_labels[token_id]) for token_id in point_ids],
            hovertemplate="Token: %{customdata[1]}<br>ID: %{customdata[0]}<extra></extra>",
            name="Tokens",
        )
    )
    if neighbors:
        neighbor_ids = [neighbor.token_id for neighbor in neighbors]
        figure.add_trace(
            go.Scattergl(
                x=coordinates[neighbor_ids, 0].tolist(),
                y=coordinates[neighbor_ids, 1].tolist(),
                mode="markers",
                marker={"color": "#fc8d62", "size": 10},
                customdata=[
                    (neighbor.token_id, projection.token_labels[neighbor.token_id], neighbor.cosine_similarity)
                    for neighbor in neighbors
                ],
                hovertemplate=(
                    "Token: %{customdata[1]}<br>ID: %{customdata[0]}<br>"
                    "Cosine similarity: %{customdata[2]:.4f}<extra></extra>"
                ),
                name="Nearest neighbors",
            )
        )
    if selected_token_id is not None:
        figure.add_trace(
            go.Scattergl(
                x=[float(coordinates[selected_token_id, 0])],
                y=[float(coordinates[selected_token_id, 1])],
                mode="markers",
                marker={"color": "#d62728", "size": 13, "symbol": "star"},
                hovertemplate=(
                    f"Selected token: {projection.token_labels[selected_token_id]}<br>"
                    f"ID: {selected_token_id}<extra></extra>"
                ),
                name="Selected token",
            )
        )
    figure.update_layout(
        height=540,
        margin={"l": 30, "r": 30, "t": 30, "b": 45},
        xaxis_title="PCA component 1",
        yaxis_title="PCA component 2",
        legend={"orientation": "h"},
    )
    page.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})
    if len(point_ids) < len(projection.token_labels):
        page.info(
            f"Showing {len(point_ids):,} of {len(projection.token_labels):,} tokens for responsiveness; "
            "the selected token and its neighbors are always included."
        )


def render_attention_heatmap(page: Any, view: AttentionView) -> None:
    """Render one causal-attention matrix with non-interactive masked future cells."""
    import plotly.graph_objects as go

    figure = go.Figure(
        go.Heatmap(
            z=view.values,
            x=view.token_labels,
            y=view.token_labels,
            colorscale="Viridis",
            colorbar={"title": "Attention weight"},
            hoverongaps=False,
            hovertemplate="Query: %{y}<br>Key: %{x}<br>Attention: %{z:.6f}<extra></extra>",
        )
    )
    figure.update_layout(
        height=max(450, len(view.token_labels) * 28),
        margin={"l": 120, "r": 30, "t": 60, "b": 120},
        plot_bgcolor="#e6e6e6",
        xaxis={"side": "top", "tickangle": -45, "automargin": True},
        yaxis={"autorange": "reversed", "automargin": True, "scaleanchor": "x", "scaleratio": 1},
    )
    page.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False})


def render_timeline_explorer(
    page: Any,
    run: TimelineRun,
    cache: TimelineCheckpointCache,
    *,
    default_prompt: str = "",
    embedding_cache: EmbeddingExplorerCache | None = None,
) -> None:
    """Render one checkpoint at a time while retaining a bounded model cache."""
    page.title("Training Checkpoint Timeline")
    if run.error:
        page.error(f"Could not discover run: {run.error}")
        return
    available = [checkpoint for checkpoint in run.checkpoints if checkpoint.available]
    unavailable = [checkpoint for checkpoint in run.checkpoints if not checkpoint.available]
    if unavailable:
        page.info(f"{len(unavailable)} checkpoint(s) are unavailable and cannot be inspected.")
    if not available:
        page.error("This run contains no usable timeline checkpoints.")
        return
    selected_step = int(
        page.select_slider(
            "Checkpoint step",
            options=[checkpoint.step for checkpoint in available],
            value=available[-1].step,
            format_func=lambda step: _checkpoint_label(next(item for item in available if item.step == step)),
        )
    )
    checkpoint = next(item for item in available if item.step == selected_step)
    try:
        session = cache.load(checkpoint)
    except (OSError, RuntimeError, ValueError) as error:
        page.error(f"Could not load checkpoint at step {checkpoint.step}: {error}")
        return
    page.caption(_checkpoint_label(checkpoint))
    _render_loss_plot(page, run)
    render_explorer(
        page,
        session,
        default_prompt=default_prompt,
        embedding_cache=embedding_cache,
        checkpoint_digest=checkpoint.sha256 if embedding_cache is not None else None,
    )


def _checkpoint_label(checkpoint: Any) -> str:
    loss = "not evaluated" if checkpoint.validation_loss is None else f"val loss {checkpoint.validation_loss:.4f}"
    return f"Step {checkpoint.step:,} — {loss}"


def _render_loss_plot(page: Any, run: TimelineRun) -> None:
    if not run.metrics:
        return
    points = [
        {"Step": metric.step, "Training loss": metric.training_loss, "Validation loss": metric.validation_loss}
        for metric in run.metrics
        if metric.training_loss is not None or metric.validation_loss is not None
    ]
    if points:
        page.subheader("Training and validation loss")
        page.line_chart(points, x="Step", y=["Training loss", "Validation loss"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Tiny LLM next-token explorer")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--checkpoint", type=Path)
    source.add_argument("--run", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args()

    import streamlit as st

    try:
        device = select_device(arguments.device)
        if arguments.checkpoint is not None:
            loaded = load_inference_checkpoint(arguments.checkpoint, map_location=device)
        else:
            run = discover_timeline_run(arguments.run)
    except (OSError, RuntimeError, ValueError) as error:
        st.title("Next-Token Explorer")
        st.error(f"Could not load checkpoint: {error}")
        st.stop()
    if arguments.checkpoint is not None:
        digest_key = "embedding-checkpoint-digest-cache"
        if digest_key not in st.session_state:
            st.session_state[digest_key] = CheckpointDigestCache()
        embedding_key = f"embedding-projection-cache:{arguments.checkpoint.resolve()}"
        if embedding_key not in st.session_state:
            st.session_state[embedding_key] = EmbeddingExplorerCache(
                arguments.checkpoint.parent / ".tiny_llm_lab" / "embedding_pca"
            )
        render_explorer(
            st,
            ExplorerSession(
                model=loaded.model,
                tokenizer=loaded.tokenizer,
                device=torch.device(device),
                config=loaded.config,
            ),
            embedding_cache=st.session_state[embedding_key],
            checkpoint_digest=st.session_state[digest_key].digest(arguments.checkpoint),
        )
        return
    cache_key = f"timeline-cache:{run.path.resolve()}:{device.type}"
    if cache_key not in st.session_state:
        st.session_state[cache_key] = TimelineCheckpointCache(run, torch.device(device))
    embedding_key = f"embedding-projection-cache:{run.path.resolve()}"
    if embedding_key not in st.session_state:
        st.session_state[embedding_key] = EmbeddingExplorerCache(run.path / "embedding_pca")
    render_timeline_explorer(
        st,
        run,
        st.session_state[cache_key],
        embedding_cache=st.session_state[embedding_key],
    )


if __name__ == "__main__":
    main()
