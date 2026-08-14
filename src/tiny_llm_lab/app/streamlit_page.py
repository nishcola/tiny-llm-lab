"""Streamlit entry point for the next-token explorer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from tiny_llm_lab.app.explorer import AttentionView, ExplorerSession, inspect_prompt
from tiny_llm_lab.training.checkpoint import load_inference_checkpoint
from tiny_llm_lab.training.trainer import select_device


def render_explorer(page: Any, session: ExplorerSession, *, default_prompt: str = "") -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Tiny LLM next-token explorer")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args()

    import streamlit as st

    try:
        device = select_device(arguments.device)
        loaded = load_inference_checkpoint(arguments.checkpoint, map_location=device)
    except (OSError, RuntimeError, ValueError) as error:
        st.title("Next-Token Explorer")
        st.error(f"Could not load checkpoint: {error}")
        st.stop()
    render_explorer(
        st,
        ExplorerSession(model=loaded.model, tokenizer=loaded.tokenizer, device=torch.device(device)),
    )


if __name__ == "__main__":
    main()
