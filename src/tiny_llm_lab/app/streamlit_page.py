"""Streamlit entry point for the next-token explorer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from tiny_llm_lab.app.explorer import ExplorerSession, inspect_next_token
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
    try:
        result = inspect_next_token(
            session, prompt, temperature=temperature, display_count=display_count
        )
    except ValueError as error:
        page.error(str(error))
        return
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
