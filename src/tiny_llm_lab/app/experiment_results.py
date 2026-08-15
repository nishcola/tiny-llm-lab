"""Read-only Streamlit rendering for verified controlled-experiment artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def render_experiment_results(page: Any, directory: str | Path) -> None:
    path = Path(directory)
    results_path = path if path.name == "results.json" else path / "results.json"
    page.title("Controlled Experiment Results")
    page.caption("Two-seed descriptive comparisons only; these results do not establish statistical significance or causal mechanisms.")
    try:
        results = json.loads(results_path.read_text(encoding="utf-8"))
        if results.get("version") != 1 or not isinstance(results.get("runs"), list) or not isinstance(results.get("summary"), list):
            raise ValueError("unsupported or incomplete results artifact")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        page.error(f"Could not load experiment results: {error}")
        return
    studies = sorted({str(run["study"]) for run in results["runs"]})
    if not studies:
        page.info("The result artifact contains no completed runs.")
        return
    study = page.selectbox("Experiment study", studies)
    summary = [row for row in results["summary"] if row["study"] == study]
    page.subheader("Condition summary")
    page.dataframe(summary, hide_index=True, width="stretch")
    if study == "tokenization":
        page.info("Token-level cross-entropy is not directly comparable across tokenizers; compare the recorded approximate bits-per-byte values instead.")
    runs = [run for run in results["runs"] if run["study"] == study]
    points = [
        {"Step": metric["step"], "Training loss": metric["training_loss"], "Validation loss": metric["validation_loss"], "Run": f"{run['condition']} / {run['seed']}"}
        for run in runs for metric in run.get("metrics", [])
    ]
    if points:
        page.subheader("Training and validation loss")
        page.line_chart(points, x="Step", y=["Training loss", "Validation loss"], color="Run")
    page.subheader("Comparable output samples")
    sample_runs = [run for run in runs if run.get("seed") == 1337] or runs
    for run in sample_runs:
        for sample in run.get("samples", []):
            page.text_area(
                f"{run['condition']} · seed {run['seed']} · {sample['prompt']}",
                value=sample["continuation"],
                height=130,
                disabled=True,
            )
