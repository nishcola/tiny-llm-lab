# Controlled Experiments Implementation Plan

**Goal:** Add the approved Milestone 10 experiment suite, artifact format, report, and read-only Streamlit view.

**Architecture:** Keep normal training/checkpoint workflows compatible. Put fixed experiment definitions, corpus preparation, execution, result validation, and reporting in a focused `experiments` package; the UI only reads verified artifacts.

## Tasks

1. Add failing configuration/model tests for selectable learned or sinusoidal positional information; implement the compatible configuration and model behavior.
2. Add failing experiment-definition and raw-split tests; implement fixed conditions, train-only tokenizer fitting, and reproducible preparation.
3. Add failing result-schema, aggregation, and sample tests; implement execution artifacts and the focused CLI command.
4. Add failing Streamlit renderer tests; implement the read-only results source and tables/charts/samples.
5. Generate the CUDA suite artifacts, Markdown report, and README documentation; run the full test suite.

## Constraints

- Preserve existing checkpoints by defaulting to learned positional embeddings and byte-BPE tokenization.
- Use exactly the approved 7 conditions and seeds 1337/2027.
- Keep generated artifacts under ignored `checkpoints/experiments/` and make conclusions descriptive only.
