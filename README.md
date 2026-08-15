# Tiny Language Model Lab

Tiny Language Model Lab is an interactive, local inspection lab for a small decoder-only transformer trained from scratch on Tiny Shakespeare. It exists to make transformer internals tangible: train a model, inspect its next-token probabilities, trace causal attention, compare checkpoints, and run limited inference-time interventions.

The goal is to show the model, data path, experiments, and tradeoffs without requiring a large GPU or a pretrained-model API.

# Gallery

<img width="820" height="846" alt="Screenshot 2026-08-14 235404" src="https://github.com/user-attachments/assets/fd647d58-eee6-49ee-bd0d-8a0a2913b8d2" />
<img width="820" height="762" alt="Screenshot 2026-08-14 235434" src="https://github.com/user-attachments/assets/e0bcaa5e-f9b7-4949-b1f7-e5f9894017c6" />
<img width="820" height="721" alt="Screenshot 2026-08-14 235421" src="https://github.com/user-attachments/assets/552795bf-df23-4840-8e60-367d6d4850c7" />
<img width="820" height="606" alt="Screenshot 2026-08-14 235322" src="https://github.com/user-attachments/assets/3e8bb6d9-66c2-4ddd-80e3-96ea2172bbcc" />

## What is implemented

From scratch with PyTorch primitives:

- byte-level BPE training and a character-tokenizer comparison baseline
- configurable decoder-only transformer: token/position embeddings, causal multi-head attention, MLP blocks, residual paths, normalization, and language-model head
- train/validation loop with seed control, gradient accumulation, evaluation, resumable checkpoints, and verified timeline snapshots
- next-token distribution and generation APIs
- targeted captures for attention, hidden states, and MLP activations
- Streamlit explorers for tokenization, predictions, attention, checkpoints, activations, embeddings, interventions, and experiment results

Libraries are deliberately limited: PyTorch for tensors/training, Streamlit and Plotly for the optional local UI, and pytest for tests. NumPy is not a direct dependency.

## Quickstart

Python 3.12 is required. The pinned profiles below are the tested installation paths. Use the CUDA profile on a compatible NVIDIA system and the CPU profile elsewhere.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements/cuda-py312.txt
python -m pip install -e . --no-deps
```

CPU fallback:

```powershell
python -m pip install -r requirements/cpu-py312.txt
python -m pip install -e . --no-deps
```

Download the public-domain training corpus, then verify the complete local pipeline with the 25-update CPU smoke run:

```powershell
tiny-llm download-data --output data/tiny_shakespeare.txt
tiny-llm train --config configs/smoke.toml
```

For the modest-GPU development run used by the lab, train the existing 1.93M-parameter configuration:

```powershell
tiny-llm train --config configs/dev.toml
streamlit run src/tiny_llm_lab/app/streamlit_page.py -- --run checkpoints/runs/<run-id>
```

The default app opens **Quick Tour**. Start with `ROMEO:` or `To be, or not to be,`, inspect predictions and attention, then switch to **Timeline**. See the full [demo runbook](docs/demo.md) for the staged versioned checkpoint bundle, publication steps, and checksum verification.

## Architecture and training

The model is a directly implemented causal decoder transformer, not a wrapped GPT class. Text is encoded with deterministic byte-level BPE, split into training and validation tokens, and passed through learned token/position embeddings, stacked masked-attention and GELU-MLP blocks, then a vocabulary projection. Training saves a full resumable state plus smaller inference snapshots that power the timeline explorer.

The detailed design, tensor-capture boundaries, checkpoint metadata, and reproducibility limits are documented in [Architecture and Training](docs/architecture.md).

## Experiments and actual results

The controlled experiment suite ran two seeds (`1337`, `2027`) for seven 2,000-update conditions on the development system. These are descriptive observations, not significance tests or causal claims.

| Comparison | Observed result |
| --- | --- |
| Attention heads | 2 heads: 2.3387 bits/byte; 4: 2.3539; 8: 2.3895, at the same 1,927,296 parameters |
| Position information | Learned: 2.3539 bits/byte; sinusoidal: 2.6250 |
| Tokenization | Byte-BPE: 2.3539 bits/byte; character baseline: 2.4460 |

Token-level loss is not comparable between tokenizers, so the tokenizer study uses approximate bits per byte. The complete table, ranges, timing, and cautions are in [Controlled Experiment Results](docs/experiment-results.md).

Re-run the suite and inspect its JSON artifacts locally:

```powershell
tiny-llm experiment run --suite controlled --data data/tiny_shakespeare.txt --device cuda
streamlit run src/tiny_llm_lab/app/streamlit_page.py -- --experiments checkpoints/experiments/controlled
```

## Hardware and limits

Development targeted limited GPU memory. The normal development configuration uses FP32, 128-token contexts, batch size 16, and two accumulation steps; it is intentionally small enough for a modest local GPU. CPU execution is supported but substantially slower.

This is a tiny corpus and tiny model. Outputs are often grammatical-looking fragments rather than reliable text; results do not generalize beyond the fixed corpus, budget, and seeds. Attention maps, activation views, and interventions are exploratory tools, not proof that a head or unit has a definitive semantic role. The project does not support large-model serving, distributed training, cloud deployment, or chatbot-quality generation.

## Tests

```powershell
python -m pytest
```

Local corpora, checkpoints, caches, secrets, and model artifacts are intentionally ignored. The repository contains source, configurations, tests, and documentation only.
