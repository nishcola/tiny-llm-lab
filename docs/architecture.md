# Architecture and Training

Tiny Language Model Lab is deliberately small enough to inspect end to end. The project implements a decoder-only transformer directly with PyTorch tensor operations and standard `torch.nn` layers; it does not wrap a pretrained GPT model.

## System flow

```text
UTF-8 corpus
  -> deterministic byte-level BPE training
  -> train/validation token splits
  -> decoder-only transformer training
  -> resumable checkpoint + timeline snapshots
  -> inference/instrumentation API
  -> Streamlit inspection lab
```

The model, training, inference, and UI layers are intentionally separate. `model/` owns the transformer and optional captures; `training/` owns evaluation, seeds, resume state, and timeline persistence; `inference/` owns sampling and next-token distributions; `app/` renders already-prepared inspection results.

## Model

The default development configuration is a 1.93M-parameter decoder-only transformer: 4 layers, 192-dimensional token representations, 4 causal-attention heads, 768-wide MLPs, learned positional embeddings, 128-token context, and 0.1 dropout. A GPT-style block applies pre-normalized causal multi-head self-attention, residual addition, pre-normalized GELU MLP, and a second residual addition. Token embeddings and the final language-modeling projection are learned from scratch.

Attention applies a causal mask, so a token can only attend to earlier positions and itself. The model exposes selected attention weights, hidden states, and post-GELU MLP activations through an explicit instrumentation request. Normal forwards retain none of these optional tensors.

## Tokenization and data

The default tokenizer learns ordered byte-pair merges from the UTF-8 corpus. Its initial vocabulary contains all 256 byte values, allowing arbitrary Unicode input without an unknown-token fallback. Tokenizer state is saved in checkpoints, alongside corpus metadata and its SHA-256 digest. A character tokenizer remains available solely for the controlled tokenization comparison.

Tiny Shakespeare is downloaded on demand and split deterministically into train and validation partitions. The repository does not redistribute the corpus or model weights.

## Training and checkpoints

Training uses AdamW, gradient clipping, periodic held-out validation, reproducible seeds where practical, and configurable gradient accumulation. A run stores a full resumable checkpoint (model, optimizer, tokenizer, config, corpus metadata, step, and RNG state) plus smaller inference-only timeline snapshots. Timeline metadata records checksums and model/tokenizer fingerprints, so a corrupted or incompatible snapshot is reported as unavailable instead of silently loaded.

`configs/dev.toml` is the practical limited-memory GPU baseline. `configs/smoke.toml` intentionally runs 25 CPU updates only to validate installation and the full data-to-checkpoint path; it is not a quality benchmark.

## Inspection lab

The Streamlit app is an analysis tool, not a chatbot. Its guided path is: enter a short prompt, inspect tokenization and next-token probabilities, choose an attention head, then move through timeline checkpoints. Advanced sections retain model interventions, MLP activation inspection, and embedding PCA. Interventions are out-of-place, inference-time transformations and never modify checkpoint weights.

## Reproducibility boundaries

The tracked configuration, fixed seeds, corpus digest, tokenizer state, checkpoint metadata, and experiment JSON make reruns inspectable. Exact floating-point results can still vary with PyTorch/CUDA versions, GPU kernels, and hardware. The pinned Python 3.12 profiles record the development environment; they are not a claim of bit-for-bit reproducibility across all machines.
