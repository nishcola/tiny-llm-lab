# AGENTS.md

## Purpose

This repository contains **Tiny Language Model Lab**, an interactive project for training and inspecting a small decoder-only transformer.

Read `PROJECT.md` before making architectural or product decisions.

The primary objective is to create a strong ML/software-engineering portfolio project, not a production chatbot.

## Working Principles

### Keep the project small enough to finish

The target machine has:

- NVIDIA GTX 1650 Super
- 32 GB system RAM

Assume limited GPU VRAM. Do not casually increase model size, sequence length, batch size, dataset size, or dependency weight.

Prefer approaches that work comfortably on this machine.

### Own the transformer implementation

Do not replace the core model with a high-level pretrained GPT implementation.

The repository should explicitly implement and expose the important transformer components so they can be inspected and instrumented.

It is acceptable to use PyTorch primitives such as `nn.Linear`, `nn.Embedding`, tensor operations, optimizers, and standard utilities.

### Build for inspection

Model internals must be accessible to the visualization layer.

When implementing model components, preserve clean ways to capture:

- attention weights
- hidden states
- MLP activations
- logits
- token embeddings

Do not tightly couple the core model to a specific UI framework.

### Prefer reproducibility over cleverness

Training and experiments should be configurable and repeatable.

Record important settings such as:

- random seed
- dataset version or source
- tokenizer version
- model configuration
- optimizer settings
- training steps
- checkpoint metadata

### Avoid unsupported claims

Interpretability visualizations are exploratory unless an experiment establishes stronger evidence.

Do not label neurons or attention heads as definitively representing concepts based only on visual inspection.

## Expected Architecture

Use modular boundaries similar to:

```text
model/
data/
tokenizer/
training/
inference/
instrumentation/
experiments/
app/
tests/
```

These names are suggestions, not rigid requirements. Preserve the separation of responsibilities even if the exact structure changes.

## Core Model Requirements

The first complete model should be a configurable decoder-only transformer with:

- token embeddings
- positional encoding or embeddings
- causal multi-head self-attention
- feed-forward / MLP blocks
- residual connections
- normalization
- final language-modeling head

The model should support CPU and CUDA execution.

Configuration should make at least these values explicit:

- vocabulary size
- context length
- embedding dimension
- number of layers
- number of attention heads
- MLP size or expansion ratio
- dropout if used

## Training Requirements

Training code should support:

- train/validation split
- periodic validation loss
- checkpoint saving
- resume from checkpoint
- deterministic or documented seeding where practical
- configurable batch size and context length
- gradient accumulation if needed for memory limits

Do not add distributed training unless the project direction changes substantially.

## Instrumentation Requirements

Instrumentation should be implemented intentionally rather than bolted on later.

Prefer stable interfaces that can return or capture selected internal tensors without forcing every normal inference call to retain all intermediate state.

Be conscious of GPU memory when storing attention maps and activations.

For long inputs or many layers, capture only what the current view needs.

## UI Requirements

The UI is an inspection lab.

The main interactions should eventually include:

- prompt input
- tokenizer visualization
- top next-token probabilities
- generation controls
- attention visualization
- embedding visualization
- checkpoint comparison
- selected activation views
- model intervention controls

Do not make a chat interface the central design.

## Experiment Requirements

Experiments should be few, controlled, and reproducible.

Prefer 2-4 strong comparisons over many loosely controlled runs.

Each experiment should save machine-readable results when practical so plots can be regenerated.

Useful metrics include:

- training loss
- validation loss
- parameter count
- runtime
- memory usage

## Testing Expectations

Core ML logic should have real tests.

At minimum, test behavior such as:

- causal masking prevents access to future tokens
- tensor shapes are correct
- model forward pass works on small synthetic input
- loss computation is finite
- checkpoint save/load preserves predictions within expected numerical tolerance
- generation obeys requested limits
- tokenizer encode/decode behavior is understood and documented

Avoid tests that only assert that code runs without checking meaningful behavior.

## Performance Rules

Before introducing an optimization, identify the actual bottleneck.

Likely constraints on the target hardware are GPU VRAM and training speed.

Good techniques include:

- smaller batches
- gradient accumulation
- shorter context lengths during early development
- mixed precision only if stable and supported by the hardware/software stack
- avoiding retention of unnecessary intermediate tensors

Do not introduce complex optimization frameworks unless measurements justify them.

## Dependency Rules

Prefer mature, commonly used dependencies.

Avoid adding a dependency for functionality that can be implemented clearly in a few lines, especially in the core transformer code.

High-level libraries are fine for peripheral functionality such as plotting, dimensionality reduction, or UI development.

## Documentation Rules

When adding a major feature, update the relevant documentation.

Important design decisions should explain:

- what was chosen
- what alternatives were considered when relevant
- why the choice fits the project's hardware and portfolio goals

The README should eventually make the project understandable to someone who has not read the source code.

## Coding Style

Favor clear, inspectable code over compressed or overly abstract code.

For ML code:

- document tensor shapes where non-obvious
- use descriptive variable names
- keep model components small enough to understand independently
- keep configuration separate from runtime state

Avoid premature abstraction.

## Change Checklist

Before completing a substantial change, verify:

1. Does it still work within the project's hardware constraints?
2. Does it preserve model inspectability?
3. Is the behavior tested?
4. Is the design simpler than the alternatives considered?
5. Does it directly support the interactive lab or experiment goals?

## Current Priority Order

Unless a later plan overrides this, prioritize work in this order:

1. reliable tiny transformer implementation
2. dataset/tokenizer pipeline
3. training and checkpointing
4. inference and next-token inspection
5. instrumentation APIs
6. interactive UI
7. checkpoint timeline
8. attention and activation views
9. embedding explorer
10. model-surgery experiments
11. polished experiment comparisons

Do not build advanced UI before the underlying model and instrumentation APIs are stable enough to support it.

## Non-Goals

Do not add the following without an explicit project decision:

- user accounts
- payments
- multi-tenant infrastructure
- large-model serving
- distributed training
- generic agent features
- RAG
- external LLM APIs as a core dependency
- social features

## Decision Rule for Agents

When requirements are ambiguous, choose the option that best satisfies all three:

- easier to understand
- easier to run on the target hardware
- more useful in a technical portfolio demo

If a proposed feature conflicts with those goals, flag the conflict before implementing it.
