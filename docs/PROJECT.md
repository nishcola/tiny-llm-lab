# Tiny Language Model Lab

## Project Summary

Tiny Language Model Lab is an interactive educational and engineering project for training, inspecting, and experimenting with a small transformer language model.

The goal is not to build a competitive chatbot. The goal is to make the internals of a transformer understandable, inspectable, and interactive while demonstrating strong machine learning and software engineering skills in a portfolio-ready project.

The system should let a user train or load a small language model, enter text, inspect the model's predictions, and explore what is happening inside the network.

## Portfolio Goal

This project should demonstrate:

- understanding of transformer architecture beyond calling a pretrained model API
- practical PyTorch implementation skills
- training and inference under realistic hardware constraints
- model instrumentation and interpretability
- data visualization and interactive UI work
- experiment design and quantitative evaluation
- clean software architecture and reproducible workflows

The project should be easy for a recruiter or engineer to understand from the README and easy to demo without requiring them to train a model first.

## Hardware Constraints

Development hardware:

- GPU: NVIDIA GTX 1650 Super
- System RAM: 32 GB

Assume limited GPU VRAM. Favor small models, modest sequence lengths, gradient accumulation where useful, efficient dataloading, and lightweight visualization.

Do not design the project around large pretrained models or workloads that require modern high-memory GPUs.

## Core Product Direction

The project is primarily an interactive model-internals lab, with a smaller experimental-comparison component.

A useful rough priority split is:

- 70% interactive inspection and interpretability
- 30% controlled experiments and comparisons

The interactive demo is the centerpiece.

## Core User Experience

A user should be able to:

1. Load a trained tiny transformer checkpoint.
2. Enter a text prompt.
3. See how the prompt is tokenized.
4. Inspect next-token probability distributions.
5. Adjust generation settings such as temperature and top-k.
6. Inspect attention patterns by layer and head.
7. Inspect intermediate activations or hidden states.
8. Explore learned token embeddings in 2D.
9. Compare model behavior across saved training checkpoints.
10. Run simple model interventions and observe how output changes.

## Model Scope

Target a small decoder-only transformer implemented directly in PyTorch.

Suggested model scale:

- approximately 5M to 30M parameters
- exact size should be configurable
- small enough to train locally on the available GPU

Important architectural components should be implemented explicitly rather than hidden behind a ready-made GPT model class:

- token embeddings
- positional information
- causal self-attention
- multi-head attention
- feed-forward network
- residual connections
- normalization
- output projection / language modeling head

Using PyTorch tensor operations and standard building blocks is expected. The educational value comes from owning the architecture and instrumentation rather than reimplementing low-level kernels.

## Tokenization

The project should support at least one practical tokenizer and leave room for comparisons.

Preferred baseline:

- BPE or another subword tokenizer trained on the selected corpus

Optional comparison:

- character-level tokenizer

The UI should expose token boundaries clearly so users can see how input text maps to model tokens.

## Training Data

Use a corpus that is small enough for repeated local experiments.

Good options include:

- Tiny Shakespeare for early development
- curated Wikipedia excerpts
- public-domain text
- a small programming-language corpus

Avoid copyrighted datasets with unclear redistribution rights.

The dataset should be easy to download or generate through documented steps.

## Main Features

### 1. Tokenizer Explorer

Show:

- raw input text
- token IDs
- token strings or byte pieces
- vocabulary information

Optional:

- side-by-side tokenizer comparison

### 2. Next-Token Explorer

Given a prompt, show:

- top predicted tokens
- probabilities or logits
- changes caused by temperature
- changes caused by top-k or top-p settings if implemented

This should update interactively.

### 3. Attention Explorer

Allow selection of:

- layer
- attention head

Visualize token-to-token causal attention weights.

The visualization should remain useful for short prompts and avoid trying to display excessively long sequences.

### 4. Embedding Explorer

Project learned token embeddings into 2D using a method such as PCA or UMAP.

Allow users to:

- search for a token
- inspect nearest neighbors
- hover or click points to see token labels

### 5. Training Timeline

Save checkpoints at meaningful training intervals.

Allow a user to enter one prompt and compare:

- next-token predictions across checkpoints
- generated continuations across checkpoints
- validation loss over time

This should make learning progress visible.

### 6. Activation Explorer

Expose selected hidden states or MLP activations for a prompt.

Keep the initial version practical. The project does not need to claim that a particular neuron has a definitive semantic meaning.

Useful features include:

- activation magnitude views
- top-activating tokens or positions
- comparison across layers

### 7. Model Surgery

Provide controlled interventions such as:

- disabling one attention head
- zeroing selected activations
- comparing output distributions before and after intervention

The interface should make clear that these interventions are exploratory and not proof of causal semantic interpretation unless the experiment actually establishes that.

## Experiment Component

Include a small number of focused experiments rather than a large collection of shallow comparisons.

Good candidates:

- model size versus validation loss
- learned versus sinusoidal positional embeddings
- number of attention heads while keeping total parameter count similar
- character-level versus subword tokenization
- effect of removing or disabling selected attention heads

Each experiment should record:

- model configuration
- parameter count
- training steps or tokens seen
- validation loss
- training time
- peak GPU memory if practical to measure
- representative outputs

Avoid making unsupported claims from tiny experiments. Report observed results directly.

## Suggested Technical Architecture

Keep the system modular so the ML code can be used independently of the UI.

Suggested areas:

- `model/` — transformer architecture and model configuration
- `data/` — dataset preparation and batching
- `tokenizer/` — tokenizer training and loading
- `training/` — training loop, evaluation, checkpointing
- `inference/` — generation and next-token prediction
- `instrumentation/` — hooks for attention, activations, hidden states
- `experiments/` — reproducible experiment definitions and results
- `app/` — interactive frontend/backend or local UI
- `tests/` — unit and integration tests

Exact folder names may change if the chosen framework suggests a cleaner structure.

## UI Direction

The UI should feel like a technical laboratory, not a chatbot.

Prioritize:

- clear controls
- readable plots
- side-by-side comparisons
- immediate feedback when parameters change
- explanations that are concise and technically accurate

Avoid decorative features that do not improve understanding.

A strong demo flow is:

1. enter a prompt
2. inspect top next-token predictions
3. open an attention view
4. disable a head
5. observe how the probability distribution changes
6. move between checkpoints to see how the model learned

## Non-Goals

Do not optimize for:

- chatbot quality
- very large parameter counts
- production-scale serving
- distributed training
- multi-user authentication
- billing
- cloud infrastructure unless later needed for deployment
- broad support for many model families

These would distract from the project's core portfolio value.

## Quality Bar

The finished project should:

- run reliably on the target development machine
- have a documented setup process
- include at least one downloadable or reproducible trained checkpoint
- include tests for core model and data behavior
- produce reproducible experiment results where feasible
- have screenshots or a short demo in the repository README
- explain architectural decisions and hardware trade-offs
- avoid misleading interpretability claims

## Success Criteria

The project is successful when a technical reviewer can answer the following after a short demo:

- What transformer did the developer implement?
- How was it trained?
- What constraints were imposed by the hardware?
- What can the inspection tools reveal?
- What experiments were run?
- What engineering trade-offs were made?

If those answers are obvious from the repository and demo, the project is serving its portfolio purpose.

## Scope Discipline

When making design decisions, prefer the smallest implementation that improves the core interactive lab.

Before adding a feature, ask:

1. Does it help users understand the model?
2. Does it demonstrate meaningful ML or engineering skill?
3. Will it improve the portfolio demo?

If the answer is no to all three, leave it out.
