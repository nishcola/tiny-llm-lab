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

## Mathematical foundations

Let $B$ be batch size, $T$ sequence length, $d$ model width, $h$ the number
of attention heads, $d_h=d/h$ one head's width, and $|\mathcal{V}|$ vocabulary
size.
The equations below describe the core forward pass without dropout; during
training, the implementation also applies dropout to embeddings, attention
weights, and MLP outputs.

### Inputs and positions

For token IDs $x_1,\ldots,x_T$, the initial representation at position $t$ is
the sum of a learned token vector and a position vector:

$$
H^{(0)}_t = E_{\mathrm{token}}[x_t] + E_{\mathrm{position}}[t].
$$

The default position table is learned. The alternative fixed table used in the
position experiment is the standard sinusoidal construction, for even index
$2i$:

$$
P_{t,2i}=\sin\left(\frac{t}{10000^{2i/d}}\right),
\qquad
P_{t,2i+1}=\cos\left(\frac{t}{10000^{2i/d}}\right).
$$

The paired sine and cosine channels give each position a distinct pattern at
multiple frequencies, so the model can use position without a learned lookup
table.

### Causal multi-head attention

Each block first applies layer normalization, then creates queries, keys, and
values with one affine projection split into three tensors:

$$
[Q,K,V] = \mathrm{LN}(H)W_{QKV}+b_{QKV}.
$$

For head $r$, the unmasked score from query position $t$ to key position $s$
is the scaled dot product

$$
S^{(r)}_{t,s}=\frac{Q^{(r)}_t(K^{(r)}_s)^\mathsf{T}}{\sqrt{d_h}}.
$$

The division by $\sqrt{d_h}$ keeps score magnitudes from growing with head
width. The causal mask turns every score for a future token into
$-\infty$:

$$
\widetilde{S}^{(r)}_{t,s} =
\begin{cases}
S^{(r)}_{t,s}, & s\leq t,\\
-\infty, & s>t.
\end{cases}
\qquad
A^{(r)}_{t,:}=\mathrm{softmax}(\widetilde{S}^{(r)}_{t,:}).
$$

Because $\exp(-\infty)=0$, the softmax assigns exactly zero attention
probability to every future position. Each head mixes only the allowed value
vectors:

$$
O^{(r)}=A^{(r)}V^{(r)},\qquad
\mathrm{Attn}(H)=
\mathrm{Concat}(O^{(1)},\ldots,O^{(h)})W_O+b_O.
$$

### Decoder block and vocabulary logits

This is a pre-normalized residual transformer. For block $\ell$, attention is
added to its input, then a normalized two-layer MLP is added:

$$
\begin{aligned}
U^{(\ell)} &= H^{(\ell-1)} +
  \mathrm{Attn}(\mathrm{LN}_1(H^{(\ell-1)})),\\
M^{(\ell)} &= \mathrm{GELU}(U^{(\ell)}W_1+b_1),\\
H^{(\ell)} &= U^{(\ell)} +
  (M^{(\ell)}W_2+b_2).
\end{aligned}
$$

The implementation uses the exact GELU activation,
$
\mathrm{GELU}(z)=z\Phi(z)
=\tfrac12z\left(1+\mathrm{erf}(z/\sqrt2)\right),
$
which smoothly gates each MLP feature. Layer normalization normalizes each
token's $d$ features before applying learned scale and bias:

$$
\mathrm{LN}(z)=\gamma\odot
\frac{z-\mu(z)}{\sqrt{\sigma^2(z)+\varepsilon}}+\beta.
$$

After the final normalization, the language-model head produces one logit for
every vocabulary item:

$$
z_t=\mathrm{LN}_{\mathrm{final}}(H^{(L)}_t)W_{\mathrm{vocab}}.
$$

### Probabilities, loss, and updates

The next-token distribution is a softmax over the logits:

$$
p_\theta(y=k\mid x_{\leq t})=
\frac{\exp(z_{t,k})}{\sum_{j=1}^{|\mathcal{V}|}\exp(z_{t,j})}.
$$

For target token $y_t$, PyTorch cross-entropy is the negative log probability.
The model averages it across the $BT$ target positions in a batch:

$$
\mathcal{L}=
-\frac{1}{BT}\sum_{b=1}^{B}\sum_{t=1}^{T}
\log p_\theta(y_{b,t}\mid x_{b,\leq t}).
$$

With $a$ gradient-accumulation microbatches, the trainer backpropagates
$\mathcal{L}_i/a$ for each one before one AdamW step. By linearity of
differentiation, this produces the average gradient

$$
g=\frac{1}{a}\sum_{i=1}^{a}\nabla_\theta\mathcal{L}_i.
$$

Before the update, global-norm clipping limits its magnitude to the configured
maximum $c$:

$$
g_{\mathrm{clip}}=g\min\left(1,\frac{c}{\lVert g\rVert_2}\right).
$$

### Comparable compression and embedding views

Token-level loss is measured in natural-log units per token, so it is not
directly comparable when tokenizers produce different numbers of tokens. The
experiment runner reports approximate bits per byte for a sampled validation
loss $\mathcal{L}_{\mathrm{val}}$, $N_{\mathrm{tok}}$ validation tokens, and
$N_{\mathrm{byte}}$ UTF-8 bytes:

$$
\mathrm{bits/byte}\approx
\frac{\mathcal{L}_{\mathrm{val}}N_{\mathrm{tok}}}
{N_{\mathrm{byte}}\ln 2}.
$$

Multiplying restores total negative log likelihood in nats, dividing by
$\ln 2$ converts nats to bits, and normalizing by bytes puts the tokenizer
conditions on the same approximate unit.

The embedding explorer projects a centered embedding matrix $X$ into two
dimensions with principal-component analysis. If
$X_c=X-\mathbf{1}\mu^\mathsf{T}=U\Sigma V^\mathsf{T}$, it displays

$$
Z=X_cV_{[:,1:2]}.
$$

In the implementation, $\mu$ and $V$ are fitted on a deterministic bounded
sample when the vocabulary is large, then every embedding is projected.
Nearest neighbors use cosine similarity in the original embedding space:

$$
\mathrm{cosine}(e_i,e_j)=
\frac{e_i^\mathsf{T}e_j}{\lVert e_i\rVert_2\lVert e_j\rVert_2}.
$$

Higher cosine similarity means the vectors point in more similar directions;
the explorer excludes the selected token itself from its ranked neighbors.

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
