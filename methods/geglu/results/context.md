# Context: the Transformer feed-forward sublayer (circa 2019-2020)

## Research question

A Transformer alternates two kinds of sublayer: multi-head self-attention, which mixes
information *across* sequence positions, and a position-wise feed-forward network (FFN),
which transforms each position's hidden vector *independently* of the others. The FFN is
where roughly two-thirds of a Transformer's parameters and a large share of its compute
live, yet it has the simplest possible form — one linear projection up to a wider hidden
dimension, one pointwise nonlinearity, one linear projection back down:

```
FFN(x) = f(x W1 + b1) W2 + b2,        d_ff = expansion-factor x d_model  (typically 4x)
```

The pointwise function `f` is the only nonlinearity in the whole sublayer, and the hidden
activation at every unit is just `f` applied to a *single* learned linear view of the input.
The question is whether this layer — the model's main per-position transformer — can be made
to fit the language-modeling objective better, *without spending more parameters or compute*,
by changing how the hidden representation is formed rather than by making it wider. Any
candidate must (1) be a drop-in replacement confined to this sublayer (no changes to
attention, normalization, the data pipeline, the optimizer schedule, or evaluation),
(2) keep the input and output shape `(batch, length, d_model)`, and (3) be matched to the
baseline FFN in both parameter count and FLOPs so that any quality change is attributable to
the *form* of the layer, not to a larger budget.

## Background

The pointwise nonlinearity `f` has its own lineage. Early units made a hard sign decision;
the sigmoid smoothed that into a differentiable "firing rate"; the rectifier
`ReLU(x) = max(x, 0) = x · 1[x>0]` (Nair & Hinton 2010; Glorot, Bordes & Bengio 2011)
replaced it as the default because gradients flow undiminished on the positive half-line
(its derivative there is exactly 1) and it is cheap and induces sparsity. ReLU makes a hard
*sign* gate: it multiplies the input by 0 or 1 depending only on `sign(x)`. Two smoother
"weight by value rather than gate by sign" alternatives became available:

- **GELU** (Hendrycks & Gimpel 2016). `GELU(x) = x · Φ(x)`, where `Φ` is the standard-normal
  CDF, `Φ(x) = P(X ≤ x), X ~ N(0,1)`. It is derived as the *expectation* of a stochastic
  mask: take a 0/1 mask `m ~ Bernoulli(Φ(x))` (keep-probability rising with `x`, motivated by
  preactivations being approximately normal, an effect strengthened by normalization layers);
  its expected transform is `Φ(x)·x + (1−Φ(x))·0 = xΦ(x)`. So GELU multiplies `x` by a smooth,
  input-dependent value in roughly `(0,1)` instead of ReLU's hard `1[x>0]`. Its derivative is
  `GELU'(x) = Φ(x) + x·φ(x)`, where `φ(x) = exp(−x²/2)/√(2π)` is the standard-normal density.
  It is computed exactly as `x·½[1 + erf(x/√2)]`, or cheaply approximated by
  `0.5x(1 + tanh[√(2/π)(x + 0.044715 x³)])` or by `x·σ(1.702 x)`.
- **Swish / SiLU** (Ramachandran, Zoph & Le 2017; the `β=1` case is the SiLU of Elfwing et al.
  2017). `Swish_β(x) = x · σ(βx)`. Found by an automated search over compositions of unary and
  binary primitives; the best-performing functions shared the structure `b(x, g(x))` — reusing
  the raw preactivation `x` multiplied by a gate `g(x)`. Swish is smooth, **non-monotonic**
  (a small "bump" below zero), unbounded above and bounded below; `β→∞` recovers ReLU and
  `β=0` gives the scaled linear map `x/2`, so it smoothly interpolates between the two. Its derivative is
  `f'(x) = σ(βx) + βx·σ(βx)(1−σ(βx)) = βf(x) + σ(βx)(1 − βf(x))`. With `β=1`, Swish and GELU
  trace nearly the same curve — both are "`x` times a smooth CDF-like gate of `x`", and indeed
  the sigmoid approximation of GELU is exactly `x·σ(1.702 x)`, a Swish with `β≈1.702`.

A second, separate line of work is about combining two linear views of an input
*multiplicatively* rather than passing one through a fixed pointwise map:

- **Multiplicative / bilinear interactions** (Mnih & Hinton 2007). Their log-bilinear
  language model predicts the next word through a *bilinear* interaction of real-valued
  distributed representations — a learned multiplicative coupling between linear projections,
  rather than an additive-then-pointwise transform. The lesson carried forward is that a
  product of two learned linear maps of the same input can express feature interactions a
  single projection-then-nonlinearity cannot.
- **Gating in sequence models.** LSTMs (Hochreiter & Schmidhuber 1997) multiply a content
  signal by learned sigmoid gates to control what information survives across many steps; the
  gate is the canonical device for *data-dependent multiplicative modulation*. The diagnostic
  finding that matters here — established on existing gated convolutional language models — is
  about how the *choice of what to put on the content path* affects gradient flow when many
  such layers are stacked. Concretely, for an LSTM-style "gated tanh unit"
  `tanh(X) ⊗ σ(X)`, the gradient is

  ```
  ∇[tanh(X) ⊗ σ(X)] = tanh'(X)∇X ⊗ σ(X) + σ'(X)∇X ⊗ tanh(X),
  ```

  in which both paths carry saturating activation-derivative factors (`0 ≤ tanh' ≤ 1`,
  `0 ≤ σ' ≤ ¼`), with no derivative-free path through the content. The corresponding signal is
  observed to attenuate as layers are stacked.

By 2019, the Transformer is the dominant sequence-modeling architecture and the transfer-
learning recipe of pretraining on a large denoising/span-corruption objective and fine-tuning
on downstream tasks is standard practice; the FFN's pointwise function (ReLU, sometimes GELU
or Swish) is treated as a minor knob and the layer's *shape* — two matrices, one pointwise
map — is taken for granted.

## Baselines

The prior art a new FFN would be measured against and reacts to:

- **ReLU FFN** (Vaswani et al. 2017; bias-free T5 form, Raffel et al. 2019).
  `FFN_ReLU(x) = max(0, xW1)W2`. Two weight matrices `W1 ∈ R^{d×d_ff}`, `W2 ∈ R^{d_ff×d}`,
  `d_ff ≈ 4d`. The hidden vector is one linear projection passed through a hard sign gate.
  **Gap:** the only input-dependence in the hidden activation is the single projection `xW1`;
  the nonlinearity is fixed and hard, with zero output and zero gradient on the entire
  negative half-line, and all hidden-unit modulation is locked to that one scalar
  preactivation.
- **GELU FFN.** `FFN_GELU(x) = GELU(xW1)W2`. Same two-matrix shape; the hard ReLU gate is
  replaced by GELU's smooth value-weighting. **Gap:** smoother and better-behaved at the
  origin, but structurally identical to the ReLU FFN — one projection, one pointwise map; the
  smooth factor `Φ(xW1)` is tied to the same scalar preactivation whose magnitude it weights.
- **Swish FFN.** `FFN_Swish(x) = Swish_1(xW1)W2`. Same shape again; the pointwise map is the
  non-monotonic `x·σ(x)`. **Gap:** identical to the above — a single projection through a
  single fixed (if smoother and non-monotonic) pointwise function, with one learned scalar per
  hidden unit determining both content and modulation.

Across all three baselines, the hidden representation at each unit is `f(one linear map of x)`.
The value-weighting that does exist (GELU's `Φ`, Swish's `σ`) is a fixed function of the same
projection whose output it scales, and the layer has exactly two weight matrices.

## Evaluation settings

The natural yardstick is the standard Transformer transfer-learning protocol that already
exists (Raffel et al. 2019), shared across variants so that only the FFN form varies:

- **Architecture.** Encoder-decoder Transformer, 12 encoder + 12 decoder layers,
  `d_model = 768`, attention `h = 12` heads with `d_k = d_v = 64`, baseline FFN hidden
  `d_ff = 3072` (4×). Any FFN variant must be re-sized to match the baseline's parameter and
  operation counts before being compared.
- **Pretraining.** Span-corruption / denoising objective (predict deleted spans) on the C4
  corpus; 524,288 steps; batch of 128 examples, each ~512 input and ~114 output tokens;
  Adafactor optimizer with an inverse-square-root learning-rate schedule and a linear decay
  over the final 10% of steps; no dropout during pretraining. (A shorter 65,536-step run,
  repeated several times, is used to gauge inter-run variability.)
- **Fine-tuning / downstream.** A fully pretrained model is fine-tuned and evaluated on the
  GLUE and SuperGLUE language-understanding benchmarks and on SQuAD question answering.
- **Metric.** Held-out-shard log-perplexity on the pretraining objective (lower is better) as
  the primary indicator of model quality, plus the downstream task scores. In a decoder-only
  GPT-style instantiation of the same idea, the analogous primary metric is validation
  cross-entropy / perplexity on the held-out language-modeling data.

## Code framework

The design slot is confined to the FFN sublayer. Everything around it already exists: the
embedding, attention, normalization, the residual wiring, the data pipeline, the optimizer and
schedule, and the training/eval loops. The base layer abstraction (a `nn.Module` taking and
returning `(batch, length, d_model)`) and the linear/activation primitives are standard. The
one slot to be filled is the form of the per-position transformation inside the FFN.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineFFN(nn.Module):
    """Existing position-wise FFN: up-project, one pointwise map, down-project.
    `activation` is the only knob (relu / gelu / swish); two weight matrices."""

    def __init__(self, config, activation=F.gelu):
        super().__init__()
        d = config.n_embd
        d_ff = 4 * d                                   # standard 4x expansion
        self.w_in = nn.Linear(d, d_ff, bias=False)
        self.w_out = nn.Linear(d_ff, d, bias=False)
        self.act = activation
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):                              # x: (B, T, d) -> (B, T, d)
        return self.dropout(self.w_out(self.act(self.w_in(x))))


class MLP(nn.Module):
    """The FFN sublayer we are free to redesign. Same input/output contract as the
    baseline: (B, T, n_embd) in, (B, T, n_embd) out, matched parameter and FLOP budget.
    The internal form of the per-position transformation is the open slot."""

    def __init__(self, config):
        super().__init__()
        # TODO: the per-position transformation we will design at matched budget.
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):                              # x: (B, T, n_embd)
        # TODO: produce the hidden representation and project back to n_embd.
        raise NotImplementedError
```

A completed `MLP` fills exactly this slot, keeping the input/output contract and the matched
parameter/FLOP budget.
