## Research question

A Transformer block alternates two sublayers: multi-head self-attention, which mixes
information *across* sequence positions, and a position-wise feed-forward network (FFN),
which transforms each position's hidden vector *on its own*, identically at every position.
The FFN is where the bulk of a Transformer's parameters and a large share of its compute
live — with the usual 4× expansion the two FFN matrices account for roughly two-thirds of the
non-embedding parameters — yet it has the simplest possible form: one linear projection up to
a wide hidden dimension, one pointwise nonlinearity, one linear projection back down:

```
FFN(x) = f(x W1 + b1) W2 + b2,        d_ff = expansion-factor × d_model  (typically 4×)
```

The pointwise function `f` is the *only* nonlinearity in the whole sublayer, and the hidden
activation at every unit is just `f` applied to a *single* learned linear view `x W1` of the
input. The question is whether changing how the hidden representation is formed inside this
sublayer — rather than widening it — can improve the layer's fit to the language-modeling
objective.

## Background

The pointwise nonlinearity `f` has its own lineage. Early units made a hard sign decision;
the sigmoid smoothed that into a differentiable "firing rate"; the rectifier
`ReLU(x) = max(x, 0) = x · 1[x>0]` (Nair & Hinton 2010; Glorot, Bordes & Bengio 2011) became
the default because the gradient flows undiminished on the positive half-line (its derivative
there is exactly 1), it is cheap, and it induces sparsity. ReLU makes a *hard sign gate*: it
multiplies the input by 0 or 1 depending only on `sign(x)`, zeroing both the output and the
gradient on the entire negative half-line. Two smoother "weight by value rather than gate by
sign" alternatives are available:

- **GELU** (Hendrycks & Gimpel 2016). `GELU(x) = x · Φ(x)`, where `Φ` is the standard-normal
  CDF, `Φ(x) = P(X ≤ x), X ~ N(0,1)`. It is derived as the *expectation* of a stochastic 0/1
  mask: take `m ~ Bernoulli(Φ(x))` (the keep-probability rising with `x`, motivated by
  preactivations being approximately normal, an effect strengthened by normalization layers);
  its expected transform is `Φ(x)·x + (1−Φ(x))·0 = xΦ(x)`. So GELU multiplies `x` by a smooth,
  input-dependent value in roughly `(0,1)` instead of ReLU's hard `1[x>0]`. Computed exactly as
  `x·½[1 + erf(x/√2)]`, or approximated cheaply by `0.5x(1 + tanh[√(2/π)(x + 0.044715 x³)])` or
  by `x·σ(1.702 x)`. The same construction with the *logistic* CDF `σ` in place of the Gaussian
  CDF gives `x·σ(x)`, which the same work names the **Sigmoid Linear Unit (SiLU)**.
- **Swish** (Ramachandran, Zoph & Le 2017). `Swish_β(x) = x · σ(βx)`, with `σ(z)=(1+e^{−z})^{−1}`
  and `β` a constant (or a trainable per-channel scalar). The `β=1` case `x·σ(x)` coincides
  with the SiLU of Elfwing, Uchibe & Doya (2017), proposed independently for reinforcement-
  learning function approximation as a "continuous, undershooting" ReLU. Swish was *found by an
  automated search* over compositions of unary/binary primitives (an RNN controller trained
  with policy gradient, plus exhaustive search for small spaces): each candidate is built from
  a "core unit" `b(u₁(x), u₂(x))` and scored by a child network's validation accuracy. Two
  findings from that search stand out. First, the best-performing functions
  overwhelmingly share the structure `b(x, g(x))` — the *raw* preactivation `x` recombined with
  some gate `g(x)` of itself (ReLU itself fits this, with `b=max` and `g(x)=0`). Second,
  *functions that use division performed poorly, because the output explodes when the
  denominator is near zero*; division was only successful when the denominator stays bounded
  away from zero. Swish's measured properties: smooth, unbounded above, bounded
  below, and — unusually — **non-monotonic**, with a small "bump" below zero (a large fraction
  of trained preactivations land in that `−5 ≤ x ≤ 0` bump region). Its derivative is
  `Swish_β'(x) = σ(βx) + βx·σ(βx)(1−σ(βx)) = β·Swish_β(x) + σ(βx)(1 − β·Swish_β(x))`, whose
  magnitude is below 1 for inputs below roughly 1.25 at `β=1` — so the *exact-1 derivative*
  that made ReLU's gradient flow well is not, by itself, the distinguishing advantage it once
  appeared to be. `β→∞` recovers ReLU and `β=0` gives the scaled linear map `x/2`, so Swish
  interpolates smoothly between linear and ReLU. With `β=1`, Swish and GELU trace nearly the
  same curve (the sigmoid approximation of GELU, `x·σ(1.702 x)`, is a Swish with `β≈1.702`).

A second, separate line of work combines two linear views of an input *multiplicatively*
rather than passing one through a fixed pointwise map:

- **Multiplicative / bilinear interactions** (Mnih & Hinton 2007). A log-bilinear language
  model predicts the next word through a *bilinear* coupling of real-valued distributed
  representations — a learned multiplicative interaction between linear projections, rather
  than an additive-then-pointwise transform. A *product* of two learned linear maps of the
  same input can express feature interactions that a single projection-then-nonlinearity cannot.
- **Gating in sequence models.** LSTMs (Hochreiter & Schmidhuber 1997) multiply a content
  signal by learned sigmoid gates to control what survives across many steps; the gate is the
  canonical device for *data-dependent multiplicative modulation*. Work on gated convolutional
  language models analyzes how the *choice of what to carry on the content path* affects
  gradient flow when many such layers are stacked. For the "gated tanh unit" `tanh(X) ⊗ σ(X)`,
  the gradient is

  ```
  ∇[tanh(X) ⊗ σ(X)] = tanh'(X)∇X ⊗ σ(X) + σ'(X)∇X ⊗ tanh(X),
  ```

  in which both paths carry a saturating activation-derivative factor (`0 ≤ tanh' ≤ 1`,
  `0 ≤ σ' ≤ ¼`), and the gradient signal is observed to attenuate as layers are stacked.

By 2019 the Transformer is the dominant sequence-modeling architecture, and the recipe of
pretraining on a large denoising / span-corruption objective and fine-tuning downstream is
standard. The FFN's pointwise function (ReLU, sometimes GELU or Swish) is treated as a minor
knob, and the layer's *shape* — two matrices, one pointwise map — is taken for granted.

## Baselines

The prior FFN variants that any new design would be measured against:

- **ReLU FFN** (Vaswani et al. 2017; bias-free T5 form, Raffel et al. 2019).
  `FFN_ReLU(x) = max(0, xW1)W2`. Two weight matrices `W1 ∈ R^{d×d_ff}`, `W2 ∈ R^{d_ff×d}`,
  `d_ff ≈ 4d`. The hidden vector is one linear projection passed through a hard sign gate.
- **GELU FFN.** `FFN_GELU(x) = GELU(xW1)W2`. Same two-matrix shape; the hard ReLU gate is
  replaced by GELU's smooth value-weighting, `Φ(xW1)` tied to the same scalar preactivation
  whose magnitude it weights.
- **Swish FFN.** `FFN_Swish(x) = Swish_1(xW1)W2`. Same shape again; the pointwise map is the
  smooth, non-monotonic `x·σ(x)`.

Across all three baselines the hidden representation at each unit is `f(one linear map of x)`.
Whatever value-weighting exists (GELU's `Φ`, Swish's `σ`) is a *fixed* function of the very
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
  over the final 10% of steps; pretraining run without dropout. (A shorter 65,536-step run,
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
schedule, and the training/eval loops. The base layer abstraction (an `nn.Module` taking and
returning `(batch, length, d_model)`) and the linear/activation primitives are standard. The
one slot to fill is the form of the per-position transformation inside the FFN.

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
