# Context: the output-layer loss for autoregressive language-model pretraining

## Research question

A decoder-only Transformer language model is pretrained by next-token prediction: at every
position the model emits a vector of logits over the vocabulary, and these are turned into a
predictive distribution by a softmax and scored by cross-entropy against the true next token.
The model is large (hundreds of millions of parameters), the vocabulary is large (tens of
thousands of tokens), and — to fit the model and run fast on the accelerator — the forward and
backward passes are done in low precision (bfloat16 activations, with a float32 master copy of
the weights). The single most consequential layer, numerically, is this final softmax: it is
the one place where a very wide vector of real-valued logits is pushed through an exponential.

The question is whether the output-layer loss formulation can be modified to improve numerical
stability under a *fixed* architecture, dataset, optimizer, and compute budget while still
optimizing the same next-token distribution. Any replacement must preserve the
maximum-likelihood scoring rule: a modification that lowers a scalar loss by rescaling logits
or changing the target distribution would no longer be the same training objective.

## Background

The objects in play at this layer.

**The softmax and its partition function.** For a single position with logit vector
`l = (l_1, ..., l_V)` over a vocabulary of size `V`, the predicted probability of token `j` is
`p_j = exp(l_j) / Z`, where `Z = sum_{k=1}^{V} exp(l_k)` is the *partition function* (the
normalizer / denominator of the softmax). Its logarithm, `log Z = log sum_k exp(l_k)`, is the
log-partition function, computed in practice by the numerically-stabilized `logsumexp`
primitive. Cross-entropy against the true next token `y` is the negative log-probability of
that token:

```
CE = -log p_y = -( l_y - log Z ) = log Z - l_y.
```

So cross-entropy decomposes exactly into "raise the true-token logit, lower the
log-partition." Two numerical facts follow directly from this form. The exponential is applied
to raw logits before normalization, so the magnitude of the values entering the softmax matters
to finite-precision arithmetic. And `log Z` is the scalar summary of the wide vocabulary
denominator that every target token's loss shares. Any output-layer stability change has to
respect the maximum-likelihood scoring rule above: it cannot change the target distribution or
replace the softmax with a different normalization and still claim to be the same training
objective.

**Low-precision arithmetic and roundoff.** Modern large-model training runs in mixed precision
(Micikevicius et al. 2017): a float32 master copy of the weights is kept for the update, but
the matrix multiplications and activations are carried in 16-bit floats for speed and memory.
The relevant format here is bfloat16, which keeps float32's 8 exponent bits (so the same huge
dynamic range, up to ~3e38) but truncates the mantissa to 7 bits, versus float32's 23. A
floating-point number stores the same number of mantissa bits across each octave `[2^k,
2^{k+1})`, so the absolute spacing between representable numbers (the unit in the last place)
*grows with magnitude*: ULP ~ `2^k · 2^{-mantissa}`. Two consequences follow directly. First,
for the same number, bfloat16's roundoff is about `2^{23-7} = 65536` times larger than
float32's. Second, *larger numbers carry larger absolute roundoff*. This interacts violently
with the softmax, because the exponential turns a small absolute error in its argument into a
large relative change in its output: `d(exp x) = exp(x)·dx`. A concrete, well-known
illustration: feed a softmax ten logits of value 128 and one of 128.5 in bfloat16; the 0.5 gap
is at the rounding threshold for that magnitude and can round away, and the softmax output for
the distinguished logit collapses from ≈0.142 to ≈0.091, a 36% swing, purely from roundoff.
The lesson is that the
*magnitude* of the numbers entering an exponential is itself a numerical liability when they
are large.

**Observed pathologies of large-LM training.** Three phenomena about existing systems set up
the problem. (1) *Logit / activation drift*: over a long run the magnitudes of the
final-softmax activations can grow too large, exposing `exp` and `logsumexp` to larger absolute
roundoff in bfloat16. (2) *Loss spikes*: large language model runs are repeatedly
interrupted by sudden spikes in the training loss, often preceded by spikes in the gradient
norm, and these become more frequent as the model gets larger; they are costly and degrade the
final model. (3) *Slow growth of the gradient norm* over the course of training, correlated
with the increasing frequency of spikes. These are not abstract worries: practitioners restart
from earlier checkpoints and skip data to escape spikes, an expensive band-aid.

## Baselines

The prior art that a new output-layer objective would be measured against and reacts to.

**Plain next-token cross-entropy.** The default: `CE = log Z - l_y`, averaged over all
non-padding positions, with hard one-hot targets and no modification. It is the correct
maximum-likelihood objective and the thing to beat.

**Label smoothing (Szegedy et al. 2015).** A widely used output-layer regularizer that softens
the hard one-hot target into a mixture with a uniform distribution: the target mass on the
true class becomes `(1-eps) + eps/V` and every other class gets `eps/V` (default `eps ≈ 0.1`).
Equivalently the loss becomes `(1-eps)·H(onehot, p) + eps·H(uniform, p)`. The motivation is to
stop the model from becoming over-confident: with hard targets the optimizer keeps pushing the
true-class logit gap toward infinity, which hurts calibration and generalization; the uniform
floor caps how large that gap wants to be.

**Gradient clipping (Pascanu et al. 2013).** The default stability mechanism for deep-network
training: when the global gradient norm exceeds a threshold, rescale the whole gradient down to
that threshold before the optimizer step. It is universally enabled in large-LM training.

**Mesh-parallelism training framework (Shazeer et al. 2018).** Not a loss, but the substrate:
a framework for expressing data- and model-parallel Transformer training across a mesh of
accelerators, in which the large-vocabulary final softmax and its cross-entropy are computed in
distributed low precision. It is the setting in which the numerical behavior of the partition
function becomes a first-class engineering concern, because the softmax over a 50k+ vocabulary,
in bfloat16, on a sharded device mesh, is exactly where the roundoff-vs-magnitude problem
above bites.

## Evaluation settings

The yardsticks for this layer.

- **Model / data / budget (fixed):** a GPT-2-Medium-class decoder-only Transformer (~355M
  parameters; 24 layers, 16 heads, `d_model = 1024`) trained on a ~7-billion-token sample of a
  web corpus (FineWeb 10BT) with the GPT-2 byte-pair tokenizer, for a fixed number of
  iterations under a fixed optimizer (AdamW) and learning-rate schedule, in bfloat16 mixed
  precision with global-norm gradient clipping at 1.0. Architecture, tokenizer, dataset,
  training loop, and evaluation are all held constant; only the output-layer loss varies.
- **Primary metric:** validation cross-entropy on held-out web text (lower is better), the
  direct measure of the modeled next-token distribution.
- **Perplexity** on standard held-out corpora (WikiText-2, LAMBADA), lower is better.
- **Downstream accuracy** on standard zero-/few-shot benchmarks (ARC-Easy, HellaSwag, PIQA,
  WinoGrande), higher is better.
- **Protocol:** identical seeds, data order, and hyperparameters across loss variants; the
  loss change must be a true drop-in at the `compute_loss` boundary and must keep training
  stable throughout (a run that spikes or diverges is a failure regardless of any momentary
  number).

## Code framework

The only thing that varies is the function that turns the model's output logits and the true
next tokens into a scalar training loss. Everything around it already exists and is fixed: the
model produces logits of shape `(B, T, V)` from its language-model head, the targets are token
ids of shape `(B, T)` with `-1` marking positions to ignore (padding / packed-example
boundaries), and the trainer calls this function inside the forward pass, backpropagates the
returned scalar, clips the global gradient norm, and steps AdamW. The data pipeline, the
Transformer, the optimizer, the schedule, the mixed-precision context, and the evaluation
harness are all in place. The single empty slot is the loss formulation itself.

```python
import torch
import torch.nn.functional as F


def compute_loss(logits, targets):
    """Turn next-token logits into the scalar training loss.

    logits:  (B, T, V) real-valued scores over the vocabulary
    targets: (B, T)    true next-token ids; entries == -1 are ignored

    Called inside the model's forward pass; the returned scalar is
    backpropagated, the global grad-norm is clipped, and AdamW steps.
    Must keep training stable throughout, and must not lower the reported
    loss by distorting the predicted distribution (e.g. via temperature)
    without genuinely improving the modeled distribution.
    """
    flat_logits = logits.view(-1, logits.size(-1))   # (B*T, V)
    flat_targets = targets.view(-1)                   # (B*T,)
    # TODO: the output-layer objective.
    #       Given the per-position logits and true tokens (with -1 ignored),
    #       return the scalar loss to backpropagate.
    pass


# existing trainer the loss plugs into (fixed; shown for context)
def train_step(model, x, y, optimizer):
    logits, loss = model(x, y)            # model calls compute_loss(logits, y) internally
    loss.backward()                       # backprop the scalar
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # reactive stability guard
    optimizer.step()                      # AdamW update
    optimizer.zero_grad(set_to_none=True)
```

The trainer supplies one logit tensor and one target tensor per step; `compute_loss` is the
one place the objective is decided.
