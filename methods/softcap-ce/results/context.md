## Research question

Train a GPT-2-style next-token language model on a fixed architecture, dataset, and optimization
budget, and find a loss-layer modification that lowers validation cross-entropy and improves
downstream language ability relative to plain next-token cross-entropy. The only thing that may
change is the function that turns the model's final logits `(B, T, V)` and the targets `(B, T)`
into a scalar training loss — the architecture, tokenizer, data pipeline, optimizer, and learning-rate
schedule are all frozen. So the question is narrow and sharp: holding everything else fixed, is there
a better-behaved objective at the softmax/cross-entropy boundary than the textbook one?

The pressure comes from a structural fact about cross-entropy with hard one-hot targets. The loss for
a single position is `−log softmax(z)_y = −z_y + logsumexp(z)`, where `z ∈ R^V` are the logits and `y`
the target index. Minimizing this drives `z_y` up and the rest down without bound: the infimum is only
approached as `z_y − max_{j≠y} z_j → +∞`. There is no finite minimizer. Left unchecked, the network is
rewarded for ever-larger logit gaps — ever more confident predictions — which is exactly the wrong
incentive for a model that must generalize and must run in low numerical precision. A good modification
would damp this runaway confidence without distorting which token the model prefers, and would do so in
a way that keeps gradients well-behaved throughout the long training run.

## Background

The default objective for autoregressive LM pretraining is per-token cross-entropy over the vocabulary,
computed from the final-layer logits via a softmax. Three facts about this layer set up the problem.

**Cross-entropy has no finite optimum and rewards confidence.** As above, `−z_y + logsumexp(z)` is
decreasing in the gap between the target logit and the rest, with no finite stationary point. On
separable or near-separable data this manifests as logit magnitudes that grow throughout training. The
phenomenon is well known from the calibration literature: deep classifiers trained to convergence on
one-hot targets become badly over-confident, assigning probabilities far higher than their empirical
accuracy warrants. Calibration and numerical stability are related but not identical: a model can have
reasonable probability gaps while still carrying large absolute logits, and the arithmetic only sees the
numbers it is asked to exponentiate.

**Low-precision arithmetic makes large logits actively dangerous.** Modern training runs in mixed
precision: weights in float32, matmuls and activations in bfloat16. bfloat16 keeps 7 mantissa bits
against float32's 23, so within any binade its roundoff error is about `2^16 ≈ 65,536×` larger, and —
because a fixed number of mantissa bits spans each `[2^k, 2^{k+1})` — *larger numbers carry larger
absolute roundoff*. The softmax/cross-entropy path is the worst place for this: it exponentiates. A
small perturbation of a large logit becomes a large multiplicative perturbation of a probability. The
standard illustration: ten logits at 128 and one at 128.5, in bfloat16, where the 0.5 gap rounds away
after the max-subtraction in softmax — the target probability swings from ≈0.142 to ≈0.091, a 36%
change, purely from roundoff. Analogous failures occur in float32 once logits are large enough.
Uncontrolled logit growth and low precision together are a recipe for training instability, and the
larger the model and vocabulary, the more exponentials there are to destabilize.

**Bounding a quantity smoothly vs. hard-clipping it has very different optimization consequences.** A
long line of stabilization work constrains activations or gradients — gradient-norm clipping for exploding
gradients, weight normalization, normalization layers. But a hard activation clamp zeroes the gradient
outside the allowed interval: the very coordinates that are most out of range stop receiving a learning
signal, and the clamp introduces a non-smooth kink. Hard update clipping is different mechanically, but
when made tight it similarly discards part of the optimizer's intended step. Empirically, update clipping
tight enough to stabilize a large sparse-expert model was found to wreck its quality, while a smooth
z-loss penalty on router logits stabilized the model without that quality loss. So if one wants to
constrain logits, the *shape* of the constraint — smooth and everywhere-differentiable vs. hard and
flat-outside — is itself load-bearing.

## Baselines

The prior loss-layer / logit-control methods a new objective is measured against and reacts to.

**Plain next-token cross-entropy (the default).** `L = −(1/N) Σ log softmax(z^{(i)})_{y_i}` over all
non-ignored positions, `z` the final logits, computed in PyTorch as
`F.cross_entropy(logits.view(-1,V), targets.view(-1), ignore_index=-1)`. Core idea: maximum likelihood
of the next token. **Gap:** as established above, it has no finite minimizer and actively pushes logit
magnitudes upward, producing over-confidence and, in low precision, instability — it imposes no control
whatsoever on how large a logit may become.

**Label smoothing (Szegedy, Vanhoucke, Ioffe, Shlens, Wojna 2015).** Replace the hard one-hot target by
a softened target with mass `1−ε` on the correct class and `ε/(V−1)` on each other class (typical
`ε≈0.1`); the loss becomes cross-entropy against this softened distribution. Core idea: stop demanding
probability 1 on the target, so the optimizer no longer chases an infinite logit gap; the implied optimal
gap between the correct logit and each non-target becomes finite, `log((1−ε)(V−1)/ε)`. This improves
calibration and generalization. **Gap:** it acts on the *target*
distribution, not on the logits themselves. It removes the *incentive* for one particular gap to diverge
but places no bound on the absolute scale of the logits, and it does not address the low-precision
roundoff problem, which depends on logit magnitude rather than on the target. It also uniformly
redistributes mass to every vocabulary item including clearly impossible ones.

**Logit / softmax z-loss (Mesh-TensorFlow softmax z-loss; router z-loss, Zoph et al. 2022; final-logit
z-loss popularized in PaLM, Chowdhery et al. 2022).** Add an auxiliary penalty on the log-partition
function: `L_z = (1/B) Σ_i (logsumexp(z^{(i)}))²`, total loss `L_CE + c_z·L_z` with `c_z≈1e-3`–`1e-4`.
Core idea: `logsumexp(z) ≈ max_j z_j` for peaked logits, so squaring and penalizing it pushes the whole
logit vector toward small magnitude, directly targeting the low-precision-roundoff failure (large numbers
into exp). It is used as a smooth stabilizer where hard update-clipping can be too destructive. **Gap:**
it is a *soft, global* penalty added
to the objective, with its own coefficient to tune and its own gradient mixed into every step; it nudges
logits to be small *on average* but enforces no hard bound on any individual logit, and it perturbs the
loss value itself (the reported number is `L_CE + c_z·L_z`, not the modeling loss alone).

**Logit clipping with a tanh activation (Bello, Pham, Le, Norouzi, Bengio 2016).** In a pointer-network
/ attention setting, replace the raw attention logits `u` feeding a softmax by `softmax(C·tanh(u))`,
where `C` "controls the range of the logits and hence the entropy" of the resulting distribution; in
their TSP experiments, clipping logits to `[−10, 10]` this way "helps with exploration and yields
marginal performance gains." Core idea: pass logits through a smooth bounded squashing function before
the softmax so they cannot run away, smoothly rather than by truncation. **Gap:** the form `C·tanh(u)`
has slope `C` at the origin, so the same constant that sets the range also rescales small logits and acts
as a temperature/entropy knob. It was used on attention scores in a small RL model, not as a final
output-logit objective for large-scale LM pretraining.

**Hard logit clamp / hard update clipping (`clamp`; Adafactor update clipping).** Constrain a quantity by
hard truncation: for logits, `torch.clamp(z, −s, s)`; for optimizer updates, cap the update norm. Core
idea: enforce an exact bound. **Gap:** a logit clamp has derivative 0 outside `[−s, s]` and 1 inside, with
a kink at the boundary — out-of-range coordinates receive no gradient through the loss-layer map. Tight
update clipping has a different derivative story, but it was observed to stabilize a large model only at
a catastrophic loss of quality. Both cases make the hardness of the constraint costly, motivating smooth
constraints over hard ones.

## Evaluation settings

- **Model / data / budget (fixed):** GPT-2 Medium (24 layers, 16 heads, `d=1024`, ~355M params),
  trained on FineWeb `sample-10BT` (HuggingFace `HuggingFaceFW/fineweb`), GPT-2 BPE tokenizer,
  ~7.1B training tokens, 13,535 iterations, micro-batch 64, gradient accumulation 8, 2-GPU DDP, mixed
  precision. Cosine learning-rate schedule with linear warmup, AdamW.
- **Primary metric:** validation cross-entropy on held-out FineWeb (lower is better). The reported
  validation loss must be the genuine modeling loss — a method may not lower it by distorting the
  probability distribution (e.g. via a temperature) without actually improving the modeled distribution.
- **Perplexity:** WikiText-2 and LAMBADA (lower is better).
- **Downstream zero/few-shot accuracy:** ARC-Easy, HellaSwag, PIQA, WinoGrande (higher is better).
- **Interface:** the modification must keep the signature `compute_loss(logits, targets)`, with
  `logits` of shape `(B, T, V)` and `targets` of shape `(B, T)`, called inside the model's forward pass,
  and must be stable for the full run.

## Code framework

The substrate is the existing nanoGPT pretraining harness. Everything around the loss is fixed: the
model produces final logits, an outer loop draws batches and steps AdamW, and a single function maps
logits and targets to the scalar that is backpropagated. The pristine version of that function is plain
cross-entropy. The one slot to be filled is the body of `compute_loss` — how the final logits are turned
into the training loss. Nothing about that transformation is settled; that is exactly what is to be
designed.

```python
import torch
import torch.nn.functional as F


# ── Loss Computation ─────────────────────────────────────────────────────────
def compute_loss(logits, targets):
    """Map final-layer logits and targets to the scalar training loss.

    logits : (B, T, V) float tensor of final-layer scores
    targets: (B, T)   int   tensor of next-token ids (ignore_index = -1)

    The objective at this boundary is what we are free to redesign.
    """
    # TODO: the loss-layer transformation we will design — how to turn the
    #       final logits into the scalar we backpropagate.
    pass


# existing training loop (fixed) the loss plugs into
def train_step(model, x, y, optimizer):
    logits = model(x)                 # frozen architecture produces final logits (B, T, V)
    loss = compute_loss(logits, y)    # <-- the only slot we may change
    loss.backward()                   # backprop through the loss into the model
    optimizer.step()                  # fixed AdamW step
    optimizer.zero_grad(set_to_none=True)
    return loss.item()
```

The forward pass hands `compute_loss` the final logits; `compute_loss` is where the redesigned objective
will live, and its output must remain a faithful modeling loss.
