## Research question

A GPT-style language model is trained by next-token prediction: the final layer produces a logit vector
over the vocabulary at every position, a softmax turns it into a distribution, and the training loss is
the cross-entropy of that distribution against the true next token. Everything about the model — the
24-layer GPT-2 Medium architecture, the FineWeb data, the AdamW schedule, the evaluation — is fixed.
The single thing being designed here is **the loss function**: the exact map from the final logits
`(B, T, V)` and the integer targets `(B, T)` to the scalar that gets backpropagated. Can a drop-in
replacement for plain cross-entropy lower held-out validation loss and improve downstream language
ability at **matched architecture, data, and optimization budget**, changing only how logits and
targets are turned into a loss — and nothing else?

## Prior art before the first rung (the objectives the first rung reacts to)

The first method on the ladder reacts to the default objective and the line of loss-layer modifications
that grew up around it. These precede the ladder; each is a different answer to "what is wrong with
plain next-token cross-entropy, and where do you intervene to fix it." Three distinct intervention
points appear, and they matter because each baseline lives at a different one.

- **Plain next-token cross-entropy (the default).** For one position with logits `z` and target `y`,
  the loss is `-log softmax(z)_y = -z_y + log Σ_j exp(z_j)`. This is the maximum-likelihood objective and
  it is correct, but it has no finite minimizer: `softmax(z)_y → 1` only as the gap `z_y - max_{j≠y} z_j`
  runs to `+∞`. On data that is even close to separable at the margin — which token-level LM data
  effectively is — the loss is a standing instruction to grow the logits without bound. Gap: it drives
  logit magnitudes upward across the run, which is over-confidence by construction and, through the
  exponentiated softmax in low precision, a source of roundoff-driven instability.
- **Label smoothing (Szegedy et al. 2016, "Rethinking the Inception Architecture").** Attack the
  *target*: replace the one-hot with `q'(k) = (1-ε)·δ_{k,y} + ε/V`, putting a positive floor under every
  class so an infinite logit gap costs infinite loss. The loss decomposes as
  `(1-ε)·H(q,p) + ε·H(u,p)` — hard cross-entropy plus a pull toward uniform. Gap: it caps the
  *difference* between the true class and the rest (the gauge-invariant part the softmax sees), not the
  absolute logit *level*; and because it fits the model to a softened distribution rather than the data,
  it changes the very objective the validation cross-entropy measures.
- **Softmax z-loss (Mesh-TensorFlow; popularized by ST-MoE, Zoph et al. 2022, and PaLM, Chowdhery et al.
  2022).** Attack the *level* directly: add an auxiliary penalty `λ·(log Σ_j exp(z_j))²` on the squared
  log-partition. Cross-entropy is exactly invariant to a uniform shift of all logits, so the overall
  level is a free gauge it never constrains; the squared-log-`Z` penalty supplies the restoring force the
  loss lacks, with a gradient `2λ·log Z·p_j` proportional to how far the level has drifted. Gap: it nudges
  magnitudes down *on average* via a coefficient mixed into every step, but enforces no hard bound on any
  single logit, and it perturbs the reported loss value rather than transforming the model's forward map.
- **Logit soft-capping (Gemma 2, Gemma Team 2024; the tanh form `s·tanh(z/s)`).** Attack the logits
  *structurally*: pass them through a smooth, bounded, strictly-increasing squash before the softmax, so
  they cannot run away while the token ranking is preserved. Gap as background, not as a finished rung:
  the canonical Gemma form caps at `s=30` symmetrically, but the precise squashing constants are a knob
  for the model size, vocabulary, and precision regime, and the choice of where to put the soft cap (and
  in what parameterization) is exactly what the medium rung on this ladder has to settle.

All four sit at the same boundary — the function that turns final logits into a loss — but they pull on
different handles: the target distribution, the absolute logit level, or the logit values themselves.

## The fixed substrate

A nanoGPT-style GPT-2 Medium pretraining loop is frozen and must not be touched: GPT-2 Medium
(24 layers, 16 heads, `n_embd=1024`, ~355M params), pre-norm bias-free `LayerNorm`, causal self-attention
(flash SDPA), tied input/output embeddings, residual-scaled init
(`c_proj` weights `N(0, 0.02/√(2·n_layer))`), AdamW (`β=(0.9,0.95)`, `weight_decay=0.1`, `grad_clip=1.0`),
a warmup+cosine schedule (`warmup_iters = 0.04·max_iters`, decay to `lr/10`), `torch.compile`, mixed
precision in **bfloat16**, and 2-GPU DDP. Data is FineWeb `sample-10BT` (GPT-2 tokenizer, ~7.1B training
tokens), trained 13,535 iterations at micro-batch 64 with gradient accumulation 8. The model's
`forward` calls `compute_loss(logits, targets)` once per training step, inside the autocast context;
`torch`, `nn`, and `F` (`torch.nn.functional`) are in scope.

## The editable interface

Exactly two regions of `nanoGPT/custom_pretrain.py` are editable, and every method on the ladder is a
fill of the same contract:

1. The **`compute_loss(logits, targets)` function** (the only objective slot). Its signature is fixed;
   `logits` has shape `(B, T, V)` and `targets` shape `(B, T)` with `-1` marking ignored positions; it
   returns a scalar to backpropagate. It may reformulate the loss, process the logits (soft-capping,
   temperature), add regularization terms (z-loss, entropy), or modify the label distribution (label
   smoothing). It must be stable throughout training and must not lower the reported loss by distorting
   the modeling distribution (e.g. via a temperature) without genuinely improving the model.
2. A **`CONFIG_OVERRIDES` dict** that may override training hyperparameters from a fixed whitelist
   (`learning_rate`, `weight_decay`, `warmup_iters`, `min_lr`, `grad_clip`). The baselines leave it
   empty; they change only the loss.

The starting point is the scaffold default — plain next-token cross-entropy with the `-1` ignore index.
Each method replaces exactly the body of `compute_loss` (and, if it chooses, `CONFIG_OVERRIDES`) and
nothing else.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py — default fill (plain next-token cross-entropy)
def compute_loss(logits, targets):
    """Compute language modeling loss from logits and targets."""
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)


# ... later in the training setup, the (default-empty) hyperparameter override hook:
# CONFIG_OVERRIDES: override training hyperparameters for your method.
# Allowed keys: learning_rate, weight_decay, warmup_iters, min_lr, grad_clip.
CONFIG_OVERRIDES = {}
```

## Evaluation settings

One seed (42). Primary metric is **FineWeb validation cross-entropy** (`val_loss`, lower is better),
measured at the end of the fixed 13,535-iteration budget — and computed with *plain* cross-entropy
regardless of the training loss, so methods stay comparable. Secondary metrics, all from the fixed eval
scripts, are word-level perplexity on **WikiText-2** and **LAMBADA** (lower is better) and zero-shot
downstream accuracy on **ARC-Easy**, **HellaSwag**, **PIQA**, and **WinoGrande** (higher is better);
`elapsed` is wall-clock seconds. Because the substrate is frozen and the budget fixed, any change in
`val_loss` is attributable to the *form* of the loss function alone.
