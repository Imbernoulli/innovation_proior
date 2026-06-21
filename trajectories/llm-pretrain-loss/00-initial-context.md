## Research question

A GPT-style language model is trained by next-token prediction: at each position the final layer outputs logits `(B, T, V)`, a softmax turns them into a distribution, and the training loss is the cross-entropy against the true next token. Architecture (GPT-2 Medium), data (FineWeb), optimizer, and evaluation are fixed. The only design choice is **the loss function** — the exact map from final logits and integer targets to the scalar that is backpropagated. Can a drop-in replacement for plain cross-entropy lower held-out validation loss and improve downstream ability at matched architecture, data, and optimization budget, changing only how logits and targets become a loss?

## Prior art / Background / Baselines

The default objective and several loss-layer alternatives define the landscape.

- **Plain next-token cross-entropy.** Map each position to `-log softmax(z)_y`; standard maximum-likelihood. Limitation: it has no finite minimizer, so on near-separable token data it continuously grows the correct logit and the overall logit scale, producing over-confident probabilities and numerical strain.
- **Label smoothing.** Mix the one-hot target with a uniform distribution over the vocabulary. Limitation: it changes the objective from fitting the observed labels to fitting a softened distribution, and it caps relative class gaps without bounding the absolute logit level.
- **Softmax z-loss.** Add a squared penalty on `log Σ_j exp(z_j)`. Limitation: it pulls the global logit level down only through a scalar coefficient, leaves individual logits unbounded, and introduces an auxiliary term not present in the validation cross-entropy.
- **Logit soft-capping.** Squash logits through a bounded monotonic function before softmax. Limitation: the bounded saturation changes the distribution shape and flattens gradients for large logits, and the cap threshold is a fixed design constant whose cost/benefit trade-off is regime-dependent.

## Fixed substrate / Code framework

The pretraining loop is frozen. Substrate: GPT-2 Medium (24 layers, 16 heads, `n_embd=1024`, ~355M params), pre-norm bias-free LayerNorm, causal flash-attention, tied input/output embeddings, residual-scaled init (`c_proj` weights `N(0, 0.02/√(2·n_layer))`), AdamW (`β=(0.9, 0.95)`, `weight_decay=0.1`, `grad_clip=1.0`), warmup+cosine schedule (`warmup_iters = 0.04·max_iters`, decay to `lr/10`), `torch.compile`, bfloat16 mixed precision, 2-GPU DDP. Data: FineWeb `sample-10BT` (GPT-2 tokenizer, ~7.1B tokens), 13,535 iterations, micro-batch 64, gradient accumulation 8. The model's `forward` calls `compute_loss(logits, targets)` once per training step inside autocast; `torch`, `nn`, and `F` are in scope.

## Editable interface

Only two regions of `nanoGPT/custom_pretrain.py` are editable:

1. The **`compute_loss(logits, targets)` function**. Signature fixed; `logits` shape `(B, T, V)`, `targets` shape `(B, T)` with `-1` for ignored positions; returns a scalar. It may reformulate the loss, process logits, add regularization, or modify the label distribution. It must be stable throughout training and must not lower reported loss by distorting the modeling distribution.
2. A **`CONFIG_OVERRIDES` dict** for a whitelist of training hyperparameters (`learning_rate`, `weight_decay`, `warmup_iters`, `min_lr`, `grad_clip`). Baselines leave it empty.

The starting point is plain next-token cross-entropy with the `-1` ignore index. Each method replaces only the body of `compute_loss` and optionally `CONFIG_OVERRIDES`.

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

One seed (42). Primary metric: FineWeb validation cross-entropy (`val_loss`, lower is better) at the end of 13,535 iterations, computed with plain cross-entropy regardless of the training loss. Secondary metrics from fixed eval scripts: word-level perplexity on WikiText-2 and LAMBADA (lower is better), and zero-shot accuracy on ARC-Easy, HellaSwag, PIQA, and WinoGrande (higher is better); `elapsed` is wall-clock seconds. Because the substrate and budget are fixed, changes in `val_loss` are attributable to the loss function's form alone.
