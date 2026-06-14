**Problem.** Plain next-token cross-entropy `-z_y + log Σ_j exp(z_j)` has no finite minimizer: it is
minimized only as the gap `z_y - max_{j≠y} z_j → +∞`. On near-separable token data it is a standing
instruction to grow the correct logit's lead without bound — over-confidence by construction, which
overfits and makes the model rigid. The only editable slot is `compute_loss(logits, targets)`; the
goal is to lower FineWeb `val_loss` and improve downstream ability at matched architecture, data, and
optimizer, by changing only the objective.

**Key idea.** Attack the *target*: stop demanding probability one. Bleed a fraction `ε` of the target
mass onto a uniform prior, `q'(k) = (1-ε)·δ_{k,y} + ε/V`. Every token now has a positive floor `ε/V`,
so an infinite logit gap costs infinite cross-entropy and the correct logit can no longer run away. The
loss decomposes as `H(q',p) = (1-ε)·H(q,p) + ε·H(u,p)` — hard cross-entropy plus an `ε`-weighted pull
toward uniform (`H(u,p) = D_KL(u‖p) + const`).

**Why (and the two task-local choices).** (1) *Training-only.* Smoothing fits the model to a softened
distribution, so it is a worse density estimator under the true one-hot likelihood — and the metric here
*is* that likelihood. Applying smoothing under `torch.is_grad_enabled()` only, and falling back to plain
cross-entropy in the no-grad eval pass, keeps `val_loss` an honest, comparable number and stays inside
the contract (no distorting the reported distribution). (2) *ε = 0.05, not 0.1.* This is a short run
(13,535 iters) on a ~50k vocabulary; the heavy `0.1` regularizer mostly biases training off the data
without the long-run overfitting it would otherwise prevent, so a lighter floor is the task-local
setting. Smoothing caps the logit *gap* (the gauge-invariant difference the softmax sees) but leaves the
absolute logit *level* — and the bfloat16 numerics that depend on it — untouched; that is the gap the
later rungs attack.

**Hyperparameters.** `smoothing = 0.05` when training, `0.0` at eval; `ignore_index = -1` for packed
boundaries; computed via the library `label_smoothing` argument over the flattened `(B·T, V)` logits.
`CONFIG_OVERRIDES` left empty — only the loss changes.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py — step 1: label smoothing (eps=0.05, training only)
def compute_loss(logits, targets):
    """Cross-entropy with label smoothing (eps=0.05) during training only.

    Label smoothing is applied only when gradients are enabled (training).
    During evaluation (@torch.no_grad()), standard cross-entropy is used
    so that val_loss remains comparable across methods.
    """
    smoothing = 0.05 if torch.is_grad_enabled() else 0.0
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), targets.view(-1),
        ignore_index=-1, label_smoothing=smoothing
    )


# training-setup hook left at the default — only the loss changed:
CONFIG_OVERRIDES = {}
```
