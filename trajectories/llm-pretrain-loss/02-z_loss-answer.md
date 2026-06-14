**Problem (from step 1).** Label smoothing landed *behind* plain cross-entropy (`val_loss` 2.3377)
because it pulls on the logit *gap* — the gauge-invariant difference the softmax sees — and trains
against a softened target while the metric is true likelihood. The genuinely free quantity it cannot
reach is the absolute logit *level*: cross-entropy is exactly invariant to a uniform shift `l → l + c`,
so the level has no restoring force, drifts upward over the run, and in bfloat16 large logits make the
softmax exponential unfaithful (absolute roundoff grows with magnitude → relative error through `exp`),
which is the spike/gradient-growth failure mode.

**Key idea.** Attack the *level*. Add an auxiliary penalty on the squared log-partition,
`L = CE + λ·(log Z)²` with `log Z = log Σ_j exp(l_j)`. The penalty is symmetric, smooth, and minimized
at `log Z = 0` (i.e. `Σ exp = 1`, the normalized-log-prob slice); `s²` is chosen over `|s|`/signed `s`
because its gradient `2s` is a *proportional* restoring force — gentle near 0, stronger on a runaway. It
changes neither the target nor (at small `λ`) the probability gaps; it only supplies the constraint
cross-entropy structurally lacks.

**Why it works.** The augmented gradient `dL/d l_j = (p_j - 1[j=y]) + 2λ·log Z·p_j` adds a
probability-weighted pull that, when `log Z > 0`, pushes down hardest on the logits contributing most to
`Z`, shrinking the level back toward 0. The log-sum-exp sandwich `max_k l_k ≤ log Z ≤ max_k l_k + log V`
guarantees that holding `log Z` near 0 caps the top logits (≤ ~11 nats spread for V≈50k), keeping the
bfloat16 exp in its faithful regime. Pinning `log Z = 0` also makes the raw logits behave like honest
normalized log-probabilities. Unlike smoothing it needs no train/eval split — it does not distort the
target, so it stays on through evaluation and `val_loss` is unaffected by the penalty term itself.

**Hyperparameters.** `λ = 1e-4` (canonical large-vocabulary coefficient): contributes ~`1e-4` against a
CE of order 1 — invisible to the modeling objective, persistent on the gauge. `log Z` via the
max-subtracting stabilized `torch.logsumexp`; the penalty masked to `targets != -1` and **mean-reduced
over exactly the same valid positions cross-entropy uses**, so the effective `λ` is the clean ratio.
`CONFIG_OVERRIDES` left empty — only the loss changes.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py — step 2: cross-entropy + softmax z-loss (lambda=1e-4)
def compute_loss(logits, targets):
    """Cross-entropy with z-loss regularization."""
    flat_logits = logits.view(-1, logits.size(-1))
    flat_targets = targets.view(-1)
    ce_loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1)
    # Z-loss: penalize large log-partition values
    log_z = torch.logsumexp(flat_logits, dim=-1)
    # Only compute z-loss for non-ignored positions
    mask = flat_targets != -1
    z_loss = (log_z[mask] ** 2).mean()
    return ce_loss + 1e-4 * z_loss


# training-setup hook left at the default — only the loss changed:
CONFIG_OVERRIDES = {}
```
