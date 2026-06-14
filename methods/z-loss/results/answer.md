# Z-loss, distilled

Z-loss is an auxiliary penalty added to next-token cross-entropy that squares the
log-partition function of the final softmax: `loss = CE + lambda·(log Z)^2`, where
`Z = sum_k exp(l_k)` is the softmax normalizer and `log Z = logsumexp(logits)`. It puts a
smooth restoring force on the overall magnitude of the logits — a quantity cross-entropy is
blind to — keeping the numbers entering the exponential small and the low-precision softmax
numerically faithful while leaving the next-token target distribution unchanged.

## Problem it solves

Cross-entropy `CE = log Z - l_y` is invariant to adding a constant to all logits
(`softmax(l + c·1) = softmax(l)`), so the *overall logit level* — equivalently `log Z` — is a
free gauge the objective never constrains. Left unconstrained it drifts upward over a long run.
In mixed-precision (bfloat16) training, large logits are exactly where the exponential becomes
unfaithful: bfloat16 has 7 mantissa bits vs float32's 23, so its absolute roundoff is ~`2^16 =
65536`× coarser and grows with magnitude, and `d(exp x) = exp(x)·dx` turns that absolute
argument error into a relative softmax error. The result is the well-known pattern of slowly
growing gradient norms tipping into sudden loss spikes that gradient clipping (a reactive,
downstream guard) does not prevent. Z-loss supplies the missing restoring force on the gauge.

## Key idea

Add a penalty that is minimized when `log Z = 0` (i.e. `Z = 1`) and grows as the logit level
drifts:

- **Why `(log Z)^2`** and not `|log Z|` or `log Z`: the penalty must be symmetric around the
  normalized-log-probability representative and have a minimum, which rules out the signed
  `log Z` (no minimum; pushes `log Z -> -inf`). Among symmetric choices, `(log Z)^2` is smooth
  at the minimum and convex, and its gradient `2·log Z` is a restoring force whose strength is
  *proportional to the violation* — gentle near 0, stronger far out — whereas `|log Z|` has a
  kink at 0 and a constant-strength pull.
- **Gradient.** With `d(log Z)/d l_j = exp(l_j)/Z = p_j`, the augmented per-position loss
  `L = (log Z - l_y) + lambda·(log Z)^2` has gradient
  `dL/d l_j = (p_j - 1{j=y}) + 2·lambda·log Z·p_j`. The first term is the usual
  predicted-minus-one-hot; the second is a probability-weighted pull on the logits that
  contribute to the partition, shrinking `log Z` toward 0 when it has drifted up. Cross-entropy
  remains the main force on the logit *gaps*; z-loss adds the missing force on the *level*.
- **It bounds the largest logit.** `max_k l_k <= log Z <= max_k l_k + log V`, equivalently
  `log Z - log V <= max_k l_k <= log Z`. Holding `log Z` near 0 keeps the largest logit from
  becoming large and positive, back in the regime where the bfloat16 exponential is faithful.
- **Bonus:** `log Z = 0 <=> sum_k exp(l_k) = 1`, so the same penalty also encourages the raw
  logits to be honest *normalized* log-probabilities, not log-probs plus an arbitrary constant.

## Why this lever, not the alternatives

- **Label smoothing** changes the *target distribution* (it fits the model to a softened
  target, trading true log-likelihood for calibration) and acts on logit *gaps* — the
  gauge-invariant part — so it does not address the free level at all. Z-loss is its
  complement: it touches the level while leaving the target distribution and CE term intact.
- **Logit clipping** clamps values *after* they are already corrupted by roundoff, and a hard
  clip is itself a discontinuity. Z-loss is smooth and acts *during* optimization to discourage
  the model from producing large logits in the first place.
- **Gradient-norm clipping** is reactive and downstream of the bad step; z-loss addresses the
  cause.

## Default coefficient

`lambda = 1e-4` for the large-vocabulary final softmax. The canonical softmax-loss
implementation gives `1e-4` as the example coefficient, and large-scale language-model training
uses the exact `10^-4 * log^2 Z` auxiliary term. The scale is deliberately small: CE per token
is `O(1)` nats and a healthy `(log Z)^2` is `O(1)`, so `1e-4` keeps the penalty far below the
primary maximum-likelihood term while its gradient `2·lambda·log Z` still supplies a persistent
restoring force on the logit level.

## Final form

```
per position with logits l and true token y:
    Z      = sum_k exp(l_k)                 # softmax partition function
    log Z  = logsumexp(l)                   # stabilized: max_k l_k + log sum_k exp(l_k - max)
    CE     = log Z - l_y                    # plain cross-entropy
    L      = CE + lambda * (log Z)^2        # + gauge-fixing restoring force

reduction: average CE and (log Z)^2 over the same valid positions (target != -1)
default:   lambda = 1e-4
```

## Working code

Fills the `compute_loss(logits, targets)` slot of the pretraining harness. `logits` is
`(B, T, V)`, `targets` is `(B, T)` with `-1` marking ignored positions. The code mirrors the
canonical structure: compute ordinary cross-entropy, compute `log_z = logsumexp(logits)`, square
that same per-position log-partition, average it over the valid positions, and add
`1e-4 * z_loss`.

```python
import torch
import torch.nn.functional as F


def compute_loss(logits, targets):
    """Cross-entropy with z-loss regularization.

    z-loss penalizes the squared log-partition (log Z)^2 of the final softmax,
    putting a smooth restoring force on the otherwise-free overall logit level
    so the logits stay small and the low-precision softmax stays numerically
    faithful. lambda = 1e-4 makes it a regularizer, not a second objective.
    """
    flat_logits = logits.view(-1, logits.size(-1))      # (B*T, V)
    flat_targets = targets.view(-1)                      # (B*T,)

    # Primary maximum-likelihood objective: CE = log Z - l_y,
    # averaged over valid positions (ignore_index=-1).
    ce_loss = F.cross_entropy(flat_logits, flat_targets, ignore_index=-1)

    # Auxiliary softmax z-loss:
    # add lambda * square(log_z), where log_z is the softmax log-partition.
    log_z = torch.logsumexp(flat_logits, dim=-1)         # log Z per position
    mask = flat_targets != -1                            # same valid positions as CE
    z_loss = (log_z[mask] ** 2).mean()                   # mean over valid positions

    return ce_loss + 1e-4 * z_loss
```

Equivalent per-position form: compute `log_z = logsumexp(logits)`, gather the true-token logits
only for valid `target != -1` positions, and return
`mean((log_z - l_y) + 1e-4 * log_z**2)` over those valid positions.
