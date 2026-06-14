# Logit soft-capping, distilled

Logit soft-capping replaces the raw logits feeding a softmax/cross-entropy with a smooth,
strictly-increasing, bounded squashing of each logit: `z ← soft_cap · tanh(z / soft_cap)`. The
transformation is the identity on normal-sized logits and a smooth asymptotic ceiling at `±soft_cap`
on large ones, so logits cannot run away, the exponentiated softmax stays numerically safe in low
precision, the token ranking (argmax) is preserved, and the gradient is nonzero for every finite input —
unlike a hard `clamp`. For the language-model loss it is applied to the final logits, which are then fed to plain
cross-entropy; the loss remains an honest negative log-likelihood of the (now better-behaved) output
head.

## Problem it solves

Cross-entropy with one-hot targets, `−z_y + logsumexp(z)`, has no finite minimizer: it decreases
monotonically as the gap `z_y − max_{j≠y} z_j → ∞`, so training drives logit magnitudes upward
indefinitely. This causes (1) over-confidence / poor calibration, and (2) in mixed-precision training,
instability — bfloat16 has ~`2^16×` larger roundoff than float32 and larger numbers carry larger
absolute roundoff, so large logits fed into the softmax's exponential turn small roundoff into large
multiplicative probability errors. Soft-capping bounds the logits without distorting their order or
freezing gradients.

## Key idea

Pass logits through a scaled tanh before the softmax:

```
z_tilde = soft_cap * tanh(z / soft_cap)
```

- **Bounded:** `z_tilde ∈ (−soft_cap, +soft_cap)` for all real `z`, so the exp inputs are bounded.
- **Identity on the bulk:** the `/soft_cap` inside makes the slope at the origin exactly 1
  (`d/dz = 1 − tanh²(z/soft_cap)`), so small logits pass through almost unchanged and only large ones
  bend — this is the difference from `C·tanh(z)` (slope `C` at 0), which is an entropy/temperature knob
  rather than an identity-plus-ceiling.
- **Smooth, live gradient:** the derivative `1 − (z_tilde/soft_cap)²` is 1 in the middle and tapers to
  (but never equals for finite `z`) 0 at the tails — out-of-range finite logits keep a task-gradient
  path, with no dead zone and no kink (contrast `clamp`, derivative `0` outside `[−s, s]`).
- **Order-preserving:** strictly increasing, so the argmax over the vocabulary is unchanged — the
  model's preferred token is preserved; only the scale is tamed.

Caps used in the canonical setting: `50.0` for attention pre-softmax logits, `30.0` for the final-layer
logits. For an LM training loss only the final logits exist in `compute_loss`, so the final-layer cap
(~30) is the relevant one.

## Sigmoid reparameterization (used in practice)

Because `tanh(u) = 2·σ(2u) − 1` (with `σ` the logistic sigmoid), the tanh cap is exactly an affine
function of a sigmoid:

```
15 * tanh(z / 15)  =  30 * sigmoid(z / 7.5) − 15
```

(verified: these two expressions both map R → (−15, 15) and agree pointwise). The additive constant
`−15` is invisible to the softmax (softmax is invariant to adding a constant to all logits), so it can be
dropped, leaving `30·σ(z/7.5)` — shifted by `+15` pointwise, but equivalent as a softmax input. The
general bounded-monotone form is `A·σ((z + B)/C)`: `A` sets the cap height (range `(0, A)`), `C` the
saturation scale, and `B` places the sigmoid's midpoint at `z = −B`. A re-tuned setting for short
nanoGPT-scale pretraining is
`A, B, C = 23.0, 5.0, 7.5`, i.e. `23·σ((z + 5)/7.5)` — strictly increasing into `(0, 23)`, smooth, and
order-preserving. This tuned curve is not pointwise identical to `15·tanh(z/15)`; it is the same
bounded-sigmoid family with a different height and center. The sigmoid form is one cheap op and folds
cleanly into a fused cap-plus-cross-entropy kernel without materializing a separate capped-logit tensor.

Gradient (for the fused backward): with `z_tilde = A·σ(u)`, `u = (z + B)/C`,

```
d z_tilde / d z = (1/C) · A · σ(u)(1 − σ(u)) = (1/C) · z_tilde · (1 − σ(u))
```

a bell peaked at the midpoint and vanishing in both tails. Composed with the usual cross-entropy
gradient `p_j − 1[j = y]`, this gives the cap-aware logit gradient directly.

## Why not the alternatives

- **Plain cross-entropy:** no finite minimizer; drives logits to grow → over-confidence + low-precision
  instability. No control on magnitude.
- **Label smoothing** with target mass `1−ε` and non-target mass `ε/(V−1)`: softens the *target*, making
  the optimal gap finite
  (`log((1−ε)(V−1)/ε)`), but places no bound on absolute logit magnitude, doesn't address roundoff, and
  optimizes a different distribution than the evaluated one.
- **z-loss** `c_z·(logsumexp z)²`: a soft global penalty that pushes magnitudes down on average, but
  adds a coefficient, mixes a force into every step, enforces no per-logit bound, and changes the
  reported loss value.
- **Hard clamp** `clamp(z, −s, s)`: bounds the range but zeroes the gradient of every out-of-range logit
  (dead coordinates) and adds a kink at `±s`; related hard update clipping can stabilize while damaging
  quality.

The useful property is a structural per-logit bound that is smooth, order-preserving, and keeps
the loss a faithful modeling cross-entropy (a nonlinear cap with saturating tails — not a uniform
temperature rescale, which would lower the loss without improving the model).

## Working code

The loss-layer slot, filled (sigmoid soft cap on the final logits, then cross-entropy):

```python
import torch
import torch.nn.functional as F


def compute_loss(logits, targets):
    """Cross-entropy with a smooth soft cap on the final logits.

    logits : (B, T, V); targets : (B, T) with ignore_index = -1.
    Each logit is mapped through a strictly-increasing bounded squash before the
    softmax, so logits cannot run away (bounding the low-precision exp) and the
    token ranking is preserved, while the gradient stays smooth and nonzero.
    """
    # Tuned sigmoid soft cap. Exact identity for the symmetric 15-cap:
    # 15*tanh(z/15) = 30*sigmoid(z/7.5) - 15, and the -15 shift is
    # invisible to softmax. A, B, C below are a retune, not a pointwise-
    # identical tanh curve.
    A, B, C = 23.0, 5.0, 7.5
    # cast to float32 before the sigmoid so the saturating exp runs in high precision
    capped_logits = A * torch.sigmoid((logits.float() + B) / C)
    return F.cross_entropy(
        capped_logits.view(-1, capped_logits.size(-1)),
        targets.view(-1),
        ignore_index=-1,
    )
```

Equivalent symmetric tanh form (the version used when also capping attention scores):

```python
import torch
import torch.nn.functional as F


def softcap(logits, cap):
    # logits <- cap * tanh(logits / cap): slope 1 at 0, smooth ceiling at +/-cap
    return cap * torch.tanh(logits / cap)


def compute_loss_tanh(logits, targets, final_cap=30.0):
    z = softcap(logits.float(), final_cap)
    return F.cross_entropy(z.view(-1, z.size(-1)), targets.view(-1), ignore_index=-1)
```

Fused cap-plus-cross-entropy (sketch): for each row, accumulate `lse = logsumexp_j A·σ((z_j+B)/C)`
online, the per-target loss is `lse − A·σ((z_y+B)/C)`, and the backward uses
`grad_z = (p_j − 1[j=y]) · (1/C)·z_tilde·(1 − σ(u))` — avoiding any materialized capped-logit tensor.
