# Generalized Cross Entropy (L_q loss), distilled

Generalized Cross Entropy (GCE), the `L_q` loss, is a drop-in classification objective for training
deep networks under label noise. It interpolates between categorical cross entropy (CCE) and mean
absolute error (MAE) with a single exponent `q`, inheriting CCE's fast learning dynamics and MAE's
provable noise tolerance. It changes only the loss — no architecture, optimizer, data-pipeline,
confusion-matrix, clean-validation-set, or auxiliary-network changes.

## Problem it solves

Training a softmax DNN when a fraction of the training labels are corrupted (e.g. flipped to a wrong
class). High-capacity networks memorize wrong labels, so clean-test accuracy degrades. We want
robustness purely from the per-example loss.

## Key idea

The per-sample gradient weight is the whole story. For a softmax output with `f_y` the probability on
the given label:

- CCE `= -log f_y` has gradient `-(1/f_y) ∇_θ f_y` — weight `f_y^{-1}`, which *over-weights*
  low-confidence (likely-noisy) samples, causing memorization of corrupted labels.
- MAE `= 2 - 2 f_y` (= `1 - f_y` up to a constant, the unhinged loss) has gradient `-∇_θ f_y` —
  weight `f_y^{0} = 1`, flat, which is noise-robust (it is **symmetric**, hence noise-tolerant) but
  gives no extra pull on hard examples, so deep training stalls on hard datasets.

The two weights `f_y^{-1}` and `f_y^{0}` are endpoints of one exponent. Let the per-sample gradient
weight be `f_y^{q-1}`; integrating `dL/dp = -p^{q-1}` with `L(1) = 0` gives the **negative Box-Cox
transformation** as the loss:

```
L_q(f(x), e_j) = (1 - f_j(x)^q) / q,    q ∈ (0, 1].
```

- `q → 0`: by L'Hôpital, `(1 - f^q)/q → (-f^q log f)|_{q=0} = -log f` = CCE.
- `q = 1`: `(1 - f)/1 = 1 - f` = MAE / unhinged loss.

Its gradient is `∂L_q/∂θ = - f_y^{q-1} ∇_θ f_y = f_y^q · ( -(1/f_y) ∇_θ f_y )`: cross entropy with
each sample's gradient scaled by `f_y^q ∈ [0,1]`, which **down-weights low-confidence (likely-noisy)
samples** (robustness vs CCE) while the `f_y^{q-1} > 1` reading shows it **up-weights hard samples
relative to MAE** (learnability). Larger `q` → more robust, harder to optimize; smaller `q` → easier,
less robust.

## Noise-tolerance guarantee

A loss is **symmetric** if `sum_{j=1}^c L(f(x), j) = C` for all `x, f`; symmetric losses are
noise-tolerant under uniform noise for `η < (c-1)/c` (Ghosh et al.), because the noisy risk is an
affine increasing transform of the clean risk, `R^η_L(f) = Cη/(c-1) + (1 - ηc/(c-1)) R_L(f)`. `L_q`
is only approximately symmetric; bounding its class-sum (for `q ∈ (0,1]`, softmax `f`):

```
(c - c^{1-q})/q  ≤  sum_{j=1}^c (1 - f_j^q)/q  ≤  (c - 1)/q
```

(upper: `f_j ≤ f_j^q`; lower: `sum f_j^q ≤ c^{1-q}` by concavity of `t^q`). Pushing these through the
same uniform-noise risk expansion gives, with `f*` the clean-risk minimizer and `f̂` the noisy-risk
minimizer and `η < 1 - 1/c`:

```
0 ≤ R^η_{L_q}(f*) - R^η_{L_q}(f̂) ≤ A,   A  = η[c^{1-q} - 1] / (q(c-1))        ≥ 0,
A' ≤ R_{L_q}(f*) - R_{L_q}(f̂) ≤ 0,      A' = η[1 - c^{1-q}] / (q(c-1-ηc))     < 0.
```

The clean-risk gap of the noisy optimum is bounded by `|A'|`, governed by the factor `c^{1-q} - 1`:
**zero at `q = 1`** (exact tolerance, recovering MAE) and widening as `q → 0` (toward CCE's
non-tolerance). For class-dependent noise with each correct label more likely than any particular wrong
label, and with `R_{L_q}(f*) = 0`, the analogous noisy-risk gap is bounded by
`B = (c^{1-q} - 1)/q · E[1 - η_y] ≥ 0`. The risk side and the gradient side agree: `q` is a
robustness↔learnability knob.

## Default

`q = 0.7`. It is a genuine hyperparameter (tune by validation accuracy; noisier data wants larger
`q`), but `0.7` is the compromise that suppresses overfitting to noise while keeping convergence close
to cross entropy on standard image benchmarks.

## Optional extension — truncated L_q

For extra robustness, flatten the loss below a confidence threshold `k`:
`L_trunc = L_q(k)` if `f_j ≤ k` else `L_q(f, e_j)`, with `L_q(k) = (1 - k^q)/q`. Below `k` the gradient
is zero, so very-low-confidence (likely-poisoned) samples are pruned. For suitable `k` this tightens the
class-sum bracket (more noise-tolerant), but it is optimized via an alternating convex search /
self-paced pruning loop (binary weights `w_i = 1` iff `f_{y_i} > k`, initialized all-1 so early training
uses all data), which touches the training loop. The plain `L_q` below is the pure drop-in objective;
`k = 0.5` is the default for the truncated variant.

## Working code

Fills the `compute_loss` slot of the fixed loss harness:

```python
import torch


class RobustLoss:
    """Generalized cross entropy (L_q) loss for label noise.

    L_q(f(x), e_y) = (1 - f_y^q) / q, averaged over the minibatch. f_y is the softmax
    probability of the given (possibly corrupted) label. q in (0,1] interpolates CCE
    (q -> 0) and MAE (q = 1): the per-sample gradient is f_y^q times CCE's, down-weighting
    low-confidence (likely-noisy) samples while keeping enough pull on hard samples to train.
    """

    def __init__(self):
        self.q = 0.7  # robustness <-> learnability tradeoff

    def compute_loss(self, logits, labels, epoch):
        probs = torch.softmax(logits, dim=1)                  # f(x)
        p = probs.gather(1, labels[:, None]).clamp_min(1e-8)  # f_y, floored for stability
        return ((1.0 - p.pow(self.q)) / self.q).mean()        # (1 - f_y^q)/q, mean over batch
```
