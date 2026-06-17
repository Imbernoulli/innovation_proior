# Symmetric Cross Entropy Learning (SCE / SL)

Symmetric Cross Entropy Learning (SL) trains a classifier under label noise by adding a
noise-tolerant counterpart to standard cross entropy:

```
ell_sl = alpha * ell_ce + beta * ell_rce
ell_ce  = - sum_k q_k log p_k        (Cross Entropy: fast convergence, not noise-robust)
ell_rce = - sum_k p_k log q_k        (Reverse Cross Entropy: symmetric => noise-tolerant)
```

where `p = softmax(z)` is the prediction and `q` is the one-hot label. Because `q_k = 0` for
non-target classes, RCE needs `log 0`, which is **defined to a finite negative constant**
`log 0 := A`. SCE is the cross-entropy analogue of symmetric KL divergence
`KL(q||p) + KL(p||q)`.

## Problem it solves

Training an accurate classifier when a fraction of training labels are flipped to wrong
classes (the test set is clean), without architectural changes, auxiliary clean data, or a
known noise-transition matrix. The motivating diagnosis: cross entropy not only overfits noisy
labels on "easy" classes but, more importantly, **under-learns "hard" classes**, leaving them
far below their clean-label accuracy. The fix must add learning pressure for hard classes while
itself being noise-tolerant.

## Key idea

- **Noise-tolerance = symmetry.** Under the symmetric-loss criterion, a loss is noise-tolerant
  under symmetric/uniform noise (rate `eta < 1 - 1/K`) if it is *symmetric*:
  `sum_{i=1}^K L(f(x), i) = C` (constant). Proof: the noisy risk is affine in the clean risk,
  `R^eta(f) = C*eta/(K-1) + (1 - eta*K/(K-1)) R(f)`, whose multiplier is positive when
  `eta < 1 - 1/K`, so argmin is preserved. CE fails this; MAE satisfies it but its gradient
  `-2 p_y(1-p_y)` vanishes for hard examples (`p_y -> 0`), so MAE converges slowly.
- **RCE is symmetric.** With `log 0 := A`, for a one-hot label `ell_rce = -A(1 - p_y)`, and
  `sum_{i=1}^K ell_rce(f(x), i) = -(K-1)A`, a constant. So RCE is noise-tolerant (Ghosh's
  theorem with `C = -(K-1)A`). It generalizes MAE: `A = -2` makes RCE exactly MAE; `A` is a
  steepness knob.
- **Two complementary forces, decoupled.** CE supplies dense, well-conditioned gradients for
  convergence; RCE supplies noise-tolerant learning pressure. Separate coefficients let you
  *lower* `alpha` to ease CE's overfitting and *raise* `beta` for more robust signal —
  independently (unlike GCE's single `q`). Easy-to-converge data wants small `alpha`;
  hard-to-converge data (e.g. CIFAR-100) wants large `alpha`.

## Why the gradient fixes under-learning

For the simplified `alpha=beta=1`, with `d p_k/d z_j = p_k(1-p_k)` if `k=j` else `-p_j p_k`,

```
d ell_sl / d z_j = (p_j - q_j) + p_j ( sum_k p_k log q_k - log q_j ).
```

- **Labeled class** (`q_j = q_y = 1`): `= (p_j - 1) - (A p_j^2 - A p_j)`, equivalently
  `(p_j - 1) + A p_j(1-p_j)`. Since `A<0`, the RCE contribution to this derivative is
  non-positive and gradient descent raises `p_y` faster; its magnitude `(-A) p_j(1-p_j)` is
  maximal at `p_j = 0.5` — biggest boost for half-learned (hard) classes, tapering to 0 as
  `p_y -> 1`. Adaptive pacing: speed up hard classes, ease off already-learned ones.
- **Wrong class** (`q_j = 0`): `= p_j - A p_j p_y`. The extra `-A p_j p_y` (>=0) suppresses
  residual wrong-class mass in proportion to `p_y`. If `p_y ~ 0` (the network rejects the
  label, as with a flip), no suppression — self-gated robustness.

## Defaults and why

- `A` (= `log 0`): a negative constant; `A = -2` is MAE. In code, clamping the one-hot label up
  to a floor `f` realizes `A = log(f)`; the Keras/TF implementation uses `f = 1e-4`, so
  `A ~= -9.21` in that implementation. Preferred over label smoothing because clamping biases
  the model only at the finite set of `q_k = 0` points and not at `q_y = 1`, whereas smoothing
  biases every point.
- `alpha, beta`: `beta` and `A` overlap (`beta * ell_rce = beta(-A)(1-p_y)`), so fix `A` and
  tune the coefficients. CIFAR-10: `alpha = 0.1, beta = 1.0`; MNIST: `alpha = 0.01, beta = 1.0`;
  CIFAR-100 (hard convergence): `alpha = 6.0, beta = 0.1`. Small `alpha` eases overfitting; too
  small slows convergence (behaves like RCE alone).

## Robustness theorem

For symmetric/uniform noise at rate `eta`, RCE (symmetric with `C = -(K-1)A`) satisfies
`R^eta(f^*) - R^eta(f) = (1 - eta*K/(K-1))(R(f^*) - R(f)) <= 0` for `eta < 1 - 1/K`, so the
clean global minimizer is also the noisy global minimizer. For asymmetric/class-conditional
noise with `eta_{yk} < 1 - eta_y` and `R(f^*) = 0`, RCE is likewise noise-tolerant
(`f^*_eta = f^*`): rewriting the risk with
`ell_rce(f,y) = -(K-1)A - sum_{k!=y} ell_rce(f,k)` leaves positive coefficients
`1 - eta_y - eta_yk`, and zero clean RCE risk forces the noisy minimizer to put zero mass on
every wrong class.

## Working code

This fills the loss slot of the fixed Keras classification harness. CE uses clipped predicted
probabilities; RCE clips the one-hot label and takes the prediction-weighted log of that label,
so `A = log(floor)`.

```python
import tensorflow as tf


def symmetric_cross_entropy(alpha, beta):
    """ell = alpha * CE + beta * RCE.

    CE  = -sum_k q_k log p_k
    RCE = -sum_k p_k log q_k
    Clipping the one-hot label to 1e-4 realizes A = log(1e-4)."""

    def loss(y_true, y_pred):
        y_true_1 = y_true
        y_pred_1 = y_pred
        y_true_2 = y_true
        y_pred_2 = y_pred

        y_pred_1 = tf.clip_by_value(y_pred_1, 1e-7, 1.0)
        y_true_2 = tf.clip_by_value(y_true_2, 1e-4, 1.0)

        ce = tf.reduce_mean(-tf.reduce_sum(y_true_1 * tf.log(y_pred_1), axis=-1))
        rce = tf.reduce_mean(-tf.reduce_sum(y_pred_2 * tf.log(y_true_2), axis=-1))
        return alpha * ce + beta * rce

    return loss
```

The Keras/TF implementation clips the prediction to `[1e-7, 1]` and the one-hot label to
`[1e-4, 1]`, then returns
`alpha * mean(-sum q log p) + beta * mean(-sum p log q)`; the label clip floor fixes
`A = log(1e-4)`.

## Relation to prior methods

- **MAE** = RCE at `A = -2`. RCE generalizes it with a steepness knob and,
  combined with CE, supplies the dense gradients MAE lacks.
- **GCE**, `(1 - p_y^q)/q`, interpolates CE (`q->0`) and MAE (`q->1`)
  with one knob; SCE instead keeps CE and *adds* a symmetric robust term, decoupling
  convergence (`alpha`) from robustness (`beta`).
- **Bootstrapping** mixes label and prediction in the target; SCE needs no
  self-target and carries a noise-tolerance guarantee.
- SCE's RCE term can be bolted onto other robust-loss / label-correction methods to add
  robustness with minimal change.
