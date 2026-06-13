# Context: selecting model complexity from a finite sample

## Research question

I am given a finite training sample of `ℓ` independent, identically distributed pairs
`(x₁,y₁),…,(x_ℓ,y_ℓ)` drawn from a fixed but unknown joint distribution `P(x,y)`. A learning machine can
implement a class of functions `f(x,w), w ∈ W`. I want to choose one function from that class whose
**expected** loss on future data — the risk

    R(w) = ∫ L(y, f(x,w)) dP(x,y)

— is as small as possible. The catch is that `P` is unknown; the only thing I can actually compute is
the loss averaged over the sample I happen to hold. The precise problem is therefore not "fit the
sample" but "choose, from a sample, a function that will be good on the distribution the sample came
from." Two sub-questions decide whether any method for this is sound: is the procedure *consistent*
(does the risk of the function it returns approach the best achievable risk as `ℓ → ∞`?), and how *fast*
does that happen for a finite `ℓ`? A solution has to answer the second question constructively, because
in practice `ℓ` is finite and often small relative to how rich the class is — and it must do so without
knowing `P`, i.e. with a *distribution-free* guarantee, otherwise the guarantee is unusable.

## Background

The only obviously implementable induction principle is **empirical risk minimization (ERM)**: replace
the unknown risk by the empirical risk

    R_emp(w) = (1/ℓ) Σᵢ L(yᵢ, f(xᵢ,w))

and return the function that minimizes it. Classical estimators are special cases — least squares is ERM
under squared loss, maximum likelihood is ERM under log loss. So any theory of ERM is a theory of those
methods too.

ERM is justified by a law of large numbers: for a *single fixed* function, `R_emp(w) → R(w)`. But the
function ERM returns is *not* fixed in advance — it is chosen *after* looking at the sample, precisely to
make `R_emp` small. The consistency of ERM was shown (Vapnik & Chervonenkis, 1968, 1971; necessary-and-
sufficient form 1989) to be equivalent to a much stronger statement — *uniform* convergence of the
empirical risk to the true risk over the **entire** class:

    Prob{ sup_{w∈W} |R(w) − R_emp(w)| > ε } → 0   as ℓ → ∞.

This is the crucial fact: generalization is a worst-case property of the *whole class*, not
of any one function. The rate at which this uniform convergence happens is controlled by a capacity
measure of the class — the **VC dimension** `h`, the largest number of points the class can label in all
`2^h` possible ways (shatter). For indicator functions with VC dimension `h`,

    Prob{ sup_w |R(w) − R_emp(w)| > ε } < (2eℓ/h)^h exp{−ε²ℓ},

a bound that is *distribution-free* — it depends on the class through `h` and on the sample through `ℓ`,
not on `P`. The VC dimension is **not** the number of free parameters: a one-parameter family
`sign(sin(αx))` shatters arbitrarily many points (infinite VC dimension), and conversely large-margin
hyperplanes in a high-dimensional space can have VC dimension far below the dimension count.

The motivating phenomenon that all of this has to explain is **overfitting**: with a rich enough class
one can drive the training error to zero and still predict future data badly. It is a documented fact
about existing learning machines — a sufficiently large neural network can memorize its training set
(zero training error) while making many errors on a held-out set. The uniform-convergence picture says
why: when the class is rich relative to the sample (the ratio `ℓ/h` is small), the gap
`sup_w|R − R_emp|` is large, so a small training error carries no guarantee about the true risk.

Three threads of prevailing wisdom frame the design space. **Bias–variance**: a class too small cannot
represent the target (bias / approximation error), a class too large fits sampling noise (variance /
estimation error); good prediction needs a balance. **Occam's razor / regularization**: prefer the
"simplest" hypothesis consistent with the data; in practice this took the form of adding a roughness or
norm penalty (ridge, weight decay, smoothing splines — Tikhonov regularization, Wahba's spline models).
**Model-order selection criteria**: choose the model that minimizes fit-plus-complexity, where
complexity is the *parameter count*. Akaike's AIC minimizes `N·log J(l) + 2l` (`l` = number of
parameters, `J(l)` = fit / MSE); Rissanen's MDL minimizes a description length `≈ N·log J + (l/2)·log l`;
Bayesian model selection penalizes via an Occam factor. Every one of these uses the *number of
parameters* as the complexity measure. The uniform-convergence result above is the standing challenge to
that choice: for nonlinear machines it is not the count but the *capacity* (VC dimension) that governs
the generalization gap, and the two can differ wildly.

## Baselines

**Empirical risk minimization (ERM) on a fixed class.** Minimize `R_emp(w)` over a single, pre-chosen
class `f(x,w), w∈W`. Core idea: trust the training error as a surrogate for the risk. Math: return
`w* = argmin_w R_emp(w)`. Limitation: the empirical risk is a trustworthy stand-in for the true risk
only when the class is poor relative to `ℓ` (the tail above is tight); on a class that is rich relative
to `ℓ` the worst-case gap is large, so if the class is fixed too large, ERM overfits, and if fixed too
small, ERM underfits. ERM gives no handle for *choosing* the class; it is justified only when `ℓ/h` is
large.

**Penalized model-order selection (AIC, MDL, Bayesian).** Fit a nested family of models of increasing
parameter count, and select the one minimizing `fit + penalty(number of parameters)`. AIC:
`min_l [N·log J(l) + 2l]`. MDL: `min_l [N·log J(l) + (l/2)·log l]`. Core idea: penalize complexity to
avoid overfitting; complexity = parameter count. Limitation: the penalty tracks the *number of
parameters*, but the generalization gap is governed by *capacity*; for nonlinear machines these diverge
(infinite-VC one-parameter families; sub-`n+1`-VC margin hyperplanes), so these criteria can both
over-penalize benign rich classes and under-penalize malign small ones. They are also derived under
asymptotic / parametric assumptions, not as distribution-free finite-sample guarantees.

**Regularization (ridge / weight decay / smoothing splines).** Minimize `R_emp(w) + γ·‖w‖²` (or a
roughness penalty). Core idea: bias the solution toward "smooth"/small-norm functions (Occam). Math:
`w_γ = argmin_w [R_emp(w) + γ‖w‖²]`; `γ` trades fit against smoothness. Limitation: `γ` is a free knob
chosen heuristically or by cross-validation, with no a-priori finite-sample risk guarantee attached to a
given `γ`; the penalty is a proxy for complexity whose link to the true generalization gap is, at
baseline, not quantified.

**Cross-validation.** Split the data; fit candidates on one part, score them on the held-out part, keep
the best. Core idea: estimate each candidate's risk directly on data it did not see. For a held-out set
of size `n`, `|R̂_V(h) − R(h)| ≤ √(log(2/δ)/(2n))` with probability `1−δ` for a fixed `h`, so the
validation risk concentrates sharply. Limitation: it spends data on the held-out set; the guarantee
degrades if hypotheses are chosen *adaptively* against the validation set; and it gives an estimate of
risk rather than an a-priori, distribution-free *bound* computable from the training sample alone.

**Separating hyperplanes.** Fit a linear decision rule `f(x,w) = θ(w·x + b)` by minimizing training
error. Core idea: a single linear threshold unit. Limitation: in `n` dimensions the class of all
hyperplanes has VC dimension `n+1`, and the class is too inflexible to reach low empirical risk on many
real problems; there is no mechanism here for controlling capacity below `n+1` or for going nonlinear.

## Evaluation settings

The natural yardstick is a function-estimation / pattern-recognition task with a finite labeled sample
and a held-out test set, measuring out-of-sample error (probability of misclassification for indicator
loss; mean-squared error for regression; log-loss for density estimation). A standard benchmark of the
period is handwritten-digit recognition from postal data (the US Postal Service zip-code corpus, with a
fixed train/test split on the order of thousands of training and a couple thousand test images), where
prediction error on the test set is the metric and a multilayer neural network trained by
back-propagation is the reference learning machine. More generally the protocol is: fix a family of
candidate classes of increasing capacity, train on the sample, and report risk on data held out from
training. Sample size `ℓ` relative to capacity `h` — the ratio `ℓ/h` — is itself a primary axis of the
setting, since the whole question of whether the training error can be trusted turns on it.

## Code framework

The primitives that already exist before any complexity-selection principle: a way to enumerate a family
of candidate classes, an ERM solver that minimizes empirical risk inside one class, and an evaluation of
empirical risk on the sample. What does *not* yet exist is the rule that turns a per-class fit into a
choice of class — that is the one empty slot.

```python
import numpy as np

def empirical_risk(predict, X, y, loss):
    # average loss on the training sample — the only computable functional
    return np.mean([loss(yi, predict(xi)) for xi, yi in zip(X, y)])

class HypothesisClass:
    """One candidate class of functions the learning machine can implement."""
    def fit_erm(self, X, y, loss):
        # minimize empirical risk WITHIN this class (least squares / max-likelihood / etc.)
        raise NotImplementedError
    def capacity(self, X):
        # a measure of this class's richness, available before fitting
        raise NotImplementedError

def candidate_classes():
    # a family of classes the designer is willing to consider
    # (e.g. ordered by some richness knob) — returned for the selector to range over
    raise NotImplementedError

def select_model(X, y, loss, delta):
    # TODO: the missing principle. Given the primitives above, decide which
    # function (and from which class) to return so that it generalizes — not
    # merely fits. The rule for making that choice is unknown here.
    pass
```
