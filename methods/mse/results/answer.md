# Least squares / mean squared error, distilled

The method of least squares chooses the unknowns of an over-determined system so that the
**sum of the squares of the residuals is a minimum**. Its empirical tensor form — the mean
of the squared elementwise differences between a prediction and a target — is the **mean squared
error (MSE)**. In Gauss's probability argument, demanding that equal direct observations return
the arithmetic mean forces a Gaussian error law, under which this same squared-error rule is the
maximum-likelihood fit. In linear algebra, it is the orthogonal projection onto the model space;
under the Gauss-Markov assumptions, it is the minimum-variance linear unbiased estimator.

## Problem it solves

Combine more (noisy) observation equations than there are unknowns into one definite estimate.
The residuals cannot all be zeroed, so a principled rule is needed to "distribute the error"
that (1) uses all the data, (2) gives a single determinate system for any number of unknowns,
(3) reduces to the arithmetic mean in the one-quantity case, and (4) has an optimality
justification.

## Key idea

Make `Φ(β) = Σ_k E_k²` a minimum, with residual `E_k = a_k + b_k x + c_k y + ⋯`. Because `Φ`
is quadratic, its stationarity condition is **linear**, giving one symmetric equation per
unknown (the *normal equations*). Squaring is the simplest sign-blind smooth choice that yields
a linear solve (so it works for any number of unknowns), reduces to the mean, equals the
maximum-likelihood estimate under Gaussian errors, and corresponds to orthogonal projection.
The absolute-value (L1) and minimax (L∞) measures are non-smooth, combinatorial, and in
practice limited to very few unknowns.

## Final form

**Objective and normal equations.** Minimize `Φ(x,y,z,…) = Σ_k (a_k + b_k x + c_k y + f_k z + ⋯)²`.
Setting `∂Φ/∂x = 0`, etc., and dropping the factor 2:

```
0 = Σab + x Σb² + y Σbc + z Σbf + ⋯
0 = Σac + x Σbc + y Σc² + z Σcf + ⋯
0 = Σaf + x Σbf + y Σcf + z Σf² + ⋯
```

`Σab = Σ_k a_k b_k`, `Σb² = Σ_k b_k²`, etc. The matrix is symmetric (the cross-coefficient
`Σbc` is shared between the x- and y-equations), and there are exactly as many equations as
unknowns, so the system is determinate. In matrix form with residual `y − Xβ`:

```
minimize ‖y − Xβ‖²   ⇒   XᵀX β̂ = Xᵀy   ⇒   β̂ = (XᵀX)⁻¹ Xᵀy   (X full column rank)
```

**Arithmetic mean as a special case.** One unknown `x`, observations `a', a'', …`:
`min Σ(a^(i) − x)²` gives `0 = Σ(a^(i) − x)`, so `x = (Σ a^(i))/n`, the arithmetic mean.

**Maximum-likelihood justification.** Model an observation as truth + error `Δ` with density
`φ(Δ)`; independent observations give joint probability `Ω = Π_i φ(Δ_i)`; with a flat prior the
most probable unknowns maximize `Ω`. Impose that for equal direct observations the most probable
value is the arithmetic mean. Writing
`g(Δ) = φ'(Δ)/φ(Δ)`, stationarity at the mean gives `Σ_i g(Δ_i) = 0` whenever `Σ_i Δ_i = 0`.
Two residuals make `g` odd; three residuals `a,b,−(a+b)` give `g(a+b)=g(a)+g(b)`; continuity
therefore forces `g(Δ)=kΔ`:

```
φ'(Δ)/φ(Δ) = k Δ   ⇒   φ(Δ) = A·exp((k/2)Δ²),  k < 0,  write k = −2h²
⇒   φ(Δ) = (h/√π)·exp(−h² Δ²)   (A = h/√π by ∫exp(−h²Δ²)dΔ = √π/h, h = precision).
```

Then `Ω = (h/√π)^N exp(−h² Σ_i Δ_i²)` is monotone decreasing in `Σ Δ_i²`, so **maximizing the
probability = minimizing the sum of squares**. For unequal precisions, `φ_i = (h_i/√π)exp(−h_i²Δ_i²)`
gives the weighted objective `Σ_i h_i² Δ_i²`. Since this density has `Var(Δ_i)=1/(2h_i²)`, the
weight is proportional to inverse variance; the common factor 2 does not affect the minimizer.
The logic is non-circular: the *postulate* is the mean; the direct-observation consistency
argument forces the Gaussian; the Gaussian forces least squares for the general equations;
reproducing the mean in the one-unknown case is a consequence, not the assumption.

**Geometry.** `‖y − Xβ‖²` is squared Euclidean distance, so its minimizer `ŷ = Xβ̂` is the
orthogonal projection of `y` onto `col(X)`; the residual is perpendicular to the model space
(`Xᵀ(y − Xβ̂) = 0` — the "normal" equations), `ŷ = X(XᵀX)⁻¹Xᵀ y = P y`. This Euclidean projection
is linear, which is why the squared-error solution is a linear system.

**Distribution-free optimality (Gauss–Markov).** With `E[ε] = 0`, `Var(ε) = σ²I`, among all
linear unbiased estimators `β̃ = Cy` (so `CX = I`), writing `C = (XᵀX)⁻¹Xᵀ + D` forces `DX = 0`,
hence `Var(β̃) = σ²(XᵀX)⁻¹ + σ²DDᵀ = Var(β̂) + σ²DDᵀ ⪰ Var(β̂)`. So the least-squares estimator
is the **best linear unbiased estimator** with no normality assumption; equality only when
`D = 0`.

## Working code

The empirical least-squares objective: the squared Euclidean residual averaged over elements
(equal precision, so uniform weights). Filling the aggregation slot:

```python
import torch.nn as nn
import torch.nn.functional as F


class CustomPredictionLoss(nn.Module):
    """Mean squared error prediction loss.

    Aggregates the elementwise residual as the mean of its square:
    (1/N) * sum_i (predicted_i - state_i)^2 over all N elements. This is
    the empirical least-squares objective; minimizing it makes the prediction the
    equal-precision best fit to the target.
    """

    def __init__(self):
        super().__init__()

    def forward(self, state, predicted):
        return F.mse_loss(predicted, state, reduction="mean")
```

Equivalent explicit form (what `F.mse_loss` computes with `reduction='mean'`):

```python
def forward(self, state, predicted):
    return ((predicted - state) ** 2).mean()
```

Dividing by `N` does not move the minimizer (it scales the objective by a constant); it keeps
the loss scale independent of the array size, which is the standard convention.
