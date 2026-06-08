# SVRG — Stochastic Variance Reduced Gradient

## Problem

Minimize a finite sum of smooth losses,

  min_w P(w) = (1/n) Σ_{i=1}^n ψ_i(w),

with each ψ_i convex and L-smooth and P γ-strongly convex (L ≥ γ > 0). Full gradient descent
converges linearly with a constant step size but costs n gradients per step. SGD costs one
gradient per step but, because the per-example gradients do not vanish at the optimum
(∇ψ_i(w*) ≠ 0 while ∇P(w*) = 0), a constant step leaves a squared-distance floor of order
ηE_i‖∇ψ_i(w*)‖²/γ around w*; the step size must decay, giving only the sublinear O(1/k) rate.
That rate is optimal for an oracle restricted to fixed-noise unbiased gradient samples. SVRG attains gradient
descent's **linear** rate with SGD's cheap step **and** a **constant** step size, using only
O(d) extra memory.

## Key idea — control variate with a periodic full-gradient snapshot

Use the classical control-variate estimator for E[X] = ∇P(w): with X = ∇ψ_i(w) and a correlated
Y = ∇ψ_i(w̃) at a reference point w̃ whose mean E[Y] = (1/n)Σ_i∇ψ_i(w̃) = ∇P(w̃) =: μ̃ is exactly
computable in one pass,

  g = X − Y + E[Y] = ∇ψ_i(w) − ∇ψ_i(w̃) + μ̃.

- **Unbiased:** E_i[g] = ∇P(w) − ∇P(w̃) + μ̃ = ∇P(w), for any reference w̃.
- **Vanishing variance:** the matched index i makes ∇ψ_i(w) − ∇ψ_i(w̃) → 0 and μ̃ = ∇P(w̃) → 0
  as w, w̃ → w*, so g → 0. The per-sample offset ∇ψ_i(w*) that floored SGD cancels against
  itself.

The exact mean μ̃ costs a full pass, so it is amortized: snapshot w̃, compute μ̃ once, run m
cheap inner steps referencing the same (w̃, μ̃), then refresh w̃. Only w̃ and the single vector
μ̃ are stored — no per-example gradient table (the gap left open by SAG/SDCA, which store an
n×d table / n duals).

## Algorithm

```
Parameters: inner length m, step size η  (constant, η < 1/(2L))
Initialize  w̃_0
for s = 1, 2, ... :
    w̃ = w̃_{s-1}
    μ̃ = (1/n) Σ_{i=1}^n ∇ψ_i(w̃)               # one full pass
    w_0 = w̃
    for t = 1, ..., m :
        pick i_t ∈ {1, ..., n} uniformly
        w_t = w_{t-1} − η ( ∇ψ_{i_t}(w_{t-1}) − ∇ψ_{i_t}(w̃) + μ̃ )
    option I (practice):  w̃_s = w_m           # last iterate (or average of inner iterates)
    option II (analysis): w̃_s = w_t,  t ~ Uniform{0, ..., m-1}
```

Per stage: n + 2m gradient evaluations; choose m = O(n) to amortize the snapshot pass. For
linear models ∇ψ_i(w) = φ'_i(w^Tx_i)x_i, so ∇ψ_i(w̃) compresses to one stored scalar per example
and need not be recomputed.

## Linear-convergence theorem (option II)

**Theorem.** Assume every ψ_i is convex and L-smooth, P is γ-strongly convex with γ > 0, and
w* = argmin P. If m is large enough that

  α = 1/(γη(1 − 2Lη)m) + 2Lη/(1 − 2Lη) < 1   (requires η < 1/(2L)),

then SVRG with option II converges geometrically in expectation:

  E[P(w̃_s) − P(w*)] ≤ α^s [P(w̃_0) − P(w*)].

At conditioning κ = L/γ = n, taking η = 0.1/L and m = 50n gives α = 1/2; more generally a
sufficient constant multiple m = O(n) gives O(n ln(1/ε)) gradient evaluations to reach accuracy ε
— versus n²ln(1/ε) for batch gradient descent — matching SAG/SDCA with O(d) memory.

### Proof

**Lemma (smoothness ⇒ gradient gap bounded by suboptimality).** Fix i and set
g_i(w) = ψ_i(w) − ψ_i(w*) − ∇ψ_i(w*)^T(w − w*). Then ∇g_i(w*) = 0, so w* minimizes the convex
L-smooth g_i and g_i(w*) = 0. Minimizing the smoothness upper bound of one gradient step,

  0 = g_i(w*) ≤ min_η [g_i(w) − η‖∇g_i(w)‖² + 0.5Lη²‖∇g_i(w)‖²] = g_i(w) − (1/2L)‖∇g_i(w)‖²,

so ‖∇ψ_i(w) − ∇ψ_i(w*)‖² ≤ 2L[ψ_i(w) − ψ_i(w*) − ∇ψ_i(w*)^T(w − w*)]. Averaging over i and using
∇P(w*) = 0,

  (1/n) Σ_i ‖∇ψ_i(w) − ∇ψ_i(w*)‖² ≤ 2L[P(w) − P(w*)].    (★)

**Second-moment bound.** With v_t = ∇ψ_{i_t}(w_{t-1}) − ∇ψ_{i_t}(w̃) + μ̃, insert ∇ψ_{i_t}(w*)
and split into a current-point bracket and a centered snapshot bracket. Using
‖a − b‖² ≤ 2‖a‖² + 2‖b‖², E‖ξ − Eξ‖² ≤ E‖ξ‖², and (★) at w_{t-1} and at w̃:

  E‖v_t‖² ≤ 2E‖∇ψ_{i_t}(w_{t-1}) − ∇ψ_{i_t}(w*)‖² + 2E‖∇ψ_{i_t}(w̃) − ∇ψ_{i_t}(w*)‖²
         ≤ 4L[P(w_{t-1}) − P(w*)] + 4L[P(w̃) − P(w*)].    (♦)

**One-step contraction.** E_{i_t}[v_t] = ∇P(w_{t-1}); with convexity
−(w_{t-1} − w*)^T∇P(w_{t-1}) ≤ P(w*) − P(w_{t-1}) and (♦),

  E‖w_t − w*‖² ≤ ‖w_{t-1} − w*‖² − 2η(1 − 2Lη)[P(w_{t-1}) − P(w*)] + 4Lη²[P(w̃) − P(w*)].

**Telescope + strong convexity.** Fix w̃ = w̃_{s-1}, sum over t = 1..m (the ‖·−w*‖² terms
telescope), take expectation, use w_0 = w̃, drop E‖w_m − w*‖² ≥ 0, use option II so
(1/m)Σ_t E[P(w_{t-1}) − P(w*)] = E[P(w̃_s) − P(w*)], and ‖w̃ − w*‖² ≤ (2/γ)[P(w̃) − P(w*)]:

  2η(1 − 2Lη)m · E[P(w̃_s) − P(w*)] ≤ (2/γ + 4Lmη²) · E[P(w̃) − P(w*)].

Dividing gives E[P(w̃_s) − P(w*)] ≤ α · E[P(w̃_{s-1}) − P(w*)]; iterating over s yields the bound. ∎

The constraint η < 1/(2L) keeps the contraction coefficient 2η(1 − 2Lη) positive; α's first term
→ 0 as m → ∞ and its second term is small for small η, so α < 1 is attainable. For smooth convex
P without strong convexity the same machinery gives O(1/T); for a nonconvex model warm-started
near a locally strongly convex minimum, the theorem applies locally for local geometric convergence
with a constant step.

## Implementation

```python
import numpy as np

class FiniteSumProblem:
    def __init__(self, X, y, reg):
        self.X, self.y, self.reg = X, y, reg
        self.n, self.d = X.shape
    def grad_i(self, w, i):
        # e.g. logistic loss + L2: -y_i/(1+exp(y_i x_i.w)) x_i + reg*w
        z = self.y[i] * (self.X[i] @ w)
        return -self.y[i] / (1.0 + np.exp(z)) * self.X[i] + self.reg * w
    def full_grad(self, w):
        return np.mean([self.grad_i(w, i) for i in range(self.n)], axis=0)

def svrg(problem, w0, eta, m, n_outer):
    """g = grad_i(w) - grad_i(w_tilde) + mu_tilde: unbiased estimate of grad P(w)
    whose variance vanishes as w, w_tilde -> w*, so constant eta < 1/(2L) converges
    geometrically. Only w_tilde and mu_tilde stored (O(d), no per-example table)."""
    w_tilde = w0.copy()
    for s in range(n_outer):
        mu_tilde = problem.full_grad(w_tilde)          # snapshot: one full pass
        w = w_tilde.copy()
        for t in range(m):
            i = np.random.randint(problem.n)
            g = problem.grad_i(w, i) - problem.grad_i(w_tilde, i) + mu_tilde
            w = w - eta * g                            # constant step, no decay
        w_tilde = w                                    # refresh reference (last iterate)
    return w_tilde
```

## Why it works, in one line

SGD's rate is capped by a *variance* floor (per-sample gradients don't vanish at w*), not bias;
a matched-index control variate with a periodically refreshed full-gradient snapshot keeps the
update unbiased while driving its variance to zero as the iterate approaches the optimum,
restoring linear convergence with a constant step size and O(d) memory.
