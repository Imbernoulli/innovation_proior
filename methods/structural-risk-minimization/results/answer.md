# Structural Risk Minimization (SRM)

## The problem it solves

From a finite i.i.d. sample of size `ℓ` drawn from an unknown `P(x,y)`, choose a predictor with small
**true** risk `R(w) = ∫ L(y,f(x,w)) dP(x,y)`. Empirical risk minimization (ERM) — minimizing the
training error `R_emp(w) = (1/ℓ)Σᵢ L(yᵢ,f(xᵢ,w))` over a fixed class — is justified only when the sample
is large relative to the class's capacity; on a rich class it overfits (drives `R_emp` to zero while `R`
stays large). SRM is the model-**selection** principle that fixes this by choosing model complexity to
minimize a guaranteed bound on `R`, not the training error.

## Key idea

The generalization gap of a class is governed by its **VC dimension** `h` (the largest number of points
it can shatter), not by its parameter count. The uniform-convergence bound gives, with probability
`1 − η`, simultaneously for all functions in a class of VC dimension `h`:

    R(w) ≤ R_emp(w) + confidence_term(h, ℓ, η),

a distribution-free bound whose confidence term grows with `h` and shrinks with `ℓ`. A single fixed
class is stuck on one point of the fit-vs-capacity tradeoff. SRM instead organizes the hypothesis space
into a **nested structure** of classes of increasing capacity, `S₁ ⊂ S₂ ⊂ ⋯` with `h₁ ≤ h₂ ≤ ⋯`, runs
ERM inside each, and returns the (class, function) pair minimizing the **guaranteed risk**. Nesting makes
the minimal empirical risk fall and the confidence term rise monotonically, so the bound has a clean
minimum at the optimal complexity.

## The bound (binary loss)

VC tail bound:

    Prob{ sup_w |R(w) − R_emp(w)| > ε } < (2eℓ/h)^h · exp{−ε²ℓ}.

Setting the right side to `η` and inverting gives, with probability `1 − η`, for all `w`:

    R(w) ≤ R_emp(w) + C₀(ℓ/h, η),
    C₀(ℓ/h, η) = √( ( h(ln(2ℓ/h) + 1) − ln η ) / ℓ ).

In the low-error regime, the relative-deviation refinement bounds `(R − R_emp)/√R` and gives the tighter,
`R_emp`-dependent confidence interval. With `A = h(ln(2ℓ/h)+1) − ln η`:

    R(w) ≤ R_emp(w) + C₁(ℓ/h, R_emp, η),
    C₁ = 2·(A/ℓ)·( 1 + √( 1 + R_emp·ℓ / A ) ),

which at `R_emp = 0` collapses (the square root becomes `1`) to the **fast rate**
`C₁(ℓ/h, 0, η) = 4C₀² = O(h ln ℓ / ℓ)`, versus the slow `O(√(h ln ℓ / ℓ))` when `R_emp` is order one. For a bounded loss `0 ≤ L ≤ B`:

    R(w) ≤ R_emp(w) + (Bε/2)( 1 + √(1 + 4R_emp/(Bε)) ),   ε = 4·(h(ln(2ℓ/h)+1) − ln η)/ℓ.

## The SRM principle

1. Provide the set of functions with an admissible nested structure `S₁ ⊂ S₂ ⊂ ⋯`,
   `h₁ ≤ h₂ ≤ ⋯` (admissible: `∪ₖ Sₖ` dense, each `hₖ` finite, each `Sₖ` totally bounded).
2. For each `Sₖ`, minimize empirical risk inside it → `αₖ` with risk `R_emp(αₖ)` and guaranteed risk
   `R(αₖ) ≤ R_emp(αₖ) + Ω(ℓ/hₖ)`.
3. Return the `(class, function)` pair minimizing the **guaranteed** risk
   `R_emp(αₖ) + Ω(ℓ/hₖ)`. As `k` rises, `R_emp(αₖ)` decreases and `Ω(ℓ/hₖ)` increases; the minimum sits
   at the complexity that balances fit against the capacity penalty.

**Consistency.** For any distribution, SRM is universally strongly consistent: with the structure level
`n(ℓ)` grown with the data, `R(α^{n(ℓ)}) → R(α₀)` with probability one, at asymptotic rate
`V(ℓ) = r_{n(ℓ)} + B_{n(ℓ)}√(h_{n(ℓ)} ln ℓ / ℓ)` (approximation error `r_n` plus capacity penalty),
provided `lim B²_{n} h_{n} ln ℓ / ℓ = 0`.

**Oracle inequality (weighting view).** Weight the classes by `w(k)`, `Σₖ w(k) ≤ 1` (e.g. `1/N` for
finitely many, `6/(π²k²)` for a countable family), spend `w(k)δ` of the failure probability on class
`k`. A union bound gives, with probability `1 − δ`, simultaneously over all `k` and `h ∈ Sₖ`,
`|R(h) − R_emp(h)| ≤ εₖ(ℓ, w(k)δ)`. SRM returning `argmin_k[R_emp(ĥₖ) + εₖ]` then satisfies

    R(ĥ) ≤ min_k { min_{h∈Sₖ} R(h) + 2·εₖ(ℓ, w(k)δ) },

i.e. it is competitive with the best class as if the right complexity were known in advance. For VC
classes with `hₖ` growing in `k`, the model-selection cost is dominated by the capacity term — selection
is essentially free.

## Why not parameter-counting (AIC/MDL/Occam)

Those penalize the number of parameters; the bound's penalty is in `h`. The two diverge: a one-parameter
family `sign(sin(αx))` has infinite VC dimension, while large-margin hyperplanes in high dimension have
VC dimension far below their parameter count. The capacity penalty is the right one and is
distribution-free; read literally as "fewer parameters," Occam's razor is not always correct.

## Structure on hyperplanes → Support Vector Machine

The class of all hyperplanes in `ℝⁿ` has VC dimension `n+1`. But `Δ`-margin separating hyperplanes with
data in a ball of radius `R` have

    h ≤ min( R²/Δ², n ) + 1,

which can be `≪ n+1` when `R²/Δ²` is the active term rather than `n`. So the **margin** indexes a structure:
larger margin → smaller-VC element. The smallest-capacity element with zero empirical risk is the
**maximal-margin (optimal) hyperplane** — an SRM construction.

    minimize  ½ w·w   subject to   yᵢ(w·xᵢ + b) ≥ 1,   i = 1,…,ℓ.

Lagrangian `L = ½w·w − Σᵢ αᵢ[yᵢ(w·xᵢ+b) − 1]`, `αᵢ ≥ 0`. Stationarity: `w = Σᵢ αᵢ yᵢ xᵢ`,
`Σᵢ αᵢ yᵢ = 0`. Dual (data enters only through inner products):

    maximize  Σᵢ αᵢ − ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ (xᵢ·xⱼ)
    subject to  Σᵢ αᵢ yᵢ = 0,   αᵢ ≥ 0.

KKT complementarity `αᵢ[yᵢ(w·xᵢ+b) − 1] = 0` ⇒ only **support vectors** (points on the margin) have
`αᵢ > 0`; the decision rule is sparse. Nonlinearity for free via a kernel `K(xᵢ,xⱼ) = φ(xᵢ)·φ(xⱼ)`
(any Mercer-admissible `K`, e.g. polynomial `(x·xᵢ+1)^d` or RBF `exp(−‖x−xᵢ‖²/σ²)`): replace every
inner product by `K`, giving `f(x) = sign(Σᵢ αᵢ yᵢ K(xᵢ,x) + b)`. Generalization is controlled by the
margin-radius ratio (`R²/Δ²`), not the raw feature-space dimension. Non-separable data: soft margin
`min ½w·w + CΣᵢ ξᵢ`, `yᵢ(w·xᵢ+b) ≥ 1 − ξᵢ`, `ξᵢ ≥ 0`, with `C` the structuring dial trading empirical risk against margin/capacity. With multipliers `αᵢ ≥ 0` for the margin constraints and `μᵢ ≥ 0` for `ξᵢ ≥ 0`, stationarity gives `w = Σᵢαᵢyᵢxᵢ`, `Σᵢαᵢyᵢ = 0`, and `C − αᵢ − μᵢ = 0`; complementarity gives `αᵢ[yᵢ(w·xᵢ+b) − 1 + ξᵢ] = 0` and `(C − αᵢ)ξᵢ = 0`, hence the soft-margin dual is

    maximize  Σᵢ αᵢ − ½ Σᵢ Σⱼ αᵢ αⱼ yᵢ yⱼ K(xᵢ,xⱼ)
    subject to  Σᵢ αᵢ yᵢ = 0,   0 ≤ αᵢ ≤ C.

## Implementation artifact

```python
import numpy as np

def empirical_risk(predict, X, y, loss):
    return np.mean([loss(yi, predict(xi)) for xi, yi in zip(X, y)])

def confidence_term(h, ell, eta):
    # C0 = sqrt((h(ln(2 ell/h)+1) - ln eta)/ell): grows with VC dim h, shrinks with ell.
    return np.sqrt((h * (np.log(2.0 * ell / h) + 1.0) - np.log(eta)) / ell)

def bounded_loss_upper_bound(R_emp, h, ell, eta, B=1.0):
    eps = 4.0 * (h * (np.log(2.0 * ell / h) + 1.0) - np.log(eta)) / ell
    return R_emp + 0.5 * B * eps * (1.0 + np.sqrt(1.0 + 4.0 * R_emp / (B * eps)))

def guaranteed_risk(R_emp, h, ell, eta):
    return R_emp + confidence_term(h, ell, eta)        # the bound SRM minimizes

class Structure:
    """Nested family S_1 subset S_2 subset ..., h_1 <= h_2 <= ... ."""
    def elements(self):
        # yield (class, vc_dimension), ordered by increasing capacity
        raise NotImplementedError

def srm_select(structure, X, y, loss, eta=0.05, use_relative_bound=False, B=1.0):
    ell = len(y)
    best = None
    for S_k, h_k in structure.elements():
        f_k = S_k.fit_erm(X, y, loss)                  # ERM within S_k
        R_emp = empirical_risk(f_k, X, y, loss)
        if use_relative_bound:
            bound = bounded_loss_upper_bound(R_emp, h_k, ell, eta, B=B)
        else:
            bound = guaranteed_risk(R_emp, h_k, ell, eta)
        if best is None or bound < best[0]:
            best = (bound, f_k)                        # smallest guaranteed risk wins
    return best[1]

def svm_dual(X, y, kernel, solve_qp, C=np.inf, tol=1e-8):
    # max sum a_i - 1/2 sum_ij a_i a_j y_i y_j K(x_i,x_j)
    # s.t. sum_i a_i y_i = 0, 0 <= a_i <= C   (margin = smallest-VC element of the structure)
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    n = len(y)
    K = np.array([[kernel(X[i], X[j]) for j in range(n)] for i in range(n)])
    Q = (y[:, None] * y[None, :]) * K
    alpha = solve_qp(Q=Q, c=-np.ones(n), Aeq=y[None, :], beq=np.array([0.0]),
                     lb=np.zeros(n), ub=np.full(n, C))
    sv = alpha > tol                                   # support vectors: only these have a_i>0
    free = sv & (alpha < C - tol) if np.isfinite(C) else sv
    bias_set = free if np.any(free) else sv
    bias = np.mean(y[bias_set] - K[bias_set] @ (alpha * y))
    def predict(x):
        score = sum(alpha[i]*y[i]*kernel(X[i], x) for i in range(n) if sv[i]) + bias
        return 1 if score >= 0.0 else -1
    return predict
```
