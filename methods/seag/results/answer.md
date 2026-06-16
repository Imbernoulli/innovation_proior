# Extra Anchored Gradient (EAG), distilled

EAG is a first-order method for smooth convex-concave minimax problems
`min_x max_y L(x, y)` that makes the **squared gradient norm** small at an accelerated,
optimal `O(1/k²)` **last-iterate** rate. It combines two mechanisms that are each only
`O(1/k)` on their own: the **extragradient** look-ahead (which defeats the rotational cycling
of gradient descent-ascent and licenses a constant step size) and **anchoring** — a Halpern-style
pull back toward the fixed starting point (which forces the *last* iterate, not just the
best-so-far, to converge and implicitly selects the solution nearest the start). Run with additive
Gaussian gradient-oracle or update perturbations, the same update is the **stochastic** EAG (SEAG):
it keeps EAG's fast transient but levels off at a noise floor, because the same `k²`-weighting that
drives the acceleration also amplifies the injected noise.

## Problem it solves

`L: R^n × R^m → R` convex in `x`, concave in `y`, with saddle operator
`G(z) = [∇_x L; -∇_y L]`, `z = (x, y)`. `G` is monotone (`⟨G(z1)-G(z2), z1-z2⟩ ≥ 0`) and
`R`-Lipschitz; a saddle point is a zero of `G`. Drive `‖∇L(z)‖² = ‖G(z)‖²` to zero, fast, in the
last iterate, unconstrained. Gradient descent-ascent diverges on the bilinear `L = xy` (the
operator is a rotation); extragradient and optimistic methods are stuck at best-iterate `O(1/k)`;
anchoring alone (SimGD-A) is last-iterate but needs a diminishing step and stalls below `O(1/k)`.

## Key idea

Put the anchor inside **both** half-steps of the extragradient predictor-corrector, with a
constant step size `α` and anchoring coefficient `β_k = 1/(k+2)`:

```
z^{k+1/2} = z^k + β_k (z^0 - z^k) - α G(z^k)          (predictor / look-ahead)
z^{k+1}   = z^k + β_k (z^0 - z^k) - α G(z^{k+1/2})     (corrector; same anchor offset, uses z^k)
```

- `β_k = 0` recovers plain extragradient. The anchor offset is relative to the current point
  `z^k` in both lines (not the look-ahead `w`).
- **Why `β_k = 1/(k+2)`.** The continuous anchored flow `ż = -G(z) - β(t)(z - z^0)` with
  `β(t) = γ/t^p` has two competing speeds: contracting (the anchor pull stabilizes and kills
  cycling) and vanishing (the pull must die to converge to a zero of `G`, not to `z^0`). `p = 1`,
  i.e. `β(t) = 1/t ↔ β_k = 1/(k+2)`, balances them and gives the fastest `O(1/t²)`; `p > 1`
  vanishes too early, `p < 1` too late. On `L = xy` the anchored flow has the closed form
  `x(t) = (y^0 cos t + x^0 sin t - y^0)/t`, `y(t) = (y^0 sin t - x^0 cos t + x^0)/t`, decaying
  like `1/t` (so `‖G‖² ~ 1/t²`) — versus the EG-flavored Moreau–Yosida flow's slow
  `exp(-λt/(1+λ²))`. Discretely, `β_k = 1/(k+2)` makes the Lyapunov coefficients `B_k = k+1`
  (linear) and `A_k = α_k(k+1)(k+2)/2` (quadratic), and quadratic `A_k` is what produces `1/k²`.
- **Why a constant step is allowed** (unlike anchoring's `(1-p)/(k+1)^p`): the extragradient
  look-ahead supplies the per-step decrease that SimGD-A had to buy with a shrinking step.

## Convergence

**Varying step (EAG-V), cleanest constant.** `α_0 ∈ (0, 3/(4R))`, recurrence

```
α_{k+1} = α_k ( 1 - 1/((k+1)(k+3)) · α_k² R² / (1 - α_k² R²) ),
```

which decreases monotonically to a positive limit `α_∞` (e.g. `α_0 = 0.618/R ⇒ α_∞ ≈ 0.437/R`).
Then

```
‖∇L(z^k)‖² ≤ 4(1 + α_0 α_∞ R²)/α_∞² · ‖z^0 - z*‖² / ((k+1)(k+2)),
```

and with `α_0 = 0.618/R` the constant is `27`.

**Constant step (EAG-C), the simple form.** `α` fixed, same `β_k = 1/(k+2)`. Under
`1 - 3αR - α²R² - α³R³ ≥ 0` and `1 - 8αR + α²R² - 2α³R³ ≥ 0` (holds for `α ∈ (0, 1/(8R)]`),

```
‖∇L(z^k)‖² ≤ 4(1 + αR + α²R²)/(α²(1 + αR)) · ‖z^0 - z*‖² / (k+1)²,
```

constant `260` at `α = 1/(8R)`.

**Proof sketch.** Lyapunov function `V_k = A_k ‖G(z^k)‖² + B_k ⟨G(z^k), z^k - z^0⟩` with
`A_k = α_k B_k/(2β_k)`, `B_{k+1} = B_k/(1-β_k)`, and
`A_{k+1} = A_k(1 - α_k²R² - β_k²)/((1-α_k²R²)(1-β_k)²)`. From the EAG identities, subtract the
nonnegative monotonicity term with weight `B_k/β_k`, subtract the nonnegative Lipschitz term with
weight `A_k/(α_k²R²)`, and use the `α`-recurrence. The result is
`V_k - V_{k+1} ≥ a‖G(z^{k+1/2})‖² + b‖G(z^{k+1})‖² - 2c⟨G(z^{k+1/2}),G(z^{k+1})⟩ ≥ 0`
with `c² = ab`, so `V_k` is nonincreasing. Then `V_k ≤ V_0 = α_0‖G(z^0)‖² ≤ α_0 R²‖z^0-z*‖²`, and a
monotonicity + Young lower bound gives
`V_k ≥ (α_∞/4)(k+1)(k+2)‖G(z^k)‖² - (1/α_∞)‖z^0-z*‖²`; combining yields the rate. For EAG-C the
same skeleton becomes `V_k - V_{k+1} ≥ Tr(M_k S_k M_kᵀ)` with `M_k = [G(z^k) G(z^{k+1/2}) G(z^{k+1})]`
and a tridiagonal `S_k ⪰ 0` (PSD cone self-dual), with `A_k` kept growing quadratically.

**Optimality (matching lower bound).** For biaffine `L(x,y) = ⟨Ax - b, y - c⟩`, first-order
iterates lie in Krylov subspaces of `A`; reducing to solving `Ax = b` by matrix-vector products
and applying Nemirovsky's matrix-Chebyshev lower bound gives
`‖∇L(z^k)‖² ≥ R²‖z^0 - z*‖² / (2⌊k/2⌋+1)² = Ω(R²/k²)`. So `O(1/k²)` is optimal up to a constant.
(EAG escapes the `O(1/k)` last-iterate lower bound for the 1-SCLI class because its `β_k = 1/(k+2)`
are non-stationary.)

## Stochastic instantiation (SEAG)

Perturb either the exact operator call, `Ḡ(z) = G(z) + ξ`, or the fixed-step update line,
`... - αG(z) + η`; the two are equivalent for fixed `α` with `η = -αξ`. Because the
gradient-norm term in `V` carries the `A_k ~ k²` weight, injected noise is amplified by the same
quadratic factor that produces the acceleration — so the method has EAG's fast `O(1/k²)` transient
but the gradient norm flattens at a floor set by `σ`, and stability needs the oracle variance
controlled on the order of `1/k`; with fixed-variance noise it accumulates error and eventually
destabilizes if pushed far enough, exactly like stochastic Nesterov for convex minimization.
Reducing the floor needs a separate variance-control mechanism, not anything in the update.

## Working code

A compact constant-step EAG-C implementation, run with optional additive Gaussian update
perturbations. The state carries the iterate, the fixed anchor `z^0`, and the index `k`.

```python
import numpy as np


def init_state(problem, initial_z, hyperparameters):
    z0 = np.asarray(initial_z, dtype=float).reshape(2 * problem.dim)
    return {
        "z": z0,                 # current iterate z^k
        "anchor_z": z0.copy(),   # the anchor z^0 (fixed)
        "step_index": 0,         # k
    }


def step(state, problem, hyperparameters):
    tau = float(hyperparameters["tau"])           # constant step α
    z = state["z"]
    anchor_z = state["anchor_z"]
    k = int(state["step_index"])

    # anchoring coefficient β_k = 1/(k+2): the 1/t anchored-flow schedule
    beta = 1.0 / (k + 2.0)

    # predictor: z^{k+1/2} = z^k + β_k (z^0 - z^k) - α G(z^k) + η
    g = problem.grad(z)
    w = z + beta * (anchor_z - z) - tau * g + problem.noise()

    # corrector: z^{k+1} = z^k + β_k (z^0 - z^k) - α G(z^{k+1/2}) + η'
    gw = problem.grad(w)
    z_next = z + beta * (anchor_z - z) - tau * gw + problem.noise()

    return {"z": z_next, "anchor_z": anchor_z, "step_index": k + 1}, z_next


def get_hyperparameters(problem_name):
    # constant step τ = α
    if problem_name == "bilinear":
        return {"tau": 0.1}
    if problem_name == "delta_nu":
        return {"tau": 1.0}
    raise KeyError(f"Unknown problem: {problem_name}")
```

Deterministic EAG is the same update with `problem.noise()` returning `0`,
optionally with the EAG-V step-size schedule `α_k` in place of a fixed `τ`.

## Relation to prior methods

- **Extragradient** (Korpelevich 1977) = EAG with `β_k = 0`: defeats cycling, constant step,
  but best-iterate `O(1/k)`.
- **Halpern iteration** (Halpern 1967; Lieder 2020) `u_{k+1} = λ_{k+1} u^0 + (1-λ_{k+1}) T(u_k)`
  for nonexpansive `T`: the anchor mechanism; implicitly regularized, last-iterate `O(1/k)`
  residual with `λ_k = 1/(k+1)`.
- **SimGD-A** (Ryu, Yuan & Yin 2019): anchoring on plain gradient steps; last-iterate but
  diminishing step `(1-p)/(k+1)^p`, rate `O(1/k^{2-2p})`, never reaches `O(1/k²)`.
- EAG = extragradient `+` anchoring in both half-steps `+` constant step, which is what breaks the
  `O(1/k)` ceiling to the optimal `O(1/k²)`. The acceleration is distinct from Nesterov's (anchoring
  damps oscillation; momentum adds it).
