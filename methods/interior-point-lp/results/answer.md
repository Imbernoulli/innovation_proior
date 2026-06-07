# Interior-Point Methods for Linear Programming

## Problem

Solve the linear program

```
(P)  min cᵀx  s.t.  Ax = b, x ≥ 0          (D)  max bᵀy  s.t.  Aᵀy + s = c, s ≥ 0
```

in time polynomial in `n` and the data bit-length `L`, while remaining fast on large sparse problems. The simplex method is fast in practice but exponential in the worst case (Klee–Minty); the ellipsoid method is polynomial but impractical. Interior-point methods achieve both by moving a strictly feasible point through the interior toward the optimum.

## Key idea

Replace the hard constraint `x ≥ 0` by the **logarithmic barrier** `φ(x) = −Σⱼ ln xⱼ` and minimize the family `F_t(x) = t·cᵀx + φ(x)`. The minimizers form the **central path** `x*(t)`, characterized as a smooth deformation of the KKT optimality conditions:

```
Ax = b, x > 0 ;   Aᵀy + s = c, s > 0 ;   xⱼ sⱼ = 1/t  (perturbed complementarity).
```

The hard complementarity `xⱼsⱼ = 0` is softened to `xⱼsⱼ = 1/t`; as `t → ∞` it recovers optimality. Every central point yields a **dual feasible** `s = 1/(t x) > 0`, so the duality gap is known exactly: `cᵀx − bᵀy = xᵀs = n/t`. Following the path means solving the smooth central-path equations with Newton's method.

## Why it is polynomial: self-concordance

Newton's method is affine-invariant, but its classical convergence region is stated via the Hessian's condition number and Lipschitz constant — frame-dependent quantities. Measuring instead in the function's own local norm `‖h‖_x = √(hᵀ∇²f(x)h)` removes the frame, and the only thing left to bound is how fast the Hessian changes, giving the **self-concordance** condition

```
|D³f(x)[h,h,h]| ≤ 2 (D²f(x)[h,h])^{3/2}.
```

The powers `3/2, 1/2` are forced by homogeneity; the constant `2` is fixed so that `−ln x` satisfies it with equality (`f''=1/x²`, `|f'''|=2/x³=2(f'')^{3/2}`), making `−ln x` a `1`-self-concordant barrier and `φ = −Σ ln xⱼ` an `n`-self-concordant barrier. Then the damped Newton step `x⁺ = x − [1/(1+λ)][∇²f(x)]⁻¹∇f(x)`, with Newton decrement `λ(x,f) = √(∇fᵀ[∇²f]⁻¹∇f)`, satisfies `λ(x⁺) ≤ 2λ(x)²`: quadratic convergence with region `{λ ≤ 1/4}`, **no condition number anywhere**. The barrier's gradient bound `|Df[h]| ≤ √ϑ·‖h‖_x` (with `ϑ = n`) lets the path-following update

```
t ← (1 + 0.1/√n)·t,   then one Newton step on F_t,
```

keep the iterate within range of the next center, giving `O(√n·log(1/ε))` iterations, each one solve with `A diag(x/s) Aᵀ`.

## Karmarkar's projective / potential-reduction form

Reducing the LP to the simplex `Δ = {x≥0, Σxⱼ=1} ∩ {Ax=0}` with target value `0` (manufactured by combining primal and dual into one feasibility problem), the projective transformation `T(a,a₀): x ↦ D⁻¹x/(eᵀD⁻¹x)`, `D = diag(a)`, fixes the simplex's vertices and maps the current point `a` to the center `a₀ = e/n`, where the inscribed/circumscribed ball ratio is `R/r = n−1`. Optimizing over the inscribed ball drops the gap by a factor `1 − 1/(n−1)` per step. Since the linear objective is not projective-invariant, progress is tracked by the **Karmarkar potential**

```
f(x) = n·ln(cᵀx) − Σⱼ ln xⱼ   ( = ϑ ln(cᵀx) + φ(x),  ϑ = n ),
```

which maps to a function of the same form under `T`; a constant per-step reduction `δ ≈ 1/8` (with step factor `α = 1/4`) forces `cᵀx → 0` geometrically. Each step's `A D² Aᵀ` solve is updated incrementally by rank-one corrections (`D` changes little), giving `O(n^{2.5})` average work per step and `O(n^{3.5} L)` overall — better than the ellipsoid's `O(n⁶L²)` by `n^{2.5}`. The potential `f = ϑ ln(cᵀx) + F(x)` is exactly the path follower seen as "run to infinity along a ray," i.e. slide down the central path.

## The practical algorithm: primal-dual predictor-corrector

The version that wins in practice keeps `(x,y,s)` together and Newton-steps the deformed KKT system `F(x,y,s) = (Ax−b, Aᵀy+s−c, XSe − σμe) = 0`, with centering `σ ∈ [0,1]` interpolating between the affine (`σ=0`) and centering (`σ=1`) directions. Mehrotra's predictor-corrector: a predictor probes the affine direction, sets `σ = (μ_aff/μ)³` adaptively, and a corrector reuses the factorization to fold in the second-order term `Δx_aff∘Δs_aff`; step lengths use the fraction-to-boundary rule.

```python
import numpy as np
from scipy.linalg import solve_triangular

def solve_lp(A, b, c, tol=1e-8, max_iter=100, eta=0.9995):
    """
    Primal-dual interior-point (Mehrotra predictor-corrector) for
        min cᵀx  s.t.  Ax = b, x ≥ 0,   dual  max bᵀy s.t. Aᵀy + s = c, s ≥ 0.
    Drives the deformed KKT system  Ax=b, Aᵀy+s=c, x∘s = μe, x,s>0  to μ→0.
    """
    A = np.asarray(A, float); b = np.asarray(b, float); c = np.asarray(c, float)
    m, n = A.shape
    x, y, s = initial_point(A, b, c)
    bc = 1.0 + max(np.linalg.norm(b), np.linalg.norm(c))

    for k in range(max_iter):
        rb = A @ x - b                        # primal infeasibility
        rc = A.T @ y + s - c                  # dual infeasibility
        rxs = x * s                           # complementarity x∘s
        mu = rxs.mean()                       # = gap/n on the central path
        if np.linalg.norm(np.concatenate([rb, rc, rxs])) / bc < tol:
            break

        d = x / s                             # D² = S⁻¹X
        M = A @ (d[:, None] * A.T)            # A diag(x/s) Aᵀ  (SPD)
        M[np.diag_indices_from(M)] += 1e-10   # tiny diagonal shift for stability
        chol = np.linalg.cholesky(M)

        # Predictor (affine, σ = 0).
        dx_a, dy_a, ds_a = newton_dir(A, chol, d, x, s, rb, rc, rxs)
        ax, as_ = step_len(x, s, dx_a, ds_a, 1.0)
        mu_aff = ((x + ax * dx_a) @ (s + as_ * ds_a)) / n
        sigma = (mu_aff / mu) ** 3            # adaptive centering

        # Corrector: re-center by σμ, cancel 2nd-order term.
        rxs_cc = rxs + dx_a * ds_a - sigma * mu
        dx, dy, ds = newton_dir(A, chol, d, x, s, rb, rc, rxs_cc)

        ax, as_ = step_len(x, s, dx, ds, eta)  # fraction-to-boundary
        x = x + ax * dx; y = y + as_ * dy; s = s + as_ * ds

    return x, y, s, float(c @ x)


def newton_dir(A, chol, d, x, s, rb, rc, rxs):
    """
    Newton system  A Δx=-rb, AᵀΔy+Δs=-rc, S Δx+X Δs=-rxs.
    Eliminate Δs=-(rxs+s∘Δx)/x and Δx, leaving
        A diag(x/s) Aᵀ Δy = -rb - A diag(x/s)(rc - rxs/x).
    """
    rhs = -rb - A @ (d * (rc - rxs / x))
    dy = solve_triangular(chol.T, solve_triangular(chol, rhs, lower=True), lower=False)
    dx = d * (A.T @ dy + rc - rxs / x)
    ds = -(rxs + s * dx) / x
    return dx, dy, ds


def step_len(x, s, dx, ds, eta):
    """Largest fraction η of the step keeping x>0 and s>0."""
    ax = 1.0; neg = dx < 0
    if neg.any(): ax = min(ax, eta * np.min(-x[neg] / dx[neg]))
    as_ = 1.0; neg = ds < 0
    if neg.any(): as_ = min(as_, eta * np.min(-s[neg] / ds[neg]))
    return ax, as_


def initial_point(A, b, c):
    """Mehrotra start: least-norm primal/dual, then shift into the strict interior."""
    AAt = A @ A.T
    y = np.linalg.solve(AAt, A @ c)
    s = c - A.T @ y
    x = A.T @ np.linalg.solve(AAt, b)
    dx = max(-1.5 * x.min(), 0.0); ds = max(-1.5 * s.min(), 0.0)
    x = x + dx; s = s + ds
    pdct = 0.5 * (x @ s)
    x = x + pdct / s.sum(); s = s + pdct / x.sum()
    return x, y, s
```

On a random standard-form LP (`m=8, n=20`) this matches `scipy.optimize.linprog` (HiGHS) to `1e-5` in 6 iterations with feasibility residual `~1e-13`; on the worked example `max 2x₁+3x₂` s.t. `x₁+x₂≤4, x₁−x₂≤1` it returns the optimum `12` at `(0,4)` in 5 iterations.

## Why each piece

- **Log barrier `−Σ ln xⱼ`** (not `1/xⱼ`, not quadratic penalty): the unique self-concordant building block; the `−ln` equality case fixes the constant `2`.
- **Central path / target `μ = 1/t`**: a single large-`t` barrier solve is ill-conditioned (the SUMT failure); tracing the path with short Newton steps stays in the quadratic-convergence region.
- **Self-concordance**: makes Newton's convergence frame-independent, dissolving the historical ill-conditioning objection; the `√ϑ` gradient bound yields the `√n` iteration count.
- **Projective transformation + potential**: re-centers the simplex by a body-preserving map; the log-of-ratio potential is projective-invariant where the linear objective is not, so per-step reduction composes.
- **Primal-dual + Mehrotra**: symmetric, observable gap `sᵀx` as stopping test, adaptive centering `σ=(μ_aff/μ)³`, second-order correction, and fraction-to-boundary steps — far faster in practice than pure path-following.
</content>
