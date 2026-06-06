# Gridsynth: optimal ancilla-free Clifford+T approximation of z-rotations

## Problem

Given an angle θ and precision ε > 0, find a single-qubit Clifford+T circuit `U` with `‖Rz(θ) − U‖ ≤ ε` (operator norm), `Rz(θ) = diag(e^{-iθ/2}, e^{iθ/2})`, of **minimal T-count** — using no ancillas, measurements, or distillation. The T-count is the cost metric because on a fault-tolerant machine Clifford gates are cheap and the `T = diag(1, e^{iπ/4})` gate is expensive.

## Key idea

A 2×2 unitary is exactly Clifford+T-representable iff all entries lie in `D[ω] = Z[1/√2, i]` (Kliuchnikov–Maslov–Mosca), and exact synthesis gives a minimal-T circuit whose T-count can be chosen as `2k − 2` for `k > 0` (and 0 for `k = 0`), where `k` is the least denominator exponent of the top-left entry `u`. So approximate `Rz(θ)` by
`U = [[u, −t†], [t, u†]]`, `u, t ∈ D[ω]`, `u†u + t†t = 1`,
and **minimize `k`**. The reduction:

1. **Closeness ⇒ a grid problem.** With `z = e^{-iθ/2}`, `‖Rz(θ) − U‖² = 2 − 2 Re(z†u)`, so `‖Rz(θ) − U‖ ≤ ε` ⟺ `⟨z⃗, u⃗⟩ ≥ 1 − ε²/2`. This carves a thin sliver `R_ε` (the **ε-region**) out of the unit disk. Unitarity forces `u` and its √2-conjugate `u•` both into the unit disk. So the necessary condition on `u` is a **two-dimensional grid problem**: find `u ∈ D[ω]` with `u ∈ A = R_ε` and `u• ∈ B = disk`, smallest denominator exponent first.

2. **Grid operators solve it efficiently.** A special grid operator `G` (real-linear, `G(Z[ω]) ⊆ Z[ω]`, `det = ±1`) preserves solutions: `u` solves `(A,B)` iff `Gu` solves `(G(A), G•(B))`. Enclosing `A, B` in ellipses and running a **skew-reduction** iteration (each step one fixed special grid operator that drops the skew `b² + β²` by a constant factor) yields, in `O(log(1/M))` operations, a `G` making both upright — then upright problems reduce to two 1-D grid problems over `Z[√2]`.

3. **Existence of `t` ⇒ a Diophantine equation.** For each candidate `u`, solve `t†t = ξ`, `ξ = 1 − u†u ∈ D[√2]`. Writing `ξ•ξ = n/2^ℓ`, this is solvable iff `ξ` is doubly positive and every prime `p ≡ 7 (mod 8)` dividing `n` occurs to even multiplicity. Constructive given the **factorization of `n`** (the only hard step).

4. **Optimality.** With a factoring oracle (Shor), the first solvable candidate is **absolutely T-count-optimal**. Without one, the implementation uses bounded factoring effort and accepts any candidate it can decide; the analysis can conservatively rely on prime `n`, because every prime `n ≡ 1 (mod 8)` makes the Diophantine equation solvable. Under the mild prime-distribution hypothesis this costs an expected `O(log(1/ε))` extra candidates and an additive `O(log log(1/ε))` in T-count over the second-to-optimal solution. The counting picture gives the typical `3 log₂(1/ε) + O(log log(1/ε))` scale for unstructured angles; `tan(θ/2) ∈ Q(√2)` is the structured regime that can force the `4 log₂(1/ε)` worst-case scale. Expected runtime is `O(polylog(1/ε))` with a factoring oracle, and under the bounded-factoring/probabilistic hypothesis without one.

## Algorithm

```
Given θ, ε:  A = ε-region,  B = unit disk
  Compute one special grid operator G making G(A), G•(B) upright (skew-reduction).
  For k = 0, 1, 2, …:
    For each u ∈ (1/√2^k)Z[ω] with u ∈ A, u• ∈ B  (scaled 2-D grid problem):
      ξ = 1 − u†u;  write ξ•ξ = n/2^ℓ;  try to factor n; solve t†t = ξ.
      On success: choose between t and ωt by the smaller denominator exponent of u+t
                  versus u+ωt; exact-synthesize the resulting unitary and stop.
```

## Code

```python
import mpmath
from rings import omega, adj, adj2, real, denomexp
from gridproblems import ConvexSet, Ellipse, unitdisk, gridpoints2_increasing
from diophantine import diophantine_dyadic, run_bounded, Success
from exact_synthesis import synthesis_u2


def epsilon_region(epsilon, theta):
    zx, zy = mpmath.cos(-theta / 2), mpmath.sin(-theta / 2)
    d = 1 - epsilon**2 / 2
    rot = mpmath.matrix([[zx, -zy], [zy, zx]])
    shape = mpmath.matrix([[4 * (1 / epsilon)**4, 0],
                           [0, (1 / epsilon)**2]])
    ellipse = Ellipse(rot @ shape @ special_inverse(rot), (d * zx, d * zy))

    def contains(point):
        x, y = point.real, point.imag
        return x*x + y*y <= 1 and zx*x + zy*y >= d

    def intersect(p, v):
        disk = solve_quadratic(v.dot(v), 2 * v.dot(p), p.dot(p) - 1)
        if disk is None:
            return empty_interval()
        t0, t1 = disk
        vz = zx * v.x + zy * v.y
        rhs = d - (zx * p.x + zy * p.y)
        if vz == 0:
            return (t0, t1) if rhs <= 0 else empty_interval()
        cut = rhs / vz
        return (max(t0, cut), t1) if vz > 0 else (t0, min(t1, cut))

    return ConvexSet(ellipse, contains, intersect)


def gridsynth_stats(rng, prec_bits, theta, effort):
    digits = mpmath.ceil(15 + 2 * prec_bits * mpmath.log10(2))
    return with_fixed_precision(digits, gridsynth_internal, rng, prec_bits, theta, effort)


def gridsynth_internal(rng, prec_bits, theta, effort):
    epsilon = 2 ** (-prec_bits)
    region = epsilon_region(epsilon, theta)
    candidates = gridpoints2_increasing(region, unitdisk())

    for u in candidates:
        rng_try, rng = split_rng(rng)
        xi = real(1 - adj(u) * u)
        status, t = run_bounded(effort, diophantine_dyadic(rng_try, xi))
        if status == Success:
            return choose_completion(u, t), status
        record_candidate(u, status)


def choose_completion(u, t):
    if denomexp(u + t) < denomexp(u + omega * t):
        return matrix2x2((u, -adj(t)), (t, adj(u)))
    return matrix2x2((u, -adj(omega * t)), (omega * t, adj(u)))


def gridsynth_gates(rng, prec_bits, theta, effort):
    U, _ = gridsynth_stats(rng, prec_bits, theta, effort)
    return synthesis_u2(U)
```

**Up to a phase.** Global phase is unobservable, and it suffices to try only `λ ∈ {1, e^{iπ/8}}` (Clifford+T determinants are discrete, so any optimal phase snaps to `e^{inπ/8}` and `ω^k` absorbs into `U` at no T-cost). Run the `λ = 1` algorithm and a twin with `U = [[u, −t†ω⁻¹],[t, u†ω⁻¹]]` for `λ = e^{iπ/8}`, then return the smaller T-count; the near-optimality comparison is against the third-to-optimal solution because the two phase branches can own the first two candidates.
