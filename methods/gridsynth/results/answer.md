# Gridsynth: optimal ancilla-free Clifford+T approximation of z-rotations

## Problem

Given an angle θ and precision ε > 0, find a single-qubit Clifford+T circuit `U` with `‖Rz(θ) − U‖ ≤ ε` (operator norm), `Rz(θ) = diag(e^{-iθ/2}, e^{iθ/2})`, of **minimal T-count** — using no ancillas, measurements, or distillation. The T-count is the cost metric because on a fault-tolerant machine Clifford gates are cheap and the `T = diag(1, e^{iπ/4})` gate is expensive.

## Key idea

A 2×2 unitary is exactly Clifford+T-representable iff all entries lie in `D[ω] = Z[1/√2, i]` (Kliuchnikov–Maslov–Mosca), and exact synthesis gives a minimal-T circuit whose T-count is `2k − 2`, where `k` is the least denominator exponent of the top-left entry `u`. So approximate `Rz(θ)` by
`U = [[u, −t†], [t, u†]]`, `u, t ∈ D[ω]`, `u†u + t†t = 1`,
and **minimize `k`**. The reduction:

1. **Closeness ⇒ a grid problem.** With `z = e^{-iθ/2}`, `‖Rz(θ) − U‖² = 2 − 2 Re(z†u)`, so `‖Rz(θ) − U‖ ≤ ε` ⟺ `⟨z⃗, u⃗⟩ ≥ 1 − ε²/2`. This carves a thin sliver `R_ε` (the **ε-region**) out of the unit disk. Unitarity forces `u` and its √2-conjugate `u•` both into the unit disk. So the necessary condition on `u` is a **two-dimensional grid problem**: find `u ∈ D[ω]` with `u ∈ A = R_ε` and `u• ∈ B = disk`, smallest denominator exponent first.

2. **Grid operators solve it efficiently.** A special grid operator `G` (real-linear, `G(Z[ω]) ⊆ Z[ω]`, `det = ±1`) preserves solutions: `u` solves `(A,B)` iff `Gu` solves `(G(A), G•(B))`. Enclosing `A, B` in ellipses and running a **skew-reduction** iteration (each step one fixed special grid operator that drops the skew `b² + β²` by a constant factor) yields, in `O(log(1/M))` operations, a `G` making both upright — then upright problems reduce to two 1-D grid problems over `Z[√2]`.

3. **Existence of `t` ⇒ a Diophantine equation.** For each candidate `u`, solve `t†t = ξ`, `ξ = 1 − u†u ∈ D[√2]`. Writing `ξ•ξ = n/2^ℓ`, this is solvable iff `ξ` is doubly positive and every prime `p ≡ 7 (mod 8)` dividing `n` occurs to even multiplicity. Constructive given the **factorization of `n`** (the only hard step).

4. **Optimality.** With a factoring oracle (Shor), the first solvable candidate is **absolutely T-count-optimal**. Without one, accept only candidates whose `n` is prime — and since every such `n ≡ 1 (mod 8)`, it is always solvable. This costs an expected `O(log(1/ε))` extra candidates and an additive `O(log log(1/ε))` in T-count over the second-to-optimal solution. For generic angles (`tan(θ/2) ∉ Q(√2)`) the result is the information-theoretic `3 log₂(1/ε) + O(log log(1/ε))`; only `tan(θ/2) ∈ Q(√2)` angles reach `4 log₂(1/ε)`. Expected runtime `O(polylog(1/ε))`, oracle or not.

## Algorithm

```
Given θ, ε:  A = ε-region,  B = unit disk
  Compute one special grid operator G making G(A), G•(B) upright (skew-reduction).
  For k = 0, 1, 2, …:
    For each u ∈ (1/√2^k)Z[ω] with u ∈ A, u• ∈ B  (scaled 2-D grid problem):
      ξ = 1 − u†u;  write ξ•ξ = n/2^ℓ;  try to factor n; solve t†t = ξ.
      On success: U = [[u, −t†],[t, u†]];  U' = T U T†;  exact-synthesize whichever
                  has smaller T-count;  output and stop.
```

## Code

```python
import mpmath
from rings import ZRootTwo, ZOmega, DRootTwo, DOmega   # ring arithmetic, conj (†), conj_sq2 (•), denomexp
from to_upright import to_upright_set_pair             # skew-reduction -> grid operator G + upright bboxes
from tdgp import solve_TDGP                             # scaled 2-D grid problem at fixed k
from diophantine import diophantine_dyadic, Result      # solve t†t = ξ given a factoring of n
from synthesis_of_cliffordT import decompose_domega_unitary  # KMM exact synthesis -> minimal-T Clifford+T
from domega_unitary import DOmegaUnitary


class EpsilonRegion:
    """Slice of the unit disk where ⟨z⃗, u⃗⟩ ≥ 1 − ε²/2, with z = e^{-iθ/2}."""
    def __init__(self, theta, epsilon):
        self.theta, self.epsilon = mpmath.mpf(theta), mpmath.mpf(epsilon)
        self.d  = mpmath.sqrt(1 - epsilon**2 / 4)         # chord depth (cosine form)
        self.zx = mpmath.cos(-theta / 2)
        self.zy = mpmath.sin(-theta / 2)
        # enclosing ellipse D = R · diag(64/ε⁴, 4/ε²) · Rᵀ, centered at d·z⃗
        R  = mpmath.matrix([[self.zx, -self.zy], [self.zy, self.zx]])
        S  = mpmath.matrix([[64 * (1/epsilon)**4, 0], [0, 4 * (1/epsilon)**2]])
        p  = mpmath.matrix([self.d * self.zx, self.d * self.zy])
        self.ellipse = Ellipse(R @ S @ R.T, p)

    def inside(self, u):                                  # in disk AND past the chord
        cos_sim = self.zx * u.real + self.zy * u.imag
        return DRootTwo.fromDOmega(u.conj * u) <= 1 and cos_sim >= self.d

    def intersect(self, u0, v):                           # {t : u0 + t v ∈ R_ε} as an interval
        a, b, c = (v.conj*v).real, (2*v.conj*u0).real, (u0.conj*u0).real - 1
        t = solve_quadratic(a, b, c)                      # disk boundary
        if t is None: return None
        t0, t1 = t
        vz  = self.zx*v.real + self.zy*v.imag             # chord half-plane
        rhs = self.d - self.zx*u0.real - self.zy*u0.imag
        if vz > 0:   return (max(t0, rhs/vz), t1)
        if vz < 0:   return (t0, min(t1, rhs/vz))
        return (t0, t1) if rhs <= 0 else None


class UnitDisk:
    def inside(self, u):
        return DRootTwo.fromDOmega(u.conj * u) <= 1
    def intersect(self, u0, v):
        a, b, c = (v.conj*v).real, (2*v.conj*u0).real, (u0.conj*u0).real - 1
        return solve_quadratic(a, b, c)


def gridsynth(theta, epsilon):
    theta, epsilon = mpmath.mpf(theta), mpmath.mpf(epsilon)
    A = EpsilonRegion(theta, epsilon)                     # u ∈ A   (closeness)
    B = UnitDisk()                                        # u• ∈ B  (Diophantine solvability)

    # one skew-reduction: grid operator G making G(A), G•(B) upright (computed once)
    transformed = to_upright_set_pair(A, B)
    tdgp_sets = (A, B, *transformed)

    k = 0
    while True:                                           # increasing least denominator exponent
        for u in solve_TDGP(*tdgp_sets, k):               # u ∈ D[ω], u ∈ A, u• ∈ B
            if (u * u.conj).residue == 0:                 # already present at exponent < k
                continue
            xi = 1 - DRootTwo.fromDOmega(u.conj * u)       # ξ = 1 − u†u ∈ D[√2]
            t = diophantine_dyadic(xi)                     # solve t†t = ξ  (Result on failure)
            if not isinstance(t, Result):
                U = DOmegaUnitary(u, t, n=0)               # [[u, −t†], [t, u†]]
                return U
        k += 1


def gridsynth_gates(theta, epsilon):
    U = gridsynth(theta, epsilon)                          # the optimal D[ω]-unitary
    circuit = decompose_domega_unitary(U, wires=[0])       # KMM exact synthesis, minimal T-count = 2k−2
    return circuit.to_simple_str()                         # e.g. "HTHTSHT...SWWW"
```

```python
# Diophantine solver  t†t = ξ,  ξ ∈ D[√2],  via factoring  n = (ξ•ξ)·2^ℓ
def diophantine_dyadic(xi):
    if xi < 0 or xi.conj_sq2 < 0:                          # doubly positive is necessary
        return Result.NO_SOLUTION
    # clear dyadic denominator with δ = 1+ω (δ†δ = λ√2 ∼ √2): reduce to ξ' ∈ Z[√2]
    xi_int, twist = clear_denominator(xi)                  # ξ' = √2^ℓ ξ,  ξ'•ξ' = n ∈ Z
    t = adj_decompose(xi_int)                              # per prime p of n:
    if isinstance(t, Result):                              #   p=2 or p≡1,3,5 mod 8 -> solvable
        return Result.NO_SOLUTION                          #   p≡7 mod 8 odd power -> NO_SOLUTION
    return fix_up_units(t, xi, twist)                      #   (uses √-1, √±2 mod p; Pollard-rho factor n)
```

```python
# skew-reduction: while the pair is skewed, apply ONE special grid operator that drops the skew,
# folding the case analysis with the X (swap) and Z (sign) symmetries.
def to_upright_ellipse_pair(ellA, ellB):
    state = normalize_pair(ellA, ellB)                     # (D, Δ): SPD, det 1, off-diagonals b, β
    Gl = Gr = GridOp.identity()
    while True:
        if state.B.b < 0:                       G = OP_Z;          # fold sign of β
        elif state.A.bias * state.B.bias < 1:   G = OP_X;          # swap to equalize biases
        elif bias_far_from_one(state):          state = shift(state, n); ...  # σ shear: bias -> ~1
        elif state.skew <= 15:                  return Gl * Gr      # upright enough -> done
        elif both_biases_near_one(state):       G = OP_R            # rotate out the tilt
        elif state.A.b >= 0 and small_bias(state): G = OP_K         # asymmetric corner
        elif state.A.b >= 0:                    G = OP_A ** n        # shear [[1,−2],[0,1]]ⁿ
        else:                                   G = OP_B ** n        # shear [[1,√2],[0,1]]ⁿ
        state, Gr = apply(state, G), G * Gr                          # accumulate
```

**Up to a phase.** Global phase is unobservable, and it suffices to try only `λ ∈ {1, e^{iπ/8}}` (Clifford+T determinants are discrete, so any optimal phase snaps to `e^{inπ/8}` and `ω^k` absorbs into `U` at no T-cost). Run the `λ = 1` algorithm (even T-counts) and a twin with `U = [[u, −t†ω⁻¹],[t, u†ω⁻¹]]` for `λ = e^{iπ/8}` (odd T-counts), interleave in increasing T-count, and return the smaller — implemented as `_gridsynth_up_to_phase`, which alternates `has_phase` and steps `k` so that the two ε-region/disk pairs (scaled by `ZRootTwo(2, ±1)`) are searched in T-count order.
