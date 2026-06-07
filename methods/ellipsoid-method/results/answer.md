# The Ellipsoid Method

## Problem

Decide, in time polynomial in the encoding length `L` of the data, whether a system of `m` linear inequalities `A x ≤ b` in `n` variables with **integer** coefficients has a real solution — and, by reduction, solve linear programs `max cᵀx s.t. A x ≤ b` in polynomial time.

## Key idea

Keep an ellipsoid guaranteed to contain the target body. Query the **center**: if it satisfies the current feasibility tolerance, accept; otherwise a violated row `A_ℓ` certifies that every admissible point lies in the half-space through the center with normal `A_ℓ`, hence in a **half-ellipsoid**. Re-enclose that half-ellipsoid in the minimum-volume ellipsoid and repeat. Each step is a **rank-one update** of the positive-definite shape matrix, and the volume shrinks by a fixed factor `e^{−1/(2(n+1))}` that depends only on the dimension. Integer data supplies (i) a starting ball of radius `2^L` that must contain a solution if one exists, and (ii) a `2^{−L}` residual margin for infeasible systems. Choosing `ε < 2^{−L}` lets the algorithm search the relaxed body `{x : A x ≤ b + ε}`: infeasible systems still have an empty relaxed body, while feasible systems gain a full-dimensional volume floor `2^{−O(nL)}`. That gives the `O(n²L)` iteration bound and an exact decision.

## The algorithm

Represent the ellipsoid as `E = { z : (z − x)ᵀ A⁻¹ (z − x) ≤ 1 }`, `A ≻ 0`, `vol(E) = √(det A)·V_n`.

- **Start.** `x₀ = 0`, `A₀ = (2^L)² I` (the ball `‖x‖ ≤ 2^L`; Lemma 1: any solution lives here).
- **Step.** At center `x`, evaluate `σ(x) = max_i (A_i x − b_i)`. For the exact central-cut step, if `σ(x) ≤ 0`, return `x`; for the integer decision wrapper, accept when `σ(x) ≤ ε` with `ε < 2^{−L}`. Otherwise let `a = A_ℓ` for a row whose residual exceeds the active tolerance. With
  ```
  b   = A a / √(aᵀ A a)
  x⁺  = x − 1/(n+1) · b
  A⁺  = n²/(n²−1) · ( A − 2/(n+1) · b bᵀ )
  ```
  the new ellipsoid `E⁺ = E(A⁺, x⁺)` is the minimum-volume ellipsoid containing the active half-ellipsoid `E ∩ { aᵀ(z−x) ≤ 0 }`, which contains the current target body (`P` in the exact step, `P_ε` in the decision wrapper), and `vol(E⁺)/vol(E) = (n/(n+1))·(n²/(n²−1))^{(n−1)/2} ≤ e^{−1/(2(n+1))}`.
- **Stop.** Run `M = O(n²L)` (e.g. `16 n² L`) iterations. The volume argument is applied to `P_ε = {x : A x ≤ b + ε}`. If the original system is feasible, `P_ε` has positive volume at least `2^{−O(nL)}` in the bounded search region, and every update preserves `P_ε ⊆ E_k`; if no center is accepted before `vol(E_k)` falls below that floor, those statements contradict each other. If the original system is infeasible, Lemma 2 gives `σ(x) ≥ 2^{−L}` everywhere, so the choice `ε < 2^{−L}` prevents a false acceptance.

**Rational arithmetic.** The `√` makes exact entries irrational with growing bit-length, which would break polynomiality. Round every entry to `p = O(nL)` bits, but first add outward slack: in the `Q`-matrix form, multiply the exact central-cut axes by a small cushion such as `2^{1/(2n²)}` before rounding; in the `A = Q Qᵀ` form, this is a scalar outward factor on the shape matrix. The cushion dominates the possible inward error from rounding, so containment survives, and the volume ratio remains below one. Total: `O(n²(n²+m)L)` operations at `O(nL)`-bit precision, `O(nm+n²)` numbers of `O(nL)` bits in memory.

## Separation And Optimization

The loop only needs a **separation oracle** with centered semantics: given the current center `x`, certify membership or return a normal `a` such that every admissible `y` satisfies `aᵀ(y−x) ≤ 0`. For an explicit system this is a row scan; for an implicit system the same localization loop runs with oracle calls in place of row scans. Optimization reduces to feasibility by binary-searching `d` and testing `P ∩ {cᵀx ≥ d}`. The intersection oracle first asks the separation oracle for `P`; if `x ∈ P` but `cᵀx < d`, it returns the violated objective cut with normal `−c`. Lemma 1 bounds the search interval by `[-2^L‖c‖, 2^L‖c‖]` for a finite integer LP optimum, and determinant bounds give the bit precision needed to recover the rational optimum.

## Working code

```python
import math
import numpy as np

def as_integer_system(Arows, brhs):
    """Validate integer data for Arows x <= brhs."""
    Arows = np.asarray(Arows, dtype=float)
    brhs = np.asarray(brhs, dtype=float).reshape(-1)
    if Arows.ndim != 2 or brhs.shape[0] != Arows.shape[0]:
        raise ValueError("expected Arows with shape (m, n) and brhs with shape (m,)")
    if Arows.shape[0] == 0:
        raise ValueError("expected at least one inequality")
    if not np.all(np.isfinite(Arows)) or not np.all(np.isfinite(brhs)):
        raise ValueError("all coefficients must be finite")
    Aint = np.rint(Arows)
    bint = np.rint(brhs)
    if not np.array_equal(Arows, Aint) or not np.array_equal(brhs, bint):
        raise ValueError("this decision wrapper expects integer coefficients")
    return Aint.astype(float), bint.astype(float)

def encoding_length(Arows, brhs):
    """Binary encoding length for an integer system Arows x <= brhs."""
    Arows, brhs = as_integer_system(Arows, brhs)
    s = sum(math.log2(abs(int(a)) + 1) for a in np.ravel(Arows))
    s += sum(math.log2(abs(int(bi)) + 1) for bi in np.ravel(brhs))
    s += math.log2(max(1, Arows.shape[0] * Arows.shape[1]))
    return int(math.ceil(s)) + 1

def residual(Arows, brhs, x):
    """sigma(x) = max_i (A_i @ x - b_i)."""
    return float(np.max(np.asarray(Arows, dtype=float) @ np.asarray(x, dtype=float) - brhs))

def separation_for_system(Arows, brhs, tol=0.0):
    """Return a center-valid cut normal, or None inside the residual tolerance."""
    Arows, brhs = as_integer_system(Arows, brhs)

    def separate(x):
        x = np.asarray(x, dtype=float)
        slack = Arows @ x - brhs
        row = int(np.argmax(slack))
        if slack[row] <= tol:
            return None
        return Arows[row].copy()

    return separate

def decide_feasibility(separate, n, R, max_iters, inflate=1.0):
    """Central-cut ellipsoid feasibility from a separation oracle.

    The oracle returns None when the center is accepted; otherwise it returns a
    normal a such that every feasible y satisfies a @ (y - x) <= 0.
    """
    if n < 2:
        raise ValueError("the central-cut update below assumes n >= 2")
    x = np.zeros(n, dtype=float)
    A = (float(R) ** 2) * np.eye(n)
    scale = n**2 / (n**2 - 1)

    for _ in range(max_iters):
        a = separate(x)
        if a is None:
            return x

        a = np.asarray(a, dtype=float)
        Aa = A @ a
        q = float(a @ Aa)
        if q <= 0:
            return None

        b = Aa / np.sqrt(q)
        x = x - b / (n + 1)
        A = inflate * scale * (A - (2.0 / (n + 1)) * np.outer(b, b))
        A = 0.5 * (A + A.T)

    return None

def decide_linear_inequalities(Arows, brhs, max_iters=None, tol=None):
    """Decide integer LP feasibility for Arows x <= brhs.

    With the default tolerance, a returned center may be epsilon-feasible
    rather than exactly feasible; the integer residual gap makes the status
    exact in the bounded-precision decision model.
    """
    Arows, brhs = as_integer_system(Arows, brhs)
    n = Arows.shape[1]
    L = encoding_length(Arows, brhs)
    R = 2.0**L
    max_iters = 16 * n * n * L if max_iters is None else max_iters
    tol = 0.5 * 2.0 ** (-L) if tol is None else tol
    sep = separation_for_system(Arows, brhs, tol=tol)
    x = decide_feasibility(sep, n, R, max_iters)
    return ("feasible", x) if x is not None else ("infeasible", None)

def maximize_linear(base_separate, c, n, L, tol=None):
    """Maximize c.x over P by bisection, using only centered separation for P."""
    c = np.asarray(c, dtype=float)
    R = 2.0**L
    max_iters = 16 * n * n * L
    tol = 2.0 ** (-L) if tol is None else tol
    bound = R * float(np.linalg.norm(c))
    if bound == 0.0:
        return decide_feasibility(base_separate, n, R, max_iters), 0.0
    lo, hi = -bound, bound
    best = None

    def with_objective_floor(d):
        def separate(x):
            cut = base_separate(x)
            if cut is not None:
                return cut
            if float(c @ x) < d:
                return -c
            return None

        return separate

    while hi - lo > tol:
        d = 0.5 * (lo + hi)
        x = decide_feasibility(with_objective_floor(d), n, R, max_iters)
        if x is None:
            hi = d
        else:
            lo, best = d, x
    return best, lo
```

The shrink factor is slow and structure-blind, but it gives the missing worst-case guarantee: linear-program feasibility, and therefore linear programming, is decidable in time polynomial in the encoding length.
