Across percolation and statistical mechanics, the same empirical pattern keeps appearing: quantities that increase with the configuration tend to be positively correlated. In a ferromagnetic Ising model, knowing one spin is up makes its neighbours more likely to be up; in independent bond percolation, knowing one region is well connected makes another connection event more likely. Griffiths' inequality and Harris's inequality captured these facts, but each proof was bolted to its own model, spin products and ferromagnetic couplings on one side and independent edge variables on the other. The task is to strip away the physics and find the minimal structural assumptions that force increasing functions to correlate.

The obstacle is that configuration spaces are only partially ordered. On a totally ordered set, positive correlation of increasing functions follows from the Chebyshev rearrangement inequality for any positive measure, because every pair of points is comparable and the differences move in the same direction. But on a partial order, incomparable pairs can make individual summands negative, so the measure itself must do work. The right abstraction turns out to be a finite distributive lattice, subsets under inclusion, products of chains, or any coordinatewise order closed under meet and join, together with a log-supermodular measure.

The FKG inequality, named for Fortuin, Kasteleyn and Ginibre, states that on a finite distributive lattice Γ, if a positive measure μ satisfies the FKG lattice condition μ(x ∧ y) μ(x ∨ y) ≥ μ(x) μ(y) for every x, y, then any two increasing real functions f and g are positively correlated: ⟨fg⟩ ≥ ⟨f⟩⟨g⟩. Equivalently, in log form with λ = log μ, the condition is supermodularity λ(x ∧ y) + λ(x ∨ y) ≥ λ(x) + λ(y). This makes precise the idea that the measure does not anti-associate coordinates: adding an element is at least as beneficial when more is already present. Product measures sit exactly on the boundary, satisfying the condition with equality, so Harris's inequality for independent percolation drops out immediately. By encoding attractive spin systems through an appropriate λ, Griffiths-type inequalities also follow.

The proof proceeds by induction on the length of the lattice. The base case of a chain is the Chebyshev argument: symmetrizing the covariance gives a sum of nonnegative products. For the inductive step, one fixes an atom a and splits the lattice into the part above a, the part not above a, and the latter shifted up by joining with a. The cross terms between the two halves reduce to showing that an increasing function has a larger average above the atom than off it. This monotonicity of averages is established using the inductive hypothesis twice: once with a decreasing indicator of the shifted region, and once with the increasing density ratio that the FKG condition guarantees. Closing the induction gives the theorem.

The FKG condition is not only sufficient but also cleanly unifies the motivating examples. In independent bond percolation the edge measure is a product, so it satisfies the lattice condition with equality and increasing connection events are positively correlated. In the random-cluster model the factor κ^{c(R)} is supermodular in the number of components for κ ≥ 1, so the same conclusion holds, and at κ = 2 the Ising correlations reappear as special cases. Thus a single theorem with two hypotheses, a distributive lattice and a log-supermodular measure, explains why good events help each other across statistical mechanics.

```python
"""
FKG inequality: numerical illustration on a finite distributive lattice.

Theorem: on a finite distributive lattice Γ, if μ > 0 satisfies
    μ(x ∧ y) * μ(x ∨ y) >= μ(x) * μ(y)   for all x, y,
then for any two increasing functions f, g:
    <fg> >= <f><g>,   where <f> = Σ_x μ(x) f(x) / Σ_x μ(x).
"""

from itertools import combinations
import math


def boolean_lattice(n):
    """Return the Boolean lattice P({0, ..., n-1}) as frozensets."""
    return [frozenset(s) for r in range(n + 1)
            for s in combinations(range(n), r)]


def meet(x, y):
    return x & y


def join(x, y):
    return x | y


def is_increasing(f, lattice):
    """Check f(x) <= f(y) whenever x ⊆ y."""
    for i, x in enumerate(lattice):
        for y in lattice[i + 1:]:
            if x <= y and f[x] > f[y]:
                return False
    return True


def check_fkg_condition(mu, lattice, eps=1e-12):
    """Verify μ(x∧y) μ(x∨y) >= μ(x) μ(y) up to tolerance."""
    for x in lattice:
        for y in lattice:
            lhs = mu[x] * mu[y]
            rhs = mu[meet(x, y)] * mu[join(x, y)]
            if rhs < lhs - eps:
                return False, (x, y, lhs, rhs)
    return True, None


def covariance(mu, f, g, lattice):
    Z = sum(mu[x] for x in lattice)
    mean_f = sum(mu[x] * f[x] for x in lattice) / Z
    mean_g = sum(mu[x] * g[x] for x in lattice) / Z
    mean_fg = sum(mu[x] * f[x] * g[x] for x in lattice) / Z
    return mean_fg - mean_f * mean_g


def make_log_supermodular_measure(n):
    """
    Build a strictly positive log-supermodular measure on P({0,...,n-1}).
    λ(R) = Σ_{r∈R} (r+1) + 0.5 |R|^2 is supermodular because the second
    difference for adding two distinct elements is 2 * 0.5 = 1 >= 0.
    """
    lattice = boolean_lattice(n)
    mu = {}
    for R in lattice:
        linear = sum(r + 1 for r in R)
        quad = 0.5 * len(R) ** 2
        mu[R] = math.exp(linear + quad)
    return lattice, mu


def make_counterexample():
    """
    A positive measure on the 2-element Boolean lattice that violates the
    FKG condition and makes two increasing functions anti-correlated.
    """
    empty, a, b, ab = frozenset(), frozenset([0]), frozenset([1]), frozenset([0, 1])
    lattice = [empty, a, b, ab]
    mu = {empty: 1.0, a: 10.0, b: 10.0, ab: 1.0}
    f = {empty: 0, a: 1, b: 0, ab: 1}  # indicator of containing element 0
    g = {empty: 0, a: 0, b: 1, ab: 1}  # indicator of containing element 1
    return lattice, mu, f, g


if __name__ == "__main__":
    # --- FKG-satisfying example on the Boolean lattice of size 3 ---
    n = 3
    lattice, mu = make_log_supermodular_measure(n)

    ok, counter = check_fkg_condition(mu, lattice)
    print("FKG condition satisfied:", ok)
    if not ok:
        print("Counterexample:", counter)

    # Two increasing functions: lattice size and sum of contained elements.
    f = {R: float(len(R)) for R in lattice}
    g = {R: float(sum(R) if R else 0) for R in lattice}
    print("f increasing:", is_increasing(f, lattice))
    print("g increasing:", is_increasing(g, lattice))

    cov = covariance(mu, f, g, lattice)
    print(f"Covariance <fg> - <f><g> = {cov:.6f}")
    assert cov >= -1e-9, "FKG inequality violated!"

    # --- Counterexample: when the FKG condition fails, so can the conclusion ---
    lattice_bad, mu_bad, f_bad, g_bad = make_counterexample()
    ok_bad, _ = check_fkg_condition(mu_bad, lattice_bad)
    cov_bad = covariance(mu_bad, f_bad, g_bad, lattice_bad)
    print("\nCounterexample FKG condition satisfied:", ok_bad)
    print(f"Counterexample covariance = {cov_bad:.6f}")
    assert not ok_bad, "Counterexample should violate the FKG condition"
    assert cov_bad < -1e-3, "Counterexample should show negative covariance"
```
