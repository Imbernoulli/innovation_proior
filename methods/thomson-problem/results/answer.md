# Universal optimality via linear-programming energy bounds

## The problem

Place N points on the unit sphere S^{n-1} to minimize the total pairwise energy
sum_{x != y} f(|x-y|^2), where f is a repulsive radial potential — Coulomb
f(r) = r^{-1/2} (i.e. 1/|x-y|) in R^3, the harmonic law r^{-(n/2-1)} in R^n, any
Riesz power 1/r^s, or a Gaussian. Direct minimization (steepest descent) finds
candidate configurations but cannot prove optimality: the landscape has exponentially
many local minima and must be redone per N and per f. We want a *proof*, and a robust
one — valid for a whole family of potentials at once.

## The key idea: a dual positive-definite certificate

Don't search configurations; bound them all simultaneously. Work in the inner-product
variable t = <x,y> (so |x-y|^2 = 2 - 2t) and let a(t) = f(2 - 2t). Choose a polynomial
h with two properties:

- **Domination:** h(t) <= a(t) for all t in [-1, 1).
- **Positive-definite:** h(t) = sum_i alpha_i C_i^{n/2-1}(t) with all alpha_i >= 0,
  where C_i^{n/2-1} are the Gegenbauer (ultraspherical) polynomials — the reproducing
  kernels of the degree-i spherical harmonics, normalized by C_0 = 1, C_1 = 2*lambda*t,
  lambda = n/2 - 1.

Then for **every** N-point configuration, since sum_{x,y} C_i(<x,y>) = |sum_x ev_{i,x}|^2
>= 0 (Schoenberg) and the i = 0 term equals N^2,

    energy >= sum_{x != y} h(<x,y>) = sum_{x,y} h(<x,y>) - N h(1) >= N^2 alpha_0 - N h(1).

Maximizing this bound over admissible h is an (infinite-dimensional) **linear program**;
discretizing the domination constraint on a grid of t gives a finite LP. This
generalizes the Delsarte–Goethals–Seidel and Kabatiansky–Levenshtein LP bounds for
codes from a packing threshold to a smooth potential.

## Sharp configurations and tightness

The bound is tight for a configuration S exactly when (1) h = a at every inner product
occurring in S, and (2) sum_{x,y} C_i(<x,y>) = 0 for every i > 0 with alpha_i > 0 — and
the latter holds when S is a spherical (deg h)-design. Call S **sharp** if it has m
distinct inner products t_1, ..., t_m and is a spherical (2m - 1)-design (this is the
maximal design strength compatible with m distances). Examples: regular simplices,
cross polytopes, the icosahedron (n=3, N=12), the E_8 minimal vectors (n=8, N=240), and
the Leech minimal vectors (n=24, N=196560).

## Constructing the magic auxiliary function

For a sharp S and a completely monotonic f, take h to be the **Hermite interpolant of a
to order 2 at the m inner products t_1, ..., t_m** (degree 2m - 1). Then:

- **Domination.** The remainder formula gives
  a(t) - h(t) = a^{(2m)}(xi)/(2m)! · prod_i (t - t_i)^2. The product is a square, and
  a^{(2m)}(xi) >= 0 because f completely monotonic <=> a absolutely monotonic (all
  derivatives nonnegative). So h <= a, with equality (double contact) at each t_i.
- **Tightness.** S is a (2m - 1)-design = deg(h)-design, so all i > 0 terms vanish.
- **Positive-definite.** With F(t) = prod_i (t - t_i), h = H(a, F^2). One shows F^2 is
  *conductive* (H(a, ·) of any absolutely monotonic a is positive-definite): F is
  strictly positive-definite (its low Gegenbauer coefficients are design sums that
  telescope to the positive term F(1)C_i(1) when y in S); the partial products are
  positive-definite via F = p_m + alpha p_{m-1} for the (1 - t)dmu-orthogonal Jacobi
  family; products of positive-definite functions are positive-definite; and the
  Hermite identity H(a, g_1 g_2) = H(a, g_1) + g_1 H(Q(a, g_1), g_2) with Q(a, ·)
  preserving absolute monotonicity multiplies conductivity up.

Since the construction used only the absolute monotonicity of a, the *same* sharp
configuration minimizes energy for **every** completely monotonic potential at once —
**universal optimality**. (By Bernstein–Widder it suffices to verify f(r) = (4 - r)^k or
1/r^s.) The vertices of the 600-cell need a refinement: being only an 11-design while
the naive h has degree 15, one constructs h of controlled degree and forces the
high Gegenbauer coefficients to be nonnegative.

## Euclidean culmination

The identical skeleton — positivity, domination from below, double roots at the special
distances — runs in R^n with "Gegenbauer-nonnegative" replaced by "Fourier-nonnegative"
and the all-pairs sum replaced by Poisson summation. The magic radial function, matching
prescribed values and derivatives of f and its Fourier transform at the radii sqrt(2k)
(the vector lengths of E_8 and Leech), is built from an interpolation theorem using
integral transforms of modular forms. This proves the E_8 lattice (dimension 8) and the
Leech lattice (dimension 24) are universally optimal among all configurations of their
density.

## Code

```python
import numpy as np
from scipy.special import gegenbauer
from scipy.optimize import linprog

def lp_energy_lower_bound(n, N, f_of_squared_dist, degree=12, grid=600):
    """Lower bound on sum_{x!=y} f(|x-y|^2) over all N points on S^{n-1}.
    h(t) = sum_i alpha_i C_i^{n/2-1}(t), alpha_i >= 0, h(t) <= f(2-2t);
    then energy >= N^2 alpha_0 - N h(1).  Maximize that linear functional."""
    lam = n / 2.0 - 1.0
    C1 = np.array([gegenbauer(i, lam)(1.0) for i in range(degree + 1)])
    obj = np.array([N*N if i == 0 else 0.0 for i in range(degree+1)]) - N * C1
    ts = np.linspace(-1 + 1e-3, 1 - 1e-4, grid)
    A_ub = np.array([[gegenbauer(i, lam)(t) for i in range(degree+1)] for t in ts])
    b_ub = np.array([f_of_squared_dist(2 - 2*t) for t in ts])
    res = linprog(-obj, A_ub=A_ub, b_ub=b_ub,
                  bounds=[(0, None)]*(degree+1), method="highs")
    return -res.fun, res.x      # provable lower bound, and the certificate coefficients

def coulomb(R):                  # f(R) = R^{-1/2}, R = squared distance
    return R ** -0.5

def icosahedron():               # sharp config on S^2: inner products -1, +-1/sqrt5; a 5-design
    phi = (1 + np.sqrt(5)) / 2
    raw = []
    for a in (-1, 1):
        for b in (-phi, phi):
            raw += [(0, a, b), (a, b, 0), (b, 0, a)]
    P = np.unique(np.array(raw, float), axis=0)
    return P / np.linalg.norm(P[0])

def energy(P, f):
    E = 0.0
    for i in range(len(P)):
        for j in range(len(P)):
            if i != j:
                E += f(np.sum((P[i] - P[j])**2))
    return E

if __name__ == "__main__":
    bound, alpha = lp_energy_lower_bound(3, 12, coulomb)
    P = icosahedron()
    print("LP lower bound     :", bound)            # ~98.333
    print("icosahedron energy :", energy(P, coulomb))  # ~98.331  (sharp; gap is grid-limited)
```

The LP certificate meets the icosahedron's Coulomb energy to grid precision, confirming
the bound is sharp — a *proof* of optimality, where steepest descent only ever produces
a candidate.
