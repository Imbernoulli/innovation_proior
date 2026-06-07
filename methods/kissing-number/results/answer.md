# The Delsarte LP bound for kissing numbers

## Problem

`tau_n` is the maximum number of non-overlapping unit spheres that can touch a central unit sphere in `R^n`. Normalizing contact directions to the unit sphere, this is the maximum `N` such that there exist unit vectors `x_1, ..., x_N` in `R^n` with `<x_i, x_j> <= 1/2` for all `i != j` (touching spheres are at least `60` degrees apart). So `tau_n = A(n, 1/2)`. Lower bounds come from constructions (lattice minimal vectors); the hard half is the upper bound — certifying that no larger configuration exists.

## Key idea

Encode a configuration by its Gram matrix `M = [<x_i, x_j>]`: ones on the diagonal, off-diagonal `<= 1/2`, positive semidefinite, rank `<= n`. Apply a polynomial `f` to the entries and sum all entries two ways.

The **Gegenbauer (ultraspherical) polynomials** `P_k^{(n)}` (recurrence `(k+n-2)P_{k+1} = (2k+n-2) t P_k - k P_{k-1}`, `P_0=1`, `P_1=t`, normalized `P_k^{(n)}(1)=1`) are the zonal spherical harmonics in one variable. By the **addition theorem**, `P_k^{(n)}(<x,y>) = (omega_n/m) sum_l S_{k,l}(x) S_{k,l}(y)` for an orthonormal harmonic basis `{S_{k,l}}`, so for any configuration

```
sum_{i,j} P_k^{(n)}(<x_i,x_j>) = (omega_n/m) sum_l ( sum_i S_{k,l}(x_i) )^2 >= 0.
```

Each Gegenbauer polynomial is a positive-definite kernel on the sphere.

## The bound (Delsarte–Goethals–Seidel; Kabatianskii–Levenshtein)

Let `f(t) = sum_{k=0}^d c_k P_k^{(n)}(t)` satisfy
- **(A1)** `f(t) <= 0` for all `t` in `[-1, 1/2]`;
- **(A2)** `c_0 > 0` and `c_k >= 0` for `k >= 1`.

Then `tau_n = A(n, 1/2) <= f(1) / c_0`.

*Proof.* Count `S = sum_{i,j} f(<x_i,x_j>)`. Lower: `S = sum_k c_k sum_{i,j} P_k^{(n)} >= c_0 N^2` (drop the `k>=1` terms, each `>= 0`). Upper: `S = N f(1) + sum_{i != j} f(<x_i,x_j>) <= N f(1)` (off-diagonal terms `<= 0` by (A1)). Hence `c_0 N^2 <= N f(1)`, so `N <= f(1)/c_0`.

Minimizing `f(1)/c_0` over admissible `f` of fixed degree is a **linear program** over the Gegenbauer coefficients after normalizing `c_0 = 1`. A grid version of `f(t) <= 0` on `[-1,s]` is a useful way to search for a candidate; the final certificate still needs the continuous sign condition, either by exact factor/sign reasoning or by reliable interval verification.

## Tightness reads off the polynomial

For the bound to be exact both inequalities must be equalities: `f` must vanish at every inner product occurring between distinct vectors of the optimal configuration (upper-tight), and the harmonic sums must vanish for every positive nonconstant Gegenbauer coefficient in the certificate (lower-tight). In the tight `E_8` and Leech cases this is the relevant spherical-design condition. So the optimal `f`'s roots are *dictated by the candidate configuration*. Interior roots get even multiplicity (so `f` only touches `0` from below, preserving (A1)); the window endpoints `-1` and `1/2` get simple roots.

- **`n = 8`, `E_8`:** normalized minimal vectors have inner products `{-1, -1/2, 0, 1/2}`. Take
  `f_8(t) = (t+1)(t+1/2)^2 t^2 (t-1/2)`. Then `f_8 <= 0` on `[-1,1/2]`, its dimension-`8` Gegenbauer coefficients are all `>= 0` (`c_0 = 3/320`), and `f_8(1)/c_0 = (9/4)/(3/320) = 240`. With `tau_8 >= 240` from `E_8`: **`tau_8 = 240`**.
- **`n = 24`, Leech:** inner products `{-1, -1/2, -1/4, 0, 1/4, 1/2}`. Take
  `f_24(t) = (t+1)(t+1/2)^2 (t+1/4)^2 t^2 (t-1/4)^2 (t-1/2)`. Coefficients nonnegative; with this normalization `f_24(1)=2025/1024`, `c_0=15/1490944`, and `f_24(1)/c_0 = 196560`. With the Leech lower bound: **`tau_24 = 196560`**.

## Levenshtein's universal polynomials

The universal LP polynomial has a closed form via Jacobi-polynomial zeros. With `(alpha,beta) = (a+(n-3)/2, b+(n-3)/2)`, `a,b in {0,1}`, largest zeros `t_k^{a,b}`, and intervals `I_{2k-1}=[t_{k-1}^{1,1}, t_k^{1,0}]`, `I_{2k}=[t_k^{1,0}, t_k^{1,1}]`, the polynomial `f_m^{(n,s)}(t) = (t-s)(T_{k-1}^{1,0})^2` (`m=2k-1`) or `(t+1)(t-s)(T_{k-1}^{1,1})^2` (`m=2k`) satisfies (A1),(A2) for `s in I_m` and yields the bound `A(n,s) <= L_m(n,s)`, with e.g.
`L_{2k-1}(n,s) = binom(k+n-3,k-1)[ (2k+n-3)/(n-1) - (P_{k-1}^{(n)}(s)-P_k^{(n)}(s))/((1-s)P_k^{(n)}(s)) ]`.
At `s=1/2`: `L_6(8,1/2)=240`, `L_10(24,1/2)=196560`. Within the degree ranges where the universal theorem applies, this is the best pure-LP bound: first up to degree `m`, and in the strengthened form up to `m+2`.

## Where pure LP stalls — Musin's relaxation

For `n = 3` the Levenshtein bound is `L_5(3,1/2) ~ 13.285`, and known higher-degree pure-LP improvements only reach about `13.18` (`> 13`); for `n = 4` pure LP cannot do better than `25` (Arestov–Babenko: optimal). The obstruction is a flexible/non-tight optimum, so `f` cannot vanish on the needed range. Relax (A1): allow `f > 0` near `t = -1`, requiring only `f <= 0` on `[t_0, 1/2]` (`-1 <= t_0 < -1/2`) and `f` decreasing on `[-1, t_0]`. For each fixed point, only the near-antipodal neighbors with inner product `<= t_0` can contribute positive off-diagonal terms; at most `mu` such neighbors fit in that cap. This gives
`tau_n <= max{h_0, ..., h_mu}/c_0`, with `h_m = max[ f(1) + sum_{j=1}^m f(<e_1, y_j>) ]` over `m` points satisfying `<e_1,y_j> <= t_0` and pairwise `<y_i,y_j> <= 1/2`. The `h_m` are nonconvex sub-optimizations. With `t_0=-0.5907`, `mu=4`, and a degree-`9` polynomial, the `n=3` upper bound is `< 13`; integrality plus the icosahedron's `12` contacts gives `tau_3 = 12`. With `t_0=-0.608`, `mu=6`, and a degree-`9` polynomial, the `n=4` bound matches the `24`-cell and gives `tau_4 = 24`.

## Lower bounds and intermediate dimensions (e.g. `n = 11`)

Constructions supply the lower-bound side. From a binary `(n,M,d)` code `C`, **Construction A** (integer vectors reducing mod `2` to `C`) and **Construction B** (even weight, sum divisible by `4`) yield lattices whose minimal-vector contact counts are read from the code's weight distribution. Construction A gives `2^d A_d(x)` contacts if `d < 4`, `2n + 16A_4(x)` if `d = 4`, and `2n` if `d > 4`; Construction B gives `2^{d-1}A_d(x)` if `d < 8`, `2n(n-1)+128A_8(x)` if `d = 8`, and `2n(n-1)` if `d > 8`. Cross-sections, laminations, and code concatenation propagate records below `n=24`; in `n=24`, Construction B reproduces the even Leech. In `n=11` the laminated lattice `Lambda_11` gives `tau_11 >= 582` against an upper bound near `870` — an open intermediate case where the record advances by better constructions, not by the certificate.

## Code

```python
import numpy as np
from numpy.polynomial import polynomial as P
from scipy.optimize import linprog

def gegenbauer_n(n, kmax):
    """Dimension-n ultraspherical polynomials, normalized G_k(1)=1, by recurrence."""
    G = [np.array([1.0])]
    if kmax >= 1:
        G.append(np.array([0.0, 1.0]))
    for k in range(1, kmax):
        tGk = np.concatenate([[0.0], G[k]])
        gkm1 = np.zeros(len(tGk)); gkm1[:len(G[k-1])] = G[k-1]
        G.append(((2*k + n - 2)*tGk - k*gkm1) / (k + n - 2))
    return G[:kmax+1]

def gegenbauer_coeffs(fpoly, n):
    """Expand f (power-basis coeffs) in the Gegenbauer basis: returns c_0..c_d."""
    f = np.array(fpoly, dtype=float); d = len(f) - 1
    G = gegenbauer_n(n, d); c = np.zeros(d+1)
    for k in range(d, -1, -1):
        c[k] = f[k] / G[k][k]
        f[:k+1] -= c[k] * G[k][:k+1]
    return c

def polynomial_from_roots(roots_with_mult):
    poly = np.array([1.0])
    for r, m in roots_with_mult:
        for _ in range(m):
            poly = P.polymul(poly, [-r, 1.0])
    return poly

def sampled_lp_candidate(n, degree, s=0.5, grid=801):
    """Grid-relaxed LP search over f(t)=sum c_k G_k(t), normalized by c_0=1."""
    G = gegenbauer_n(n, degree)
    t = np.linspace(-1.0, s, grid)
    A = np.column_stack([P.polyval(t, g) for g in G])
    objective = np.ones(degree + 1)
    bounds = [(1.0, 1.0)] + [(0.0, None)] * degree
    res = linprog(objective, A_ub=A, b_ub=np.zeros(grid), bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    return res.fun, res.x

def certificate_bound(fpoly, n, s=0.5, grid=2001):
    """Numerically screen the Delsarte conditions and return f(1)/c_0."""
    poly = np.array(fpoly, dtype=float)
    c = gegenbauer_coeffs(poly, n)
    assert c[0] > 0 and np.all(c[1:] >= -1e-9), "Gegenbauer coefficients must be nonnegative"
    t = np.linspace(-1.0, s, grid)
    assert np.max(P.polyval(t, poly)) <= 1e-8, "f must be nonpositive on [-1, s]"
    return P.polyval(1.0, poly) / c[0]

print(certificate_bound(polynomial_from_roots(
    [(-1,1),(-0.5,2),(0.0,2),(0.5,1)]), 8))                              # 240.0
print(certificate_bound(polynomial_from_roots(
    [(-1,1),(-0.5,2),(-0.25,2),(0.0,2),(0.25,2),(0.5,1)]), 24))           # 196560.0
```
