# Context: bounding the kissing number

## Research question

How many non-overlapping unit spheres can simultaneously touch ("kiss") a central unit sphere in `R^n`? Call this number `tau_n`. Two touching outer spheres have centers at distance `>= 2`, and each center lies at distance exactly `2` from the central sphere's center; projecting the contact points onto the central sphere, the problem becomes: place as many unit vectors `x_1,...,x_N` on the sphere `S^{n-1}` as possible so that the angle between any two is at least `60` degrees, i.e.

```
<x_i, x_j> <= 1/2   for all i != j.
```

So `tau_n = A(n, 1/2)`, where `A(n, s)` is the maximum number of unit vectors on `S^{n-1}` with pairwise inner product at most `s`. The question splits cleanly into two halves with completely different characters. A **lower bound** `tau_n >= N` is *constructive*: exhibit `N` vectors that obey the angle constraint and you are done. An **upper bound** `tau_n <= M` is the hard half: you must certify that *no* configuration of `M+1` vectors can exist, ruling out a continuum of possibilities at once. The dispute between Newton (who said `tau_3 = 12`) and Gregory (`13`) in 1694 is exactly the difficulty of the upper bound made concrete — the lower bound `tau_3 >= 12` is a drawing; the upper bound `tau_3 <= 12` took until 1953.

What a satisfying solution would need: a single, dimension-independent *machine* that takes the constraint `<x_i,x_j> <= 1/2` and emits a finite numerical certificate `tau_n <= M`, ideally one tight enough to meet a known construction so the answer is pinned exactly.

## Background

The contact configurations that matter are the short vectors of remarkable lattices. In `n = 8`, the root lattice `E_8` (all integer vectors with coordinates all even or all odd and coordinate-sum divisible by `4`) has `240` shortest nonzero vectors, all of the same length, with minimum distance equal to that length — rescaled to the unit sphere they kiss, giving `tau_8 >= 240`. Concretely the `240` are the `112` vectors of type `(0^6, +-2^2)` and the `128` of type `(+-1^8)` with an even number of minus signs. In `n = 24`, the Leech lattice has `196560` shortest vectors (types `(0^16, +-2^8)` even sign count: `97152`; `(0^22, +-4^2)`: `1104`; `(+-1^23, -+3)`: `98304`), all of equal length, giving `tau_24 >= 196560`. In `n = 3` the icosahedron's `12` vertices give `tau_3 >= 12`; in `n = 4` the `24`-cell (the `D_4` minimal vectors) gives `tau_4 >= 24`.

The deep structural fact that makes upper bounds tractable lives on the sphere, not in the lattice. Functions on `S^{n-1}` decompose into spherical harmonics by degree, and the **Gegenbauer (ultraspherical) polynomials** `P_k^{(n)}` are the one-variable shadow of that decomposition. For fixed `n` they are defined by `P_0^{(n)} = 1`, `P_1^{(n)} = t`, and the three-term recurrence

```
(k + n - 2) P_{k+1}^{(n)}(t) = (2k + n - 2) t P_k^{(n)}(t) - k P_{k-1}^{(n)}(t),
```

normalized so `P_k^{(n)}(1) = 1`, and orthogonal on `[-1, 1]` with respect to the weight `(1 - t^2)^{(n-3)/2}` — the weight that arises when you integrate a function of `<x, y>` over `S^{n-1}`. For `n = 3` these are the Legendre polynomials; for `n = 4`, Chebyshev polynomials of the second kind. The single load-bearing fact about them is the **addition theorem** (traceable to Herglotz, via Müller): if `{S_{k,l}}_{l=1..m}` is an orthonormal basis for the degree-`k` spherical harmonics (dimension `m = m(k,n) = binom(k+n-2, k) + binom(k+n-3, k-1)`), then

```
P_k^{(n)}(<x, y>) = (omega_n / m) * sum_{l=1}^m S_{k,l}(x) S_{k,l}(y),
```

with `omega_n` the surface area of `S^{n-1}`. This expresses `P_k^{(n)}(<x,y>)` as an inner product of feature vectors `(S_{k,1}(x), ..., S_{k,m}(x))` — a positive-definite kernel on the sphere (Schoenberg's characterization of positive-definite functions on spheres, 1942). The empirical lesson from the lattice configurations is equally important: in the candidate optima the set of inner products that *actually occur* between distinct vectors is tiny and very symmetric — for normalized `E_8` it is exactly `{-1, -1/2, 0, 1/2}`, and for Leech `{-1, -1/2, -1/4, 0, 1/4, 1/2}`. These configurations are also spherical designs (their harmonic moments vanish to high degree). Those two facts — few inner products, high design strength — are properties of the *configurations*, knowable by inspecting the lattices, and they are what a tight upper-bound certificate will have to exploit.

## Baselines

**Spherical-trigonometry case analysis (Schütte–van der Waerden 1953).** The first rigorous `tau_3 <= 12`. One studies `13` points on `S^2` with pairwise angular distance `>= 60` degrees and derives a contradiction by analyzing the induced graph of "close" pairs and the areas of spherical triangles / Delaunay-type cells, bounding total area against `4*pi`. It is correct but a heavy, ad hoc case enumeration tied to `n = 3`; it gives no leverage in higher dimensions and does not scale.

**Coxeter's bound (1963), completed by Böröczky (1978).** A general upper bound `A(n, s) <= A_{CB}(n, s) = 2 F_{n-1}(alpha) / F_n(alpha)`, where `F_n` is the Schläfli function (`F_0 = F_1 = 1`, `F_{n+1}(alpha) = (2/pi) integral F_{n-1}(beta(t)) dt`) and `alpha = (1/2) arccos( s / (1 + (n-2)s) )`. It rests on a simplex-bound conjecture proved later by Böröczky. It is genuinely general but, for the kissing case `s = 1/2`, numerically weak; it only becomes competitive when `s` is close to `1` (e.g. it gives `A(4, cos(pi/5)) = 120`, met by the `600`-cell). For pinning `tau_8` or `tau_24` it is far too loose.

**Fejes Tóth's bound (1943).** Bounds the *minimum distance* `D(n, M)` of an `M`-point code: `D(n,M) <= (4 - 1/sin^2(phi_M))^{1/2}` with `phi_M = pi M / (6(M-2))`. Attained only for `M = 3, 4, 6, 12`. It constrains distance given cardinality rather than cardinality given the angle, so it does not directly bound `tau_n`, and is sharp only at a handful of `M`.

**Constructive lower bounds via codes (Leech–Sloane 1971).** For the lower-bound half: a binary `(n, M, d)` code `C` yields a lattice (Construction A: integer vectors whose reduction mod `2` lies in `C`; Construction B adds even-weight and sum-divisible-by-`4`), whose minimal vectors give a kissing configuration. The contact count is read off from the weight distribution of `C`: for Construction A it is `2^d A_d(x)` if `d < 4`, `2n + 16 A_4(x)` if `d = 4`, and `2n` if `d > 4`; Construction B similarly gives `2^{d-1} A_d(x)` if `d < 8`, `2n(n-1) + 128 A_8(x)` if `d = 8`, and `2n(n-1)` if `d > 8`. Cross-sections and laminations then propagate configurations between dimensions, and concatenation of codes (Dodunekov–Ericson–Zinoviev) yields most record lower bounds below dimension `24`. These give `tau_n >= ...` but say nothing about the matching upper bound — leaving wide gaps in intermediate dimensions (e.g. dimension `11`, where the laminated lattice `Lambda_11` gives `tau_11 >= 582` against a much larger upper bound).

The gap these baselines leave: there is no uniform, computable upper-bound machine. The trigonometric proof is one-off; Coxeter–Böröczky is uniform but loose; Fejes Tóth bounds the wrong quantity. None of them can certify `tau_8 <= 240` to *meet* the `E_8` construction.

## Evaluation settings

The yardstick is the table of kissing numbers for `n` up to about `32`, comparing the best known *lower* bound (a construction) against the best known *upper* bound (a certificate) in each dimension. The dimensions that decide whether a method is principled: `n = 3` and `n = 4` (small, optimum non-unique, the classical hard cases); `n = 8` and `n = 24` (the `E_8` and Leech configurations, the candidates for exactness); and intermediate dimensions such as `n = 5, ..., 23` where lower and upper bounds disagree and `n = 11` in particular, where the best certificate is far above the best construction. Inner products are normalized so the kissing constraint is `<x_i, x_j> <= 1/2`. A method is judged by (a) how tight an upper bound it yields per dimension and (b) whether, in `n = 8` and `n = 24`, it meets the construction exactly.

## Code framework

Pre-existing numerical primitives: a way to generate orthogonal polynomials by their three-term recurrence, polynomial arithmetic (multiply, evaluate, change of basis), and a linear-program / convex solver. The open slot is a routine that takes a candidate one-variable polynomial and decides whether it turns into a numerical upper bound for a spherical code with angle parameter `s`.

```python
import numpy as np
from numpy.polynomial import polynomial as P

def gegenbauer_n(n, kmax):
    """Degree-0..kmax orthogonal polynomials attached to integration over S^{n-1},
    by the standard three-term recurrence, normalized to value 1 at t=1."""
    pass

def gegenbauer_coeffs(fpoly, n):
    """Re-express a polynomial (power-basis coeffs) in the ultraspherical basis.
    Generic linear algebra over the polynomial recurrence."""
    pass

def polynomial_from_roots(roots_with_mult):
    """Build a one-variable polynomial from prescribed real roots."""
    pass

def certificate_bound(fpoly, n, s=0.5):
    """Return the upper-bound value produced by a candidate polynomial."""
    # TODO: identify the coefficient and sign checks that make f(1)/c_0 valid.
    pass
```

The empty `certificate_bound` check is the whole problem: what algebraic conditions make `f(1)/c_0` a valid certificate, and how should `f` be chosen so that this number is small?
