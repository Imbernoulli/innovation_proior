OK, let me think this through from scratch. The question is how many unit spheres can kiss a central unit sphere in `R^n`. Touching means the outer centers sit at distance `2` from the middle center and at distance `>= 2` from each other; divide everything by `2` and the contact directions are unit vectors `x_1, ..., x_N` on `S^{n-1}` with the angle between any two at least `60` degrees. The cosine of `60` is `1/2`, so the entire problem is: maximize `N` subject to `<x_i, x_j> <= 1/2` for `i != j`. Call the max `tau_n`.

There are two completely different jobs here and I should not confuse them. If I want to show `tau_n >= 240`, I just hand you `240` vectors that obey the constraint — that is a construction, and the lattices already give me those: `E_8`'s shortest vectors, the Leech lattice's shortest vectors, the icosahedron in `R^3`. The construction half is "easy" in the sense that it is an existence proof by exhibition. The murderous half is the *upper* bound. To say `tau_n <= 240` I have to rule out every possible configuration of `241` vectors — a continuum of them — with a finite argument. That asymmetry is the whole reason Newton and Gregory could argue about `tau_3` for centuries: the picture of `12` is easy, the proof that `13` is impossible is not.

So let me stare at the upper-bound half. What do I have for it right now? The honest answer is: almost nothing uniform. Schütte and van der Waerden proved `tau_3 <= 12`, but the way they did it is a hand-to-hand fight on the `2`-sphere — you take `13` points with pairwise angle `>= 60`, look at which pairs are "close" to `60`, build the graph of those pairs, triangulate, and bound the total spherical area of the cells against `4*pi`. It works, but it is a case analysis welded to `n = 3`. Push to `n = 4` and the spherical triangles become spherical tetrahedra and the bookkeeping explodes; there is no machine here, just heroic effort per dimension. Coxeter has a genuinely general bound through the Schläfli function, completed by Böröczky — `A(n,s) <= 2 F_{n-1}(alpha)/F_n(alpha)` — but when I plug in `s = 1/2` it is loose. It is good only when the angle is small, close to `s -> 1` (it nails the `600`-cell, `A(4, cos pi/5) = 120`), and useless for kissing. Fejes Tóth bounds the minimum distance for a fixed cardinality, which is the wrong direction — I want cardinality for a fixed angle. So: I have one ad hoc `n=3` proof and one uniform-but-weak bound. Neither can certify `tau_8 <= 240` to *meet* the `E_8` construction. I want a single machine that eats the constraint `<x_i,x_j> <= 1/2` and spits out a number.

Let me think about what object actually encodes a configuration. I have `N` unit vectors; stack them as columns of `X` in `R^{n x N}`. The natural object is the Gram matrix `M = X^T X`, an `N x N` matrix whose `(i,j)` entry is `<x_i, x_j>`. What do I know about `M`? Four things. Its diagonal is all ones (unit vectors). Its off-diagonal entries are all `<= 1/2` (the constraint). It is positive semidefinite (it is a Gram matrix). And it has rank at most `n` (the vectors live in `R^n`). That last one is where the dimension enters — it is the only place `n` shows up, and it had better be the lever.

Now what can I *do* with a Gram matrix? Here is the only general tool I can think of that turns "this matrix is PSD of bounded rank" into a scalar: apply some function `f` entrywise to `M` and sum all the entries. Why summing all entries? Because if I can show the entrywise-transformed matrix `[f(<x_i,x_j>)]` has a nonnegative sum of entries, and *separately* bound that same sum from above using the constraint on the off-diagonal entries, I will trap `N`. Let me try to make that concrete and see if it even has a chance.

Sum of all entries of `[f(<x_i,x_j>)]`. Split diagonal from off-diagonal:

```
sum_{i,j} f(<x_i,x_j>) = N * f(1) + sum_{i != j} f(<x_i,x_j>).
```

The diagonal gives `N f(1)` exactly. Now suppose — this is the wish — that `f(t) <= 0` for every `t` in the range where the off-diagonal inner products live, i.e. for `t` in `[-1, 1/2]`. Then every off-diagonal term is `<= 0`, so

```
sum_{i,j} f(<x_i,x_j>) <= N f(1).
```

Good, that is an *upper* bound on the total sum, and crucially it is linear in `N`. If I can independently show the total sum is at least something *quadratic* in `N`, I win: quadratic `<=` linear forces `N` to be bounded. So I need a lower bound on `sum_{i,j} f(<x_i,x_j>)` that scales like `N^2`. Where would an `N^2` come from? From the constant term. If `f` "contains" a positive multiple of the constant function `1`, then summing that piece over all `i,j` gives that multiple times `N^2`. So the plan is to write `f` as (positive constant) `*1` plus other stuff, and arrange that the "other stuff" contributes a *nonnegative* amount to the total sum so I can drop it.

That is the crux: I need a family of building-block functions `g` such that `sum_{i,j} g(<x_i,x_j>) >= 0` for *every* configuration on the sphere, and I want to expand my `f` in those building blocks with nonnegative coefficients. Which functions of the inner product have a guaranteed-nonnegative sum over any point set on the sphere? This is a positive-definiteness question. A function `g` on `[-1,1]` is positive-definite on `S^{n-1}` if the matrix `[g(<x_i,x_j>)]` is PSD for every point set — and a PSD matrix has nonnegative sum of entries (sandwich it between the all-ones vector: `1^T G 1 >= 0`). So I want positive-definite functions on the sphere, and I want a *basis* of them.

This is exactly where the geometry of `S^{n-1}` should hand me the answer. Functions of `<x,y>` that are positive-definite on the sphere are characterized by Schoenberg: they are the nonnegative combinations of the Gegenbauer (ultraspherical) polynomials `P_k^{(n)}`. Let me re-derive *why* the Gegenbauer polynomials are the right building blocks, because if I just assert it I have not understood it. The reason is the addition theorem. Decompose functions on `S^{n-1}` into spherical harmonics by degree. Let `{S_{k,l}}_{l=1}^{m}` be an orthonormal basis for the degree-`k` harmonics, `m = m(k,n)` the dimension of that space. The addition theorem says the degree-`k` zonal kernel — the rotation-invariant piece, which can only depend on `<x,y>` — is, up to a constant,

```
P_k^{(n)}(<x, y>) = (omega_n / m) * sum_{l=1}^m S_{k,l}(x) S_{k,l}(y).
```

Stare at the right-hand side. It is a sum of products `S_{k,l}(x) S_{k,l}(y)` — it is literally an inner product of the feature vector `v_k(x) = (S_{k,1}(x), ..., S_{k,m}(x))` with `v_k(y)`. So `P_k^{(n)}(<x,y>)` is a kernel `<v_k(x), v_k(y)>`. Now sum over the configuration:

```
sum_{i,j} P_k^{(n)}(<x_i, x_j>) = (omega_n/m) sum_{l=1}^m sum_{i,j} S_{k,l}(x_i) S_{k,l}(x_j)
   = (omega_n/m) sum_{l=1}^m ( sum_{i} S_{k,l}(x_i) )^2 >= 0.
```

There it is — a sum of squares, nonnegative for *any* configuration, with no constraint at all. That is the building block I needed. The Gegenbauer polynomials are not pulled out of a hat; they are the zonal harmonics, and the addition theorem makes each one a sum-of-squares-positive kernel on the sphere. (And `P_0^{(n)} = 1` is the constant function, whose sum over `i,j` is exactly `N^2` — that is my quadratic term.)

Let me also pin where the dimension lives in these polynomials, because that is the lever from rank `n`. They satisfy `P_0 = 1`, `P_1 = t`, and `(k+n-2)P_{k+1}(t) = (2k+n-2) t P_k(t) - k P_{k-1}(t)`, with `n` sitting right in the coefficients; they are orthogonal under the weight `(1-t^2)^{(n-3)/2}` on `[-1,1]`, which is exactly the measure you get pushing the uniform measure on `S^{n-1}` down to the inner-product coordinate. For `n=3` they are Legendre; for `n=4`, Chebyshev of the second kind. Different dimension, different polynomials — good, the machine is dimension-aware.

Now assemble it. Take any polynomial `f` and expand it in the Gegenbauer basis for dimension `n`:

```
f(t) = sum_{k=0}^d c_k P_k^{(n)}(t).
```

Impose two conditions. First, `c_0 > 0` and `c_k >= 0` for `k >= 1` — the nonnegative-combination condition. Second, `f(t) <= 0` for all `t` in `[-1, 1/2]` — the constraint window. Then the two-way count closes:

Lower bound on the total sum,
```
sum_{i,j} f(<x_i,x_j>) = sum_{k=0}^d c_k sum_{i,j} P_k^{(n)}(<x_i,x_j>)
   >= c_0 sum_{i,j} P_0^{(n)}(<x_i,x_j>) = c_0 N^2,
```
where I threw away every `k >= 1` term because each is a nonnegative coefficient times a nonnegative sum-of-squares. Upper bound, from the diagonal/off-diagonal split and `f <= 0` on the off-diagonal range,
```
sum_{i,j} f(<x_i,x_j>) = N f(1) + sum_{i != j} f(<x_i,x_j>) <= N f(1).
```
Chain them: `c_0 N^2 <= N f(1)`, so

```
tau_n = A(n, 1/2) <= f(1) / c_0.
```

That is the machine. Pick *any* admissible `f` — any polynomial that is a nonnegative Gegenbauer combination and sits `<= 0` on `[-1, 1/2]` — and `f(1)/c_0` is a rigorous upper bound on the kissing number. The looseness of every choice of `f` is allowed; I just want the *best* one, the one minimizing `f(1)/c_0`. And minimizing a ratio over polynomials with linear constraints (`c_k >= 0`, `f(t) <= 0` on a grid of `t`) is a linear program once I fix the degree `d` — normalize `c_0 = 1` and minimize `f(1)`, or normalize `f(1)` and maximize `c_0`. The centuries-old upper-bound problem has become: solve an LP.

Now the beautiful part: is this machine ever *tight*? Let me ask what it would take for `tau_n = f(1)/c_0` exactly, for some configuration of `N = tau_n` vectors. I traced through two inequalities; both must be equalities. The upper inequality `sum_{i != j} f(<x_i,x_j>) <= 0` is tight iff every off-diagonal term is zero, i.e. `f(<x_i,x_j>) = 0` for *every* inner product that actually occurs between distinct vectors in the configuration. The lower inequality is tight iff `c_k sum_{i,j} P_k^{(n)} = 0` for all `k >= 1`, i.e. the configuration's harmonic moments vanish up to degree `d` — it is a spherical design of strength `d`. So the magic polynomial is not something I search for blindly. If I have a candidate optimal configuration, I *read off* the set of inner products that occur, and I demand that `f` vanish exactly there. The roots of `f` are dictated by the geometry of the candidate.

Let me do `n = 8`. The candidate is the `240` shortest vectors of `E_8`, normalized to the unit sphere. By symmetry the only inner products that occur between distinct normalized roots are `{-1, -1/2, 0, 1/2}` — a tiny set, which is exactly the gift these special configurations give. So I want `f` to vanish at `-1, -1/2, 0, 1/2`. But I also need `f(t) <= 0` on the whole interval `[-1, 1/2]`, not just at those four points. If a root in the interior of the interval were simple, `f` would change sign there and become positive just past it. To keep `f <= 0` while touching zero at an interior root, that root must have *even* multiplicity — `f` kisses zero from below. The endpoints are different: at `t = -1` (left end) and `t = 1/2` (right end of the window) a simple root is fine, since `f` only needs to be `<= 0` on one side. So put simple roots at `-1` and `1/2`, double roots at the interior points `-1/2` and `0`:

```
f_8(t) = (t + 1) (t + 1/2)^2 t^2 (t - 1/2).
```

Degree `6`. Check the sign on `[-1, 1/2]`: `(t+1) >= 0`, the two squares `>= 0`, and `(t - 1/2) <= 0`, so the product is `<= 0` throughout — condition on the window holds. Now I owe the Gegenbauer-positivity check: expand `f_8` in the dimension-`8` Gegenbauer basis and confirm `c_0 > 0` and all `c_k >= 0`. There is no slick reason this has to hold; it is a property of this specific configuration that it is a tight design, and the check either passes or it does not. Expanding (the recurrence makes this mechanical), the coefficients come out all nonnegative — `c_0` turns out to be `3/320` once `f_8` is taken with leading normalization, and the ratio is what I care about. Computing `f_8(1)/c_0`: `f_8(1) = (2)(3/2)^2 (1)(1/2) = (2)(9/4)(1/2) = 9/4`, and dividing by `c_0 = 9/4 / 240` gives exactly `240`. So `tau_8 <= 240`. The `E_8` construction gives `tau_8 >= 240`. They meet: `tau_8 = 240`, exactly, with a one-line polynomial certificate where the old proofs had none.

The same recipe runs for `n = 24` and the Leech lattice. The normalized shortest vectors have one more pair of inner products — the set is `{-1, -1/2, -1/4, 0, 1/4, 1/2}`. Same logic: simple roots at the ends `-1` and `1/2`, even (double) roots at the interior values `-1/2, -1/4, 0, 1/4`:

```
f_24(t) = (t + 1)(t + 1/2)^2 (t + 1/4)^2 t^2 (t - 1/4)^2 (t - 1/2).
```

Degree `10`. On `[-1, 1/2]` it is `<= 0` by the same sign accounting. Expand in the dimension-`24` Gegenbauer basis — coefficients all nonnegative, `c_0 > 0`; with this normalization `f_24(1) = 2025/1024` and `c_0 = 15/1490944`, so `f_24(1)/c_0 = 196560`. That meets the Leech construction exactly: `tau_24 = 196560`. The fact that in dimensions `8` and `24` the candidate configurations are *so* symmetric that they have only a handful of inner products, and that they are tight designs, is exactly what makes the LP certificate snap shut.

I do not want to be re-guessing the polynomial dimension by dimension, though. Is there a closed-form `f` that the LP itself wants? Push on the structure. For a given `n` and target inner product `s` (here `s = 1/2`), a degree-limited improving polynomial should have its roots at the largest possible inner products consistent with the design/quadrature structure — and those are governed by the zeros of the Gegenbauer-adjacent **Jacobi** polynomials. Let me set up the parameters Levenshtein-style: consider Jacobi polynomials `P_k^{(alpha,beta)}` with `(alpha, beta) = (a + (n-3)/2, b + (n-3)/2)`, `a, b in {0,1}` (the Gegenbauer case is `a = b = 0`), and let `t_k^{a,b}` be the largest zero. These largest zeros interlace, `t_{k-1}^{1,1} < t_k^{1,0} < t_k^{1,1}`, and they carve `[-1,1)` into consecutive intervals `I_m`: `I_{2k-1} = [t_{k-1}^{1,1}, t_k^{1,0}]` and `I_{2k} = [t_k^{1,0}, t_k^{1,1}]`. For `s` in `I_m` the universal polynomial is

```
f_m^{(n,s)}(t) = (t - s) (T_{k-1}^{1,0}(t,s))^2                 if m = 2k-1,
              = (t + 1)(t - s) (T_{k-1}^{1,1}(t,s))^2          if m = 2k,
```

where the `T` are the Christoffel–Darboux / kernel polynomials built from the Jacobi family. This is the same shape I hand-built for `E_8` — a factor that handles the window endpoints `(t - s)` (and `(t+1)` in the even case) times a perfect square that supplies the even interior roots and the nonnegative Gegenbauer coefficients. Levenshtein proved these `f_m^{(n,s)}` satisfy both admissibility conditions for all `s` in `I_m` (in fact all their Gegenbauer coefficients are positive there), so plugging into the machine gives the universal bound `A(n,s) <= L_m(n,s)`, a clean closed form in binomials, e.g.

```
L_{2k-1}(n,s) = binom(k+n-3, k-1) [ (2k+n-3)/(n-1) - (P_{k-1}^{(n)}(s) - P_k^{(n)}(s)) / ((1-s) P_k^{(n)}(s)) ].
```

For `s = 1/2`: `tau_8 <= L_6(8,1/2) = L_7(8,1/2) = 240` and `tau_24 <= L_10(24,1/2) = L_11(24,1/2) = 196560` — the closed form recovers the two exact cases, and the hand-built `f_8, f_24` are special instances. Among improving polynomials up to degree `m`, and in the strengthened range up to `m+2`, these universal polynomials are best possible; if a gap remains there, a comparable-degree pure LP certificate will not close it.

Which is exactly the wall in low dimensions. Run the machine at `n = 3`, `s = 1/2`: the Levenshtein bound is `L_5(3,1/2) ~ 13.285`, and higher-degree pure-LP improvements only drop it to about `13.18`. That is `< 14` but not `< 13` — it cannot prove `tau_3 <= 12`. At `n = 4` I get `<= 25.55`, so `tau_4` is `24` or `25`, and it is a theorem (Arestov–Babenko) that *no* pure-LP polynomial does better than `25` here. Why does the machine stall exactly here? Trace the tightness condition again. Tightness needs `f` to vanish on the *entire* set of occurring inner products of a tight, essentially unique configuration. In `n=8, 24` the candidate is rigid and is a tight design, so a finite root set suffices. In `n = 3` the optimum is *not* unique — there is a positive-dimensional family of `12`-point configurations with `tau_3 = 12` — so to be tight the certificate would have to vanish on a whole *range* of inner products, not finitely many points, and a polynomial that is `<= 0` on `[-1,1/2]` and zero on a subinterval is identically zero on it. The machine has no room. The `n=4` `24`-cell is unique but is not a tight enough design for the bound to land on `24`. The flexibility of the optimum is precisely what defeats pure LP.

So the off-diagonal terms I threw away — `sum_{i != j} f(<x_i,x_j>) <= 0` — are leaking. I was too generous: I bounded that whole sum by zero. Can I recover some of it instead of discarding it? The wasteful region is near `t = -1`: a point's near-antipode. Let `f` be *positive* near `t = -1` (give up `f <= 0` on `[-1, t_0]` for some `t_0` with `-1 <= t_0 < -1/2`, only keeping `f <= 0` on `[t_0, 1/2]`), and instead of dropping the off-diagonal terms with inner product in `[-1, t_0]`, bound those terms by a local geometric count. Concretely: for a fixed point `x_i`, how many other points `x_j` can have `<x_i, x_j> <= t_0`? Those `x_j` all lie in the spherical cap opposite `x_i` (angular radius set by `t_0`), and they themselves obey pairwise `<= 1/2`, so the cap holds at most some small number `mu` of them. Then the leaked sum per point is at most the maximum of `f(1) + sum_{j} f(<x_i, x_j>)` over configurations of `m <= mu` points in that cap — call the maximum over `m` of these `max{h_0, ..., h_mu}`. Redo the count with this in place of the careless `<= N f(1)` and I get

```
tau_n <= max{h_0, h_1, ..., h_mu} / c_0,
```

provided `f` is a nonnegative Gegenbauer combination, `f <= 0` on `[t_0, 1/2]`, and `f` is decreasing on `[-1, t_0]` (so the cap configurations are controlled). The cost is honest: computing `h_m` for `m >= 2` is a nonconvex optimization over point configurations in a cap (it needs estimates on how close `M` points on `S^{n-1}` can be), so this is no longer a clean LP. But the relaxation is what was missing. For `n = 3`, choosing `t_0 = -0.5907`, taking `mu = 4`, and a degree-`9` polynomial of this relaxed type, the bound finally drops below `13`, giving `tau_3 = 12` — a uniform-flavored proof that does not reduce to the `1953` spherical-triangle enumeration. For `n = 4`, with `t_0 = -0.608`, `mu = 6`, and a degree-`9` polynomial, it drops to `24`, settling `tau_4 = 24`.

That handles the upper bounds. The other half — the lower-bound construction — is where the intermediate dimensions like `n = 11` actually get decided, because there the certificate sits far above any known configuration and the record *is* the best construction. The principled way to build configurations is not random placement; it is to borrow the combinatorial structure of error-correcting codes. Take a binary `(n, M, d)` code `C`. Construction A builds a lattice from it: a point `(x_1, ..., x_n)` with integer coordinates is a center iff its reduction mod `2` is a codeword of `C`. The minimal vectors of this lattice are a kissing configuration, and — this is the payoff of going through codes — the number of contacts is read straight off the weight distribution of `C`: `2^d A_d(x)` if `d < 4`, `2n + 16A_4(x)` if `d = 4`, and only the `2n` coordinate contacts if `d > 4`, where `A_d(x)` counts codewords at distance `d` from `x`. Construction B refines this (restrict to even-weight codewords with coordinate sum divisible by `4`), changing the contact count to `2^{d-1} A_d(x)` if `d < 8`, `2n(n-1) + 128A_8(x)` if `d = 8`, and `2n(n-1)` if `d > 8` — and remarkably, applied to the right code in `n = 24`, Construction B reproduces the even part of the Leech lattice, closing the loop with the upper-bound story there. Below dimension `24` you then chain these: take cross-sections to drop a dimension, stack layers (laminations) to climb one, and concatenate codes (put a small code in selected coordinate blocks of a larger one) to squeeze out more contacts. In dimension `11` the laminated lattice `Lambda_11` minimal vectors give `tau_11 >= 582`, while the best LP/Levenshtein-type certificate sits up around `870` — so `11` is exactly the kind of intermediate case where the answer is unknown and *progress means a better construction*, found by smarter code/lattice engineering rather than by the bound. The construction half and the certificate half are the two jaws; in `8` and `24` they close on the same number, and everywhere in between the gap is the open problem.

Let me put the certificate machine into code, mirroring the two-way count exactly. Generate the dimension-`n` Gegenbauer polynomials by their recurrence; expand a chosen `f` in that basis to read off `c_0` and confirm all `c_k >= 0`; check the sign condition on the kissing window; evaluate `f(1)` and return `f(1)/c_0`. For the exact cases I instantiate the configuration-forced polynomial `f_8` / `f_24`; the same primitives are what a sampled LP would feed to a linear solver in dimensions where the roots are not already forced.

```python
import numpy as np
from numpy.polynomial import polynomial as P

def gegenbauer_n(n, kmax):
    # dimension-n ultraspherical polys, normalized G_k(1)=1, by the recurrence
    # (k+n-2)G_{k+1} = (2k+n-2) t G_k - k G_{k-1};  G_0=1, G_1=t.
    G = [np.array([1.0]), np.array([0.0, 1.0])]
    for k in range(1, kmax):
        tGk = np.concatenate([[0.0], G[k]])               # t * G_k
        gkm1 = np.zeros(len(tGk)); gkm1[:len(G[k-1])] = G[k-1]
        G.append(((2*k+n-2)*tGk - k*gkm1) / (k + n - 2))  # next polynomial
    return G[:kmax+1]

def gegenbauer_coeffs(fpoly, n):
    # re-express f (power-basis coeffs) in the Gegenbauer basis: c_0..c_d
    f = np.array(fpoly, dtype=float); d = len(f) - 1
    G = gegenbauer_n(n, d); c = np.zeros(d+1)
    for k in range(d, -1, -1):                            # peel from the top
        c[k] = f[k] / G[k][k]                             # match leading coeff
        f[:k+1] -= c[k] * G[k][:k+1]
    return c

def polynomial_from_roots(roots_with_mult):
    poly = np.array([1.0])
    for r, m in roots_with_mult:
        for _ in range(m):
            poly = P.polymul(poly, [-r, 1.0])             # multiply by (t - r)
    return poly

def certificate_bound(fpoly, n, s=0.5, grid=2001):
    # f must have nonnegative Gegenbauer coefficients and be nonpositive
    # on the off-diagonal window; then tau_n <= f(1)/c_0.
    poly = np.array(fpoly, dtype=float)
    c = gegenbauer_coeffs(poly, n)
    assert c[0] > 0 and np.all(c[1:] >= -1e-9), "Gegenbauer coefficients must be nonnegative"
    t = np.linspace(-1.0, s, grid)
    assert np.max(P.polyval(t, poly)) <= 1e-8, "f must be nonpositive on [-1, s]"
    return P.polyval(1.0, poly) / c[0]

# E8: inner products {-1, -1/2(double), 0(double), 1/2}
print(certificate_bound(polynomial_from_roots(
    [(-1,1), (-0.5,2), (0.0,2), (0.5,1)]), 8))                    # 240.0
# Leech: {-1, -1/2(2), -1/4(2), 0(2), 1/4(2), 1/2}
print(certificate_bound(polynomial_from_roots(
    [(-1,1), (-0.5,2), (-0.25,2), (0.0,2), (0.25,2), (0.5,1)]), 24)) # 196560.0
```

The whole chain, in one breath: the kissing constraint makes the Gram matrix PSD of rank `n` with ones on the diagonal and off-diagonal `<= 1/2`; the addition theorem turns each Gegenbauer polynomial of the inner product into a sum of squares over the configuration, so any nonnegative Gegenbauer combination has a nonnegative entry-sum; counting `sum_{i,j} f(<x_i,x_j>)` two ways — `>= c_0 N^2` from the constant part, `<= N f(1)` from forcing `f <= 0` on the constraint window — traps `N <= f(1)/c_0`; minimizing that ratio is a linear program; and demanding tightness *reads the optimal polynomial straight off* the candidate configuration's inner products, which in dimensions `8` and `24` are so few and so symmetric that the bound lands exactly on the `E_8` and Leech constructions. Where pure LP stalls (`n = 3, 4`) the window must be relaxed near the antipode and a small geometric cap count substituted, and where no construction meets the certificate (`n = 11` and its neighbors) the gap is the open problem.
