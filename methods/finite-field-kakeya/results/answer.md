# The polynomial method for finite-field Kakeya

## Problem

For `F = F_q`, a Kakeya set `K ⊆ F^n` contains a full line
`{ y + a x : a ∈ F }` in every vector direction `x`. The goal is a lower bound
`|K| ≥ c_n q^n` with `c_n` independent of `q`.

## Homogeneous polynomial bound

For a `(delta,gamma)`-Kakeya set, take

    d = floor(q min{delta,gamma}) - 2.

When `d≥0`, if `|K| < C(d+n-1,n-1)`, then a nonzero homogeneous degree-`d`
polynomial `g` vanishes on `K`. Homogeneity extends the zeros to the cone through
`K`. For each of at least `delta q^n` directions `y`, the inequality
`d+2≤gamma q` makes the zero-direction singleton irrelevant, so the chosen line
`z+a y` supplies at least `d+2` distinct parameters with points in `K`. After
discarding a possible zero parameter, rescaling gives `d+1` zeros of `g` on a
line through `y`, unless `z=0`, in which case `g(y)=0` directly. Thus `g`
vanishes on at least `delta q^n` points.
Schwartz-Zippel allows at most `d q^{n-1}` zeros for a nonzero degree-`d`
polynomial, and `d/q < delta`, a contradiction.

So

    |K| ≥ C(d+n-1,n-1).

For a full Kakeya set, `d=q-2`, giving `C(q+n-3,n-1) ≈ q^{n-1}/(n-1)!`.
Since `K^r` is Kakeya in `F_q^{nr}`, this amplifies to
`|K| ≥ C_{n,epsilon} q^{n-epsilon}` for every fixed `epsilon>0`.

## Leading-form upgrade

The missing factor of `q` comes from counting only homogeneous forms. Use all
polynomials of degree at most `q-1`, a space of dimension

    C(n+q-1,n).

If `|K|` were smaller than this, a nonzero `P` of degree at most `q-1` would
vanish on `K`. Write `P=P_0+...+P_t`, with `P_t` the highest nonzero homogeneous
part. For a line `b+a y ⊆ K`, the univariate polynomial `P(b+a y)` has degree
`t<q` and all `q` field elements as roots, so it is identically zero. The
coefficient of `a^t` in this restriction is exactly `P_t(y)`; in particular, at
top degree `q-1`, `coeff_{a^{q-1}} P(b+a y)=P_{q-1}(y)`.

Every direction occurs, so `P_t` vanishes on all of `F_q^n`. Schwartz-Zippel
forces `P_t=0`; applying the same argument to the next highest degree peels the
polynomial down to a constant, which is also zero because `P` vanishes on `K`.
Therefore no such nonzero `P` exists, and

    |K| ≥ C(q+n-1,n)
        = (q+n-1)(q+n-2)...q / n!
        ≈ q^n/n!.

## Multiplicity sharpening

Requiring multiplicity-`m` vanishing at each point of `K` imposes
`C(m+n-1,n)` linear conditions per point, so interpolation succeeds when

    C(m+n-1,n) |K| < C(d+n,n).

The multiplicity Schwartz-Zippel bound is

    sum_{a ∈ F_q^n} mult(P,a) ≤ d q^{n-1}.

With `ell` a large multiple of `q`, choose `d = ell q - 1` and
`m = 2 ell - ell/q`. Then `d < ell q` and `(m-ell)q > d-ell`, which are the
degree inequalities needed for the line-restriction multiplicity step. The
leading homogeneous part is forced to vanish to multiplicity `ell` at every point
of `F_q^n`; since `ell q^n > d q^{n-1}`, the multiplicity zero-counting bound
kills it. Thus interpolation cannot occur under

    |K| < C(d+n,n) / C(m+n-1,n).

For the chosen parameters,

    C(d+n,n) / C(m+n-1,n)
      = prod_{i=1}^n (ell q - 1 + i)/(2 ell - ell/q - 1 + i),

and letting `ell → ∞` gives

    |K| ≥ (q/(2 - 1/q))^n ≥ q^n/2^n.

## Code

```python
from itertools import product
from math import comb
import random

def make_field(q):                       # q prime in these executable checks
    return list(range(q))

def monomials_up_to_degree(n, d):
    return [e for e in product(range(d + 1), repeat=n) if sum(e) <= d]

def homogeneous_monomials(n, d):
    return [e for e in product(range(d + 1), repeat=n) if sum(e) == d]

def peval_mon(exps, pt, q):
    t = 1
    for v, e in zip(pt, exps):
        t = (t * pow(v, e, q)) % q
    return t

def peval(poly, pt, q):
    return sum((c % q) * peval_mon(e, pt, q) for e, c in poly.items()) % q

def nullspace_dim(rows, ncols, q):
    A = [row[:] for row in rows]
    r = 0
    rank = 0
    for c in range(ncols):
        pivot = next((i for i in range(r, len(A)) if A[i][c] % q), None)
        if pivot is None:
            continue
        A[r], A[pivot] = A[pivot], A[r]
        inv = pow(A[r][c], q - 2, q)
        A[r] = [(x * inv) % q for x in A[r]]
        for i in range(len(A)):
            if i != r and A[i][c] % q:
                f = A[i][c]
                A[i] = [(A[i][k] - f * A[r][k]) % q for k in range(ncols)]
        r += 1
        rank += 1
        if r == len(A):
            break
    return ncols - rank

def schwartz_zippel_bound(d, q, n):
    return d * q ** (n - 1)

def vanishing_dim(point_set, q, n, d):
    mons = monomials_up_to_degree(n, d)
    rows = [[peval_mon(e, pt, q) for e in mons] for pt in point_set]
    return nullspace_dim(rows, len(mons), q)

def vanishing_polynomial_exists(point_set, q, n, d):
    return vanishing_dim(point_set, q, n, d) > 0

def contains_full_line(point_set, q, direction):
    F = make_field(q)
    for base in product(F, repeat=len(direction)):
        if all(tuple((base[i] + a * direction[i]) % q for i in range(len(direction))) in point_set for a in F):
            return True
    return False

def is_plane_kakeya(point_set, q):
    F = make_field(q)
    directions = [(1, m) for m in F] + [(0, 1)]
    return all(contains_full_line(point_set, q, direction) for direction in directions)

def quadratic_plane_kakeya(q):
    F = make_field(q)
    K = set()
    inv4 = pow(4, q - 2, q)
    for m in F:
        oy = (-(m * m) * inv4) % q
        for a in F:
            K.add((a % q, (oy + a * m) % q))
    for b in F:
        K.add((0, b))
    return K

def lagrange_top_coefficient(vals, q):
    F = make_field(q)
    lead = 0
    for i, a in enumerate(F):
        denom = 1
        for j, aa in enumerate(F):
            if i != j:
                denom = (denom * ((a - aa) % q)) % q
        lead = (lead + vals[i] * pow(denom, q - 2, q)) % q
    return lead

def line_direction_certificate_holds(q, n):
    F = make_field(q)
    random.seed(1000 + 10 * q + n)
    mons = homogeneous_monomials(n, q - 1)
    P_top = {e: random.randrange(q) for e in mons}
    if all(c == 0 for c in P_top.values()):
        P_top[mons[0]] = 1
    for y in product(F, repeat=n):
        for b in product(F, repeat=n):
            vals = [
                peval(P_top, tuple((bi + a * yi) % q for bi, yi in zip(b, y)), q)
                for a in F
            ]
            if lagrange_top_coefficient(vals, q) != peval(P_top, y, q):
                return False
    return True

def kakeya_lower_bound(q, n):
    return comb(q + n - 1, n)

def run_checks():
    for n in range(1, 5):
        for d in range(0, 7):
            assert len(monomials_up_to_degree(n, d)) == comb(n + d, n)
            assert len(homogeneous_monomials(n, d)) == comb(d + n - 1, n - 1)

    random.seed(1)
    for q, n in [(5, 2), (7, 2), (5, 3)]:
        F = make_field(q)
        for _ in range(200):
            d = random.randint(1, q - 1)
            P = {e: random.randrange(q) for e in monomials_up_to_degree(n, d)}
            if all(c == 0 for c in P.values()):
                continue
            zeros = sum(1 for pt in product(F, repeat=n) if peval(P, pt, q) == 0)
            assert zeros <= schwartz_zippel_bound(d, q, n)

    assert line_direction_certificate_holds(5, 2)
    assert line_direction_certificate_holds(7, 2)

    for q in (5, 7, 11):
        K = quadratic_plane_kakeya(q)
        assert is_plane_kakeya(K, q)
        d = q - 1
        assert not vanishing_polynomial_exists(K, q, 2, d)
        print(
            f"q={q}: |K|={len(K)}, "
            f"C(q+1,2)={kakeya_lower_bound(q, 2)}, "
            f"nullspace={vanishing_dim(K, q, 2, d)}"
        )

    print("OK")

if __name__ == "__main__":
    run_checks()
```
