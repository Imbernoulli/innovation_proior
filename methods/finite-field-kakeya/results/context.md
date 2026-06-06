# Context: Lower bounds for Kakeya sets over finite fields

## Research question

Let `F = F_q` be a finite field and work in `F^n`. A Kakeya set is a set
`K ⊆ F^n` that contains a full affine line in every vector direction: for each
`x ∈ F^n` there is an offset `y ∈ F^n` such that

    { y + a x : a ∈ F }

lies inside `K`. The zero direction is harmless; the useful directions are the
nonzero vectors, with scalar multiples representing the same projective
direction. The finite-field Kakeya problem asks how small such a set can be.
The target lower bound has the form

    |K| ≥ c_n q^n,

where `c_n` depends only on the dimension and not on the field size. This is the
finite-field analogue of the Euclidean Kakeya question: containing a line segment
in every direction should force full-dimensional size. Over a finite field, the
measure-theoretic language disappears and the problem becomes a sharp
cardinality statement.

## Background

**Directional richness.** A Kakeya set may reuse points heavily by choosing
different offsets for different directions, but it cannot avoid representing all
directions. The central obstruction has to come from that directional spread,
not from a naive line count: many different lines can share points, and a lower
bound must survive that overlap.

**Small constructions.** In `F_q^2` for odd `q`, a standard quadratic
construction takes, for each slope `m`, the line in direction `(1,m)` through
`(0, -m^2/4)`, together with one vertical line. For `q = 5, 7, 11` this produces
sets of sizes `17, 31, 71`, with ratios to `q^2` moving toward `1/2`. Products of
such examples show that constants near `2^{-n}` are the natural scale; a proof
cannot hope for a lower bound close to the full `q^n` points.

**Polynomial dimension counts.** The vector space of polynomials in `n`
variables of total degree at most `d` has the monomial basis
`x_1^{e_1}...x_n^{e_n}` with `e_1 + ... + e_n ≤ d`, so its dimension is

    C(n+d, n).

The homogeneous degree-`d` subspace has dimension

    C(d+n-1, n-1).

Vanishing at a specified point is one homogeneous linear equation in the
coefficients. If a set `S` has fewer points than the relevant polynomial-space
dimension, linear algebra gives a nonzero polynomial in that space vanishing on
all of `S`.

**Zero counting.** The Schwartz-Zippel bound says that a nonzero polynomial of
total degree at most `d` over `F_q^n` has at most

    d q^{n-1}

zeros. In one variable, this is the familiar fact that a nonzero polynomial of
degree `d` has at most `d` roots. In particular, a univariate polynomial of
degree at most `q-1` that vanishes at all `q` field elements is the zero
polynomial.

**Homogeneity and lines through the origin.** If `g` is homogeneous of degree
`d` and `g(x)=0`, then `g(c x)=c^d g(x)=0` for every `c ∈ F`. Thus ordinary
vanishing on a set automatically extends to the cone through the set. That cone
operation is useful when a line in direction `y` meets `K` many times: after
rescaling those intersection points, one obtains many zeros on a line passing
through `y`.

**Multiplicity bookkeeping.** A sharper polynomial method asks a polynomial to
vanish to multiplicity `m` at each point, meaning all Hasse derivatives of order
less than `m` vanish. At a single point this imposes

    C(m+n-1, n)

linear conditions. The multiplicity version of Schwartz-Zippel is

    sum_{a ∈ F_q^n} mult(P,a) ≤ d q^{n-1}

for a nonzero polynomial of total degree at most `d`. Dvir-Kopparty-Saraf-Sudan
combine these counts with leading-form multiplicity propagation: with
`ell` a multiple of `q`, `d = ell q - 1`, and `m = 2 ell - ell/q`, the
inequalities `d < ell q` and `(m-ell)q > d-ell` make the line-restriction
multiplicity step work. The leading homogeneous part is forced to vanish to
multiplicity `ell` at every point of `F_q^n`. Interpolation would be possible
if

    C(m+n-1, n) |K| < C(d+n, n),

so the contradiction gives

    |K| ≥ C(d+n,n) / C(m+n-1,n).

For the chosen `d` and `m`, this ratio is

    prod_{i=1}^n (ell q - 1 + i)/(2 ell - ell/q - 1 + i),

which tends to `(q/(2 - 1/q))^n`, and hence gives the clean `q^n/2^n`
sharpening.

## Baselines

- **Wolff incidence bound, `|K| ≥ C_n q^{(n+2)/2}`.** This counts incidences
  between points and lines and uses the limited overlap of lines in different
  directions. It gives the first nontrivial finite-field lower bound, but for
  large `n` its exponent is far below the desired `n`.

- **Additive-combinatorial bounds, about `C_n q^{4n/7}` in general dimension.**
  Bourgain, Katz-Tao, Mockenhaupt-Tao, Rogers, and related arguments connect a
  small Kakeya set to unexpectedly small sumsets such as `A + r B` for many
  field elements `r`. Sum-product growth then rules this out. These methods are
  powerful but lose a fixed fraction of the exponent, leaving an exponent gap
  rather than only a constant-factor gap.

- **Low-dimensional incidence refinements.** Special arguments in dimensions
  such as `3` and `4` improve the exponent in those cases, but the mechanisms do
  not scale into a uniform `q^n` theorem for every fixed `n`.

- **Homogeneous polynomial counting.** Counting only homogeneous degree-`d`
  forms gives `C(d+n-1,n-1)` coefficients. With `d = q-2` this reaches
  `≈ q^{n-1}/(n-1)!`, and the product trick converts it to `q^{n-epsilon}` for
  any fixed `epsilon > 0`. The missing factor of `q` comes from using only one
  homogeneous degree slice rather than all polynomials of degree at most `q-1`.

The common gap is that the earlier approaches either translate directionality
through lossy incidence/additive estimates or count too few polynomial
coefficients.

## Evaluation settings

The main yardstick is the worst-case lower bound for `|K|` as a function of `q`
for fixed `n`, compared against `q^n`. Exact constants matter after the exponent
is correct: the quadratic planar construction and its products put the natural
scale near `2^{-n} q^n`.

A second yardstick is the statistical relaxation. A `(delta,gamma)`-Kakeya set
has at least `delta q^n` vector directions such that, in each selected direction,
some affine line meets `K` in at least `gamma q` points. A robust lower bound
should degrade in a controlled way as `delta` and `gamma` decrease.

The finite-field instances include prime and prime-power fields. The executable
checks here use small prime fields, where ordinary integer residues implement
field arithmetic directly.

## Code framework

The scaffold uses finite-field arithmetic, monomial enumeration, polynomial
evaluation, Gaussian elimination over `F_q`, and explicit line checks. The
certification routine is left as the algebraic step that turns Kakeya line data
into a lower bound.

```python
from itertools import product
from math import comb

def make_field(q):           # q prime in the executable checks
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
    return sum(c * peval_mon(e, pt, q) for e, c in poly.items()) % q

def nullspace_dim(rows, ncols, q):
    """Dimension of {c : A c = 0} over F_q."""
    pass

def schwartz_zippel_bound(d, q, n):
    return d * q**(n - 1)

def vanishing_dim(point_set, q, n, d):
    mons = monomials_up_to_degree(n, d)
    rows = [[peval_mon(e, pt, q) for e in mons] for pt in point_set]
    return nullspace_dim(rows, len(mons), q)

def vanishing_polynomial_exists(point_set, q, n, d):
    return vanishing_dim(point_set, q, n, d) > 0

def contains_full_line(point_set, q, direction):
    """Does point_set contain an affine line in the given direction?"""
    pass

def is_plane_kakeya(point_set, q):
    """Check one representative of each projective direction in F_q^2."""
    pass

def quadratic_plane_kakeya(q):
    """Quadratic small Kakeya construction in F_q^2 for odd prime q."""
    pass

def line_direction_certificate_holds(q, n):
    """Return whether the algebraic certificate attached to line directions holds."""
    pass

def kakeya_lower_bound(q, n):
    # TODO: certify the algebraic lower-bound step.
    pass
```
