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
line through `y`. If `z=0`, this already gives `g(y)=0`; if `z≠0`, the
restriction of `g` to that line is a univariate polynomial of degree at most
`d` with `d+1` roots, so it is identically zero and in particular vanishes at
`y`. Thus `g` vanishes on at least `delta q^n` points.
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
`t<q` and all `q` field elements as roots, so it is identically zero. Lower
homogeneous pieces cannot contribute to the coefficient of `a^t`, and the
degree-`t` piece contributes exactly `P_t(y)`.

Every nonzero direction occurs. If `t>0`, homogeneity gives `P_t(0)=0`, so
`P_t` vanishes on all of `F_q^n`. Schwartz-Zippel then forces `P_t` to be the
zero polynomial, contradicting the choice of `t`. If `t=0`, then `P` is a
nonzero constant, which cannot vanish on the nonempty set `K`. Therefore no such
nonzero `P` exists, and

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
`m = 2 ell - ell/q`. If interpolation produced a nonzero `P` of actual degree
`d*≤d`, then `d*≥m≥ell`. For every Hasse derivative of weight `w<ell`, the
line restriction has degree at most `d*-w` and has all `q` field points as roots
with multiplicity at least `m-w`. The inequality

    (m-w)q > d* - w

follows from `(m-w)q - (d-w) = (q-1)(ell-w)+1`. Thus those restrictions are zero,
and their leading coefficients force every derivative of the top homogeneous
part of weight below `ell` to vanish in every direction. The top part vanishes to
multiplicity `ell` at every point of `F_q^n`; since
`ell q^n > d q^{n-1} ≥ d* q^{n-1}`, the multiplicity zero-counting bound kills
it. Thus interpolation cannot occur under

    |K| < C(d+n,n) / C(m+n-1,n).

For the chosen parameters,

    C(d+n,n) / C(m+n-1,n)
      = prod_{i=1}^n (ell q - 1 + i)/(2 ell - ell/q - 1 + i),

and letting `ell → ∞` gives

    |K| ≥ (q/(2 - 1/q))^n ≥ q^n/2^n.
