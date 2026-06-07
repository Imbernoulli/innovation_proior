# Context: reduced bases for lattices

## Research question

A lattice `L` of rank `n` in `R^m` is the set of integer combinations of `n`
linearly independent vectors `b_1, ..., b_n`. The same `L` is spanned by
infinitely many different bases: any two bases differ by a unimodular integer
matrix (an element of `GL_n(Z)`, determinant `±1`). Some of these bases are
"good" — their vectors are short and nearly orthogonal — and some are terrible,
with long, skewed vectors that obscure the geometry of `L`. The practical
problem is: given an arbitrary, possibly very skewed, integer basis, transform
it by integer operations into one of the good bases — in particular one whose
first vector is short — and do it in time polynomial in `n` and in the bit-size
of the input.

This matters because a short lattice vector is the answer to many number-
theoretic and algorithmic questions. The shortest vector itself (the Shortest
Vector Problem, SVP) is hard to compute exactly in high dimension, so the
achievable goal is weaker: return a vector that is *provably* short relative to
the true shortest one, with a guarantee that degrades gracefully — ideally only
exponentially in `n` but with a small base — and in *polynomial* time. A
solution would have to (i) define precisely what "reduced enough" means, (ii)
give a finite sequence of unimodular moves that reaches that state from any
start, and (iii) prove the move-count is polynomially bounded.

## Background

**Lattices and the determinant.** Writing the basis as columns, the determinant
`d(L) = |det(b_1, ..., b_n)|` is a positive real that does not depend on which
basis is chosen — it is the covolume of `L`. So any honest measure of basis
quality must be one that the unimodular freedom cannot inflate away.

**Gram-Schmidt orthogonalization.** From any independent `b_1, ..., b_n` one
builds an orthogonal sequence `b*_1, ..., b*_n` and real coefficients `mu_{i,j}`
(for `j < i`) by

    b*_i = b_i - sum_{j<i} mu_{i,j} b*_j ,   mu_{i,j} = <b_i, b*_j> / <b*_j, b*_j>.

Here `b*_i` is the component of `b_i` orthogonal to the span of `b_1, ..., b_{i-1}`,
and the partial spans agree: `span(b_1,...,b_i) = span(b*_1,...,b*_i)`. Two facts
make the `b*_i` the "real" lengths of the lattice. First, because the `b*_i` are
pairwise orthogonal, `d(L) = prod_i |b*_i|` — the product of GS lengths is the
basis-invariant covolume. Second, any lattice vector
`x = sum r_i b_i = sum r'_i b*_i` with `r_i` integers has `r'_i = r_i` for the
largest index `i` with `r_i != 0` (because the change of coordinates from `b` to
`b*` is upper-triangular with ones on the diagonal); hence `|x| >= |b*_i|`. So
the `|b*_i|` lower-bound the lengths of all lattice vectors that "reach" index
`i`. The `b*_i` themselves are generally *not* in the lattice — only the `b_i`
are — but their lengths are the invariant the geometry cares about.

A basis is "good" exactly when these GS lengths do not collapse too quickly as
`i` grows: if `|b*_i|` stays comparable to `|b*_{i-1}|`, then no lattice vector
can be much shorter than `|b_1|`, and `|b_1|` itself is controlled.

**Classical reduction theory.** The idea of choosing a canonical, near-orthogonal
basis goes back to the theory of quadratic forms: Lagrange (1773) and Gauss
(1801) for binary (rank-2) forms, then Hermite (1850), Korkine and Zolotarev
(1873), and Minkowski (1896) in higher rank. Their notions of a "reduced" basis
give the strongest geometric guarantees, but the algorithms that compute them
are not known to run in polynomial time as the rank grows — Minkowski-reduction
in particular involves searching for shortest vectors. Two pieces of this
classical theory are load-bearing here:

- *Minkowski's convex-body bound / successive minima.* In a lattice of rank `i`
  and covolume `D_i`, there is a nonzero vector of squared length at most
  `(4/3)^{(i-1)/2} D_i^{2/i}`. This is what eventually lower-bounds the GS
  lengths and turns "the potential keeps dropping" into "the potential cannot
  drop forever".

- *Gauss/Lagrange rank-2 reduction.* For two vectors `v_1, v_2`, repeatedly
  subtract `round(<v_1,v_2>/<v_1,v_1>)` copies of `v_1` from `v_2`, then swap if
  `|v_2| < |v_1|`; iterate. This is the lattice analogue of the Euclidean
  algorithm — Euclid subtracts integer multiples to shrink integers, this
  subtracts vector multiples to shrink a vector's component along another — and
  it terminates with the genuinely shortest basis of the rank-2 lattice. The
  product `|v_1||v_2|` strictly decreases at each swap, which is why it stops.
  This rank-2 procedure is the only fully-understood reduction step, and it works
  precisely because in two dimensions the rounding plus swap cannot loop.

**The gap.** Classical high-rank reduction gives the best bases but no
polynomial-time algorithm; the rank-2 Gauss step is polynomial and exact but
only handles two vectors. What is missing is a *relaxed* notion of reducedness,
weak enough that a Gauss-style rounding-and-swapping process provably converges
in polynomially many steps on any rank, yet strong enough that the output still
guarantees a short `b_1`.

## Baselines

**Hermite/Minkowski-reduced bases.** A Minkowski-reduced basis takes `b_1` to be
a shortest nonzero lattice vector, `b_2` shortest independent of `b_1`, and so on.
The guarantees are essentially optimal, but producing such a basis requires
solving SVP-like subproblems; no polynomial-time algorithm in the rank is known.
This is the quality target one would like to approximate cheaply.

**Gauss/Lagrange rank-2 reduction.** Core idea and math as above: size-reduce
`v_2` against `v_1` by rounding the Gram-Schmidt coefficient, swap on
`|v_2| < |v_1|`, repeat until no swap occurs. Output is provably optimal in rank
2 and the algorithm is fast. The limitation is dimensional: it gives no direct
handle on rank `n > 2`, where there is no single "the" shortest basis reachable
by a local two-vector move and where a naive generalization (reduce every pair)
need not terminate or guarantee anything.

**Brute-force / enumeration of short vectors.** One can in principle enumerate
lattice points within a ball to find short vectors, but the number of points
grows so fast with rank that this is infeasible beyond small `n`; it gives the
exact answer but not in polynomial time.

## Evaluation settings

The natural yardsticks are intrinsic and worst-case rather than benchmark
datasets. For an input integer basis with `|b_i|^2 <= B`:

- The quality of the output is measured by how short `b_1` is relative to the
  basis-invariant scale `d(L)^{1/n}` and relative to the true shortest vector
  `lambda_1(L)` — i.e. an approximation factor as a function of rank `n`.
- The cost is measured in arithmetic operations on integers and in the bit-length
  of those integers, as a function of `n` and `log B`. The relevant regime is
  "polynomial in `n` and `log B`", and within that one cares about the exponent
  of `n` and whether all arithmetic can be kept exact in integers of controlled
  size.
- The downstream setting that motivates the cost target is using the reduction
  as a subroutine inside larger number-theoretic algorithms (factoring integer
  polynomials, simultaneous Diophantine approximation), where it is called
  repeatedly and must therefore be cheap and exact.

## Code framework

What exists before the method: exact rational arithmetic, the inner product, and
Gram-Schmidt. The reduction logic itself — what counts as reduced, which move to
make, when to stop — is the empty slot.

```python
from fractions import Fraction


def dot(u, v):
    return sum(Fraction(a) * Fraction(b) for a, b in zip(u, v))


def gram_schmidt(B):
    """b*_i = b_i - sum_{j<i} mu_{i,j} b*_j,  mu_{i,j} = <b_i,b*_j>/<b*_j,b*_j>."""
    n = len(B)
    Bstar = [None] * n
    mu = [[Fraction(0) for _ in range(n)] for _ in range(n)]
    for i in range(n):
        bi = [Fraction(x) for x in B[i]]
        Bstar[i] = bi[:]
        for j in range(i):
            mu[i][j] = dot(B[i], Bstar[j]) / dot(Bstar[j], Bstar[j])
            Bstar[i] = [a - mu[i][j] * c for a, c in zip(Bstar[i], Bstar[j])]
    return Bstar, mu


def reduce_basis(B):
    """Apply unimodular (det +-1) integer moves to B until it is 'reduced enough'.
    TODO: decide what 'reduced' means (a per-vector size condition plus a
          condition on the Gram-Schmidt lengths), which local move restores it,
          and prove the move-count is polynomial.
    """
    # TODO: size-reduction step (round a Gram-Schmidt coefficient, subtract).
    # TODO: the test on adjacent Gram-Schmidt lengths that triggers a swap.
    # TODO: the swap, and how to advance/retreat through the basis.
    pass
```
