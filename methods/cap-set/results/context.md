# Context: bounding the largest 3-AP-free subset of F_3^n

## Research question

Fix the field F_3 and the vector space F_3^n. A subset A ⊆ F_3^n is *3-AP-free*
(a "cap") if it contains no three-term arithmetic progression: no distinct
x, y, z ∈ A with x + z = 2y. Over F_3, multiply by 2 = −1 and this is exactly
x + y + z = 0, i.e. A contains no nontrivial *line* {b, b+r, b+2r}, r ≠ 0.
Write r_3(F_3^n) for the maximum size of such a set.

The question is the *growth rate* of r_3(F_3^n) as n → ∞. The whole space has
3^n points. Is r_3(F_3^n) as large as (3 − ε)^n for every ε > 0 (so the trivial
3^n is essentially right up to a sub-exponential factor), or is it bounded by
c^n for some fixed c < 3 (an honest *exponential* saving)? F_3^n is the cleanest
model for Roth's theorem on 3-AP-free subsets of {1,…,N}, and the size of the
largest cap is one of the central quantities in additive combinatorics.

## Background

The largest-3-AP-free-set problem has a long history. Roth (1952, 1953) showed a
3-AP-free A ⊆ {1,…,N} has |A| = O(N / log log N), later sharpened by Heath-Brown,
Szemerédi, Bourgain, Sanders, Bloom. Roth's problem is essentially the cyclic
group Z_N; it is natural to move to other finite abelian groups, and the vector
spaces (Z/3Z)^n sit at one extreme.

Two ingredients dominate the prevailing technology for upper bounds. First, the
Fourier transform on F_3^n: a set with no 3-AP has a large Fourier coefficient,
which one converts (via a density-increment or directly in this group) into
structure. Second, in the F_p^n setting Meshulam (1995) observed the argument is
*elementary*: it gives r_3(F_3^n) ≤ 2·3^n / n directly, no density increment
needed, because the relevant "Bohr set" is just a subspace. Bateman–Katz (2012)
pushed this to O(3^n / n^{1+ε}) by a deep analysis of the spectrum, staying
inside the Fourier/density-increment world.

Croot–Lev–Pach (2016) treat a neighboring problem: progression-free subsets of
(Z/4Z)^n. Their result gives an exponential upper bound c^n with c < 4, and the
proof uses the polynomial method rather than Fourier analysis. The regime is the
base ring fixed while n grows. The argument is carried out over Z/4Z, a ring
rather than a field, with its own coset bookkeeping, and is stated for that
two-term difference setting.

On the constructive (lower-bound) side, intuition from the integers can mislead.
Behrend (1946) built a large 3-AP-free subset of {1,…,N} by placing integers on
a sphere of fixed radius: a sphere is strictly convex, so it carries no three
collinear points, and a careful pigeonholing over radii yields a set of size
N·exp(−c√log N) — only a sub-polynomial factor below N. Over F_3 there is no
metric convexity to exploit; spheres {x : Σ x_i^2 = r} in F_3^n do not avoid
lines in the same way, and the principled constructions are of a different kind
(below).

A purely combinatorial fact organizes the constructions: 3-AP-freeness is closed
under direct products. If A ⊆ F_3^k and B ⊆ F_3^m are caps, then A × B ⊆ F_3^{k+m}
is a cap, because a line in A × B projects to a line (or a point) in each factor,
and at least one factor would have to contain a genuine line. Hence a single cap
of size M in some fixed dimension k tensors up: A^t ⊆ F_3^{kt} is a cap of size
M^t, giving the asymptotic per-dimension lower bound M^{1/k}.

## Baselines

**Meshulam's Fourier bound (1995).** A 3-AP-free A ⊆ F_3^n has a nontrivial
Fourier coefficient of size ≥ (density)^2 · |A| up to constants; iterating on the
hyperplane where that character is large (a subspace, so no density-increment
machinery is required) gives r_3(F_3^n) ≤ 2·3^n / n. Core idea: count 3-APs via
Σ_ξ â(ξ)^3; a 3-AP-free set forces a large non-principal coefficient, which
restricts the set to a hyperplane and recurses.

**Bateman–Katz (2012).** A deeper spectral analysis: if the large spectrum
is "spread out" one wins more, if it is "additively structured" one also wins,
and a dichotomy across these cases improves the bound to O(3^n / n^{1+ε}) for an
absolute ε > 0. Core idea: structure of the set of large Fourier coefficients.

**Croot–Lev–Pach (2016).** For progression-free subsets of (Z/4Z)^n, CLP
translate the obstruction into a problem over an F_2-vector space and use
low-degree polynomials. The published application is wrapped around Z/4Z's cosets
and treats the two-term difference a−b.

**Product / tensor constructions (lower bounds).** The largest caps in small
dimensions are known: the dimension-1 cap has size 2, then 4, 9, 20, 45, 112 for
n = 2,…,6. By the product principle a cap of size M in dimension k gives
r_3(F_3^n) ≥ M^{n/k} for n a multiple of k, i.e. base M^{1/k}: 20^{1/4} ≈ 2.1147,
112^{1/6} ≈ 2.1955. Calderbank–Fishburn (1994) refined the construction (gluing
rather than a raw Cartesian power) to base ≈ 2.2104. Edel (2004) introduced a
more elaborate product — a union of caps satisfying compatibility conditions is a
cap — reaching base ≈ 2.217389.

## Evaluation settings

The yardstick is the asymptotic growth rate of r_3(F_3^n) as n → ∞, reported as
the base c in c^n (with sub-exponential factors ignored), and the explicit
constant where one can be named. The relevant regimes: the trivial upper bound
3^n; the Fourier upper bounds of the form 3^n / poly(n); and the constructive
lower bounds c^n with c around 2.2. Sanity checks live in tiny dimensions
(n ≤ 6), where the exact maximum cap sizes 2, 4, 9, 20, 45, 112 are known and any
claimed bound must respect them; the small caps also feed the product
constructions. The quantities are exact counts and exact analytic constants, so
"evaluation" is a derivation plus a finite numeric confirmation, not a benchmark.

## Code framework

The available computational primitives are: an exact count of monomials by
degree over F_3, brute-force verification that a small set is a cap, and the
product-construction lower bound. An upper-bound certificate can plug into the
empty slot below.

```python
import math

def degree_counts(n):
    # [x^k] (1+x+x^2)^n counts reduced monomials of degree k over F_3.
    poly = [1]
    for _ in range(n):
        new = [0]*(len(poly)+2)
        for i, c in enumerate(poly):
            new[i] += c; new[i+1] += c; new[i+2] += c
        poly = new
    return poly

def m_d(n, d):
    c = degree_counts(n)
    d = int(math.floor(d))
    return sum(c[k] for k in range(min(d, len(c)-1)+1))

def is_cap(points):
    # brute check: no distinct x,y,z in the set with x+y+z = 0
    S = set(points); pts = list(points)
    for i in range(len(pts)):
        for j in range(i+1, len(pts)):
            x, y = pts[i], pts[j]
            z = tuple((-(a+b)) % 3 for a, b in zip(x, y))
            if z != x and z != y and z in S:
                return False
    return True

def product_lower_bound(cap_size, k):
    # a cap of size M in dim k -> per-dimension lower bound M^(1/k)
    return cap_size ** (1.0/k)

def upper_bound(n):
    # TODO: return the upper-bound certificate for dimension n
    pass

def upper_bound_constant():
    # TODO: return the limiting exponential base for the upper bound
    pass
```
