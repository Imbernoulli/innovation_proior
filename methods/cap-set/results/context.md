# Context: bounding the largest 3-AP-free subset of F_3^n

## Research question

Fix the field F_3 and the vector space F_3^n. A subset A ⊆ F_3^n is *3-AP-free*
(a "cap") if it contains no three-term arithmetic progression: no distinct
x, y, z ∈ A with x + z = 2y. Over F_3, multiply by 2 = −1 and this is exactly
x + y + z = 0, i.e. A contains no nontrivial *line* {b, b+r, b+2r}, r ≠ 0.
Write r_3(F_3^n) for the maximum size of such a set.

The question is the *growth rate*. The whole space has 3^n points; a cap can
clearly be much smaller. The sharp dichotomy at stake: is r_3(F_3^n) as large as
(3 − ε)^n for every ε > 0 (so the trivial 3^n is essentially right up to a
sub-exponential factor), or is it bounded by c^n for some fixed c < 3 (an
honest *exponential* saving)? F_3^n is the cleanest model for Roth's theorem on
3-AP-free subsets of {1,…,N}; a definitive answer here would say whether the
"finite field model" admits an exponential improvement that the integer setting
has resisted. A solution must produce an upper bound whose exponential rate is
c < 3, and ideally an explicit c — and it must do so by a mechanism that
escapes whatever barrier has held the previous methods to within a polynomial
factor of 3^n.

## Background

The largest-3-AP-free-set problem has a long history. Roth (1952, 1953) showed a
3-AP-free A ⊆ {1,…,N} has |A| = O(N / log log N), later sharpened by Heath-Brown,
Szemerédi, Bourgain, Sanders, Bloom. Roth's problem is essentially the cyclic
group Z_N; it is natural to move to other finite abelian groups, and the vector
spaces (Z/3Z)^n sit at one extreme. The size of the largest cap is one of the
central quantities in additive combinatorics.

Two ingredients dominate the prevailing technology for upper bounds. First, the
Fourier transform on F_3^n: a set with no 3-AP has a large Fourier coefficient,
which one converts (via a density-increment or directly in this group) into
structure. Second, in the F_p^n setting Meshulam (1995) observed the argument is
*elementary*: it gives r_3(F_3^n) ≤ 2·3^n / n directly, no density increment
needed, because the relevant "Bohr set" is just a subspace. Every refinement
after that — Bateman–Katz (2012), pushing to O(3^n / n^{1+ε}) by a deep analysis
of the spectrum — stayed inside the Fourier/density-increment world and saved
only a *polynomial* factor in n off the trivial 3^n. After roughly fifteen years
that polynomial-in-n saving was the ceiling, and there was no consensus that an
exponential improvement to c^n with c < 3 was even true.

On the constructive (lower-bound) side, intuition from the integers can mislead.
Behrend (1946) built a large 3-AP-free subset of {1,…,N} by placing integers on
a sphere of fixed radius: a sphere is strictly convex, so it carries no three
collinear points, and a careful pigeonholing over radii yields a set of size
N·exp(−c√log N) — only a sub-polynomial factor below N. One might hope the same
"points on a sphere" idea makes caps in F_3^n nearly all of the space. It does
not: F_3 has no metric convexity to exploit, spheres {x : Σ x_i^2 = r} in F_3^n
do not avoid lines in any comparably strong way, and the genuinely *principled*
constructions are of a different, weaker kind (below). So the lower-bound side is
exponential with a small base, far from 3.

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
restricts the set to a hyperplane and recurses. Gap: the saving is one factor of
n per step and the recursion loses, so it never beats 3^n by more than a
polynomial factor.

**Bateman–Katz (2012).** A much deeper spectral analysis: if the large spectrum
is "spread out" one wins more, if it is "additively structured" one also wins,
and a dichotomy across these cases improves the bound to O(3^n / n^{1+ε}) for an
absolute ε > 0. Core idea: structure of the set of large Fourier coefficients.
Gap: still O(3^n / poly(n)) — the same exponential 3^n, just a better polynomial.
No mechanism here produces c^n with c < 3, and the consensus was unclear whether
that was even the truth.

**Product / tensor constructions (lower bounds).** The largest caps in small
dimensions are known: |F_3^1|-cap 2, then 4, 9, 20, 45, 112 for n = 2,…,6. By
the product principle a cap of size M in dimension k gives r_3(F_3^n) ≥ M^{n/k}
for n a multiple of k, i.e. base M^{1/k}: 20^{1/4} ≈ 2.1147, 112^{1/6} ≈ 2.1955.
Calderbank–Fishburn (1994) refined the construction (gluing rather than a raw
Cartesian power) to base ≈ 2.2104. Edel (2004) introduced a more elaborate
product — a union of caps satisfying compatibility conditions is a cap — reaching
base ≈ 2.217389, which stood as the best principled lower bound for more than a
decade. Gap: these are explicit constructions bounded above by what the small
base caps allow; they leave a wide multiplicative gap to any upper bound and tell
us nothing about why an upper bound below 3^n should hold.

## Evaluation settings

The yardstick is the asymptotic growth rate of r_3(F_3^n) as n → ∞, reported as
the base c in c^n (with sub-exponential factors ignored), and the explicit
constant where one can be named. The relevant regimes: the trivial upper bound
3^n; the Fourier ceiling 3^n / poly(n); and the constructive lower bounds c^n
with c around 2.2. A meaningful new upper bound must have exponential rate
c < 3 and, ideally, a closed-form c. Sanity checks live in tiny dimensions
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
