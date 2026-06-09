# Context: the minimum-shadow problem for uniform set families

## Research question

Fix a uniformity $k$ and a count $m$. Take any family $\mathcal{F}$ of $m$ distinct $k$-element
sets. Each $k$-set has $k$ subsets of size $k-1$; collect all of them across the family and call the
result the **shadow** $\partial\mathcal{F}$ — the $(k-1)$-uniform family of all $(k-1)$-subsets that
sit inside some member of $\mathcal{F}$. The question is: **how small can $|\partial\mathcal{F}|$ be,
given $|\mathcal{F}| = m$?**

This is the discrete isoperimetric question on a single layer of the Boolean lattice: a family is a
region on the $k$-th layer, the shadow is its lower boundary, and we want the region whose boundary is
as small as possible. The maximisation direction is uninteresting — every $k$-set contributes $k$
subsets, so $|\partial\mathcal{F}| \le k|\mathcal{F}|$, with equality when the sets are pairwise
disjoint. The minimisation direction is the hard and useful one.

It matters because the shadow controls the "cost of the boundary" of any set system, and that cost
shows up everywhere a downward-closed structure does. The motivating instance is the characterisation
of which integer vectors $(f_0, f_1, f_2, \dots)$ can arise as the face-count vector of an abstract
simplicial complex — number of vertices, edges, triangles, and so on. A complex is downward closed, so
the number of $(k-1)$-faces is at least the size of the shadow of its $k$-faces. A tight lower bound on
the shadow would turn "which $f$-vectors are realisable?" into an explicit arithmetic condition. The
same minimum-shadow quantity is what one needs to deduce sharp bounds on intersecting families and
antichains. So the precise minimum of $|\partial\mathcal{F}|$ as a function of $m$ and $k$ is the
object worth pinning down — not just its order of magnitude, but the exact value and the families that
attain it.

## Background

The arena is the Boolean lattice $B_n = (\mathcal{P}([n]), \subseteq)$ sliced into layers
$X^{(r)} = \{A \subseteq [n] : |A| = r\}$. For a family $\mathcal{A} \subseteq X^{(r)}$ the lower
shadow is $\partial\mathcal{A} = \{B \in X^{(r-1)} : B \subset A \text{ for some } A \in \mathcal{A}\}$
and the upper shadow is its dual on $X^{(r+1)}$. The shadow is intrinsic to the family: it does not
depend on the ambient ground set, only on the abstract sets.

A first, crude bound comes from double counting. Each $k$-set has exactly $k$ shadow-subsets, and each
$(k-1)$-set is a subset of at most $n-(k-1)$ different $k$-sets over $[n]$, so
$k|\mathcal{F}| \le (n-k+1)|\partial\mathcal{F}|$, giving $|\partial\mathcal{F}| \ge
\frac{k}{n-k+1}|\mathcal{F}|$. This is linear in $|\mathcal{F}|$ and depends on $n$; it is far from
tight and it does not even identify the extremal configuration. The real answer is sublinear and
$n$-free, and the whole difficulty is getting it exactly.

A heuristic for the extremal shape is concentration: to keep the boundary small, one wants the sets to
share their $(k-1)$-subsets as much as possible, which suggests packing them onto as few elements as
possible. The cleanest packing is a **clique**: take all $k$-subsets of a small ground set $[a]$, i.e.
$\mathcal{F} = [a]^{(k)}$ with $|\mathcal{F}| = \binom{a}{k}$. Its shadow is all of $[a]^{(k-1)}$, of
size $\binom{a}{k-1}$ — every $(k-1)$-subset of $[a]$ is reused by many $k$-sets. But a clique only
exists when $m$ is exactly a binomial coefficient $\binom{a}{k}$; for a general count $m$ there is no
obvious canonical family, and even at round counts "the clique looks best" is not a proof.

Several standard objects from finite-set combinatorics are available as raw material. Numbers admit
**binomial (cascade) representations** — "base-binomial" analogues of binary expansion writing a
number as a sum of binomial coefficients with strictly decreasing top arguments and indices. And one
can order $k$-sets in various ways; among them are the **lexicographic order** (by the smallest
differing element) and the **colexicographic order**, $A < B \iff \max(A \triangle B) \in B$, in which
sets containing larger elements come later. Whether any of these is the right language for the problem
is not settled here. Both were familiar objects in the field.

The field around this question — extremal set theory of the early-to-mid 1960s — already had two of
its central pillars in place. **Sperner's theorem** (1928) bounds the size of an antichain in $B_n$ by
the middle binomial coefficient. The **Erdős–Ko–Rado theorem** (Erdős, Ko, Rado 1961) bounds an
intersecting family of $r$-sets by $\binom{n-1}{r-1}$ for $n \ge 2r$. Both are statements about a
single layer or about how layers interlock, and both are really statements about shadows and
co-shadows in disguise — which is precisely why a sharp shadow inequality would subsume and refine
them. The prevailing technique of the era was clever direct counting (the EKR original proof, the
permutation/cycle method); the systematic "make the family more extremal without making it worse"
machinery — shifting and compression — was only beginning to be articulated as a general tool.

## Baselines

These are the prior results and techniques a sharp shadow bound would have to beat or build on.

- **The double-counting / averaging bound.** As above, $|\partial\mathcal{F}| \ge
  \frac{k}{n-k+1}|\mathcal{F}|$. Core idea: count incidences between $k$-sets and their
  $(k-1)$-subsets two ways. Gap: it is linear, it depends on $n$, and it is loose by a wide margin —
  the true minimum is governed by binomial coefficients, not a linear ratio, and is attained on a
  clique, which this bound does not see.

- **The clique guess.** Conjecture that the minimiser is $[a]^{(k)}$ when $m = \binom{a}{k}$, giving
  $|\partial\mathcal{F}| \ge \binom{a}{k-1}$. Core idea: maximal sharing of subsets. Gap: it only
  covers the special counts $m = \binom{a}{k}$ and says nothing in between; and "it looks optimal" is
  not a proof — extremal combinatorics is full of plausible guesses that fail, so the clique guess
  needs both an exact extension to all $m$ and a proof that no cleverer family does better.

- **Sperner's theorem (1928).** An antichain in $B_n$ has size at most $\binom{n}{\lfloor n/2\rfloor}$.
  Core idea: chain decomposition / the LYM inequality $\sum_A 1/\binom{n}{|A|} \le 1$. Gap for the
  present purpose: it controls *how many* sets an antichain can have in total but says nothing about
  the boundary between adjacent layers, i.e. nothing quantitative about shadows of fixed-size families.

- **Erdős–Ko–Rado (Erdős, Ko, Rado 1961).** An intersecting family $\mathcal{A} \subseteq X^{(r)}$
  with $2 \le r < n/2$ has $|\mathcal{A}| \le \binom{n-1}{r-1}$, attained by the star of all $r$-sets
  through a fixed point. Core idea: the cyclic-permutation counting argument. Gap: it is one specific
  intersection statement; it does not provide the general layer-to-layer boundary inequality, although
  (as becomes clear later) it is exactly the kind of result a sharp shadow bound should yield as a
  corollary by passing to complements.

- **The shifting/compression heuristic (in embryo).** The operation "if a set uses a large element,
  swap it for a smaller unused one" preserves cardinality and intuitively makes a family more
  concentrated. Core idea: monotone transformations toward a canonical extremal family. Gap at this
  point: it was folklore-level rather than a packaged theorem — its effect on the shadow had not been
  analysed, and it had not been turned into a proof of any extremal bound.

## Evaluation settings

The natural yardsticks are exactness and attainment, not a benchmark number. A proposed bound is
measured against:

- **All $(m,k)$, not just nice ones.** The bound must be stated for every count $m$ and every
  uniformity $k$, not only for the round counts $m = \binom{a}{k}$ where a clique exists.

- **Tightness / extremal families.** For each $m$ there must be an explicit family achieving the
  bound, and ideally a characterisation of *when* the extremal family is unique up to relabelling
  (e.g. whether it is forced to be a clique at $m = \binom{a}{k}$).

- **Closed-form usability.** Ideally a version that is easy to compute with and to plug into other
  proofs, rather than only an exact-but-cumbersome arithmetic condition.

- **Downstream corollaries.** Whether the bound recovers known landmark results — the Erdős–Ko–Rado
  theorem and the characterisation of $f$-vectors of simplicial complexes — since a correct sharp
  shadow inequality should imply them cleanly.

## Code framework

The computational scaffold is small: enumerate $k$-subsets, compute a shadow by brute force, and
evaluate binomial coefficients. The missing slots are the predicted minimum shadow and the family that
attains it.

```python
from itertools import combinations
from math import comb

def shadow(family):
    """Lower shadow: all (k-1)-subsets contained in some k-set of the family."""
    out = set()
    for S in family:
        S = frozenset(S)
        for x in S:
            out.add(frozenset(S - {x}))
    return out

def all_k_sets(n, k):
    """The k-th layer of the Boolean lattice over [n]."""
    return [frozenset(c) for c in combinations(range(1, n + 1), k)]

def min_shadow_size(m, k):
    """
    TODO: the minimum of |shadow(F)| over all families F of m k-sets,
    as a function of m and k.
    """
    pass

def extremal_family(m, k):
    """
    TODO: an explicit family of m k-sets attaining the minimum shadow.
    """
    pass

def brute_force_min_shadow(n, m, k):
    """Reference: search all m-subsets of the k-th layer for the smallest shadow."""
    best = None
    for fam in combinations(all_k_sets(n, k), m):
        s = len(shadow(fam))
        if best is None or s < best:
            best = s
    return best
```
