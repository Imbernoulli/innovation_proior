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

The guiding intuition for the extremal shape is concentration: to keep the boundary small, pack the
sets onto as few elements as possible, because overlapping subsets get shared. The cleanest packing is
a **clique**: take all $k$-subsets of a small ground set $[a]$, i.e. $\mathcal{F} = [a]^{(k)}$ with
$|\mathcal{F}| = \binom{a}{k}$. Its shadow is all of $[a]^{(k-1)}$, of size $\binom{a}{k-1}$ — every
$(k-1)$-subset of $[a]$ is reused by many $k$-sets, so the boundary is as economical as possible.
When $m$ is not exactly a binomial coefficient $\binom{a}{k}$, the natural move is to take the largest
clique that fits, then a clique one layer thinner on top of it, and so on — a **greedy clique
decomposition** of $m$.

Two pieces of standard machinery make this precise. The first is the **binomial (cascade)
representation** of a natural number: a "base-binomial" analogue of binary expansion in which $m$ is
written as a sum of binomial coefficients with strictly decreasing top arguments and indices. The
second is the **colexicographic order** on $k$-sets, $A < B \iff \max(A \triangle B) \in B$, in which
sets containing larger elements come later; its initial segments are exactly the greedy clique-stacks,
and they are independent of the ambient $n$. Both were familiar objects in finite-set combinatorics.

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
  point: it was folklore-level rather than a packaged theorem; nobody had yet shown that some such
  operation provably never increases the shadow and drives an arbitrary family all the way to the
  colex initial segment. Establishing exactly that is the work.

## Evaluation settings

The natural yardsticks are exactness and attainment, not a benchmark number. A proposed bound is
measured against:

- **All $(m,k)$, not just nice ones.** The bound must be stated for every count $m$ and every
  uniformity $k$, via the cascade representation, and must reduce to $\binom{a}{k-1}$ when
  $m = \binom{a}{k}$.

- **Tightness / extremal families.** For each $m$ there must be an explicit family achieving the
  bound — the candidate being the first $m$ sets in colex order — and ideally a characterisation of
  *when* the extremal family is unique up to relabelling (e.g. forced to be a clique at
  $m = \binom{a}{k}$).

- **Closed-form usability.** A version that is easy to compute with — extending binomial coefficients
  $\binom{x}{k} = x(x-1)\cdots(x-k+1)/k!$ to real $x$ — so the bound can be plugged into other proofs
  without manipulating cascades, agreeing with the exact bound at the integer points $m = \binom{n}{k}$.

- **Downstream corollaries.** Whether the bound recovers known landmark results — the Erdős–Ko–Rado
  theorem via complementation, and the characterisation of $f$-vectors of simplicial complexes — since
  a correct sharp shadow inequality should imply them cleanly.

## Code framework

The computational scaffold is small: enumerate $k$-subsets, compute a shadow by brute force, evaluate
binomial coefficients, and order sets. The missing slots are the closed-form predicted minimum shadow
and the extremal construction.

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

def order_key(S):
    # TODO: the order on k-sets whose initial segments are the extremal families.
    pass

def min_shadow_size(m, k):
    """
    TODO: the closed-form minimum of |shadow(F)| over all families F of m
    k-sets. Should depend only on m and k, not on the ambient n, and equal
    comb(a, k-1) when m == comb(a, k).
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
