# Context — Largest intersecting families of equal-sized sets

## Research question

Fix a ground set `[n] = {1, 2, …, n}`. A family `F` of subsets is called **intersecting** if every two of its members share at least one element: `A ∩ B ≠ ∅` for all `A, B ∈ F`. The question is extremal: **how large can an intersecting family be?**

Unrestricted, the answer is easy and not very interesting. Pair each subset `A ⊆ [n]` with its complement `[n] \ A`; the two are disjoint, so an intersecting family can hold at most one of every such pair. Hence `|F| ≤ 2^{n−1}`, and the bound is met by, e.g., all sets containing a fixed element. The half-of-everything answer is forced by complementation and tells us little about the *structure* of intersecting families.

The problem comes alive when all members are required to have the **same size** `k`. So: among all families of `k`-element subsets of `[n]` that are pairwise intersecting, what is the **maximum cardinality**, and which families achieve it? This is the central extremal question for `k`-uniform intersecting set systems. It matters because `k`-uniform intersecting families are the combinatorial backbone of countless results — they model "any two committees share a member", coding-theoretic distance constraints, and they are the prototype on which the whole field of extremal set theory tests its tools. A clean, tight bound with a clean extremal characterization is the goal; a bound that is correct but whose proof does not *explain* the answer (why this number, why this hypothesis) is unsatisfying.

There is one regime to dispose of immediately. If `n < 2k`, then any two `k`-subsets of `[n]` already intersect (two disjoint `k`-sets would need `2k > n` points), so *every* `k`-uniform family is intersecting and the maximum is the trivial `\binom{n}{k}`. The interesting regime is therefore `n ≥ 2k`, where being intersecting is a genuine restriction.

## Background

**A natural large intersecting family — the "star."** Fix one element `x ∈ [n]` and take every `k`-set that contains `x`:
`S_x = { A ∈ \binom{[n]}{k} : x ∈ A }.`
Any two such sets share `x`, so `S_x` is intersecting, and `|S_x| = \binom{n−1}{k−1}` (choose the remaining `k−1` elements from the other `n−1`). This is the obvious construction and the obvious conjecture is that, for `n ≥ 2k`, no intersecting `k`-family beats it: the maximum is `\binom{n−1}{k−1}`. The conjecture is supported by the small cases one can check by hand, but a *proof* for all `n, k` is what is wanted, together with the question of whether the star is the *only* optimum.

**The complementary-pair phenomenon at the boundary.** Exactly at `n = 2k`, each `k`-set has a unique disjoint complement, and `\binom{2k}{k}` sets split into `\tfrac12\binom{2k}{k}` complementary pairs. Picking one set from each pair gives an intersecting family of size `\tfrac12\binom{2k}{k} = \binom{2k−1}{k−1}` — the same as the star bound, but generally *not* a star. So whatever the answer is, the extremal *structure* must be different at `n = 2k` than for `n > 2k`. This is a pre-method fact about the design space that any complete account has to respect.

**Sperner's theorem and the antichain template (Sperner 1928).** The model result for this whole line of questions is Sperner's: the largest **antichain** in the Boolean lattice `2^{[m]}` (a family no member of which contains another) has size `\binom{m}{⌊m/2⌋}`, the middle binomial coefficient, achieved by taking all subsets of size `⌊m/2⌋`. Two features of Sperner's proof are part of the standard toolkit here. (i) The **LYM inequality**: for any antichain `A`, `\sum_{\ell} |A_\ell| / \binom{m}{\ell} ≤ 1`, where `A_\ell` is the part of `A` at level `\ell`; the bound on `|A|` follows by maximizing this weighted sum. (ii) **Symmetric chain decomposition**: `2^{[m]}` decomposes into saturated chains symmetric about the middle level, and an antichain meets each chain at most once, so its size is at most the number of chains, which equals the middle rank number. Both are *normalizing/averaging* ideas — weighting each set by `1/\binom{m}{\ell}`, or distributing sets across structured chains — that turn a global extremal count into a local per-chain or per-level statement.

**Counting in two ways.** The other standard tool is double counting: count the same set of incidences two different ways and equate. Sperner's own auxiliary lemma (the form invoked in the standard treatment of intersecting families) is of exactly this type: for the `\ell_0`-sets in a family and the `(\ell_0+1)`-sets that cover them, counting covering pairs both ways yields an inequality `n_0 (m − \ell_0) ≤ n_1 (\ell_0 + 1)` relating the count of small sets to the count of their covers. Such shadow/cover inequalities are the engine of the existing proofs of the intersecting-family bound.

**Vertex-transitivity of the intersection graph.** Build the graph `G` whose vertices are the `\binom{n}{k}` many `k`-subsets, with two joined when they intersect. An intersecting family is exactly a **clique** of `G`. Any permutation of `[n]` permutes the `k`-sets and preserves intersection, so `G` is **vertex-transitive**: the automorphism group acts transitively on vertices, and every `k`-set "looks the same" from the group's point of view. Vertex-transitive graphs support clean clique/coclique bounds via averaging over the automorphism group — a structural symmetry that is available here for free, before any clever construction.

## Baselines

**The induction-plus-compression proof (the proof on the table).** The known route to the bound is an induction. One fixes the parameters and chooses, among all intersecting families of the given size, one that minimizes a potential (the total sum of all elements over all sets), forcing the family into a "compressed"/"pushed-down" canonical position. Then a covering lemma of Sperner type (count, two ways, the pairs of an `\ell_0`-set and an `(\ell_0+1)`-set containing it, giving `n_0(m − \ell_0) ≤ n_1(\ell_0+1)`) lets one trade the family for a related family on a smaller ground set or smaller subset size, and a double induction — over the ground-set size and over the gap between the actual and the minimal intersection size — drives the bound `\binom{n−1}{k−1}` down to base cases. The method is correct and it is the standard reference proof. **Its limitations**: the argument is long and case-heavy (it splits into multiple sub-cases at each induction step), and the compression bookkeeping obscures the answer — at the end one has *verified* `\binom{n−1}{k−1}` but the proof does not make transparent *why* this is the number, nor *why* the hypothesis is exactly `n ≥ 2k`. The constant `\binom{n−1}{k−1}` and the threshold `2k` fall out of the induction rather than being explained by it. It also handles the extremal-structure question (uniqueness of the star, the boundary at `n = 2k`) only with extra work layered on top.

**Sperner's theorem as a baseline bound.** Sperner gives `\binom{m}{⌊m/2⌋}` for the *antichain* problem, not the intersecting problem; an antichain need not be intersecting and vice versa, so Sperner does not directly bound `|F|`. **Its limitation here**: it is the right *shape* of result (a tight extremal bound with a clean extremal family) and supplies the toolkit (LYM, chains, double counting), but the constraint it handles — no containment — is different from intersection, so its proof does not transfer mechanically. It is the aspirational template, not a solution.

**The trivial half-of-everything bound.** For unrestricted intersecting families, complementation gives `|F| ≤ 2^{n−1}`. **Its limitation**: it ignores uniformity entirely; restricting to size `k` it only says `|F| ≤ \binom{n}{k}`, which is far above `\binom{n−1}{k−1}` (a factor `n/k`) and says nothing for the `k`-uniform problem.

## Evaluation settings

This is a theorem-proving problem, so the "evaluation" is the standard for an extremal combinatorics result. A complete solution is judged on: (1) **correctness and tightness** — a bound on `|F|` that is met by an explicit construction (so it cannot be improved), valid for all `n ≥ 2k` and all `1 ≤ k`; (2) **extremal characterization** — identifying exactly which families attain the bound, including the qualitatively different situation expected at the boundary `n = 2k` versus `n > 2k`; (3) **explanatory force** — whether the proof makes the value of the constant and the role of the hypothesis `n ≥ 2k` transparent, rather than producing them as the output of opaque bookkeeping; (4) **economy** — brevity and the avoidance of long case analyses, the implicit yardstick against which the induction proof is found wanting. The natural test instances are small parameters one can enumerate by hand or by brute force — e.g. `n = 4, k = 2` (bound `\binom{3}{1} = 3`), `n = 5, k = 2` (bound `\binom{4}{1} = 4`), `n = 6, k = 3` (boundary case `n = 2k`, bound `\binom{5}{2} = 10`) — used to confirm a candidate bound and to inspect whether the extremal families are stars or boundary constructions.

## Code framework

The artifact here is a proof, not a program; but a candidate bound and a candidate extremal characterization can be sanity-checked by enumeration on small parameters. The pre-method scaffold below uses only generic primitives — generating `k`-subsets, testing the intersecting property, computing binomial coefficients — and leaves one empty slot for whatever counting argument the proof will turn out to rest on.

```python
from itertools import combinations
from math import comb

def k_subsets(n, k):
    """All k-element subsets of {0,...,n-1} as frozensets."""
    return [frozenset(s) for s in combinations(range(n), k)]

def is_intersecting(family):
    """True iff every two sets in the family share an element."""
    fam = list(family)
    for i in range(len(fam)):
        for j in range(i + 1, len(fam)):
            if not (fam[i] & fam[j]):
                return False
    return True

def star(n, k, x=0):
    """The family of all k-subsets containing the fixed point x."""
    return [s for s in k_subsets(n, k) if x in s]

def max_intersecting_by_search(n, k):
    """Brute-force largest intersecting k-family of [n] (tiny n,k only)."""
    sets = k_subsets(n, k)
    best = 0
    # exhaustive / greedy search over subfamilies; feasible only for small n,k
    # used purely to check a conjectured bound on small cases
    raise NotImplementedError  # a brute-force check value, not the argument itself

def conjectured_bound(n, k):
    # TODO: the quantity the proof will show bounds |F| from above
    pass

def bound_argument(n, k):
    """
    The core of the proof, to be filled in.
    Whatever structure the proof introduces to bound |F| for an
    intersecting k-uniform family on [n] (n >= 2k) goes here, and the
    pieces below it (verification on small cases) check it.
    """
    # TODO: the argument that yields the upper bound on |F|
    pass

def verify_small_cases():
    """Check the conjectured bound against brute force for small n, k."""
    for (n, k) in [(4, 2), (5, 2), (6, 3)]:
        assert len(star(n, k)) == comb(n - 1, k - 1)
        # TODO: compare conjectured_bound(n, k) with the search optimum
```
