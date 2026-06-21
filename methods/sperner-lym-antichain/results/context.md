# Context: how large can a family of subsets be if none contains another?

## Research question

Fix a finite ground set $[n] = \{1, 2, \dots, n\}$ and look at its power set $2^{[n]}$, the collection of all $2^n$ subsets. Call a family $\mathcal{F} \subseteq 2^{[n]}$ an **antichain** (a "distinguished system" of subsets) if no member of $\mathcal{F}$ is contained in another: for all distinct $A, B \in \mathcal{F}$, neither $A \subseteq B$ nor $B \subseteq A$. The containment relation $\subseteq$ partially orders $2^{[n]}$; an antichain is a set of pairwise *incomparable* subsets.

The question is purely extremal: **how large can an antichain in $2^{[n]}$ be?** Write $w(n)$ for the maximum possible $|\mathcal{F}|$. We want $w(n)$ exactly — a closed form — together with a characterization of *which* families attain it.

Why it matters. This is the most basic quantitative fact about the subset-containment order, and it is the seed of an entire subject (extremal set theory). A clean answer would say precisely how much "incomparable information" the subset lattice can hold, and the method that yields it tends to generalize to many neighboring extremal problems (intersecting families, shadows, profiles of set systems). The difficulty is that an antichain can mix subsets of many different sizes, so naive size-by-size counting does not immediately bound the total.

## Background

**The subset lattice as a graded poset.** Partition $2^{[n]}$ into $n+1$ *layers* (or *levels*) by cardinality: layer $k$ is $\binom{[n]}{k}$, the set of all $k$-element subsets, of which there are $\binom{n}{k}$. Containment only ever goes up in cardinality, so any single layer $\binom{[n]}{k}$ is automatically an antichain: two distinct $k$-sets are never nested. This gives an immediate **lower bound** $w(n) \ge \max_k \binom{n}{k}$.

**Where is $\binom{n}{k}$ largest?** The binomial coefficients along a row of Pascal's triangle rise and then fall — they are *unimodal*. The single largest entry sits in the middle: at $k = \lfloor n/2 \rfloor$ for even $n$ (a unique peak), and tied at $k = \tfrac{n-1}{2}$ and $k = \tfrac{n+1}{2}$ for odd $n$. So the best single layer has size $\binom{n}{\lfloor n/2 \rfloor}$, and
$$
w(n) \ \ge\ \binom{n}{\lfloor n/2 \rfloor}.
$$
The whole problem is whether this lower bound is the truth — whether a cleverly *mixed* antichain, drawing sets from several layers at once, can ever beat the single best layer.

**Shadows and shades.** For a family $\mathcal{A}$ of $k$-sets, its **(lower) shadow** $\partial \mathcal{A}$ is the family of all $(k-1)$-sets contained in some member of $\mathcal{A}$; dually its **(upper) shade** is the family of all $(k+1)$-sets containing some member. Counting incidences between a layer and the layer just below it is a standard tool: each $k$-set has exactly $k$ subsets of size $k-1$, and each $(k-1)$-set is contained in exactly $n-(k-1)$ sets of size $k$. These incidence counts let one compare the size of a family to the size of its shadow, and they are the classical handle on "how a family spreads across adjacent layers."

**Double counting and pigeonhole.** The workhorse of finite combinatorics: count one collection of objects in two different ways and equate or compare the two totals; or, if many objects are forced into few boxes, some box is crowded. These are the only heavy tools the field had, and they are what one reaches for when a quantity must be bounded.

**Permutations of $[n]$.** There are $n!$ linear orderings of the ground set — a basic counted object always available to a combinatorial argument, and a natural thing to count against when a quantity built out of factorials needs an interpretation.

**Diagnostic facts about mixed antichains.** Small cases show mixing is real but constrained. For $n=2$: layers have sizes $1,2,1$; the largest antichain is $\{\{1\},\{2\}\}$, size $2 = \binom{2}{1}$, and any attempt to add $\emptyset$ or $\{1,2\}$ breaks incomparability. For $n=3$: layer sizes $1,3,3,1$; one can take all three $2$-sets (size $3$) or all three $1$-sets, but a mix like $\{\{1\},\{2,3\}\}$ is an antichain of size only $2$ and cannot be grown to beat $3$. The recurring observation: **whenever you place a set of size $k$ into an antichain, you forfeit all of its subsets and all of its supersets**, and the bookkeeping of that forfeiture across many sizes is exactly what makes the mixed problem hard to bound directly.

## Baselines

These are the approaches on the table for pinning $w(n)$.

**1. Layer-by-layer counting.** Let $a_k$ be the number of size-$k$ sets in an antichain $\mathcal{F}$, so $|\mathcal{F}| = \sum_k a_k$ with $a_k \le \binom{n}{k}$. Bounding each $a_k$ separately gives $|\mathcal{F}| \le \sum_k \binom{n}{k} = 2^n$, which is uselessly weak — it ignores the interaction *between* layers entirely. The gap it leaves: it never uses the antichain condition to couple the $a_k$ across different $k$, so it cannot see why you can't be large in every layer at once.

**2. The shadow / compression method.** Take an antichain and try to push all of its mass toward the middle layer without shrinking it. Concretely, replace the top-layer members (those of size $k > n/2$) by their lower shadow, and the bottom-layer members by their upper shade, repeating until everything sits in the middle. Using the incidence counts above, one shows the shadow of a family in a sufficiently high layer is at least as large as the family — so each replacement does not decrease the count and keeps the family an antichain — and after finitely many steps the family lives in a single middle layer, where its size is at most $\binom{n}{\lfloor n/2\rfloor}$. This is a genuine, complete route to the bound. Its costs: it requires a sharp shadow-size lemma with a delicate boundary case at $k$ near $(n+1)/2$; it proceeds by an induction over the replacement steps with careful verification that the antichain property survives each step; and it only ever yields the size bound $|\mathcal{F}| \le \binom{n}{\lfloor n/2\rfloor}$ as an inequality between integers, giving no extra structural information for free. It is heavy machinery for a one-number answer.

**3. Chain covers of the lattice.** Try to partition all of $2^{[n]}$ into disjoint chains (towers $S_0 \subset S_1 \subset \cdots$). If $2^{[n]}$ splits into $t$ disjoint chains, then any antichain meets each chain at most once, so $|\mathcal{F}| \le t$. This reduces the problem to "how few chains can cover the cube," but constructing an explicit economical chain partition is itself a nontrivial combinatorial construction, and absent such a construction this route gives no number at all. The gap: it converts the question into another existence problem rather than resolving it.

## Evaluation settings

This is a theorem, so the "evaluation" is mathematical adequacy, judged against fixed yardsticks rather than data:

- **Sanity on small $n$.** The claimed formula must reproduce the exhaustively checkable cases: $n=0$ ($w=1$), $n=1$ ($w=1$), $n=2$ ($w=2$), $n=3$ ($w=3$), $n=4$ ($w=6=\binom{4}{2}$).
- **Tightness.** The upper bound on $w(n)$ must match the layer lower bound $\binom{n}{\lfloor n/2\rfloor}$ — an inequality that does not meet the construction is only half an answer.
- **Equality characterization.** A complete answer also says *which* antichains are extremal, not merely how big they can be.
- **Robustness / generality.** A proof is judged partly by how cheaply it extends — to weighted versions, to the dual intersecting-family problems, and to the "profile" inequalities relating the $a_k$ — so a self-contained argument with few moving parts is preferred over one that needs a special construction or a delicate induction.

## Code framework

No training pipeline is involved; the artifact is a theorem with a proof. The only computational thing that is natural here is a brute-force checker over small $n$ — used purely to *sanity-check* a claimed formula and equality cases against ground truth, written entirely in pre-method vocabulary (subsets, the containment relation, the unimodal binomial coefficients). The reasoning will fill the one open slot: the actual upper-bound argument.

```python
from itertools import combinations
from math import comb

def all_subsets(n):
    U = range(n)
    return [frozenset(s) for k in range(n + 1) for s in combinations(U, k)]

def is_antichain(family):
    fam = list(family)
    for i in range(len(fam)):
        for j in range(len(fam)):
            if i != j and fam[i] <= fam[j]:   # fam[i] subset of fam[j] -> comparable
                return False
    return True

def largest_layer(n):
    # the obvious construction: the biggest single cardinality-layer
    k = n // 2
    return comb(n, k)

def brute_force_max_antichain(n):
    # exhaustive ground truth for tiny n: size of the largest antichain in 2^[n]
    subs = all_subsets(n)
    best = 0
    # (exponential; only for n <= 4 or 5, to validate a claimed formula)
    from itertools import combinations as C
    for r in range(len(subs) + 1):
        for fam in C(subs, r):
            if is_antichain(fam):
                best = max(best, r)
    return best

def upper_bound(n):
    # TODO: the argument that bounds every antichain in 2^[n] from above.
    #       Returns the proven upper bound on |F|; must match the construction.
    pass

def extremal_families(n):
    # TODO: the families that attain the maximum (the equality characterization).
    pass
```
