# Context: assigning distinct representatives to a family of sets

## Research question

I am handed a finite collection of sets $T_1, T_2, \dots, T_n$, all drawn from a common finite ground set $S$, and I am asked to pick one element from each set so that the chosen elements are all *distinct*: a tuple $(a_1, \dots, a_n)$ with $a_i \in T_i$ for every $i$ and $a_i \neq a_j$ whenever $i \neq j$. Such a tuple is called a **system of distinct representatives** (an SDR), and the underlying set $\{a_1, \dots, a_n\}$ is a **transversal** of the family.

The blunt question is: *when does such a tuple exist?* For some families it plainly does, for others it plainly does not, and I want a clean **necessary and sufficient condition** — one that I can check, and that tells me *why* an assignment is impossible when it is.

This matters because the same combinatorial object reappears everywhere. The "marriage" phrasing is one instance: $n$ people, each with a set of acceptable partners; can everyone be paired off simultaneously with someone acceptable, no partner shared? A scheduling instance: $n$ tasks, each runnable only on a subset of machines; can each task get its own machine? A representation instance from algebra: a subgroup $H$ of finite index $n$ in a group $G$ has $n$ left cosets and $n$ right cosets — can I choose $n$ elements that are *simultaneously* a left-coset transversal and a right-coset transversal? Each of these is the SDR question wearing different clothes. A single sharp criterion would settle all of them at once.

What a solution must achieve: (i) a condition on the *family* that is **equivalent** to the existence of an SDR — not merely sufficient, not merely necessary; (ii) a condition that is **structural** — it should point at the obstruction when no SDR exists, not just announce failure; (iii) ideally, a condition whose verification or whose witness-of-failure is something I can actually *produce*, not just assert.

## Background

The natural arena is the **bipartite graph**. Put the indices $\{1, \dots, n\}$ on one side (call it $A$) and the ground-set elements $S$ on the other side (call it $B$), and draw an edge from index $i$ to element $x$ exactly when $x \in T_i$. Then an SDR is precisely a set of edges, one per index, no two sharing an endpoint — i.e. a set of pairwise-disjoint edges that touches every vertex of $A$. A set of pairwise-disjoint edges is a **matching**; one that touches every vertex of a given side is said to **saturate** that side. So:

> *an SDR for $(T_1, \dots, T_n)$ exists $\iff$ the bipartite graph $G$ has a matching that saturates $A$.*

This translation is the load-bearing reframing. It lets me reason about edges and vertices instead of tuples of representatives, and it connects the SDR question to a body of early-twentieth-century work on bipartite graphs and 0/1 matrices.

The vocabulary I have available:

- A **matching** $M$ is a set of edges no two of which share a vertex. A vertex with no edge of $M$ on it is **exposed** (or unmatched, or open); a vertex with an edge of $M$ on it is **matched** or **saturated**. A matching is **perfect** if no vertex is exposed.
- The **neighborhood** of a set $S \subseteq A$ is $N(S) = \{\, b \in B : b \text{ is joined to some } a \in S \,\}$.
- A **vertex cover** $C$ is a set of vertices touching every edge: no edge has both endpoints outside $C$.
- An **alternating path** with respect to a matching $M$ is a path whose edges alternate between edges of $M$ and edges not in $M$.

Several known facts sit in this background, all from before or around the time the SDR question is being asked:

- **Weak duality between matchings and covers.** Any matching $M$ and any vertex cover $C$ satisfy $|M| \le |C|$, because each edge of $M$ needs a distinct cover vertex (the edges of a matching share no endpoints, so one cover vertex cannot pay for two of them). This is a one-line lower bound on covers and upper bound on matchings, but it is only an *inequality*.
- **König 1931 and Egerváry 1931.** Dénes König proved that *for bipartite graphs the weak inequality is an equality*: the maximum size of a matching equals the minimum size of a vertex cover. Jenő Egerváry proved the equivalent statement in the language of 0/1 (and weighted) matrices, on the nonvanishing of a generalized term. These are minimax theorems: an optimum is certified by a dual object of equal size.
- **Frobenius 1917.** In matrix language, a square 0/1 matrix has a nonzero term in its determinant expansion (a permutation hitting only ones) exactly under a rank-type condition on its supports — an algebraic ancestor of the same phenomenon.
- **Menger 1927.** Karl Menger proved that the maximum number of internally/edge-disjoint paths between two vertices equals the minimum size of a separating cut. Bipartite matching embeds into this by adding a source joined to all of $A$ and a sink joined to all of $B$; a system of disjoint source-to-sink paths is a matching. So Menger's min-cut/max-paths duality is a more general minimax that contains the matching one.

The recurring shape of all of these is **minimax / duality**: a maximum primal object equals a minimum dual object. That is the prevailing wisdom of the field at this moment — optimality is proved by exhibiting a dual certificate of equal value.

There is also a known *negative* fact that bounds the ambition: the SDR question genuinely depends on **finiteness**. For infinite families the situation degrades — one can write down an infinite family in which every finite subcollection looks fine yet no global system of distinct representatives exists — so any clean criterion I find should be understood as a statement about finite families.

## Baselines

These are the tools already on the table that a criterion for SDRs would be measured against or built from.

**König's minimax theorem (1931).** *In a bipartite graph, $\max |M| = \min |C|$*, the maximum matching size equals the minimum vertex-cover size. Core content: optimality of a matching is certified by a cover of equal size. As a route to the SDR question it is powerful — if I could compute or bound the minimum cover I would know the maximum matching, and an $A$-saturating matching exists iff the maximum matching has size $n = |A|$. **Limitation it leaves open:** König's theorem speaks about the *sizes* of optimal matchings and covers; it does not, by itself, hand me a condition phrased directly in terms of the *family of sets* — a statement of the form "the family has an SDR iff [some property of the $T_i$]." It tells me an SDR exists iff the minimum cover has size $n$, but "the minimum cover has size $n$" is not yet a transparent, checkable condition on the neighborhoods $N(S)$. There is a gap between the size-of-cover certificate and a condition I can read off the sets themselves.

**Egerváry's matrix theorem (1931).** The same minimax in 0/1- and weighted-matrix language: a condition for an injective choice of a nonzero entry per row. Core content identical to König's, transposed to matrices. **Limitation:** same as König — it is a statement about matrices and term-rank, one further translation step away from "any $k$ of the sets contain at least $k$ elements," and it does not localize the obstruction.

**Menger's theorem (1927).** $\max$ edge/vertex-disjoint $s$–$t$ paths $=\min$ separating cut. Core content: the most general minimax in this family; matching is a special case via the source/sink gadget. **Limitation:** it is *more* general than I need, so it carries overhead — to use it I must build the flow network and reason about cuts, and the resulting "cut" certificate, when pushed back through the gadget, is again a cover-like object, not a clean condition on the sets $T_i$. It proves existence but does not, in raw form, produce the set-system obstruction.

**Frobenius's determinant condition (1917).** A rank/support condition for a square 0/1 matrix to have a nonzero diagonal term under some permutation. Core content: the algebraic precursor. **Limitation:** stated for the *square* (perfect-matching) case and in determinant language; it does not directly give the asymmetric "saturate $A$" version for $n$ sets over a possibly larger ground set, and it is an algebraic existence statement rather than a combinatorial witness.

**Greedy / exhaustive assignment.** The naive baseline: try to assign representatives one index at a time, backtracking when stuck; or, for the decision question, search over assignments. Core content: it is correct and it terminates. **Limitation:** it is opaque and expensive — when it fails it gives no *reason*, only "no assignment found after searching," and in the worst case the search is exponential. It provides no insight into *which* sets are the obstruction.

The common thread across the minimax baselines: each proves *that* an optimal matching is as large as some dual object, but none of them is yet a condition phrased on the **collection of sets and their unions** — a condition that, on failure, fingers the specific subcollection responsible.

## Evaluation settings

This is a mathematical existence question, so the "evaluation" is by proof and by canonical instances rather than by benchmark numbers. The natural yardsticks that exist already:

- **Small families one can check by hand.** Families like $T_1 = \{a\}$, $T_2 = \{a\}$ (obstructed — two sets, one element between them) versus $T_1 = \{a\}$, $T_2 = \{a, b\}$ (an SDR exists). The criterion must give the right verdict on these and on slightly larger hand-built families designed to be tight.
- **The bipartite-graph / 0/1-matrix instances** used for König's and Egerváry's theorems, where one already knows the maximum matching and minimum cover, so a new criterion can be cross-checked against the known optimum.
- **The canonical applications as test cases:** the coset-transversal problem for a finite-index subgroup; the marriage/assignment phrasing; later, the extension of a partial Latin rectangle to a full Latin square (where each column forbids the letters already used, and a new row is exactly an SDR of the "still-available" sets). A criterion is judged by whether it cleanly resolves these.
- **Tight (extremal) families** where the criterion holds with no slack — exactly $|N(S)| = |S|$ for some $S$ — since these are where a correct condition must still permit an SDR and an incorrect one would fail.

The metric throughout is correctness of the equivalence (no false positives, no false negatives) and the *informativeness* of the witness when an SDR does not exist.

## Code framework

This problem is settled by a theorem and its proof, not by a program; there is no benchmark to run. But the proof I am after is constructive — it should either produce a saturating assignment or produce the obstruction — so it is useful to lay out a minimal computational scaffold in pre-method vocabulary, mirroring the structure the final argument will take. The pieces below are the generic, already-available primitives (a bipartite graph, matchings, neighborhoods, alternating-path search); the one empty slot is the existence criterion and the routine that decides it.

```python
from collections import defaultdict

class BipartiteGraph:
    """Indices A on one side, ground-set elements B on the other.
    adj[a] = the set T_a of elements joined to index a."""
    def __init__(self, A, B, adj):
        self.A = list(A)
        self.B = list(B)
        self.adj = {a: set(adj[a]) for a in A}   # adj[a] == T_a

    def neighborhood(self, S):
        """N(S): every element joined to some index in S."""
        out = set()
        for a in S:
            out |= self.adj[a]
        return out


def is_matching_saturating_A(graph, M):
    """A matching (dict index->element, injective) saturates A iff it
    covers every index in A."""
    if len(set(M.values())) != len(M):
        return False
    return set(M.keys()) == set(graph.A)


def alternating_search(graph, M, root):
    """Generic prior-art primitive: from an exposed index `root`, explore
    the graph respecting the matching M (alternate non-matching / matching
    edges). Returns the structure reachable this way -- the raw material any
    matching argument works from. (Standard alternating exploration.)"""
    # ... breadth-first alternation from root, standard ...
    pass


# --- the slot the discovery fills -----------------------------------------

def admits_SDR(graph):
    """Decide whether the family (T_a) has a system of distinct
    representatives, and if not, expose why.
    TODO: the criterion we will discover, and the procedure that
          either returns a saturating matching or returns the obstruction."""
    pass
```

The final argument will fill `admits_SDR` (and use `alternating_search` and `neighborhood` as its working parts); these stubs are exactly the slots the discovered criterion and its constructive proof occupy.
