# Universal Optimality of Dijkstra via Beyond-Worst-Case Heaps

## Problem

Order the vertices of a directed graph `G` (`n` vertices, `m` arcs, source `s`, non-negative arc
lengths) by distance from `s` — the **distance order problem**. Dijkstra with a Fibonacci heap solves
it in `O(m + n log n)`, worst-case optimal, but provably wasteful on easy graphs (e.g. a path hanging
off a star). The goal is **universal optimality**: a single algorithm that, on *every* graph topology
`G`, costs within a constant factor of the best any correct algorithm achieves on `G`, where cost is
measured under a worst-case choice of weights (and, in the time model, of incidence-list order):

    ∃ c, ∀ G, ∀ correct A':   max_w cost_A(G, w)  ≤  c · max_w cost_{A'}(G, w).

## Key idea

Dijkstra's running time is `O(m)` plus the cost of its heap operations, and the delete-mins are the only
super-linear term. Replace the size-based delete-min bound with a **working-set bound**: a delete-min
returning `x` costs `O(log W(x))`, where the **working-set size** `W(x)` is the number of items inserted
during `x`'s residence in the heap (every other operation stays `O(1)` amortized, *including*
decrease-key). Then:

1. **The locality of Dijkstra makes `Σ_v log W(v) = O(log D)`**, where `D` is the number of distance
   orders of `(G, s)`. A parent and child in Dijkstra's search tree have disjoint, ordered residence
   windows; an interval-counting lemma turns the per-vertex recency costs into the topological quantity
   `log D`. So Dijkstra with such a heap runs in `O(m + log D)` time = **universally time-optimal**.
2. **A small fix closes the comparison gap.** Plain Dijkstra does `O(F + log D)` comparisons, off by an
   additive `O(n)` from the lower bound `Ω(F − n + 1 + log D)`; this matters only on bottleneck-heavy
   graphs (where `log D` is small). Keeping *bottlenecks* out of the heap — their order is forced and
   their distances propagate by additions — yields `O(F − n + 1 + log D)` comparisons = **universally
   comparison-optimal**.
3. **The required heap is built** by stacking Fibonacci-quality heaps whose sizes grow doubly-
   exponentially in recency, achieving the working-set bound *with* `O(1)` decrease-key — a combination
   no prior heap provided.

Here `D` = number of distance orders of `(G, s)`; `F` = max number of forward arcs over all distance
orders (`F = m` for undirected graphs); `log` is base 2; `W(x)` re-counts a re-inserted item as new.

## Lower bounds (the target)

**Time.** Any correct deterministic distance-order algorithm needs `Ω(m + log D)` time.
- `Ω(m)`: with the weighting `c(v_iv_j) = max(0, j − i)` making a chosen order unique, an unaccessed
  incidence entry can be lengthened-to-`1/2`/inserted to break the output; so essentially all of the
  incidence structure must be read — `≥ max(n−2, m−2n+2) = Ω(m)`.
- `Ω(log D)`: information theory. `D` weightings, each making a distinct order unique, force `D` distinct
  leaves in the comparison decision tree (perturb away `=0` outcomes), so depth `≥ ⌈log D⌉`.

**Comparisons.** `Ω(F − n + 1 + log D)`.
- `Ω(log D)`: as above.
- `Ω(F − n + 1)`: give forward arc `v_iv_j` length `j − i`, non-forward arcs a common huge length. A run
  of `≤ F − n` comparisons yields `≤ F − n` equality constraints; with the `n − 1` path-edge equations
  that is `< F` equations in `F` variables, so one can slide a forward-arc length to `0` (exactly one
  `v_iv_j`, `j > i+1`) without changing any comparison outcome, breaking the order — contradiction.

## Main theorems

**Theorem 1 (working-set heap ⇒ time optimality).** Dijkstra implemented with a heap having the
working-set bound runs in `O(m + log D)` time and `O(F + log D)` comparisons on any graph. Hence it is
universally optimal in time, and universally optimal in comparisons up to an additive `O(n)`.

*Proof.* Inserts (`n`) and decrease-keys (`≤ F − n + 1`) are `O(1)` each, contributing `O(m)`. It
remains to bound delete-mins by `O(n + log D)`. Number the vertices `v_1,…,v_n` by insertion order; for
`v_i` let `[a_i = i, b_i]` be the insertion-indices spanning its residence, so `W(v_i) = b_i − a_i + 1`.
In the search tree `T` (arc `u_iv_i` that first labeled `v_i`), every arc `v_iv_j` satisfies `b_i < a_j`
(the parent is deleted before the child is inserted), so it is an arc of the interval DAG `I`
(`[a_i,b_i] → [a_j,b_j]` iff `b_i < a_j`). Thus every topological order of `I` is one of `T`, hence a
distance order of `G`, giving `D ≥ D(I)`. By the interval lemma below,
`Σ_i log W(v_i) = Σ_i log(b_i − a_i + 1) = O(log D(I)) = O(log D)`. So delete-mins cost
`O(n + log D)`. ∎

**Lemma (interval bound).** For `n` integer intervals `[a_i, b_i] ⊆ [1, n]` with induced partial order
`P` (`i ≺ j` iff `b_i < a_j`) and `e(P)` linear extensions, `Σ_i log(b_i − a_i + 1) = O(log e(P))`.

*Proof.* Reindex so `R_1,…,R_m` is a maximum disjoint subfamily (left to right). Let
`A = {x ∈ ℝ^n : x_i = mid(R_i) for i ≤ m, x_i ∈ R_i for i > m}`, with `Vol(A) = ∏_{i>m}|R_i|`; a
distinct-coordinate `x ∈ A` realizes a linear extension `L` of `P`. The `m+1` gaps between consecutive
spine-midpoints each have length `≥ 1`; for fixed `L`, the `n − m` free coordinates occupy `≤ n − m`
gaps, leaving `≥ 2m + 1 − n` gaps unoccupied, so the occupied region `G_L` has `|G_L| ≤ 2(n − m)` and
`Vol(A_L) ≤ |G_L|^{n−m}/(n−m)! ≤ (2(n−m))^{n−m}/(n−m)! ≤ (2e)^{n−m}`. Summing,
`Σ_{i>m} ln|R_i| = ln Vol(A) ≤ ln(e(P)·(2e)^{n−m}) = ln e(P) + (1+ln2)(n−m)`. Disjointness gives
`Σ_{i≤m} ln|R_i| ≤ Σ(|R_i|−1) = n − m` (via `ln x ≤ x − 1`). Adding and converting base:
`Σ_i log|R_i| ≤ log e(P) + (1 + 2/ln2)(n − m)`. Finally `n − m ≤ log e(P)`: the spine is a longest
chain of `P` (length `m`), so stratifying the `n` intervals by chain height gives `m` nonempty strata with
`Σ(L_i − 1) = n − m`, and ordering freely within strata yields `e(P) ≥ ∏ L_i! ≥ 2^{n−m}`. Hence the sum is
`O(log e(P))`. ∎

**Theorem 2 (comparison optimality via bottlenecks).** Let the **level** `ℓ(v)` be the minimum number
of vertices on a path `s → v`, and a **bottleneck** a vertex alone on its level. If `G` has `b`
bottlenecks then `log D ≥ (n − b)/2` (levels without a bottleneck have `≥ 2` vertices ⇒ at most `(n+b)/2`
levels; a BFS tree gives `D ≥ ∏|V_i|! ≥ 2^{n − #levels} ≥ 2^{(n−b)/2}`). Hence plain Dijkstra is already
comparison-optimal unless almost all vertices are bottlenecks. **Dijkstra with lookahead** — which finds
bottlenecks by BFS (linear, no comparisons), keeps them out of the heap, propagates their distances by
additions (an unmarked bottleneck `v` has a unique next-level vertex `w` with `d*(w) = d*(v) + c(vw)`),
and splices them into the output by exponential/binary search — runs in `O(m + log D)` time and
`O(F − n + 1 + log D)` comparisons. It is universally optimal in both time and comparisons. (A
**recursive Dijkstra** that recurses at each bottleneck and maintains the output in a homogeneous finger
search tree achieves the same bounds.)

*Proof sketch.* Non-bottleneck inserts number `n − b = O(log D)`; decrease-keys `≤ F − n + 1`. Delete-min
cost is `O(log D)` by Theorem 1 applied to a fictitious run that inserts-then-deletes each bottleneck
(working sets only grow; each bottleneck has `W = 1`). The exponential searches cost
`Σ_v O(1 + log|B(v)|) = O(log D)`, since the disjoint bottleneck groups `B(v)` give
`D ≥ ∏_v(|B(v)| + 1)`. ∎

**Theorem 3 (the heap exists).** There is a heap with the working-set bound — `O(1)` amortized insert,
decrease-key, find-min, and delete-min returning `x` in `O(log W(x))` amortized — supporting
decrease-key (no meld required).

*Construction.* An **outer heap** is a list of **inner heaps** `H_1, H_2, …`, each a fast heap (`O(1)`
all ops but delete-min, `O(log size)` delete-min, supports meld; Fibonacci / hollow / rank-pairing),
with invariant `i < j ⇒` every item in `H_i` was inserted after every item in `H_j`. Insert creates a
one-item `H_0`, melds the smallest pair `H_j, H_{j+1}` with `|H_j| + |H_{j+1}| ≤ 2^{2^{j+1}}` (else
reindexes all up by 1), and reindexes `H_0,…,H_{j−1}` up by 1.

*Why it works.* Invariants: `|H_i| ≤ 2^{2^i}`; if `H_i` changes in an insert then beforehand
`|H_{i−1}| > 2^{2^{i−1}} − 2^{2^{i−2}}`, so an item in `H_i` (i>1) has `W > 2^{2^{i−2}} − 2^{2^{i−3}} ≥
2^{2^{i−3}}`. Delete-min of `x ∈ H_i` costs `O(log|H_i|) = O(2^i)`, and `log W(x) ≥ 2^{i−3} = 2^i/8`, so
it is `O(log W(x))`. Insert is `O(1)` amortized: charge `1` to each of `H_0,…,H_{j−1}`, split among
items; an item in `H_i` is charged `≤ 1/2^{2^{i−2}}` and bumped one index, total
`Σ_{i≥0} 1/2^{2^{i−2}} = O(1)`. There are `≤ 1 + log log n` inner heaps. Routing in `O(1)` amortized:
**union-find with link-by-index** (root = higher index) for decrease-key (`x` gains `≤ j` ancestors,
charged to its delete-min); a **one-word suffix-minimum bit vector** with `Next/Prev` by mask/shift for
find-min/delete-min (the instance is `O(log log n)` bits). For the general case where items are never
deleted, Gabow–Tarjan fixed-tree union-find replaces link-by-index; in Dijkstra `n` is known and every
item is deleted, so neither rebuilding nor that fallback is needed. ∎

## Algorithm

```text
# ---------- Universally time-optimal: Dijkstra + working-set heap ----------
procedure DijkstraDistanceOrder(G, s):
    for v in V: d[v] ← +inf; state[v] ← UNLABELED
    d[s] ← 0; state[s] ← LABELED; H ← MakeHeap(); Insert(s, H); L ← []
    while FindMin(H) ≠ NULL:
        v ← DeleteMin(H)                       # O(log W(v))
        state[v] ← SCANNED; append v to L
        for arc vw out of v:
            if state[w] = UNLABELED:
                d[w] ← d[v] + c(vw); state[w] ← LABELED; p[w] ← v; Insert(w, H)
            elif state[w] = LABELED and d[v] + c(vw) < d[w]:
                d[w] ← d[v] + c(vw); p[w] ← v; DecreaseKey(w, d[w], H)
    return L                                   # O(m + log D) time, O(F + log D) comparisons

# ---------- Universally comparison-optimal: Dijkstra with lookahead ----------
procedure DijkstraWithLookahead(G, s):
    compute ℓ(·) by BFS; bottlenecks ← {v alone on its level}     # O(m), 0 comparisons
    mark bottleneck v iff level ℓ(v)+1 has ≥ 2 vertices
    init d, state, p; L ← []; H ← MakeHeap()
    B ← bottlenecks in level order up to and incl. first marked
    repeat until B and H empty:
        if H nonempty and (B empty or minLevel(B).d > FindMin(H).d):          # Case 1
            v ← DeleteMin(H); Scan(v); append v to L                          #   O(log W(v))
        else:                                                                # Case 2
            scan B in level order: d[next] ← d[cur] + c(cur,next)            #   additions only
            if B drains below FindMin(H):                                    #   Subcase 2a
                move all of B to L (level order); refill B to next marked
            else:                                                            #   Subcase 2b
                v ← FindMin(H)
                x ← largest-d bottleneck in B with d ≤ d[v],
                      via exponential/binary search from p[v]                #   O(1+log j)
                move bottlenecks of B up to x into L (level order)
    return L                          # O(m + log D) time, O(F − n + 1 + log D) comparisons

# ---------- The working-set outer heap ----------
# inner heaps H_1, H_2, ... : Fibonacci-quality, support meld; i<j ⇒ H_i items newer than H_j
procedure Insert(x, H):
    H_0 ← inner heap {x}
    if ∃ j: |H_j| + |H_{j+1}| ≤ 2^(2^(j+1)):
        j ← min such; H_{j+1} ← Meld(H_j, H_{j+1}); reindex H_0..H_{j-1} up by 1
    else: reindex every H_i up by 1
    record x in union-find; repair suffix-minimum bits of changed heaps      # O(1) amortized
procedure DecreaseKey(x, k, H):
    H_i ← Find(x); DecreaseKey(x, k, H_i)                                     # link-by-index, O(1) am.
    if x is now min of H_i: repair suffix-minimum bits
procedure FindMin(H):  j ← Next(1) on bitvector b; return FindMin(H_j)        # one-word op, O(1)
procedure DeleteMin(H):
    j ← Next(1); x ← DeleteMin(H_j)                                           # O(log|H_j|)=O(log W(x))
    repair suffix-minimum bits for indices ≤ j                                # charged to log W(x)
    return x
```
