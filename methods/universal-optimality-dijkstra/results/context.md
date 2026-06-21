# Context

## Research question

Dijkstra's algorithm with a Fibonacci heap solves single-source shortest paths on a graph with
non-negative arc lengths in `O(m + n log n)` time, and there are graphs on which this bound cannot
be beaten by *any* correct algorithm. So in the worst case Dijkstra is optimal, and one might think
the story is over. It is not. A worst-case bound is a statement about the single hardest family of
inputs; it says nothing about whether the algorithm is wasteful on the inputs one actually has. Take
a graph that is a long path of `r` vertices hanging off a star of `t` leaves at the source. Sorting
the `t` leaves genuinely needs `Θ(t log t)` comparisons, but the `r` path vertices are forced into a
unique order by the path itself and ought to cost nothing — total `O(r + t log t)`. Dijkstra with a
Fibonacci heap instead pays `Θ(log t)` for *each* of the `r` path deletions, because all `t` leaves
sit in the heap the whole time, giving `Ω(r log t)`. For `r = t²` that is asymptotically worse than
the best possible on this very graph. The algorithm is worst-case optimal yet provably suboptimal on
this easy instance.

The question this raises: can a single, fixed, comparison-based shortest-path algorithm be made *as
fast as the best possible algorithm on every individual graph at once* — not just on the worst graph?
"As fast as possible on a given graph" has to be made precise (a tailor-made algorithm that hard-codes
one instance's answer is trivially fast and meaningless), and one needs a cost model in which such a
claim can even be stated and a matching lower bound proved. The target is a guarantee of the form:
for every graph topology `G`, the algorithm's cost under a worst-case choice of arc weights is within
a constant factor of what the best correct algorithm achieves on `G` under *its* worst-case weights.
Achieving this for a classical sequential algorithm — rather than just in the worst case — is the goal.

## Background

**The shortest-path / vertex-ordering setting.** `G` is a directed graph with `n` vertices, `m` arcs,
a source `s` from which all vertices are reachable, and a non-negative length `c(vw)` on each arc.
`d*(v)` denotes the true (shortest-path) distance from `s`. Dijkstra's greedy method maintains a
*current distance* `d(v)`, keeps vertices in three states (unlabeled / labeled / scanned), and
repeatedly scans the minimum-current-distance labeled vertex, relaxing its outgoing arcs. Because
arc lengths are non-negative, a vertex's current distance equals its true distance when it is scanned,
and vertices are scanned in non-decreasing order of true distance. Three distinct outputs can be asked
of the algorithm: the true distances, a shortest-path tree, and an ordering of the vertices by
distance. The last — call it the **distance order problem** — is the focus here. A *distance order* is
a total order `L` of the vertices for which *some* non-negative weighting makes the true distances
distinct and increasing along `L`. There is a clean combinatorial characterization: `L` is a distance
order iff every vertex `w ≠ s` has an incoming *forward arc* (an arc `vw` with `v` before `w` in `L`).
Write `D` for the number of distinct distance orders of `(G, s)` and `F` for the maximum number of
forward arcs of any distance order (for an undirected graph `F = m`). `D` is a purely topological
quantity, and `log D` is the natural information-content of "which order are we in."

**Heaps and the cost of Dijkstra.** An efficient Dijkstra stores the labeled vertices in a heap keyed
by current distance: `n` inserts, at most `n` delete-mins, at most `m − n + 1` decrease-keys, plus
`O(m)` work outside the heap. So the running time is `O(m)` plus the heap cost, and the heap is the
whole ball game. The relevant heaps, in historical order:
- **Binary heap** (Williams 1964): every operation `O(log n)`; Dijkstra runs in `O(m log n)`.
- **Fibonacci heap** (Fredman & Tarjan 1987), invented expressly to speed up Dijkstra: insert,
  decrease-key, meld in `O(1)` amortized; delete-min in `O(log n)` amortized. Dijkstra runs in
  `O(m + n log n)`, worst-case optimal for the distance order problem. Hollow heaps and rank-pairing
  heaps achieve the same bounds; all of these support melding cheaply.
- **Pairing heap** (Fredman, Sedgewick, Sleator & Tarjan 1986), a self-adjusting heap with no node
  augmentation. Fredman (1999) showed pairing heaps *cannot* achieve `O(1)` decrease-key (a
  `Ω(log log n)` lower bound for a natural model); Iacono (2000) showed pairing heaps support `O(1)`
  insert/meld and `O(log n)` extract-min, and that *if decrease-key is not a supported operation*, the
  pairing heap admits a fine-grained bound in which the cost of deleting an item depends on how recently
  it was inserted rather than on the heap's size. Later work (Elmasry; Elmasry, Farzan & Iacono)
  obtained stronger bounds of this fine-grained kind, again only without decrease-key.

**Beyond-worst-case analysis.** The dissatisfaction with worst-case bounds has a literature. The
strongest refinement, **instance optimality** (Fagin, Lotem & Naor 2001; Afshani, Barbay & Chan 2017
for geometric problems), asks an algorithm to be within `O(1)` of the best correct algorithm on *every
single input*. It is the gold standard and correspondingly rare; for the distance order problem it is
unachievable, because on a fixed weighted instance an algorithm that simply outputs a correct order
and verifies it runs in linear time, so no instance-optimal bound can be both nontrivial and matched.
A weaker, more attainable notion was developed in **distributed** computing under the name *universal
optimality* (Haeupler, Wajc & Zuzic 2021, and a line of work on distributed minimum spanning tree,
minimum cut, and approximate shortest paths): parameterize by the graph topology but take the worst
case over weights, so the benchmark is "best possible on this topology." Whether such a guarantee can
be obtained for a *sequential* algorithm in the standard model was open. The self-adjusting-data-
structure tradition supplies the relevant intuition: splay trees (Sleator & Tarjan 1985) have a
*working-set* bound — accessing a recently accessed element is cheap — and the splay-tree dynamic
optimality conjecture is exactly an instance-optimality question; the analogous question for a broad
class of heaps (tournament heaps) was answered negatively (Munro, Peng, Wild & Zhang 2019), so one
cannot expect a fully self-adjusting heap to be the vehicle.

**The motivating phenomenon.** The path-off-a-star example above
illustrates the contrast between Dijkstra's worst-case bound and per-graph performance. A Fibonacci
heap charges each delete-min by the *current heap size*, while on this graph the deleted path vertices
were inserted and removed in immediate succession — there is strong *locality* in the sequence of heap
operations that the size-based bound does not reflect.

**A neighbouring fact about distances vs. ordering.** The three Dijkstra outputs are not equally hard.
Computing only the distances (not a sorted order) is doable faster than Dijkstra in
the comparison-addition model (Duan, Mao, Mao, Shu & Yin 2025, `O(m log^{2/3} n)`), so the
*distance order* problem is strictly harder than the distances-only problem.

**A counting tool for orderings of intervals.** A recurring combinatorial object is a family of integer
intervals `{[a_i, b_i] ⊆ [1, k]}` and the partial order in which `[a_i, b_i] ≺ [a_j, b_j]` when
`b_i < a_j`. Recent work on sorting from a directed acyclic graph (van der Hoog, Rotenberg &
Rutschmann 2025), itself building on Cardinal, Fiorini, Joret, Jungers & Munro 2013 and on Kahn & Kim
1992 on entropy and sorting, provides a sharp bound relating the *widths* of the intervals to the
*number of linear extensions* of this partial order: the sum of `log(width)` over the intervals is
`O(log #extensions)`. This is the kind of statement that converts a per-element "recency" cost into a
global "how many orderings are possible" quantity.

## Baselines

**Dijkstra + binary heap (Williams 1964).** Insert/decrease-key/delete-min all `O(log n)`; total
`O(m log n)`. Every per-operation cost is tied to the *whole* current heap size.

**Dijkstra + Fibonacci heap (Fredman & Tarjan 1987).** `O(1)` amortized insert and decrease-key,
`O(log n)` amortized delete-min, total `O(m + n log n)`. This is worst-case optimal for distance
ordering: there exist graphs (a single star, where the leaves must be fully sorted) forcing
`Ω(n log n)` comparisons for any correct algorithm. The delete-min bound is `log(current heap size)`.

**Pairing heap and its working-set-style bounds (Fredman–Sedgewick–Sleator–Tarjan 1986; Iacono 2000;
Elmasry; Elmasry–Farzan–Iacono).** These heaps have fine-grained bounds where the deletion cost
of an item depends on how recently it entered rather than on the heap size. These results assume
decrease-key is *not* a supported operation; Dijkstra performs up to `m − n + 1` decrease-keys.

**Instance-optimal algorithms (Fagin–Lotem–Naor 2001; Afshani–Barbay–Chan 2017).** The benchmark is
the best correct algorithm on each individual input. Where attainable (top-`k` aggregation; planar
convex hull under order-oblivious instance optimality) this is the strongest guarantee. For distance
ordering, a fixed weighted instance is solved in linear time by an algorithm that knows the answer and
only verifies it, so there is no nontrivial instance-optimal benchmark to match.

**Universally-optimal distributed algorithms (Haeupler–Wajc–Zuzic 2021 and follow-ups).** In the
distributed (e.g. CONGEST) model, "best possible on this topology, worst case over weights" has been
achieved for several problems. These results live in a communication-complexity model with its own
notion of cost and techniques (e.g. low-congestion shortcuts).

## Evaluation settings

The yardsticks are two cost models, both pre-existing and standard, in which a lower bound can be proved
and an upper bound matched:

- **Comparison model.** The algorithm knows the graph's vertices and arcs and has oracle access to the
  arc lengths; it pays `1` for each comparison of two linear functions of arc lengths and may do no
  other arithmetic on lengths. Cost = number of comparisons. The natural lower-bound machinery here is
  the information-theoretic / decision-tree argument for sorting.
- **Time model.** The graph is given by incidence lists of outgoing arcs (each storing its length); the
  algorithm must *traverse* the lists to discover arcs (`1` unit to access the first arc of a list or
  the arc after a given one), and may compare linear functions only of already-accessed arcs (`1` unit
  each). This model is deliberately generous to the algorithm (it is free to compute the linear
  functions it compares), which only strengthens lower bounds. Cost = total units. The comparison model
  is strictly stronger than the time model: on a path with extra back-arcs, distance ordering costs `0`
  comparisons but `Ω(m)` time.

In both models the benchmark is *per-topology worst-case over weights* (and, for the time model, over
incidence-list permutations): an algorithm is universally optimal in time (resp. comparisons) if on
every graph its worst-case time (resp. comparison count) is within `O(1)` of the best any correct
algorithm achieves on that graph for its own worst case. The relevant problem-instance families used to
exhibit gaps are the single star (forces `Ω(n log n)` comparisons), the simple path and the
path-with-back-arcs (force `0` comparisons but `Ω(m)` time), and the path-off-a-star (separates
Fibonacci-Dijkstra from the per-graph optimum).

## Code framework

The landing artifact for this problem is a theorem with its proof and the algorithm in pseudocode, not
production code. The scaffold below gives the greedy Dijkstra harness driven by an abstract heap
interface, and the shape of an optimality argument, leaving empty the pieces a solution would have to
supply.

```text
# ---- abstract heap interface (the operations Dijkstra needs) ----
interface Heap:
    MakeHeap() -> H
    Insert(x, H)             # x has a predefined key
    DecreaseKey(x, k, H)     # k < current key of x
    FindMin(H) -> item or NULL
    DeleteMin(H) -> item     # remove and return a min-key item

# ---- the existing efficient-Dijkstra harness, parameterized by a heap ----
def Dijkstra(G, s, Heap):
    for v in V(G):
        d[v] = +inf;  state[v] = UNLABELED
    d[s] = 0;  state[s] = LABELED
    H = Heap.MakeHeap();  Heap.Insert(s, H)
    order = []                       # vertices in scanned (distance) order
    while FindMin(H) != NULL:
        v = Heap.DeleteMin(H)        # next vertex to scan
        state[v] = SCANNED;  order.append(v)
        for arc vw in outgoing(v):   # relax
            if state[w] == UNLABELED:
                d[w] = d[v] + c(vw);  state[w] = LABELED;  Heap.Insert(w, H)
            elif state[w] == LABELED and d[v] + c(vw) < d[w]:
                d[w] = d[v] + c(vw);  Heap.DecreaseKey(w, d[w], H)
    return order

# ---- the heap the analysis will require ----
class TODOHeap(Heap):
    # All standard heaps (binary, Fibonacci, ...) implement this interface, but
    # charge each DeleteMin by the *current size* of the heap.
    # TODO: the heap we will design.
    pass

# ---- the optimality argument, stated against a per-topology lower bound ----
# LowerBound(G) := best worst-case-over-weights cost achievable on topology G
#                  by any correct distance-order algorithm.
def claim_universally_optimal(Dijkstra_with_TODOHeap):
    # Want: for every graph G, worst-case cost of the algorithm on G
    #       is O( LowerBound(G) ).
    # This needs (a) a usable closed form for LowerBound(G), and
    #            (b) a bridge from the heap's per-operation costs on the
    #                operation sequence Dijkstra generates to that closed form.
    # TODO: the lower bound, and the bridge from heap cost to it.
    pass
```
