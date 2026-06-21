Let me start from the thing that bothers me. Dijkstra's algorithm with a Fibonacci heap runs in
`O(m + n log n)`, and I can prove that's worst-case optimal for producing the vertices sorted by
distance: a single star — a source `s` joined to `n − 1` leaves — forces any correct algorithm to sort
the leaf weights, which is `Ω(n log n)` comparisons. So on paper the question is closed. But "closed
in the worst case" is a statement about one bad family of graphs. It tells me nothing about whether the
algorithm is wasteful on the graph in front of me. And I have a nagging example where it clearly is.

Take a star of `t` leaves `u_1,…,u_t` hanging off `s`, and also a long path `s, v_1, v_2, …, v_r`
coming out of `s`. Suppose the path is "short": the distance to the far end `v_r` is less than every
single edge `s u_i`. Run Dijkstra. It deletes `s`, then walks down the path deleting `v_1, v_2, …, v_r`
in order, and only then starts pulling out the `u_i`. But here's the thing: every leaf `u_i` was
inserted at the very first step, when `s` was scanned. So while Dijkstra is deleting the path vertices,
all `t` leaves are sitting in the heap the entire time. With a Fibonacci heap each of those `r`
deletions costs `Θ(log(heap size)) = Θ(log t)` — total `Ω(r log t)`.

Now, is that necessary? The leaves genuinely have to be sorted among themselves; that's `Θ(t log t)`,
unavoidable. But the path vertices? Their order is forced — `v_1` before `v_2` before … by the path
itself, there's literally nothing to decide. I can solve the whole graph by sorting the `t` leaf
distances (`O(t log t)`), summing along the path to get each `v_j`'s distance (`O(r)`, no comparisons),
and merging the two sorted lists (`O(r + t)`). Total `O(r + t log t)`. If `r = t²`, that's `O(t²)`
against Dijkstra's `Ω(t² log t)`. Dijkstra, on this graph, is a `log t` factor off the best possible —
and the best possible is achieved by an algorithm I can write down. Dijkstra is worst-case optimal and
yet, on this perfectly nice graph, demonstrably suboptimal. That's the itch.

So what do I actually want? Not instance optimality. I know that's hopeless here: fix the weights, and
the "algorithm" that just prints the correct order and verifies it in one linear pass beats everything;
there's no nontrivial per-*instance* benchmark to match. The trick the path-off-a-star is pointing at
is that the difficulty lives in the *topology*, not in the particular numbers. The leaves are hard
because the star topology permits `t!` orderings; the path is easy because its topology permits exactly
one. So the right benchmark parameterizes by the graph `G` and takes the worst case over the weights:
an algorithm `A` is what I'll call *universally optimal* if there's a constant `c` such that for every
graph `G` and every correct algorithm `A'`,

    max over weights w  Time_A(G, w)   ≤   c · max over weights w  Time_{A'}(G, w).

"Best possible on this exact graph." This notion exists in distributed computing — people have
universally optimal distributed algorithms for spanning trees, cuts, approximate shortest paths — but I
don't know of it being achieved for a *sequential* algorithm in the standard model. That's the prize:
can a single fixed Dijkstra-like algorithm be optimal on *every* graph at once.

Before I can chase a matching upper bound I need to know what I'm matching. So: what is the per-graph
lower bound for ordering vertices by distance?

Let me pin down the problem cleanly first. I'll call it the *distance order problem*: output a total
order `L` of the vertices that is non-decreasing by true distance for the given weights. Notice three
different things Dijkstra produces — the distances, a shortest-path tree, and this order — and the
order is the hardest of the three up to constant factors: given the order I can recover distances and a
tree in `O(m)` extra work, but not obviously vice versa. (And in fact computing distances *alone* turns
out to be doable faster than Dijkstra — there's an `O(m log^{2/3} n)` algorithm for distances-only — so
the distance *order* is genuinely the harder, more Dijkstra-shaped problem. Good: that means it's the
right problem to ask Dijkstra-optimality about.)

When is an order `L` even a valid distance order? Claim: `L` is a distance order iff every vertex
`w ≠ s` has an incoming *forward* arc, meaning an arc `vw` with `v` before `w` in `L`. One direction:
if `L` comes from some weighting with distinct increasing distances, the last arc `vw` on a shortest
path to `w` has `d*(w) = d*(v) + c(vw) > d*(v)`, so `v` precedes `w` — there's the forward arc. The
other direction is constructive and worth doing because it'll seed the lower bounds: given `L =
[v_1,…,v_n]` where each `v_j ≠ s` has some incoming arc `v_i v_j` with `i < j`, set that arc's length to
`j − i` and every other arc's length to `n`. Then the true distance of `v_i` is exactly `i − 1`, the
distances are distinct and increasing along `L`, and `L` is a distance order. (For an undirected graph
the same works, giving the reverse arc the same length.) Clean. Let `D` be the number of distance
orders of `(G,s)` and `F` the maximum number of forward arcs of any distance order. These are purely
topological numbers. `log D` smells like the right "information content" of the answer.

Now the lower bounds. Two of them, one for time and one for comparisons, and I'll need a careful cost
model to even state them. Let me fix the model: in the *comparison model* the algorithm has the graph
combinatorially and oracle access to weights, paying `1` per comparison of two linear functions of arc
lengths, no other weight arithmetic. In the *time model* the graph arrives as incidence lists the
algorithm must traverse to discover arcs, `1` unit per list-step, same comparison rule. The time model
is deliberately too generous to the algorithm (it gets to compute the linear functions for free) — that
only makes the lower bounds stronger, which is what I want.

Lower bound on time: any correct algorithm needs `Ω(m)`. Adversary argument. Fix any distance order
`L = [v_1,…,v_n]` and the weighting `c(v_i v_j) = max(0, j − i)`, so `d*(v_i) = i − 1` and `L` is the
*unique* true order. Now suppose the algorithm fails to access some incidence-list entry, say it never
walks past arc `v_i v_j` with `j > i + 1`. Then I sneak in: lower that arc's length to `1/2` (or add
such an arc if it isn't there). The algorithm, having never looked, behaves identically and still
outputs `L`. But now there's a path `s → v_i → v_j` of length `(i − 1) + 1/2 < i`, while every path to
`v_{i+1}` has length `≥ i`, so `v_j` must precede `v_{i+1}` in any correct order — `L` is wrong. So a
correct algorithm must access essentially all of the incidence structure: at least `max(n − 2,
m − 2n + 2)` entries, which with `n > 2` is `Ω(m)`. (Time only, of course — in the comparison model you
*know* the arcs, so this `Ω(m)` doesn't apply there; that's the gap between the two models, and it's
real: a path with extra back-arcs needs `0` comparisons but `Ω(m)` time.)

Lower bound on comparisons, first piece: `Ω(log D)`. This is the sorting information-theory argument.
Model the algorithm as a decision tree where each node compares a linear function of weights to `0`
(three outcomes: `<`, `=`, `>`). For each of the `D` distance orders, take a weighting making it the
unique true order (the `j − i` construction). Whenever a comparison comes out "`= 0`," perturb a weight
with nonzero coefficient by an arbitrarily tiny amount to push it off zero, small enough not to disturb
any other outcome or the distinctness of distances — always possible since weights are arbitrary
non-negative reals. After perturbation, the `D` weightings follow `D` distinct root-to-leaf paths, no
"`=`" outcomes, so the tree has `≥ D` leaves and a leaf at depth `≥ ⌈log D⌉`. Hence `≥ ⌈log D⌉`
comparisons in the worst case. (Average depth `≥ ⌊log D⌋`, and by Yao's principle this survives for
randomized algorithms too.) So `log D` is a genuine comparison lower bound — and now I see why I wanted
it: it's small exactly when the topology forces the order, which is the path's situation.

Lower bound on comparisons, second piece: `Ω(F − n + 1)`. This one I almost missed, but the time bound
hinted at it — there's a cost just to *certify* you haven't missed a forward arc. Take a distance order
with `F` forward arcs; give forward arc `v_i v_j` length `j − i`, and all non-forward arcs a common
huge length, so the algorithm's behavior depends only on the forward-arc lengths. Simplify the decision
tree to functions of forward-arc variables only. Suppose the path taken makes `≤ F − n` comparisons. I
turn each comparison inequality `a·c > 0` into the equality `a·c = (its current value)`; that's `≤ F − n`
linear equations in the forward-arc lengths. Add the `n − 1` equations fixing the lengths of the path
edges `v_i v_{i+1}` to their current values — total `≤ F − 1` equations in `F` variables. Fewer
equations than variables, so the solution set is a positive-dimensional affine space containing the
current weighting; slide along a line in it, decreasing some variable, until a forward-arc length hits
`0`. I can nudge so that exactly one such arc `v_i v_j` with `j > i + 1` becomes length `0` while all
comparison outcomes are unchanged. But then `v_j` ought to sit immediately after (or before) `v_i`, so
`L` is no longer the unique correct order — and the algorithm made the same comparisons, so it's wrong.
Hence `≥ F − n + 1` comparisons. Combine the two: comparisons `≥ Ω(F − n + 1 + log D)`, time
`≥ Ω(m + log D)`. (For undirected graphs `F = m`, so the comparison bound reads `m − n + 1 + log D`.)

There's my target, drawn: time `Ω(m + log D)`, comparisons `Ω(F − n + 1 + log D)`. Now, can Dijkstra
hit `O(m + log D)`?

Dijkstra is `O(m)` plus heap cost. The inserts and decrease-keys: there are `n` and at most `F − n + 1`
of them respectively, and with a Fibonacci-quality heap each is `O(1)` amortized, so `O(F) = O(m)`
total. Everything funnels into the delete-mins. With a Fibonacci heap a delete-min costs `O(log n_i)`
where `n_i` is the heap size at that moment — and that's exactly what overcharges the path-off-a-star.
So the whole question becomes: is there a heap whose delete-min cost, summed over a Dijkstra run, is
`O(n + log D)` instead of `O(n log n)`?

What should the delete-min cost depend on, if not heap size? Stare at the path-off-a-star again. The
path vertices were each inserted and then almost immediately deleted — they had tiny *residence*. The
leaves sat around forever. The Fibonacci bound, `log(heap size)`, can't tell these apart; both are
charged by the population at deletion time. What I want is a bound that's cheap precisely when an item
is deleted soon after it was inserted — a bound sensitive to *temporal locality* in the operation
sequence. This is the same spirit as the working-set bound for splay trees, where a recently-touched
element is cheap to access.

Let me make it concrete. For an item `x`, define its **working set** as the set of items inserted into
the heap between the time `x` is inserted and the time `x` is deleted (including `x`), and its
**working-set size** `W(x)` as the count. I'll ask for a heap with the **working-set bound**: every
operation except delete-min is `O(1)` amortized, and a delete-min returning `x` costs `O(log W(x))`. To
feel why this is the right knob, take the operation sequence `Insert(999), Insert(999), …, Insert(999)`
(`k − 1` long-lived items) followed by `Insert(1), Del, Insert(1), Del, …` (the quick ones). A regular
heap charges each `Del` `Θ(log(heap size)) = Θ(log k)`. A working-set heap charges each `Del`
`O(log(items inserted since this item entered)) = O(log 1) = O(1)`. On the path-off-a-star, each path
vertex `v_j` is deleted with only `O(1)` insertions having happened since it entered — its working set
is just itself — so each costs `O(1)`, and the total drops to `O(r + t log t)`, exactly the optimum I
computed by hand. The working-set bound is the heap property that fixes my motivating example.

But fixing one example isn't a theorem. I need: in *any* Dijkstra run on *any* graph,
`Σ_v log W(v) = O(log D)`. If that holds, delete-mins cost `O(n + log D)`, Dijkstra is `O(m + log D)`,
universally time-optimal. So I have to relate the working-set sizes — a property of the *operation
sequence Dijkstra happens to generate* — to `log D`, a property of the *graph topology*. Why on earth
would those be connected? Because Dijkstra's operation sequence isn't arbitrary; it has structure
inherited from the graph.

Here's the structure. Number the vertices `v_1, v_2, …, v_n` in the order Dijkstra *inserts* them. For
each `v_i`, let the interval `[a_i, b_i]` be `a_i = i` and `b_i =` the insertion-index of the last
vertex inserted before `v_i` is deleted. Then `W(v_i) = b_i − a_i + 1` exactly: the working set is the
window of insertion-indices during `v_i`'s residence. Now consider the *search tree* `T` of the run:
for each `v_i ≠ s`, the arc `u_i v_i` whose relaxation first labeled `v_i`. These arcs form a spanning
tree rooted at `s`. The crucial locality fact: if `v_i v_j` is an arc of `T`, then `v_i` was *deleted
before* `v_j` was *inserted*. Why — `v_j` only becomes labeled (and thus inserted) when `u_i = v_i` is
scanned, and a vertex is scanned exactly when it's deleted. So `b_i < a_j`: the residence windows of a
parent and child in `T` are *disjoint and ordered*. The parent's window closes before the child's opens.

That disjointness is the bridge. Build the interval DAG `I`: a vertex per interval `[a_i, b_i]`, and an
arc from `[a_i,b_i]` to `[a_j,b_j]` whenever `b_i < a_j` (window `i` entirely before window `j`). The
locality fact says every `T`-arc `v_i v_j` corresponds to an `I`-arc, so any topological order of `I`,
read through the map `[a_i,b_i] ↦ v_i`, is a topological order of `T` — and distinct topological orders
of `I` give distinct ones of `T`. And every topological order of `T` is a distance order of `G` (by the
forward-arc characterization: each non-root has its `T`-parent as a preceding incoming arc). So

    D  ≥  (number of distance orders of T)  ≥  (number of topological orders of I) =: D(I).

Therefore `log D ≥ log D(I)`. If I can show `Σ_i log W(v_i) = Σ_i log(b_i − a_i + 1) = O(log D(I))`,
I'm done. That is a pure statement about intervals: the sum of the logarithms of the interval *widths*
is `O(log #topological orders of the interval order)`. Let me prove it, because it's the linchpin and I
don't want to wave at it.

Setup for the interval lemma: `n` integer intervals `R_1, …, R_n`, each `⊆ [1,n]` (here `n` plays the
role of `k`; widths `|R_i| = b_i − a_i + 1`), inducing the partial order `P` where `i ≺ j` iff
`b_i < a_j` (disjoint, `i` left of `j`); let `e(P)` be the number of linear extensions of `P` (that's
`D(I)`). Claim: `Σ_i log|R_i| = O(log e(P))`. I'll prove it with a volume argument, which is slicker
than charging.

Reindex so that `R_1, R_2, …, R_m` is a *maximum-size* set of pairwise-disjoint intervals, sorted left
to right (`m` of them). Define the polytope

    A = { x ∈ ℝ^n : x_i = mid(R_i) for i ≤ m,  x_i ∈ R_i for i > m }.

Pin the `m` "spine" coordinates at their interval midpoints; let the other `n − m` roam in their
intervals. Then `Vol(A) = ∏_{i>m} |R_i|`. Any `x ∈ A` with distinct coordinates induces a linear
extension `L` of `P` by the rule "`i ≺ j` iff `x_i < x_j`" — call `x` a *realization* of `L`. Different
realizations can give the same `L`; let `A_L` be the points of `A` realizing `L`.

Claim 1: `Vol(A_L) ≤ (2e)^{n−m}` for each `L`. Look at the `m + 1` "gap" intervals carved out by the
spine midpoints: `I = (−.5, mid R_1), (mid R_1, mid R_2), …, (mid R_m, n + .5)`. Because the spine is a
*maximum* disjoint family with each `|R_i| ≥ 1`, consecutive midpoints are `≥ 1` apart, so each gap has
length `≥ 1`. Now, fixing `L` fixes, for each free coordinate `i > m`, *which gap it lives in* — any two
realizations of the same `L` agree on that, since the relative order of all coordinates is fixed by `L`.
Call a gap *occupied* if some free coordinate sits in it. There are `n − m` free coordinates, so at most
`n − m` occupied gaps, hence at least `(m + 1) − (n − m) = 2m + 1 − n` *unoccupied* gaps. Each unoccupied
gap has length `≥ 1`, and the total span is `n + 1`, so the union `G_L` of occupied gaps has length
`|G_L| ≤ (n + 1) − (2m + 1 − n) = 2(n − m)`. Confine `A_L` inside `{ free coords ∈ G_L, respecting `L`'s
order }`; its volume is `|G_L|^{n−m} / (n − m)!` (ordered points in a region of measure `|G_L|`), so

    Vol(A_L) ≤ (2(n−m))^{n−m} / (n−m)!  ≤  (2e)^{n−m},

using `k! ≥ (k/e)^k`. Good.

Claim 2: sum over the spine-free coordinates. `Σ_{i>m} ln|R_i| = ln Vol(A) ≤ ln Σ_{L ⊇ P} Vol(A_L)`
(`A` is partitioned by which extension each point realizes, and not every extension need be realizable,
so the volume is at most the sum over *all* extensions). By Claim 1 there are `e(P)` extensions each of
volume `≤ (2e)^{n−m}`, so

    Σ_{i>m} ln|R_i|  ≤  ln( e(P) · (2e)^{n−m} )  =  ln e(P) + (1 + ln 2)(n − m).

Claim 3: the spine coordinates. `R_1,…,R_m` are disjoint in `[1,n]`, so `Σ_{i≤m} |R_i| ≤ n`, and by
`ln x ≤ x − 1`, `Σ_{i≤m} ln|R_i| ≤ Σ_{i≤m}(|R_i| − 1) = n − m`.

Add Claims 2 and 3 and convert to base-2 logs:

    Σ_{i=1}^n log|R_i|  ≤  log e(P)  +  (1 + 2/ln 2)(n − m).

Last step: `n − m ≤ log e(P)`. A chain in `P` is a sequence of pairwise `≺`-related intervals — i.e.
pairwise disjoint and left-to-right — so the spine `R_1,…,R_m`, being a *maximum* disjoint family, is a
*longest chain* in `P`, of length `m`. Stratify all `n` intervals by chain height: let `L_i` (for
`i = 1,…,m`) be the number of intervals whose longest chain down to a `≺`-minimal interval has exactly `i`
intervals. Every height from `1` to `m` is hit (the spine alone realizes each), and the heights partition
the `n` intervals, so `Σ_i L_i = n` over `m` strata, giving `Σ_i (L_i − 1) = n − m`. Two intervals of equal
height are incomparable in `P` (neither can lie on the other's chain), so ordering arbitrarily *within*
each stratum and stacking the strata in height order yields a distinct linear extension of `P` every time:
`e(P) ≥ ∏_{i=1}^m L_i! ≥ ∏_i 2^{L_i − 1} = 2^{n − m}`, i.e. `n − m ≤ log e(P)`.
Therefore `Σ log|R_i| = O(log e(P))`. The lemma holds. ∎

Substituting back — `|R_i| = W(v_i)`, `e(P) = D(I) ≤ D` — gives `Σ_v log W(v) = O(log D(I)) = O(log D)`.
So delete-mins cost `O(n + log D)`, and Dijkstra with a working-set heap runs in `O(m + log D)` time:
**universally optimal in time**. And in comparisons it does `O(F + log D)`, since the delete-min
comparisons are also `O(n + log D)` and the rest is `O(F)`. That matches the comparison lower bound
`Ω(F − n + 1 + log D)` — except for an additive `O(n)`. Hmm.

That `O(n)` gap is annoying, and it's not always negligible: it's `Θ(n)` whenever `log D ≪ n`. The
path-off-a-star is exactly such a case (`log D` is roughly `t log t`, possibly `≪ n = r + t + 1`). So
plain Dijkstra, even with a perfect working-set heap, is universally time-optimal but not yet
comparison-optimal. Where does the extra `O(n)` come from? From doing one comparison per vertex just to
get it into and out of the heap — `n` heap inserts and `n` delete-mins, each touching a comparison even
when the order is forced. To be comparison-optimal I must *avoid spending comparisons on vertices whose
position is already determined*.

When is `log D` actually small compared to `n`? Let me find the obstruction. Define the **level**
`ℓ(v)` of a vertex as the minimum number of vertices on a path from `s` to `v`, and call `v` a
**bottleneck** if it is the *only* vertex on its level. Claim: if `G` has `b` bottlenecks then
`log D ≥ (n − b)/2`. Proof: every level without a bottleneck has `≥ 2` vertices, so if there are `ℓ`
levels, `2(ℓ − b) ≤ n − b`, giving `ℓ ≤ (n + b)/2`. Take a BFS tree `T`; its topological orders include
every order that respects levels, so `D ≥ ∏_i |V_i|! ≥ ∏_i 2^{|V_i|−1} = 2^{n−ℓ}`, hence
`log D ≥ n − ℓ ≥ (n − b)/2`. So `log D` is small only when *almost all* vertices are bottlenecks.

That's the whole game. If at most `(1 − ε)n` vertices are bottlenecks then `log D = Ω(n)` and my `O(n)`
slack is absorbed — already comparison-optimal. The only graphs that defeat me are bottleneck-heavy
ones, and on a bottleneck I am wasting a comparison. So I must handle bottlenecks specially: never spend
a comparison ordering them when their order is forced.

What's special about a bottleneck `v`? It's the unique vertex on its level, so it *dominates* every
vertex on higher levels — every path from `s` to anything beyond `v`'s level goes through `v`. Two
consequences. First, an arc into `v` from a *higher* level is useless (it would have to come back through
`v`), so it can be ignored. Second — and this is the lever — call a bottleneck *unmarked* if the
next-higher level also has just one vertex (so it's a bottleneck too) or `v` is the top level, and
*marked* otherwise. If `v` is an unmarked bottleneck and `w` is the unique vertex on the next level, then
since `v` dominates `w` and `w` is exactly one level up, a shortest path to `w` ends with the arc `vw`,
so `d*(w) = d*(v) + c(vw)`. That's the true distance of `w` from one *addition*, no comparison. So along
a run of consecutive unmarked bottlenecks I can propagate true distances by additions alone, free of
comparisons. And there are few marked bottlenecks: each marked one has `≥ 2` vertices on the next level,
both non-bottlenecks, so the number of marked bottlenecks is at most `n − b`.

Now I can make Dijkstra comparison-optimal: don't put bottlenecks in the heap at all. Find all
bottlenecks first — a plain BFS from `s` computes levels in `O(m)` time and *zero* comparisons. Then run
Dijkstra inserting only the non-bottlenecks, and handle the bottlenecks on the side. Keep an array `B`
of pending bottlenecks, in level order, up to and including the next marked one (so each refill is a run
of unmarked bottlenecks ending at a marked one). When the run's first distance is known, the rest follow
by additions (the Lemma above). Then I splice `B` into the output list against the stream of vertices
coming out of the heap.

The splice is where a few comparisons happen, and I want them cheap. When the heap's minimum is a
non-bottleneck `v` with distance `d(v)`, I need to move out all pending bottlenecks with distance `≤
d(v)`. Rather than scan them one by one, I do an *exponential/binary search* in `B` starting from `p(v)`
(the parent of `v`, often already near the right spot): compare `d(v)` to the `1`st, `2`nd, `4`th, `8`th,
… pending bottleneck until I bracket it, then binary-search inside. If the right cutoff is the `j`-th
pending bottleneck, this costs `O(1 + log j)`. The point of exponential search is that it pays for the
*distance moved*, not the size of `B`.

Correctness is the usual Dijkstra invariant with a twist: a vertex's current distance equals its true
distance when scanned; the output list is non-decreasing by true distance; and — the twist — I break
ties so a bottleneck is emitted before a non-bottleneck of equal distance. That tie-break guarantees a
vertex's parent always precedes it in the output, so the output is a topological order of the search
tree, hence a distance order. (When `v`'s parent `p(v)` is a bottleneck and `v` isn't, `v` can't leave
the heap until all bottlenecks up to `p(v)` have been processed and emitted — so `p(v)` lands first.)

Efficiency, term by term. The non-bottleneck inserts number `n − b = O(log D)` (by the bottleneck
lemma). Decrease-keys: at most `F − n + 1`, `O(1)` each, `O(F − n + 1)` total — within budget. The
delete-mins: I reuse the working-set analysis, but I have to account for the bottlenecks that never
entered the heap. Trick: in the analysis only, *pretend* each bottleneck is inserted-then-immediately-
deleted right before it's scanned. These fictitious operations only *grow* the working sets of the
real (non-bottleneck) items, never shrink them, and each fictitious bottleneck has working set `1`. So
`Σ_v log W(v) = O(log D)` still holds, and the real delete-mins cost `O(log D)`. The searches of `B`:
let `B(v)` be the bottlenecks moved out during `v`'s search step (excluding ones below `p(v)`'s level if
`p(v)` moves then). These `B(v)` are disjoint, and inserting `v` anywhere among its `|B(v)| + 1` slots
gives a distinct distance order, so `D ≥ ∏_v (|B(v)| + 1)`, i.e. `Σ_v log(|B(v)| + 1) ≤ log D`. Since
each search costs `O(1 + log|B(v)|)`, total search cost is `O(n − b + log D) = O(log D)` — same interval
flavor of argument as before. Everything sums to `O(m + log D)` time and `O(F − n + 1 + log D)`
comparisons. That hits *both* lower bounds. **Universally optimal in time and comparisons.**

(There's a second route to the same end that I like for its symmetry: instead of pulling bottlenecks out
of the heap, run Dijkstra *recursively*. Run from `s`; the moment you'd scan the next bottleneck `v`,
suspend and recurse from `v` with a fresh heap, `v`'s true distance as its starting key. Compute all
bottleneck distances by descending into the recursion, then finish the runs in decreasing level order;
later runs may decrease-key into earlier heaps, which is exactly the bookkeeping that keeps it correct.
Maintain the output list with a *homogeneous finger search tree*, which inserts an item `k` positions
from a finger in `O(1 + log k)` — and `Σ log(gap) = O(log D)` by the very same interval argument. No
melds are ever needed even though several heaps coexist, so my heap-without-meld will serve. Same
bounds, different mechanism; good to have because it shows the result isn't an artifact of the lookahead
trick.)

So the algorithm is settled, *conditional on one thing I've been assuming the whole time*: a heap with
the working-set bound that *also* supports `O(1)` decrease-key. I have not built it. And I can't just
grab one off the shelf.

Why not? The locality-sensitive heaps that exist — Iacono's pairing-heap analysis, and the stronger
working-set-type bounds after it — all have a fatal caveat for me: they hold only when *decrease-key is
not a supported operation*. Iacono showed the pairing heap, deleting an item, can be charged by how
recently it entered rather than the heap size — exactly my working-set bound — but the moment you allow
decrease-key the analysis breaks, and in fact Fredman proved pairing heaps can't even do `O(1)`
decrease-key. Dijkstra does up to `F − n + 1` decrease-keys and they *must* be `O(1)` amortized. So the
existing working-set heaps are useless to me here. I need a new construction that marries `O(1)`
decrease-key (which Fibonacci-style heaps give) with the working-set delete-min bound (which the
self-adjusting heaps give, but only without decrease-key). Nobody has both at once. That's the real
technical work.

Let me think about what "working-set bound" is asking of the *structure*. Deleting `x` should cost
`O(log W(x))`, where `W(x)` counts insertions during `x`'s residence — i.e. roughly the items that
arrived *after* `x` and are still around. The Fibonacci bound is `log(total population)`. The difference
between "items inserted after `x`" and "all items" is items inserted *before* `x`. So I want the
structure to, somehow, not charge `x` for items older than it. Idea: keep the items *roughly sorted by
insertion time*, freshest first, and segregate them so that deleting a fresh item only touches the fresh
part.

Concretely: maintain a list of separate heaps `H_1, H_2, …, H_k`, each an ordinary fast heap (anything
of Fibonacci quality — `O(1)` everything except delete-min, `O(log size)` delete-min, supports meld;
Fibonacci, hollow, or rank-pairing heaps all qualify — I'll use one black-box and not care which). Call
these *inner heaps* and the whole thing an *outer heap*. Keep the invariant: **every item in `H_i` was
inserted after every item in `H_j` whenever `i < j`** (smaller index = more recent). Then for any item
`x` in `H_i`, *all* of `H_1, …, H_{i−1}` were inserted after `x`, so they're all in `x`'s working set —
which means `H_{i−1}`'s *size* is a *lower bound* on `x`'s working *set* size `W(x)`. That's the lever:
if I delete `x` from `H_i`, the cost is `O(log|H_i|)`, and I want that to be `O(log W(x))`; it suffices
that `|H_{i−1}|` (a lower bound on `W(x)`) be polynomially related to `|H_i|` — say `log|H_{i−1}| ≥
c · log|H_i|`. So I need the inner heaps' sizes to *grow fast* as the index drops.

How fast? If sizes merely doubled (`|H_i| ≈ 2^i`), then `log|H_i| ≈ i` and `log|H_{i−1}| ≈ i − 1` —
that's only an additive gap, the ratio `→ 1`, fine for delete-min but, as I'll see, it kills the insert
amortization. Push harder: let `|H_i|` grow *doubly* exponentially, `|H_i| ≈ 2^{2^i}`. Then
`log|H_i| ≈ 2^i` and `log|H_{i−2}| ≈ 2^{i−2} = 2^i / 4`, so the working-set lower bound for an item in
`H_i` is within a constant factor of the delete-min cost *in the log* — and as a bonus the number of
inner heaps is only `O(log log n)`, which will let me handle the "which heap holds the global minimum"
problem almost for free. Let me commit to doubly-exponential and make the bookkeeping precise.

Sizes are controlled at insertion via melding. To insert `x`: create a new one-item inner heap `H_0`;
then find the smallest `j` such that `H_j` and `H_{j+1}` *together* are small enough — I'll define "small
enough" as `|H_j| + |H_{j+1}| ≤ 2^{2^{j+1}}` — and replace `H_{j+1}` by the meld of `H_j` and the old
`H_{j+1}`; finally reindex `H_0, …, H_{j−1}` up by one (so `H_0` becomes `H_1`, etc.). If no such `j`
exists, just reindex everyone up by one (and `H_0` becomes `H_1`). The other operations are routing:
find-min/delete-min find the inner heap holding a global minimum and delegate; decrease-key finds the
inner heap holding `x` and delegates.

Now verify the three invariants this is designed to keep. (i) *Order*: by induction, melds and
reindexings preserve "smaller index ⇒ more recently inserted" — a meld combines `H_j` into `H_{j+1}`
where everything in `H_j` is newer than everything in `H_{j+1}`, and the new `H_0` is the newest of all.
(ii) *Upper bound* `|H_i| ≤ 2^{2^i}` at all times: induction on operations. A meld into `H_{j+1}` only
happens when the combined size is `≤ 2^{2^{j+1}}`, which is the bound for index `j + 1`; reindexing `H_0`
(size `1 ≤ 2^{2^1}`) to `H_1` is fine; nothing else grows a heap. (iii) *Lower bound when a heap
changes*: if `i > 0` and `H_i` changes during an insertion, then at the start (after creating `H_0`),
`|H_{i−1}| > 2^{2^{i−1}} − 2^{2^{i−2}}`. For `i = 1`: `H_0` has `1` item and `1 > 2^{2^0} − 2^{2^{−1}}`,
true. For `i > 1`: since `H_i` changes, the insertion did *not* meld `H_{i−2}` and `H_{i−1}` (it found a
larger `j`), so `|H_{i−2}| + |H_{i−1}| > 2^{2^{i−1}}`; by the upper bound `|H_{i−2}| ≤ 2^{2^{i−2}}`, so
`|H_{i−1}| > 2^{2^{i−1}} − 2^{2^{i−2}}`. Combined with the order invariant, every item in `H_{i−1}` is in
the working set of every item in `H_i`, giving the corollary: an item in `H_i` (for `i > 1`) has working-
set size `> 2^{2^{i−2}} − 2^{2^{i−3}} ≥ 2^{2^{i−3}}` (carrying the lower bound through one reindexing —
`H_{i−2}` becomes `H_{i−1}` during the insertion that grows `H_i`).

Delete-min cost. An item `x` in `H_i` costs `O(log|H_i|) ≤ O(log 2^{2^i}) = O(2^i)` to delete (the inner
heap is Fibonacci-quality). If `i = 1`, that's `O(1)`, fine. If `i > 1`, by the corollary
`log W(x) ≥ log 2^{2^{i−3}} = 2^{i−3} = 2^i / 8`, so the delete cost `O(2^i) = O(8 · 2^{i−3}) =
O(log W(x))`. There it is: the doubly-exponential growth makes `H_{i−2}`'s size lower-bound (a stand-in
for `W(x)`) match `H_i`'s size upper-bound to within a factor of `8` in the log. Delete-min is within the
working-set bound. (And the `/8` is exactly why I needed doubly- and not singly-exponential: with
`|H_i| = 2^i` the cost is `O(i)` while `log W ≥ i − c`, a constant *additive* gap only — but the insert
amortization needs the gaps between sizes to grow fast, and that's the other half.)

Insert cost — this is the half that forces doubly-exponential. An insertion that changes heaps
`H_0, …, H_{j−1}` (highest changed index `j`) costs `O(j)`; normalize to `j` units, charge `1` to each
of `H_0, …, H_{j−1}`, and split each heap's unit equally among its items. By the lower-bound invariant,
an item in `H_i` is charged at most `1 / (2^{2^{i−1}} − 2^{2^{i−2}}) ≤ 1 / 2^{2^{i−2}}` for this
insertion. Every time an item is charged, the insertion bumps its index. So over its whole life an item
accrues total charge at most `Σ_{i≥0} 1 / 2^{2^{i−2}}`, and *this sum converges* (the terms shrink doubly-
exponentially) — to a constant. Hence each insertion is `O(1)` amortized. Had the sizes only grown
singly-exponentially, the per-charge bound would be `~1/2^{i}` and the sum still converges, but the
*delete-min* side fails; doubly-exponential is the sweet spot where *both* the delete-min log-ratio and
the insert charge-series behave. So: `O(1)` amortized insert, decrease-key, find-min; delete-min within
the working-set bound — provided I can do the two routing tasks (find the inner heap holding a given
item; find the inner heap holding the global minimum) in `O(1)` amortized each.

Routing task 1 — find the inner heap containing item `x`, for decrease-key. This is exactly the
disjoint-set-union (union-find) problem: each inner heap is a set of its items, melds are unites, and
decrease-key is a find. Generically union-find is `O(α(n))` amortized, essentially constant; but I can
get *truly* `O(1)` amortized here by exploiting structure. Use the classic compressed-tree union-find
with one twist — *link by index*: when melding `H_j` into `H_{j+1}`, make `H_{j+1}`'s root the new root.
Because an item's index only ever *increases* over its life, it acquires at most `j` ancestors before it
reaches `H_j`; a find on `x` costs `O(1)` plus `O(1)` per ancestor-change, at most `O(j)` total over all
finds of `x`. And `x` is deleted from some `H_j` where, if `j > 1`, `log W(x) ≥ 2^{j−3}` — so I charge
the `O(j) = O(log W(x))` to `x`'s delete-min, which is within budget. Every find and unite is `O(1)`
amortized. (In the general case where some items are never deleted, link-by-index isn't enough; there I'd
fall back to Gabow–Tarjan fixed-tree union-find, which is `O(1)` amortized when the set of possible
unites — here `Unite(i, i+1)` for consecutive insertion-ids — is known in advance. But Dijkstra deletes
everything it inserts, so the simple link-by-index suffices.)

Routing task 2 — find the inner heap holding the global minimum, for find-min/delete-min. Here the
doubly-exponential payoff returns: there are only `1 + log log n` inner heaps (since the highest index
`j` satisfies, after the insertion creating it, that the outer heap holds `> 2^{2^{j−1}}` items, so
`j ≤ 1 + log log n`). Call `H_i` a *suffix minimum* if it's nonempty and its minimum key beats every
nonempty `H_{i'}` with `i' > i`. Keep a bit vector `b` with `b_i = 1` iff `H_i` is a suffix minimum;
it's only `1 + log log n` bits, so it fits in *one machine word*. Find-min is then "smallest index with
`b_i = 1`", `Next(1)`; delete-min does that, deletes from that inner heap, and re-walks the heaps of
index `≤ j` to repair suffix-minima, costing `O(j)` — charged, as above, to `log W(x)`. Decrease-key may
flip a bounded number of bits, charged to the flips that set them. The `Next/Prev` queries on a single-
word bit vector are a constant number of mask-and-shift operations (or a `log n`-size lookup table) —
`O(1)`. This is a tiny instance of union-split-find; van Emde Boas would give `O(log log)` per query in
general, but my instance is so small (`log log n` bits) that one word handles it in `O(1)`. (Both
routing structures, and the bit vector, want an a-priori bound on `n`; a loose one suffices, and in
Dijkstra `n` is known up front, so no rebuilding is ever needed.)

That closes the loop. I have an outer heap with `O(1)` amortized insert, decrease-key, find-min, and
delete-min within the working-set bound, *with* `O(1)` decrease-key — the combination that didn't exist
before — built black-box on top of any Fibonacci-quality inner heap. Dijkstra driven by it runs in
`O(m + log D)` time on every graph (universally time-optimal), and the bottleneck-aware extensions push
the comparison count down to `O(F − n + 1 + log D)` (universally comparison-optimal), both matching the
topological lower bounds I drew at the start.

Tracing the whole chain: the path-off-a-star shows Fibonacci-Dijkstra overcharges easy graphs, which
demands a per-graph rather than worst-case benchmark — universal optimality. To match it I first prove
the per-graph lower bounds `Ω(m + log D)` time and `Ω(F − n + 1 + log D)` comparisons. The delete-mins
are the only loose term, so I ask for a heap charged by *recency* — the working-set bound — and I prove
its total cost is `O(log D)` because Dijkstra's parent-before-child insertion windows are disjoint, which
through an interval-counting lemma collapses `Σ log W(v)` into `log D`. The residual `O(n)` comparison
slack matters only when almost every vertex is a bottleneck, and bottlenecks have forced order whose
distances propagate by additions, so I keep them out of the heap and splice them in by exponential
search (or recurse), spending no comparisons where the lower bound permits none. Finally I build the
required heap by stacking Fibonacci-quality heaps whose sizes grow doubly-exponentially in recency, which
makes the older neighbor's size a constant-factor-in-the-log proxy for an item's working set (cheap
delete-min) while making the meld-charge series converge (cheap insert), with union-find and a one-word
suffix-minimum bit vector doing the `O(1)` routing.

    procedure Dijkstra-distance-order(G, s):
        # G: graph, s: source; output: vertices in a true distance order.
        # H is the working-set outer heap (Insert/DecreaseKey O(1) amortized,
        #    DeleteMin within the working-set bound O(log W(x))).
        for v in V:  d[v] ← +inf; state[v] ← UNLABELED
        d[s] ← 0;  state[s] ← LABELED;  H ← MakeHeap();  Insert(s, H)
        L ← empty list
        while FindMin(H) ≠ NULL:
            v ← DeleteMin(H);  state[v] ← SCANNED;  append v to L   # cost O(log W(v))
            for arc vw out of v:
                if state[w] = UNLABELED:
                    d[w] ← d[v] + c(vw); state[w] ← LABELED; p[w] ← v; Insert(w, H)
                elif state[w] = LABELED and d[v] + c(vw) < d[w]:
                    d[w] ← d[v] + c(vw); p[w] ← v; DecreaseKey(w, d[w], H)
        return L
        # Time O(m + log D), comparisons O(F + log D): universally time-optimal.

    # ---- comparison-optimal variant: keep bottlenecks out of the heap ----
    procedure Dijkstra-with-lookahead(G, s):
        compute levels by BFS; bottlenecks ← {v : v alone on its level}   # O(m), 0 comparisons
        mark bottleneck v iff level ℓ(v)+1 has ≥ 2 vertices
        initialize d, state, p as above; L ← empty; H ← MakeHeap()
        B ← bottlenecks in level order up to and incl. the first marked one
        repeat until B and H are empty:
            if H nonempty and (B empty or min-level(B).dist > FindMin(H).dist):   # Case 1
                v ← DeleteMin(H); Scan(v); append v to L                          #   O(log W(v))
            else:                                                                # Case 2
                scan B in level order, propagating d[next] ← d[cur] + c(cur,next) # additions only
                if B drains entirely below FindMin(H):                            #   Subcase 2a
                    move all of B to L (level order); refill B to next marked bottleneck
                else:                                                            #   Subcase 2b
                    v ← FindMin(H)
                    x ← largest-distance bottleneck in B with dist ≤ d[v],
                        found by exponential/binary search from p[v]              #   O(1+log j)
                    move bottlenecks of B up to x into L (level order)
        return L
        # Time O(m + log D), comparisons O(F − n + 1 + log D): universally optimal in both.

    # ---- the heap H: an outer heap of doubly-exponentially sized inner heaps ----
    # inner heaps H_1, H_2, ...: any Fibonacci-quality heap (O(1) all ops but DeleteMin,
    #   O(log size) DeleteMin, supports meld); invariant: i<j ⇒ every item in H_i newer than H_j.
    procedure Insert(x, H):
        H_0 ← new inner heap holding x
        if ∃ j with |H_j| + |H_{j+1}| ≤ 2^(2^(j+1)):
            j ← smallest such; H_{j+1} ← Meld(H_j, H_{j+1}); reindex H_0..H_{j-1} up by 1
        else:
            reindex every H_i up by 1
        union-find: record x's set; suffix-minimum bit vector b: repair bits for changed heaps
        # amortized O(1): meld-charge Σ 1/2^(2^(i-2)) converges
    procedure DecreaseKey(x, k, H):
        H_i ← Find(x)                       # union-find with link-by-index, O(1) amortized
        DecreaseKey(x, k, H_i); update suffix-minimum bits if x became H_i's min   # O(1) amortized
    procedure DeleteMin(H):
        j ← Next(1) on b                    # smallest-index suffix minimum, one word op, O(1)
        x ← DeleteMin(H_j)                  # O(log |H_j|) = O(log W(x)) by Cor. (size ratio /8)
        repair suffix-minimum bits for indices ≤ j                                  # charged to log W(x)
        return x
