Let me start from what actually hurts. I have discrete observational data from some unknown DAG `G*`, and I want its CPDAG back. The classical tools I'd reach for — PC, GES — come with a clean theorem: under faithfulness they return a graph in `MEC(G*)`. And in practice they disappoint, badly, and they disappoint in a *patterned* way: their accuracy is fine on sparse graphs and falls off a cliff as the average degree climbs. That pattern is the clue. Why would density specifically break a search that's provably correct? Because the correctness leans on faithfulness — the assumption that every independence in the distribution is one the graph structure forces — and faithfulness is fragile exactly where edges crowd together. A near-deterministic relation, or two paths that nearly cancel, produces a partial correlation so small that any finite-sample CI test reads it as an independence, and the geometry says the *volume* of distributions sitting near such a violation grows with density. So PC deletes a true edge because a test said "independent," GES's local score does the analogous thing, and on dense graphs that's happening all over the graph at once. The theorem is true; the assumption it rests on is the problem. I don't want a faster faithful algorithm. I want correctness under something weaker than faithfulness, *and* I want it to scale to dense graphs with dozens or hundreds of variables. Those two wants usually trade off, and the whole game is to get both.

So what's the weakest assumption anyone has correctness under? There's a beautiful answer from Raskutti and Uhler. Forget CI tests; rank graphs by edge count. Among all Markovian DAGs — all `G` with `I(G) ⊆ I(P)`, every separation in the graph a true independence — call the one with the fewest edges the sparsest. If `G*` is the *unique* sparsest Markovian DAG (up to its equivalence class), then outputting that sparsest graph gives you `MEC(G*)`. This condition — they call it the sparsest Markov representation, u-frugality — is strictly weaker than faithfulness: there are unfaithful distributions where it still holds, because a near-cancellation that hides an edge just makes the *true* graph look sparser, and sparsity is exactly what you're optimizing for. So the razor is better. How do they search? Over **orderings**. Fix a permutation `π` of the vertices; for each vertex `k`, let its parents be the smallest subset of its predecessors in `π` that makes `k` independent of the rest of the predecessors — its Markov boundary relative to the prefix. The induced `G_π` is automatically acyclic (every edge points forward in `π`), Markovian, and minimal — no subgraph of it is Markovian. Then they enumerate all `π`, build each `G_π`, and keep the sparsest.

That last sentence is where it dies. All `π` means `m!` permutations. Nine variables and you're done. The assumption is the best available and the algorithm to exploit it is intractable — a stark statistical-computational tradeoff. I want the assumption without the `m!`.

The obvious fix is to stop enumerating and start *searching* the ordering space greedily. Teyssier and Koller did exactly this years earlier, and the efficiency argument is worth pinning down because it's the reason ordering-search is the right substrate. The ordering space has `2^{O(m log m)}` elements against `2^{O(m^2)}` structures — vastly smaller. The branching factor of a local move is `O(m)`, not `O(m^2)`. You never check acyclicity, because an ordering can't produce a cycle. And the score decomposes over vertices, `score(G_π) = Σ_i score(X_i, Pa(i))`, so if your local move is an adjacent transposition — swap the vertices at positions `i` and `i+1` — then *only those two vertices* change which predecessors they see; every other family's parent set and score are untouched. Cache the per-family scores and a move costs you re-scoring two families. So hill-climb the orderings with adjacent transpositions, restarts, a tabu list. It's fast, it's simple, it's hard to beat empirically. But — and this is the wall — it has *no consistency guarantee whatsoever*. It's a heuristic. The adjacent-transposition neighborhood is purely local; there's nothing tying its moves to the equivalence-class structure that a correctness proof would need, so it can and does stall at an ordering whose `G_π` isn't even P-minimal. I have a tractable search with no theorem, and an intractable search with the best theorem. I need to import the theorem into the tractable search.

What would it take to give the greedy ordering search a correctness proof? I need my moves to be able to reach `MEC(G*)` from anywhere. There's machinery for navigating *equivalence classes* of DAGs: Chickering's theory of covered edges. An edge `i → j` is covered when `i` and `j` share all their other parents, `Pa(i) = Pa(j) \ {i}`. Reversing a covered edge keeps you in the same MEC — and conversely, two Markov-equivalent DAGs differing in `k` orientations are linked by exactly `k` covered-edge reversals. More: if `I(H) ⊆ I(G)`, there's a sequence of covered-edge reversals and edge additions taking `H` to `G` while `I(·) ⊆ I(G)` holds at every step — a Chickering sequence. This is the engine GES rides. So the natural plan, the one Solus, Wang and Uhler pursued, is to graft Chickering's covered-edge navigation onto the sparsest-permutation search.

Their construction is geometric and it's illuminating. The orderings are the vertices of a polytope, the permutohedron `A_v`, whose edges are adjacent transpositions. Many orderings induce the *same* `G_π`, so contract every transposition that corresponds to a genuine CI relation of `P`; what's left is the DAG associahedron `A_v(P)` — a polytope whose vertices are the distinct induced DAGs (the SGS-minimal ones) and whose edges connect DAGs differing by a single covered-arrow reversal. Now greedily walk this polytope toward sparser DAGs. There are two ways to walk. One walks the *edges* of `A_v(P)` directly via DAG-changing moves — the ESP variant, the edge-sparsest-permutation idea. The other walks weakly-decreasing covered-arrow-reversal sequences in DAG space, Chickering-style — the TSP variant, the triangle one they actually run. And both come with identifiability assumptions, the "edge" and "triangle" assumptions, that sit neatly *between* faithfulness and u-frugality: weaker than faithfulness, stronger than sparsest-Markov-representation. That's progress — a relaxation of faithfulness that's also operational. But staring at it, I see real friction.

First, the ESP/edge variant requires *building the polytope* `A_v(P)`. They flag this themselves: it's inefficient, because to take a step you only need the *neighbors* of your current vertex, not the whole polytope — and they never give an operational, implementable version of ESP at all. It's a theorem with no algorithm. Second, the TSP/triangle variant — the one that runs — does its real work in DAG space. A single move is: reverse a covered arrow, take a linear extension of the resulting DAG, recompute the minimal I-map. So you're constantly translating between the permutation you're searching over and the DAG you're scoring, an awkward back-and-forth between two representations. Third, there's an unsettled correctness claim — they assert the triangle variant can be correct even under some unfaithfulness via a small example, and that example turns out to rest on a distribution that isn't even a semigraphoid, which every distribution is, so the *necessity* of faithfulness for these variants is left murky. And fourth, empirically the triangle variant doesn't scale its accuracy to moderate or large graphs.

Two representations. Permutation space for the search, DAG space for the move. Every time I want to take a covered-arrow reversal I drop into the DAG, reverse, lift back to a permutation. That seam is the source of the awkwardness *and* the inefficiency. What if the move could happen entirely in permutation space? What if there were a single operation on `π` that *is* a covered-edge reversal of `G_π`, with no detour through the DAG?

Let me try to construct it. Take `π` and a covered edge `j → k` in `G_π`. Write `π = ⟨δ1, j, δ2, k, δ3⟩`, where `δ1` is everything before `j`, `δ2` everything strictly between `j` and `k`, `δ3` everything after `k`. I want a new permutation whose induced DAG is `G_π` with `j → k` reversed to `k → j`. The most naive idea: just swap `j` and `k` in place — but they're not adjacent, there's `δ2` between them, so an in-place swap isn't even an adjacent transposition and I can't reason about it cleanly. Let me first slide `k` left until it sits right after `j`. Can I? Sliding `k` past a vertex `i ∈ δ2` is an adjacent transposition of `i` and `k`, and it leaves `G_π` unchanged exactly when `X_i ⊥ X_k` given `i`'s predecessors — i.e. when `i` is *not* a parent of `k`. Here's where "covered" earns its keep. If `j → k` is covered, then `Pa(j) = Pa(k) \ {j}`, so `k`'s parents are `j` together with `j`'s parents, and `j`'s parents all live in `δ1` (they precede `j`). That means *no vertex in `δ2` is a parent of `k`*. So I can slide `k` leftward past every vertex of `δ2`, one adjacent transposition at a time, each one DAG-preserving, until `π` becomes `⟨δ1, j, k, δ2, δ3⟩` with `j` and `k` now adjacent and `G` unchanged. And if an in-between vertex were an ancestor of `k`, the last vertex before `k` on that directed path would be a parent of `k`; coveredness would make that vertex a parent of `j`, so it would have to occur before `j`, not inside `δ2`. Now `j → k` is an honest adjacent pair, and swapping them — `⟨δ1, k, j, δ2, δ3⟩` — is a single adjacent transposition that, because `X_j ⊥̸ X_k` given their shared context, *does* change the DAG: it flips `j → k` to `k → j`. And since reversing a covered edge stays in the MEC, the new ordering's DAG is `G_π` with exactly that one arrow reversed. So my operation, in the covered case, is: **move `k` to just before `j`**, i.e. `⟨δ1, k, j, δ2, δ3⟩`. One permutation operation, no DAG detour. I'll call it a tuck.

Let me write down what a tuck has to do in general, because I want it defined for *any* edge `j → k`, not just covered ones — I'll need the generality later. The subtlety is `δ2`. When the edge is covered, `δ2` has no ancestors of `k` that matter and I can shove `k` all the way to just after `j`. In general `δ2` contains some vertices that are ancestors of `k` and some that aren't. The ancestors of `k` *have* to stay before `k` (a causal order can't put an ancestor after its descendant), and they should come along with `k`; the non-ancestors can stay where they are. So: split `δ2` into `γ` = the vertices of `δ2` that are ancestors of `k` in `G_π`, in order, and `γ^c` = the rest. Then

```
tuck(π, j, k) = ⟨δ1, γ, k, j, γ^c, δ3⟩     if j → k ∈ E(G_π),     else π.
```

The ancestors `γ` slide forward with `k` to sit before `j`; `k` and `j` swap; the
non-ancestors `γ^c` stay after. In the covered case `γ` is empty: if a `δ2` vertex were an
ancestor of `k`, the last vertex before `k` on that path would be a parent of `k`, hence
also a parent of `j`, and therefore forced before `j`. So the covered tuck collapses exactly
to `⟨δ1, k, j, δ2, δ3⟩` — my move above.
Good, the general definition is consistent with the covered case.

Now I have to prove this tuck actually does what I claimed. Let me take the covered case carefully, because correctness lives here. Let `τ = tuck(π, j, k)` and let `H` be `G_π` with `j → k` reversed. First, `τ` is a causal order of `H`: I showed `π' = ⟨δ1, j, k, δ2, δ3⟩` induces the same DAG as `π` (the slide was DAG-preserving), `τ` is the adjacent swap of `j, k` from `π'`, and `π'` is a causal order of `G_π`, so swapping the one covered pair makes `τ` a causal order of `H`. Second, `E(G_τ) ⊆ E(H)`: there's a general fact that if `π` is a causal order of a Markovian `H`, then the minimal-I-map `G_π` is a subgraph of `H` — because each vertex's Markov boundary among its predecessors is contained in its parent set in `H`. Apply it: `τ` is a causal order of `H`, `H` is Markovian (it's in the MEC of the Markovian `G_π`), so `E(G_τ) ⊆ E(H)`. Third, `|E(G_τ)| ≤ |E(G_π)|`: since reversal preserves edge count, `|E(H)| = |E(G_π)|`, and `E(G_τ) ⊆ E(H)` gives `|E(G_τ)| ≤ |E(G_π)|`. Fourth, `I(G_π) ⊆ I(G_τ)`: from `E(G_τ) ⊆ E(H)` and the fact that fewer edges means *more* independences, `I(H) ⊆ I(G_τ)`, and `I(G_π) = I(H)` because they're Markov-equivalent. So a covered tuck never *increases* the edge count and never *loses* an independence — it either reverses an arrow within the MEC (same edge count) or drops to a strictly sparser DAG (deletes one or more edges). That second possibility is a bonus the Chickering DAG-walk doesn't get for free: a single tuck can reverse a covered edge *and* delete an edge in the same move, where Chickering would need a reversal step and then a separate deletion step. The permutation operation is doing two DAG operations at once.

Let me make that equivalence to TSP precise, because if covered tucks exactly reproduce what TSP does, I've got TSP for free in pure permutation space — call that tier GRaSP₀. Define a ct-sequence (covered-tuck sequence) as a chain of permutations where each is the covered tuck of the previous and no induced DAG repeats. I claim: every Chickering sequence TSP would consider, from `G_π` down to some SGS-minimal `H`, corresponds to a ct-sequence ending at the same `H`. Going from a ct-sequence to a Chickering sequence: if a covered tuck keeps the edge count, it's a single covered-edge reversal (the equal-count case of my lemma forces `G_τ = H`); if it drops the count, it's a reversal followed by edge deletions, which is exactly a Chickering sequence with a "turning point" where reversals give way to deletions. Going the other way, from TSP's Chickering sequence to a ct-sequence: each covered reversal in the sequence is realized by a covered tuck of the corresponding causal order, by the lemma. So the two are interchangeable, and the DAG returned by TSP from a given start equals the DAG induced by GRaSP₀'s output. And because a single tuck can fuse a reversal with deletions, the ct-sequence is generally *shorter* than the Chickering sequence — so GRaSP₀ is not just equivalent but more efficient. The two-representation seam is gone: I never leave permutation space.

Now correctness of GRaSP₀. I need: if `G_π` is not P-minimal, a ct-sequence reaches a P-minimal DAG. `G_π` not P-minimal means there's a Markovian `H` with `I(G_π) ⊂ I(H)` — strictly more independences. Chickering's theorem gives a sequence of covered-edge reversals plus an edge deletion from `G_π` toward `H` keeping `I(·) ⊆`, and by the equivalence I just established that sequence is a ct-sequence, and the strict gain in independences means I land somewhere strictly more independent, hence eventually at a P-minimal DAG. So unbounded GRaSP₀ always lands on a P-minimal `G_τ`. And under faithfulness the P-minimal DAGs *are* exactly `MEC(G*)` (a known equivalence: faithful = Pearl-minimal in that setting). So GRaSP₀ is correct and pointwise consistent under faithfulness. Same theorem as TSP, now in permutation space.

Here's a thing I notice while proving necessity, and it surprises me. I'd assumed faithfulness was sufficient but maybe not necessary for TSP — the GSP folks suggested TSP might be correct under some unfaithfulness. Let me check by chasing the logic. Define a DAG as uniquely-P-minimal if it's P-minimal and every P-minimal DAG is in its MEC. Claim: the faithful DAGs are *exactly* the uniquely-P-minimal DAGs. One direction: a faithful `G` is P-minimal, and any other P-minimal `G'` must have `I(G') = I(G)` (neither can strictly contain the other's independences without violating minimality), so `G'` is in the MEC — uniquely P-minimal. The other direction, by contraposition: suppose `G` is unfaithful but P-minimal. Then there's a CI relation `ψ` in `P` not encoded by `G`. I can construct a DAG `G^0` whose *only* independence is `ψ` (take a complete graph oriented by index, delete the one adjacency the independence concerns — every other pair stays d-connected). Extend `G^0` to a P-minimal `G^1` with `I(G^0) ⊆ I(G^1) ⊆ I(P)`. Since `ψ ∈ I(G^1)` but `ψ ∉ I(G)`, `G^1` is *not* in `G`'s MEC. So now I have two P-minimal DAGs in different MECs — `G` is not uniquely P-minimal. Contraposition done: faithful = uniquely-P-minimal. And that *forces* faithfulness to be necessary for TSP/GRaSP₀: under detectable unfaithfulness there's a P-minimal `G` not in `MEC(G*)`, Chickering moves can only travel within `G`'s own MEC from there (they preserve `I(·) ⊆`, and `G` being P-minimal means you can't get strictly more independent), so starting from a causal order of `G` the search returns `G`, which is wrong. The earlier suggestion that TSP escapes unfaithfulness was mistaken — and the mistaken example used a distribution that isn't a semigraphoid, which is illegitimate since every distribution is one. So GRaSP₀, exactly like TSP, is correct iff faithfulness holds. Which means GRaSP₀ alone hasn't bought me anything over GES on the assumption front. I'm back at the wall: I've made the search clean and efficient, but its razor is still faithfulness. I need to *weaken the razor*, and the way to do that has to be in how I pick which edges to tuck.

So far I only tuck *covered* edges. Why covered? Because covered reversal stays in the MEC, so a covered tuck either moves within the equivalence class or sparsifies — it's the conservative move. The associahedron picture says there's a richer set of moves: ESP walks *edges of the polytope*, which are DAG-changing moves, not just within-MEC reversals. What edge of `G_π` corresponds to an associahedron edge? Let me reason about a DAG-changing walk — a sequence of adjacent transpositions, all DAG-preserving except the last, which flips some `j → k` to `k → j`. For that last AT to be performable, `j` and `k` have to have become adjacent in the permutation, with all of `k`'s relevant context already arranged. When can I always do this? I need to be able to slide all of `δ2` out of the way — the ancestors of `k` to before `j`, the non-ancestors to after `k` — by DAG-preserving ATs. That works precisely when there's no vertex `l` in `δ2` that is *both* a descendant of `j` and an ancestor of `k`, because such an `l` would be trapped: it can't go before `j` (it's a descendant of `j`) and can't go after `k` (it's an ancestor of `k`). "No `l` is simultaneously a descendant of `j` and an ancestor of `k`" is exactly the condition that there's **no unidirectional directed path from `j` to `k` other than the edge `j → k` itself**. Call such an edge **singular**. So the associahedron edges — the DAG-changing walks, the ESP moves — correspond exactly to tucks of *singular* edges. Every covered edge is singular (a covered edge can't be on a longer `j ⤳ k` path, or it wouldn't be covered), so covered ⊆ singular. If I let myself tuck singular edges, not just covered ones, I'm doing ESP — and doing it operationally, in permutation space, with one tuck per associahedron edge, *without ever building the polytope*. That's the operational ESP that Solus et al. never wrote down. Call it GRaSP₁. And ESP's razor — the "edge assumption" — is strictly weaker than TSP's. So enlarging the set of tuckable edges from covered to singular weakens the assumption. That's the lever.

Now I can see the whole ladder. The set of edges I'm willing to tuck is the knob. Covered edges → GRaSP₀ ≡ TSP. Singular edges → GRaSP₁ ≡ ESP. And the obvious next rung: **all edges**. Don't restrict at all — tuck any `j → k ∈ E(G_π)`. Call it GRaSP₂. Define the edge-type filter

```
E^t(G) = covered edges      if t = 0
       = singular edges     if t = 1
       = all edges          if t = 2,           with E^0(G) ⊆ E^1(G) ⊆ E^2(G).
```

Each tier `t` runs the same DFS but tucks edges in `E^t`. Why would tucking a *non-singular* edge ever help? Because a non-singular edge tuck makes a move neither a within-MEC reversal nor a clean associahedron edge — it's a permutation rearrangement that can jump to a DAG the smaller move-sets can't reach, and on an unfaithful distribution that jump can find a strictly sparser permutation where ESP and TSP stall. The monotonicity of the razors is what I should check: since each higher tier never returns a denser permutation than the lower one (I run the lower tier first and only accept strict improvements), GRaSP₂ is correct under a strictly *weaker* causal razor than GRaSP₁, which is strictly weaker than GRaSP₀. I can even exhibit it: a four-variable unfaithful model where a path cancellation hides one independence, on which GRaSP₁ gets stuck at a five-edge DAG from some initial permutations while GRaSP₂ recovers the four-edge truth from every start. The relaxation is real, not cosmetic. I should be honest about the ceiling, though: GRaSP₂ is *not* correct under u-frugality alone — there's a five-variable counterexample where even tucking any edge fails to reach the sparsest permutation from some start. So the ladder lands strictly between ESP's razor and the sparsest-Markov-representation ideal. That's still the best operational razor in this family, and it's the dense-graph regime where it pays off.

Let me settle the DFS itself, because "tuck edges in `E^t`" needs an actual search. Greedy single tucks aren't enough — a sparsifying tuck might only become available *after* a score-neutral one, so I need to descend through equal-score tucks looking for a strict improvement. The structure: at the current permutation, scan candidate edges; tuck one; if the score strictly improves, accept and restart; if the score is unchanged (a within-MEC move), recurse one level deeper to look for an improvement reachable from there; if it's worse, undo. Bound the recursion by a depth `d`. Unbounded (`d = m!`) is the version with the correctness theorem. But in finite samples, unbounded DFS is exponential and the deep tucks rarely pay, so I cap `d` small — depth three is the sweet spot in practice: deep enough to follow short equal-score chains to a sparsification, shallow enough to stay fast. And I run the tiers in order — GRaSP₀ then GRaSP₁ then GRaSP₂ — so each tier starts from the previous tier's already-improved permutation and the statistics improve monotonically.

One refinement on the DFS that the implementation wants. At the *root* of a DFS descent, for the top tier I actually want to run, I should be willing to tuck any parent edge — that's where the relaxation lives. But once I'm *inside* a recursive descent (chasing equal-score chains), restrict to *covered* tucks. The reason: the deep descent is meant to traverse *within* an equivalence class to find a hidden sparsification, and within-MEC traversal is exactly covered-edge reversal; letting deep recursion tuck arbitrary edges would let it wander off into unrelated DAGs and explode the search. So: any parent edge at the top, covered-only as I go deeper. And to avoid looping forever between Markov-equivalent DAGs, I track the set of edges I've flipped along an equal-score path and refuse to revisit a flip-set I've already seen.

Now the sample version, because everything above is the oracle story with "score = negative edge count" and a CI oracle. With finite data I have neither. Replace negative-edge-count with a decomposable, locally consistent score, and replace the CI-test-based parent selection with score-based parent selection. BIC works in general: `BIC_D(G) = Σ_i BIC_D(X_i, Pa(i))`, decomposable, and locally consistent — in the large-sample limit, adding `j → k` raises `BIC` iff `X_j ⊥̸ X_k | Pa(k)`. So "is `j` a parent of `k`" becomes a score comparison, no significance threshold. For the discrete benchmarks I want the discrete analogue, BDeu: a marginal likelihood under a Dirichlet prior with an equivalent-sample-size pseudo-count spread uniformly over cells, which is what makes it score-equivalent across an MEC. Its local score for vertex `i` with parents `Pa_i`, in terms of the cell counts `N_ijk` (records with `X_i` in state `k` under parent-configuration `j`) and `N_ij = Σ_k N_ijk`, with `r_i` the number of states of `X_i`, `q_i = Π_{p∈Pa_i} r_p` the number of parent configurations, and `α` the sample-prior:

```
BDeu_D(X_i, Pa_i) = Σ_j [ lgamma(α/q_i) − lgamma(N_ij + α/q_i)
                          + Σ_k ( lgamma(N_ijk + α/(r_i q_i)) − lgamma(α/(r_i q_i)) ) ]
                    + |Pa_i|·log(s/vm) + (vm − |Pa_i|)·log(1 − s/vm),
```

where the last line is a structure prior (`s` = structure-prior, `vm = m − 1`) that mildly
penalizes parent count. The defaults `α = 1`, `s = 1`, depth `= 3` are the conservative
choices: a single pseudo-observation of prior mass, a neutral structure prior, a shallow DFS.

How do I select each vertex's parents from its prefix? Grow-shrink. Grow: start from the empty parent set, repeatedly add the predecessor that most improves the local score while any addition improves it; this overshoots to a superset of the true Markov boundary. Shrink: from the grown set, repeatedly remove any variable whose removal improves the score, while any does; this trims back to exactly the unique Markov boundary in the large-sample limit (on a compositional graphoid). Grow-then-shrink realizes the minimal-I-map-from-an-ordering construction with nothing but the decomposable score — no hypothesis test anywhere. That's the `markov_boundary` slot from the harness.

And here's where the Teyssier-Koller caching insight comes back, generalized. When I tuck, I rearrange a contiguous block of the permutation, and *only the vertices in that block* see a changed prefix; every vertex outside the block keeps its parents and score. So I should recompute parents/scores only for the affected positions. But I can do better than recompute-from-scratch: the grow and shrink traces for a given vertex depend only on *which* predecessors are available, so I can cache them in a tree keyed by the prefix — a grow-shrink *tree* (GST), one per vertex. Each node of the tree caches its grow branches (sorted so the best-scoring growth is tried first) and its shrink removals; tracing for a particular prefix walks the tree, following only the branches whose vertex is actually in the prefix, growing and shrinking lazily and reusing every node it has built before. Across the thousands of re-scorings a DFS performs, the GST turns the cost of a vertex's parent search from "rerun grow-shrink" into "walk a cached trie," which is the discrete-search analogue of "only two families recompute on an adjacent swap."

Let me also nail the tuck *implementation* in array terms, since I defined it abstractly. To tuck the vertex at position `i` (that's `k`) toward position `j` (that's the parent `j`'s position), I need to move `k` together with its ancestors that currently sit between `j` and `i` to just after position `j`. So: collect the ancestors of `k` (transitive closure of its parents); walk positions from `j+1` up to `i`; whenever the vertex at the current position is an ancestor of `k`, pop it out and insert it at the front of the block (right after `j`), maintaining a running shift. After the moves, the ancestors `γ` and `k` itself have slid forward to sit just past `j` and `j` has dropped back — which is exactly `⟨δ1, γ, k, j, γ^c, δ3⟩`. Then `update` re-derives, for every position from `j` to `i`, that vertex's parents and local score from its new prefix via the cached GST.

So let me assemble it — filling the order-space-search slot of the harness with the top-tier, depth-bounded covered/any tuck DFS over a random initial permutation, scored by BDeu via cached grow-shrink trees, ending in a DAG-to-CPDAG conversion. I need to be careful with score signs in code: the BDeu helper computes the local expression with the structure-prior term, the `Order` constructor negates the empty-parent call it makes during setup, and then the actual family scores used by initialization and every tuck update come from `GST.trace`, which calls the no-cache score and compares larger returned values.

```python
import random, sys, time, warnings
from typing import Any, Dict, List, Optional
import numpy as np
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.score.LocalScoreFunction import local_score_BDeu, local_score_BIC_from_cov
from causallearn.search.PermutationBased.gst import GST          # cached grow-shrink tree
from causallearn.score.LocalScoreFunctionClass import LocalScoreClass
from causallearn.utils.DAG2CPDAG import dag2cpdag


class Order:
    """A working permutation plus each vertex's selected parents, its local score,
    and the running edge count. Initialized to a random order with empty parents."""
    def __init__(self, p, score):
        self.order = list(range(p))
        self.parents = {}
        self.local_scores = {}
        self.edges = 0
        random.shuffle(self.order)                       # random initial permutation
        for i in range(p):
            y = self.order[i]
            self.parents[y] = []
            self.local_scores[y] = -score.score(y, [])   # causal-learn negates this setup score

    def get(self, i): return self.order[i]
    def set(self, i, y): self.order[i] = y
    def index(self, y): return self.order.index(y)
    def insert(self, i, y): self.order.insert(i, y)
    def pop(self, i=-1): return self.order.pop(i)
    def get_parents(self, y): return self.parents[y]
    def set_parents(self, y, ps): self.parents[y] = ps
    def get_local_score(self, y): return self.local_scores[y]
    def set_local_score(self, y, s): self.local_scores[y] = s
    def get_edges(self): return self.edges
    def set_edges(self, e): self.edges = e
    def bump_edges(self, b): self.edges += b
    def len(self): return len(self.order)


def grasp(X, score_func="local_score_BIC_from_cov", depth=3, parameters=None, verbose=True,
          node_names=None):
    X = X.copy()
    n, p = X.shape
    if n < p:
        warnings.warn("The number of features is much larger than the sample size!")

    # For discrete data, choose BDeu; this branch uses local_score_BDeu defaults:
    # sample_prior=1, structure_prior=1, and r_i_map inferred from X.
    if score_func == "local_score_BDeu":
        score = LocalScoreClass(data=X, local_score_fun=local_score_BDeu, parameters=None)
    elif score_func == "local_score_BIC_from_cov":
        score = LocalScoreClass(data=X, local_score_fun=local_score_BIC_from_cov,
                                parameters=parameters or {"lambda_value": 2})
    else:
        raise Exception("Unknown function!")

    gsts = [GST(i, score) for i in range(p)]              # one cached grow-shrink tree / vertex
    node_names = node_names or [("X%d" % (i + 1)) for i in range(p)]
    nodes = [GraphNode(name) for name in node_names]
    G = GeneralGraph(nodes)

    order = Order(p, score)
    # score each vertex against the vertices preceding it in the initial order (grow-shrink)
    for i in range(p):
        y = order.get(i)
        y_parents = order.get_parents(y)
        candidates = [order.get(j) for j in range(0, i)]         # the prefix = predecessors
        order.set_local_score(y, gsts[y].trace(candidates, y_parents))
        order.bump_edges(len(y_parents))

    # iterate the DFS until no tuck improves the total score
    while dfs(depth - 1, set(), [], order, gsts):
        if verbose:
            sys.stdout.write("\rGRaSP edge count: %i    " % order.get_edges()); sys.stdout.flush()

    # read the DAG off the final order, convert to its CPDAG
    for y in range(p):
        for x in order.get_parents(y):
            G.add_directed_edge(nodes[x], nodes[y])
    return dag2cpdag(G)


def dfs(depth, flipped, history, order, gsts):
    """One DFS pass of tucks. `flipped` tracks the covered-reversal flip-set along an
    equal-score path; `history` blocks revisiting a flip-set (no infinite within-MEC loop)."""
    cache = [{}, {}, {}, 0]
    indices = list(range(order.len())); random.shuffle(indices)
    for i in indices:
        y = order.get(i)
        y_parents = order.get_parents(y); random.shuffle(y_parents)
        for x in y_parents:
            covered = set([x] + order.get_parents(x)) == set(y_parents)   # is x->y covered?
            # at the root (history empty) tuck ANY parent edge (the tier-2 relaxation);
            # deeper in the descent restrict to covered tucks (within-MEC traversal only)
            if len(history) > 0 and not covered:
                continue
            j = order.index(x)
            for k in range(j, i + 1):                       # snapshot the affected block
                z = order.get(k)
                cache[0][k] = z; cache[1][k] = order.get_parents(z)[:]
                cache[2][k] = order.get_local_score(z)
            cache[3] = order.get_edges()

            tuck(i, j, order)                              # the permutation move
            edge_bump, score_bump = update(i, j, order, gsts)   # re-derive affected families

            if score_bump > 1e-6:                          # strict improvement: accept, restart
                order.bump_edges(edge_bump); return True

            if score_bump > -1e-6:                         # score-neutral (within-MEC) move
                flipped = flipped ^ set(
                    [tuple(sorted([x, z])) for z in order.get_parents(x)
                     if order.index(z) < i])
                if len(flipped) > 0 and flipped not in history:
                    history.append(flipped)
                    if depth > 0 and dfs(depth - 1, flipped, history, order, gsts):
                        return True
                    del history[-1]

            for k in range(j, i + 1):                       # undo: restore the block
                z = cache[0][k]; order.set(k, z)
                order.set_parents(z, cache[1][k]); order.set_local_score(z, cache[2][k])
            order.set_edges(cache[3])
    return False


def update(i, j, order, gsts):
    """Recompute parents and local scores for positions j..i after a tuck (only the
    affected block changed prefix); returns (edge change, score change)."""
    edge_bump = old_score = new_score = 0
    for k in range(j, i + 1):
        z = order.get(k); z_parents = order.get_parents(z)
        edge_bump -= len(z_parents); old_score += order.get_local_score(z)
        z_parents.clear()
        candidates = [order.get(l) for l in range(0, k)]   # z's new prefix
        s = gsts[z].trace(candidates, z_parents)           # cached grow-shrink
        order.set_local_score(z, s)
        edge_bump += len(z_parents); new_score += s
    return edge_bump, new_score - old_score


def tuck(i, j, order):
    """Move the vertex at position i (=k) and its ancestors lying in (j, i] to just after
    position j: pi = <d1, gamma, k, j, gamma^c, d3>."""
    ancestors = []
    get_ancestors(order.get(i), ancestors, order)
    shift = 0
    for k in range(j + 1, i + 1):
        if order.get(k) in ancestors:                      # an ancestor of k in the block
            order.insert(j + shift, order.pop(k)); shift += 1


def get_ancestors(y, ancestors, order):
    ancestors.append(y)
    for x in order.get_parents(y):
        if x not in ancestors:
            get_ancestors(x, ancestors, order)
```

Let me trace the causal chain that got me here. The classical faithful searches break on dense graphs precisely because faithfulness fails there, so I needed correctness under a weaker razor; the sparsest-permutation razor is that weaker razor but its exhaustive `m!` search is intractable. Greedy ordering search is tractable but has no theorem. To import a theorem I needed Chickering's covered-edge navigation, and the existing graft of it onto permutation search lived awkwardly across two representations and never gave an operational polytope walk. The fix was a single permutation operation, the tuck, which in the covered case *is* a covered-edge reversal — collapsing to "move `k` before `j`" because no in-between vertex can be an ancestor of `k` — so it reproduces TSP entirely inside permutation space (GRaSP₀), even fusing a reversal and a deletion into one move. That gave the same theorem but also the same razor, faithfulness, which a clean necessity argument (faithful = uniquely-P-minimal) showed is unavoidable for this tier. The way past it was to enlarge *which edges* I tuck: singular edges turn the tuck into an operational associahedron edge-walk (GRaSP₁ ≡ ESP, weaker razor, no polytope construction), and tucking *any* edge (GRaSP₂) relaxes the razor strictly further — verifiably recovering truths the lower tiers stall on, though still short of u-frugality alone. The finite-sample code follows the top-tier version directly: descend through score-neutral tucks bounded by a small depth (three in practice), tuck any parent edge at the root and only covered edges deeper, and use a flip-set history to prevent within-MEC loops. The finite-sample instantiation replaces negative-edge-count with the decomposable, locally consistent BDeu score and the CI oracle with grow-shrink parent selection, cached per vertex in a grow-shrink tree so that each tuck — which only perturbs a contiguous block of the permutation — re-derives just the affected families against the trie. The final ordering's DAG, read off the parents, is converted to its CPDAG.
