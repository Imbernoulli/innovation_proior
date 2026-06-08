I keep running into the same wall, and it always wears the same disguise. I have an object I want to build — a colouring, an assignment, a configuration — and I want to prove it *exists* by the only general method I trust for this, which is to make the object at random and show the random one is good with positive probability. I lay out the bad events `A_1, …, A_n`, where `A_i` says "the random object fails at spot `i`," and a good object is exactly an outcome on which none of the `A_i` fire. So I want `P(Ā_1 … Ā_n) > 0`.

Two ways to get there are easy and I lean on them constantly. If the bad events are independent of each other and each has probability below one, then `P(Ā_1 … Ā_n) = ∏_i (1 − P(A_i))`, and a product of positive numbers is positive — it doesn't matter how big the `P(A_i)` are individually, or how huge their sum is. That's the strongest mode there is. And if I can't get independence, there's always the union bound: if `Σ_i P(A_i) < 1` then `P(⋃ A_i) ≤ Σ_i P(A_i) < 1`, so there's room left over and `P(Ā_1 … Ā_n) > 0`.

But the problems that actually stop me live exactly in the gap between these two. Picture a `k`-uniform hypergraph and the question of whether I can two-colour the vertices with no edge monochromatic. Colour each vertex red or blue by an independent fair coin; let `A_f` be "edge `f` came out monochromatic," so `P(A_f) = 2 · 2^{-k} = 2^{-(k-1)}` — small, exponentially small in `k`. Lovely. Now how many edges does the hypergraph have? Possibly millions. So `Σ_f P(A_f)` is gigantic, way past one, and the union bound is dead on arrival: it tells me the hypergraph is two-colourable only if the *total* number of edges is below `2^{k-1}`. And independence? The events `A_f` are absolutely not independent — two edges that share even one vertex are correlated through that shared coin.

What nags me is that the union-bound verdict feels morally wrong. Take a hypergraph with an astronomical number of edges, but arranged so that any given edge only touches a few others — most pairs of edges are disjoint. Disjoint edges use *disjoint* coins, so `A_f` and `A_g` for disjoint `f, g` are genuinely independent. The whole system is "mostly independent," lightly stitched together at the seams where edges meet. Intuitively such a thing should be two-colourable no matter how many edges it has, because each edge is only really fighting with its handful of neighbours. But the union bound charges me the full `2^{-(k-1)}` for every single edge, including all the disjoint ones it should be giving me for free, and so it caps me at a *global* edge count when the real obstruction is plainly *local*. The structural results I know about this — Lovász's and Woodall's theorems that a non-two-colourable uniform hypergraph must hide a high-degree vertex, or violate `|⋃_{E∈H'} E| > |H'|` somewhere — all point the same way: the obstruction to colouring is a *local* density, a few edges piled up, not a head-count of edges. So the condition I'm hunting for should be local too.

So here's the precise thing I want, stated honestly: many bad events, each one unlikely, their probabilities summing to far more than one, *not* independent — but with the dependence confined, so that each `A_i` is entangled with only a few of the others and independent of everything else. Does a system like that still leave positive probability that none of the bad events fires? And if so, exactly how sparse does the entanglement have to be?

Let me get the dependence structure written down before anything else. For each `A_i` there's a small set of "neighbour" events it can interact with, and `A_i` is independent of all the rest. I'll draw a graph `G` on the events, joining `A_i` to the ones it might depend on, so that `A_i` is independent of the events it is *not* joined to. Call the non-neighbours of `A_i` its "far" events.

I have to be careful about what "independent of the far events" means, and this is a place I could easily fool myself. It is not enough that `A_i` be independent of each far event one at a time. I need `A_i` independent of the far events *as a whole* — independent of every Boolean combination of them, i.e. `P(A_i | any conjunction of far events and their complements) = P(A_i)`. Pairwise independence is strictly weaker and would sink me. The clean reminder of why: take `x_1, x_2, x_3` independent uniform bits in `Z/2Z` and set `A_i = "x_{i+1} + x_{i+2} = 0"` cyclically. Any two of these are independent — knowing one parity tells you nothing about another single parity — yet all three together are not, since any two of the `A_i` force the third. So for those events the empty graph is *not* a valid dependency graph even though no two events are correlated; I'd need at least a couple of edges. The graph I want is built from genuine *mutual* independence between `A_i` and its far set, not from "which pairs happen to be uncorrelated." Good — that's the trap noted; I'll insist on mutual independence from the far set throughout.

Now, what is the quantity I should actually be trying to bound? The union bound failed because it works with the raw `P(A_i)`. But the thing that's truly exact, dependence or no dependence, is the chain rule:

    P(Ā_1 … Ā_n) = P(Ā_1) · P(Ā_2 | Ā_1) · P(Ā_3 | Ā_1 Ā_2) · ⋯ · P(Ā_n | Ā_1 … Ā_{n−1}).

This product is positive exactly when every factor is positive, and the `k`-th factor is `1 − P(A_k | Ā_1 … Ā_{k−1})`. So everything comes down to the *conditional* probability of a bad event *given that some of the others were avoided*. That's the right object — not the unconditional `P(A_k)` the union bound clings to, but `P(A_k | some of the others avoided)`. If I can show every such conditional stays bounded away from `1`, every factor in the product is bounded away from `0`, and the product is positive. So the target sharpens to: bound `P(A_i | a set of the other Ā)` by something strictly below `1`.

Let me just try to bound `P(A_1 | Ā_2 … Ā_n)` and see what the locality buys me. If `d = 0`, then there are no neighbours and the product formula has already solved the problem, so assume `d ≥ 1`. The instinct is to pull apart the conditioning into the part `A_1` actually depends on and the part it doesn't. Say `A_1`'s neighbours in `G` are `v_2, …, v_q`; since the maximum degree is `d`, I have `q − 1 ≤ d`, i.e. `q ≤ d + 1`. Split the conditioning bar accordingly and use the definition of conditional probability:

    P(A_1 | Ā_2 … Ā_n) = P(A_1 Ā_2 … Ā_q | Ā_{q+1} … Ā_n) / P(Ā_2 … Ā_q | Ā_{q+1} … Ā_n).

Here `Ā_2 … Ā_q` are the neighbours and `Ā_{q+1} … Ā_n` are the far events. Now I lean on independence for the numerator. Dropping the `Ā_2 … Ā_q` from inside the joint event only enlarges it, so `P(A_1 Ā_2 … Ā_q | Ā_{q+1} … Ā_n) ≤ P(A_1 | Ā_{q+1} … Ā_n)`. And `A_1` is independent of the far events `{A_{q+1}, …, A_n}` as a set — that's exactly the property I built the graph to have — so `P(A_1 | Ā_{q+1} … Ā_n) = P(A_1)`. The numerator is at most `P(A_1)`. The conditioning on the far events evaporated, which is the whole point of locality: `A_1` doesn't feel them.

Now the denominator, `P(Ā_2 … Ā_q | Ā_{q+1} … Ā_n)` — the probability that none of `A_1`'s `≤ d` neighbours fired, given the far events were avoided. I want this *not too small*, because it sits underneath. Write it as `1 −` the probability that *at least one* neighbour fired, and union-bound that one piece — and here the union bound is fine, because I'm only summing over the `≤ d` neighbours, a *local* set, not over everything:

    P(Ā_2 … Ā_q | Ā_{q+1} … Ā_n) = 1 − P(A_2 + ⋯ + A_q | Ā_{q+1} … Ā_n) ≥ 1 − Σ_{i=2}^{q} P(A_i | Ā_{q+1} … Ā_n).

So the denominator is at least `1 − Σ_{i=2}^{q} P(A_i | far avoided)`. But each term in that sum is again a *conditional probability of a bad event given some others were avoided* — the very kind of quantity I'm in the middle of bounding. That's the signal to set up an induction. More precisely, for a term such as `P(A_i | Ā_{q+1} … Ā_n)`, I can look at the smaller family consisting of `A_i` together with those far events; the same degree and independence conditions are inherited. If I can carry, as an inductive hypothesis, a uniform bound `P(A_i | any others avoided) ≤ β` on every such conditional, then the denominator is `≥ 1 − (q−1) β ≥ 1 − d β`, and the whole thing reads

    P(A_1 | Ā_2 … Ā_n) ≤ P(A_1) / (1 − d β).

Now I want this to *also* be `≤ β`, so the bound reproduces itself and the induction closes. That asks for `P(A_1) / (1 − dβ) ≤ β`, i.e. `P(A_1) ≤ β(1 − dβ)`. The right side is a downward parabola in `β`, maximized at `β = 1/(2d)`, where it equals `(1/(2d))(1 − 1/2) = 1/(4d)`. So the cleanest place to stand is `β = 1/(2d)`, and the hypothesis I should carry is

    P(A_i | any subset of the others avoided) ≤ 1/(2d),

provided `P(A_i) ≤ 1/(4d)`. Let me check it shuts: with `β = 1/(2d)`, the denominator `≥ 1 − d·(1/(2d)) = 1 − 1/2 = 1/2`, and `q − 1 ≤ d` so the sum has at most `d` terms each `≤ 1/(2d)` — good, exactly `≤ 1/2`. Then

    P(A_1 | Ā_2 … Ā_n) ≤ P(A_1) / (1/2) = 2 P(A_1) ≤ 2 · 1/(4d) = 1/(2d).

It reproduces. So I've got a theorem, and let me state it the way it actually came out. Let `G` be a finite graph, maximum degree `d ≥ 1`, with an event `A_i` attached to each vertex `v_i`, each `A_i` independent of the set of its far events. If `P(A_i) ≤ 1/(4d)` for every `i`, then `P(Ā_1 … Ā_n) > 0`. The proof is the stronger claim `P(A_1 | Ā_2 … Ā_n) ≤ 1/(2d)`, by induction on `n`. Base `n = 1`: there's nothing to condition on, `P(A_1) ≤ 1/(4d) ≤ 1/(2d)`, fine. Inductive step is exactly the split I just did — numerator `≤ P(A_1) ≤ 1/(4d)` by independence-from-the-far-set, denominator `≥ 1 − Σ_{i=2}^{q} P(A_i | far avoided) ≥ 1 − (q−1)/(2d) ≥ 1/2` by applying the inductive hypothesis to the smaller family formed by that neighbour and the far events in its conditioning set, so the ratio is `≤ (1/(4d))/(1/2) = 1/(2d)`. And once every conditional `P(A_k | Ā_1 … Ā_{k−1})` is `≤ 1/(2d) < 1`, every factor `1 − P(A_k | earlier avoided)` in the chain-rule product is positive, so `P(Ā_1 … Ā_n) > 0`. There it is — local dependence is enough, and the explicit price is `P(A_i) ≤ 1/(4d)`.

Let me immediately cash this out on the colouring problem that started the whole thing, to make sure the local condition really does what I hoped. Take a `k`-uniform hypergraph `H` and colour each vertex red or blue by an independent fair coin. For an edge `f`, the bad event `A_f` is that all `k` vertices of `f` have the same colour, so `P(A_f) = 2 · 2^{-k} = 2^{-(k-1)}`. Build the dependency graph as the "line graph": join `A_f` to `A_g` exactly when `f ∩ g ≠ ∅`. Two edges that meet share a vertex, hence a coin, so I cannot claim independence and I *do* join them; two disjoint edges use disjoint coins, and `A_f` is then independent of the whole set of `A_g` over disjoint `g` — that's the mutual independence I need, and it's clear here because the colouring is by independent coins on disjoint vertex sets. If every edge meets at most `d` other edges, the maximum degree is `d`, and my condition `P(A_f) ≤ 1/(4d)` becomes `2^{-(k-1)} ≤ 1/(4d)`, i.e. `d ≤ 2^{k-1}/4 = 2^{k-3}`. So: if every edge of a `k`-uniform hypergraph meets at most `2^{k-3}` other edges, the hypergraph is two-colourable — **no matter how many edges it has.** A *local* degree bound has completely replaced the union bound's *global* edge count of `2^{k-1}`. That's exactly the trade I wanted: the disjoint edges cost me nothing.

And the same machine gives a SAT statement for free. A `k`-CNF is a conjunction of clauses, each a disjunction of `k` literals on distinct variables. Assign every variable a uniform random truth value; let `A_i` be "clause `i` is violated," which happens only when all `k` of its literals come out false, so `P(A_i) = 2^{-k}` exactly. Two clauses are entangled only if they share a variable; join them in `G` precisely then. If each clause shares a variable with at most `d` others, the condition `2^{-k} ≤ 1/(4d)`, i.e. `d ≤ 2^{k-2}`, guarantees a satisfying assignment exists, regardless of the total number of clauses. Local sparsity implies satisfiability.

But now I sit with the constant `1/(4d)` and it bothers me, because the union-bound step in the denominator was lazy, and I want to know how much I overpaid. Two places leak. First, I forced a *single* threshold `1/(2d)` on every event; but in the SAT and colouring problems all the bad events have equal probability, sure, yet in general I might have bad events of wildly different likelihoods, and a uniform bound is wasteful — a very unlikely event should be allowed more dangerous neighbours, and a likely one fewer. Second, I bounded the denominator by `1 − Σ (neighbour conditionals)`, a first-order union bound, when the true joint probability that none of the neighbours fired is more like a *product* `∏ (1 − something)` — and products beat sums. Let me redo the bound keeping both ideas: give each event `A_i` its own slack `x_i ∈ [0,1)`, and aim to prove the product-form conditional bound

    P(A_i | any subset of the others avoided) ≤ x_i.

What hypothesis on `P(A_i)` makes this go through? Run the same split. Bound `P(A_i | ∩_{j∈S} Ā_j)` for `i ∉ S`. Partition `S` into the part touching `A_i`'s neighbours, `S_1 = S ∩ N(i)`, and the far part `S_2 = S \ S_1`. By conditional probability,

    P(A_i | ∩_{j∈S} Ā_j) = P(A_i ∩_{j∈S_1} Ā_j | ∩_{l∈S_2} Ā_l) / P(∩_{j∈S_1} Ā_j | ∩_{l∈S_2} Ā_l).

Numerator: drop the `∩_{S_1} Ā_j` to enlarge the event, `≤ P(A_i | ∩_{S_2} Ā_l)`, and `A_i` is independent of the far set `S_2`, so this `= P(A_i)`. Denominator: list `S_1 = {j_1, …, j_r}` (note `r ≤ |N(i)| ≤ d`) and telescope by the chain rule,

    P(∩_{j∈S_1} Ā_j | ∩_{l∈S_2} Ā_l)
    = ∏_{t=1}^{r} P(Ā_{j_t} | Ā_{j_1} … Ā_{j_{t-1}} ∩ ∩_{l∈S_2} Ā_l),

and each factor is `1 − P(A_{j_t} | some others avoided) ≥ 1 − x_{j_t}` by the inductive hypothesis (the conditioning set has size `|S_2| + t − 1 < |S|`, which is what licenses the induction — I'm inducting on `|S|`). So the denominator is `≥ ∏_{t=1}^{r} (1 − x_{j_t}) ≥ ∏_{j ∈ N(i)} (1 − x_j)`, the last step because adding the missing neighbour factors, each in `(0,1]`, can only decrease the product. Putting the two together,

    P(A_i | ∩_{j∈S} Ā_j) ≤ P(A_i) / ∏_{j∈N(i)} (1 − x_j).

For this to be `≤ x_i` and close the induction, I need precisely

    P(A_i) ≤ x_i ∏_{j∈N(i)} (1 − x_j).

That's the asymmetric hypothesis, and it's beautiful — the product over neighbours is exactly what the telescoped denominator produces, so the two halves of the proof fit together with nothing wasted. Let me confirm the base case and the conclusion. Base `S = ∅`: `P(A_i) ≤ x_i ∏_{N(i)}(1 − x_j) ≤ x_i` since every factor `≤ 1`, good. And once `P(A_i | earlier avoided) ≤ x_i` for all `i`, the chain rule gives

    P(Ā_1 … Ā_n) = ∏_{k=1}^{n} P(Ā_k | Ā_1 … Ā_{k-1}) = ∏_{k} (1 − P(A_k | earlier avoided)) ≥ ∏_{k} (1 − x_k) > 0,

a strict positive lower bound, not just positivity. So the general statement: events `A_1, …, A_n` with a dependency graph, real numbers `x_i ∈ [0,1)` with `P(A_i) ≤ x_i ∏_{j∈N(i)} (1 − x_j)` for every `i`; then `P(Ā_1 … Ā_n) ≥ ∏_i (1 − x_i) > 0`.

Now back to the constant, with the sharper tool in hand. In the symmetric case — all `P(A_i) ≤ p`, every event independent of all but at most `d` others — I get to *choose* the `x_i`. If `d = 0`, full independence already settles positivity as soon as `p < 1`, so the interesting case is `d ≥ 1`. By symmetry take all the slacks equal, `x_i = x`. The hypothesis `P(A_i) ≤ x_i ∏_{j∈N(i)}(1 − x_j)` is implied by `p ≤ x (1 − x)^d`, since `A_i` has at most `d` neighbours and `(1−x) ≤ 1`. So I want to choose `x ∈ (0,1)` to make `x(1−x)^d` as large as possible — that's the most slack the proof can extract. Differentiate: `d/dx [x(1−x)^d] = (1−x)^d − d x (1−x)^{d-1} = (1−x)^{d-1}(1 − x − dx)`, which vanishes at `x = 1/(d+1)`. Plug it in:

    x(1−x)^d = (1/(d+1)) · (1 − 1/(d+1))^d = (1/(d+1)) · (d/(d+1))^d.

Now what is `(d/(d+1))^d = (1 − 1/(d+1))^d` as a number? It's the kind of expression that converges to `1/e`, and I should pin the direction of the inequality rather than wave at it. Set `m = d + 1`, so the factor is `(1 − 1/m)^{m-1}`. I know `(1 − 1/m)^m` increases up to `1/e` and `(1 − 1/m)^{m-1}` *decreases* down to `1/e`, so `(1 − 1/m)^{m-1} ≥ 1/e` for all `m ≥ 2` — the `m-1` version sits *above* the limit. Let me double-check that's the version I have: my exponent is `d = m − 1`, yes, the `m-1` one, so `(1 − 1/(d+1))^d ≥ 1/e`. (Quick sanity check at `d = 1`: `(1/2)^1 = 0.5 ≥ 1/e ≈ 0.368`. At `d = 2`: `(2/3)^2 = 0.444 ≥ 0.368`. Decreasing toward `1/e` from above — consistent.) Therefore

    x(1−x)^d ≥ (1/(d+1)) · (1/e) = 1 / (e(d+1)).

So with `x = 1/(d+1)`, the hypothesis `p ≤ x(1−x)^d` is satisfied as soon as `p ≤ 1/(e(d+1))`, i.e.

    e · p · (d + 1) ≤ 1.

That's the symmetric form. The exact finite threshold produced by the equal choice of `x` is `(1/(d+1))(d/(d+1))^d`; the `e` form is the clean universal sufficient condition obtained by replacing the geometric factor with its lower bound `1/e`, not by pretending the finite factor is exactly `1/e`. Compared with the first crude pass, the exact product bound is never worse than `1/(4d)` and improves once `d > 1`; the simplified `1/(e(d+1))` version is asymptotically sharper and already beats `1/(4d)` for `d ≥ 3`. That is the right way to read the constant: `e` is the limiting exchange rate between a uniform slack and a degree-`d` neighbourhood.

Let me re-run the two applications with the tightened constant to see the real reach. Hypergraph two-colouring: `p = 2^{-(k-1)}`, and `e p (d+1) ≤ 1` means `d + 1 ≤ 2^{k-1}/e`, so an edge may meet up to about `2^{k-1}/e − 1` other edges and the hypergraph is still two-colourable, for arbitrarily many edges. As a sanity check on a clean case: a `k`-uniform `k`-regular hypergraph (every vertex in exactly `k` edges) has each edge meeting at most `k(k−1)` others — its `k` vertices each pull in `k−1` more edges — so `d = k(k−1)` and the condition `e(k(k−1)+1) 2^{-(k-1)} ≤ 1` holds once `2^{k-1}` outgrows `e(k^2 − k + 1)`, which it does for all `k` from `9` on; so every `k`-uniform `k`-regular hypergraph with `k ≥ 9` is two-colourable. The exponential `2^{k-1}` crushes the polynomial degree — the whole point is that locality is cheap. And SAT: `p = 2^{-k}`, so `e 2^{-k}(d+1) ≤ 1` lets each clause share a variable with up to `2^k/e − 1` others while a satisfying assignment is still forced; the total number of clauses is irrelevant.

There's one more refinement that falls straight out of the asymmetric form and is worth keeping, because the symmetric `e p (d+1) ≤ 1` is the wrong tool when the bad events have very different probabilities — the moment that happens, a single uniform `p` and `d` throws away everything. Suppose instead each `P(A_i) < 1/2` and, for every `i`, the *total* probability mass of its neighbours is small, `Σ_{j∈N(i)} P(A_j) ≤ 1/4`. Choose `x_i = 2 P(A_i)`, which is in `[0,1)` since `P(A_i) < 1/2`. Then, bounding the product below by its first-order union-bound,

    x_i ∏_{j∈N(i)} (1 − x_j) ≥ x_i (1 − Σ_{j∈N(i)} x_j) = 2 P(A_i)(1 − Σ_{j∈N(i)} 2 P(A_j)) ≥ 2 P(A_i)(1 − 1/2) = P(A_i),

so the asymmetric hypothesis `P(A_i) ≤ x_i ∏(1 − x_j)` holds and `P(Ā_1 … Ā_n) > 0`. (The step `∏(1 − x_j) ≥ 1 − Σ x_j` is the standard product-vs-sum inequality, and `Σ x_j = 2 Σ P(A_j) ≤ 1/2`.) This is the version to reach for whenever the bad events are of genuinely different natures — different sizes, different probabilities — and it costs nothing beyond the general theorem.

Let me close the loop by saying out loud the causal chain, because it's short and it's the whole story. The probabilistic method has two comfortable modes — full independence, where a product of complementary probabilities stays positive, and the union bound, where a global budget `Σ P(A_i) < 1` does the job — and the problems I care about fall in the crevice between them: many unlikely bad events, summing past one, not independent, but only *locally* dependent. The union bound dies there because it pays full price for events that are disjoint and hence free. The fix is to stop bounding the raw `P(A_i)` and instead bound the conditional probability of a bad event given that some others were avoided — the exact factor the chain rule expansion of `P(Ā_1 … Ā_n)` is built from — and to set up an induction on the size of the conditioning set. Splitting that conditioning into the neighbours and the far events lets independence collapse the numerator to `P(A_i)` and lets the chain rule telescope the denominator into a product over neighbours; matching the two forces the hypothesis `P(A_i) ≤ x_i ∏_{j∈N(i)}(1 − x_j)`, which yields `P(Ā_1 … Ā_n) ≥ ∏(1 − x_i) > 0`. Optimising a single uniform slack `x = 1/(d+1)` across a degree-`d` neighbourhood gives the exact product `(1/(d+1))(d/(d+1))^d`, and the lower bound `(d/(d+1))^d ≥ 1/e` gives the memorable symmetric condition `e · p · (d + 1) ≤ 1`. And that one inequality converts a hopeless global count into a mild local degree bound, so that a `k`-uniform hypergraph is two-colourable when each edge meets at most about `2^{k-1}/e` others — however many edges there are — and a `k`-CNF is satisfiable when each clause shares variables with at most about `2^k/e` others, however many clauses there are.

Here is the theorem in final form, plus a small certificate helper that checks the hypotheses numerically.

```python
from math import e, prod

# ---- The Lovász Local Lemma -----------------------------------------------
# Setup. Bad events A_1,...,A_n on a common probability space, with a dependency
# graph: each A_i is (mutually) independent of the SET of all events it is not
# joined to (its "far" events). N(i) = neighbours of i.
#
# GENERAL (asymmetric) form.  If there exist x_i in [0,1) with
#         P(A_i) <= x_i * prod_{j in N(i)} (1 - x_j)        for all i,
# then    P(no A_i occurs) >= prod_i (1 - x_i) > 0.
#
# Proof (recap, lived above): induct on |S| to show P(A_i | ∩_{j in S} Ā_j) <= x_i.
# Split S into S1 = S ∩ N(i) and S2 = S \ S1. Independence from the far set S2
# collapses the numerator to P(A_i); the chain rule lower-bounds the denominator
# by prod_{j in N(i)}(1 - x_j); matching them gives the bound. Then the chain
# rule on P(Ā_1...Ā_n) = prod_k (1 - P(A_k | earlier)) >= prod_k (1 - x_k).

def general_lll_certificate(p, neighbours, x):
    """Certify the asymmetric hypothesis P(A_i) <= x_i * prod_{j in N(i)} (1 - x_j)."""
    for i in range(len(p)):
        rhs = x[i] * prod(1.0 - x[j] for j in neighbours[i])
        if not (0.0 <= x[i] < 1.0 and p[i] <= rhs + 1e-12):
            return False
    return True  # => P(no bad event) >= prod_i (1 - x_i) > 0


# SYMMETRIC corollary.  All P(A_i) <= p, each A_i independent of all but <= d others.
# Take x_i = 1/(d+1); then x_i*(1-x_i)^d = (1/(d+1))*(1 - 1/(d+1))^d >= 1/(e(d+1)),
# using (1 - 1/(d+1))^d >= 1/e.  So the hypothesis holds whenever
#         e * p * (d + 1) <= 1.
def symmetric_lll(p, d):
    """True => a good object (avoiding every bad event) provably exists."""
    if d == 0:
        return p < 1.0
    return e * p * (d + 1) <= 1.0


# ---- Application 1: k-uniform hypergraph 2-colouring (property B) ----------
# Colour each vertex red/blue by an independent fair coin. A_f = "edge f mono",
# P(A_f) = 2^{-(k-1)}. A_f depends only on edges meeting f. If each edge meets
# <= d others and e * 2^{-(k-1)} * (d+1) <= 1, the hypergraph is 2-colourable.
def hypergraph_two_colourable(k, max_edge_intersections):
    p = 2.0 ** (-(k - 1))
    return symmetric_lll(p, max_edge_intersections)

# k-uniform k-regular: every edge meets <= k(k-1) others.
def k_regular_two_colourable(k):
    return hypergraph_two_colourable(k, k * (k - 1))


# ---- Application 2: k-SAT --------------------------------------------------
# Uniform random assignment. A_i = "clause i violated", P(A_i) = 2^{-k} exactly.
# Clauses depend only when they share a variable. If each clause shares a variable
# with <= d others and e * 2^{-k} * (d+1) <= 1, a satisfying assignment exists.
def k_sat_satisfiable(k, max_shared_clauses):
    return symmetric_lll(2.0 ** (-k), max_shared_clauses)


if __name__ == "__main__":
    # k-uniform k-regular hypergraphs are 2-colourable from k = 9 on.
    for k in range(2, 12):
        print(k, k_regular_two_colourable(k))   # flips to True at k = 9
    # k-SAT: each clause may share variables with up to ~2^k/e - 1 others.
    k = 5
    bound = int(2 ** k / e) - 1
    print("k=5: satisfiable up to d =", bound, k_sat_satisfiable(k, bound))
```
