OK, let me think this through from scratch. I want the global minimum cut of an undirected graph — the cheapest way to split the vertices into two nonempty sides. The textbook route is to lean on max-flow/min-cut: a single s–t min cut is a single max flow, but here there's no fixed s and t, so I'd have to try all the ways the two sides could fall. If I'm clever I fix one vertex s as a permanent source and run n−1 max-flow computations against every possible t, because whatever the global min cut is, it puts s on one side and *somebody* on the other. That works, but it drags in the whole flow apparatus — residual graphs, augmenting paths, push-relabel — and the cost is n−1 flows, something like O(mn) per flow, so O(mn²)-ish. It feels enormously heavy for a quantity that's really just "what's the weakest seam in this graph." I want something elementary. Let me stare at the structure of the problem instead and see if there's a shortcut the flow methods are throwing away.

Here's a fact so cheap it's almost embarrassing. Sum the degrees of all vertices: every edge gets counted exactly twice, once at each endpoint, so Σ_u deg(u) = 2m. The average degree is 2m/n. Now isolate a single vertex u: putting just u on one side and everyone else on the other is a perfectly valid cut, and its value is exactly deg(u). So the global min-cut value k can't exceed deg(u) for *any* u — in particular it can't exceed the *minimum* degree, and certainly not the average degree. So k ≤ 2m/n. Rearrange: m ≥ nk/2.

Let me sit with what that inequality is telling me, because it's more than a bound on k. Read it the other way: a graph whose min cut is k must have at least nk/2 edges, but the cut itself only has k of them. So the cut edges are a *vanishing fraction* of all edges — at most k out of ≥ nk/2, i.e. a fraction ≤ 2/n. If I reach into the graph and grab a uniformly random edge, the chance it's one of the k cut edges is at most k/m ≤ k/(nk/2) = 2/n. For any decent-sized graph that's tiny. A random edge is almost surely *internal* to one side of the min cut. The min cut is a needle; the graph is overwhelmingly hay. The flow algorithms compute as if every edge mattered equally — but the thing I'm looking for is sparse, and that sparsity is begging to be used.

So how do I *use* "a random edge is almost never a cut edge"? Searching for a cut is really making a sequence of decisions: for each pair of vertices, are they on the same side or opposite sides? Is there an operation that lets me *commit* "these two are on the same side"? Yes — contract the edge between them. Merge u and v into one supernode; its incident edges are u's edges plus v's edges; the edge(s) that ran directly between u and v become self-loops and I throw them away; edges that both had to a common neighbor w become parallel edges and I *keep* them, because their multiplicity records how many original edges run between the merged blob and w. The graph becomes a multigraph and loses one vertex.

Why is committing-to-same-side the right primitive and not, say, deleting edges? Because deletion would change which cuts are even available — drop the wrong edge and I might disconnect the graph or destroy the min cut. Contraction is conservative in exactly the direction I want: when I merge u and v, I'm restricting attention to cuts that keep u and v together. Every cut of the contracted multigraph is a genuine cut of the original graph (the one that keeps all merged pairs together), so contraction can never *invent* a cut smaller than the true min cut — the min cut of the contracted graph is always ≥ k. And it stays equal to k for as long as I never contract an actual cut edge. That's the whole game: if I only ever contract internal edges, the min cut is preserved; the only way to ruin it is to contract one of the k cut edges and force two vertices that belong on opposite sides into the same supernode.

Now combine the two observations. A random edge is a cut edge with probability ≤ 2/n. Contracting a non-cut edge preserves the min cut. So: just keep contracting *uniformly random* edges. Each contraction is almost surely safe. Keep going until only two supernodes remain — at that point the parallel edges between them are precisely a cut of the original graph, and if I got lucky and never touched a cut edge, it's *the* min cut. Read off the number of edges between the last two supernodes and that's my cut value.

Let me be careful about "uniformly random edge," because there's a tempting cheaper version. I could pick a random *vertex* and then a random neighbor. But that over-samples high-degree vertices and double-counts — the probability of selecting a given edge would be deg-weighted, not uniform, and then "P[edge is a cut edge] ≤ 2/n" no longer cleanly holds. The clean analysis needs the edge chosen uniformly over the *m* edges. In an adjacency-list multigraph I can get that by picking an endpoint with probability proportional to its degree and then a uniform incident edge — overall uniform over edge-endpoints, which is what I want. (The vertex-first version is actually the appealing *heuristic* reading — random edges come from dense regions, so contraction tends to glue together things that belong together — but for a guarantee I'll keep it uniform over edges.)

Now, does it actually work often enough? One contraction is safe with probability ≥ 1 − 2/n. But there are n − 2 contractions to do (from n vertices down to 2), and the danger *grows* as the graph shrinks. Let me track it honestly. Fix a particular min cut C of value k. After some contractions the graph has i supernodes. The key thing: the min cut of the *current* multigraph is still ≥ k (contraction never lowers it), so the very same handshake argument applies to the current graph — its min cut is ≥ k, so it has ≥ ik/2 edges, so a uniformly random current edge is a C-edge with probability ≤ k/(ik/2) = 2/i. Therefore

  P[the contraction at i supernodes avoids C] ≥ 1 − 2/i = (i − 2)/i.

C survives the whole run only if it survives every contraction, from i = n down to i = 3 (the last contraction takes 3 supernodes to 2). Multiply:

  P[C survives] ≥ Π_{i=3}^{n} (i − 2)/i = (n−2)/n · (n−3)/(n−1) · (n−4)/(n−2) · … · 2/4 · 1/3.

Stare at this product for a second. Each numerator (i−2) cancels against the denominator two fractions later. After all the cancellation, the only denominators left standing are the two largest, n and n−1, and the only numerators left are the two smallest, 2 and 1. So the whole thing collapses to

  Π = (2 · 1)/(n · (n−1)) = 2/(n(n−1)) = 1/C(n,2).

So one run of "contract random edges down to two supernodes" returns the min cut with probability at least 2/(n(n−1)). That's small — it goes to zero like 1/n² — but it is *positive and quantified*, which is exactly what the bare heuristic lacked.

A small probability is fine as long as it's bounded below, because independent repetition crushes failure exponentially. If one run succeeds with probability p ≥ 2/(n(n−1)), then T independent runs all fail with probability at most (1 − p)^T ≤ e^{−pT}. To make that 1/n I need pT ≈ ln n, i.e. T ≈ (1/p) ln n ≈ (n(n−1)/2) ln n = C(n,2) ln n. So: run the contraction C(n,2) ln n ≈ ½ n² ln n times, keep the smallest cut seen, and the answer is the true global min cut except with probability ≤ 1/n. Each run is n − 2 contractions; with an adjacency-list multigraph each contraction is O(n) work (splice one supernode's edge list into another's), so a run is O(n²) and the whole thing is O(n⁴ log n). Flow-free, two lines to state, provably correct with high probability. That alone justifies the approach over the flow machinery on the conceptual axis, even if n⁴ is not yet fast.

Before I optimize, let me notice something the analysis gives me for free, because it's a sanity check that the bound is tight in spirit. Every *distinct* min cut C survives a single run with probability ≥ 1/C(n,2), and on any one run the events "this run outputs exactly C" are mutually exclusive across different C's (a run outputs one cut). So if there were N distinct min cuts, summing their disjoint success probabilities, N · (1/C(n,2)) ≤ 1, giving N ≤ C(n,2). A graph can have at most O(n²) distinct minimum cuts. That falls straight out of the same telescoping — a strong hint the 1/C(n,2) bound is the real rate, not a loose artifact.

Now, n⁴ bothers me. Where is the work being wasted? Look back at the telescoping product and *where the risk lives*. The early factors — (n−2)/n, (n−3)/(n−1), … — are all extremely close to 1. When i is large, the per-step kill probability 2/i is negligible; contracting the first edge of a thousand-vertex graph essentially never hits the cut. All the danger is concentrated at the *end*, when i is small and 2/i is no longer tiny. The product is overwhelmingly dragged down by its last few factors. Yet the naive algorithm treats every run as fully independent from scratch: it redoes the (almost-always-safe, almost-free-of-risk) early contractions over and over, once per repetition. That's the waste. The expensive early contractions — expensive in *count*, cheap in *risk* — are being recomputed millions of times when they almost never go wrong.

So I want to *share* the safe early work across many runs and only pour extra independent effort into the dangerous late part. Concretely: do the random contractions down to some intermediate size *once*, and only after the graph has shrunk — where the per-step failure probability has climbed — should I branch into multiple independent continuations. How far should the shared prefix go? Use the telescoping bound for partial survival. By the same product, the probability that C survives from n supernodes down to t supernodes is

  Π_{i=t+1}^{n} (i−2)/i = (t(t−1))/(n(n−1)) = C(t,2)/C(n,2) ≈ (t/n)².

I want to contract until this partial survival probability is about ½ — that's the natural stopping point, the place where it becomes "as likely as not" that I've already killed the cut, so it's the moment to hedge. Set (t/n)² = ½, i.e. t = n/√2. So: contract down to roughly n/√2 supernodes, at which point the min cut is still intact with probability ≥ ½.

Now the hedge. If a single contraction-to-n/√2 keeps the cut with probability ≥ ½, then to be reasonably sure *some* continuation keeps it, I should make more than one independent continuation from that point — and the cheapest meaningful number is two. Make two independent contracted copies down to n/√2, and recurse on each, solving each smaller instance the same way; return the smaller of the two cuts found. Why exactly two? Because the expected number of the two branches that still contain the intact min cut is 2 × ½ = 1 — I'm running a critical branching process, tuned so that in expectation one surviving copy is carried forward at every level, neither dying out nor exploding. Three or more branches would multiply the running time without being needed; one branch would just be the original algorithm with no hedging. Two is the knife's edge.

Let me write the recursion and check both the time and the success probability, because the whole bet is that this tuning makes them line up. From a graph on n supernodes, if n is below a small constant (say 6), just contract all the way down to 2 by brute force — at that size everything is O(1) and the recursion has no room to help. Otherwise set t = 1 + ⌈n/√2⌉ (the +1 keeps t strictly below n so the recursion makes progress, the ⌈·⌉ keeps it an integer), contract two independent copies of G down to t supernodes, recurse on both, and return the minimum of the two results.

Time first. Contracting from n down to ~n/√2 costs O(n²) (that many contractions, each O(n)), and I do it twice and recurse twice:

  T(n) = 2·T(n/√2) + O(n²).

Solve it. The branching factor is 2 and the size shrinks by √2 each level, so a subproblem of size n/√2 has the *same* O((n/√2)²) = O(n²/2) cost, doubled by the two branches back to O(n²) per *level* — the work is the same Θ(n²) at every level of the recursion. The number of levels is how many times I can divide n by √2 before hitting the constant base case: log_{√2} n = 2 log₂ n = Θ(log n) levels. Same work per level times Θ(log n) levels gives

  T(n) = O(n² log n).

(That's the critical case of the master theorem: n^{log_{√2} 2} = n^{2}, and the per-call cost is also Θ(n²), so a log factor multiplies in.) So one call to the recursive procedure costs O(n² log n) — already as cheap as roughly *one* run of the naive algorithm, and we'll see it's far more reliable.

Now the payoff: the success probability. Let P(n) be the probability that a recursive call on an n-vertex graph returns the true min cut. A call succeeds if at least one of its two branches both *preserves* the cut through the contraction to n/√2 (probability ≥ ½) *and* then succeeds recursively on the smaller instance (probability P(n/√2)). So one branch succeeds with probability ≥ ½·P(n/√2), and with two independent branches,

  P(n) ≥ 1 − (1 − ½·P(n/√2))².

This is where the design pays off. Compare it to the naive algorithm, whose per-run success P decays like 1/n² — geometrically catastrophic. Here, expand the square: writing p = P(n/√2), one step gives P(n) ≥ 2·(½p) − (½p)² = p − p²/4. So as I go *up* one level (from a subproblem of size n/√2 to size n), the success probability barely decreases — it drops only by the quadratically-small p²/4, not by a constant factor. Set q(d) for the success probability at recursion depth d (depth measured from the base case); the recurrence q ↦ q − q²/4 with the depth growing like log n turns the decay from geometric into *harmonic*: the solution behaves like P(n) = Θ(1/log n). Tracking it more carefully, 1/P picks up about ¼ per level and there are Θ(log n) levels, so 1/P(n) ≈ ½ log n and P(n) = Ω(1/log n).

That's the leap. The naive algorithm pays a 1/n² success rate and so needs ~n² repetitions; the recursion, by sharing the cheap-but-safe early contractions and only branching where the risk concentrates, pays just a 1/log n success rate — at the cost of only an O(n² log n) running time per call. The two effects multiply in my favor.

Finish the bookkeeping. One recursive call costs O(n² log n) and succeeds with probability Ω(1/log n). To drive the overall failure below 1/poly(n), repeat the whole recursive call O(log n / P(n)) = O(log n · log n) = O(log² n) times and take the best — failure ≤ (1 − P)^{O(log n / P)} ≤ e^{−Ω(log n)} = 1/poly(n). Total time:

  O(n² log n) · O(log² n) = O(n² log³ n).

So the global min cut, with high probability, in O(n² log³ n) — flow-free, and an order of magnitude past the naive n⁴.

One more thing I want to record, because it connects this to machinery I already trust. There's an equivalent way to do a single random contraction run without writing a contraction routine at all: assign every edge an independent uniform random weight, run Kruskal's algorithm to grow the minimum spanning tree merging components, and stop just before the last merge — i.e., remove the single heaviest edge that Kruskal would add, splitting the spanning forest into two components. Processing edges in increasing random-weight order and merging their components via union-find is *exactly* contracting edges in a uniformly random order; the two components you're left with when you withhold the final union are the two supernodes of a contraction run. So a contraction run is a random-weight MST computation with the last edge dropped, and decades of union-find and MST optimization can be pointed straight at min cuts. Good — the primitive isn't exotic; it's Kruskal in disguise.

Let me write it. The multigraph is an adjacency map {node: [neighbors…]} with parallel edges as repeated entries and self-loops never stored. Contraction down to t supernodes, the cut readout, the plain repeat-and-keep-best, and the recursive Karger–Stein procedure with its two branches.

```python
import copy
import math
import random


def contract(graph, t):
    """Contract uniformly random edges until t supernodes remain.

    Adjacency multigraph {node: [neighbor, ...]}; parallel edges are
    repeated neighbors, self-loops are never stored.
    """
    g = copy.deepcopy(graph)
    while len(g) > t:
        # uniform over edges: endpoint with prob proportional to degree,
        # then a uniform incident edge -> uniform over edge-endpoints,
        # so P[contract a min-cut edge] <= 2/(current #supernodes).
        u = random.choices(list(g.keys()),
                           weights=[len(g[v]) for v in g])[0]
        w = random.choice(g[u])               # the edge (u, w) to merge
        for x in g[w]:                         # redirect w's edges to u,
            if x != u:                         # dropping the u-w self-loop
                g[u].append(x)
        for x in g[w]:
            g[x].remove(w)
            if x != u:
                g[x].append(u)
        del g[w]
    return g


def cut_value(g):
    """Crossing-edge count once two supernodes remain."""
    return len(g[next(iter(g))])


def karger_min_cut(graph, trials):
    """Plain Karger: contract to 2, repeat, keep the smallest cut.
    trials = ceil(C(n,2) * ln n) makes failure <= 1/n."""
    best = math.inf
    for _ in range(trials):
        best = min(best, cut_value(contract(graph, 2)))
    return best


def fast_min_cut(graph):
    """Karger-Stein: share safe early contractions, branch where risk grows.
    Contract to t ~ n/sqrt(2) (min cut survives w.p. >= 1/2), branch twice."""
    n = len(graph)
    if n <= 6:                                 # base case: brute contraction
        return cut_value(contract(graph, 2))
    t = 1 + math.ceil(n / math.sqrt(2))        # survival to t is >= 1/2
    g1 = contract(graph, t)                    # two independent continuations
    g2 = contract(graph, t)
    return min(fast_min_cut(g1), fast_min_cut(g2))


def karger_stein_min_cut(graph, trials):
    """One call succeeds w.p. Omega(1/log n); O(log^2 n) trials -> 1/poly(n)."""
    best = math.inf
    for _ in range(trials):
        best = min(best, fast_min_cut(graph))
    return best
```

So the causal chain: the handshake identity forces m ≥ nk/2, which means a random edge is a min-cut edge with probability only ≤ 2/n; contraction commits two vertices to the same side and never lowers the min cut, so contracting random edges almost surely preserves the cut; telescoping the per-step survival from n down to 2 supernodes gives a clean 2/(n(n−1)) success per run, repeatable to high confidence in O(n⁴ log n). Then, noticing the risk lives entirely in the *late* contractions, I share the safe early prefix down to n/√2 (the survival-½ threshold), branch into exactly two independent continuations (so one survivor is carried forward in expectation), and recurse — turning a 1/n² success rate into Ω(1/log n) at O(n² log n) per call, for a global minimum cut in O(n² log³ n) with high probability.
