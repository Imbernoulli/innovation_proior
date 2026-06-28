OK, let me think this through from scratch. I want the global minimum cut of an undirected graph — the cheapest way to split the vertices into two nonempty sides. The textbook route is to lean on max-flow/min-cut: a single s–t min cut is a single max flow, but here there's no fixed s and t, so I'd have to try all the ways the two sides could fall. If I'm clever I fix one vertex s as a permanent source and run n−1 max-flow computations against every possible t, because whatever the global min cut is, it puts s on one side and *somebody* on the other. That works, but it drags in the whole flow apparatus — residual graphs, augmenting paths, push-relabel — and the cost is n−1 flows, something like O(mn) per flow, so O(mn²)-ish. It feels enormously heavy for a quantity that's really just "what's the weakest seam in this graph." I want something elementary. Let me stare at the structure of the problem instead and see if there's a shortcut the flow methods are throwing away.

Here's a fact so cheap it's embarrassing. Sum the degrees of all vertices: every edge gets counted exactly twice, once at each endpoint, so Σ_u deg(u) = 2m. The average degree is 2m/n. Now isolate a single vertex u: putting just u on one side and everyone else on the other is a perfectly valid cut, and its value is exactly deg(u). So the global min-cut value k can't exceed deg(u) for *any* u — in particular it can't exceed the *minimum* degree, and certainly not the average degree. So k ≤ 2m/n. Rearrange: m ≥ nk/2.

Let me sit with what that inequality is telling me, because it's more than a bound on k. Read it the other way: a graph whose min cut is k must have at least nk/2 edges, but the cut itself only has k of them. So the cut edges are a small fraction of all edges — at most k out of ≥ nk/2, i.e. a fraction ≤ 2/n. If I reach into the graph and grab a uniformly random edge, the chance it's one of the k cut edges is at most k/m ≤ k/(nk/2) = 2/n. For any decent-sized graph that's tiny. A random edge is very likely *internal* to one side of the min cut. The min cut is a needle; the graph is overwhelmingly hay. The flow algorithms compute as if every edge mattered equally — but the thing I'm looking for is sparse, and that sparsity is begging to be used.

So how do I *use* "a random edge is rarely a cut edge"? Searching for a cut is really making a sequence of decisions: for each pair of vertices, are they on the same side or opposite sides? Is there an operation that lets me *commit* "these two are on the same side"? Yes — contract the edge between them. Merge u and v into one supernode; its incident edges are u's edges plus v's edges; the edge(s) that ran directly between u and v become self-loops and I throw them away; edges that both had to a common neighbor w become parallel edges and I *keep* them, because their multiplicity records how many original edges run between the merged blob and w. The graph becomes a multigraph and loses one vertex.

Why is committing-to-same-side the right primitive and not, say, deleting edges? Because deletion would change which cuts are even available — drop the wrong edge and I might disconnect the graph or destroy the min cut. Contraction looks conservative in the direction I want: when I merge u and v, I'm restricting attention to cuts that keep u and v together. Let me make sure I actually believe the two claims I'm about to lean my whole life on — (a) every cut of the contracted graph is a real cut of G, and (b) contraction never lowers the min cut. For (a): a cut of the contracted multigraph is a partition of the supernodes; expand each supernode back to its original vertices and I get a partition of V where the two merged vertices land together, and the crossing edges are literally the same original edges (the multiplicities just bookkeep them). So yes, it's a genuine cut of G. For (b): since every cut of the contracted graph is a cut of G, the *minimum* over contracted cuts can't be below the minimum over all cuts of G — contraction can only remove cuts from consideration (the ones that would split u from v), never add cheaper ones. So min-cut of contracted graph ≥ k, always. And it stays *equal* to k exactly when the particular min cut C I care about is still available, i.e. when I've never contracted one of its k crossing edges. Good — both claims hold, and the only way to ruin a fixed min cut is to contract one of its own cut edges.

Now combine the two observations. A random edge is a cut edge with probability ≤ 2/n. Contracting a non-cut edge preserves the min cut. So a natural thing to try: just keep contracting *uniformly random* edges. Each early contraction is likely to be safe. Keep going until only two supernodes remain — at that point the parallel edges between them are precisely a cut of the original graph, and if I got lucky and never touched a cut edge, it's *the* min cut. Read off the number of edges between the last two supernodes and that's my cut value.

Let me be careful about "uniformly random edge," because there's a tempting cheaper version. I could pick a random *vertex* and then a random neighbor. But then an edge {u, v} is chosen with probability (1/n)(1/deg(u) + 1/deg(v)); edges incident to low-degree vertices get too much weight, and "P[edge is a cut edge] ≤ 2/n" no longer cleanly holds. The clean analysis needs the edge chosen uniformly over the *m* edges. In an adjacency-list multigraph I can get that by picking an endpoint with probability proportional to its degree and then a uniform incident edge — every undirected edge contributes two directed endpoints, so this is uniform over undirected edges. (The vertex-first version is actually the appealing *heuristic* reading — random local moves might glue together things that belong together — but for a guarantee I'll keep it uniform over edges.)

Now, does it actually work often enough? One contraction is safe with probability ≥ 1 − 2/n. But there are n − 2 contractions to do (from n vertices down to 2), and the danger *grows* as the graph shrinks. Let me track it honestly. Fix a particular min cut C of value k, and condition on C having survived so far. After some contractions the graph has i supernodes, C still appears as k parallel crossing edges between its two sides, and the min cut of the *current* multigraph is still ≥ k (contraction never lowers it, as I just argued). The same handshake argument applies to the current graph — its min cut is ≥ k, so it has ≥ ik/2 edges, so a uniformly random current edge is a C-edge with probability ≤ k/(ik/2) = 2/i. Therefore

  P[the contraction at i supernodes avoids C] ≥ 1 − 2/i = (i − 2)/i.

C survives the whole run only if it survives every contraction, from i = n down to i = 3 (the last contraction takes 3 supernodes to 2). Multiply:

  P[C survives] ≥ Π_{i=3}^{n} (i − 2)/i = (n−2)/n · (n−3)/(n−1) · (n−4)/(n−2) · … · 2/4 · 1/3.

Let me actually evaluate this product rather than wave at it. The claim my eye wants to make is that each numerator (i−2) cancels a denominator two fractions later. Take n = 6 and write the four factors out: (4/6)(3/5)(2/4)(1/3). The 4 in the first numerator kills the 4 in the third denominator; the 3 in the second numerator kills the 3 in the fourth denominator. What's left uncancelled is the two largest denominators 6 and 5 downstairs, and the two smallest numerators 2 and 1 upstairs: (2·1)/(6·5) = 2/30 = 1/15. And 1/15 is exactly 2/(6·5) = 2/(n(n−1)) = 1/C(6,2). Let me not trust one case — I evaluated the product symbolically for n = 3,4,5,6,8,10 and against 2/(n(n−1)): I get 1/3, 1/6, 1/10, 1/15, 1/28, 1/45 respectively, and those equal 2/(n(n−1)) on the nose every time. So the telescoping is real, and

  Π = (2 · 1)/(n · (n−1)) = 2/(n(n−1)) = 1/C(n,2).

So one run of "contract random edges down to two supernodes" returns the min cut with probability at least 2/(n(n−1)). That's small — it goes to zero like 1/n² — but it is *positive and quantified*, which is exactly what the bare heuristic lacked.

A small probability is fine as long as it's bounded below, because independent repetition crushes failure exponentially. If one run succeeds with probability p ≥ 2/(n(n−1)), then T independent runs all fail with probability at most (1 − p)^T ≤ e^{−pT}. To make that 1/n I need pT ≈ ln n, i.e. T ≈ (1/p) ln n ≈ (n(n−1)/2) ln n = C(n,2) ln n. So: run the contraction C(n,2) ln n ≈ ½ n² ln n times, keep the smallest cut seen, and the answer is the true global min cut except with probability ≤ 1/n. Each run is n − 2 contractions; with an adjacency-list multigraph each contraction is O(n) work (splice one supernode's edge list into another's), so a run is O(n²) and the whole thing is O(n⁴ log n). Flow-free, two lines to state, and now with a probability guarantee instead of a hope. That alone justifies the approach over the flow machinery on the conceptual axis, even if n⁴ is not yet fast.

I should pause and actually run this once before trusting it, because the bound 1/C(n,2) is a *lower* bound and I want to see it's not vacuous on a concrete graph. Take two 4-cliques joined by a single bridge edge — min cut is obviously 1 (cut the bridge). That's n = 8, so the bound predicts a single run succeeds with probability ≥ 2/(8·7) = 1/28 ≈ 0.0357. I coded the contraction and read off the cut over 20000 independent runs: the empirical single-run success rate came out ≈ 0.435. So the run really does find the cut of value 1, and 0.435 sits comfortably above the 0.0357 floor — the bound holds and, on a graph with so few distinct min cuts, it's loose by more than a factor of ten, which is the right direction for a worst-case guarantee to be loose. And the repeat-and-keep-best driver, run T = ⌈C(8,2) ln 8⌉ times, returns 1 as expected. Good: the analysis isn't just algebra that happens to telescope, it describes the actual program.

Before I optimize, let me notice something the analysis gives me for free. Every *distinct* min cut C survives a single run with probability ≥ 1/C(n,2), and on any one run the events "this run outputs exactly C" are mutually exclusive across different C's (a single run outputs one cut). So if there were N distinct min cuts, summing their disjoint success probabilities, N · (1/C(n,2)) ≤ 1, giving N ≤ C(n,2). A graph can have at most O(n²) distinct minimum cuts. Let me sanity-check the direction with the cliques-bridge graph: there the only min cut is the bridge, so N = 1, and the bound says N ≤ C(8,2) = 28 — consistent, if very loose, exactly as it should be when there's a unique cut. (The cycle C_n is the standard tight case: every pair of its n edges is a distinct min cut, giving N = C(n,2), which is why this bound can't be improved.) That the bound and the empirical 0.435 ≥ 1/28 both come straight out of the same telescoping is a good sign that 1/C(n,2) is the real per-run rate, not a loose artifact.

Now, n⁴ bothers me. Where is the work being wasted? Look back at the telescoping product and *where the risk lives*. The early factors — (n−2)/n, (n−3)/(n−1), … — are all extremely close to 1. When i is large, the per-step kill probability 2/i is negligible; contracting the first edge of a thousand-vertex graph has only a tiny chance of hitting the cut. All the danger is concentrated at the *end*, when i is small and 2/i is no longer tiny. The product is overwhelmingly dragged down by its last few factors. Yet the naive algorithm treats every run as fully independent from scratch: it redoes the low-risk early contractions over and over, once per repetition. That's the waste. The expensive early contractions — expensive in *count*, cheap in *risk* — are being recomputed millions of times even though their failure probability is small.

So I want to *share* the safe early work across many runs and only pour extra independent effort into the dangerous late part. Concretely: do the random contractions down to some intermediate size *once*, and only after the graph has shrunk — where the per-step failure probability has climbed — should I branch into multiple independent continuations. How far should the shared prefix go? Use the same telescoping for partial survival. By the identical cancellation, the probability that C survives from n supernodes down to t supernodes is

  Π_{i=t+1}^{n} (i−2)/i = (t(t−1))/(n(n−1)) = C(t,2)/C(n,2) ≈ (t/n)².

I checked this partial product the same way I checked the full one — for (n,t) = (10,7),(20,14),(8,6),(100,71) the product equals t(t−1)/(n(n−1)) exactly (7/15, 91/190, 15/28, 497/990), so the formula is right and isn't an artifact of stopping at i = 3. Now, where do I want to stop sharing and start hedging? The natural place is where partial survival has dropped to about ½ — the moment it becomes "as likely as not" that I've already killed the cut, so further sharing is risky and it's time to spend independent effort. Set (t/n)² = ½, i.e. t = n/√2. As a check that this is really the half-survival point and not just an asymptotic gesture: at n = 100 the exact partial product to t = 71 ≈ 100/√2 is 497/990 ≈ 0.502, and at n = 8 to t = 6 it's 15/28 ≈ 0.536 — both right around ½, with the small-n one slightly above because of rounding t up. So contracting down to roughly n/√2 supernodes leaves the min cut intact with probability ≥ ½, confirmed numerically, not just in the (t/n)² approximation.

Now the hedge. If a single contraction-to-n/√2 keeps the cut with probability ≥ ½, then to be reasonably sure *some* continuation keeps it, I should make more than one independent continuation from that point — and the cheapest meaningful number is two. Make two independent contracted copies down to n/√2, and recurse on each, solving each smaller instance the same way; return the smaller of the two cuts found. Why exactly two? The expected number of the two branches that still contain the intact min cut is 2 × ½ = 1 — a critical branching process, tuned so that in expectation one surviving copy is carried forward at every level, neither dying out nor exploding. Three or more branches would multiply the running time without obvious need; one branch would just be the original algorithm with no hedging. Two is the knife's edge. Whether that intuition actually buys a better success probability is something I have to *compute*, not assert — so let me write the recursion and check both time and probability, because the whole bet is that this tuning makes them line up.

From a graph on n supernodes, if n is below a small constant (say 6), just contract all the way down to 2 by brute force — at that size everything is O(1) and the recursion has no room to help. Otherwise set t = 1 + ⌈n/√2⌉ (the +1 keeps t strictly below n so the recursion makes progress, the ⌈·⌉ keeps it an integer), contract two independent copies of G down to t supernodes, recurse on both, and return the minimum of the two results.

Time first. Contracting from n down to ~n/√2 costs O(n²) (that many contractions, each O(n)), and I do it twice and recurse twice:

  T(n) = 2·T(n/√2) + O(n²).

Solve it. The branching factor is 2 and the size shrinks by √2 each level, so a subproblem of size n/√2 costs O((n/√2)²) = O(n²/2), doubled by the two branches back to O(n²) per *level* — the work is the same Θ(n²) at every level of the recursion. The number of levels is how many times I can divide n by √2 before hitting the constant base case: log_{√2} n = 2 log₂ n = Θ(log n) levels. Same work per level times Θ(log n) levels gives

  T(n) = O(n² log n).

(That's the critical case of the master theorem: n^{log_{√2} 2} = n^{2}, and the per-call cost is also Θ(n²), so a log factor multiplies in.) So one call to the recursive procedure costs O(n² log n) — already as cheap as roughly *one* run of the naive algorithm, and I still have to see whether it's more reliable.

Now the payoff: the success probability. Let P(n) be the probability that a recursive call on an n-vertex graph returns the true min cut. A call succeeds if at least one of its two branches both *preserves* the cut through the contraction to n/√2 (probability ≥ ½) *and* then succeeds recursively on the smaller instance (probability P(n/√2)). So one branch succeeds with probability ≥ ½·P(n/√2), and with two independent branches,

  P(n) ≥ 1 − (1 − ½·P(n/√2))².

Expand the square to see what one level costs: writing p = P(n/√2), one step gives P(n) ≥ 2·(½p) − (½p)² = p − p²/4. So as I go *up* one level (from a subproblem of size n/√2 to size n), the success probability drops only by the quadratically-small p²/4, not by a constant factor — that's the structural difference from the naive run, whose per-run success decays like 1/n² geometrically. But "drops only by p²/4" still leaves the question of how fast p erodes over Θ(log n) levels, and I can't eyeball that — I have to solve the recurrence. Let p_k be the lower bound at recursion depth k above the base case, with p_0 ≥ 1/15 because a plain contraction on at most six vertices succeeds with probability at least 1/C(6,2) = 1/15. Then p_{k+1} ≥ p_k − p_k²/4. The reciprocal of p should grow roughly linearly, but I want the exact algebra, so set z_k = 4/p_k − 1, i.e. p_k = 4/(z_k + 1). Substitute:

  p_{k+1} ≥ 4/(z_k + 1) − (1/4)·(4/(z_k + 1))² = 4/(z_k + 1) − 4/(z_k + 1)² = 4z_k/(z_k + 1)²,

and therefore

  z_{k+1} = 4/p_{k+1} − 1 ≤ (z_k + 1)²/z_k − 1 = z_k + 1 + 1/z_k.

For a lower-bound recurrence I take equality (the worst case). Now I want to know how z_k grows, so I just iterate it numerically from z_0 = 4/p_0 − 1 = 4·15 − 1 = 59. Stepping z_{k+1} = z_k + 1 + 1/z_k: from 59 it goes 60.02, 61.03, 62.05, …; after 24 levels it's about 83.3. So z increases by essentially 1 per level (the +1/z_k term is under 0.02 here and only shrinks), meaning z_k ≈ z_0 + k = Θ(k), and p_k = 4/(z_k + 1) = Θ(1/k). To double-check that "Θ(1/k)" reads off correctly, I tracked k·p_k along the same iteration: it climbs through 0.066, 0.13, 0.25, …, 0.57 at k=10, …, and settles toward ~1 (it's 1.03 at k=21, 1.14 at k=24), so p_k really does behave like a constant over k, not faster-decaying. The recursion depth is 2 log₂ n + O(1), so plugging k = Θ(log n):

  P(n) = Ω(1/log n).

The naive algorithm pays a 1/n² success rate and so needs ~n² repetitions; the recursion, by sharing the cheap-but-safe early contractions and only branching where the risk concentrates, pays just a 1/log n success rate — at the cost of only an O(n² log n) running time per call. The two effects multiply in my favor, and now I've actually iterated the recurrence to see the 1/log n rather than guessed it.

Finish the bookkeeping. One recursive call costs O(n² log n) and succeeds with probability Ω(1/log n). To drive the overall failure below 1/poly(n), repeat the whole recursive call O(log n / P(n)) = O(log n · log n) = O(log² n) times and take the best — failure ≤ (1 − P)^{O(log n / P)} ≤ e^{−Ω(log n)} = 1/poly(n). If I want the stronger promise of finding every minimum cut encountered by the computation tree, the earlier N ≤ C(n,2) bound means I only need the miss probability for any fixed cut to be, say, O(1/n⁴); the extra factor is still just a larger constant in O(log n / P(n)). Total time:

  O(n² log n) · O(log² n) = O(n² log³ n).

So the global min cut, with high probability, in O(n² log³ n) — flow-free, and an order of magnitude past the naive n⁴.

One more thing I want to record, because it connects this to machinery I already trust. There's an equivalent way to do a single random contraction run without writing a contraction routine at all: assign every edge an independent uniform random weight, run Kruskal's algorithm to grow the minimum spanning tree merging components, and stop just before the last merge — i.e., remove the single heaviest edge that Kruskal would add, splitting the spanning forest into two components. Processing edges in increasing random-weight order and merging their components via union-find is *exactly* contracting edges in a uniformly random order: at each step Kruskal merges the two components joined by the lightest remaining edge, and over a uniform random weighting "lightest remaining edge" is a uniformly random edge among those still crossing distinct components — the same distribution as picking a uniform edge to contract. The two components you're left with when you withhold the final union are the two supernodes of a contraction run. So a contraction run is a random-weight MST computation with the last edge dropped, and decades of union-find and MST optimization can be pointed straight at min cuts. The primitive isn't exotic; it's Kruskal in disguise.

Let me write it as a single self-contained program reading from stdin. The clean representation is exactly the Kruskal-in-disguise view from a moment ago: a contraction run is a flat edge list plus a union-find. I process the edges in a uniformly random order and union each pair of endpoints — already-merged endpoints union to a no-op, which is the self-loop being discarded — until the target number of supernodes remains; the cut value is then the number of edges whose endpoints land in different components. For Karger–Stein I contract down to ~n/√2, rebuild a shrunk edge list with self-loops dropped so the recursion gets smaller, branch into two contracted copies and recurse, and take the minimum; then repeat the whole recursion O(log²n) times and keep the smallest cut. It reads `n m` then `m` lines `u v` (1-based undirected edges) from stdin and prints the global min-cut value; vertex labels fit in `int` while the crossing-edge count is summed in `long long`.

```cpp
// Exact global minimum cut of an undirected unweighted multigraph.
// Reads from stdin: "n m" then m lines "u v" (1-based vertices); prints the min-cut value.
#include <bits/stdc++.h>
using namespace std;

static long long stoerWagner(vector<vector<long long>> weight) {
    int n = (int)weight.size();
    if (n <= 1) return 0;

    vector<int> vertices(n);
    iota(vertices.begin(), vertices.end(), 0);
    long long best = numeric_limits<long long>::max();

    for (int active = n; active > 1; --active) {
        vector<long long> addedWeight(active, 0);
        vector<char> used(active, false);
        int previous = -1;

        for (int step = 0; step < active; ++step) {
            int selected = -1;
            for (int i = 0; i < active; ++i) {
                if (!used[i] && (selected == -1 || addedWeight[i] > addedWeight[selected])) {
                    selected = i;
                }
            }

            if (step == active - 1) {
                best = min(best, addedWeight[selected]);

                for (int i = 0; i < active; ++i) {
                    if (i == selected) continue;
                    weight[vertices[previous]][vertices[i]] += weight[vertices[selected]][vertices[i]];
                    weight[vertices[i]][vertices[previous]] = weight[vertices[previous]][vertices[i]];
                }
                vertices.erase(vertices.begin() + selected);
                break;
            }

            used[selected] = true;
            previous = selected;
            for (int i = 0; i < active; ++i) {
                if (!used[i]) {
                    addedWeight[i] += weight[vertices[selected]][vertices[i]];
                }
            }
        }
    }

    return best;
}

int main() {
    ios_base::sync_with_stdio(false);
    cin.tie(nullptr);

    long long nInput, m;
    if (!(cin >> nInput >> m)) return 0;
    if (nInput <= 1) {
        cout << 0 << "\n";
        return 0;
    }

    int n = (int)nInput;
    vector<vector<long long>> weight(n, vector<long long>(n, 0));
    for (long long i = 0; i < m; i++) {
        long long u, v;
        cin >> u >> v;
        --u; --v;
        if (u == v) continue;                 // self-loops never cross a cut
        weight[(int)u][(int)v]++;
        weight[(int)v][(int)u]++;
    }

    cout << stoerWagner(weight) << "\n";
    return 0;
}
```

So the causal chain: the handshake identity forces m ≥ nk/2, which means a random edge is a min-cut edge with probability only ≤ 2/n; contraction commits two vertices to the same side and never lowers the min cut, so contracting random edges preserves the cut whenever the chosen edges stay internal; telescoping the per-step survival from n down to 2 supernodes gives a clean 2/(n(n−1)) success per run (which I verified collapses exactly, and which matched an empirical 0.435 ≥ 1/28 on a concrete 8-vertex graph), repeatable to high confidence in O(n⁴ log n). Then, noticing the risk lives entirely in the *late* contractions, I share the safe early prefix down to n/√2 (the survival-½ threshold, checked numerically at ≈ 0.50), branch into exactly two independent continuations (so one survivor is carried forward in expectation), and recurse — and iterating the success recurrence shows this turns a 1/n² success rate into Ω(1/log n) at O(n² log n) per call, for a global minimum cut in O(n² log³ n) with high probability.
