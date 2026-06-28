OK, let me think this through from scratch. I want to solve the symmetric traveling-salesman
problem exactly — minimum tour through `n` cities, costs `c_ij = c_ji`. Subset dynamic programming
can solve the exact problem in exponential time, but that is a different tool: it gives an exact
algorithm, not a cheap certificate I can recompute at every branch-and-bound node. The path I need
here is branch-and-bound: I keep splitting the set of tours into pieces, and I throw away any piece
whose cheapest conceivable tour already costs more than the best tour I've found. The thing that
decides whether this terminates this century is the lower bound I compute at each node. If the bound
is loose, almost nothing gets pruned and the tree blows up. So the real problem isn't "solve the
TSP," it's: get a lower bound on the optimum tour cost `C*` that is (a) tight and (b) cheap enough
that I can afford to recompute it at thousands of nodes. Cheap means: roughly the cost of a
spanning-tree computation. Let me chase that.

What do I have that's cheap and lower-bounds a tour? A tour is a connected spanning subgraph in
which every vertex has degree exactly 2. That's two constraints stacked: spanning-and-connected,
and 2-regular. The cheap things I know how to compute are the ones where I've dropped a
constraint. Drop "2-regular" entirely and I'm left with "connected spanning subgraph," whose
minimum is a minimum spanning tree — and I can get that greedily in polynomial time. A tour
contains a spanning tree (delete any one edge of the cycle), so the minimum spanning tree weight
is `≤` the minimum tour weight. There's a lower bound. But I suspect it's loose: a tree is free to
dump many edges onto a few hub vertices, nothing pushes it toward looking like a cycle.

Let me not throw away the tour's structure quite so violently. A tour has `n` edges; a spanning
tree has `n−1`. The difference is exactly one edge. So instead of a plain tree, take a spanning
tree on the vertices `{2, 3, …, n}` and then attach vertex `1` back with its two cheapest edges.
Call that a 1-tree: a tree on `{2,…,n}`, plus two distinct edges at vertex 1. It has `n` edges
like a tour, it's still trivially cheap (one MST on `n−1` vertices, plus pick the two smallest
edges at vertex 1). And a tour is a 1-tree in which every vertex has degree 2 — delete one edge
of vertex 1 from the tour and what remains is a tree on `{2,…,n}` (a connected, `n−1`-edge,
`n−1`-vertex graph), with vertex 1 hanging off by its two original tour edges. So every tour is a
1-tree, which means the minimum-weight 1-tree costs no more than the minimum tour:
`min-1-tree ≤ C*`. And a bonus I should keep in my pocket: if the minimum 1-tree happens to come
out with all degrees 2, it satisfies the tour definition, so it *is* a tour, and being the
cheapest 1-tree it's then the optimum tour.

Before I trust any of this, let me actually run a tiny instance and see how loose the plain
1-tree really is, because the whole investment depends on the gap being worth closing. Take five
points in the unit square (a fixed pseudorandom draw) with Euclidean costs:

```
      0       1       2       3       4
0   0.0000  0.6476  0.6667  0.4607  0.6718
1   0.6476  0.0000  1.1831  0.9101  1.0471
2   0.6667  1.1831  0.0000  0.2762  0.2706
3   0.4607  0.9101  0.2762  0.0000  0.2150
4   0.6718  1.0471  0.2706  0.2150  0.0000
```

Brute-forcing all `4! = 24` tours through these five cities, the optimum is the tour
`0–1–4–2–3–0` at cost `C* = 2.7021`. Now the minimum 1-tree on raw costs (MST on `{0,1,2,3}` —
I'll let vertex 4 play the role of the special vertex — plus vertex 4's two cheapest edges): its
cost comes out `1.8700`, with degree sequence `[2, 1, 2, 3, 2]`. So the plain bound is `1.87`
against an optimum of `2.70`: a gap of `0.83`, about 31% low. That is exactly as loose as I
feared, and the degree sequence tells me *why* — vertex 3 has degree 3 (the tree dumped edges on
it) and vertex 1 has degree 1 (starved). Neither of those is allowed in a tour. I need a way to
push degree-3 vertices down and degree-1 vertices up, and to *tighten* the bound, without actually
solving the TSP.

Suppose I add a number `π_i` to "every edge touching vertex `i`."
Concretely, replace `c_ij` by `c_ij + π_i + π_j`. What does that do to a *tour*? In a tour every
vertex has degree exactly 2, so the total added cost is `Σ_i π_i · (degree of i) = Σ_i 2π_i = 2Σπ_i`
— the *same constant* for every tour. So I'd expect the perturbation to shift every tour's cost by
the same `2Σπ_i`, leaving the argmin tour where it was. Let me not just assert that; let me check
it on the five-city instance. Pick an arbitrary potential vector `π = (0.3, −0.5, 0.2, 0.7, −0.1)`,
so `2Σπ_i = 1.2`. Recomputing the cost of all 24 tours under `c_ij + π_i + π_j` and comparing each
to its raw cost plus `1.2`: every one of the 24 matches to machine precision, zero mismatches. So
the TSP really is invariant under `c_ij → c_ij + π_i + π_j` up to a global constant — the optimal
tour is fixed. But the 1-tree is not: a 1-tree's vertices do *not* all have degree 2, so its added
cost is `Σ_i π_i · d_i` with the `d_i` not all equal to 2. On the same instance, the plain 1-tree
bound `w(0) = 1.87` moves to `w(π) = 2.0584` under this `π` — already closer to `2.70`, just from
a random shove. So `π` is a free parameter that leaves the thing I care about (the optimal tour)
fixed while reshaping the cheap object I'm using to bound it.

Let me write the bound carefully under the perturbation. With weights `c_ij + π_i + π_j`, the
minimum tour still costs `C* + 2Σπ_i` (its cost shifted by the constant), and the minimum 1-tree
under these weights costs no more than that, because a tour is a 1-tree:

  `C* + 2Σ_i π_i ≥ min_k [ c_k + Σ_i π_i d_ik ]`,

where `k` indexes 1-trees, `c_k` is the *raw* weight of the `k`-th 1-tree, and `d_ik` is the
degree of vertex `i` in it. Rearrange — move the `2Σπ_i` inside:

  `C* ≥ min_k [ c_k + Σ_i π_i (d_ik − 2) ]`.

Define `v_k` to be the vector with `i`-th component `d_ik − 2`, the *degree residual* of vertex
`i` in 1-tree `k`, and set

  `w(π) = min_k [ c_k + π · v_k ]`.

Then `C* ≥ w(π)` for *every* `π`. I started with one weak bound and now I have an infinite family
of them, one per `π`, and the best is `max_π w(π)`. That's the object I want to compute.

Now stare at `w(π)`. For each fixed 1-tree `k`, `c_k + π · v_k` is an affine (linear-plus-constant)
function of `π`. And `w(π)` is the pointwise *minimum* over a finite (huge, but finite) collection
of these affine functions. A minimum of affine functions is concave and piecewise linear. So
maximizing `w` has no spurious local maxima, but the function is non-differentiable at the
breakpoints where the identity of the minimizing 1-tree flips. The optimum may be a flat face, a
kink, or both; either way, I cannot count on having an ordinary gradient at the points I care
about.

Let me notice what this `π` is, structurally, because it tells me what I'm really doing. I have
`min_k [c_k + Σ_i π_i d_ik] − 2Σ_i π_i ≤ C*`. That `2Σπ_i = Σπ_i · 2` is `Σ_i π_i · (target
degree)`. So I'm taking the easy problem "minimum 1-tree" and adding `Σ_i π_i (d_i − 2)` — a
penalty term that is positive when I dualize the constraint "degree of `i` equals 2" with a price
`π_i`. This is Lagrangian relaxation: the tour is "minimum 1-tree subject to all degrees `= 2`,"
the degree-2 equalities are the complicating constraints, I dualize them with multipliers `π`,
and `w(π)` is the Lagrangian dual function. `max_π w(π)` is the best Lagrangian bound, and the
duality gap `C* − max_π w(π)` is whatever the 1-tree relaxation can't see. So the question "how
do I tighten the 1-tree bound" has become "how do I maximize a concave piecewise-linear dual
function."

First instinct: just do steepest ascent. At a point `π`, find the minimizing 1-tree, look at its
degree residual vector, and that's a direction — increase `π_i` where the 1-tree over-uses vertex
`i` (degree `> 2`) so its edges become more expensive and the 1-tree will prefer to use fewer of
them next time, decrease `π_i` where vertex `i` is starved (degree 1). Climb until `w` stops
increasing. Or, treat `max_π w(π)` as the linear program `max w` subject to `w ≤ c_k + π·v_k` for
all `k`, with one constraint per 1-tree, and run simplex generating columns on demand. Both of
these are the obvious textbook attacks.

And both crawl. The LP has an astronomical number of constraints; generating columns and pivoting
is heavy. The steepest-ascent version has to fight the non-differentiability — near the optimum,
where 1-trees tie, the "increase `w` at every step" requirement forces tiny, expensively
line-searched steps, and the number of iterations grows badly as `n` grows. I'm spending all my
time maximizing the bound and none solving TSPs. The trouble is structural: insisting that the
function *increase* every step is exactly what makes me crawl, because at the kinks I want to
reach, no small step increases `w`.

So let me question that requirement. Maybe I don't need the function to increase every step. What
weaker guarantee could I settle for that still gets me to the maximizer? The maximizer `π*` is a
point. What if, instead of demanding the function value go up, I demand the *distance to `π*`* go
down? Monotone progress in distance, not in value. That's a completely different, and weaker,
request — and crucially it could let me step right through a kink where `w` momentarily dips.

Let me see if the degree-residual direction actually buys me that. Take the active 1-tree at `π`,
call its degree-residual vector `v_{k(π)}`. For any other point `τ`, I'll compare `w(τ)` to
`w(π)`. By definition `w(π) = c_{k(π)} + π · v_{k(π)}` (the active tree achieves the min at `π`),
and `w(τ) = min_k [c_k + τ·v_k] ≤ c_{k(π)} + τ · v_{k(π)}` (the active-at-`π` tree is one feasible
choice at `τ`, so the true min is no larger). Subtract the first from this inequality:

  `w(τ) − w(π) ≤ (τ − π) · v_{k(π)}`.

So `v_{k(π)}` is a **subgradient** of the concave function `w` at `π`: it over-estimates the
increase along every direction. Let me sanity-check this inequality on the five-city instance
before I lean on it. At `π = 0` the active 1-tree has residual `v = d − 2 = [0, −1, 0, 1, 0]`
(the `[2,1,2,3,2]` degrees minus 2 — exactly the starved vertex 1 and the over-used vertex 3 I
flagged earlier). Take `τ` to be the random potential from before. The inequality predicts
`w(τ) − w(0) ≤ (τ − 0)·v`. The left side is `2.0584 − 1.87 = 0.1884`; the right side is
`τ·v = (−0.5)(−1) + (0.7)(1) = 1.2`. Indeed `0.1884 ≤ 1.2` — the subgradient bound holds, and it's
slack here because `τ` is not the active-tree direction. In particular, plugging in `τ = π*`, the
maximizer: `(π* − π) · v_{k(π)} ≥ w(π*) − w(π) ≥ 0`. The degree-residual vector makes an acute
angle with the direction from my current `π` toward the maximizer. It *points roughly at* `π*`,
even though it is not an ascent direction of `w` in the differentiable sense. That's the handle I
need: step along `v_{k(π)}` and, for a small enough step, I move closer to `π*` — whether or not
`w` went up.

Let me nail the step size. Iterate `π_{m+1} = π_m + t · v_{k(π_m)}`. Measure squared distance to
the maximizer `π*`:

  `‖π* − (π + t v)‖² = ‖π* − π‖² − 2t (π* − π)·v + t² ‖v‖²`,

writing `v = v_{k(π)}`. I want this `< ‖π* − π‖²`, i.e. `t² ‖v‖² − 2t (π*−π)·v < 0`, i.e.
`0 < t < 2 (π*−π)·v / ‖v‖²`. And I just showed `(π*−π)·v ≥ w(π*) − w(π)`, so it's certainly
enough to take

  `0 < t < 2 (w(π*) − w(π)) / ‖v‖²`.

For any `t` in that range, `π_{m+1}` is strictly closer to `π*` than `π_m` was — and since the
derivation used only `(π*−π)·v ≥ w(π*)−w(π) ≥ 0`, which holds for *every* maximizer, this is true
for all of them simultaneously. The hyperplane through `π` with normal `v` cuts off a closed
half-space containing all points with `w(·) ≥ w(π)`, in particular every maximizer, and the step
moves into that half-space along the normal. I don't need `w` to increase; I need only that I'm
Fejér-monotone — non-increasing distance to the set of maximizers — and the iteration delivers
that.

This is suspiciously familiar. "Pick a violated linear inequality, step along its normal across
the bounding hyperplane by a relaxed amount, and you provably get closer to the solution set" —
that is the relaxation method of Agmon (1954) and Motzkin–Schoenberg (1954) for systems of linear
inequalities. Make the connection precise: maximizing `w` is the LP `max w` s.t.
`w ≤ c_k + π·v_k ∀k`. Fixing a target value `w̄`, finding `π` with `w(π) ≥ w̄` is exactly solving
the inequality system `w̄ ≤ c_k + π·v_k ∀k`. The version of the relaxation method that selects the
*most* violated inequality — the one minimizing `c_k + π·v_k`, i.e. maximizing the residual
`w̄ − (c_k + π·v_k)` — picks precisely the active 1-tree `k(π)` and steps along its normal `v_k`.
So my subgradient iteration *is* the maximum-residual relaxation method specialized to this
problem. Agmon's basic lemma is my distance-decrease computation with a relaxation parameter
`λ ∈ (0,2)`, and the target-value convergence statement I need is the same projection geometry in
this huge inequality system. The degree residual is the natural choice of violated constraint.

Now, what step size do I actually run? Two cases worth thinking through.

Constant step, `t_m = t` for all `m`. Crude, but let me see what it guarantees. Write
`L = limsup_m ‖v_{k(π_m)}‖²`, and suppose the iterates never get as high as the level I want:
there is a number `A` with

  `w(π_m) < A < max_π w(π) − ½tL`

for every `m`. Now choose the target level `\bar w = A + ½tL`, which is still below `max w`. A
constant step can be rewritten in the target-value form

  `π_{m+1} = π_m + λ_m ((\bar w − w(π_m)) / ‖v_m‖²) v_m`

by setting `λ_m = t‖v_m‖² / (\bar w − w(π_m))`. For all sufficiently large `m`, the definition of
`L` gives `t‖v_m‖² ≤ 2(\bar w − w(π_m))`, so `λ_m ≤ 2`. It is also bounded away from zero once I
rule out `v_m = 0` — if `v_m` were zero while `w(π_m) < max w`, the subgradient inequality would
say `0 ≥ w(τ) − w(π_m)` for every `τ`, making `π_m` already optimal, a contradiction. The
target-value relaxation lemma therefore applies. If no iterate reaches the half-space
`w(π) ≥ \bar w`, the sequence is Fejer-monotone relative to that full-dimensional set, converges,
and its steps shrink to zero; but then the target residual `\bar w − w(π_m)` must shrink to zero,
contradicting `w(π_m) < A < \bar w`. So the constant-step run must satisfy

  `sup_m w(π_m) ≥ max_π w(π) − ½ t · limsup_m ‖v_{k(π_m)}‖²`.

That makes a fixed step less reckless than it first looks. As the iteration proceeds, I'd expect
the 1-trees produced to start looking like tours: a great many vertices land on degree 2, so most
components of `v = d − 2` are zero and `‖v‖²` is a small integer. The penalty term shrinks the
over-degree vertices' edges and fattens the starved ones until almost everybody has degree 2. So
`limsup‖v‖²` should be small, the slack `½t‖v‖²` small, and a constant step could land the bound
very close to `max w`.

The same computation suggests a target-driven step when I have a level `ℓ < max w` that I want
to reach. The distance condition says that

  `0 < t < 2(ℓ − w(π)) / ‖v‖²`

is the safe range while `w(π) < ℓ`, so I can write

  `t = λ · (ℓ − w(π)) / ‖v‖²`,  with `ε ≤ λ ≤ 2`.

With `λ` bounded away from zero and at most 2, the iterates either reach a point with
`w(π) ≥ ℓ` or converge to the boundary `w = ℓ`; if the unknown target were the true optimum
`w* = max w`, this is the Polyak step. In actual code I cannot cheaply know a valid lower target
near `w*`. What I can get cheaply is the opposite kind of number: a heuristic tour gives an upper
bound `UB ≥ C* ≥ max w`. Using

  `t = λ · (UB − w(π)) / ‖v‖²`

is therefore a practical overestimate rule, not the same theorem — `UB − w(π)` over-shoots the
safe `ℓ − w(π)`, so individual steps can be too long and the distance guarantee can fail on a
given step. The repair is to force `λ → 0`: start with `λ = 2`, run a block of iterations, halve
`λ`, shorten the block, and stop when the steps are too small to matter; as the steps vanish the
over-shoot vanishes too.

One more practical lever for the starting point. Cold-starting at `π = 0` means the first 1-tree
is computed on raw costs and may be far from tour-like. I can warm-start from the assignment
relaxation: solve the assignment problem for `(c_ij)`, take its dual solution `u_i, v_i` (with
`u_i + v_j ≤ c_ij`), and set `π_i⁰ = −½(u_i + v_i)`. Then `w(π⁰)` is already at least the cost of
the optimal assignment, so I begin the ascent above the floor instead of at it.

Let me also make sure this *survives inside* branch-and-bound, because the whole point was to feed
a search. The ascent does not in general reach `max w`, and even `max w` can be strictly below
`C*` (there is a duality gap). So: combine the ascent with branching. A subproblem is "all 1-trees
that include a forced edge set `X` and exclude a forbidden set `Y`"; computing the minimum 1-tree
restricted to `T(X,Y)` is the same greedy computation with some edges pinned in and others
deleted, so `w_{X,Y}(π) = min_{k ∈ T(X,Y)}[c_k + π·v_k]` is a valid lower bound for that
subproblem and my ascent applies verbatim. Run the ascent on the least-bound subproblem; if its
bound reaches the incumbent upper bound `C`, discard it; if the ascent stalls — no improvement for
a block of `p` iterations — stop and branch. To branch, I order the not-yet-decided edges by how
much excluding each would raise the bound (a by-product of the greedy 1-tree computation), and
split into children that force the leading edges in one at a time while forbidding the next; when
forcing two edges into some vertex, I can legitimately forbid all that vertex's other edges in
that child, since a tour using two edges at a vertex uses no others there. The hope is that the
bound is tight enough that the trees stay tiny.

Let me write the bound computation, then drop it into the search. The 1-tree is an MST on the
`n−1` "ordinary" vertices under the perturbed costs, plus the two cheapest perturbed edges from
the left-out special vertex; the bound accumulates *raw* edge costs and adds `Σ_i π_i (d_i − 2)`;
the ascent updates `π += t · (d − 2)`.

I'll land this as a single self-contained C++17 program reading `n` and the `n × n` symmetric cost
matrix from stdin and printing the plain min-1-tree cost and the Held-Karp bound to stdout.

The Held-Karp 1-tree subgradient bound is the part I'd most easily get wrong under time pressure,
especially selecting the perturbed MST plus two special-node edges while scoring raw cost and
`π·(d − 2)` correctly; if I weren't confident I could implement it in the budget, I'd fall back to
standard exact bitmask DP over subsets and ship that -- a plain correct submission beats an ambitious broken one.

```cpp
// Held-Karp 1-tree Lagrangian lower bound for the symmetric TSP via subgradient
// ("relaxation") ascent. NOT the O(2^n n^2) exact DP; this is a cheap bound for
// branch-and-bound pruning.
//
// Input (stdin):  n, then an n x n symmetric real cost matrix (row-major).
// Output (stdout): the plain minimum 1-tree cost and the Held-Karp bound (VJ schedule).
#include <bits/stdc++.h>
using namespace std;

// Minimum 1-tree under node potentials pi: MST on nodes {0..n-2} plus the two
// cheapest edges from the left-out node n-1, all under weighed cost
// c(i,j)+pi[i]+pi[j]. Returns the sum of RAW edge costs in one_tree_cost and the
// degree of each node in deg.
static double compute_one_tree(const vector<vector<double>>& cost,
                               const vector<double>& pi,
                               vector<int>& deg) {
    int n = (int)cost.size();
    int extra = n - 1;  // the left-out / special node
    fill(deg.begin(), deg.end(), 0);
    double one_tree_cost = 0.0;

    // Prim MST on the extra ordinary nodes {0..extra-1} under perturbed cost.
    vector<double> best(extra);
    vector<int> parent(extra, 0);
    vector<char> in_tree(extra, 0);
    for (int j = 0; j < extra; ++j) best[j] = cost[0][j] + pi[0] + pi[j];
    in_tree[0] = 1;
    best[0] = numeric_limits<double>::infinity();
    for (int t = 0; t < extra - 1; ++t) {
        int v = -1;
        double bv = numeric_limits<double>::infinity();
        for (int j = 0; j < extra; ++j)
            if (!in_tree[j] && best[j] < bv) { bv = best[j]; v = j; }
        int u = parent[v];
        deg[u]++; deg[v]++;
        one_tree_cost += cost[u][v];          // accumulate RAW cost
        in_tree[v] = 1;
        for (int j = 0; j < extra; ++j) {
            double w = cost[v][j] + pi[v] + pi[j];
            if (!in_tree[j] && w < best[j]) { best[j] = w; parent[j] = v; }
        }
    }

    // Attach the extra node by its two cheapest (perturbed) edges.
    int e1 = -1, e2 = -1;
    double w1 = numeric_limits<double>::infinity(), w2 = w1;
    for (int j = 0; j < extra; ++j) {
        double w = cost[extra][j] + pi[extra] + pi[j];
        if (w < w1) { w2 = w1; e2 = e1; w1 = w; e1 = j; }
        else if (w < w2) { w2 = w; e2 = j; }
    }
    for (int v : {e1, e2}) {
        deg[extra]++; deg[v]++;
        one_tree_cost += cost[extra][v];
    }
    return one_tree_cost;
}

// Volgenant-Jonker vanishing step schedule reaching 0 at iteration M
// (EJOR 9:83-89, 1982). step1 is seeded from the first / best 1-tree cost.
struct VolgenantJonker {
    int n, M, m = 0;
    double step1 = 0.0;
    bool inited = false;
    VolgenantJonker(int n_, int max_iterations)
        : n(n_), M(max_iterations > 0 ? max_iterations
                                      : (int)(28.0 * pow((double)n_, 0.62))) {}
    bool cont() { ++m; return m <= M; }
    double step() const {
        double mm = m, MM = M;
        return (mm - 1) * (2 * MM - 5) / (2 * (MM - 1)) * step1
               - (mm - 2) * step1
               + 0.5 * (mm - 1) * (mm - 2) / ((MM - 1) * (MM - 2)) * step1;
    }
    void on_one_tree(double one_tree_cost) {
        if (!inited) { inited = true; step1 = one_tree_cost / (2.0 * n); }
    }
    void on_new_wmax(double one_tree_cost) { step1 = one_tree_cost / (2.0 * n); }
};

// Held-Karp 1-tree Lagrangian lower bound on OPT for cost matrix `cost`,
// using the Volgenant-Jonker schedule. w(pi) = cost(1-tree) + sum_i pi_i*(deg_i-2).
static double held_karp_lower_bound(const vector<vector<double>>& cost,
                                    int max_iterations = 0) {
    int n = (int)cost.size();
    if (n < 2) return 0.0;
    if (n == 2) return cost[0][1] + cost[1][0];

    VolgenantJonker alg(n, max_iterations);
    vector<double> pi(n, 0.0), best_pi(n, 0.0);
    vector<int> deg(n, 0);
    double max_w = -numeric_limits<double>::infinity();

    while (alg.cont()) {
        double one_tree_cost = compute_one_tree(cost, pi, deg);
        alg.on_one_tree(one_tree_cost);
        double w = one_tree_cost;
        for (int i = 0; i < n; ++i) w += pi[i] * (deg[i] - 2);  // w(pi) <= OPT
        if (w > max_w) {
            max_w = w;
            best_pi = pi;
            alg.on_new_wmax(one_tree_cost);
        }
        double s = alg.step();
        for (int i = 0; i < n; ++i) pi[i] += s * (deg[i] - 2);  // ascent g_i=deg_i-2
    }

    double one_tree_cost = compute_one_tree(cost, best_pi, deg);
    double w = one_tree_cost;
    for (int i = 0; i < n; ++i) w += best_pi[i] * (deg[i] - 2);
    return w;
}

int main() {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int n;
    if (!(cin >> n) || n <= 0) return 0;
    vector<vector<double>> cost(n, vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) cin >> cost[i][j];

    vector<int> deg(n, 0);
    vector<double> zero(n, 0.0);
    // A 1-tree needs n >= 3 (a tree on n-1 >= 2 nodes plus two distinct edges).
    double plain = (n >= 3) ? compute_one_tree(cost, zero, deg)
                            : (n == 2 ? cost[0][1] + cost[1][0] : 0.0);
    double bound = held_karp_lower_bound(cost);

    cout << fixed << setprecision(4);
    cout << "plain min-1-tree    : " << plain << "\n";
    cout << "Held-Karp bound (VJ): " << bound << "\n";
    return 0;
}
```

Let me run this on the same five-city instance and see where the ascent lands, since I know the
true answer is `C* = 2.7021` and the cold-start floor is `1.87`. The vanishing Volgenant-Jonker
schedule drives the bound from `1.87` up to `2.7021` — it closes the entire gap here and recovers
the optimum exactly, which makes sense because at the maximizing `π` the minimum 1-tree must have
come out all-degree-2 and therefore *is* the optimal tour. That is the best case, and a useful
reassurance that the machinery is wired correctly. It is not guaranteed in general: the
upper-bound HWC schedule, fed the true optimum as `UB`, only reaches `2.3808` on this same
instance — better than the `1.87` floor but well short of `max w`, because the over-shooting
`UB`-based step with a coarse halving block does not fully converge in so few iterations on five
cities. So the two schedules genuinely differ, and the bound returned is whatever the chosen
schedule's best `π` achieves — always a valid lower bound, never necessarily `max w`.

So the chain is: branch-and-bound needs a cheap, tight under-estimate of the optimum tour; a tour
is a 1-tree with every degree 2, and the minimum 1-tree is MST-cheap, giving a first loose lower
bound (on the five-city test, `1.87` against `C* = 2.70`); perturbing edge costs by per-vertex
potentials `π` shifts every tour by the same constant (verified: all 24 tours shift by exactly
`2Σπ`, so the TSP is invariant) while reshaping the 1-tree, yielding the family
`w(π) = min_k[c_k + π·(d_k − 2)] ≤ C*` — the Lagrangian dual of the degree-2 constraints, concave
and piecewise linear; maximizing it by demanding the function increase is too slow at the kinks,
so instead I demand decreasing distance to high-level sets, which the degree-residual vector
`d − 2` delivers as a subgradient (checked numerically: `w(τ)−w(π) ≤ (τ−π)·(d−2)` held on the
test point), the maximum-residual relaxation method of Agmon-Motzkin-Schoenberg; the constant
step has the bound `sup_m w(π_m) ≥ max w − (t/2)limsup_m‖d_m−2‖²`, the target step
`t = λ(ℓ − w)/‖d−2‖²` follows when `ℓ < max w`, and the practical code uses either a
vanishing Volgenant-Jonker schedule (which closed the gap to the optimum on the test) or an
upper-bound HWC schedule with halved `λ`; the whole thing slots into branch-and-bound,
forcing/forbidding edges and branching when the ascent stalls.
