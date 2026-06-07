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
is `≤` the minimum tour weight. There's my lower bound. But it's loose: a tree is free to dump
many edges onto a few hub vertices, nothing pushes it toward looking like a cycle.

Let me not throw away the tour's structure quite so violently. A tour has `n` edges; a spanning
tree has `n−1`. The difference is exactly one edge. So instead of a plain tree, take a spanning
tree on the vertices `{2, 3, …, n}` and then attach vertex `1` back with its two cheapest edges.
Call that a 1-tree: a tree on `{2,…,n}`, plus two distinct edges at vertex 1. It has `n` edges
like a tour, it's still trivially cheap (one MST on `n−1` vertices, plus pick the two smallest
edges at vertex 1), and here's the clean part — a tour is *precisely* a 1-tree in which every
vertex has degree 2. So every tour is a 1-tree, which means the minimum-weight 1-tree costs no
more than the minimum tour: `min-1-tree ≤ C*`. And a bonus I should keep in my pocket: if the
minimum 1-tree happens to come out with all degrees 2, it *is* a tour, and then it's the optimum
tour. Nice. But generically it won't — some vertex will have degree 3 or more and some will hang
off with degree 1. The bound is still loose. I need a way to *tighten* it without actually
solving the TSP.

Suppose I add a number `π_i` to "every edge touching vertex `i`."
Concretely, replace `c_ij` by `c_ij + π_i + π_j`. What does that do to a *tour*? In a tour every
vertex has degree exactly 2, so the total added cost is `Σ_i π_i · (degree of i) = Σ_i 2π_i = 2Σπ_i`
— the *same constant* for every tour. So the perturbation shifts every tour's cost by the same
`2Σπ_i`; the argmin tour doesn't move; the TSP is invariant under `c_ij → c_ij + π_i + π_j`. But
what does it do to a *1-tree*? A 1-tree's vertices do *not* all have degree 2, so the added cost
is `Σ_i π_i · d_i` with the `d_i` not all equal to 2 — the perturbation changes which 1-tree is
minimal, and by how much. So I have a free parameter vector `π` that leaves the thing I care
about (the optimal tour) fixed while reshaping the cheap object I'm using to bound it. That's a
lever.

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
time maximizing the bound and none solving TSPs. This is the wall: insisting that the function
*increase* every step is exactly what makes me crawl, because at the kinks I want to reach, no
small step increases `w`.

So let me give up on "increase `w` each step." What weaker guarantee could I settle for that still
gets me to the maximizer? The maximizer `π*` is a point. What if, instead of demanding the
function value go up, I demand the *distance to `π*` go down*? Monotone progress in distance, not
in value. That's a completely different, and weaker, request.

Let me see if the degree-residual direction actually buys me that. Take the active 1-tree at `π`,
call its degree-residual vector `v_{k(π)}`. For any other point `τ`, I'll compare `w(τ)` to
`w(π)`. By definition `w(π) = c_{k(π)} + π · v_{k(π)}` (the active tree achieves the min at `π`),
and `w(τ) = min_k [c_k + τ·v_k] ≤ c_{k(π)} + τ · v_{k(π)}` (the active-at-`π` tree is one feasible
choice at `τ`, so the true min is no larger). Subtract the first from this inequality:

  `w(τ) − w(π) ≤ (τ − π) · v_{k(π)}`.

So `v_{k(π)}` is a **subgradient** of the concave function `w` at `π`: it over-estimates the
increase along every direction. In particular, plug in `τ = π*`, the maximizer:
`(π* − π) · v_{k(π)} ≥ w(π*) − w(π) ≥ 0`. The degree-residual vector makes an acute angle with
the direction from my current `π` toward the maximizer. It *points roughly at* `π*`, even though
it is not an ascent direction of `w` in the differentiable sense. That's exactly the handle I
need: step along `v_{k(π)}` and, for a small enough step, I move closer to `π*` — whether or not
`w` went up.

Let me nail the step size. Iterate `π_{m+1} = π_m + t · v_{k(π_m)}`. Measure squared distance to
the maximizer `π*`:

  `‖π* − (π + t v)‖² = ‖π* − π‖² − 2t (π* − π)·v + t² ‖v‖²`,

writing `v = v_{k(π)}`. I want this `< ‖π* − π‖²`, i.e. `t² ‖v‖² − 2t (π*−π)·v < 0`, i.e.
`0 < t < 2 (π*−π)·v / ‖v‖²`. And I just showed `(π*−π)·v ≥ w(π*) − w(π)`, so it's certainly
enough to take

  `0 < t < 2 (w(π*) − w(π)) / ‖v‖²`.

For any `t` in that range, `π_{m+1}` is strictly closer to `π*` than `π_m` was — and this holds
for *every* maximizer simultaneously. The hyperplane through `π` with normal `v` cuts off a
closed half-space containing all points with `w(·) ≥ w(π)`, in particular every maximizer, and
the step moves into that half-space along the normal. I don't need `w` to increase; I need only
that I'm Fejér-monotone — non-increasing distance to the set of maximizers — and the iteration
delivers that.

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

That makes a fixed step less reckless than it first looks. As the iteration proceeds, the 1-trees
produced start to look like tours: a great many vertices land on degree 2, so most components of
`v = d − 2` are zero and `‖v‖²` is a small integer. The penalty term shrinks the over-degree
vertices' edges and fattens the starved ones until almost everybody has degree 2. So
`limsup‖v‖²` is small, the slack `½t‖v‖²` is small, and a constant step can land the bound very
close to `max w`.

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

is therefore a practical overestimate rule, not the same theorem. The repair is to force
`λ → 0`: start with `λ = 2`, run a block of iterations, halve `λ`, shorten the block, and stop
when the steps are too small to matter.

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
that child, since a tour using two edges at a vertex uses no others there. Because the bound is so
tight, the trees stay tiny.

Let me write the bound computation, then drop it into the search. The 1-tree is an MST on the
`n−1` "ordinary" vertices under the perturbed costs, plus the two cheapest perturbed edges from
the left-out special vertex; the bound accumulates *raw* edge costs and adds `Σ_i π_i (d_i − 2)`;
the ascent updates `π += t · (d − 2)`.

```python
import math
import numpy as np

def _prim_mst(weighed):
    """Minimum spanning tree on a dense perturbed-cost matrix."""
    k = weighed.shape[0]
    in_tree = np.zeros(k, dtype=bool)
    best = weighed[0].copy()
    parent = np.zeros(k, dtype=int)
    in_tree[0] = True
    best[0] = np.inf
    edges = []
    for _ in range(k - 1):
        v = int(np.argmin(np.where(in_tree, np.inf, best)))
        edges.append((parent[v], v))
        in_tree[v] = True
        upd = (~in_tree) & (weighed[v] < best)
        best[upd] = weighed[v][upd]
        parent[upd] = v
    return edges

def compute_one_tree(cost, pi):
    """Minimum 1-tree under node potentials pi: raw cost plus degrees."""
    n = cost.shape[0]
    extra = n - 1
    weighed = cost + pi[:, None] + pi[None, :]
    sub_edges = _prim_mst(weighed[:extra, :extra])
    degrees = np.zeros(n, dtype=int)
    one_tree_cost = 0.0
    for u, v in sub_edges:
        degrees[u] += 1
        degrees[v] += 1
        one_tree_cost += cost[u, v]
    order = np.argsort(weighed[extra, :extra])
    for v in (int(order[0]), int(order[1])):
        degrees[extra] += 1
        degrees[v] += 1
        one_tree_cost += cost[extra, v]
    return one_tree_cost, degrees

class VolgenantJonker:
    """Vanishing schedule used as the default evaluator."""
    def __init__(self, n, max_iterations=0):
        self.n = n
        self.M = max_iterations if max_iterations > 0 else int(28 * n ** 0.62)
        self.step1 = 0.0
        self.m = 0
        self._init = False

    def cont(self):
        self.m += 1
        return self.m <= self.M

    def step(self):
        m, M = self.m, self.M
        return ((m - 1) * (2 * M - 5) / (2 * (M - 1)) * self.step1
                - (m - 2) * self.step1
                + 0.5 * (m - 1) * (m - 2) / ((M - 1) * (M - 2)) * self.step1)

    def on_one_tree(self, one_tree_cost):
        if not self._init:
            self._init = True
            self.step1 = one_tree_cost / (2 * self.n)

    def on_new_wmax(self, one_tree_cost):
        self.step1 = one_tree_cost / (2 * self.n)

class HeldWolfeCrowder:
    """Upper-bound Polyak-style evaluator with lambda halving."""
    def __init__(self, n, upper_bound):
        self.n = n
        self.UB = upper_bound
        self.num_iter = 2 * n
        self.lam = 2.0
        self.it = 0
        self._step = 0.0

    def cont(self):
        if self.it >= self.num_iter:
            self.num_iter //= 2
            if self.num_iter < 2:
                return False
            self.it = 0
            self.lam /= 2
        else:
            self.it += 1
        return True

    def step(self):
        return self._step

    def on_one_tree(self, one_tree_cost, w, degrees):
        norm = float(np.sum((degrees - 2) ** 2))
        self._step = self.lam * (self.UB - w) / norm if norm > 0 else 0.0

    def on_new_wmax(self, one_tree_cost):
        pass

def held_karp_lower_bound(cost, algorithm="VJ", upper_bound=None, max_iterations=0):
    """Return the 1-tree Lagrangian lower bound."""
    cost = np.asarray(cost, dtype=float)
    n = cost.shape[0]
    if n < 2:
        return 0.0
    if n == 2:
        return cost[0, 1] + cost[1, 0]

    if algorithm == "HWC":
        if upper_bound is None:
            raise ValueError("HWC needs an upper_bound on OPT")
        alg = HeldWolfeCrowder(n, upper_bound)
    else:
        alg = VolgenantJonker(n, max_iterations)

    pi = np.zeros(n)
    best_pi = pi.copy()
    max_w = -math.inf
    w = 0.0
    while alg.cont():
        one_tree_cost, degrees = compute_one_tree(cost, pi)
        if isinstance(alg, HeldWolfeCrowder):
            alg.on_one_tree(one_tree_cost, w, degrees)
        else:
            alg.on_one_tree(one_tree_cost)
        g = degrees - 2
        w = one_tree_cost + float(np.dot(pi, g))
        if w > max_w:
            max_w = w
            best_pi = pi.copy()
            alg.on_new_wmax(one_tree_cost)
        pi = pi + alg.step() * g

    one_tree_cost, degrees = compute_one_tree(cost, best_pi)
    return one_tree_cost + float(np.dot(best_pi, degrees - 2))
```

So the chain is: branch-and-bound needs a cheap, tight under-estimate of the optimum tour; a tour
is a 1-tree with every degree 2, and the minimum 1-tree is MST-cheap, giving a first loose lower
bound; perturbing edge costs by per-vertex potentials `π` shifts every tour by the same constant
(so the TSP is invariant) while reshaping the 1-tree, yielding the family `w(π) = min_k[c_k +
π·(d_k − 2)] ≤ C*` — the Lagrangian dual of the degree-2 constraints, concave and piecewise
linear; maximizing it by demanding the function increase is too slow at the kinks, so instead I
demand decreasing distance to high-level sets, which the degree-residual vector `d − 2` delivers as
a subgradient (the maximum-residual relaxation method of Agmon-Motzkin-Schoenberg); the constant
step has the bound `sup_m w(π_m) ≥ max w − (t/2)limsup_m‖d_m−2‖²`, the target step
`t = λ(ℓ − w)/‖d−2‖²` follows when `ℓ < max w`, and the practical code uses either a
vanishing Volgenant-Jonker schedule or an upper-bound HWC schedule with halved `λ`; the whole thing
slots into branch-and-bound, forcing/forbidding edges and branching when the ascent stalls.
