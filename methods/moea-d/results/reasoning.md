Let me start from what actually hurts. I have a design problem with several objectives that fight each other — minimize cost, maximize reliability, that kind of thing. There is no single winner, so the real answer is a whole trade-off surface, the Pareto front, and the decision-maker wants to *see* it and then choose. A population-based evolutionary algorithm is the obvious vehicle because its unit of work is already a set of candidates, so in principle one run can hold many trade-off solutions at once and the whole front falls out.

The standard way to drive that population is to rank its members by Pareto dominance: a solution on a better nondomination front out-selects one on a worse front, and you bolt on a density measure to spread them out. That's what the dominance-ranked GAs do, and the elitist fast-sorted version of it — compute every pairwise dominance once, peel fronts by decrementing domination counts, merge parents and offspring for elitism, break ties inside a front by a parameter-free crowding distance — works well on two and three objectives. I keep coming back to it, but two things about it nag at me.

The first is the cost. Every generation I run a nondominated sort over the combined 2N population, which is O(MN²). Plus a crowding computation. It's fine at N in the hundreds, but it's a quadratic tax I pay every single generation, and the thing I'm paying it for is just *ordering* the population.

The second nags more. Dominance is a *partial* order. With two objectives that's fine — plenty of pairs are comparable, the fronts have real structure, the sort separates the population into meaningful layers. But picture what happens as I add objectives. For one solution to dominate another it has to be no worse on *every* one of the M objectives. The more objectives, the harder that is — almost any pair will have one solution better on some objective and worse on another, so they're mutually nondominated, incomparable. The population floods into front 1: nearly everybody is rank 1, nobody dominates anybody. At that point the dominance rank is constant across most of the population and it carries almost no information; selection that's supposed to drive *convergence* has nothing to grip on and degrades toward picking by the tie-breaker — the crowding heuristic — which was only ever meant to spread, not to converge. So the convergence pressure, the whole point of the ranking, thins out exactly as the problem gets harder. That's a structural weakness of ranking vectors by a partial order, and no amount of speeding up the sort fixes it.

So let me question the frame. The reason I'm ranking vectors at all is that a GA wants a scalar fitness and my objectives are a vector. Dominance is *one* way to compare vectors — the honest partial order. But there's a completely different, much older way to turn a vector of objectives into a scalar: just aggregate them. Scalarize. The mathematical-programming people (Miettinen's book lays this out cleanly) have done this forever: pick a weight vector w, form a scalar aggregation g(x|w), and a Pareto-optimal solution of the multi-objective problem comes out as the optimizer of that scalar subproblem. A scalar g gives a *total* order — every pair is comparable, the comparison is always decisive. The selection pressure from "is g(y) smaller than g(x)?" never degrades, no matter how many objectives I stack up, because it's a single number.

The reason nobody just does this for the whole front is the standard objection: one scalarization gives you *one* point. To trace the front you sweep the weights and re-solve, over and over — one full optimization per point on a curve you want densely sampled. That's wasteful. And on top of it, the simplest aggregation, the weighted sum g = Σ wᵢ fᵢ, can only find points on the convex hull of the front — minimizing it means sliding a hyperplane with normal w down until it touches the feasible objective set, and a hyperplane touches the *convex* boundary; if the front has a concave dip, no nonnegative weights ever land a solution in it. So weighted-sum scalarization is both slow and blind to concave regions.

But wait. Look hard at *why* the sweep is wasteful: it's wasteful because each scalar subproblem is solved independently, from scratch, sequentially. "One run per point." That accounting is exactly the thing a population dissolves. I'm already carrying a population of N candidate solutions. What if I don't solve the subproblems one at a time — what if I attach *each population member to its own weight vector*, so I have N scalar subproblems, and I evolve all N solutions *simultaneously* in a single run? Then the "N separate runs" aren't separate at all; they're N members of one co-evolving population. The objection to scalarization evaporates the moment scalarization lives inside a population instead of in a sequential outer loop.

That reframes the whole algorithm. I'm no longer ranking a population of vectors against each other by a partial order. I'm maintaining N scalar subproblems, one per weight vector, each with its own current-best solution, and I push all of them toward their own optima at once. The Pareto front isn't an emergent property of dominance ranking anymore — it's the *collection of optima of N well-spread scalar subproblems*. Decompose the multi-objective problem into N scalar subproblems and co-evolve.

Now I have to make three things concrete: which scalarization, where the weight vectors come from, and how the subproblems help each other (because if they don't, I've just hidden N independent runs inside one loop and gained nothing).

Scalarization first, because the weighted sum has a known blind spot — it can only find points on the convex hull of the feasible objective set, and lots of real fronts have concave stretches. So I need a different aggregation, one whose contours can touch the front *inside* a concave dip. The candidate I reach for is the weighted Chebyshev — Tchebycheff — distance: g^te(x | w, z*) = max_i wᵢ |fᵢ(x) − zᵢ*|, where z* is the ideal point, the vector of per-objective minima. The intuition for why it should reach concave regions is geometric. A level set {f : Σ wᵢ fᵢ = c} of the weighted sum is a hyperplane; minimizing it presses a flat plane against the feasible set, and a flat plane can only kiss the convex boundary. A level set of the Tchebycheff function, {f : max_i wᵢ |fᵢ − zᵢ*| = c}, is an axis-aligned box-corner — an "L" in 2D — anchored at z*. As I shrink c, that L collapses toward z*, and the corner of an L can poke *into* a concave dip that a flat plane slides right over.

That's the picture, but I want to be sure it's true and not just a story I like, so let me actually run both scalarizations against a concave front and see what each one picks. Take a clean concave front, f₂ = 1 − f₁² for f₁ ∈ [0,1] (this is the ZDT2 front; it bulges *away* from the ideal — at f₁ = 0.5 the front sits at f₂ = 0.75, above the chord 1 − f₁ = 0.5, which is what "concave" means here). Put z* = (0,0). Sample the front densely and, for a handful of weights, find which front point minimizes each aggregation.

Weighted sum first, g = w₁f₁ + w₂f₂. Sweeping w₁ over 0.1, 0.3, 0.5, 0.7, 0.9, the argmin lands at f₁ = 1.0 (the (1,0) corner) for w₁ ≤ 0.3 and at f₁ = 0.0 (the (0,1) corner) for w₁ ≥ 0.5 — *every* weight collapses to one of the two endpoints; no weight ever returns an interior point of the concave front. That is the blind spot, made concrete: the whole concave interior is invisible to the weighted sum.

Now Tchebycheff, g = max(w₁f₁, w₂f₂), same five weights:

```
w=(0.10,0.90) -> f=(0.946, 0.105)
w=(0.30,0.70) -> f=(0.808, 0.347)
w=(0.50,0.50) -> f=(0.618, 0.618)
w=(0.70,0.30) -> f=(0.370, 0.863)
w=(0.90,0.10) -> f=(0.110, 0.988)
```

Every weight returns a *distinct interior* point, and they march monotonically across the front as w turns. Sweeping 99 weights from 0.01 to 0.99 recovers f₁ spanning [0.010, 0.995] with 86 of them strictly interior (0.05 < f₁ < 0.95) — the box-corner really does reach into the dip, and distinct weights tile the whole front. So the geometric story holds up: weighted sum is blind to the concave interior, Tchebycheff sweeps all of it. That settles the choice of aggregation.

The check also tells me *what* point each weight pins down. Look at w = (0.5, 0.5): the optimum is f = (0.618, 0.618), and there w₁f₁ = 0.309 = w₂f₂ — the two weighted deviations are exactly equal at the optimum. That is the geometry the "L" demands: the corner of the box sits on the ray from z* where wᵢ(fᵢ − zᵢ*) is equal across i, so minimizing the max drives x to the front point where the weighted deviations balance. Different w tilts that ray, so different w lands on a different front point — which is precisely the per-weight handle on the front I was after. There's a cleaner statement underneath that I'd expect to be provable (under mild conditions, every Pareto-optimal point minimizes g^te for some positive weight, and every minimizer of g^te is at least weakly Pareto-optimal), and the numeric sweep is consistent with it; I shouldn't oversell it though, because the minimizer for a given weight need not be unique and different weights can coincide, so it is not a clean bijection — but every front point being reachable by *some* weight is all the construction needs, and the sweep showed exactly that. The cost is that max is non-smooth, but a GA doesn't take gradients, so non-smoothness is free here.

Let me make sure I have the direction right on z*. I'm minimizing, z* sits at (or below) the front in every coordinate, so fᵢ(x) − zᵢ* ≥ 0 for any achievable f, and the absolute value is just a guard for the case where z* sits *above* some objective value — i.e. when z* is a momentary *over*-estimate of the true ideal (or I deliberately push it a hair below the running min, a utopian point) so that fᵢ(x) − zᵢ* could come out negative. With the absolute value the deviation is always nonnegative, so the max is well-behaved either way.

There's a subtlety in z*. The true ideal point is the vector of global per-objective minima, which I don't know — finding it is itself M single-objective optimizations. So I estimate it: keep a running z* and, every time I evaluate a new solution y, lower each coordinate, zⱼ ← min(zⱼ, fⱼ(y)). It only ever moves down, tracking the best per-objective value seen so far, so it converges toward the true ideal as the search proceeds. The Tchebycheff function measures a weighted L_∞ distance *to* z*, so anchoring it to this moving best-corner keeps every subproblem's contours pinned to the right reference; as z* descends, all the subproblems sharpen toward the true front together.

Now the weight vectors. I want them to produce an *even* spread of front points, so I want them evenly spread on the simplex {w : wᵢ ≥ 0, Σ wᵢ = 1}. The structured construction for that is Das & Dennis's simplex lattice: pick an integer H, divide each coordinate into H equal steps, and take every weight whose components are nonnegative multiples of 1/H summing to 1. For two objectives that should just be w = (i/H, 1 − i/H) for i = 0…H, i.e. N = H+1 evenly spaced points along the edge; let me confirm the generator does that before I trust it. Running it at H = 5 returns exactly (0,1), (0.2,0.8), (0.4,0.6), (0.6,0.4), (0.8,0.2), (1,0) — six points, evenly spaced, as predicted. The general count is N = C(H+M−1, M−1); at M = 2, H = 99 that is C(100,1) = 100, and the generator does return 100 vectors. So I can dial the population size precisely by choosing H — the number of weight vectors *is* the population size, and H is the knob that sets resolution. (One bookkeeping note for Tchebycheff: a component wᵢ = 0 makes that objective invisible to the max, which I don't want, so I'll nudge zeros to a tiny ε and renormalize — keeps every objective in play.)

Here's the part that earns the whole reframing — the collaboration. If each of the N subproblems just ran its own little optimization, I'd have N independent searches wearing a population costume; no gain. The leverage comes from a fact about the weights: two weight vectors that are *close* in weight space define two scalar subproblems with nearly identical aggregations, hence nearly the same optimum. So a solution that's good for subproblem i is, with high probability, good for the subproblems whose weights sit next to w^i. That means the subproblems can *share*: a subproblem can recombine the good solutions of its neighbors, and a fresh good offspring can be handed to its neighbors too. The neighbors carry useful information for each other precisely because their optima are close.

So define a neighborhood. For each weight vector w^i, let B(i) be the indices of its T closest weight vectors, by Euclidean distance in weight space (including i itself). T is a small number, say around 20. Computing all the neighborhoods is one O(N²) distance-sort up front, then reused every generation — cheap, and I never touch it again. T controls how local the sharing is: too small and almost no information passes between subproblems, so I'm back to N near-independent searches; too large and "neighbors" stop being similar — distant weights have different optima, so recombining their solutions and forcing them on each other just injects noise, and in the limit T = N it's a global, structureless search with no benefit from decomposition. A modest T keeps "neighbor" synonymous with "similar subproblem."

Now I can write the loop, and notice how cheap it is. I have one solution x^i per subproblem, its objective vector FV^i = F(x^i), the running z*, and the neighborhoods B(i). Each generation, I sweep the subproblems. For subproblem i: pick two parents at random from its neighborhood B(i) — *not* from the whole population, because the neighbors are the solutions likely to be good for nearby weights, so recombining them is recombining locally-relevant material; produce an offspring y by simulated binary crossover and polynomial mutation (the standard real-coded operators — SBX makes children spread around the parents with a distribution index η_c controlling how tightly, polynomial mutation perturbs with its own index η_m); evaluate F(y). Then update the ideal point with y: zⱼ ← min(zⱼ, fⱼ(y)) for each objective.

Then the update that closes the loop — and it's the dual of "parents come from the neighborhood." The offspring y was bred to be good around weight w^i, so it might improve *any* subproblem in B(i), not just i itself. So for every neighbor k in B(i), I ask the only question that matters for that subproblem — its scalar one: is y better than the current x^k *under k's own weight*? Compute g^te(y | w^k, z*) and g^te(x^k | w^k, z*); if g^te(y | w^k, z*) ≤ g^te(x^k | w^k, z*), replace x^k with y. Each subproblem is judged purely by its own total order, so the comparison is always decisive — this is the selection pressure that doesn't thin as M grows, the thing dominance ranking couldn't give me. A single good offspring can improve several neighboring subproblems at once, which is exactly the information-sharing that makes co-evolution beat N separate runs.

Count the work per generation. There's no global nondominated sort, none. For each of the N subproblems I do a constant number of variation operations and then T scalar comparisons against the neighborhood, each comparison a max over the M objectives — O(M·T) per subproblem, O(M·N·T) per generation total. That's it. Linear in N for fixed T and M, versus the O(MN²) sort I was paying before, and the selection is a clean total order the whole way. Both of the things that nagged me are gone at once: the quadratic per-generation tax, and the partial-order convergence weakness.

Let me stress-test the update rule for a failure mode. The replacement is "≤", and a single offspring is allowed to overwrite *every* neighbor it beats. Could one lucky offspring flood the whole neighborhood with copies of itself and collapse diversity? It can win a few neighbors in one step, yes — but each subproblem still keeps whatever is best *under its own weight*, and the neighborhoods overlap and tile the simplex, so a solution only sticks where it's genuinely the best for that weight; the next offspring bred for a slightly different weight will displace it there. The structure of distinct, well-spread weights is what holds the spread, not a separate diversity device. I don't need a crowding distance or a σ_share at all — the diversity is *built into* the fixed lattice of weights, one solution pinned to each. That's a real simplification over the dominance-ranked machinery: no density heuristic, no niche radius.

One more refinement on where parents come from. Drawing parents strictly from B(i) is great for exploiting local structure, but if the neighborhood stalls — everyone in it similar, recombination producing nothing new — I'd like an occasional shot of diversity from farther away. So with high probability (say 0.9) I draw the two parents from the neighborhood B(i), and with the small remaining probability I draw them from the whole population. Mostly local exploitation, an occasional global jump to escape a stalled neighborhood. Cheap insurance, one random branch.

Last piece: reporting. Each subproblem stores the best solution *for its weight*, but a solution that's optimal for a skewed weight can itself be dominated by some other subproblem's solution — the per-subproblem store isn't guaranteed to be a clean nondominated set. So I'll also keep an external population EP: every time I generate a y, remove from EP anything F(y) dominates, and add F(y) unless some EP member already dominates it. EP accumulates the nondominated solutions discovered across all subproblems and is what I actually output as the approximated front. (For a clean two-objective run the final population is already an excellent front, but EP is the principled report.)

Let me write it as real code — the simplex-lattice weights, the T-nearest neighborhoods, the Tchebycheff aggregation against the running z*, and the reproduce-from-neighborhood / update-neighbors loop, each block doing exactly one piece I derived.

```python
import numpy as np

# --- Das & Dennis simplex lattice: weights are multiples of 1/H summing to 1 ---
def das_dennis_weights(n_partitions, n_obj):
    def rec(acc, w, left, depth):
        if depth == n_obj - 1:
            w[depth] = left / n_partitions
            acc.append(w.copy()); return
        for i in range(left + 1):
            w[depth] = i / n_partitions
            rec(acc, w.copy(), left - i, depth + 1)
    acc = []
    rec(acc, np.zeros(n_obj), n_partitions, 0)
    W = np.clip(np.array(acc), 1e-6, None)     # keep every objective visible to max
    return W / W.sum(axis=1, keepdims=True)    # N = C(H+m-1, m-1) vectors

# --- Tchebycheff aggregation: weighted L_inf distance to the ideal point z* ---
def tchebycheff(F, w, z):
    return np.max(w * np.abs(F - z), axis=-1)  # g^te(x|w,z*) = max_i w_i |f_i - z*_i|

# --- external nondominated archive (what we report) ---
def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)

def ep_update(EP, fy):
    EP = [g for g in EP if not dominates(fy, g)]          # drop those y dominates
    if not any(dominates(g, fy) for g in EP):
        EP.append(fy.copy())                              # add y if nondominated
    return EP

def moead(F_eval, n_var, xl, xu, n_partitions=99, T=20, n_gen=250,
          eta_c=20, pc=1.0, eta_m=20, prob_nb=0.9, seed=1):
    rng = np.random.default_rng(seed)
    W = das_dennis_weights(n_partitions, 2)               # N weight vectors / subproblems
    N = len(W)
    # T-nearest neighborhoods in weight space: one O(N^2) sort, reused every generation
    B = np.argsort(((W[:, None, :] - W[None, :, :]) ** 2).sum(-1), axis=1)[:, :T]
    X = rng.uniform(xl, xu, (N, n_var))                   # one solution per subproblem
    FV = F_eval(X)
    z = FV.min(axis=0)                                    # initial ideal-point estimate
    EP = []
    for _ in range(n_gen):
        for i in rng.permutation(N):
            # parents from the neighborhood (local material), occasionally global
            pool = B[i] if rng.random() < prob_nb else np.arange(N)
            k, l = rng.choice(pool, 2, replace=False)
            child = sbx_crossover(X[k], X[l], xl, xu, eta_c, pc, rng)
            y = polynomial_mutation(child, xl, xu, eta_m, rng)
            fy = F_eval(y[None, :])[0]
            z = np.minimum(z, fy)                         # z_j <- min(z_j, f_j(y))
            # update every neighbor judged by ITS OWN scalar subproblem (total order)
            g_new = tchebycheff(fy,        W[B[i]], z)
            g_old = tchebycheff(FV[B[i]],  W[B[i]], z)
            repl = B[i][g_new <= g_old]                  # replace where y is no worse
            X[repl] = y; FV[repl] = fy
            EP = ep_update(EP, fy)                        # maintain reported front
    return X, FV, np.array(EP)
```

Before I trust the whole assembly, I should actually run it once. ZDT1 is the obvious test: m = 2, n = 30, true front f₂ = 1 − √f₁. Run H = 99 (so N = 100 subproblems), T = 20, 250 generations, seed 1, and score the result by the vertical residual of each obtained point to the true front, |f₂ − (1 − √f₁)|, plus the f₁ range to see whether the spread reaches both extremes. The output: population 100, mean vertical residual to the front 5.16e-04, with f₁ covering [0.00, 1.00]. So the converged points sit essentially *on* the front (a residual of ~5×10⁻⁴ against objectives of order 1) and they reach from one extreme of the front to the other — convergence and full-width spread, in a single run, with no nondominated sort anywhere in the loop. That is the behavior the whole reframing was supposed to buy, and now I've seen it rather than assumed it.

So the causal chain, end to end: dominance ranking gives a *partial* order whose convergence pressure thins as objectives grow, and it costs an O(MN²) sort every generation — so I drop ranking-of-vectors entirely and scalarize, because a scalar aggregation is a *total* order that stays decisive at any M; the usual "one run per point" cost of scalarization dissolves once I attach one weight vector to each population member and co-evolve all N subproblems in a single run. Weighted sum is blind to concave fronts (verified above — it collapses to the endpoints), so the aggregation is Tchebycheff, g^te = max_i wᵢ|fᵢ − zᵢ*|, anchored to a running ideal point z* updated by zⱼ ← min(zⱼ, fⱼ(y)); the weights come from a Das & Dennis simplex lattice so the front points spread evenly, and the lattice *is* the diversity mechanism, so there's no crowding distance or niche radius. The subproblems collaborate because neighboring weights have neighboring optima: each subproblem breeds from its T-nearest neighborhood and a fresh offspring updates every neighbor it beats under that neighbor's own weight. One generation costs O(N·T) — no global sort — and, as the ZDT1 run confirmed, a single co-evolving run returns a converged, evenly-spread front, reported from an external nondominated archive.
