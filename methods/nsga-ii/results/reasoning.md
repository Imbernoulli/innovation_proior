Let me start from what actually hurts. I have a design problem with several objectives that fight each other — minimize cost, maximize reliability, that kind of thing. There is no single winner: any solution I might call "best" is beaten on some objective by another. So the real answer is a whole set, the trade-off surface — the points where I can't do better on one objective without doing worse on another. The decision-maker wants to *see* that surface and then choose. My job is to produce it.

The textbook approach is to scalarize: pick weights w_i, optimize Σ w_i f_i(x), get one point. To get the surface I sweep the weights and re-solve, over and over. Two things bother me about this. First, it's wasteful — one full optimization per point on a curve I want densely sampled. Second, and worse: a weighted sum can only ever find points on the convex hull of the front. Geometrically, minimizing Σ w_i f_i means sliding a hyperplane with normal w down until it touches the feasible objective set; it touches the *convex* boundary. If the true front has a concave dip, no choice of nonnegative weights ever lands a solution in that dip — the hyperplane skips across it. So on a non-convex front, scalarization isn't just slow, it's blind to entire regions. That's disqualifying for the goal of recovering the whole front.

So scalarization is the wrong frame. What I actually want is to hold many trade-off solutions at once and push them all toward the front while keeping them spread out. The moment I phrase it as "hold many solutions at once," a population-based evolutionary algorithm is the obvious vehicle — its unit of work is already a *set* of candidates, evolved by selection, crossover, mutation. One run, one population, ideally the whole front falls out.

But there's an immediate gap. A GA needs a scalar fitness to select on, and I deliberately refuse to scalarize the objectives — that's the whole point. So how do I rank a population of vectors? The honest ordering on vectors is Pareto dominance: x dominates y iff x is no worse in every objective and strictly better in at least one. That's a partial order — most pairs are mutually nondominated, which is fine, that's what a front *is*.

Let me look at what people have already built on this, because I don't want to reinvent the dead ends.

Schaffer's VEGA (1985) is the oldest. Split the population into M chunks, select chunk m purely by objective m, then shuffle and recombine. Clever and simple — but each chunk is just a single-objective selector pulling toward one objective's optimum, so the population speciates toward the M extremes and starves the middle of the front. It's a weighted-sum in disguise (selecting by one objective is the corner case of weights), so it inherits the convex-front blindness and, on top, can't even populate the interior. No dominance anywhere. Not enough.

Fonseca & Fleming's MOGA (1993) finally uses dominance: rank an individual by 1 + (how many population members dominate it), so all nondominated guys get rank 1. Map rank to fitness. That gives genuine convergence pressure. But for spread they bolt on fitness sharing — and sharing drags in a parameter, σ_share, which I'll come back to because it's a recurring tax.

Srinivas & Deb's NSGA (1994) is the one I'm really standing on. Instead of "count your dominators," it *sorts* the population into layers: peel off all nondominated solutions as front 1 and give them a big dummy fitness; remove them; peel the nondominated solutions of what's left as front 2 with a smaller dummy fitness; repeat. Front-by-front this guarantees better-front solutions out-select worse-front ones — clean convergence pressure. Within each front it spreads using fitness sharing.

NSGA works, but staring at it I see three specific costs, and each one is a thing I'd want to remove.

First cost — the sorting is expensive. Let me actually count it, because the exponent matters. To find front 1: take a solution p, compare it against every other solution to decide if anything dominates it. Each dominance check between two M-vectors is M comparisons, O(M). There are N−1 others, so deciding p's status is O(MN). Do that for all N solutions to extract front 1: O(MN^2). Now remove front 1 and repeat on the remainder to get front 2 — another O(MN^2) in the worst case, because almost everything could still be there. The pathological case is N fronts each holding a single solution; then I pay O(MN^2) about N times → O(MN^3) overall. That cubic in N is exactly why NSGA chokes at large population sizes. Storage, at least, is cheap — O(N), I'm just tracking who's been peeled.

Second cost — it's non-elitist. Each generation I build offspring and they *replace* the parents. A solution that landed beautifully on the front in generation t can simply be gone in generation t+1 if the variation operators don't reproduce it. That's throwing away hard-won progress, and elitism — keeping the best around — is known to both speed up and stabilize GA convergence. I want elitism.

Third cost — σ_share. Sharing spreads a front by dividing each individual's fitness by a niche count Σ_j sh(d_ij), with sh(d) = 1 − (d/σ_share)^α inside radius σ_share and 0 outside. The whole behavior hinges on σ_share, which the user has to pick; too large and everything shares, too small and nothing does. There are guidelines but no principled parameter-free choice. And there's a hidden second cost here: to compute each individual's niche count I compare it against every other individual in the front — that's O(N^2) work per generation just for diversity, on top of the sort.

So I have a concrete wish list: kill the O(MN^3), add elitism, and remove σ_share (ideally making diversity cheaper than O(N^2) too). Let me take them one at a time, because each has a clean fix and they compose.

The sort first. Why is the naive version cubic? Because I keep *re-comparing* the same pairs — every time I peel a front, the survivors get compared against each other all over again. The waste is obvious once I name it: dominance between any fixed pair p, q is a fact that never changes during the sort. I should compute every pairwise relationship exactly *once* and then reuse it.

So let me do a single O(MN^2) pass over all ordered pairs, and for each solution p record two things. One, n_p: how many solutions dominate p — a counter. Two, S_p: the actual *set* of solutions that p dominates — a list. One pass, every pair touched once, O(M) per comparison: O(MN^2) time. The price is memory — S_p can hold up to N−1 entries each, so storage jumps from O(N) to O(N^2). I'll take that trade; memory is cheap and the cubic time was the killer.

Now the fronts come almost for free. Front 1 is exactly the solutions with n_p = 0 — nobody dominates them. Here's the move that avoids ever re-comparing: take each p in the current front and walk its dominated set S_p; for each q in S_p, decrement n_q by one — because p, one of q's dominators, has now been "removed" into a completed front. Whenever some n_q hits zero, every solution that used to dominate q is already in an earlier front, so q belongs to the *next* front; collect it. Process the whole current front this way to assemble the next one, set its rank, and continue until no solutions remain.

Let me make sure this is really O(MN^2) and not secretly cubic again. The expensive part — the M-comparison dominance checks — happens only in the *initial* pass that builds all the n_p and S_p: that's O(MN^2), done once. The front-peeling phase afterward does no dominance checks at all, just integer decrements. How many decrements? Look at the loop structure two ways. The outer loop "for each p in current front" runs, summed over all fronts, exactly N times total — every individual sits in exactly one front, so it's visited once as a front member. The inner loop "for each q in S_p" runs at most N−1 times per p. So the decrement phase is O(N^2). Combined with the O(MN^2) initial pass, the whole sort is O(MN^2). I've knocked a full factor of N off NSGA's sort, paying only O(N^2) storage. Good — first wish granted.

Elitism next. The non-elitism came from offspring *replacing* parents. The fix is to make parents and offspring compete on equal footing. So at generation t, with parents P_t and offspring Q_t each of size N, combine them: R_t = P_t ∪ Q_t, size 2N. Now run my fast nondominated sort on the *combined* set. Any solution that was excellent among the parents is right there in R_t, competing — it can't be silently dropped. Then I fill the next parent population P_{t+1} from R_t front by front: take all of front 1, then all of front 2, and so on, until adding the next whole front would overflow N. Because the best fronts of the combined 2N always survive, elitism is automatic — no external archive to maintain, unlike SPEA's bookkeeping, no separate elite set. (Rudolph had floated this parent-plus-offspring nondominated merge and even proved it converges, but he left it with no diversity mechanism at all — pure convergence, no spread, which is what kept it from being usable.)

There's a subtlety in "fill front by front." It almost always happens that some front F_l overflows — the fronts up to F_{l−1} fit inside N, but F_{l−1} plus F_l exceeds N. I can take only part of F_l. *Which* part? Every solution in F_l is mutually nondominated — they're all equally good on convergence. So the tie-breaker has to be diversity: from this last accepted front, keep the solutions that best spread the population, and drop the rest. That's precisely where I need a diversity measure — and it's my chance to kill σ_share.

What did sharing actually need σ_share *for*? To define a radius — "within this distance you're crowding me." But I'm only ever comparing solutions *within a single front*, and a front is a one-dimensional-ish manifold sitting in objective space. Instead of asking "how many neighbors fall inside a radius I have to guess," I can ask a parameter-free question: "how much empty room is around this solution along the front?" The emptiness is just the gap to its nearest neighbors on each side — and I can read those gaps off directly, no radius needed.

Concretely, for a given front, take one objective m at a time, sort the front's solutions by f_m. For an interior solution i, its two neighbors in this sorted order, i−1 and i+1, straddle it; the gap f_m(i+1) − f_m(i−1) is how much room it has along objective m. Do this for every objective and add the gaps up. Geometrically that sum is the perimeter of the cuboid whose opposite corners are the two nearest neighbors — the bigger the cuboid, the more isolated i is, the more I want to keep it to preserve spread. A solution sitting in a tight cluster has tiny gaps on all sides, small cuboid, small crowding distance — it's expendable because its neighbors already represent its region.

Two details fall out of thinking about it carefully. The objectives can be on wildly different scales — one in dollars, one as a probability — and if I just add raw gaps, the large-range objective swamps the sum. So I normalize each objective's gap by that objective's range over the front, (f_m^max − f_m^min). Now every objective contributes commensurably. The crowding distance of i becomes the sum over m of (f_m(i+1) − f_m(i−1)) / (f_m^max − f_m^min).

The second detail is the boundary solutions — the ones that are smallest or largest in some objective after sorting. They have no neighbor on one side, so the cuboid is undefined. These are the *extremes* of the front, exactly the points I most want to keep, because losing them shrinks the range of trade-offs I report. So I assign them infinite crowding distance — they always survive the truncation. That also handles the edges of the sort cleanly.

And look at the cost. To compute crowding distances for a front I sort it once per objective: M sorts of up to N elements, O(MN log N). Compare that with sharing's O(N^2) per front — cheaper. And there is *no parameter*: nothing to set, nothing to guess. Both diversity wishes granted at once — σ_share gone, and the work dropped from quadratic to N log N.

Now I have, for every solution, two attributes: its nondomination rank (which front), and its crowding distance (how isolated within that front). I need one ordering that fuses them, used wherever I select. The priority is not symmetric: convergence must win over diversity — a solution on a better front is better, full stop, regardless of how crowded it is, because getting onto the front is the primary goal and spread is only meaningful *among* equally-converged solutions. Only when two solutions are on the *same* front does spread decide. So the ordering is lexicographic. Define the crowded-comparison operator ≺_n: i ≺_n j if i_rank < j_rank, or if i_rank = j_rank and i_distance > j_distance. Lower rank wins; on a rank tie, larger crowding distance (more isolated, more valuable for spread) wins. That's the single comparator I'll use everywhere selection happens.

So "which part of the overflowing front F_l to keep" has a clean answer: sort F_l by ≺_n in descending order — which, within one front, just means descending crowding distance — and take the first N − |P_{t+1}| of them. The infinite-distance extremes go in first, then the most isolated interior points, and the tightly-clustered ones get dropped. Exactly the diversity-preserving truncation I wanted.

Let me also use ≺_n in the selection that *makes* the offspring, so the same convergence-then-diversity pressure drives variation, not just survival. Binary tournament selection: pick two solutions at random, keep the ≺_n-better one, repeat to fill the mating pool; then crossover and mutate. Now every selection in the algorithm — both building the mating pool and truncating the survivors — is governed by one consistent rule.

Let me assemble the generation. Start with parents P_t. By binary tournament under ≺_n, plus crossover and mutation, make offspring Q_t of size N. Combine R_t = P_t ∪ Q_t (size 2N). Fast-nondominated-sort R_t into F_1, F_2, …. Initialize P_{t+1} empty; add whole fronts F_1, F_2, … while they still fit within N. For the first front F_l that would overflow, compute crowding distances on F_l, sort it by ≺_n descending, and add its best N − |P_{t+1}| solutions to fill P_{t+1} exactly. Then from P_{t+1}, by tournament-under-≺_n, crossover, mutation, build Q_{t+1}, and loop. The very first generation is the only special case: there are no previous best to merge with, so I just sort P_0, then do selection/crossover/mutation to get Q_0; from generation 1 on, the merge-and-truncate machinery runs.

One efficiency note on the sort inside the loop. I'm sorting R_t of size 2N, but I only need enough fronts to fill N survivors. The moment the accumulated fronts reach N members I can stop sorting — there's no reason to rank solutions I'm about to discard. So the nondominated sort early-exits once enough fronts are found.

Now the complexity of one whole generation, which is what I really care about. The nondominated sort on the 2N combined population is O(M(2N)^2). Crowding-distance assignment is O(M(2N)log(2N)). The ≺_n sort of a front is O(2N log(2N)). The dominating term is the nondominated sort: O(MN^2) per generation. So the entire algorithm is O(MN^2) — I've removed a full factor of N relative to NSGA's O(MN^3), while *also* adding elitism and *removing* σ_share. The cost I paid is space: O(N^2), from storing the S_p sets. That's the deal, and it's a good one — the cubic time was the thing actually limiting population size.

A couple of pieces remain to make this run on real-valued problems. Variation: with real-coded variables I want crossover and mutation that respect the continuous geometry. Simulated binary crossover produces two children spread around the parents, with a distribution index η_c controlling how tightly the children cluster near the parents (larger η_c → children closer to parents); polynomial mutation perturbs a variable by a polynomial-distributed step with its own index η_m. These are the standard real-coded operators and they slot straight into the tournament→crossover→mutation pipeline.

Constraints, finally. Many real problems are constrained, and I'd rather not introduce a penalty weight — that's just σ_share's cousin, another knob. The clean trick is to fold feasibility *into* the dominance relation itself, as a pre-emptive layer: between two solutions, a feasible one always beats an infeasible one; between two infeasible ones, the one with smaller total constraint violation wins; between two feasible ones, ordinary Pareto dominance decides. No penalty parameter, and the entire ≺_n / sort / truncate machinery is untouched — it just runs on this constrained-domination relation. Feasibility becomes the outermost lexicographic key, sitting above front-rank, which sits above crowding distance.

Let me write it as real code. I'll keep the fast sort with explicit n_p and S_p, the cuboid crowding distance with normalized objectives and infinite boundaries, the ≺_n comparator, and the combine-sort-truncate loop — each block doing exactly one of the pieces I derived.

```python
import numpy as np

# --- dominance on objective vectors (minimization) ---
def dominates(f_a, f_b):
    # a dominates b iff a <= b in every objective and a < b in at least one
    return np.all(f_a <= f_b) and np.any(f_a < f_b)

# --- fast nondominated sort: one O(MN^2) pass, then peel by decrement ---
def fast_nondominated_sort(F):
    n = len(F)
    S = [[] for _ in range(n)]       # S_p: solutions that p dominates
    n_dom = np.zeros(n, dtype=int)   # n_p: number of solutions that dominate p
    fronts = [[]]
    for p in range(n):
        for q in range(p + 1, n):    # touch each unordered pair once
            if dominates(F[p], F[q]):
                S[p].append(q)        # p dominates q
                n_dom[q] += 1
            elif dominates(F[q], F[p]):
                S[q].append(p)        # q dominates p
                n_dom[p] += 1
    for p in range(n):
        if n_dom[p] == 0:             # nobody dominates p -> front 1
            fronts[0].append(p)
    # peel: decrement counters, no more dominance checks
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:     # all of q's dominators are in earlier fronts
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()                      # last appended front is empty
    return fronts

# --- crowding distance: normalized cuboid perimeter, infinite at boundaries ---
def crowding_distance(F_front):
    n, m = F_front.shape
    dist = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)     # all are boundary -> always kept
    for obj in range(m):
        order = np.argsort(F_front[:, obj])
        f = F_front[order, obj]
        rng = f[-1] - f[0]
        dist[order[0]] = dist[order[-1]] = np.inf      # extremes preserved
        if rng == 0:
            continue
        # interior i gets the normalized gap between its two neighbors
        dist[order[1:-1]] += (f[2:] - f[:-2]) / rng
    return dist

# --- crowded-comparison: rank first (convergence), then crowding (diversity) ---
def crowded_less(rank_i, cd_i, rank_j, cd_j):
    return (rank_i < rank_j) or (rank_i == rank_j and cd_i > cd_j)

# --- elitist survival: combine, sort, fill front by front, truncate last by cd ---
def survival(F, n_survive):
    fronts = fast_nondominated_sort(F)
    survivors, rank = [], np.empty(len(F), dtype=int)
    cd = np.zeros(len(F))
    for r, front in enumerate(fronts):
        front = np.array(front)
        d = crowding_distance(F[front])
        for idx, gi in enumerate(front):
            rank[gi] = r
            cd[gi] = d[idx]
        if len(survivors) + len(front) <= n_survive:
            survivors.extend(front.tolist())          # whole front fits
        else:
            k = n_survive - len(survivors)            # last (splitting) front
            keep = front[np.argsort(-d)][:k]          # largest crowding distance
            survivors.extend(keep.tolist())
            break
    return np.array(survivors), rank, cd

def binary_tournament(rank, cd, n, rng):
    a, b = rng.integers(0, n), rng.integers(0, n)
    return a if crowded_less(rank[a], cd[a], rank[b], cd[b]) else b

def nsga2(problem, pop_size=100, n_gen=200, eta_c=15, pc=0.9, eta_m=20, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(problem.xl, problem.xu, (pop_size, problem.n_var))
    F = problem.evaluate(X)
    _, rank, cd = survival(F, pop_size)               # rank/cd for the first selection
    for _ in range(n_gen):
        # mating pool by tournament under crowded-comparison
        idx = [binary_tournament(rank, cd, len(X), rng) for _ in range(pop_size)]
        Xp = X[idx]
        Xc = sbx_crossover(Xp, problem.xl, problem.xu, eta_c, pc, rng)
        Xc = polynomial_mutation(Xc, problem.xl, problem.xu, eta_m, rng)
        Fc = problem.evaluate(Xc)
        # R_t = P_t U Q_t, then elitist truncation back to pop_size
        Xr = np.vstack([X, Xc]); Fr = np.vstack([F, Fc])
        surv, rank, cd = survival(Fr, pop_size)
        X, F = Xr[surv], Fr[surv]
    return X, F
```

The causal chain, end to end: scalarization can't recover a non-convex front and wastes a run per point, so I switch to a population and rank it by Pareto dominance instead of a scalar; NSGA already does dominance-front sorting but pays O(MN^3) by re-comparing pairs every peel, is non-elitist, and needs σ_share — so I compute every pairwise relation once as n_p and S_p and peel fronts by decrementing counters, dropping the sort to O(MN^2) at O(N^2) memory; I get elitism for free by merging parents and offspring into R_t and filling the next generation from the best fronts of the combined 2N; and I replace sharing with a parameter-free crowding distance — the normalized cuboid perimeter to each solution's nearest neighbors, infinite at the extremes — folded with rank into a single lexicographic crowded-comparison operator that drives both tournament selection and the truncation of the splitting front. One generation costs O(MN^2), governed by the sort, and the whole thing returns a converged, evenly-spread front in a single run with no diversity parameter.
