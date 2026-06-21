Many real design problems carry several conflicting objectives at once — minimize cost while maximizing reliability, minimize weight while maximizing stiffness — and there is no single winner: any solution I might call best is beaten on some objective by another. The honest answer is a whole *set*, the Pareto-optimal trade-off surface, where a solution dominates another iff it is no worse in every objective and strictly better in at least one. The decision-maker wants to see that surface and then choose, which means a good approximation needs two things at once: convergence, the returned points sitting on or near the true front, and diversity, those points spread evenly across it, extremes included. The textbook move is to scalarize — pick weights $w_i$, minimize $\sum_i w_i f_i(x)$, get one point, and sweep the weights to trace the curve. That is wasteful, one full optimization per sampled point, and it is also geometrically blind: minimizing $\sum_i w_i f_i$ slides a hyperplane with normal $w$ down until it touches the feasible objective set, and a supporting hyperplane only ever touches the *convex* boundary. If the true front has a concave dip, no choice of nonnegative weights lands a solution in that dip — the hyperplane skips across it. So on a non-convex front scalarization is not merely slow, it cannot recover entire regions of the answer.

Once I phrase the goal as "hold many trade-off solutions at once and push them all toward the front while keeping them spread," a population-based evolutionary algorithm is the obvious vehicle, because its unit of work is already a set of candidates evolved by selection, crossover and mutation. The catch is that a GA selects on a scalar fitness, and I refuse to scalarize — so I rank a population of vectors by Pareto dominance directly. The relevant prior art already does this. Schaffer's VEGA (1985) selects each sub-population by a single objective, which is weighted-sum selection in disguise and starves the middle of the front. Fonseca & Fleming's MOGA (1993) ranks by $1 + (\text{number of dominators})$ for genuine convergence pressure, but bolts on fitness sharing for spread, dragging in a radius parameter. Srinivas & Deb's NSGA (1994) is the one I stand on: instead of counting dominators it *sorts* the population into nondomination layers — peel all nondominated solutions off as front 1, remove them, peel front 2 from the rest, and so on — which guarantees better-front solutions out-select worse-front ones, and within each front it spreads by sharing. NSGA works, but it carries three specific costs I want gone. Its sort is $O(MN^3)$: extracting one front is $O(MN^2)$, and the pathological case of $N$ singleton fronts pays that $N$ times. It is non-elitist — offspring replace parents, so a solution that landed beautifully on the front in one generation can simply vanish in the next. And its sharing needs a user-supplied radius $\sigma_{\text{share}}$, with the spread quality riding on a value nobody can principle-out, at $O(N^2)$ cost per generation.

I propose NSGA-II, and it removes all three costs while composing them into one $O(MN^2)$-per-generation algorithm. Take the sort first. The cubic blowup comes from re-comparing the same pairs every time a front is peeled, yet the dominance relation between any fixed pair never changes during the sort. So I compute every pairwise relationship exactly once in a single $O(MN^2)$ pass, and for each solution $p$ record two things: the domination count $n_p$, how many solutions dominate $p$, and the dominated set $S_p$, the actual list of solutions that $p$ dominates. Front 1 is then exactly $\{p : n_p = 0\}$, nobody dominates them. To build the next front I walk each $p$ in the current front and, for every $q \in S_p$, decrement $n_q$ by one — because $p$, one of $q$'s dominators, has now been retired into a completed front; the instant some $n_q$ hits zero, all of $q$'s dominators sit in earlier fronts, so $q$ belongs to the next one. The peeling phase does no dominance checks at all, only integer decrements. Counting it carefully: the expensive $O(M)$ checks happen only in the one initial $O(MN^2)$ pass; the outer "for each $p$ in current front" loop runs $N$ times total across all fronts since every individual sits in exactly one front, and the inner "for each $q \in S_p$" loop runs at most $N-1$ times, so the decrement phase is $O(N^2)$. The whole sort is $O(MN^2)$, a full factor of $N$ off NSGA, paid for with $O(N^2)$ storage for the $S_p$ sets — a good trade, since cubic *time* was the thing actually limiting population size.

Elitism comes next, and the fix for "offspring replace parents" is to make parents and offspring compete on equal footing. At generation $t$ I combine parents $P_t$ and offspring $Q_t$, each of size $N$, into $R_t = P_t \cup Q_t$ of size $2N$, run the fast sort on the combined set, and fill the next parent population $P_{t+1}$ front by front from the best of $R_t$ until adding the next whole front would overflow $N$. Any solution excellent among the parents is right there in $R_t$ and cannot be silently dropped, so elitism is automatic with no external archive to maintain — unlike SPEA's bookkeeping, and unlike Rudolph's parent-plus-offspring merge, which had this elitism and even a convergence proof but no diversity mechanism at all. The subtlety is that almost always some front $F_l$ overflows: the fronts up to $F_{l-1}$ fit inside $N$, but $F_{l-1}\cup F_l$ exceeds it, and I can take only part of $F_l$. Since every solution in $F_l$ is mutually nondominated and equally good on convergence, the tie-breaker must be diversity — keep the solutions that best spread the population — and this is exactly where I kill $\sigma_{\text{share}}$.

What did sharing need a radius for? To define "within this distance you crowd me." But I only ever compare solutions within a single front, which is a thin manifold in objective space, so instead of asking "how many neighbors fall inside a radius I must guess," I ask the parameter-free question "how much empty room is around this solution along the front," and read that room straight off the gaps to its nearest neighbors. For a given front, for each objective $m$ I sort the front's solutions by $f_m$; an interior solution $i$ gets the gap between its two straddling neighbors in that order, and summing over objectives gives a normalized side-length measure of the cuboid whose opposite corners are $i$'s nearest neighbors — bigger cuboid, more isolated, more valuable to keep; a tight cluster yields tiny gaps and a small crowding distance, so its members are expendable. Two details matter. The objectives can be on wildly different scales, so each gap is normalized by that objective's range over the front, giving the crowding distance
$$d_i = \sum_{m=1}^{M} \frac{f_m(i+1) - f_m(i-1)}{f_m^{\max} - f_m^{\min}},$$
and dividing this sum by $M$ in code changes no comparison since $M$ is a positive constant across the front. And the boundary solutions, smallest or largest in some objective, have no neighbor on one side; these are precisely the extremes of the front, the points I most want to keep so as not to shrink the reported range of trade-offs, so I assign them infinite crowding distance and they always survive truncation. The cost is $M$ sorts of up to $N$ elements, $O(MN\log N)$, cheaper than sharing's $O(N^2)$, and with nothing to tune.

Now every solution carries two attributes — its nondomination rank, which front it is on, and its crowding distance, how isolated it is within that front — and I fuse them into one ordering, the crowded-comparison operator $\prec_n$: $i \prec_n j$ if $i_{\text{rank}} < j_{\text{rank}}$, or if $i_{\text{rank}} = j_{\text{rank}}$ and $i_{\text{distance}} > j_{\text{distance}}$. The priority is deliberately asymmetric and lexicographic, because convergence must win over diversity: a solution on a better front is better full stop, since getting onto the front is the primary goal and spread is only meaningful *among* equally-converged solutions, so only a rank tie lets crowding distance decide, and on that tie the more isolated (larger distance) point wins. This resolves "which part of the overflowing $F_l$ to keep": sort $F_l$ by $\prec_n$ descending — within one front, just descending crowding distance — and take the first $N - |P_{t+1}|$, so the infinite-distance extremes enter first, then the most isolated interior points, and the clustered ones are dropped. I carry the same pressure into the selection that *makes* offspring, via binary tournament: pick two solutions at random and prefer the feasible one when the other is infeasible, then prefer the one that dominates (or, in the rank variant, the one that wins by $\prec_n$), and otherwise prefer the larger crowding distance. Constraints fold straight into the dominance relation as a pre-emptive layer with no penalty weight — between two solutions a feasible one beats an infeasible one, between two infeasible ones the smaller total constraint violation wins, between two feasible ones ordinary Pareto dominance decides — so feasibility becomes the outermost lexicographic key above front-rank above crowding distance, and the entire $\prec_n$/sort/truncate machinery runs unchanged on this constrained-domination relation. Assembling one generation: from $P_t$ make $Q_t$ by tournament, simulated binary crossover with distribution index $\eta_c$ and polynomial mutation with index $\eta_m$; combine $R_t = P_t \cup Q_t$; fast-sort it (early-exiting the moment enough fronts fill $N$, since I will discard the rest); add whole fronts while they fit and truncate the splitting one by crowding distance to fill $P_{t+1}$ exactly. The dominating term is the sort on $2N$, so a generation costs $O(MN^2)$ at $O(N^2)$ space — and the result is a converged, evenly-spread front from a single run with no diversity parameter.

```python
import numpy as np

def dominance_relation(f_a, f_b, cv_a=0.0, cv_b=0.0):
    # constraint violation is compared before objectives, with no penalty weight
    if cv_a > 0.0 or cv_b > 0.0:
        if cv_a < cv_b:
            return 1
        if cv_b < cv_a:
            return -1
        return 0

    a_better = np.any(f_a < f_b)
    b_better = np.any(f_b < f_a)
    if a_better and not b_better:
        return 1
    if b_better and not a_better:
        return -1
    return 0

def dominates(f_a, f_b, cv_a=0.0, cv_b=0.0):
    return dominance_relation(f_a, f_b, cv_a, cv_b) == 1

def fast_nondominated_sort(F, CV=None, n_stop_if_ranked=None):
    n = len(F)
    CV = np.zeros(n) if CV is None else np.asarray(CV).reshape(-1)
    S = [[] for _ in range(n)]          # S_p: solutions dominated by p
    n_dom = np.zeros(n, dtype=int)      # n_p: number of solutions dominating p
    fronts = [[]]
    for p in range(n):
        for q in range(p + 1, n):       # each unordered pair compared once
            rel = dominance_relation(F[p], F[q], CV[p], CV[q])
            if rel == 1:
                S[p].append(q); n_dom[q] += 1
            elif rel == -1:
                S[q].append(p); n_dom[p] += 1
    for p in range(n):
        if n_dom[p] == 0:
            fronts[0].append(p)
    n_ranked = len(fronts[0])
    i = 0
    while fronts[i] and (n_stop_if_ranked is None or n_ranked < n_stop_if_ranked):
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    nxt.append(q)
        if not nxt:
            break
        fronts.append(nxt)
        n_ranked += len(nxt)
        i += 1
    return fronts

def crowding_distance(F_front):
    n, m = F_front.shape
    dist = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for obj in range(m):
        order = np.argsort(F_front[:, obj])
        f = F_front[order, obj]
        rng = f[-1] - f[0]
        dist[order[0]] = dist[order[-1]] = np.inf       # boundaries always kept
        if rng == 0:
            continue
        dist[order[1:-1]] += (f[2:] - f[:-2]) / rng     # normalized side-length sum
    return dist / m

def crowded_less(rank_i, cd_i, rank_j, cd_j):
    return (rank_i < rank_j) or (rank_i == rank_j and cd_i > cd_j)

def survival(F, n_survive, CV=None):
    CV = np.zeros(len(F)) if CV is None else np.asarray(CV).reshape(-1)
    fronts = fast_nondominated_sort(F, CV, n_stop_if_ranked=n_survive)
    survivors = []
    rank = np.empty(len(F), dtype=int)
    cd = np.zeros(len(F))
    for r, front in enumerate(fronts):
        front = np.array(front)
        d = crowding_distance(F[front])
        for idx, gi in enumerate(front):
            rank[gi] = r; cd[gi] = d[idx]
        if len(survivors) + len(front) <= n_survive:
            survivors.extend(front.tolist())
        else:
            k = n_survive - len(survivors)
            survivors.extend(front[np.argsort(-d)][:k].tolist())   # split last front by cd
            break
    return np.array(survivors), rank, cd

def binary_tournament(F, rank, cd, CV, n, rng, tournament_type="dom"):
    a, b = rng.integers(0, n), rng.integers(0, n)

    if CV[a] > 0.0 or CV[b] > 0.0:
        if CV[a] < CV[b]:
            return a
        if CV[b] < CV[a]:
            return b
    elif tournament_type == "dom":
        rel = dominance_relation(F[a], F[b])
        if rel == 1:
            return a
        if rel == -1:
            return b
    elif tournament_type == "rank":
        if crowded_less(rank[a], cd[a], rank[b], cd[b]):
            return a
        if crowded_less(rank[b], cd[b], rank[a], cd[a]):
            return b

    if cd[a] > cd[b]:
        return a
    if cd[b] > cd[a]:
        return b
    return rng.choice([a, b])

def evaluate(problem, X):
    out = problem.evaluate(X)
    if isinstance(out, tuple):
        F, CV = out
    else:
        F, CV = out, np.zeros(len(out))
    return np.asarray(F), np.asarray(CV).reshape(-1)

def nsga2(problem, pop_size=100, n_gen=200, eta_c=15, pc=0.9, eta_m=20, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(problem.xl, problem.xu, (pop_size, problem.n_var))
    F, CV = evaluate(problem, X)
    _, rank, cd = survival(F, pop_size, CV)
    for _ in range(n_gen):
        idx = [binary_tournament(F, rank, cd, CV, len(X), rng) for _ in range(pop_size)]
        Xc = sbx_crossover(X[idx], problem.xl, problem.xu, eta_c, pc, rng)
        Xc = polynomial_mutation(Xc, problem.xl, problem.xu, eta_m, rng)
        Fc, CVc = evaluate(problem, Xc)
        Xr, Fr, CVr = np.vstack([X, Xc]), np.vstack([F, Fc]), np.r_[CV, CVc]
        surv, rank_r, cd_r = survival(Fr, pop_size, CVr)   # elitist truncation to N
        X, F, CV = Xr[surv], Fr[surv], CVr[surv]
        rank, cd = rank_r[surv], cd_r[surv]
    return X, F
```
