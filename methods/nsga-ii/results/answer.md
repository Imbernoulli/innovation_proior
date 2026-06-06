# NSGA-II: Fast Elitist Nondominated-Sorting Genetic Algorithm

## Problem

Multi-objective optimization returns not a single optimum but a *Pareto front* of trade-off solutions — those that cannot be improved in one objective without worsening another. Scalarizing into a weighted sum yields one point per run and cannot reach the concave regions of a non-convex front. NSGA-II finds a well-converged, evenly-spread approximation of the entire front in a single run of a population-based evolutionary algorithm, with no diversity parameter and O(MN²) per-generation cost.

## Key ideas

1. **Fast nondominated sort — O(MN²).** Compute every pairwise dominance relation *once*: for each solution p store the domination count n_p (how many dominate p) and the dominated set S_p (those p dominates). Front 1 = {p : n_p = 0}. To build the next front, for each p in the current front decrement n_q for every q ∈ S_p; whenever n_q reaches 0, q joins the next front. The peeling phase does no dominance checks, only decrements. This replaces the original NSGA's O(MN³) repeated-rescan sort, at the cost of O(N²) storage for the S_p sets.

2. **Elitism by merging.** Combine parents and offspring, R_t = P_t ∪ Q_t (size 2N), and sort the combined set. Fill the next population P_{t+1} front by front from the best of R_t. Because every parent competes against its offspring, good solutions are never lost — no external archive needed.

3. **Crowding distance — parameter-free diversity.** Within a front, for each objective m sort by f_m; an interior solution i receives the normalized neighbor gap (f_m(i+1) − f_m(i−1)) / (f_m^max − f_m^min), summed over objectives. This is the perimeter of the cuboid spanned by i's nearest neighbors — large = isolated = valuable. Boundary (extreme) solutions get ∞ so the front's range is preserved. Cost O(MN log N), with no σ_share to tune (versus fitness sharing's O(N²) and its required radius).

4. **Crowded-comparison operator ≺_n.** A lexicographic order: i ≺_n j if i_rank < j_rank, or (i_rank = j_rank and i_distance > j_distance). Convergence (front rank) dominates; diversity (crowding distance) breaks ties. Used in both binary tournament selection and in truncating the last accepted front.

## Algorithm (one generation)

```
R_t = P_t ∪ Q_t                              # combine parents and offspring (size 2N)
F = (F_1, F_2, …) = fast-nondominated-sort(R_t)
P_{t+1} = ∅;  i = 1
until |P_{t+1}| + |F_i| ≤ N:                  # add whole fronts while they fit
    crowding-distance-assignment(F_i)
    P_{t+1} = P_{t+1} ∪ F_i;  i = i + 1
Sort(F_i, ≺_n)                               # last (splitting) front, descending
P_{t+1} = P_{t+1} ∪ F_i[1 : (N − |P_{t+1}|)] # fill exactly N by crowding distance
Q_{t+1} = make-new-pop(P_{t+1})              # tournament(≺_n) + SBX + polynomial mutation
t = t + 1
```

Per-generation complexity: nondominated sort O(M(2N)²), crowding assignment O(M(2N)log 2N), ≺_n sort O(2N log 2N) → overall **O(MN²)**, governed by the sort; space **O(N²)**. The sort early-exits once enough fronts fill N. Real-coded variation uses simulated binary crossover (index η_c) and polynomial mutation (index η_m). Constraints are handled by constrained domination: feasible beats infeasible; among infeasible, smaller total violation wins; among feasible, ordinary dominance — no penalty parameter.

## Code

```python
import numpy as np

def dominates(f_a, f_b):
    # minimization: a dominates b iff a <= b in all objectives and a < b in at least one
    return np.all(f_a <= f_b) and np.any(f_a < f_b)

def fast_nondominated_sort(F):
    n = len(F)
    S = [[] for _ in range(n)]          # S_p: solutions dominated by p
    n_dom = np.zeros(n, dtype=int)      # n_p: number of solutions dominating p
    fronts = [[]]
    for p in range(n):
        for q in range(p + 1, n):       # each unordered pair compared once
            if dominates(F[p], F[q]):
                S[p].append(q); n_dom[q] += 1
            elif dominates(F[q], F[p]):
                S[q].append(p); n_dom[p] += 1
    for p in range(n):
        if n_dom[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:                     # peel fronts by decrementing counters
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
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
        dist[order[1:-1]] += (f[2:] - f[:-2]) / rng     # normalized cuboid perimeter
    return dist

def crowded_less(rank_i, cd_i, rank_j, cd_j):
    return (rank_i < rank_j) or (rank_i == rank_j and cd_i > cd_j)

def survival(F, n_survive):
    fronts = fast_nondominated_sort(F)
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

def binary_tournament(rank, cd, n, rng):
    a, b = rng.integers(0, n), rng.integers(0, n)
    return a if crowded_less(rank[a], cd[a], rank[b], cd[b]) else b

def nsga2(problem, pop_size=100, n_gen=200, eta_c=15, pc=0.9, eta_m=20, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(problem.xl, problem.xu, (pop_size, problem.n_var))
    F = problem.evaluate(X)
    _, rank, cd = survival(F, pop_size)
    for _ in range(n_gen):
        idx = [binary_tournament(rank, cd, len(X), rng) for _ in range(pop_size)]
        Xc = sbx_crossover(X[idx], problem.xl, problem.xu, eta_c, pc, rng)
        Xc = polynomial_mutation(Xc, problem.xl, problem.xu, eta_m, rng)
        Fc = problem.evaluate(Xc)
        Xr, Fr = np.vstack([X, Xc]), np.vstack([F, Fc])   # R_t = P_t U Q_t
        surv, rank, cd = survival(Fr, pop_size)            # elitist truncation to N
        X, F = Xr[surv], Fr[surv]
    return X, F
```

This mirrors the structure of widely-used implementations (e.g. `pymoo`'s `NSGA2`): a `RankAndCrowding` survival operator built on a fast nondominated sort plus per-front crowding distance, a binary tournament keyed on the crowded-comparison operator, and SBX (η_c = 15, p_c = 0.9) with polynomial mutation (η_m = 20) at population size 100. Default tournament uses dominance directly between the two candidates, falling back to crowding distance when neither dominates; feasibility/constraint-violation is compared first when present.
