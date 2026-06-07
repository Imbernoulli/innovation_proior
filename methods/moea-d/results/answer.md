# MOEA/D — Multiobjective Evolutionary Algorithm based on Decomposition

## Problem

Multi-objective optimization returns not a single optimum but a *Pareto front* of trade-off solutions. The dominance-based approach ranks a population by Pareto dominance — a *partial* order whose convergence pressure weakens as the number of objectives grows (almost every pair becomes mutually nondominated, so nearly everyone is rank 1 and selection loses its grip), and whose per-generation nondominated sort costs O(MN²). MOEA/D instead **decomposes** the problem into N scalar subproblems via a set of uniformly spread weight vectors, attaches one solution to each subproblem, and co-evolves them all in a single run. Each subproblem is a *total* order, so selection stays decisive at any M, and selection/update touch only a small neighborhood, costing O(N·T) per generation with no global sort.

## Key idea

1. **Decompose into N scalar subproblems.** Lay down N evenly spread weight vectors w¹…wᴺ on the simplex and turn the multi-objective problem into N scalarized subproblems, one per weight. The Pareto front is then the collection of the N subproblem optima — recovered in a single run, not one run per point.

2. **Tchebycheff aggregation (reaches concave fronts).** Subproblem i minimizes
   g^te(x | wⁱ, z*) = max₁≤ⱼ≤ₘ wⱼⁱ |fⱼ(x) − zⱼ*|,
   where z* is the ideal point (running per-objective minimum). Unlike the weighted sum g^ws = Σⱼ wⱼ fⱼ(x), whose hyperplane contours only touch the convex hull of the front, the Tchebycheff function's box-corner contours can touch concave regions, and every Pareto-optimal point is the minimizer for some weight. (For three or more objectives the **PBI** alternative g^pbi = d₁ + θd₂, with d₁ = ‖(F(x)−z*)ᵀw‖/‖w‖ the distance along the reference direction and d₂ = ‖F(x) − (z* + d₁w)‖ the perpendicular distance off it, gives a smoother, more even spread for a penalty θ > 0.)

3. **Uniform weights = built-in diversity.** Weight vectors come from the Das & Dennis simplex lattice: components are nonnegative multiples of 1/H summing to 1, giving N = C(H+M−1, M−1) vectors. The population size equals N. Because each solution is pinned to a distinct, evenly spread weight, the diversity is structural — no crowding distance, no niche radius.

4. **Neighborhood collaboration.** B(i) = the T weight vectors nearest wⁱ (Euclidean, in weight space). Neighboring weights define near-identical subproblems with near-identical optima, so subproblems share: parents are drawn from B(i), and a fresh offspring updates *every* neighbor it improves under that neighbor's own weight. This is what makes co-evolution beat N independent runs.

5. **Ideal-point update + neighbor replacement.** After evaluating offspring y, update z by zⱼ ← min(zⱼ, fⱼ(y)) for each j. Then for each k ∈ B(i), replace xᵏ with y iff g^te(y | wᵏ, z*) ≤ g^te(xᵏ | wᵏ, z*). An external population EP collects the nondominated solutions found, and is the reported front.

## Algorithm (vanilla MOEA/D, Tchebycheff)

```
Input: weight vectors W = {w^1,…,w^N} (Das–Dennis lattice), neighbor size T
Init:  B(i) = indices of the T nearest weight vectors to w^i (incl. i)
       x^i ~ random; FV^i = F(x^i);  z_j = min_i FV^i_j;  EP = ∅
repeat for n_gen generations:
  for i = 1..N (random order):
     pick k,l from B(i)         # with prob 0.9 from B(i), else from {1..N}
     y = mutate(crossover(x^k, x^l))           # SBX (η_c) + polynomial mutation (η_m)
     evaluate F(y)
     z_j = min(z_j, f_j(y))      for all j                 # update ideal point
     for each k in B(i):                                   # update neighbors
        if  g^te(y | w^k, z) <= g^te(x^k | w^k, z):
            x^k = y ;  FV^k = F(y)
     EP = nondominated update of EP with F(y)
output EP (approximated Pareto front)
```

Per-generation cost: O(N·T) scalar comparisons, **no global nondominated sort**. The neighborhoods are computed once in O(N²). Tchebycheff is the default for M ≤ 2; PBI for M ≥ 3.

## Code

Self-contained MOEA/D with the Das–Dennis lattice, T-nearest neighborhoods, Tchebycheff aggregation against a running ideal point z*, and the reproduce-from-neighborhood / update-neighbors loop. Returns the population and the external nondominated archive on ZDT1. Structure mirrors `pymoo`'s `MOEAD` (`NeighborhoodSelection` with `prob_neighbor_mating`, neighbors via `argsort(cdist(W, W))[:, :T]`, ideal point `np.min(F, axis=0)`, replacement where the decomposed offspring value is no worse) and its `Tchebicheff`/`PBI` decompositions.

```python
import numpy as np


def das_dennis_weights(n_partitions, n_obj):
    # uniform simplex lattice: components are multiples of 1/H summing to 1
    def rec(acc, w, left, depth):
        if depth == n_obj - 1:
            w[depth] = left / n_partitions
            acc.append(w.copy()); return
        for i in range(left + 1):
            w[depth] = i / n_partitions
            rec(acc, w.copy(), left - i, depth + 1)
    acc = []
    rec(acc, np.zeros(n_obj), n_partitions, 0)
    W = np.clip(np.array(acc), 1e-6, None)      # keep every objective visible to max
    return W / W.sum(axis=1, keepdims=True)     # N = C(H+m-1, m-1) vectors


def tchebycheff(F, w, z):
    return np.max(w * np.abs(F - z), axis=-1)   # max_i w_i |f_i - z*_i|


def dominates(a, b):
    return np.all(a <= b) and np.any(a < b)


def ep_update(EP, fy):
    EP = [g for g in EP if not dominates(fy, g)]
    if not any(dominates(g, fy) for g in EP):
        EP.append(fy.copy())
    return EP


def sbx_crossover(p1, p2, xl, xu, eta, pc, rng):
    c = p1.copy()
    if rng.random() <= pc:
        for i in range(len(p1)):
            if rng.random() <= 0.5 and abs(p1[i] - p2[i]) > 1e-14:
                x1, x2 = min(p1[i], p2[i]), max(p1[i], p2[i])
                u = rng.random()
                beta = 1.0 + 2.0 * (x1 - xl[i]) / (x2 - x1)
                alpha = 2.0 - beta ** (-(eta + 1))
                bq = (u * alpha) ** (1 / (eta + 1)) if u <= 1 / alpha \
                    else (1 / (2 - u * alpha)) ** (1 / (eta + 1))
                c[i] = 0.5 * ((x1 + x2) - bq * (x2 - x1))
    return np.clip(c, xl, xu)


def polynomial_mutation(x, xl, xu, eta, rng):
    x = x.copy(); pm = 1.0 / len(x)
    for i in range(len(x)):
        if rng.random() <= pm:
            d1 = (x[i] - xl[i]) / (xu[i] - xl[i])
            d2 = (xu[i] - x[i]) / (xu[i] - xl[i])
            u, mp = rng.random(), 1 / (eta + 1)
            if u < 0.5:
                dq = (2 * u + (1 - 2 * u) * (1 - d1) ** (eta + 1)) ** mp - 1
            else:
                dq = 1 - (2 * (1 - u) + 2 * (u - 0.5) * (1 - d2) ** (eta + 1)) ** mp
            x[i] = np.clip(x[i] + dq * (xu[i] - xl[i]), xl[i], xu[i])
    return x


def moead(F_eval, n_var, xl, xu, n_partitions=99, T=20, n_gen=250,
          eta_c=20, pc=1.0, eta_m=20, prob_nb=0.9, seed=1):
    rng = np.random.default_rng(seed)
    W = das_dennis_weights(n_partitions, 2)                 # N subproblems
    N = len(W)
    B = np.argsort(((W[:, None, :] - W[None, :, :]) ** 2).sum(-1), axis=1)[:, :T]
    X = rng.uniform(xl, xu, (N, n_var))                     # one solution per subproblem
    FV = F_eval(X)
    z = FV.min(axis=0)                                      # ideal-point estimate
    EP = []
    for _ in range(n_gen):
        for i in rng.permutation(N):
            pool = B[i] if rng.random() < prob_nb else np.arange(N)
            k, l = rng.choice(pool, 2, replace=False)
            child = sbx_crossover(X[k], X[l], xl, xu, eta_c, pc, rng)
            y = polynomial_mutation(child, xl, xu, eta_m, rng)
            fy = F_eval(y[None, :])[0]
            z = np.minimum(z, fy)                           # z_j <- min(z_j, f_j(y))
            g_new = tchebycheff(fy,       W[B[i]], z)
            g_old = tchebycheff(FV[B[i]], W[B[i]], z)
            repl = B[i][g_new <= g_old]                     # replace neighbors y beats
            X[repl] = y; FV[repl] = fy
            EP = ep_update(EP, fy)
    return X, FV, np.array(EP)


def zdt1(X):
    f1 = X[:, 0]
    g = 1 + 9 * X[:, 1:].sum(axis=1) / (X.shape[1] - 1)
    f2 = g * (1 - np.sqrt(f1 / g))
    return np.column_stack([f1, f2])


if __name__ == "__main__":
    X, F, EP = moead(zdt1, 30, np.zeros(30), np.ones(30),
                     n_partitions=99, n_gen=250, seed=1)
    d = np.abs(F[:, 1] - (1 - np.sqrt(np.clip(F[:, 0], 0, 1))))   # vs true PF f2=1-sqrt(f1)
    print(f"pop {len(F)}  EP {len(EP)}  mean|dist to PF| {d.mean():.2e}  f1∈[{F[:,0].min():.2f},{F[:,0].max():.2f}]")
```

For three or more objectives, swap the Tchebycheff aggregation for PBI:

```python
def pbi(F, w, z, theta=5.0):
    wn = w / np.linalg.norm(w, axis=-1, keepdims=True)
    d1 = np.sum((F - z) * wn, axis=-1)                  # along the reference direction
    d2 = np.linalg.norm((F - z) - d1[..., None] * wn, axis=-1)   # perpendicular off it
    return d1 + theta * d2
```
