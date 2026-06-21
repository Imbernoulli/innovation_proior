The task is to evolve, in a single run, a good finite approximation of the Pareto-optimal set of a vector objective $f(x) = (f_1(x), \dots, f_m(x))$ with all components minimized. Because the objectives conflict there is no single optimum but a whole front, and the set I return has to be two things at once: close to the true front (convergence) and uniformly spread along it with the boundary trade-offs kept (diversity). That makes selection — not the variation operators, which are fine — the hard part. The only quality signal intrinsic to the problem is Pareto dominance, where $a \succ b$ iff $a$ is no worse on every objective and strictly better on at least one. But dominance is only a *partial* order: two individuals that each win on a different axis are mutually indifferent, and with three or more objectives that indifference is the normal case rather than the exception. A fitness built purely on dominance therefore leaves most of the population tied, and a tied population is directionless — selection cannot push it anywhere.

The inherited strength scheme tried to refine dominance into a real number but only half-succeeded. It keeps a population $P$ and an external archive $\bar P$ of elite nondominated solutions, gives each archive member $i$ a strength $S(i)$ equal to the count of population members it dominates (normalized into $[0,1)$), lets that strength be the archive member's fitness, and then assigns a population member $j$ the fitness $F(j) = 1 + \sum_{i \in \bar P,\, i \succeq j} S(i)$. This buys elitism for free — archive members land in $[0,1)$, population members at $\ge 1$ — but it is fatally coarse: two population members dominated by the *same* set of archive members get *identical* fitness no matter how they relate to each other, and in the extreme where the archive holds a single individual the whole population collapses into at most two fitness levels and selection degrades to random search, precisely when the archive is small, which is precisely early on. The failure is that strength was assigned only to the archive, so dominated individuals are ranked solely by who sits above them and never by what they themselves do. Separately, that scheme reduced an over-full archive by clustering, which merges members to centroids and tends to drop the outermost solutions — exactly the boundary trade-offs a good spread most needs — while letting the archive size drift unpredictably.

I propose SPEA2 (Strength Pareto Evolutionary Algorithm 2). It rests on a single fine-grained scalar fitness, dominance first and density only as a tie-breaker, paired with a fixed-size, boundary-preserving elite archive. The first move is to stop restricting strength to the archive. Give *every* individual in the combined pool $P_t + \bar P_t$ a strength equal to the number of individuals it dominates,
$$S(i) = \big|\{\, j \in P_t + \bar P_t : i \succ j \,\}\big|.$$
Now each individual carries a number saying how much of the pool it sits above, so two individuals dominated from above by the same set are no longer forced to tie — they generally dominate different numbers of individuals below. But $S(i)$ has the wrong sign to minimize directly and does not yet encode how *dominated* an individual is, which is what measures distance from the front. So I build the raw fitness as the summed strengths of $i$'s dominators,
$$R(i) = \sum_{j \in P_t + \bar P_t,\; j \succ i} S(j).$$
This is doing exactly the right thing. If nobody dominates $i$ the sum is empty and $R(i) = 0$, so $R(i)=0$ characterizes the nondominated front cleanly; if $i$ is dominated, $R(i)$ grows, and it grows *more* when $i$ is dominated by individuals that themselves dominate many others. Weighting by the dominator's strength rather than just counting dominators is the load-bearing choice: a dominator that covers a huge swath of the pool sits far out toward the front, so being below it means I am deep in the interior and far from the trade-off surface, whereas a dominator that barely dominates anything is near the back itself, a milder verdict. Minimizing $R$ thus pushes individuals toward the front, with $R=0$ being the front.

Raw fitness solves convergence but reintroduces the tie problem in a new place: every nondominated individual has $R(i)=0$, yet those are exactly the points I keep and whose spread I care about. So I need a second, independent signal — density — to prefer the lonely points over the crowded ones. A hyper-grid histogram (as in PESA) is cheap but resolution-dependent: its verdict depends entirely on where the grid lines fall. A per-objective crowding distance (as in NSGA-II) is an axis-sort quantity defined relative to neighbors along each axis and only within one front, not a genuine density of the point cloud. I want a true metric density, one scalar per individual, with a bandwidth that adapts to how dense the local region is. The $k$-th nearest neighbor estimator from statistics gives exactly that: density is a decreasing function of the distance to one's $k$-th nearest data point, so a near $k$-th neighbor means crowded and a far one means isolated, and the bandwidth adapts automatically. Following Silverman's square-root-of-sample-size smoothing rule, I take $k = \sqrt{N + \bar N}$ over the combined pool. For each $i$, compute the Euclidean objective-space distance to every other individual, sort, read off the $k$-th value $\sigma_i^k$, and turn it into a contribution that is high for crowded (since I minimize) by inverting:
$$D(i) = \frac{1}{\sigma_i^k + 2}.$$
The $+2$ is not decoration. It keeps the denominator strictly positive even when $\sigma_i^k = 0$ (a duplicate in objective space), so $D$ never divides by zero; and more importantly it forces $D(i) \in (0,1)$, since $\sigma \ge 0$ gives a denominator $\ge 2$ and hence $D \le 1/2$. That strict bound is what makes the final fitness work:
$$F(i) = R(i) + D(i).$$
Because $R$ is a nonnegative integer (a sum of integer strengths) and $0 < D < 1$, density can reorder individuals that share the same integer $R$ but can never cross an integer level. Dominance-strength is the primary key, density the strictly secondary tie-breaker. In particular every nondominated individual has $R=0$ and so $F < 1$, while every dominated individual has $R \ge 1$ and so $F \ge 1$ — which means the single test $F(i) < 1$ extracts the current nondominated front exactly. Had I used $+1$, a nondominated duplicate ($\sigma=0$) would reach $D=1$ and land at $F=1$, colliding with the first dominated level and breaking that clean test; the $+2$ leaves the needed margin.

With fitness settled, environmental selection forms the fixed-size archive $\bar P_{t+1}$ from $P_t + \bar P_t$. Holding the archive at a constant size $\bar N$ (instead of letting it drift) keeps the mating-pool size and the elitism pressure steady. First copy every nondominated individual via the $F(i) < 1$ test. If the result has exactly $\bar N$ members, done. If it has fewer, rather than leave the archive underfull I top it up with the *best dominated* individuals — those with smallest $F$, the ones closest to breaking onto the front. If it has more, I must truncate, and this is where clustering went wrong. The requirement dictates the operator: remove from the densest region so the survivors stay uniform, and never remove a boundary solution. Both goals point at the nearest-neighbor distance. At each step remove the individual whose distance to its nearest neighbor is smallest — by definition the one in the densest spot — and because a boundary point of the front sits at an end and so has a comparatively large nearest-neighbor distance, it is never the minimum and is never removed. Boundary preservation is automatic, not a patch. Ties (two pairs equally close) are broken by looking further out — compare the second-nearest neighbor, then the third, and so on — so the removal rule is "drop the individual with the lexicographically smallest sorted neighbor-distance vector,"
$$i \le_d j \;:\Leftrightarrow\; \big(\forall\, 0<k<|\bar P_{t+1}|:\ \sigma_i^k = \sigma_j^k\big)\ \vee\ \big(\exists\, 0<k<|\bar P_{t+1}|:\ (\forall\, 0<l<k:\ \sigma_i^l = \sigma_j^l)\ \wedge\ \sigma_i^k < \sigma_j^k\big),$$
iterating until the archive is back to $\bar N$. Mating selection is then deliberately simple: the archive already *is* the converged, well-spread elite, so I draw parents from it by binary tournament with replacement (pick two, keep the lower $F$, undecided pairs broken at random), then apply recombination and mutation to produce $P_{t+1}$ and loop. The cost per generation is $O(M^2)$ for $S$ and $R$ with $M = N + \bar N$, and $O(M^2 \log M)$ on average for the density and truncation (governed by the sorting, $O(M^3)$ worst case for truncation).

The implementation makes two order-preserving shortcuts. It copies nondominated individuals using the raw test $R < 1$ (equivalent to $F < 1$ because $D < 1$) and computes density only when it must rank dominated candidates to fill an underfull archive or when truncating, and it keeps distances *squared* wherever only neighbor ordering matters, since squaring is monotone and changes neither which neighbor is nearest nor the lexicographic tie-break. It uses $K = \sqrt{M}$ over the combined pool.

```python
import math
import random
from copy import deepcopy

from deap import tools


def sel_spea2(individuals, k):
    """DEAP-style SPEA2 environmental selection: keep k individuals from a
    combined pool by strength/raw fitness, density fill, and truncation."""
    N = len(individuals)
    M = len(individuals[0].fitness.values)          # number of objectives
    K = math.sqrt(N)                                # k-th neighbour ~ sqrt(sample size)

    strength_fits = [0] * N                          # S(i) = # individuals i dominates
    fits = [0.0] * N                                 # -> R(i), then R(i) + D(i)
    dominating_inds = [[] for _ in range(N)]         # i's dominators

    # Strength S(i) and the dominator lists
    for i, ind_i in enumerate(individuals):
        for j, ind_j in enumerate(individuals[i + 1:], i + 1):
            if ind_i.fitness.dominates(ind_j.fitness):
                strength_fits[i] += 1
                dominating_inds[j].append(i)
            elif ind_j.fitness.dominates(ind_i.fitness):
                strength_fits[j] += 1
                dominating_inds[i].append(j)

    # Raw fitness R(i) = sum of i's dominators' strengths; R(i)=0 <=> nondominated
    for i in range(N):
        for d in dominating_inds[i]:
            fits[i] += strength_fits[d]

    # Copy all nondominated (R(i) < 1)
    chosen_indices = [i for i in range(N) if fits[i] < 1]

    if len(chosen_indices) < k:                      # underfull -> add density, fill best-dominated
        for i in range(N):
            distances = [0.0] * N
            for j in range(i + 1, N):
                dist = 0.0
                for m in range(M):
                    val = (individuals[i].fitness.values[m]
                           - individuals[j].fitness.values[m])
                    dist += val * val                # squared Euclidean, order-preserving
                distances[j] = dist
            kth_dist = _kth_smallest(distances, 0, N - 1, K)
            fits[i] += 1.0 / (kth_dist + 2.0)        # F(i) = R(i) + D(i), D < 1
        rest = [(fits[i], i) for i in range(N) if i not in chosen_indices]
        rest.sort()
        chosen_indices += [i for _, i in rest[:k - len(chosen_indices)]]

    elif len(chosen_indices) > k:                    # overfull -> truncate densest, keep boundary
        n = len(chosen_indices)
        dist = [[0.0] * n for _ in range(n)]
        order = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                d = 0.0
                for m in range(M):
                    val = (individuals[chosen_indices[i]].fitness.values[m]
                           - individuals[chosen_indices[j]].fitness.values[m])
                    d += val * val
                dist[i][j] = d
                dist[j][i] = d
            dist[i][i] = -1                           # self is never a neighbour
        for i in range(n):                           # sort each row: sigma_i^1, sigma_i^2, ...
            for j in range(1, n):
                m = j
                while m > 0 and dist[i][j] < dist[i][order[i][m - 1]]:
                    order[i][m] = order[i][m - 1]
                    m -= 1
                order[i][m] = j

        size = n
        to_remove = []
        while size > k:
            min_pos = 0                              # i <=_d j: lexicographically smallest = densest
            for i in range(1, n):
                for j in range(1, size):
                    di = dist[i][order[i][j]]
                    dm = dist[min_pos][order[min_pos][j]]
                    if di < dm:
                        min_pos = i
                        break
                    elif di > dm:
                        break
                    # tie: compare next-nearest neighbour
            for i in range(n):
                dist[i][min_pos] = float("inf")
                dist[min_pos][i] = float("inf")
                for j in range(1, size - 1):
                    if order[i][j] == min_pos:
                        order[i][j], order[i][j + 1] = order[i][j + 1], min_pos
            to_remove.append(min_pos)
            size -= 1
        for idx in reversed(sorted(to_remove)):
            del chosen_indices[idx]

    return [individuals[i] for i in chosen_indices]


def _kth_smallest(array, begin, end, i):
    if begin == end:
        return array[begin]
    q = _partition(array, begin, end)
    cnt = q - begin + 1
    if i < cnt:
        return _kth_smallest(array, begin, q, i)
    return _kth_smallest(array, q + 1, end, i - cnt)


def _partition(array, begin, end):
    p = random.randint(begin, end)
    array[begin], array[p] = array[p], array[begin]
    x = array[begin]
    i, j = begin - 1, end + 1
    while True:
        j -= 1
        while array[j] > x:
            j -= 1
        i += 1
        while array[i] < x:
            i += 1
        if i < j:
            array[i], array[j] = array[j], array[i]
        else:
            return j


class CustomMOEA:
    """SPEA2 strategy: strength-Pareto fitness + k-NN density tie-breaker,
    with a fixed-size, boundary-preserving elite archive."""

    def __init__(self, pop_size, n_obj, n_var, bounds,
                 cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
        self.archive = []

    def select(self, population, k):
        # Binary tournament on the archive (the elite front); dominance decides a
        # pair, an undecided pair is broken at random.
        pool = self.archive if self.archive else population
        chosen = []
        for _ in range(k):
            a, b = random.sample(pool, 2)
            if a.fitness.dominates(b.fitness):
                chosen.append(deepcopy(a))
            elif b.fitness.dominates(a.fitness):
                chosen.append(deepcopy(b))
            else:
                chosen.append(deepcopy(random.choice((a, b))))
        return chosen

    def vary(self, parents):
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds
        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 0.9:               # crossover probability
                tools.cxSimulatedBinaryBounded(
                    offspring[i], offspring[i + 1],
                    eta=self.cx_eta, low=lo, up=hi)
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values
        for ind in offspring:
            tools.mutPolynomialBounded(
                ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob)
            del ind.fitness.values
        return offspring

    def survive(self, population, offspring):
        combined = population + offspring
        survivors = sel_spea2(combined, self.pop_size)
        nd = [ind for ind in survivors
              if not any(o.fitness.dominates(ind.fitness) for o in survivors)]
        self.archive = [deepcopy(ind) for ind in nd[:self.pop_size]]
        return survivors

    def on_generation(self, gen, population):
        pass
```
