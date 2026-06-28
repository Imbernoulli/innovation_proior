Let me start from what actually goes wrong when I try to evolve a good approximation of a Pareto front, because the trouble is not in the search operators — crossover and mutation are fine — it is in *selection*. I have a population of decision vectors, each with a tuple of objective values, all to be minimized, and I want to end up with a set that hugs the true trade-off front and is spread evenly along it, extremes included. The only quality signal that is intrinsic to the problem is Pareto dominance: `a` dominates `b` if `a` is no worse in every objective and strictly better in at least one. So my first instinct is to rank by dominance and let selection do the rest. But dominance is a partial order, and that is the whole headache. Two individuals that each win on a different objective are incomparable — indifferent — and dominance says nothing about which to prefer. With two objectives that is occasional; with three or more it is the *normal* state, because the more axes you have the easier it is for any two points to each be better on some axis. So a fitness built purely on dominance leaves most of the population tied, and a tied population is a directionless one: selection cannot push it anywhere. That is the first wall, and it is structural, not a tuning issue.

Now, there is an existing strength-based idea I inherited that already tries to refine dominance into a real number, so let me reconstruct it carefully and see exactly where it breaks, because the fix will come from staring at the break. The idea: keep a regular population `P` and an external archive `P̄` of elite nondominated solutions. Give each archive member `i` a *strength* `S(i)` equal to the number of population members it dominates, normalized by `N+1` so it lands in `[0,1)`, and let that strength *be* the archive member's fitness. Then a population member `j` gets fitness `F(j) = 1 + Σ_{i ∈ P̄, i ≽ j} S(i)` — one plus the summed strengths of the archive members that cover it. Minimize `F`. This is clever in two ways. Archive members have fitness in `[0,1)` and population members have fitness `≥ 1`, so archive members are automatically fitter and elitism falls out for free. And a population member dominated by *strong* archive members (members that dominate a lot) is penalized more than one dominated by weak archive members, which is a first attempt at "how deep in the dominated region are you."

But let me actually compute what happens in a concrete configuration, because that is where I expect it to fall apart. Take an archive holding a single front member `a = (1,1)` (minimization), a normalizing `N = 4`, and four population members `p=(2,5)`, `q=(5,2)`, `r=(2,2)`, `s=(9,9)`. The archive member `a` dominates all four, so `S(a) = 4/(N+1) = 0.8`. Each of the four population members is dominated by `a` and by nothing else in the archive (there is nothing else in the archive), so each gets `F = 1 + S(a) = 1.8`. I write them out: `F(p) = F(q) = F(r) = F(s) = 1.8`. The fitnesses are *literally identical* — one distinct value across the whole population. And this is not a coincidence of the numbers: the sum `Σ_{i ∈ P̄, i ≽ j} S(i)` does not depend on `j` at all once the set of dominators is fixed, so any two members dominated by the same archive set must tie. The damning part: `r = (2,2)` dominates `p = (2,5)` outright, yet `F(r) = F(p)`. The fitness is blind to all structure among the dominated individuals, and with an archive of size one the entire population collapses to a single fitness level, where selection is pure coin-flipping. So the inherited method degrades to random search exactly when the archive is small — which is precisely early on. Wall.

Why did this happen? Because strength was assigned *only* to the archive, so the dominated individuals are ranked solely by who sits above them, never by what they themselves do. The information I am throwing away is each individual's *own* relation to the rest of the pool — how many *it* dominates. So let me not restrict strength to the archive. Let me give a strength to *every* individual in the union of archive and population:

  `S(i) = | { j ∈ P_t + P̄_t : i ≻ j } |`,

the number of individuals `i` dominates, where `+` is multiset union over population and archive and `≻` is dominance. Now every individual, archive or not, carries a number that says how much of the pool it sits above.

`S(i)` by itself is the wrong sign to use as fitness directly — a high `S` means `i` is good (it dominates a lot), so I would want to *minimize* something that is *low* for good individuals. And it does not yet encode "how dominated am I," which is what actually measures distance from the front. So let me build the quantity I want on top of the strengths. For each individual `i`, look at who dominates it, and sum *their* strengths:

  `R(i) = Σ_{ j ∈ P_t + P̄_t , j ≻ i } S(j)`.

Let me read this off a worked pool rather than trust the prose, because the whole claim is that `R` measures convergence and I should see it do so. Five points, minimization: `a=(1,4)`, `b=(2,2)`, `c=(4,1)`, `d=(3,3)`, `e=(5,5)`. The domination edges are `a≻e`, `b≻d`, `b≻e`, `c≻e`, `d≻e`. So the strengths are `S(a)=1` (dominates only `e`), `S(b)=2` (dominates `d` and `e`), `S(c)=1`, `S(d)=1` (dominates only `e`), `S(e)=0`. Now the raw fitnesses:

- `a`, `b`, `c` are dominated by nobody, so each has an empty dominator set: `R(a)=R(b)=R(c)=0`.
- `d` is dominated only by `b`, so `R(d) = S(b) = 2`.
- `e` is dominated by `a,b,c,d`, so `R(e) = S(a)+S(b)+S(c)+S(d) = 1+2+1+1 = 5`.

That output is exactly what I hoped for and now I can see *why*. `R = 0` falls out precisely for `{a,b,c}`, the three nondominated points — so `R(i) = 0` characterizes the current front cleanly, no separate test needed. The interior point `d` gets `2`, and the deepest point `e` gets `5`: `R` grows as you sink into the dominated region. And the weighting matters in the right way: `e` is below `b`, the strongest point (`S(b)=2`), and that strong dominator contributes `2` to `e`'s raw fitness, marking `e` as deep. If I had merely *counted* dominators instead, `e` would score `4` and `d` would score `1`, which orders them the same way here, but the strength weighting is what keeps the measure graded by *how far out toward the front* each dominator sits rather than by a flat headcount — a point buried under several front-anchoring solutions reads as deeper than one buried under a single mediocre one. So minimizing `R` pushes individuals toward the front, `R = 0` *is* the front, and the ranking among the dominated is by depth. This is fine-grained in a way the inherited scheme never was: it ranks every individual, not just the archive, by both what it dominates (through its dominators' strengths) and what dominates it.

Now the first wall reappears in a new form, and the worked example shows it plainly: `R(a)=R(b)=R(c)=0`. Raw fitness has measured convergence, but it gives me *no* signal to choose among the points that are already on the front — and those are exactly the points I keep, the ones whose spread I care about. If I stop here, the three front members are tied and I have no way to prefer a point in a sparse region over one crammed next to its neighbors. So I cannot avoid bringing in a second, independent signal: *density*. I need to know, for each individual, how crowded its neighborhood in objective space is, so I can prefer the lonely ones and spread the front out.

How should I measure density? Let me think about what is available and why I would reject the obvious options. One option is a hyper-grid: divide objective space into cells and count how many individuals share a cell. That is cheap but its verdict depends entirely on where I draw the grid lines and how big the cells are; two points a hair apart land in different cells and read as uncrowded, two points far apart in one big cell read as crowded. It is coarse and resolution-dependent. Another option, used by a contemporary method, is a per-objective crowding distance: for each objective sort the front and give each point the gap to its neighbors along that axis, summed over axes. That is reasonable but it is a sort-along-each-axis quantity, defined relative to axis-neighbors, not a genuine density of the point cloud, and it lives only inside a single front. I want something that is a true metric density — a function of actual distances between points — and gives me one scalar per individual that I can fold straight into the fitness.

The natural tool is the k-th nearest neighbor density estimator from statistics. The idea there: the density at a point is a *decreasing* function of the distance to its `k`-th nearest data point — if your `k`-th neighbor is close, you are in a dense region; if it is far, you are isolated. It adapts the bandwidth to the local density automatically, which is exactly what I want, because the front can be densely sampled in one region and sparse in another. The standard smoothing choice in that literature is to take `k` on the order of the square root of the sample size — large enough not to be fooled by a single nearby point, small enough to stay local. My sample is the combined pool of size `N + N̄`, so I set

  `k = sqrt( N + N̄ )`.

Concretely: for each individual `i`, compute the Euclidean distance in objective space to every other individual, sort those distances in increasing order, and read off the `k`-th one; call it `σ_i^k`. A small `σ_i^k` means crowded, a large one means isolated. I need to turn that into a fitness contribution that is *high for crowded* (since I minimize), so I take the inverse:

  `D(i) = 1 / ( σ_i^k + c )`,

for some additive constant `c`. The constant has to keep the denominator positive when `σ_i^k = 0` — which happens whenever an individual has a duplicate in objective space, its `k`-th neighbor sitting right on top of it — so `D` never divides by zero. Any `c > 0` does that. But the *value* of `c` is not free, because I am about to add density to raw fitness:

  `F(i) = R(i) + D(i)`,

and `R(i)` is a *nonnegative integer* (a sum of integer strengths). The design intent is dominance-strength first, density only as a tie-break: all nondominated individuals have `R = 0`, and I want every one of them to land at `F < 1`; every dominated individual has `R ≥ 1`, and I want none of them to slip below that boundary, so that the single test `F < 1` extracts the front. That forces a constraint on `c` — adding `D` must never push a front member up to `1`. The dangerous case is the worst case for `D`, which is the densest possible point: a duplicate with `σ = 0`, giving `D = 1/c`. So I need `1/c < 1`. Let me check the two natural candidates numerically. With `c = 1`: a nondominated duplicate gets `D = 1/1 = 1`, so `F = 0 + 1 = 1`, which fails the *strict* `F < 1` test — the duplicate collides with the first dominated raw-fitness level. With `c = 2`: the same duplicate gets `D = 1/2 = 0.5`, so `F = 0 + 0.5 = 0.5 < 1` — safe. And in general with `c = 2` the denominator is `≥ 2` for any `σ ≥ 0`, so `0 < D ≤ 1/2 < 1` always. That `0 < D < 1` is the load-bearing property: adding `D` can reorder individuals that share the same integer `R`, but it can never cross an integer raw-fitness level, so `F(i) = R(i) + D(i)` is a layered key — rank by how dominated you are, break ties among the equally dominated (and especially among the `R = 0` front) by how isolated you are. The `+2` is the smallest clean choice that gives that strict margin.

Let me pause on the cost, because I will be doing this every generation. Computing all pairwise objective-space distances and sorting them to find each `σ_i^k` is the dominant term, `O(M^2 log M)` with `M = N + N̄`. The strengths and raw fitness are `O(M^2)` — for every pair I check dominance once and bump the dominator's strength, then for every individual I sum its dominators' strengths. So fitness assignment is `O(M^2 log M)` overall, dominated by the density sort. That is affordable for the population sizes in play.

So fitness is settled. Now the archive — environmental selection — which is the other half and where the inherited method's diversity reduction failed. I want a *fixed*-size archive `P̄` of size `N̄` (the inherited one let the size drift, which makes selection pressure unpredictable; a constant size keeps the number of mating candidates fixed and the elitism steady). Each generation I form `P̄_{t+1}` from the combined pool `P_t + P̄_t`. The first step follows from the fitness I just built: copy every nondominated individual, i.e. every one with `F(i) < 1`. I want to be sure `F(i) < 1` and `R(i) = 0` really are the same set, since the worked pool is what convinces me — there `R = 0` for `{a,b,c}` and `F = 0 + D < 1` for each since `D < 1`; conversely if `i` is dominated, `R(i) ≥ 1` so `F(i) ≥ 1`. The single test `F(i) < 1` therefore extracts the current nondominated front, which is the payoff of having pinned `D < 1`. Set

  `P̄_{t+1} = { i ∈ P_t + P̄_t : F(i) < 1 }`.

Three cases. If `|P̄_{t+1}| = N̄` exactly, done. If `|P̄_{t+1}| < N̄`, the front is too small to fill the archive — and rather than leave it underfull (which would shrink the mating pool), I top it up with the *best dominated* individuals: sort the remaining pool (those with `F ≥ 1`) by `F` ascending and take the first `N̄ − |P̄_{t+1}|`. They are the least-bad of the dominated set, the ones closest to breaking onto the front. If `|P̄_{t+1}| > N̄`, the front overflows and I must *truncate* — and this is exactly where the inherited clustering went wrong by discarding the extremes.

Let me reason out the truncation from the requirement, because the requirement dictates the operator. I want to remove individuals one at a time until I am down to `N̄`, and I want each removal to (a) come from the most crowded region, so the remaining set stays uniform, and (b) avoid removing a boundary/extreme solution, since the extremes are the spread I most want to keep. Both goals seem to point at the same quantity I already compute: the distance to the nearest neighbor. The individual to remove at each step is the one with the *smallest* distance to its nearest neighbor — by definition the one in the densest spot. So let me trace this on a small front and watch whether the extremes really do survive, rather than just assert it. Take five front points: `L=(0,10)` and `R=(10,0)` at the two ends, and a crowded cluster in the middle, `m1=(4,6)`, `m2=(4.5,5.5)`, `m3=(5,5)`, with `m1`–`m2` very close. Truncate to three. Using squared distances, each point's sorted neighbor-distance vector is:

- `m2`: nearest is `m1` at `0.5`, then `m3` at `0.5`, then far ends — vector `(0.5, 0.5, 40.5, 60.5)`.
- `m1`: nearest `m2` at `0.5`, then `m3` at `2.0`, … — vector `(0.5, 2.0, …)`.
- `m3`: nearest `m2` at `0.5`, then `m1` at `2.0`, … — vector `(0.5, 2.0, …)`.
- `L` and `R`: their nearest neighbor is a middle point, tens of units away (e.g. `L`–`m1` is `36`), so their vectors start at `36`, `40`, …

The smallest first entry is `0.5`, shared by `m1`, `m2`, `m3`; the lexicographically smallest *vector* among those is `m2`'s, because its second entry `0.5` beats the `2.0` of the others. Remove `m2`. Recompute on `{L, m1, m3, R}`: now `m1` and `m3` are each other's nearest neighbor at `2.0`, the smallest in the set, so `m1` goes next. Survivors: `{L, m3, R}`. Both boundary points survived; the two removals came from the crowded middle. The mechanism is now clear from the trace: a boundary point sits at an end of the front, so its nearest neighbor is comparatively far, its vector starts large, and it is not the lexicographic minimum.

I should be honest about how strong that property is, though, because "boundary points are never removed" is the kind of claim I should test at its edge before I believe it. Construct an adversarial front: put *two* points near the left extreme, `L1=(0,10)` and `L2=(0.1,9.9)`, a single middle point `M=(5,5)`, and a right extreme `R=(10,0)`; truncate to three. Now `L1` and `L2` are each other's nearest neighbor at distance `≈0.02`, far smaller than anything else, so the lexicographic minimum is `L2` (or `L1`) — and a *near-extreme* point gets removed. So the honest statement is not "the boundary is provably never touched"; it is that the operator always removes whatever sits in the densest spot, and an *isolated* extreme is by construction sparse and therefore safe — which is the ordinary situation, since a well-spread front has its ends far from their neighbors. The extremes are preserved exactly when they are genuinely extreme; two near-coincident extremes can lose one of themselves, and that is the correct behavior, not a bug. That is a real improvement over clustering, which merges groups to centroids and so can drop a true extreme even when it is isolated.

The lexicographic rule also needs a tie-break written down precisely, because "smallest distance to nearest neighbor" can tie: two pairs of points equally close. If two individuals tie on their nearest neighbor, compare their *second* nearest, then third, and so on — exactly what I did by hand to separate `m2` from `m1`/`m3` above. Formally, define `i ≤_d j` (read: `i` is at least as crowded as `j`, so `i` is the one to drop) by

  `i ≤_d j  :⇔  [ ∀ 0 < k < |P̄_{t+1}| : σ_i^k = σ_j^k ]  OR  [ ∃ 0 < k < |P̄_{t+1}| : ( ∀ 0 < l < k : σ_i^l = σ_j^l ) ∧ σ_i^k < σ_j^k ]`,

where `σ_i^k` is `i`'s distance to its `k`-th nearest neighbor *within the current `P̄_{t+1}`*. In words: `i` loses if its sorted distance vector is lexicographically smaller than `j`'s; if the vectors are equal all the way (true duplicates) it does not matter which goes. At each iteration remove an `i` that satisfies `i ≤_d j` for all `j`, recompute, repeat until `|P̄_{t+1}| = N̄`. The cost: building each individual's sorted distance list is `O(M^2)`, and the worst case for the whole truncation is `O(M^3)` because in principle every removal could force re-examining everything, but since individuals usually already differ at the second or third nearest neighbor, the tie-break rarely goes deep and the average is `O(M^2 log M)`, governed again by the sorting.

That completes environmental selection. Mating selection is now simple and I should not overcomplicate it: the archive `P̄_{t+1}` already *is* the converged, well-spread elite, so I draw parents from it by binary tournament with replacement — pick two archive members at random, keep the one with the lower `F`, repeat to fill the mating pool. Restricting mating to the archive (rather than the whole population) focuses reproduction on the current best front, which is what the archive was built to represent. Then apply recombination and mutation to the mating pool to produce the next population `P_{t+1}`, increment `t`, and loop back to fitness assignment.

Let me lay out the full loop so I can see it whole. Start with a random population `P_0` and an empty archive `P̄_0`. Each generation: compute `S`, `R`, `D`, and `F = R + D` over `P_t + P̄_t`; environmental-select `P̄_{t+1}` (copy `F < 1`, then truncate-or-fill to exactly `N̄`); if the stopping criterion holds, return the nondominated members of `P̄_{t+1}` as the answer; otherwise binary-tournament on `P̄_{t+1}` for the mating pool, vary it into `P_{t+1}`, and repeat. Two selection mechanisms, both reading the same fine-grained fitness, with a fixed-size boundary-preserving archive.

Now I want to land this on the actual environmental selector I would run, and I will be precise about how the abstract formulas become operations, because the implementation makes two order-preserving choices. I have a combined pool, and I need, per individual, its strength and then its raw fitness. The selector can copy nondominated individuals using `R < 1` before computing density, because the strict `D < 1` margin makes `R = 0`, `F < 1`, and nondominance the same test. Density is only needed when the archive is underfull and I must rank dominated candidates by `F`; when the archive is overfull, the lexicographic truncation operator uses sorted neighbor distances directly. For those neighbor comparisons I can keep squared Euclidean distances and skip the square roots, because squaring is monotone and does not change which neighbor is nearest or how the lexicographic tie-break orders two distance lists — which is why the worked truncation above used squared distances throughout and still ordered the removals correctly. For the smoothing index, the implementation uses `K = sqrt(M)` where `M` is the combined-pool size, the same square-root sample-size rule as `k = sqrt(N + N̄)`.

Here is the selector — strength, raw fitness, the nondominated copy, then the underflow fill or overflow truncation — written the way it actually runs:

```python
import math
import random
from copy import deepcopy

from deap import tools


def sel_spea2(individuals, k):
    """DEAP-style SPEA2 environmental selection: keep k individuals from a
    combined pool by strength/raw fitness, density fill, and truncation."""
    N = len(individuals)
    M = len(individuals[0].fitness.values)        # number of objectives
    K = math.sqrt(N)                              # k-th neighbour, ~ sqrt(sample size)

    strength_fits = [0] * N                       # S(i) = # individuals i dominates
    fits = [0.0] * N                              # becomes R(i), then R(i)+D(i)
    dominating_inds = [[] for _ in range(N)]      # for each i, the list of its dominators

    # S(i): count whom i dominates; record each i's dominators for the R step
    for i, ind_i in enumerate(individuals):
        for j, ind_j in enumerate(individuals[i + 1:], i + 1):
            if ind_i.fitness.dominates(ind_j.fitness):
                strength_fits[i] += 1
                dominating_inds[j].append(i)
            elif ind_j.fitness.dominates(ind_i.fitness):
                strength_fits[j] += 1
                dominating_inds[i].append(j)

    # R(i) = sum of the strengths of i's dominators; R(i)=0  <=>  i nondominated
    for i in range(N):
        for d in dominating_inds[i]:
            fits[i] += strength_fits[d]

    # Copy all nondominated individuals: those with raw fitness R(i) < 1 (i.e. == 0)
    chosen_indices = [i for i in range(N) if fits[i] < 1]

    if len(chosen_indices) < k:                   # archive underfull -> fill with best-dominated
        # Add density D(i) = 1/(sigma_i^K + 2) using squared distances.
        for i in range(N):
            distances = [0.0] * N
            for j in range(i + 1, N):
                dist = 0.0
                for m in range(M):
                    val = (individuals[i].fitness.values[m]
                           - individuals[j].fitness.values[m])
                    dist += val * val             # squared Euclidean, order-preserving
                distances[j] = dist
            kth_dist = _kth_smallest(distances, 0, N - 1, K)   # sigma_i^K, squared
            density = 1.0 / (kth_dist + 2.0)      # D(i): the "+2" keeps D in (0,1)
            fits[i] += density                    # F(i) = R(i) + D(i)
        # take the best-dominated by F to top up the archive to size k
        rest = [(fits[i], i) for i in range(N) if i not in chosen_indices]
        rest.sort()
        chosen_indices += [i for _, i in rest[:k - len(chosen_indices)]]

    elif len(chosen_indices) > k:                 # archive overfull -> truncate densest, keep boundary
        n = len(chosen_indices)
        dist = [[0.0] * n for _ in range(n)]      # full pairwise (squared) distance matrix
        order = [[0] * n for _ in range(n)]       # per-row indices sorted by increasing distance
        for i in range(n):
            for j in range(i + 1, n):
                d = 0.0
                for m in range(M):
                    val = (individuals[chosen_indices[i]].fitness.values[m]
                           - individuals[chosen_indices[j]].fitness.values[m])
                    d += val * val
                dist[i][j] = d
                dist[j][i] = d
            dist[i][i] = -1                        # self never counts as a neighbour
        for i in range(n):                        # sort each row -> sigma_i^1, sigma_i^2, ...
            for j in range(1, n):
                m = j
                while m > 0 and dist[i][j] < dist[i][order[i][m - 1]]:
                    order[i][m] = order[i][m - 1]
                    m -= 1
                order[i][m] = j

        size = n
        to_remove = []
        while size > k:
            # i <=_d j: lexicographically smallest sorted-distance vector = most crowded
            min_pos = 0
            for i in range(1, n):
                for j in range(1, size):
                    di = dist[i][order[i][j]]
                    dm = dist[min_pos][order[min_pos][j]]
                    if di < dm:                   # i is closer at this rank -> i more crowded
                        min_pos = i
                        break
                    elif di > dm:                 # i is farther -> keep current min, stop tie-break
                        break
                    # equal -> compare the next nearest neighbour (the sigma^l tie-break)
            for i in range(n):                    # drop min_pos from everyone's neighbour lists
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
    """ith smallest element without full sort (randomized-select, expected O(n))."""
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
```

That selector is the whole contribution; the rest is the strategy wrapper that the surrounding evolutionary loop calls each generation — mating by binary tournament on the archive, the standard bounded variation, and survival via the selector above, after which I refresh the archive to the surviving nondominated front so the next generation's tournaments draw from it:

```python
class CustomMOEA:
    """Strength-Pareto strategy: fine-grained dominance-strength fitness with a
    k-NN density tie-breaker, plus a fixed-size, boundary-preserving elite archive."""

    def __init__(self, pop_size, n_obj, n_var, bounds,
                 cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
        self.archive = []                          # the external elite set P_bar

    def select(self, population, k):
        # Mating selection: binary tournament with replacement, on the ARCHIVE
        # (the elite front) when it exists, else the population. Dominance decides
        # a pair; an undecided (mutually nondominated) pair is broken at random.
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
            if random.random() < 0.9:              # crossover probability
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
        # Environmental selection over the combined pool -> fixed-size survivors
        survivors = sel_spea2(combined, self.pop_size)
        # Refresh the archive to the surviving nondominated front
        nd = [ind for ind in survivors
              if not any(o.fitness.dominates(ind.fitness) for o in survivors)]
        self.archive = [deepcopy(ind) for ind in nd[:self.pop_size]]
        return survivors

    def on_generation(self, gen, population):
        pass
```

Tracing the causal chain back through the whole thing: I started stuck because Pareto dominance is a partial order and leaves most of a multi-objective population mutually indifferent, so any fitness built only on dominance ties everyone. The inherited strength scheme tried to refine dominance but assigned strength only to the archive, and the worked case showed every population member dominated by the same archive set tying at `1.8` — even one that dominates another — collapsing to a single fitness level when the archive was small. Giving a strength `S(i) = #(i dominates)` to *every* individual and defining raw fitness `R(i) = Σ` of the strengths of `i`'s dominators fixed that: on the five-point pool, `R = 0` fell out exactly for the front `{a,b,c}` while the interior point scored `2` and the deepest `5`, so `R` grades depth into the dominated region, weighted by how strong the dominators are. Raw fitness still tied all front members at `R = 0`, so I added a density tie-breaker from k-th nearest neighbor density estimation, `D(i) = 1/(σ_i^k + 2)` with `k = sqrt(N + N̄)`, where the `+2` was forced by the numeric check that `+1` lets a duplicate reach `D = 1` and break the strict `F < 1` test while `+2` holds it at `0.5`. The archive is held at a fixed size: copy all `F < 1`, fill underflow with the best dominated by `F`, and on overflow truncate by iteratively removing the lexicographically-densest individual — traced on a five-point front, this removed the crowded middle and kept both ends, and the adversarial check pinned down the honest scope: isolated extremes are preserved because they are the sparsest, while two near-coincident extremes can correctly lose one. Mating is binary tournament with replacement on that archive. The implementation computes strengths and raw fitness in `O(M^2)` pairwise passes, uses squared distances wherever only neighbor ordering matters, and drops into the standard evolutionary loop as the three hooks the loop calls each generation: `select` (binary tournament on the archive), `vary` (bounded SBX plus polynomial mutation), and `survive` (the environmental-selection core above, after which the archive is refreshed to the surviving nondominated front).
