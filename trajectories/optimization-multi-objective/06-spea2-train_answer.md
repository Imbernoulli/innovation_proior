NSGA-III did what I climbed to it for and left one residual weakness. The pivotal prediction held: on DTLZ2 it beat NSGA-II on HV (2.7896 vs 2.7407) and IGD (0.0521 vs 0.0647), and on the 2-objective ZDTs the imposed grid out-spread NSGA-II decisively (ZDT1 spread 0.287 against 0.343). But on DTLZ2 spread it is only 0.621 — barely better than NSGA-II's 0.653 and worse than RVEA's 0.383. NSGA-III converges beautifully but its spread is only as even as the *fixed reference grid* it optimizes coverage of, and a fixed grid is only a proxy for the actual point-cloud density the task scores with the Spread metric. Where the grid's directions do not match where the front actually wants points, the spread comes out merely adequate. Every method so far measures diversity either against a fixed external structure (decomposition's weights, NSGA-III's reference lines) or with a geometry-blind estimate (NSGA-II's cuboid); none measures it as a *true metric density of the actual point cloud* — the distance from each solution to its real nearest neighbors in objective space — and uses that both to grade the front and to decide who to drop.

I propose **SPEA2**: strength-Pareto fitness with k-NN density. The trouble with dominance is that it is a partial order — two points each better on a different objective are incomparable, so a fitness built purely on dominance ties most of the population. The strength-Pareto fix refines dominance into a real number, and I want the fine-grained version. Give every individual in the combined pool a strength $S(i)$ = the number of individuals it dominates. Then define raw fitness

$$R(i) = \sum_{j\,:\,j \succ i} S(j),$$

the sum of the strengths of $i$'s dominators. Read what this does: if nobody dominates $i$, the sum is empty and $R(i) = 0$ — so $R = 0$ cleanly marks the nondominated front; if $i$ is dominated, $R(i)$ grows, and it grows *more* when $i$ is dominated by individuals that themselves dominate many others, because being below a point that sits far out toward the front means $i$ is deep in the interior. So $R$ is a graded "how far back am I," not a flat count, and minimizing it pushes the population toward the front. This is finer than NSGA-II's integer front-rank: it ranks by both what you dominate (through your dominators' strengths) and what dominates you.

But raw fitness ties every front member at $R = 0$, which is exactly the set whose spread I care about, so I bring in an independent density signal — and here is the move that beats NSGA-III. Rather than a grid, measure density with the $k$-th nearest-neighbor estimator from statistics: density at a point is a decreasing function of the distance to its $k$-th nearest *actual* data point, with $k = \sqrt{N + N_{\text{archive}}}$ — large enough not to be fooled by one nearby point, small enough to stay local. For each individual compute the Euclidean distance in objective space to every other individual, sort, read off the $k$-th, call it $\sigma_i^k$, and set

$$D(i) = \frac{1}{\sigma_i^k + 2},$$

high for crowded, low for isolated. The $+2$ is load-bearing: with $\sigma \ge 0$ the denominator is $\ge 2$, so $D \in (0,1)$, which means adding it to the integer $R$ — the final fitness $F(i) = R(i) + D(i)$ — can reorder individuals sharing the same $R$ but can never cross an integer boundary. So $F$ is a layered key: dominance-strength first, real-point-cloud density second, with $F < 1$ exactly the nondominance test. This density is the thing NSGA-III lacked — it is a function of *actual distances between solutions*, the same quantity the Spread metric scores, not a proxy for it.

The archive — environmental selection — is the other half. Hold it at a fixed size $N_{\text{archive}}$ (a constant keeps selection pressure steady). Each generation, form it from the combined pool: copy every nondominated individual ($F < 1$); if that underfills, top up with the best-dominated by $F$ ascending (the ones closest to breaking onto the front); if it overflows, *truncate* by iteratively removing the individual with the smallest distance to its nearest neighbor — by definition the densest point — breaking ties lexicographically on the next-nearest, then the next, and so on. This truncation has a property the whole ladder has been chasing: a boundary/extreme solution sits at an end of the front, so it has a *large* nearest-neighbor distance, so it is never the minimum and never removed. Boundary preservation is automatic from "remove the closest-pair member," not patched on — and keeping the extremes is precisely what makes the spread metric (which penalizes a front that fails to reach its ends) come out low.

The task's landing is close to the canonical wrapper. The harness delegates `survive` entirely to DEAP's `tools.selSPEA2(combined, pop_size)` — exactly the strength/raw-fitness, k-NN density, lexicographic boundary-preserving truncation I just derived — and then refreshes the archive to the nondominated members of the survivors (`get_nondominated(selected)`, capped at `pop_size`). Mating selection (`select`) is a binary tournament *on the archive* when it exists (else the population), decided by **dominance** directly — pick two archive members, keep the dominator, break a mutually-nondominated pair at random — focusing reproduction on the current elite front; this is dominance, not the full $F$-fitness tournament, and because the archive is already the converged well-spread elite, dominance is enough pressure. Variation is the shared SBX ($\eta_c = 20$, probability **0.9** — back to NSGA-II's value, not the 1.0 the reference-point methods used) plus polynomial mutation ($\eta_m = 20$, $p_m = 1/n$). One cost note that explains a number I will see: the k-NN density and lexicographic truncation compute full pairwise distance matrices over the combined pool every generation, $O(M^2\log M)$ with $M = 2N$ — markedly heavier than the other rungs' sorts, so this is the slowest rung in wall-clock. That is the price of measuring true point-cloud density rather than reading a fixed grid.

I expect a clean win: on spread, strength-Pareto should clearly beat NSGA-III everywhere — ZDT1 spread dropping into the 0.10–0.15 band and DTLZ2 spread down near 0.10, a step-change, not an increment — because the k-NN truncation optimizes real inter-point distances while preserving the extremes by construction; and on convergence a wash to slight win, HV and IGD around NSGA-III's level, because the method is just as elitist and its fine-grained $R$ converges at least as hard. The crucial thing is that the spread win must come *without* a convergence regression; if it did not, this rung would merely re-trade convergence for diversity like AGE-MOEA did. The open question it leaves is whether one can keep this true-density spread while *directly* maximizing the headline HV metric rather than approximating it through dominance-strength — the gap between optimizing a diversity surrogate and optimizing the scored indicator itself.

```python
# EDITABLE region of deap/custom_moea.py (lines 297-441) — step 6: SPEA2
class CustomMOEA:
    """SPEA2: Strength Pareto Evolutionary Algorithm 2."""

    def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var
        self.archive = []

    def select(self, population, k):
        """Binary tournament selection using SPEA2 fitness from archive."""
        # Use archive for selection if available, otherwise population
        pool = self.archive if self.archive else population
        # Binary tournament on dominance
        selected = []
        for _ in range(k):
            i1, i2 = random.sample(range(len(pool)), 2)
            a, b = pool[i1], pool[i2]
            if a.fitness.dominates(b.fitness):
                selected.append(deepcopy(a))
            elif b.fitness.dominates(a.fitness):
                selected.append(deepcopy(b))
            else:
                selected.append(deepcopy(random.choice([a, b])))
        return selected

    def vary(self, parents):
        """SBX crossover + polynomial mutation."""
        offspring = [deepcopy(ind) for ind in parents]
        lo, hi = self.bounds

        for i in range(0, len(offspring) - 1, 2):
            if random.random() < 0.9:
                tools.cxSimulatedBinaryBounded(
                    offspring[i], offspring[i + 1],
                    eta=self.cx_eta, low=lo, up=hi,
                )
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        for ind in offspring:
            if random.random() < 1.0:
                tools.mutPolynomialBounded(
                    ind, eta=self.mut_eta, low=lo, up=hi, indpb=self.mut_prob,
                )
                del ind.fitness.values

        return offspring

    def survive(self, population, offspring):
        """SPEA2 survival: strength fitness + kNN density truncation."""
        combined = population + offspring

        # Use DEAP's built-in SPEA2 selection
        selected = tools.selSPEA2(combined, self.pop_size)

        # Update archive with non-dominated solutions
        nd = get_nondominated(selected)
        self.archive = [deepcopy(ind) for ind in nd[:self.pop_size]]

        return selected

    def on_generation(self, gen, population):
        pass
```
