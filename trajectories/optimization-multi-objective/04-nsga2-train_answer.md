RVEA vindicated the convergence repair and exposed what reference-vector selection trades away. On DTLZ2 it did what I bet it would: HV recovered to 2.7886 — the best 3-objective convergence on the ladder so far — with IGD 0.0548 and a tight spread of 0.383 across all three seeds. But the cost showed up where I warned the un-normalized APD and the hard-wired horizon would bite: on DTLZ1, seed 42 blew up (spread 3.45, IGD 0.110, HV 0.944) while the other two seeds were fine, dragging the mean to spread 1.43. RVEA's wound is its *instability and parameter sensitivity*: a hard-wired $t_{\max} = 400$ that mistimes the ramp on shorter runs, an un-normalized angle penalty that lets a niche poach and blow up a seed, and an adaptation schedule that can churn the niches. Every one of those is a knob or a fixed scale I had to guess. So I want the convergence-and-diversity balance back, but with *no fragile diversity parameter at all* — no $\theta$, no $\alpha$ ramp, no horizon, no adaptation schedule.

I propose **NSGA-II**: elitist survival with parameter-free diversity. It closes the three defects of the late-90s layered-ranking-plus-fitness-sharing template together, and the third closure is the one that matters here. Take elitism first, because it reframes the rest. SPEA's route is an external archive — a second data structure with its own sizing, pruning, clustering — but I do not need one to stop good parents from disappearing. Cheaper: don't throw the parents away. Merge parents $P_t$ and offspring $Q_t$ into one combined pool $R_t = P_t \cup Q_t$ of size $2N$ and select survivors from that. Every parent competes for a survivor slot on equal footing with every child, whole better-ranked fronts carry forward when they fit, and when a front overflows I choose its least-crowded representatives. Elitism becomes a property of the survival pool, with no archive — but it doubles the size of the thing I sort, so I must fix the cost.

The naive non-dominated sort is $O(M N^3)$ because to find front 2 it re-examines dominations already computed for front 1, peeling $N$ times. Dominance relations between pairs are fixed, so I compute each pair once. In one $O(M N^2)$ pass I record, for each solution $p$, a domination count $n_p$ (how many dominate it) and the set $S_p$ it dominates. Then peeling is almost free: every $p$ with $n_p = 0$ is in front 1; to get the next front, walk each $p$ in the current front, decrement $n_q$ for each $q \in S_p$, and any $q$ hitting zero joins the next front. Each solution sits in exactly one front, so the outer loop runs $N$ times total and the inner decrement at most $N(N-1)$ times — $O(N^2)$ peeling, $O(M N^2)$ overall, trading the cubic for storage of $S_p$. The harness gets this through `tools.sortNondominated`, DEAP's fast non-dominated sort.

The heart of why NSGA-II beats RVEA on stability is the **crowding distance** — diversity with *no* tuning parameter. Fitness sharing asks "how crowded is your neighborhood?" and answers with a kernel of radius $\sigma_{\text{share}}$; RVEA asks the analogous question and answers with an angle penalty scaled by a $(t/t_{\max})^\alpha$ ramp against a guessed horizon. Both have a knob. I can estimate local crowding with *no* radius and *no* horizon by reading it straight off the neighbors along the front. Within a single front, sort by each objective in turn; for an interior solution, the distance between its two flanking neighbors along that objective, normalized by the front's extent in that objective,

$$\text{cd}(i) \mathrel{+}= \frac{f_m[i+1] - f_m[i-1]}{f_m^{\max} - f_m^{\min}},$$

summed over objectives $m$, is the width of the slab the point occupies. A large crowding distance means a sparse region — keep it; a small one means a dense cluster — expendable. The front's own spacing sets the scale automatically: no radius, no ramp, no horizon. And the two boundary solutions of each objective — the extremes I most want to keep so the population reaches the ends of the front — get infinite distance and always survive truncation. This is the knob-free diversity RVEA's APD lacked, and it is why RVEA threw a blow-up seed on DTLZ1 while NSGA-II will not: there is no horizon to mistime and no un-normalized angle to let a niche poach. The harness's `compute_crowding_distance` does exactly this — infinite extremes, summed normalized neighbor gaps.

One lexicographic comparison fuses the two readings: prefer the lower non-domination rank (convergence first, full stop), break ties by the larger crowding distance (diversity among equals). It is a strict priority, never a weighted sum with a coefficient I would have to tune — the same anti-knob spirit. The harness uses it in two places. In `select`, mating is `tools.selTournamentDCD` — a binary tournament on dominance then crowding, after crowding has been assigned per front — so parents favor near-front, sparse-region solutions. In `survive`, environmental selection enforces full rank-then-crowding: take whole fronts in rank order while they fit, then sort the critical front by crowding distance descending and take the top slots to reach exactly $N$. Variation stays the shared real-coded pair at the field-standard settings: SBX ($\eta_c = 20$) at crossover probability **0.9**, polynomial mutation ($\eta_m = 20$) at $p_m = 1/n$. The 0.9 is worth noting against the prior rungs — RVEA and the reference-point methods crossover at probability 1.0, while NSGA-II leaves one in ten pairs uncrossed, keeping crossover the main engine while preserving some parents intact. These are held fixed so the difference from the other rungs is the selection strategy, not the operators. (`selTournamentDCD` requires the population size divisible by four; the task's sizes 100 and 120 satisfy this.)

I expect NSGA-II to show tight per-seed numbers on every problem with no DTLZ1 blow-up seed — the parameter-free crowding has no horizon to mistime — and to *beat* RVEA on the 2-objective ZDTs on both HV and spread, where the gentle ramp under-pressured RVEA. The one place I am least sure it wins is DTLZ2 convergence: its crowding distance is the $L_1$ cuboid that AGE-MOEA's whole premise said mis-reads a curved 3-objective front, so DTLZ2 HV should land *below* RVEA's 2.7886, plausibly around the decomposition level. That is not a contradiction of the ranking — NSGA-II sits above RVEA on *aggregate* because it wins the 2-objective problems and the stability across seeds, even while conceding 3-objective convergence — and it tells the rungs above exactly what to do: replace the cuboid crowding with a diversity device that reads the 3-objective geometry correctly.

```python
# EDITABLE region of deap/custom_moea.py (lines 297-441) — step 4: NSGA-II
class CustomMOEA:
    """NSGA-II: Non-dominated Sorting Genetic Algorithm II."""

    def __init__(self, pop_size, n_obj, n_var, bounds, cx_eta=20.0, mut_eta=20.0, mut_prob=None):
        self.pop_size = pop_size
        self.n_obj = n_obj
        self.n_var = n_var
        self.bounds = bounds
        self.cx_eta = cx_eta
        self.mut_eta = mut_eta
        self.mut_prob = mut_prob if mut_prob is not None else 1.0 / n_var

    def select(self, population, k):
        """Binary tournament selection with crowding distance."""
        fronts = tools.sortNondominated(population, len(population), first_front_only=False)
        for front in fronts:
            compute_crowding_distance(front)
        return tools.selTournamentDCD(population, k)

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
        """NSGA-II survival: non-dominated sorting + crowding distance."""
        combined = population + offspring
        fronts = tools.sortNondominated(combined, self.pop_size, first_front_only=False)

        next_gen = []
        for front in fronts:
            if len(next_gen) + len(front) <= self.pop_size:
                next_gen.extend(front)
            else:
                remaining = self.pop_size - len(next_gen)
                compute_crowding_distance(front)
                front.sort(key=lambda x: x.fitness.crowding_dist, reverse=True)
                next_gen.extend(front[:remaining])
                break

        return next_gen

    def on_generation(self, gen, population):
        pass
```
