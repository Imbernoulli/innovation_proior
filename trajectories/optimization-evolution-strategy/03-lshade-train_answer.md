The GA landed almost exactly on the three-way bet. Rastrigin dropped to a mean of 8.05 at 30D and 113.8 at 100D — a decisive win over DE's 256.8 and 925.8, the per-coordinate polynomial mutation doing on the separable egg-carton what a coupled difference-vector move could not. But the two regressions I feared are right there: Rosenbrock rose to a mean of 138.3 (far worse than DE's seed-42 32.2) because axis-aligned SBX and polynomial mutation have no notion of the diagonal valley, and Ackley sat at 1.12, nowhere near the basin DE's anisotropic differences should have solved. And the convergence columns show every method still improving at the wall — they ran out of budget, not converged. So I now hold a contradictory lesson: DE's self-scaling anisotropic difference vector is the right *mutation* but its single global $F$/$CR$ is catastrophic on Rastrigin and its fixed `rand` base is slow; the GA's per-coordinate locality fixes Rastrigin but throws away the anisotropy that owns the valley. No fixed setting of either is right on all four landscapes. The parameters must learn themselves from what is working on *this* problem, right now.

I propose L-SHADE — success-history adaptive DE with linear population reduction — and the core insight is that the signal for learning $F$ and $CR$ is already sitting in the selection step: every generation some trials beat their parents and some do not, and the winners' $(F_i, CR_i)$ are evidence of what works on this landscape. The mutation strategy comes first. Plain DE/rand/1 is diversity-preserving but slow; the greedy `current-to-best/1` is fast on a bowl but funnels the whole population into the incumbent's basin — the Rastrigin death. The fix is to pull toward a *random one of the top few* rather than the single best:

$$v_i = x_i + F_i\,(x_{p\text{best}} - x_i) + F_i\,(x_{r_1} - x_{r_2}),$$

with $x_{p\text{best}}$ drawn from the top $N\cdot p$ individuals. Here $p$ is a greediness dial — small $p$ is aggressive, a moderate $p$ spreads the attraction over several good basins so the population does not all rush one hole — and I sample $p_i$ per individual from $[p_\text{min}, p_\text{max}]$ with $p_\text{min}=2/N_\text{init}$, $p_\text{max}=0.2$, widening the spread of guides for free. The difference term gets a second gift: if both $x_{r_1}$ and $x_{r_2}$ come from the live population, the difference shrinks toward zero as the population converges, losing diversification exactly when I might be stuck. So I keep the parents that just *lost* selection in an external archive $A$ — they encode regions the search recently chose to leave — and draw $x_{r_2}$ from $P \cup A$ ($r_1$ from $P$ only), letting the difference reach back to a recently-abandoned region without enlarging the live population. The archive is capped at the current population size, random members dropped on overflow: a reservoir of recent history, not a museum that biases the search.

The adaptation is where the design earns its keep, and the asymmetry between $F$ and $CR$ is deliberate. $CR$ is a probability that should settle near a stable value, so I sample $CR_i = \mathcal{N}(M_{CR}[r], 0.1)$ clamped to $[0,1]$ — a tight Normal keeps it near its learned center. $F$ controls mutation magnitude, and the failure mode I fear most is $F$ collapsing small too early and the search going quiet, so $F$ wants a distribution that keeps *proposing* large values even as the center drifts down: Cauchy, $F_i = M_F[r] + 0.1\tan(\pi(\text{rand}-0.5))$, fat-tailed about the same center. The truncations respect what each parameter is: $F>1$ truncates to 1, but $F\le 0$ I *resample* (a non-positive scaling inverts or kills the mutation), while $CR$ clamps at both ends because both endpoints are meaningful. Summarizing successful $F$ into a new center cannot use the arithmetic mean — on a converging population the successes skew small, the center drifts down, mutation dies, and I manufacture the premature convergence I am avoiding. The weighted Lehmer mean resists that pull,

$$\text{mean}_L(S) = \frac{\sum_k w_k S_k^2}{\sum_k w_k S_k},$$

squaring in the numerator and first-power in the denominator so larger successes pull the ratio up (it is provably $\ge$ the weighted arithmetic mean, equal only when all successes coincide). The weights $w_k = \Delta f_k / \sum_l \Delta f_l$ come from each winner's fitness improvement $\Delta f_k$, because a trial that improved a lot is stronger evidence than one that barely squeaked past. I use the weighted Lehmer mean for $F$ and a weighted arithmetic mean for $CR$ — "let $CR$ settle, but never let $F$ go quiet."

A single steering pair $(M_F, M_{CR})$ (JADE-style) is fragile: one unlucky generation whose winners carry mediocre parameters slides the one center, and next generation the *entire* population samples from the contaminated value — and on a hard multimodal problem where success is noisy and rare, that is the normal weather. The cure is redundancy: keep $H$ slots $M_F = M_{CR} = [0.5,\dots]$, each individual picks an index $r$ uniformly, and each generation writes the winners' summary into *one* slot, cycling round-robin with a counter $k$. A contaminated summary then lands in one of $H$ slots, only about $1/H$ of the population draws from it next generation, and it is overwritten within a few. Round-robin matters — overwriting every slot each generation would just be one effective center — and I take $H=6$. (A generation with no winners updates nothing; the weighted means carry a $10^{-30}$ denominator guard.)

That is a genuinely self-tuning DE, but it leaves the third knob — $N$ — fixed, and $N$ is the one whose pain the convergence columns screamed. Early I want a *large* $N$: broad coverage of the box and a large diverse pool of successes per generation for good adaptation statistics. Late, once the population has localized, a large $N$ is pure waste — hundreds of evaluations per generation for one tiny refinement, when what I want is *many* generations of small precise moves. With fixed $N$ the budget buys $\text{budget}/N$ generations, full stop. So I start large and shrink linearly,

$$N_\text{next} = \text{round}\!\left(N_\text{init} + (N_\text{min} - N_\text{init})\,\frac{\text{gen}+1}{n_\text{generations}}\right),$$

floored at $N_\text{min}=4$ (pinned by the operator, which needs $x_i$, a distinct $x_{p\text{best}}$, $x_{r_1}$, $x_{r_2}$). As $N$ shrinks each generation costs less, so the same remaining budget buys *more* generations — manufacturing the late many-small-steps regime out of budget a fixed large $N$ would have burned on redundant exploration, and respecting DE's self-scaling because a smaller converged population still has small differences. I keep the schedule deterministic and monotone rather than reactive: I have spent two rungs deleting hand-tuned parameters and will not reintroduce a fistful of meta-parameters to govern when and how much to resize. When $N$ must drop I delete the *worst* individuals (elitist shrink accelerates late focusing) and trim the archive to match. One departure from the canonical constant is load-bearing: the natural choice is $N_\text{init}=18D$, sized for large competition budgets, but this harness runs a *fixed, small* generation count, and at 100D $18D=1800$ would starve the search of the generations the linear reduction needs (degrading Rastrigin-100D rather than helping). So I use $N_\text{init}=\text{pop\_size}$ as given and let the reduction to $N_\text{min}=4$ do its work, keeping the budget identical to the other rungs.

```python
# EDITABLE region of deap/custom_evolution.py (lines 87-225) - step 3: L-SHADE

def custom_select(population: list, k: int, toolbox=None) -> list:
    """Not used in L-SHADE (adaptive DE handles selection internally)."""
    return population[:k]


def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    """Not used in L-SHADE (binomial crossover built into run_evolution)."""
    return ind1, ind2


def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
    """Not used in L-SHADE (adaptive mutation built into run_evolution)."""
    return (individual,)


def run_evolution(
    evaluate_func: Callable,
    dim: int,
    lo: float,
    hi: float,
    pop_size: int,
    n_generations: int,
    cx_prob: float,
    mut_prob: float,
    seed: int,
) -> Tuple[list, list]:
    """L-SHADE: Success-History based Adaptive DE with Linear Population Reduction.

    - Adaptive F (Cauchy) and CR (Normal) from success history
    - current-to-pbest/1 mutation with external archive
    - Linear population size reduction from N_init to N_min
    """
    random.seed(seed)
    np.random.seed(seed)

    # --- Hyperparameters ---
    # The canonical choice is N_init = 18·D, but on small fixed budgets (as in
    # our 400 pop × 1000 gen setting) that value starves the search of
    # generations: on Rastrigin-100D, N_init=1800 with matched total-eval
    # budget degraded from 128 → 313. Use pop_size as given and the
    # canonical N_min = 4, which lets the linear population
    # reduction actually run. Budget stays identical to CMA-ES/DE/GA.
    H = 6  # History size (canonical default)
    N_init = pop_size
    N_min = 4  # Minimum population size
    p_min = 2.0 / N_init  # Minimum p for pbest
    p_max = 0.2  # Maximum p for pbest

    toolbox = base.Toolbox()
    toolbox.register("individual", make_individual, toolbox, dim, lo, hi)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_func)

    # Initialize population
    pop = toolbox.population(n=N_init)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Success history for F and CR
    M_F = [0.5] * H
    M_CR = [0.5] * H
    k = 0  # History index

    # External archive of inferior solutions
    archive = []

    fitness_history = []

    for gen in range(n_generations):
        N_current = len(pop)

        # Collect successful F and CR values and their fitness improvements
        S_F = []
        S_CR = []
        delta_f = []  # fitness improvement for weighting

        trial_list = []
        F_list = []
        CR_list = []

        for i in range(N_current):
            # Sample F from Cauchy(M_F[r], 0.1), truncate to (0, 1]
            r = random.randint(0, H - 1)
            while True:
                F_i = M_F[r] + 0.1 * np.random.standard_cauchy()
                if F_i > 0:
                    break
            F_i = min(F_i, 1.0)

            # Sample CR from Normal(M_CR[r], 0.1), clamp to [0, 1]
            CR_i = np.random.normal(M_CR[r], 0.1)
            CR_i = max(0.0, min(1.0, CR_i))

            F_list.append(F_i)
            CR_list.append(CR_i)

            # current-to-pbest/1 mutation
            # Choose p_i uniformly from [p_min, p_max]
            p_i = random.uniform(p_min, p_max)
            n_pbest = max(1, int(round(p_i * N_current)))
            sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values[0])
            pbest = random.choice(sorted_pop[:n_pbest])

            # Select r1 from pop (r1 != i)
            candidates = list(range(N_current))
            candidates.remove(i)
            r1 = random.choice(candidates)

            # Select r2 from pop + archive (r2 != i and r2 != r1)
            union = list(range(N_current + len(archive)))
            union_exclude = {i, r1}
            union_avail = [x for x in union if x not in union_exclude]
            if not union_avail:
                union_avail = [x for x in union if x != i]
            r2_idx = random.choice(union_avail)
            if r2_idx < N_current:
                x_r2 = pop[r2_idx]
            else:
                x_r2 = archive[r2_idx - N_current]

            # Mutation: v = x_i + F * (pbest - x_i) + F * (x_r1 - x_r2)
            mutant = creator.Individual([
                pop[i][j] + F_i * (pbest[j] - pop[i][j]) + F_i * (pop[r1][j] - x_r2[j])
                for j in range(dim)
            ])

            # Binomial crossover
            j_rand = random.randint(0, dim - 1)
            trial = creator.Individual([
                mutant[j] if (random.random() < CR_i or j == j_rand) else pop[i][j]
                for j in range(dim)
            ])

            # Clip to bounds
            for j in range(dim):
                trial[j] = max(lo, min(hi, trial[j]))

            trial.fitness.values = toolbox.evaluate(trial)
            trial_list.append(trial)

        # Selection and success history update
        new_pop = []
        for i in range(N_current):
            trial = trial_list[i]
            if trial.fitness.values[0] <= pop[i].fitness.values[0]:
                if trial.fitness.values[0] < pop[i].fitness.values[0]:
                    S_F.append(F_list[i])
                    S_CR.append(CR_list[i])
                    delta_f.append(abs(pop[i].fitness.values[0] - trial.fitness.values[0]))
                    # Add inferior parent to archive
                    archive.append(creator.Individual(pop[i][:]))
                new_pop.append(trial)
            else:
                new_pop.append(pop[i])

        pop = new_pop

        # Update success history
        if S_F:
            weights = np.array(delta_f)
            weights = weights / (weights.sum() + 1e-30)

            # Weighted Lehmer mean for F
            S_F_arr = np.array(S_F)
            mean_F = np.sum(weights * S_F_arr ** 2) / (np.sum(weights * S_F_arr) + 1e-30)
            M_F[k] = mean_F

            # Weighted arithmetic mean for CR
            S_CR_arr = np.array(S_CR)
            mean_CR = np.sum(weights * S_CR_arr)
            M_CR[k] = mean_CR

            k = (k + 1) % H

        # Trim archive to at most N_current
        while len(archive) > N_current:
            archive.pop(random.randint(0, len(archive) - 1))

        # Linear population size reduction
        N_next = int(round(N_init + (N_min - N_init) * (gen + 1) / n_generations))
        N_next = max(N_min, N_next)
        if N_next < len(pop):
            # Remove worst individuals
            pop.sort(key=lambda ind: ind.fitness.values[0])
            pop = pop[:N_next]

        best_fit = min(ind.fitness.values[0] for ind in pop)
        fitness_history.append(best_fit)

        if (gen + 1) % 50 == 0 or gen == 0:
            avg_fit = sum(ind.fitness.values[0] for ind in pop) / len(pop)
            print(
                f"TRAIN_METRICS gen={gen+1} best_fitness={best_fit:.6e} "
                f"avg_fitness={avg_fit:.6e}",
                flush=True,
            )

    best_ind = min(pop, key=lambda ind: ind.fitness.values[0])
    return best_ind, fitness_history
```
