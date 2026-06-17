**Problem (from step 3).** L-SHADE is the strongest baseline but lost Rastrigin-100d to the GA
(mean 132.5 vs 114): its single `F` ties the elite-pull and random-perturbation strengths together, so
the search committed to exploitation and shrank the population before exploring the 100-D egg-carton
enough.

**Key idea (jSO).** Keep all of L-SHADE; add the **weighted mutation**
`v = x_i + Fw*(x_pbest - x_i) + F*(x_r1 - x_r2)` with `Fw = 0.7F / 0.8F / 1.2F` at budget fractions
`<0.2 / <0.4 / else`, decoupling the elite pull from the perturbation and sliding the balance from
exploration-heavy early to exploitation-heavy late. Reinforced by: `CR` memory biased high (0.8) with
one slot *frozen* at (0.9,0.9); early phase rails (`CR` floored, `F` capped); blended memory update
`(new+old)/2`; and a *decreasing* pbest fraction `p: 0.25 -> 0.125`.

**Why it should beat L-SHADE.** The weaker early elite pull + broader early `p` + high-biased `CR`
memory keep the population exploring the high-D multimodal landscape longer before committing — directly
the correction Rastrigin-100d needed — while the late stronger pull sharpens convergence on the
problems L-SHADE already won.

**Hyperparameters.** `H=5` (slot 4 frozen at 0.9,0.9, slots 0-3 adapt); `M_F` init 0.5, `M_CR` init
0.8; `p_max=0.25`, `p_min=0.125`; `Fw` schedule (0.7,0.8,1.2); `CR>=0.7` (frac<0.25), `>=0.6`
(frac<0.5); `F<=0.7` (frac<0.6); `N_min=4`, archive cap = current N. Harness departure (as in step 3):
`N_init=pop_size`, not the canonical `round(25 log(D) sqrt(D))`, to keep the generation budget identical
to the baselines. Operator stubs are no-ops; the whole search lives in `run_evolution`.

```python
# EDITABLE region of deap/custom_evolution.py (lines 87-225) - finale: jSO (weighted current-to-pbest-w/1)

def custom_select(population: list, k: int, toolbox=None) -> list:
    """Not used in jSO (adaptive DE handles selection internally)."""
    return population[:k]


def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    """Not used in jSO (binomial crossover built into run_evolution)."""
    return ind1, ind2


def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
    """Not used in jSO (adaptive mutation built into run_evolution)."""
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
    """jSO: weighted current-to-pBest-w/1 success-history adaptive DE with linear
    population reduction (Brest, Maucec & Boskovic, CEC 2017).

    - Memory biased high (M_CR=0.8) with one frozen slot at (0.9, 0.9)
    - Weighted mutation factor Fw on the pbest term, phase-scheduled
    - Early phase rails: CR floored, F capped; decreasing pbest fraction p
    - Blended memory update (new + old) / 2
    """
    random.seed(seed)
    np.random.seed(seed)

    # --- Hyperparameters (Brest et al., CEC 2017) ---
    # Canonical N_init = round(25*log(D)*sqrt(D)), but on this fixed small-budget
    # harness that starves the generation count, so use pop_size as given (same
    # departure as the L-SHADE baseline) and let linear reduction to N_min=4 run.
    H = 5                 # memory slots; last one frozen, slots 0..H-2 adapt
    N_init = pop_size
    N_min = 4
    p_max, p_min = 0.25, 0.125

    toolbox = base.Toolbox()
    toolbox.register("individual", make_individual, toolbox, dim, lo, hi)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_func)

    pop = toolbox.population(n=N_init)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Memory: M_F init 0.5, M_CR init 0.8; last slot frozen at (0.9, 0.9).
    M_F = [0.5] * H
    M_CR = [0.8] * H
    M_F[H - 1] = 0.9
    M_CR[H - 1] = 0.9
    k = 0  # round-robin write index over the H-1 adaptable slots

    archive = []
    fitness_history = []

    for gen in range(n_generations):
        N_current = len(pop)
        frac = (gen + 1) / n_generations   # budget fraction proxy

        # Decreasing pbest fraction p: p_max -> p_min over the run
        p = p_max - (p_max - p_min) * frac
        n_pbest = max(2, int(round(p * N_current)))
        sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values[0])

        S_F = []
        S_CR = []
        delta_f = []
        trial_list = []
        F_list = []
        CR_list = []

        for i in range(N_current):
            r = random.randint(0, H - 1)

            # CR ~ Normal(M_CR[r], 0.1); terminal slot (-1) => CR = 0; clamp to [0,1]
            if M_CR[r] == -1:
                CR_i = 0.0
            else:
                CR_i = np.random.normal(M_CR[r], 0.1)
                CR_i = max(0.0, min(1.0, CR_i))
            # Early phase rails on CR
            if frac < 0.25:
                CR_i = max(CR_i, 0.7)
            elif frac < 0.5:
                CR_i = max(CR_i, 0.6)

            # F ~ Cauchy(M_F[r], 0.1); resample if <= 0; truncate to 1
            while True:
                F_i = M_F[r] + 0.1 * np.random.standard_cauchy()
                if F_i > 0:
                    break
            F_i = min(F_i, 1.0)
            # Early phase cap on F
            if frac < 0.6:
                F_i = min(F_i, 0.7)

            # Weighted mutation factor Fw by budget phase
            if frac < 0.2:
                Fw = 0.7 * F_i
            elif frac < 0.4:
                Fw = 0.8 * F_i
            else:
                Fw = 1.2 * F_i

            F_list.append(F_i)
            CR_list.append(CR_i)

            pbest = random.choice(sorted_pop[:n_pbest])

            # r1 from pop (r1 != i)
            candidates = list(range(N_current))
            candidates.remove(i)
            r1 = random.choice(candidates)

            # r2 from pop + archive (r2 != i, r2 != r1)
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

            # Weighted donor: v = x_i + Fw*(pbest - x_i) + F*(x_r1 - x_r2)
            mutant = creator.Individual([
                pop[i][j] + Fw * (pbest[j] - pop[i][j]) + F_i * (pop[r1][j] - x_r2[j])
                for j in range(dim)
            ])

            # Binomial crossover with guaranteed donor coord j_rand
            j_rand = random.randint(0, dim - 1)
            trial = creator.Individual([
                mutant[j] if (random.random() < CR_i or j == j_rand) else pop[i][j]
                for j in range(dim)
            ])

            for j in range(dim):
                trial[j] = max(lo, min(hi, trial[j]))

            trial.fitness.values = toolbox.evaluate(trial)
            trial_list.append(trial)

        # Greedy selection + success collection
        new_pop = []
        for i in range(N_current):
            trial = trial_list[i]
            if trial.fitness.values[0] <= pop[i].fitness.values[0]:
                if trial.fitness.values[0] < pop[i].fitness.values[0]:
                    S_F.append(F_list[i])
                    S_CR.append(CR_list[i])
                    delta_f.append(abs(pop[i].fitness.values[0] - trial.fitness.values[0]))
                    archive.append(creator.Individual(pop[i][:]))
                new_pop.append(trial)
            else:
                new_pop.append(pop[i])

        pop = new_pop

        # Blended memory update over adaptable slots only (skip frozen last slot)
        if S_F and k != H - 1:
            weights = np.array(delta_f)
            weights = weights / (weights.sum() + 1e-30)

            S_F_arr = np.array(S_F)
            mean_F = np.sum(weights * S_F_arr ** 2) / (np.sum(weights * S_F_arr) + 1e-30)
            M_F[k] = 0.5 * (mean_F + M_F[k])

            S_CR_arr = np.array(S_CR)
            if M_CR[k] == -1 or max(S_CR) == 0:
                M_CR[k] = -1   # terminal: CR locked to 0
            else:
                mean_CR = np.sum(weights * S_CR_arr ** 2) / (np.sum(weights * S_CR_arr) + 1e-30)
                M_CR[k] = 0.5 * (mean_CR + M_CR[k])

            k = (k + 1) % (H - 1)

        # Trim archive to at most N_current
        while len(archive) > N_current:
            archive.pop(random.randint(0, len(archive) - 1))

        # Linear population size reduction
        N_next = int(round(N_init + (N_min - N_init) * (gen + 1) / n_generations))
        N_next = max(N_min, N_next)
        if N_next < len(pop):
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
