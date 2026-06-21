L-SHADE confirmed its whole thesis — one adaptive algorithm *is* right across landscapes where the two fixed methods each failed somewhere. Ackley fell to a mean of $9.4\times10^{-12}$ (solved, near machine precision, and converged early at generation 258 rather than burning the budget); Rosenbrock to a mean of 13.3 (beating DE's 32.2 and crushing the GA's 138 as the adaptive $CR$ recovered the coordinated valley descent); Rastrigin-30d to a mean of 7.30, edging the GA's 8.05. But the one I flagged as uncertain slipped: Rastrigin-100d came back at a mean of **132.5**, *worse* than the GA's 113.8. The convergence column is the tell — at 100D the search was still improving at generation ~995 of 1000, so it did not converge to a wrong answer and sit there; it was still moving but too slowly, having spent its early generations on a population too small ($N_\text{init}=\text{pop\_size}$, not a large multiple of $D$) to cover 100 dimensions. The deliberate small $N_\text{init}$ — forced by this fixed-budget harness — traded exploration for refinement, and on the hardest multimodal case the population committed to exploitation, and shrank toward it, before exploring the egg-carton enough. I need more effective exploration in the first half of the run *without* enlarging the population budget, which rules out the obvious bigger-$N_\text{init}$ fix and forces a subtler one: extract more exploration from the same population by changing *how* each individual moves early.

I propose jSO — weighted current-to-pBest-w/1 adaptive DE — which keeps all of L-SHADE and fixes the one place its mutation is structurally crude. Look at the donor $v_i = x_i + F_i(x_{p\text{best}} - x_i) + F_i(x_{r_1} - x_{r_2})$: the *same* $F_i$ multiplies two structurally different terms. The first, $x_{p\text{best}} - x_i$, is a directed pull toward the elite — pure exploitation. The second, $x_{r_1} - x_{r_2}$, is the self-scaling random difference — the exploration. Tying both to one $F_i$ forces the elite-pull strength and the perturbation strength to move in lockstep, but the Rastrigin-100d failure is exactly a balance problem: the search committed to the elite before perturbing enough to find the right basin. So I give the elite-pull term its own weighted factor $F_w$, leaving $F_i$ on the random difference, and phase-schedule it on the budget fraction:

$$v_i = x_i + F_w\,(x_{p\text{best}} - x_i) + F_i\,(x_{r_1} - x_{r_2}), \qquad F_w = \begin{cases} 0.7\,F_i & \text{frac} < 0.2 \\ 0.8\,F_i & \text{frac} < 0.4 \\ 1.2\,F_i & \text{otherwise.} \end{cases}$$

Early $F_w < F_i$ (perturb more than pull — explore without committing); late $F_w > F_i$ (pull more than perturb — refine hard once the basin is found). This is the heart of the improvement: it decouples the two roles $F$ was forced to play and slides their balance from exploration-heavy early to exploitation-heavy late, which is exactly the correction Rastrigin-100d needed and which a single $F_i$ cannot express.

Several smaller refinements reinforce the same exploration-early/exploitation-late arc, each saving budget L-SHADE wasted. The memory starts biased: L-SHADE initializes every slot neutral at 0.5 and re-discovers on every problem that high $CR$ helps early, so I initialize $M_{CR}$ at 0.8 (high $CR$ giving coordinated multi-coordinate moves on a spread population) and leave $M_F$ at 0.5 — a free head start the adaptation overrides if a near-separable problem prefers low $CR$. I also *freeze* the last memory slot at $(M_F, M_{CR})=(0.9, 0.9)$ and never update it; because each individual picks its memory index uniformly, about $1/H$ of the population always samples near $(0.9, 0.9)$ regardless of what the rest of the memory decides — cheap permanent insurance against the whole population going quiet, which is one description of the Rastrigin-100d collapse. Hard phase rails keep early sampling in the explore-coordinated regime and then relax: for the first quarter of the budget floor $CR$ at 0.7, for the first half floor it at 0.6, and for the first 0.6 cap $F$ at 0.7. And the memory update is *blended* rather than overwritten — $M[k] \leftarrow (\text{mean}_{WL}(S) + M[k]_\text{old})/2$ — so a single noisy generation moves a slot at most halfway to its summary and the slot retains its history, a stabilizer on the adaptation trajectory.

The pbest greediness $p$ gets the same phase treatment: where L-SHADE drew $p_i$ from a fixed interval, I make $p$ *decrease* over the run, $p = p_\text{max} - (p_\text{max} - p_\text{min})\,\text{nfes}/\text{max\_nfes}$ with $p_\text{max}=0.25$, $p_\text{min}=0.125$. Early the elite guide is drawn from the top quarter (diverse, several good basins represented); late from the top eighth (sharp convergence) — the same exploration-to-exploitation arc as the weighted mutation, applied to where the elite guide comes from, directly widening early exploration on the high-D multimodal case that failed. The constants carry over from the adaptive-DE substrate: memory size $H=5$ with the last slot frozen, so four slots adapt and the round-robin counter cycles over those four; $N_\text{min}=4$ (current-to-pbest/1 needs four distinct individuals); the archive caps at the current population size with random deletion on overflow; Cauchy on $F$ (resample if $\le 0$, truncate to 1) for mutation-strength diversity; Normal on $CR$ (clamp to $[0,1]$) for stability, with the terminal-$CR=0$ rule locking the one-coordinate regime on slots whose successful $CR$ are all zero. And, as in the L-SHADE rung, the one place this departs from the canonical constant is $N_\text{init}=\text{pop\_size}$ rather than $\text{round}(25\log(D)\sqrt{D})$: a large initial population would starve this fixed-small-budget harness of generations, so the weighted mutation and the phase schedule must do their work *within* the same population budget as every other rung, not by enlarging it.

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
