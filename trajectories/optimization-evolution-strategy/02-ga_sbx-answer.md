**Problem (from step 1).** DE's single global `CR=0.9` couples nine of ten coordinates per trial, which is
catastrophic on the separable Rastrigin egg-carton (172 at 30D, 926 at 100D) — there is no local,
per-coordinate move and no operator that changes one axis at a time.

**Key idea.** A real-coded GA whose operators are derived to carry binary single-point crossover's
behavior into continuous space *locally*. SBX samples a spread factor β from the closed-form polynomial
density `C(β)=0.5(η+1)β^η` (peaked at β=1, so children hug parents) and builds the two mean-preserving,
β-spread children `c = 0.5[(1±β)x1+(1∓β)x2]`. Polynomial mutation perturbs a single coordinate with a
boundary-aware density that can never leave `[lo,hi]`, at per-coordinate rate `1/n` — the one-axis-at-a-time
move Rastrigin rewards. Tournament(3) supplies scale-free mild selection pressure.

**Why it should beat DE on the separable problem.** The per-coordinate polynomial mutation fixes one axis
at a time instead of betting the whole vector on a coupled difference-vector move.

**Hyperparameters.** SBX `eta_c=20`, polynomial mutation `eta_m=20`, `indpb=1/n`, tournament size 3;
`cx_prob=0.9`, `mut_prob=0.2`, `pop_size=200`, `n_generations=500` (harness). This is the scaffold default
fill — all three operator stubs filled, the default generational loop.

```python
# EDITABLE region of deap/custom_evolution.py (lines 87-225) - step 2: GA (tournament + SBX + polynomial)

def custom_select(population: list, k: int, toolbox=None) -> list:
    """Tournament selection with tournament size 3."""
    return tools.selTournament(population, k, tournsize=3)


def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    """Simulated Binary Crossover (SBX), eta=20."""
    tools.cxSimulatedBinary(ind1, ind2, eta=20.0)
    return ind1, ind2


def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
    """Polynomial bounded mutation, eta=20, indpb=1/dim."""
    tools.mutPolynomialBounded(
        individual, eta=20.0, low=lo, up=hi,
        indpb=1.0 / len(individual)
    )
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
    """Standard GA with tournament selection, SBX crossover, polynomial mutation."""
    random.seed(seed)
    np.random.seed(seed)

    toolbox = base.Toolbox()
    toolbox.register("individual", make_individual, toolbox, dim, lo, hi)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_func)

    pop = toolbox.population(n=pop_size)
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    fitness_history = []

    for gen in range(n_generations):
        offspring = custom_select(pop, len(pop), toolbox)
        offspring = [toolbox.clone(ind) for ind in offspring]

        for i in range(0, len(offspring) - 1, 2):
            if random.random() < cx_prob:
                custom_crossover(offspring[i], offspring[i + 1])
                del offspring[i].fitness.values
                del offspring[i + 1].fitness.values

        for i in range(len(offspring)):
            if random.random() < mut_prob:
                custom_mutate(offspring[i], lo, hi)
                del offspring[i].fitness.values

        for ind in offspring:
            clip_individual(ind, lo, hi)

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = list(map(toolbox.evaluate, invalid_ind))
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

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
