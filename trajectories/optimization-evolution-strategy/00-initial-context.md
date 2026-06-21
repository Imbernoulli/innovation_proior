## Research question

Continuous black-box minimization: minimize a scalar objective `f(x)` over a box of real vectors. `f` is nonlinear, multimodal, ill-conditioned, and available only through function queries (feed a vector, get a number). The task is to design the **evolutionary strategy** — selection, recombination, mutation, and the loop that drives them. The benchmark functions, bounds, fitness types, and evaluation harness are fixed. The target is low `f` on a panel of standard test functions: multimodal Rastrigin, curved Rosenbrock valley, flat-then-needle Ackley, at 30D and 100D.

## Prior art / Background / Baselines

Current population-based continuous optimizers:

- **Binary genetic algorithms.** Represent real vectors as bit strings and apply selection, single-point crossover, and bit-flip mutation.
- **Classical evolution strategies.** Mutate a parent by adding isotropic normal noise `x ← x + N(0, σ²)` and adapt the global step size σ with the 1/5 success rule or by co-evolving it.
- **Real-coded crossovers.** Recombine parent reals directly (linear blending, or interval sampling like BLX-α). BLX-α samples uniformly inside a box around the parents.
- **Simulated annealing.** Escape local minima by accepting uphill moves with probability `exp(−Δ/T)` while cooling `T`.

## Fixed substrate / Code framework

Frozen and not to be touched: the benchmark functions (`rastrigin`, `rosenbrock`, `ackley`, each returning a one-tuple `(value,)`), their bounds, DEAP single-objective minimization types (`creator.FitnessMin` with `weights=(-1.0,)` and `creator.Individual` as a `list`), and the evaluation harness below the editable region. Two helpers may be reused: `make_individual(toolbox, dim, lo, hi)` and `clip_individual(individual, lo, hi)`. The harness uses `pop_size=200`, `n_generations=500` (group budget), `cx_prob=0.9`, `mut_prob=0.2`, and reports `best_fitness` (final best value, lower is better) and `convergence_gen` (first generation within 1% of the final best).

## Editable interface

Exactly one region is editable — lines 87–225 of `deap/custom_evolution.py`, with four definitions:

- `custom_select(population, k, toolbox)` — return `k` selected individuals.
- `custom_crossover(ind1, ind2)` — recombine two parents in place, return the pair.
- `custom_mutate(individual, lo, hi)` — perturb one individual in place, return a one-tuple.
- `run_evolution(...)` — the full loop; must return `(best_individual, fitness_history)` and print `TRAIN_METRICS gen=G best_fitness=F avg_fitness=A` every 50 generations.

A method that restructures the algorithm can ignore the operator stubs and write the whole search inside `run_evolution`; an operator-suite method fills all three operators and uses the default generational loop. `numpy`, `scipy`, `math`, `random`, and `deap.{base,creator,tools,cma}` are available.

The starting scaffold is below — tournament selection, simulated binary crossover (η=20), polynomial bounded mutation (η=20, indpb=1/n), in a generational loop. Each step replaces exactly this region.

```python
# EDITABLE region of deap/custom_evolution.py (lines 87-225) - scaffold default fill

def custom_select(population: list, k: int, toolbox=None) -> list:
    """Default: tournament selection with tournament size 3."""
    return tools.selTournament(population, k, tournsize=3)


def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    """Default: simulated binary crossover (SBX), eta=20."""
    tools.cxSimulatedBinary(ind1, ind2, eta=20.0)
    return ind1, ind2


def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
    """Default: polynomial mutation, eta=20, indpb=1/dim."""
    tools.mutPolynomialBounded(
        individual, eta=20.0, low=lo, up=hi,
        indpb=1.0 / len(individual)
    )
    return (individual,)


def run_evolution(
    evaluate_func: Callable, dim: int, lo: float, hi: float,
    pop_size: int, n_generations: int, cx_prob: float, mut_prob: float, seed: int,
) -> Tuple[list, list]:
    """Default generational loop: select -> clone -> crossover -> mutate -> clip -> evaluate -> replace."""
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

## Evaluation settings

Four minimization benchmarks (lower is better), each run at seed 42 (the leaderboard aggregates three internal seeds into a mean):

| Benchmark | Function | Dim | Domain | Global min |
|---|---|---|---|---|
| rastrigin-30d | Rastrigin (multimodal, separable) | 30 | [-5.12, 5.12] | 0 |
| rosenbrock-30d | Rosenbrock (curved valley, non-separable) | 30 | [-5, 10] | 0 |
| ackley-30d | Ackley (flat plateau, needle at origin) | 30 | [-32.768, 32.768] | 0 |
| rastrigin-100d | Rastrigin (high-dimensional, hidden eval) | 100 | [-5.12, 5.12] | 0 |

Metrics per benchmark: `best_fitness` (final best value) and `convergence_gen` (first generation within 1% of final best). Budget: `pop_size=200`, `n_generations=500` for the 30D functions (1000 for 100D), `cx_prob=0.9`, `mut_prob=0.2`.
