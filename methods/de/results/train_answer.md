Continuous black-box global optimization asks for a vector of real parameters that minimizes a cost we can only sample. The cost is nonlinear, non-differentiable, and multimodal, and each evaluation may be the read-out of a slow simulation or physical experiment. Existing direct-search methods all struggle with the same hidden choice: the scale of the random perturbation applied to a candidate. Greedy local search such as Nelder-Mead self-scales by reflecting a simplex, but a single simplex collapses into a basin and cannot escape. Evolution strategies and genetic algorithms keep a population, yet their mutation distribution is set by an external mechanism, such as sigma adaptation, success rules, or fixed distribution indices, which must itself be tuned and can mis-track the geometry. Simulated annealing can climb out of local minima, but only by accepting uphill moves through a temperature schedule that brings many control variables and a large evaluation budget. The common failure is that the step size is imposed from outside rather than read from the actual state of the search.

The way out is to let the population supply its own scale. When the population is spread out, the typical distance between members is large; when it has converged, that distance is small. The vector difference between two random members therefore already carries the right magnitude and direction for a perturbation: large and exploratory early, small and refining late, and naturally stretched along curved valleys because the population aligns with them. Adding a scaled version of that difference to a third member gives a mutation whose scale and orientation self-adapt with no schedule and no covariance matrix.

This is Differential Evolution. For each target member x_i of the population, choose three other distinct members x_r1, x_r2, x_r3 and form the mutant v = x_r1 + F (x_r2 - x_r3). The difference term samples the population's current spread, so F is only a fixed multiplier on a vector that is already correctly sized and oriented; a value around 0.5 is robust. If F is too small the steps die out before the optimum is reached, while if it is too large the search overshoots and fails to settle. Next, mix the mutant with the target coordinate by coordinate: for each dimension j, take the mutant value with probability CR, otherwise keep the target value, but force one randomly chosen coordinate to come from the mutant so the trial is never identical to the target. CR near 0.9 makes the trial mostly the full difference-vector move, which is good for non-separable problems such as Rosenbrock; lower values change fewer coordinates and suit separable problems. Finally, accept the trial into the next generation only if its cost is no worse than its target's cost. This one-to-one greedy selection keeps the best cost monotone while preserving diversity, because a member that is merely the best in its own lineage survives instead of being crowded out by globally superior points. That diversity keeps the difference vectors varied, so trapped members can be pulled toward better basins.

The algorithm needs only three controls: the scale factor F, the crossover rate CR, and the population size NP. NP must be at least four to allow three distinct helpers and is usually several times the dimension D; too small and the difference vectors stagnate, too large and evaluations are wasted. Initialize uniformly over the box unless a good nominal point is known. The canonical scheme is DE/rand/1/bin: random base, one difference vector, binomial crossover.

```python
import random
from typing import Tuple, Callable
from deap import base, creator, tools

if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


def make_individual(toolbox, dim: int, lo: float, hi: float):
    return creator.Individual([random.uniform(lo, hi) for _ in range(dim)])


def clip_individual(individual, lo: float, hi: float):
    for i in range(len(individual)):
        individual[i] = max(lo, min(hi, individual[i]))
    return individual


def make_generation(population: list, toolbox, dim: int, lo: float, hi: float) -> list:
    """One DE/rand/1/bin generation with F = 0.5, CR = 0.9."""
    F = 0.5
    CR = 0.9
    if len(population) < 4:
        raise ValueError("DE/rand/1/bin needs pop_size >= 4")

    next_population = list(population)

    for i, target in enumerate(population):
        candidates = list(range(len(population)))
        candidates.remove(i)
        r1, r2, r3 = random.sample(candidates, 3)
        x_r1, x_r2, x_r3 = population[r1], population[r2], population[r3]

        mutant = creator.Individual(
            [x_r1[j] + F * (x_r2[j] - x_r3[j]) for j in range(dim)]
        )

        j_rand = random.randrange(dim)
        trial = creator.Individual(
            [mutant[j] if (random.random() < CR or j == j_rand) else target[j]
             for j in range(dim)]
        )
        clip_individual(trial, lo, hi)
        trial.fitness.values = toolbox.evaluate(trial)

        if trial.fitness.values[0] <= target.fitness.values[0]:
            next_population[i] = trial

    return next_population


def run_evolution(evaluate_func: Callable, dim: int, lo: float, hi: float,
                  pop_size: int, n_generations: int,
                  cx_prob: float, mut_prob: float, seed: int) -> Tuple[list, list]:
    """Differential Evolution: DE/rand/1/bin."""
    random.seed(seed)
    _ = (cx_prob, mut_prob)

    toolbox = base.Toolbox()
    toolbox.register("individual", make_individual, toolbox, dim, lo, hi)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_func)

    pop = toolbox.population(n=pop_size)
    for ind in pop:
        ind.fitness.values = toolbox.evaluate(ind)

    fitness_history = []

    for gen in range(n_generations):
        pop = make_generation(pop, toolbox, dim, lo, hi)
        best_fit = min(ind.fitness.values[0] for ind in pop)
        fitness_history.append(best_fit)

    best_ind = min(pop, key=lambda ind: ind.fitness.values[0])
    return best_ind, fitness_history
```
