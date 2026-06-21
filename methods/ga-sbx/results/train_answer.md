The problem is continuous black-box optimization over a box of real variables, where we would like to use a genetic algorithm but cannot afford the pathologies of binary encoding. Binary-coded GAs search discrete spaces well because single-point crossover propagates short, high-fitness substrings, but forcing continuous variables through a fixed-length bit string caps precision, freezes the real-to-string mapping, and creates Hamming cliffs: value-adjacent reals can be bit-distant, so a small move in real space requires flipping many bits at once. Existing real-coded alternatives are also unsatisfying. Wright's linear crossover is essentially deterministic, producing only three candidate children from any pair of parents, so its search power is tiny. Eshelman and Schaffer's BLX-alpha samples uniformly from a widened interval, which is genuinely stochastic but flat, ignoring the strong concentration near the parents that single-point crossover actually exhibits. Evolution-strategy recombination is either discrete or averaging and does little exploration on its own. What is needed is a real-coded crossover that reproduces the distributional behavior of single-point binary crossover directly on the raw reals.

The method I propose is Simulated Binary Crossover, abbreviated SBX. It is a real-parameter recombination operator that reconstructs the two decisive properties of single-point crossover on decoded reals: the children's mean equals the parents' mean, and the spread factor beta = |c1 - c2| / |p1 - p2| is distributed with a sharp peak near beta = 1, meaning children usually land close to their parents but occasionally appear farther out. SBX models this by defining a polynomial density for the spread factor on the contracting side 0 <= beta <= 1 as C(beta) = 0.5 (eta + 1) beta^eta, where eta >= 0 is a single distribution index. The expanding side beta > 1 is not chosen independently; it is forced by the symmetry that crossing the children of a contracting crossover at the same site returns the original parents. This pairs beta with 1/beta and gives E(beta) = beta^{-2} C(1/beta) = 0.5 (eta + 1) / beta^(eta + 2). Each side carries probability one half. Large eta squeezes the density toward beta = 1, so offspring hug their parents and the search is exploitative; eta = 0 makes the contracting half uniform while preserving the expanding tail, giving a broader, more exploratory search. Because beta is a ratio rather than an absolute distance, the same operator automatically explores widely when parents are far apart and narrows its focus as the population converges.

Sampling from this density is cheap because the polynomial choice makes the cumulative distribution closed-form and invertible. Draw u uniformly from [0, 1]; if u <= 0.5, set beta = (2u)^{1/(eta+1)}, and if u > 0.5, set beta = (1 / (2(1-u)))^{1/(eta+1)}. With beta in hand, the two offspring are built by the symmetric mean-preserving combination c1 = 0.5 [(1 + beta) x1 + (1 - beta) x2] and c2 = 0.5 [(1 - beta) x1 + (1 + beta) x2]. Their sum equals x1 + x2, so the midpoint is preserved, and their difference equals beta (x1 - x2), so the spread is exactly beta. SBX is normally applied to every coordinate of a selected mating pair, with the outer crossover probability deciding whether the pair recombines at all.

SBX is only one operator; a complete real-coded GA also needs mutation and selection. For mutation I use the polynomial bounded mutation operator, which perturbs a coordinate with the same polynomial shape but is capped so it can never leave the box. It mutates each coordinate independently with probability 1/n, the real-coded analogue of the binary per-bit rate 1/L that keeps about one coordinate changed per pass. For selection I use tournament selection with tournament size 3, which needs no fitness scaling or sorting and works through pairwise comparisons. The default indices eta_c = eta_m = 20 give a balanced regime where children and mutants stay near their sources while a thinning tail still permits the occasional large jump.

```python
import random
import numpy as np
from typing import Callable, Tuple
from deap import base, tools


def custom_select(population: list, k: int, toolbox=None) -> list:
    """Tournament selection, tournament size 3; Fitness weights define best."""
    return tools.selTournament(population, k, tournsize=3)


def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    """Simulated Binary Crossover (SBX), distribution index eta_c = 20."""
    eta = 20.0
    for i, (x1, x2) in enumerate(zip(ind1, ind2)):
        u = random.random()
        if u <= 0.5:                              # contracting: beta = (2u)^(1/(eta+1))
            beta = 2.0 * u
        else:                                     # expanding: beta = (1/(2(1-u)))^(1/(eta+1))
            beta = 1.0 / (2.0 * (1.0 - u))
        beta **= 1.0 / (eta + 1.0)
        # Mean-preserving, beta-spread children.
        ind1[i] = 0.5 * (((1 + beta) * x1) + ((1 - beta) * x2))
        ind2[i] = 0.5 * (((1 - beta) * x1) + ((1 + beta) * x2))
    return ind1, ind2


def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
    """Polynomial bounded mutation, index eta_m = 20, per-variable rate 1/n."""
    eta = 20.0
    indpb = 1.0 / len(individual)
    for i in range(len(individual)):
        if random.random() <= indpb:
            x = individual[i]
            delta_1 = (x - lo) / (hi - lo)         # normalized room below x
            delta_2 = (hi - x) / (hi - lo)         # normalized room above x
            u = random.random()
            mut_pow = 1.0 / (eta + 1.0)
            if u < 0.5:                            # downward, capped at -delta_1
                xy = 1.0 - delta_1
                val = 2.0 * u + (1.0 - 2.0 * u) * xy ** (eta + 1.0)
                delta_q = val ** mut_pow - 1.0
            else:                                  # upward, capped at +delta_2
                xy = 1.0 - delta_2
                val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * xy ** (eta + 1.0)
                delta_q = 1.0 - val ** mut_pow
            x = x + delta_q * (hi - lo)
            individual[i] = min(max(x, lo), hi)    # numerical safety clip
    return (individual,)


def run_evolution(
    evaluate_func: Callable, dim: int, lo: float, hi: float,
    pop_size: int, n_generations: int, cx_prob: float, mut_prob: float, seed: int,
) -> Tuple[list, list]:
    """Generational real-coded GA: tournament + SBX + polynomial mutation."""
    random.seed(seed)
    np.random.seed(seed)

    toolbox = base.Toolbox()
    toolbox.register("individual", make_individual, toolbox, dim, lo, hi)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_func)

    pop = toolbox.population(n=pop_size)
    for ind, fit in zip(pop, map(toolbox.evaluate, pop)):
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

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        pop[:] = offspring
        fitness_history.append(min(ind.fitness.values[0] for ind in pop))

    best_ind = min(pop, key=lambda ind: ind.fitness.values[0])
    return best_ind, fitness_history
```
