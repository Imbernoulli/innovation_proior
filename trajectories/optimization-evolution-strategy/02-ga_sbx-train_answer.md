DE/rand/1/bin came back as a clean confirmation of the one worry I flagged, plus a worse surprise. The headline failure is Rastrigin: a mean best fitness of 256.8 at 30D and **925.8** at 100D — catastrophic on a separable egg-carton. The cause is structural, not a bug. DE's single global $CR=0.9$ forces nine of every ten coordinates to move together each trial, which is precisely wrong when the coordinates are independent: the search cannot fix one axis at a time, it must coordinate a near-full-dimensional jump that almost always lands in a worse pit, and at 100D it was still improving at the wall (convergence around generation 968). The failure is the absence of a *local, per-coordinate* perturbation and a learner that does not bet the whole vector on every move. That is what I build now — working directly on reals, because forcing them through a binary encoding imports the Hamming cliff ($0111$ and $1000$ are adjacent integers but differ in every bit), which is itself fatal on a grid of pits where the good move is a small shift in one coordinate.

I propose a real-coded genetic algorithm built from operators derived to carry binary single-point crossover's behavior into continuous space *locally*: tournament selection, simulated binary crossover (SBX), and polynomial bounded mutation. The derivation of SBX is the load-bearing part. Take two parent bitstrings, cut at a random site $k$, swap the tails. Writing a decoded value as $x = B\cdot 2^k + A$, each child keeps its own parent's high part $B$ and takes the other's low part $A$, so $y_1 + y_2 = (B_1+B_2)2^k + (A_1+A_2) = x_1 + x_2$: the children's mean equals the parents' mean. Single-point crossover does not drift the per-variable centroid, it spreads symmetrically around it. The amount of spread is the ratio $\beta = |c_1 - c_2| / |p_1 - p_2|$, and because it is a *ratio*, the same $\beta$ yields close children when parents are close and distant children when far apart — so the operator self-anneals as the population converges, the property DE got from its difference vectors. The shape of $\beta$'s distribution is what BLX-$\alpha$ (flat density on a widened interval) gets wrong: computing $\beta(k)$ on the cleanest case and converting site-counts to a density (proportional to the reciprocal slope) shows single-point crossover overwhelmingly produces children *close to* their parents, the chance of a far-out child thinning out.

So I design the contracting density $C(\beta)$ on $[0,1]$ to increase toward $\beta=1$, be trivial to sample (it runs on every coordinate of every mating every generation), and carry a concentration knob. The simplest such family is the power law $C(\beta) = c\,\beta^{\eta}$ with distribution index $\eta \ge 0$, fixed by the half-mass requirement $\int_0^1 c\,\beta^{\eta}\,d\beta = 0.5$, giving $c = 0.5(\eta+1)$ and

$$C(\beta) = 0.5(\eta+1)\,\beta^{\eta}.$$

The contracting/expanding symmetry (crossing the two children at the same site returns the parents, so $\beta$ on $[0,1]$ maps one-to-one to $1/\beta$ on $(1,\infty)$) forces the expanding branch to $E(\beta) = 0.5(\eta+1)/\beta^{\eta+2}$, whose mass integrates to exactly $0.5$ — so the power law is *consistent* with the structure, not merely convenient. Sampling is closed-form with one power per coordinate: invert the contracting CDF $0.5\,\beta^{\eta+1} = u$ for $u \le 0.5$ to get $\beta = (2u)^{1/(\eta+1)}$, and the expanding CDF to get $\beta = (1/(2(1-u)))^{1/(\eta+1)}$ for $u > 0.5$. Then turn $\beta$ into the two mean-preserving, $\beta$-spread children

$$c_1 = \tfrac12[(1+\beta)x_1 + (1-\beta)x_2], \qquad c_2 = \tfrac12[(1-\beta)x_1 + (1+\beta)x_2],$$

whose sum is $x_1+x_2$ and whose separation is $\beta(x_1-x_2)$. I fix $\eta=20$, a moderate value that peaks the density sharply at $\beta=1$ so children hug good parents — exploitative but not frozen, and crucially *local*: a $\beta$ near 1 changes each coordinate only slightly, which is exactly the per-axis locality Rastrigin demanded and DE's $CR=0.9$ denied.

Crossover alone is not a complete GA — escaping a prematurely-collapsed basin is the whole game on Rastrigin — so I add polynomial mutation with the same character: perturb one variable to a nearby value, small perturbations more likely than large, never stepping outside $[\text{lo},\text{hi}]$. A bare perturbation $\delta$ with density $\propto (1-|\delta|)^{\eta_m}$ on $[-1,1]$ would pile mass on a boundary wall when clipped, so I fold the boundary distance into the inversion itself: with $\delta_1 = (x-\text{lo})/(\text{hi}-\text{lo})$ and $\delta_2 = (\text{hi}-x)/(\text{hi}-\text{lo})$, for $u<0.5$ take $\delta_q = [2u + (1-2u)(1-\delta_1)^{\eta_m+1}]^{1/(\eta_m+1)} - 1$ and for $u\ge 0.5$ the mirror, then $x' = x + \delta_q(\text{hi}-\text{lo})$. At the lower bound $\delta_1=0$ forces $\delta_q=0$ (no downward move possible) so the density bends to the box rather than being clipped to it. The per-coordinate rate is $\text{indpb}=1/n$, the real-coded analogue of the binary $1/L$, so on average exactly one variable changes per pass — enough diversity without tearing good solutions apart, and *one coordinate at a time*, the local per-axis move Rastrigin rewards. Selection is tournament: grab $t$ individuals at random, keep the best, repeat. No sorting and no fitness scaling — minimization is handled by the $\text{FitnessMin}$ weight, not a rewritten rule — and the size $t=3$ is mild pressure that lets the population converge without collapsing diversity too fast. This is exactly the scaffold default fill, in the standard generational loop: tournament-select a parent pool, clone, apply SBX to each pair with probability $\text{cx\_prob}=0.9$, send each individual through polynomial mutation with probability $\text{mut\_prob}=0.2$, clip into the box, re-evaluate the changed, and replace the population generationally.

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
