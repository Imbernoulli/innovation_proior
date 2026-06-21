Every direct-search method on this panel comes down to one question: by how much, and in what direction, do I perturb a candidate point? The scaffold I am handed answers it with simulated binary crossover and polynomial mutation under a fixed distribution index $\eta=20$ — a legitimate genetic algorithm, but stare at how it sets scale. The SBX spread and the mutation width are governed by a number I fix once, by hand, that does not contract as the population converges and does not align with the shape of a valley. That is the disease shared across the lineage: evolution strategies make $\sigma$ the scale but pay for a second optimization problem (the 1/5 rule, self-adapted $\sigma$, a full $D\times D$ covariance) to control it; simulated annealing pays with a cooling schedule and an enormous evaluation appetite. In every case the perturbation's scale is set by an *external device* sitting outside the actual configuration of the points — a temperature, a controlled $\sigma$, a fixed width — brittle when the landscape disagrees and a tuning burden everywhere. I do not want a better device. I want the scale to not be a device at all.

So I propose Differential Evolution in its DE/rand/1/bin form, and what makes it work is that it reads the move scale straight off the population. A population is a distribution of points: early it is scattered across the box and the cloud is large; late, as it converges, the cloud is tight around the optimum. That spread — the typical distance between members — is exactly the quantity I keep wanting $\sigma$ to be, and it tracks the phase of the search for free because the population literally contracts as it converges. The simplest object built from the population whose magnitude *is* that spread is the difference of two members. Pick two members $x_{r_2}$ and $x_{r_3}$ at random; $x_{r_2}-x_{r_3}$ is, by construction, a sample of the current typical pairwise distance — large while exploring, small while refining. Add a scaled copy to a third member to get the mutant

$$v = x_{r_1} + F\,(x_{r_2} - x_{r_3}).$$

I never estimated a distribution; I subtracted two points I already hold. The self-scaling is automatic, and it comes with a second gift I almost miss — orientation. Difference vectors inherit the *shape* of the population cloud, not just its size. On Rosenbrock's long curved valley the population, selected toward low cost, spreads out *along* the thin ribbon and stays narrow across it, so two random members differ far more along the valley than across it and $x_{r_2}-x_{r_3}$ points preferentially down it. The perturbation is anisotropic in exactly the way the landscape is — the correlated, rotation-aware search ES needs an explicit covariance for, obtained here for nothing because the differences sample a cloud that has already aligned itself with the contours.

The details each carry a real choice. The base $x_{r_1}$ is *random*, not the current best: always perturbing the best converges fast on an easy bowl but concentrates every trial around one point and destroys the diversity that lets a population escape local minima — fatal on Rastrigin's grid of pits — so the cautious general default is the "rand" scheme. The factor $F$ is a single dial on a vector that is already pointed and sized correctly; I keep it because the raw inter-member hop is not necessarily the *optimal* step length, but I do not adapt it over time because the difference vector already carries the time-adaptation. The conventional range is $0 < F \le 2$, and I take $F=0.5$. The indices $r_1, r_2, r_3$ must be distinct from each other (if $r_2=r_3$ the difference is the zero vector, a wasted trial) and from the target $i$ (the move should be informed by *other* members, not coupled to the vector it competes against), which needs at least four members — trivially satisfied at $\text{pop\_size}=200$.

The mutant does not become the trial wholesale, because moving all $D$ coordinates of $x_{r_1}$ at once is wasteful on a separable problem where I would rather change a few coordinates and let the good values of the others ride along. So I mix the mutant with the target coordinate by coordinate — binomial crossover. Build the trial $u$ by taking $u_j = v_j$ with probability $CR$, else $u_j = x_{i,j}$, each coordinate an independent coin. $CR$ is a dial on the dependency structure: near 1 almost every coordinate comes from the mutant (the full difference-vector move, right for a non-separable problem like Rosenbrock where coordinates must move together); near 0 only a coordinate or two change (right for a separable problem optimized axis by axis). For the general non-separable case I lean high, $CR=0.9$, knowing it is the wrong setting for separable Rastrigin. One degeneracy remains: if every coin says "keep the target," $u=x_i$ and the evaluation is wasted, so I pick a random index $j_\text{rand}$ up front and always take that coordinate from $v$ — $u_j = v_j$ if $\text{random}() < CR$ or $j = j_\text{rand}$ — which rules out the all-target copy.

Acceptance is greedy and one-to-one: the trial $u$ competes only against its own target $x_i$ and takes the slot if $\text{cost}(u) \le \text{cost}(x_i)$, otherwise $x_i$ stays. This is exactly right on two counts. It is one-to-one, so each trial replaces only the single member it was built from — keeping $\text{pop\_size}$ fixed and, crucially, preserving diversity far better than a global "keep the best $N$" truncation, which would let a few good basins crowd out everything and collapse the spread that feeds the self-scaling differences. And it is elitist per slot, so the best cost is monotone non-increasing — the method never throws away its best find. Greedy acceptance does not trap the search the way plain greedy descent does, because a single member sliding into a local basin cannot drag the others, and difference vectors from members exploring elsewhere keep proposing jumps that pull a trapped member out. DE's three control variables are just $F$, $CR$, and $\text{pop\_size}$ — none a schedule or an adapted matrix. Selection is greedy inside the loop, mutation is the difference vector, crossover is binomial, so the scaffold's three operator stubs are left as no-ops and the whole search lives in `run_evolution`, with the trial overwriting `pop[i]` in place during the sweep (the standard DE-with-immediate-replacement form the scaffold's single loop invites).

```python
# EDITABLE region of deap/custom_evolution.py (lines 87-225) - step 1: DE/rand/1/bin

def custom_select(population: list, k: int, toolbox=None) -> list:
    """Not used directly in DE (greedy selection is built into run_evolution)."""
    return population[:k]


def custom_crossover(ind1: list, ind2: list) -> Tuple[list, list]:
    """Not used directly in DE (binomial crossover is built into run_evolution)."""
    return ind1, ind2


def custom_mutate(individual: list, lo: float, hi: float) -> Tuple[list]:
    """Not used directly in DE (DE mutation is built into run_evolution)."""
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
    """Differential Evolution: DE/rand/1/bin.

    F (scale factor) = 0.5, CR (crossover rate) = 0.9.
    Greedy selection: trial replaces target only if fitness improves.
    """
    random.seed(seed)
    np.random.seed(seed)

    F = 0.5   # Scale factor (Storn & Price recommended default)
    CR = 0.9  # Crossover rate

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
        for i in range(len(pop)):
            # Select three distinct random individuals (not i)
            candidates = list(range(len(pop)))
            candidates.remove(i)
            r1, r2, r3 = random.sample(candidates, 3)
            x_r1, x_r2, x_r3 = pop[r1], pop[r2], pop[r3]

            # DE/rand/1 mutation
            mutant = creator.Individual([
                x_r1[j] + F * (x_r2[j] - x_r3[j])
                for j in range(dim)
            ])

            # Binomial crossover
            j_rand = random.randint(0, dim - 1)
            trial = creator.Individual([
                mutant[j] if (random.random() < CR or j == j_rand) else pop[i][j]
                for j in range(dim)
            ])

            # Clip to bounds
            for j in range(dim):
                trial[j] = max(lo, min(hi, trial[j]))

            # Evaluate trial
            trial.fitness.values = toolbox.evaluate(trial)

            # Greedy selection
            if trial.fitness.values[0] <= pop[i].fitness.values[0]:
                pop[i] = trial

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
