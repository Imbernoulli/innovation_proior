Let me start from what actually goes wrong when I try to minimize one of these costs. I have a scalar `f(x)` over a real vector `x` of dimension `D`, and it is the nasty kind: nonlinear, with kinks so no derivative, riddled with local minima, and sometimes I cannot even write it down — it is the output of a circuit simulation or a physical measurement that takes minutes per call. So I am doing direct search: I can only ever poke `f` at points and compare the numbers it returns. The skeleton of every direct-search method I know is the same two steps — take a vector, generate a *variation* of it, then decide whether to keep the variation. And the more I think about where these methods actually live or die, the more it all collapses onto one question: by how much, and in what direction, do I perturb? If I perturb too hard the search thrashes and never settles; too gently and it either crawls or sinks into the first local pit and stays. And the cruel part is that the right amount is not a constant — early on, when I am scanning a wide domain, I want big jumps; late, when I am polishing a solution, I want tiny ones — and it is different for every cost surface. So the real problem, underneath "global optimization," is: where does the *scale* of my perturbation come from, and how does it stay right as the search moves through its phases, without me hand-tuning it for each problem?

Let me lay out the tools I have and look hard at exactly how each one answers that scale question, because the gap will tell me what to build. Plain greedy direct search: generate some variation, accept it only if cost drops. Fast, but it has no answer to the scale question at all — and with strict downhill-only acceptance it just rolls into the nearest local minimum and dies there. So I need at least one of the two known safeguards against trapping. One is to relax greed: simulated annealing accepts an uphill move of size `Δ` with probability about `exp(−Δ/T)` and cools `T` over time, so early on it can climb out of pits and later it reverts to greedy. It works, but look at the cost: a temperature *schedule* is a whole apparatus of control variables — an adaptive version exposes something like a dozen knobs, of which a couple dominate — and worst of all it is slow, burning enormous numbers of function evaluations, which is exactly what I cannot afford when each call is a slow simulation. The other safeguard is population: run many vectors at once, so a vector stuck in a bad basin can be rescued by the better-placed members, and so I get parallelism for free across the expensive evaluations. Population feels more promising for my constraints. So let me look at the population methods and see how *they* answer the scale question.

Evolution strategies. Mutate a parent by adding a normal perturbation coordinate-wise, `x_j ← x_j + N(0, σ²)`. Now σ *is* the scale, and the whole tradition is built around controlling it: Rechenberg's 1/5 rule nudges σ up or down so that about one mutation in five succeeds — a beautiful idea, derived on the sphere and the inclined ridge — and later you self-adapt a per-coordinate σ, or even a full covariance, by evolving those strategy parameters alongside `x`. So ES *does* answer the scale question, but it answers it by bolting on a second optimization problem: now I am also tuning, or adapting, σ (and maybe a `D×D` covariance), with its own learning machinery that can lag or mistrack the true geometry, and its own extra control variables. Genetic algorithms have the same disease in a blunter form. Selection, crossover, mutation on a population; on continuous problems you use real-coded crossover that blends parents and a mutation that perturbs a gene by some fixed-width distribution. The mutation — the thing that injects fresh variation — draws from a *fixed* distribution whose width I set once, by hand. It does not shrink as the population converges, it does not align with the shape of the valley; it is right for at most one phase of one problem.

So here is the pattern, and it is the same pattern in every one of them. The perturbation's scale is set by an *external device* sitting outside the actual configuration of my candidate points: a temperature schedule, a separately controlled σ, a fixed mutation width. Each device is a thing I have to tune, and each is brittle when the landscape does not match what it assumed. That is the wall. I do not want a better device for setting the scale. I want the scale to *not be a device at all*.

Is that even possible? There is one method that hints it is. Nelder–Mead. It keeps `D+1` vertices and it generates new vertices by reflecting, expanding, and contracting the figure *relative to its own current spread*. The size of the next move is read straight off how big the simplex currently is — no σ, no schedule, no user scale. When the simplex is big the moves are big; as it collapses into a basin the moves shrink with it. That is exactly the self-scaling behavior I am hunting. The trouble is it is a single figure of `D+1` points, a purely local minimizer that, once it slides into a basin, contracts and cannot climb back out — and even annealed it is not strong enough for global search. But it carries the idea I want: *extract the scale of the next move from the configuration of the points you are already holding.* Nelder–Mead does this with one tiny simplex. I have a whole population. What if I extract the move scale from the population itself?

Let me stare at what a population actually contains, as a distribution of points in the search space. Early in the run the members are scattered widely across the box, so the cloud is large. Late in the run, as they converge, the cloud is tight around the optimum. That spread — the typical distance between members — is precisely the quantity I keep wanting σ to be: big while exploring, small while refining, and it *already tracks the phase of the search on its own*, for free, with no schedule, because the population literally contracts as it converges. The information I have been paying ES and SA to supply is sitting right there in the geometry of the points I am already carrying. The question is just how to turn "the population is spread this much" into an actual perturbation vector I can add.

The naive thing would be to estimate the population's covariance and sample a Gaussian from it — but that is just ES with extra steps, a `D×D` matrix to estimate and store and keep current, the very machinery I am trying to delete. I want something that gives me a perturbation *of the right scale and orientation* without ever forming a distribution. So: what is the simplest object built from the population whose typical magnitude *is* the population's spread? The difference between two members. Pick two population vectors `x_{r2}` and `x_{r3}` at random and form `x_{r2} − x_{r3}`. This vector's length is, by construction, a sample of the typical pairwise distance in the current population — so its magnitude is large when the cloud is large and small when the cloud is small. It is a *sample* drawn from the population's own current distribution of differences. I never estimated anything; I just subtracted two points I already have. Add a scaled copy of it to a third member as the perturbation:

    v = x_{r1} + F · (x_{r2} − x_{r3}).

And now the self-scaling is automatic and exact, not approximate. Early, members are far apart, `x_{r2} − x_{r3}` is a big vector, so the step is big and exploratory. Late, members have clustered, the difference is tiny, so the step is tiny and refining. The perturbation scale rides the population's contraction with zero parameters and zero schedule. This is the thing ES needed a 1/5 rule or a self-adapted σ to fake, falling out of pure subtraction.

There is a second gift here that I almost missed, and it is the orientation. The difference vectors are not isotropic — they inherit the *shape* of the population cloud, not just its size. Picture a long curved valley like Rosenbrock, where the good region is a thin bent ribbon. The population, being selected toward low cost, spreads out *along* that ribbon and stays thin across it. So two members picked at random tend to differ much more along the valley than across it, and `x_{r2} − x_{r3}` points preferentially down the valley. The perturbation is automatically anisotropic in exactly the way the landscape is — it proposes moves along the directions where the population has discovered it can move productively. This is correlated, rotation-aware search, the thing ES needs an explicit covariance matrix to achieve, and I am getting it for nothing because the differences are samples of a cloud that has already aligned itself with the contours. The population is doing the covariance estimation for me, implicitly, by where its members sit.

So this difference-vector perturbation is the core. Let me now nail down the details, because each one is a real choice with a failure mode if I get it wrong. The base vector `x_{r1}` that I perturb — random, or the current best? If I always perturb the best, the search gets greedy and converges fast on easy unimodal problems, but it concentrates all trials around one point and loses the diversity that lets the population escape local minima — dangerous on the multimodal costs I most care about. A randomly chosen base keeps exploration broad and the population diverse. So the cautious, general default is a random base: scheme "rand." (I can keep the greedier best-based variant in my back pocket for non-critical, easy functions, but it is not the workhorse.) The factor `F`: it amplifies the already-adapted difference vector. Why have it at all, if the difference is already the right scale? Because the *raw* typical difference is not necessarily the *optimal* step length — I may want to take a bit less than a full inter-member hop to converge cleanly, or a bit more to push exploration on a rugged surface. `F` is a single dial on a vector that is already pointed and sized correctly, so it is robust and easy to choose, unlike σ which had to *be* the whole scale. If `F` is too small the steps die out before the population reaches the optimum — premature convergence; if `F` is too large the trials overshoot and the population never settles. The allowed range is positive, conventionally written `0 < F ≤ 2`; a value around a half to one keeps the mean step useful, and values above one are only occasional tools for extra exploratory reach. I will take `F = 0.5` as the safe task default — and crucially I do *not* adapt it over time, because the difference vector already carries the time-adaptation; `F` only sets a fixed multiplier on it.

Now the indices. I need `r1, r2, r3` distinct from each other, and also distinct from the target index `i` whose slot I am trying to improve. Why distinct from each other? If `r2 = r3` the difference is the zero vector and there is no perturbation at all — a wasted trial. Why distinct from `i`? Because the move should be informed by *other* members, not coupled to the very vector I am about to replace; mixing the target into its own perturbation correlates the proposal with the thing it competes against. Picking three indices all different from `i` means I need at least four distinct members in the population, so `NP ≥ 4`. And `NP` itself, the population size, sets how rich my supply of difference vectors is: too small and the set of available differences is thin, the search stagnates for lack of varied directions; too large and I waste evaluations. Something on the order of several to ten times the dimension `D` gives enough difference diversity to span the search directions without bloating the evaluation count. Initialize the population uniformly over the box, since I have no prior; if I happened to know a nominal solution I would seed around it with small random deviations, but uniform is the honest default.

Let me sanity-check the mutation by itself before I complicate it. Suppose every trial vector were just `v = x_{r1} + F·(x_{r2} − x_{r3})`, accepted greedily if it beats its target. Does this actually work as a search? The members improve one slot at a time, the cloud contracts toward low-cost regions, the differences shrink with it, the steps anneal themselves — yes, the dynamics are sound. But I have a worry: the perturbation moves *all* `D` coordinates of `x_{r1}` at once, by the full difference vector. On a separable problem, where coordinates are independent, perturbing every coordinate simultaneously is wasteful — I would rather sometimes change just a few coordinates and let the good values of the others ride along. More generally I want a way to control *how many* coordinates of the mutant actually make it into the trial, to inject extra diversity and to tune the search to the dependency structure of the problem. So I will not let the mutant become the trial wholesale; I will *mix* the mutant with the target, coordinate by coordinate. This is the crossover step.

Concretely, build the trial `u` from the mutant `v` and the target `x_i` per coordinate: take `u_j = v_j` with probability `CR`, otherwise `u_j = x_{i,j}`. Each coordinate decided by an independent coin — a binomial choice across the `D` coordinates. The crossover rate `CR ∈ [0,1]` is now a dial on the dependency structure: with `CR` near 1 almost every coordinate comes from the mutant, so the trial is essentially the full difference-vector move — which is what I want on a *non-separable* problem like Rosenbrock, where the coordinates must move together along the curved valley and changing them one at a time would crawl. With `CR` near 0 only a coordinate or two changes per trial, which suits a *separable* problem where I can optimize axes nearly independently. So `CR` lets one algorithm cover both regimes. For the general non-separable case I will lean high, `CR ≈ 0.9`.

But wait — there is a degenerate case in this coordinate-wise coin flipping. If every coin comes up "keep the target" (all `randb(j) >= CR`, which can happen, especially for small `CR`), then `u = x_i` exactly: the trial is identical to the target, I evaluate `f` on a point I already know, and the whole iteration is wasted. I have to *force at least one coordinate* to come from the mutant. So I pick one random coordinate index `rnbr(i)` up front and always take that coordinate from `v`, regardless of its coin. Then the rule is: `u_j = v_j` if `randb(j) < CR` *or* `j = rnbr(i)`, else `u_j = x_{i,j}`. Now the construction cannot choose the all-target copy; at least one coordinate is structurally inherited from the mutant, and except for the accidental case where that mutant value equals the target value, the trial is a new point. That forced-coordinate fix is small but it is the difference between a method that spends each evaluation on a real proposal and one that sometimes spins.

(There is another way to choose which coordinates cross over: take a *contiguous block* of coordinates from the mutant, of a random length drawn so that longer blocks are geometrically less likely — an "exponential" scheme. It is a reasonable alternative, but the independent per-coordinate binomial choice is cleaner to reason about, has no positional bias from where the block happens to start, and gives `CR` a direct reading as "fraction of coordinates taken from the mutant." I will use binomial.)

Now acceptance. I have a trial `u` for target slot `i`. Greedy and one-to-one: compare `u` against *its own target* `x_i`, and let `u` take the slot in the next generation if and only if its cost is no worse — `cost(u) ≤ cost(x_i)`; otherwise `x_i` stays. Two things to notice about why this exact form is right. First, it is one-to-one — each trial competes only with the single member it was built to replace, not against the whole population. That keeps `NP` fixed and, more importantly, preserves diversity far better than a global "keep the best `NP`" truncation would: truncation would let a few good basins crowd out everything and collapse the population's spread (and with it my self-scaling difference vectors); one-to-one replacement lets a member that is merely the best *in its own lineage* survive even if it is mediocre globally, so the cloud stays spread and keeps supplying varied differences. Second, it is elitist per slot: since a slot only ever changes to something no worse, no slot can degrade, so the best cost in the population is monotone non-increasing — the method never throws away its best find. And does monotone, greedy, per-slot acceptance trap me in local minima the way plain greedy search did? No — and this is where the population earns its keep. A single member sliding into a local basin cannot drag the others with it, and the difference vectors from members exploring *other* basins keep proposing jumps that can pull a trapped member out. Greed plus population diversity gives me the best of both: monotone progress *and* escape from local minima, without ever accepting an uphill move the way annealing has to.

Let me assemble the whole loop now and see it whole. One generation sweeps the population: for each target index `i`, pick three other distinct members at random, form the difference-vector mutant `v = x_{r1} + F·(x_{r2} − x_{r3})`, binomial-cross it with `x_i` (forcing one coordinate from `v`) to get the trial `u`, clip `u` back into the box, evaluate `f(u)`, and replace `x_i` with `u` iff `u` is no worse. After `NP` such competitions the generation is done; record the best cost; repeat. The control variables are just `F`, `CR`, and `NP` — three numbers, all easy to set and robust, none of them a schedule or an adapted matrix. That count is the payoff: the scale-and-orientation problem that cost every prior method a tuning apparatus has been absorbed into the difference vector, which costs nothing.

Before I write it as code, let me name the family so the choices are explicit. The base vector is "rand," I use one difference vector, and the crossover is "bin" — DE/rand/1/bin. (Swap "rand"→"best" for the greedy base; use two differences for "/2"; that gives a small zoo of related schemes, e.g. a best-vector-plus-two-differences variant, but rand/1/bin is the robust general workhorse and the one I will ship.) Now the code fills the single generation slot in the population harness. The rule is cleanest as one population update, because the proposal for a target depends on three other members and the acceptance comparison belongs to that same target. I keep the fixed scaffold controls for the task, `F = 0.5` and `CR = 0.9`, and I build the next generation from the current one so the `G -> G+1` bookkeeping is explicit.

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
        # Three other members, mutually distinct and different from target i.
        candidates = list(range(len(population)))
        candidates.remove(i)
        r1, r2, r3 = random.sample(candidates, 3)
        x_r1, x_r2, x_r3 = population[r1], population[r2], population[r3]

        # DE/rand/1 mutation: v = x_r1 + F * (x_r2 - x_r3).
        mutant = creator.Individual(
            [x_r1[j] + F * (x_r2[j] - x_r3[j]) for j in range(dim)]
        )

        # Binomial crossover, with one forced coordinate from the mutant.
        j_rand = random.randrange(dim)
        trial = creator.Individual(
            [mutant[j] if (random.random() < CR or j == j_rand) else target[j]
             for j in range(dim)]
        )
        clip_individual(trial, lo, hi)
        trial.fitness.values = toolbox.evaluate(trial)

        # Greedy one-to-one selection: trial replaces target iff its cost is no worse.
        if trial.fitness.values[0] <= target.fitness.values[0]:
            next_population[i] = trial

    return next_population


def run_evolution(evaluate_func: Callable, dim: int, lo: float, hi: float,
                  pop_size: int, n_generations: int,
                  cx_prob: float, mut_prob: float, seed: int) -> Tuple[list, list]:
    """Differential Evolution, scheme DE/rand/1/bin."""
    random.seed(seed)
    _ = (cx_prob, mut_prob)  # retained for harness compatibility; F and CR are fixed above

    toolbox = base.Toolbox()
    toolbox.register("individual", make_individual, toolbox, dim, lo, hi)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_func)

    pop = toolbox.population(n=pop_size)           # init uniformly over the box
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

So the whole causal chain is this. I was stuck because every direct-search method I had answered the perturbation-scale question with an external device — a temperature schedule, a controlled or self-adapted σ, a fixed mutation width — each one a tuning burden and each one brittle when its assumptions did not match the landscape. Nelder–Mead showed it was possible to read the move scale off the configuration of points already in hand, but only for one collapsing simplex, only locally. I had a whole population, and its spread already *is* the scale I kept wanting: big while exploring, small while converging, and shaped like the valley because the members align with the contours. The difference of two random members is a sample of that spread, so `v = x_{r1} + F·(x_{r2} − x_{r3})` perturbs at exactly the right scale and orientation with no estimated distribution and no schedule — `F` is a single robust dial on an already-correct vector, not the whole scale. Binomial crossover with a forced coordinate mixes the mutant into the target so `CR` can tune how many coordinates move while ruling out the all-target copy. Greedy one-to-one acceptance makes the best cost monotone while one-to-one replacement preserves the diversity that lets the population escape local minima. Three control variables — `F`, `CR`, `NP` — no covariance, no cooling, no σ-adaptation, and the target competitions can be evaluated independently once the current generation is fixed. The scale problem dissolved into a subtraction.
