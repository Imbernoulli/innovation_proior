The search is the whole point, and the scaffold I start from does no search at all. Its default fill picks parents uniformly at random with no regard for fitness, its crossover returns a copy of `parent1`, and its mutation returns a copy of the parent. The loop's elitism dutifully carries forward whichever tree of generation zero happened to score lowest, but nothing ever improves on it, because no operator moves anyone in program space and no selection prefers good over bad. The outcome of a run is decided entirely by the luck of the initial draw. The floor I have to build first is therefore the one Koza built: turn the empty contract into a genuine Darwinian loop over expression trees.

I propose a standard, Koza-style genetic program. The first thing to fix is what is being searched, because the structure of the space forces every later choice. I have a sample $(X, y)$ and want an explicit formula that fits it. The candidates — expression trees — have no fixed size: one might be three nodes, another three hundred, a flat sum or a deep composition. There is no fixed-dimensional parameter vector, so there is nothing to take a gradient of. But I *can* score a candidate: run it on the data and measure error. That is precisely the interface a genetic algorithm needs — a population, a fitness, and operators that build new candidates from old — and nothing more; it never asks for smoothness or differentiability. So the engine is right, and the substrate has already settled the representation: the individual *is* the tree, and `get_all_nodes()` hands me every subtree with its parent pointer so I can splice. The load-bearing fact behind the whole design is that *any subtree of a valid expression is itself a valid expression* — so swapping subtrees between two parents always produces a syntactically valid, evaluable child. That is why crossover works on trees where it shattered on flat strings, and the substrate's protected operators close the other half: any subtree's output is a legal input to any operator.

The fitness is the least-squares error of the candidate's output against the targets, the mean squared error,
$$\text{MSE}(\text{tree}) = \frac{1}{N}\sum_{t=1}^{N}\big(y_t - \hat y_t\big)^2,$$
with lower better. Squared rather than absolute error because least squares is the objective I actually care about, it penalizes large misses heavily, and — the part that matters for selection — it gives a clean total order on candidates, which is all selection ever needs to ask. I keep `fitness_function` as *raw* MSE with no penalty baked in, because the outer loop tracks the best tree by exactly this number for elitism, stopping, and the final report; quietly mixing a complexity penalty into it would make the loop preserve and report the wrong tree. Any shaping a later rung wants belongs in selection, not here.

Selection is where the default is most obviously broken — uniform random choice applies zero fitness pressure, so the population drifts rather than improves. Holland's original answer, fitness-proportionate selection, has two failure modes I can see before coding it. Early in a run, one lucky tree with far lower error grabs an enormous reproductive share, floods the next generation with its children, and the population collapses onto one family — crossover then has nothing distinct to recombine, and the search converges prematurely. Late in a run the fitnesses compress into a narrow band, the proportionate shares become nearly equal, and selection loses its ability to discriminate exactly when I most want to prefer the slightly-better trees. Both come from the same root: proportionate selection cares about the *magnitudes* of fitness, and those magnitudes are badly behaved at both ends of a run and differ by orders of magnitude across three benchmarks. What I actually want is for selection to care only about *which* tree is better, not by how much. So I draw $k$ individuals at random and let the single lowest-error one win — a tournament of size $k$. It only ever *compares* fitnesses, never sums or weights them, so it is invariant to their scale and spread: a tree a thousand times better is treated like one barely better, it just wins its tournaments, and no super-fit individual can swamp the next generation because being selected still requires being drawn into one. Late in the run the better tree still wins whenever it meets a worse contender, so pressure is set by $k$ rather than by the accidental spread of scores. $k$ is the single greediness knob — $k=1$ is drift, large $k$ is nearly always-take-the-best — and $k=7$ is the standard default: strong enough to drive improvement, loose enough that average trees occasionally win and diversity survives. It is also cheap, $O(k)$ per draw with no global sum. Selection returns copies, so a parent can be drawn and reused without being mutated in place.

Crossover is the core operator. To make a child I copy `parent1` (so the original is never damaged — it may be selected again), copy `parent2` as the donor, pick a node in each, and replace the copy's chosen subtree with the donor's; because both endpoints are valid subexpressions the child is always valid. For *how* to pick the two points I commit to the plainest choice — uniform over the nodes. The offspring point is drawn from $1 \dots \text{size}-1$, excluding the root at index 0 so a single crossover does not wholesale-replace the offspring with the donor in the common case, and the donor point from $0 \dots \text{size}-1$ so the donor may contribute its whole tree. There is a known refinement — weighting function nodes more heavily than terminals, since in a binary-operator tree most nodes are leaves and a uniform point therefore usually swaps a lone terminal rather than a meaningful subexpression — but I deliberately decline it here: the point of standard GP is the plainest Koza-style loop, and the 90/10 weighting would muddy the question of what the bare engine does. The depth cap is the bloat brake and it lives right here: if crossover produces a child deeper than the cap, I reject it and return a copy of `parent1`.

Crossover has a built-in ceiling — it can only ever shuffle material already present in the population. If a primitive or a useful constant was never drawn in generation zero, or died out when a subpopulation went extinct, no recombination brings it back, because crossover splices existing pieces and cannot invent one. So I need a source of fresh material: subtree mutation, mirroring crossover, picks a node in a copy of the parent, deletes the subtree there, and grafts in a *freshly generated* random subtree via `generate_tree('grow', 3, n_features)`. The shallow grow at depth 3 keeps the injected noise small while reintroducing operators and constants the gene pool may have lost. Mutation is disruptive — it throws away a working subtree for random noise — so its rate stays small; the search leans overwhelmingly on crossover to assemble building blocks and uses mutation sparingly to keep the pool from going extinct. Same depth cap: too deep, return the parent unchanged.

The generation loop assembles these. Elitism comes first — copy the single best individual by raw MSE forward unconditionally, because the operators are stochastic and could destroy the current best, and I never want the best-of-generation to go backward. Then fill the rest by rolling a uniform random number per child: below the crossover rate (0.9), do subtree crossover on two tournament-selected parents; between 0.9 and 0.9 plus the mutation rate (0.95), mutate one tournament-selected parent; otherwise (the remaining 0.05) reproduce — copy one tournament-selected parent unchanged. Crossover at 0.9 is the workhorse, mutation at 0.05 the diversity drip, reproduction the chance for a good tree to pass forward intact. Repeat until the population reaches `pop_size`. Over 50 generations on a population of 500, with the best-ever tree tracked outside, the run reports the best tree it ever found.

What this floor should do is the entire reason to run it, because it is the rung every later rung will diagnose. On a *reachable* target — Koza-3's $x^5 - 2x^3 + x$, just repeated multiplication and addition of $x$, well inside the $\{\text{add},\text{sub},\text{mul}\}$ part of the function set and the depth cap — I expect steady, high R² across all three seeds. The harder benchmarks are where the bare engine should show its seams: Nguyen-7 must assemble logarithms and then extrapolate to a wider test range than it trained on, and Nguyen-10 must compose a product of a sine and a cosine in two variables. Both require finding and protecting a specific transcendental building block, and here the failure mode tournaments only *partly* fix is premature convergence — a single lineage that gets an early lead on the small training sample dominates the tournaments, the population homogenizes onto its *form*, and on a transcendental target that form is often wrong. Because the stuck point depends on which lineage won the early tournaments, the outcome should be strongly seed-dependent: some seeds land near 1.0, others lock onto a wrong form, and on the extrapolating Nguyen-7 a wrong form can diverge off the training range badly enough that R² goes sharply negative before the floor catches it at 0. The signature I expect, then, is high seed-to-seed variance on the transcendental benchmarks with Koza-3 steady and high — and that variance points straight at the next lever: selection, the thing that decides which lineage wins.

```python
def fitness_function(tree, X, y):
    """MSE fitness — lower is better."""
    y_pred = safe_evaluate(tree, X)
    return float(np.mean((y - y_pred) ** 2))


def selection(population, fitnesses, n_select, tournament_size=7):
    """Tournament selection."""
    selected = []
    pop_size = len(population)
    for _ in range(n_select):
        candidates = random.sample(range(pop_size), min(tournament_size, pop_size))
        best = min(candidates, key=lambda i: fitnesses[i])
        selected.append(population[best].copy())
    return selected


def crossover(parent1, parent2, n_features, max_depth=17):
    """Standard subtree crossover."""
    offspring = parent1.copy()
    donor = parent2.copy()

    # Pick random crossover points
    off_size = offspring.size()
    don_size = donor.size()
    if off_size <= 1 or don_size <= 1:
        return offspring

    off_point = random.randint(1, off_size - 1)
    don_point = random.randint(0, don_size - 1)

    # Extract donor subtree
    donor_nodes = donor.get_all_nodes()
    donor_subtree = donor_nodes[don_point][0].copy()

    # Replace in offspring
    off_nodes = offspring.get_all_nodes()
    node, parent, child_idx = off_nodes[off_point]
    if parent is not None:
        parent.children[child_idx] = donor_subtree
    else:
        offspring = donor_subtree

    # Reject if too deep
    if offspring.depth() > max_depth:
        return parent1.copy()

    return offspring


def mutation(parent, n_features, max_depth=17):
    """Subtree mutation — replace a random subtree with a new random tree."""
    offspring = parent.copy()
    tree_size = offspring.size()
    if tree_size <= 1:
        return generate_tree('grow', 3, n_features)

    mut_point = random.randint(1, tree_size - 1)
    new_subtree = generate_tree('grow', 3, n_features)

    nodes = offspring.get_all_nodes()
    node, par, child_idx = nodes[mut_point]
    if par is not None:
        par.children[child_idx] = new_subtree
    else:
        offspring = new_subtree

    if offspring.depth() > max_depth:
        return parent.copy()

    return offspring


def evolve_one_generation(population, fitnesses, X_train, y_train,
                          n_features, pop_size,
                          crossover_rate=0.9, mutation_rate=0.05,
                          max_depth=17):
    """Standard GP generation with elitism."""
    new_population = []

    # Elitism: keep best
    elite_idx = int(np.argmin(fitnesses))
    new_population.append(population[elite_idx].copy())

    while len(new_population) < pop_size:
        r = random.random()
        if r < crossover_rate:
            parents = selection(population, fitnesses, 2)
            child = crossover(parents[0], parents[1], n_features, max_depth)
        elif r < crossover_rate + mutation_rate:
            parents = selection(population, fitnesses, 1)
            child = mutation(parents[0], n_features, max_depth)
        else:
            parents = selection(population, fitnesses, 1)
            child = parents[0]
        new_population.append(child)

    return new_population[:pop_size]
```
