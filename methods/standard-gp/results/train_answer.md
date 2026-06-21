The recurring task across the empirical sciences is to take a finite sample of measurements — pairs $(x, y)$ of inputs and a measured output — and recover the *mathematical relationship* that produced them, written as an explicit symbolic formula a human can read and manipulate: $y = \log(x+1) + \log(x^2+1)$, $y = 2\sin(x)\cos(z)$, a polynomial, a rational function. The tools one reaches for first do not even fit the question. Linear, quadratic, and higher-order polynomial regression each hand me a fixed *template* — $y = a_0 + a_1 x + a_2 x^2 + \cdots$ — and solve in closed form for the coefficients $a_i$ that minimize $\sum (y - \hat y)^2$. That is exact and beautiful when I already know the shape of the answer, but the whole difficulty is that I do not: I do not know whether the truth is a polynomial, a sum of sinusoids, something with a logarithm, or a ratio. If the data really came from $\log(x+1)+\log(x^2+1)$, no polynomial template can ever recover it — a transcendental function is not a finite polynomial — and least squares just returns a high-order polynomial that wiggles through the sample and generalizes terribly. The missing piece is not a better coefficient solver; it is the ability to choose the functional *form* itself, and that form is precisely the answer, not an input.

Reframing this as a search exposes what makes it unlike any optimization I already know. I want to search the space of *all expressions* for one that fits the data, but the objects I am searching over have no fixed size: an expression might be three symbols or three hundred, a flat sum or a deeply nested composition. There is no fixed-dimensional parameter vector — the number of "knobs" is not even constant from one candidate to the next — so gradient descent has no $\theta$ to move and coefficient fitting has no fixed set of coefficients to solve. The objects are hierarchical, combinatorial, and non-numeric; I can score a candidate but I cannot differentiate through this space or even write down a coordinate system for it. What I do have for scoring-only search is Holland's genetic algorithm: keep a whole population of candidates, score each with a fitness function, and breed the population forward like natural selection — let the fitter ones make more offspring, recombine pairs, sprinkle in mutation, repeat for generations, and watch average fitness climb. It asks almost nothing of the space — never a gradient, never smoothness — only that I can evaluate fitness and that I have operators to make new candidates from old ones. The wall is that every bit of the genetic algorithm's clean theory — the schema counting, the building-block story — lives on the *fixed-length string* chromosome. Forcing my candidates into a fixed-length string means deciding their size up front, which is deciding before I start how big the answer is allowed to be; and a one-point crossover that cuts a flat string at a random byte slices through the middle of a nested subexpression and produces garbage that does not parse. The selection-and-recombination engine is exactly right; the string chromosome is the one piece that does not reach the problem.

So I keep the engine and change what the individual *is*. I propose Standard Genetic Programming: run Holland's Darwinian loop directly on a population of **expression trees**. An expression like $0.234\,z + x - 0.789$ is not flat — it is $+$ applied to $(0.234\,z)$ and to $(x-0.789)$, which decompose further — it is a parse tree, with operators at the internal nodes and variables and constants at the leaves. Two structural facts make this representation load-bearing: a tree can be any size and any shape, so there is no fixed length to commit to; and **any subtree of a valid expression is itself a valid expression**. Let the individual *be* the tree — genotype and phenotype are the same object, over a chosen *function set* $F$ (the operators I allow: $+, -, \cdot$, protected $/$, and $\sin, \cos, \log, \exp$ where needed) and a *terminal set* $T$ (the input variables plus constants). Selection is unchanged: fitness is fitness, I can pick the fitter trees regardless of shape. The whole question collapses onto crossover, and the answer is forced by the subtree fact. The disease of string crossover was that cutting at a random point destroys structure; with trees I have a unit of structure that is *always self-contained* — the subtree. Pick a node in each parent (on copies, so the originals are never damaged) and swap the two subtrees rooted there. Because a subtree is itself a valid expression, *every* such swap replaces a valid subexpression with another valid subexpression, so the child always parses and always evaluates — the closure problem that wrecked string crossover simply does not exist here, solved by the choice of representation. And it does exactly what crossover is meant to do in the building-block picture: useful subexpressions like $\log(x+1)$ or $\sin(x)$ or a good constant subtree are the building blocks, and subtree exchange recombines them across individuals while selection steers trials toward the parents that carry them.

Around that core, each remaining choice falls out of a concrete failure mode. Closure has a second, sneakier face: because crossover relentlessly reconstructs arbitrary operator-operand combinations, an evolved tree will contain $\mathrm{div}(\cdot,\, x-x)$ with a zero divisor, or $\log$ of a negative number, or $\exp$ of a huge number, and ordinary arithmetic then returns $\mathrm{NaN}$ or $\mathrm{Inf}$ and poisons every fitness comparison. I cannot forbid these subtrees, so I make the *primitives themselves* total: protected division returns $1.0$ on a near-zero divisor; protected log uses $\log(|a|)$ and returns $0$ near zero; protected exp clips its exponent to a safe range such as $[-10, 10]$; and as a final net, after evaluation any stray $\mathrm{NaN}/\mathrm{Inf}$ is replaced and the output clipped finite. This is the *evaluation-safety* half of closure; the partner is *type consistency* — every function takes and returns the same real-valued type, so any subtree can legally sit in any argument slot, which is exactly what unrestricted subtree crossover demands. $F$ must also satisfy *sufficiency*: it has to be *able* to express a solution or the search is doomed to approximate — transcendental targets need $\sin, \cos, \log, \exp$, whereas $\{+, -, \cdot\}$ suffices for a polynomial like $x^6 - 2x^4 + x^2$; since an unused primitive rarely hurts but a missing one is fatal, I err toward including what the target plausibly needs. Real coefficients like $0.234$ or the $2$ in $2\sin(x)\cos(z)$ cannot be enumerated in a finite $T$, so I add one **ephemeral random constant** terminal: whenever it is selected during tree construction, I draw a random number and *freeze* it permanently into that leaf. The initial population thus scatters random constants, and from then on crossover only *moves and combines* them — a $\cdot$ node scales a subtree, a $+$ node combines $-0.402$ and $-0.583$ into something near $-0.985$ — so coefficients are *evolved* by selection rather than solved by any numerical optimizer.

For generation zero I want trees varied in both size and shape, since the initial spread is the raw material everything downstream recombines. The `full` method (always choose operators until the depth limit, then terminals) gives bushy trees of uniform depth but poor *shape* variety; the `grow` method (choose freely from operators or terminals) gives irregular shapes but a size distribution wildly sensitive to the terminal-to-function ratio, so a primitive set with many terminals produces stunted trees regardless of the limit. Neither alone suffices, so I use **ramped half-and-half**: ramp the depth limit over a range (say 2 up to 6) and at each depth build half the trees with `full` and half with `grow`, getting a spread of sizes and, within each size, a spread of shapes; the initial limit stays modest (around 6) so generation zero is not enormous. For selection I deliberately reject Holland's fitness-proportionate default. Proportionate selection depends on the *magnitudes* of the fitness values, and that dependence fails at both ends of a run: early, one lucky low-error individual takes a huge share and floods the next generation with copies, collapsing diversity so crossover has nothing distinct to recombine; late, as errors compress into a narrow band the shares become nearly equal and selection pressure evaporates just when I still want to prefer slightly-better trees. What I want is for selection to care only about *which* tree is better, not by how much — invariant to the scale and spread of the scores. So I use **tournament selection**: draw $k$ individuals uniformly with replacement and let the single lowest-error one be the selected parent. It only ever *compares* fitnesses, never sums or weights them, so a tree a thousand times better is treated like one barely better — it simply wins its tournaments — and no super-fit individual can swamp the next generation because it still has to be drawn into a finite number of tournaments. Selection pressure is set by $k$ rather than by the accidental spread of scores: $k=1$ is drift, large $k$ is near-greedy, and a moderate $k = 7$ is a sensible default, cheap at $O(k)$ per draw with no global sum.

The operators that build the new generation then follow, with their rates forced by the dynamics. Crossover is subtree exchange, but *how* I pick the crossover point matters: with binary operators dominating, the branching factor is near two, so most nodes are leaves, and a uniform point choice would mostly swap a single terminal — a trivial exchange that barely moves me in program space. I therefore bias the choice with the **90/10 convention**, weighting each function node by $0.9$ and each terminal by $0.1$ before normalizing across the actual nodes, so an internal node is nine times as likely as a terminal and the typical crossover swaps a genuine subexpression. Crossover has a ceiling, though — it can only shuffle material already present in the population, so if a needed primitive like $\cos$ is absent from every tree, no recombination can ever bring it back. The cure is **subtree mutation**: pick a node, delete its subtree, and graft a freshly grown random subtree (mechanically, cross the tree against a fresh random tree — "headless chicken" mutation), injecting primitives and constants the gene pool may lack. Mutation is disruptive, so its rate stays small while crossover does the assembly. The leftover probability goes to **reproduction**: copy a selected individual unchanged, the survival half of survival-of-the-fittest. With crossover at $0.9$ and mutation at $0.05$, reproduction takes the remaining $0.05$: roll a uniform $r$; below $0.9$ do crossover on two tournament-selected parents, between $0.9$ and $0.95$ mutate one, otherwise copy one. Two further safeguards are forced. **Elitism**: since the stochastic operators can destroy the current best, I copy the single best individual of each generation forward unconditionally, making the population's best monotone non-worsening at the cost of one slot, and report the best tree ever seen as the run's result. And a **depth cap** against **bloat** — the phenomenon where trees grow without fitness gain by accumulating inert "intron" subexpressions, because near a fitness plateau larger trees are statistically more likely to survive crossover unharmed, so selection quietly rewards size for its protective value; unchecked, this balloons the trees and destroys the very interpretability I wanted. A hard cap (say 17) rejects any over-deep offspring and returns a parent copy instead, deep enough for any expression I would actually want yet a firm ceiling, with the smaller init limit keeping generation zero compact. Finally, fitness is just the mean squared error of the candidate against the targets over the sample points, $\mathrm{MSE} = \frac{1}{n}\sum (\hat y - y)^2$, lower being better — the least-squares objective, and exactly the ordering a tournament needs to rank by. Run that loop for a fixed number of generations on a ramped-half-and-half initial population and report the best tree ever seen, and the discovered formula comes out of the data without my ever having chosen its form.

```python
import random
import numpy as np


def _is_function_node(node):
    return bool(getattr(node, "children", ()))


def _choose_subtree_index(tree):
    """Choose a crossover point with function nodes weighted 0.9 and leaves 0.1."""
    nodes = tree.get_all_nodes()
    weights = [0.9 if _is_function_node(node) else 0.1
               for node, _parent, _child_idx in nodes]
    return random.choices(range(len(nodes)), weights=weights, k=1)[0]


def _replace_subtree(tree, point, replacement):
    """Replace the subtree at point in a copied tree, including replacement at the root."""
    _node, parent, child_idx = tree.get_all_nodes()[point]
    if parent is None:
        return replacement.copy()
    parent.children[child_idx] = replacement.copy()
    return tree


def fitness_function(tree, X, y):
    """MSE fitness over the fitness cases — lower is better."""
    y_pred = safe_evaluate(tree, X)
    return float(np.mean((y - y_pred) ** 2))


def selection(population, fitnesses, n_select, tournament_size=7):
    """Tournament selection: sample k contenders, keep the best (min MSE). Rank-only, so it is
    invariant to fitness scale and holds selection pressure constant."""
    selected = []
    pop_size = len(population)
    k = max(1, min(tournament_size, pop_size))
    for _ in range(n_select):
        candidates = [random.randrange(pop_size) for _ in range(k)]
        best = min(candidates, key=lambda i: fitnesses[i])
        selected.append(population[best].copy())     # copy: parents may be reused
    return selected


def crossover(parent1, parent2, n_features, max_depth=17):
    """Subtree crossover: graft a subtree of parent2 into parent1. Any subtree is a valid
    expression, so the offspring is always valid (closure)."""
    offspring = parent1.copy()
    donor = parent2.copy()

    off_point = _choose_subtree_index(offspring)
    don_point = _choose_subtree_index(donor)
    donor_subtree = donor.get_all_nodes()[don_point][0]
    offspring = _replace_subtree(offspring, off_point, donor_subtree)

    if offspring.depth() > max_depth:                 # bloat brake
        return parent1.copy()
    return offspring


def mutation(parent, n_features, max_depth=17):
    """Subtree mutation: replace a random subtree with a freshly grown random subtree —
    injects material crossover cannot reintroduce."""
    offspring = parent.copy()

    # Headless-chicken mutation: build a random donor and graft one of its subtrees.
    donor = generate_random('half and half', min(6, max_depth), n_features)
    donor_point = _choose_subtree_index(donor)
    new_subtree = donor.get_all_nodes()[donor_point][0]

    mut_point = _choose_subtree_index(offspring)
    offspring = _replace_subtree(offspring, mut_point, new_subtree)

    if offspring.depth() > max_depth:
        return parent.copy()
    return offspring


def evolve_one_generation(population, fitnesses, X_train, y_train,
                          n_features, pop_size,
                          crossover_rate=0.9, mutation_rate=0.05,
                          max_depth=17):
    """One generation: elitism, then breed by tournament selection + crossover / mutation /
    reproduction in the given proportions."""
    new_population = []

    elite_idx = int(np.argmin(fitnesses))            # elitism: best survives
    new_population.append(population[elite_idx].copy())

    while len(new_population) < pop_size:
        r = random.random()
        if r < crossover_rate:                        # subtree crossover (~0.9)
            parents = selection(population, fitnesses, 2)
            child = crossover(parents[0], parents[1], n_features, max_depth)
        elif r < crossover_rate + mutation_rate:      # subtree mutation (~0.05)
            parents = selection(population, fitnesses, 1)
            child = mutation(parents[0], n_features, max_depth)
        else:                                         # reproduction (remainder)
            parents = selection(population, fitnesses, 1)
            child = parents[0]
        new_population.append(child)

    return new_population[:pop_size]
```
