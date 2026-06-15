I start with the failure mode I can actually observe. I evolve a population of expression trees for symbolic regression, and after a quiet opening where the average size barely moves, the mean node count often starts climbing generation after generation while the fitness stops improving. I do not want to confuse this with legitimate growth: a hard target may need a larger expression. The bad case is growth that buys no fitness. It makes evaluation slower, makes the final expression harder to understand, and usually makes generalisation worse.

The old practical answer is to make large programs pay rent. In the maximised-fitness convention, I select on a penalised fitness

  f_p(x) = f(x) - c*l(x),

where l(x) is program size and c is a constant pressure against length. The original f still matters for recognising a solution; the penalty is only meant to steer selection. This is sensible as a soft constraint or MDL-flavoured tradeoff, but it leaves me with the only question that matters: what should c be? If c is too small, bloat continues. If c is too large, GP treats shrinking as the real objective and collapses toward tiny useless programs. Worse, the good value changes with the problem, primitive set, population, selection scheme, and generation. Zhang and Mühlenbein's adaptive coefficient idea tells me that changing c over time can help, but it still does not tell me what value gives a specified change in mean size. I need to compute the pressure from the current population rather than tune it.

So I ask what determines the expected change in mean size. The size evolution equation gives me the starting point. For selection followed by symmetric subtree crossover, where crossover does not favour one parent order over the other, the expected next mean size depends only on which length classes selection chooses:

  E[mu(t+1)] = sum_l l*p(l,t).

Here p(l,t) is the probability that a selection event chooses a size-l program. If Phi(l,t) is the fraction of the current population with that size, then the current mean is mu(t) = sum_l l*Phi(l,t), so

  E[Delta mu] = sum_l l*(p(l,t) - Phi(l,t)).

This already tells me the right lever. Crossover may reshape the distribution, but under the symmetry condition it does not move the mean in expectation. The mean moves only because selection over- or under-samples length classes. If long programs are picked more often than their population share, the mean grows; if short programs are favoured, the mean shrinks. To control bloat, I have to make selection neutral with respect to the length advantage it is currently exploiting.

To turn that into an equation for fitness, I first use fitness-proportionate selection because there p(l,t) has a closed form. If fbar(l,t) is the average fitness among size-l programs and fbar(t) is the population mean fitness, then

  p(l,t) = Phi(l,t)*fbar(l,t)/fbar(t).

Substituting this into the size-change equation gives

  E[Delta mu] = (1/fbar(t))*sum_l l*(fbar(l,t) - fbar(t))*Phi(l,t).

This looks almost like a covariance, except covariance wants l - mu(t), not l. I check whether inserting l - mu(t) changes anything. I split l into (l - mu(t)) + mu(t):

  sum_l l*(fbar(l,t) - fbar(t))*Phi(l,t)
    = sum_l (l - mu(t))*(fbar(l,t) - fbar(t))*Phi(l,t)
      + mu(t)*sum_l (fbar(l,t) - fbar(t))*Phi(l,t).

The first term is Cov(l,f). The second term vanishes because the population-weighted deviation of size-class fitness from the population mean is zero:

  sum_l (fbar(l,t) - fbar(t))*Phi(l,t) = fbar(t) - fbar(t) = 0.

So the cross-term with mu(t) disappears for a precise reason, not by handwaving, and I get

  E[Delta mu] = Cov(l,f)/fbar(t).

This is Price's theorem in the language of program size: the expected one-generation change in the mean of a heritable feature is its covariance with fitness divided by mean fitness. For this problem, the important reading is operational. In the maximised-fitness convention, a positive covariance between length and fitness means selection has a length advantage to exploit, so mean size grows.

Now I bring the size penalty back in without assuming it is linear. I write the penalised fitness as

  f_p(x,t) = f(x) - g(l(x),t),

where g is any generation-dependent function of size. Reusing the same covariance identity with f_p gives

  E[Delta mu] = Cov(l,f_p)/fbar_p
              = Cov(l,f - g)/(fbar - gbar)
              = (Cov(l,f) - Cov(l,g))/(fbar - gbar).

For no expected growth, I set E[Delta mu] = 0. Provided the denominator is not zero, the condition is simply

  Cov(l,g) = Cov(l,f).

This is the core design condition. The penalty must have exactly the same covariance with size as the raw fitness currently has. At the no-growth point, the mean-fitness denominator drops out because I am forcing the numerator to zero.

If I choose the traditional linear penalty, g(l,t) = c(t)*l, the condition becomes

  Cov(l,g) = Cov(l,c(t)*l) = c(t)*Cov(l,l) = c(t)*Var(l).

Setting this equal to Cov(l,f) forces

  c(t) = Cov(l,f)/Var(l).

That is the coefficient I need for zero expected size change under the linear pressure. It is dynamic because both pieces of the ratio change during the run. It is also the ordinary least-squares slope of fitness against size: the per-node fitness advantage selection currently sees. Subtracting that slope times length removes the linear size-fitness advantage from the selection signal.

I check the generalisation to make sure I am not only fitting the traditional special case. If I use a power penalty

  g(l,t) = c(t)*l^k,

then the same no-growth condition gives

  c(t) = Cov(l,f)/Cov(l,l^k).

The case k = 1 reduces to Cov(l,f)/Var(l). If I instead centre the linear penalty,

  g(l,t) = c(t)*(l - mu(t)),

then Cov(l,mu(t)) = 0 because mu(t) is constant across the population, so Cov(l,g) = c(t)*Var(l) again. The covariance condition, not the surface form of the penalty, is the real object.

Freezing size is only one target. If I want the expected mean size to follow a chosen trajectory gamma, I impose

  E[mu(t+1)] = gamma(t+1).

Since E[mu(t+1)] = mu(t) + E[Delta mu], the constraint is

  (Cov(l,f) - Cov(l,g))/(fbar - gbar) + mu(t) = gamma(t+1).

For g(l,t) = c(t)*l^k, I have gbar = c(t)*E[l^k] and Cov(l,g) = c(t)*Cov(l,l^k). Writing delta = gamma(t+1) - mu(t), I solve

  (Cov(l,f) - c(t)*Cov(l,l^k))/(fbar - c(t)*E[l^k]) = delta,

which gives

  c(t) = (Cov(l,f) - delta*fbar)/(Cov(l,l^k) - delta*E[l^k])
       = (Cov(l,f) - (gamma(t+1) - mu(t))*fbar)
         /(Cov(l,l^k) - (gamma(t+1) - mu(t))*E[l^k]).

For k = 1 this becomes

  c(t) = (Cov(l,f) - (gamma(t+1) - mu(t))*fbar)
         /(Var(l) - (gamma(t+1) - mu(t))*mu(t)).

If I set gamma(t+1) = mu(t), delta is zero and I recover Cov(l,f)/Var(l). If I instead use the initial mean as the setpoint, gamma(t+1) = mu(0), the formula becomes

  c(t) = (Cov(l,f) - (mu(0) - mu(t))*fbar)
         /(Cov(l,l^k) - (mu(0) - mu(t))*E[l^k]).

This matters because the zero-growth equation controls the mean only in expectation. In a finite population, sampling noise can let the realised mean drift. Anchoring to mu(0) adds a restoring term: if the population drifts above or below the initial mean, the next coefficient pushes it back.

I also keep the selection-scheme boundary clear. The exact covariance derivation uses fitness-proportionate selection. The size evolution equation itself is broader, because it is written in terms of selection probabilities p(l,t), so it applies before I specify how selection probabilities are produced. Tournament selection does not give me the same simple p(l,t) = Phi(l,t)*fbar(l,t)/fbar(t) expression, so the coefficient is not an exact theorem for every tournament. But the ratio still estimates the linear size-fitness slope in the current population, and tournament selection amplifies the same ordering. Cancelling that slope is therefore the natural practical coefficient, and it is the one that carries over without changing the operator.

Now I translate the math into the task implementation. The theoretical equations use a maximised fitness and write the penalty as f - c*l. The MLS-style symbolic-regression scaffold gives me mean squared error, where lower is better. Under that convention, making large trees worse means selecting on

  penalized_error = MSE + c*l.

This matches the gplearn sign convention: `_Program.fitness()` returns `raw_fitness_ - c*len(program)*metric.sign`, and for MSE `metric.sign = -1`, so the penalised value is raw error plus c times length. Selection minimises that value. The raw MSE remains the truth for elitism, stopping, and reporting; the penalty is only a parent-selection signal.

For the coefficient, I preserve the gplearn automatic rule exactly at the measurement step:

  auto_c = np.cov(length, fitness)[1, 0] / np.var(length).

In the MLS-Bench edit surface, the function receives raw lower-better fitnesses, so I compute that ratio from the current lengths and raw MSE vector, guard the degenerate zero-variance case, and then apply the task's practical clamp to [0, 0.001]. The lower clamp prevents a negative coefficient from rewarding length under the lower-better convention, and the upper clamp keeps one noisy generation from making selection ignore fit. The generation loop is then straightforward: keep the raw-MSE elite, build the penalised vector `MSE + c*length`, and use that vector only inside tournament selection while crossover, mutation, and reproduction fill the rest of the population.

Putting it into the four contract functions plus the per-generation loop, the size-control logic is the few lines that compute c and form the penalised vector; everything else is the generic GP machinery I already had.

```python
import random
import numpy as np

# GP toolkit primitives already provided:
#   safe_evaluate(tree, X) -> predictions
#   generate_tree(method, max_depth, n_features) -> Tree   (method = 'grow' | 'full')
#   Tree.copy() / Tree.size() / Tree.depth() / Tree.get_all_nodes()


def fitness_function(tree, X, y):
    # Raw MSE, lower is better. The size penalty lives in selection only, so the
    # main loop recognises solutions and tracks the best by their true fit.
    y_pred = safe_evaluate(tree, X)
    return float(np.mean((y - y_pred) ** 2))


def selection(population, fitnesses, n_select, tournament_size=7):
    # Plain tournament selection; handed the penalised vector, it becomes size-aware.
    selected = []
    pop_size = len(population)
    for _ in range(n_select):
        candidates = random.sample(range(pop_size), min(tournament_size, pop_size))
        best = min(candidates, key=lambda i: fitnesses[i])   # lower is better
        selected.append(population[best].copy())
    return selected


def crossover(parent1, parent2, n_features, max_depth=17):
    # Standard subtree crossover: graft a random subtree of parent2 into parent1.
    offspring = parent1.copy()
    donor = parent2.copy()
    off_size, don_size = offspring.size(), donor.size()
    if off_size <= 1 or don_size <= 1:
        return offspring
    off_point = random.randint(1, off_size - 1)
    don_point = random.randint(0, don_size - 1)
    donor_subtree = donor.get_all_nodes()[don_point][0].copy()
    node, parent, child_idx = offspring.get_all_nodes()[off_point]
    if parent is not None:
        parent.children[child_idx] = donor_subtree
    else:
        offspring = donor_subtree
    if offspring.depth() > max_depth:          # depth cap (Koza): reject, keep a parent
        return parent1.copy()
    return offspring


def mutation(parent, n_features, max_depth=17):
    # Standard subtree mutation: replace a random subtree with a fresh random one.
    offspring = parent.copy()
    tree_size = offspring.size()
    if tree_size <= 1:
        return generate_tree('grow', 3, n_features)
    mut_point = random.randint(1, tree_size - 1)
    new_subtree = generate_tree('grow', 3, n_features)
    node, par, child_idx = offspring.get_all_nodes()[mut_point]
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
    # c = Cov(len, fit) / Var(len): the OLS slope of fitness on size, the value
    # that cancels the size-fitness coupling and sets E[Delta mu] = 0 this gen.
    lengths = np.array([tree.size() for tree in population], dtype=float)
    raw_fit = np.array(fitnesses, dtype=float)
    length_var = float(np.var(lengths))
    if length_var > 1e-15:
        auto_c = float(np.cov(lengths, raw_fit)[1, 0]) / length_var
        c = min(max(auto_c, 0.0), 0.001)       # no negative c (would reward bloat); mild ceiling
    else:
        c = 0.0

    # Penalty ADDED (lower = better here): penalised = MSE + c * size. Selection only.
    penalized = [f + c * l for f, l in zip(fitnesses, lengths)]

    new_population = []
    # Elitism on RAW fitness: the true-best individual is always preserved.
    new_population.append(population[int(np.argmin(fitnesses))].copy())

    while len(new_population) < pop_size:
        r = random.random()
        if r < crossover_rate:
            parents = selection(population, penalized, 2)
            child = crossover(parents[0], parents[1], n_features, max_depth)
        elif r < crossover_rate + mutation_rate:
            parent = selection(population, penalized, 1)[0]
            child = mutation(parent, n_features, max_depth)
        else:
            child = selection(population, penalized, 1)[0]
        new_population.append(child)

    return new_population[:pop_size]
```

The chain now holds together. I begin with bloat as a selection-caused mean-size drift. The size evolution equation tells me mean size changes only through length-class selection probabilities. Fitness-proportionate selection turns that change into Price's covariance formula. A general size penalty changes the numerator from Cov(l,f) to Cov(l,f) - Cov(l,g). No growth therefore requires Cov(l,g) = Cov(l,f). The linear choice g = c*l forces c = Cov(l,f)/Var(l), the power and target-tracking formulas follow by the same one-unknown equation, and the implementation uses the same covariance-over-variance rule while respecting the lower-is-better MSE sign convention and the MLS-Bench function surface.
