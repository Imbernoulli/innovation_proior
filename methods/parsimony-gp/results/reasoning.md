I start with the failure mode I can actually observe. I evolve a population of expression trees for symbolic regression, and after a quiet opening where the average size barely moves, the mean node count often starts climbing generation after generation while the fitness stops improving. I do not want to confuse this with legitimate growth: a hard target may need a larger expression. The bad case is growth that buys no fitness. It makes evaluation slower, makes the final expression harder to understand, and usually makes generalisation worse.

The old practical answer is to make large programs pay rent. In the maximised-fitness convention, I select on a penalised fitness

  f_p(x) = f(x) - c*l(x),

where l(x) is program size and c is a constant pressure against length. The original f still matters for recognising a solution; the penalty is only meant to steer selection. This is sensible as a soft constraint or MDL-flavoured tradeoff, but it leaves me with the only question that matters: what should c be? If c is too small, bloat continues. If c is too large, GP treats shrinking as the real objective and collapses toward tiny useless programs. Worse, the good value changes with the problem, primitive set, population, selection scheme, and generation. Zhang and Mühlenbein's adaptive coefficient idea tells me that changing c over time can help, but it still does not tell me what value gives a specified change in mean size. I would rather compute the pressure from the current population than tune it by hand.

So I ask what determines the expected change in mean size. The size evolution equation gives me a starting point. For selection followed by symmetric subtree crossover, where crossover does not favour one parent order over the other, the expected next mean size depends only on which length classes selection chooses:

  E[mu(t+1)] = sum_l l*p(l,t).

Here p(l,t) is the probability that a selection event chooses a size-l program. If Phi(l,t) is the fraction of the current population with that size, then the current mean is mu(t) = sum_l l*Phi(l,t), so

  E[Delta mu] = sum_l l*(p(l,t) - Phi(l,t)).

I want to be sure I am reading this correctly before I lean on it, so I put numbers to it. Take four programs with sizes l = (2, 3, 5, 8), each its own singleton size class with population share 1/4, so mu(t) = 4.5. Give them raw fitnesses f = (1, 2, 2, 5) under the maximised convention, so fbar(t) = 2.5; I deliberately let the larger programs be fitter so there is a length advantage to detect. Under fitness-proportionate selection p is just f/sum(f) = (1, 2, 2, 5)/10, and sum_l l*p = (2+6+10+40)/10 = 5.8. So E[Delta mu] = 5.8 - 4.5 = 1.3: the mean grows by 1.3 nodes in one generation, purely because selection over-samples the long-and-fit programs. That is bloat in miniature, and it confirms what the equation is supposed to say — crossover never entered the calculation, only selection did. The mean moves because selection over- or under-samples length classes: if long programs are picked more often than their population share, the mean grows. To control bloat I have to neutralise whatever length advantage selection is currently exploiting.

To turn that into an equation for the fitness I am selecting on, I use fitness-proportionate selection, because there p(l,t) has a closed form. If fbar(l,t) is the average fitness among size-l programs and fbar(t) is the population mean fitness, then

  p(l,t) = Phi(l,t)*fbar(l,t)/fbar(t).

Substituting this into the size-change equation gives

  E[Delta mu] = (1/fbar(t))*sum_l l*(fbar(l,t) - fbar(t))*Phi(l,t).

This looks almost like a covariance between size and fitness, except a covariance wants l - mu(t), not l. So I check whether replacing l with l - mu(t) changes the sum. I split l into (l - mu(t)) + mu(t):

  sum_l l*(fbar(l,t) - fbar(t))*Phi(l,t)
    = sum_l (l - mu(t))*(fbar(l,t) - fbar(t))*Phi(l,t)
      + mu(t)*sum_l (fbar(l,t) - fbar(t))*Phi(l,t).

The first term is exactly Cov(l,f). The second term has the constant mu(t) pulled out front, multiplying

  sum_l (fbar(l,t) - fbar(t))*Phi(l,t) = sum_l fbar(l,t)*Phi(l,t) - fbar(t)*sum_l Phi(l,t)
                                       = fbar(t) - fbar(t)*1 = 0,

because the population-weighted average of the size-class mean fitnesses is just the overall mean fitness, and the shares sum to one. So the cross-term vanishes identically, and

  E[Delta mu] = Cov(l,f)/fbar(t).

Before trusting the algebra I run it against the same four programs. Population covariance Cov(l,f) = sum Phi*(l-mu)*(f-fbar) = [(-2.5)(-1.5) + (-1.5)(-0.5) + (0.5)(-0.5) + (3.5)(2.5)]/4 = [3.75 + 0.75 - 0.25 + 8.75]/4 = 13/4 = 3.25. Then Cov(l,f)/fbar = 3.25/2.5 = 1.3 — the same 1.3 I got by direct enumeration of sum_l l*p. The covariance form and the brute-force form agree, so the cross-term really did drop out and I have not lost a constant somewhere.

This is Price's theorem written in the language of program size: the expected one-generation change in the mean of a heritable feature is its covariance with fitness divided by mean fitness. The operational reading is what I care about. In the maximised-fitness convention, a positive covariance between length and fitness means selection has a length advantage to exploit, so the mean grows — exactly the +1.3 I just watched happen.

Now I bring the size penalty back in, and I deliberately do not assume it is linear yet. I write the penalised fitness as

  f_p(x,t) = f(x) - g(l(x),t),

where g is any generation-dependent function of size. Reusing the same covariance identity with f_p in place of f gives

  E[Delta mu] = Cov(l,f_p)/fbar_p
              = Cov(l,f - g)/(fbar - gbar)
              = (Cov(l,f) - Cov(l,g))/(fbar - gbar),

since covariance is linear in its argument and the mean of a difference is the difference of means. For no expected growth I set E[Delta mu] = 0. Provided the denominator is not zero, that forces the numerator to zero, i.e.

  Cov(l,g) = Cov(l,f).

So the penalty has to carry exactly the same covariance with size as the raw fitness currently does — no more, no less. At this no-growth point the mean-fitness denominator drops out of the condition entirely, because I am setting the numerator to zero regardless of what the denominator is.

If I now pick the traditional linear penalty, g(l,t) = c(t)*l, the left side becomes

  Cov(l,g) = Cov(l,c(t)*l) = c(t)*Cov(l,l) = c(t)*Var(l).

Setting this equal to Cov(l,f) gives

  c(t) = Cov(l,f)/Var(l).

I should check that this c actually zeroes the growth and is not just an artefact of pushing symbols around, so I go back to the four programs. There Var(l) = sum Phi*(l-mu)^2 = [6.25 + 2.25 + 0.25 + 12.25]/4 = 21/4 = 5.25, so c = 3.25/5.25 = 0.6190. The penalised fitnesses are f_p = f - c*l = (1 - 1.238, 2 - 1.857, 2 - 3.095, 5 - 4.952) = (-0.238, 0.143, -1.095, 0.048). Two things jump out. First, recomputing the proportionate-selection mean with these f_p values gives E[mu(t+1)] = 4.5 to machine precision, so E[Delta mu] = 0: the coefficient does exactly what it was designed to do on this example. And Cov(l, c*l) comes out to 3.25, matching Cov(l,f) on the nose, which is the same statement seen from the other side. Second — and this is the part I would have missed without the numbers — some of the f_p are negative. Fitness-proportionate selection, which is how I derived the identity, needs non-negative weights; a penalised fitness that can go negative is not literally a proportionate-selection scheme any more. So the formula for c is exact as a statement about covariances and means, but the proportionate-selection story behind it is fragile the moment the penalty bites hard. I file that away as something the implementation has to respect.

The coefficient c = Cov(l,f)/Var(l) is the ordinary least-squares slope of fitness regressed on size: the per-node fitness advantage selection currently sees. Subtracting that slope times length removes the linear size-fitness coupling from the selection signal. It is dynamic because both the numerator and denominator drift over the run.

I check the generalisation, partly to make sure I have not just curve-fitted the one linear special case. With a power penalty

  g(l,t) = c(t)*l^k,

the no-growth condition Cov(l,g) = Cov(l,f) becomes c(t)*Cov(l,l^k) = Cov(l,f), so

  c(t) = Cov(l,f)/Cov(l,l^k),

and k = 1 collapses to Cov(l,f)/Var(l) as it must, since Cov(l,l) = Var(l). And if I centre the linear penalty,

  g(l,t) = c(t)*(l - mu(t)),

then Cov(l, mu(t)) = 0 because mu(t) is a constant across the population, so Cov(l,g) = c(t)*Var(l) again — the same c. The covariance condition, not the surface form of the penalty, is the real object; shifting the penalty by a population-wide constant cannot change a covariance.

Freezing size is only one target. If I want the expected mean size to follow a chosen trajectory gamma, I impose

  E[mu(t+1)] = gamma(t+1).

Since E[mu(t+1)] = mu(t) + E[Delta mu], the constraint is

  (Cov(l,f) - Cov(l,g))/(fbar - gbar) + mu(t) = gamma(t+1).

For g(l,t) = c(t)*l^k I have gbar = c(t)*E[l^k] and Cov(l,g) = c(t)*Cov(l,l^k). Writing delta = gamma(t+1) - mu(t) and clearing the denominator,

  Cov(l,f) - c(t)*Cov(l,l^k) = delta*(fbar - c(t)*E[l^k]),

which solves to

  c(t) = (Cov(l,f) - delta*fbar)/(Cov(l,l^k) - delta*E[l^k])
       = (Cov(l,f) - (gamma(t+1) - mu(t))*fbar)
         /(Cov(l,l^k) - (gamma(t+1) - mu(t))*E[l^k]).

For k = 1 this is

  c(t) = (Cov(l,f) - (gamma(t+1) - mu(t))*fbar)
         /(Var(l) - (gamma(t+1) - mu(t))*mu(t)).

Setting gamma(t+1) = mu(t) makes delta zero and recovers Cov(l,f)/Var(l), which is the consistency check I wanted: the target-tracking formula degenerates to the freeze formula when the target is "stay where you are." Using the initial mean as the setpoint, gamma(t+1) = mu(0), gives

  c(t) = (Cov(l,f) - (mu(0) - mu(t))*fbar)
         /(Cov(l,l^k) - (mu(0) - mu(t))*E[l^k]).

This matters because the zero-growth equation controls the mean only in expectation. In a finite population, sampling noise can let the realised mean drift even when E[Delta mu] = 0. Anchoring to mu(0) adds a restoring term: if the population has drifted above mu(0), then mu(0) - mu(t) is negative and the coefficient is pushed up to pull size back down, and vice versa.

I also keep the selection-scheme boundary honest. The exact covariance derivation used fitness-proportionate selection, where p(l,t) = Phi(l,t)*fbar(l,t)/fbar(t). Tournament selection does not give me that clean p(l,t), so c = Cov(l,f)/Var(l) is not an exact theorem for an arbitrary tournament. But I noticed above that proportionate selection itself breaks once the penalty drives f_p negative, so I was never going to ship literal proportionate selection anyway. What survives across schemes is the diagnostic content of the ratio: it is the linear size-fitness slope in the current population, and tournament selection, being rank-based, amplifies that same ordering. Cancelling the slope is therefore the natural practical coefficient even where it is not a theorem, and it carries over without my having to touch the operator.

Now I translate the math into the task implementation. The theory uses a maximised fitness and writes the penalty as f - c*l. The scaffold I have to fill gives me mean squared error, where lower is better. Under that convention, making large trees worse means selecting on

  penalized_error = MSE + c*l,

i.e. the penalty flips sign because the objective flipped direction. This matches the gplearn convention: `_Program.fitness()` returns `raw_fitness_ - c*len(program)*metric.sign`, and for MSE `metric.sign = -1`, so the penalised value is raw error plus c times length, and selection minimises it. The raw MSE stays the truth for elitism, stopping, and reporting; the penalty is only a parent-selection signal.

For the coefficient I reuse the gplearn automatic rule at the measurement step:

  auto_c = np.cov(length, fitness)[1, 0] / np.var(length).

Here I want to trace the numpy defaults rather than assume they match my derivation, because np.cov and np.var disagree on degrees of freedom. np.cov uses ddof = 1 (sample covariance, dividing by N-1); np.var uses ddof = 0 (population variance, dividing by N). On my four programs np.cov(l,f)[1,0] = 13/3 = 4.333 and np.var(l) = 5.25, so auto_c = 0.8254 — which is exactly 4/3 times the 0.6190 I computed with matched population statistics, i.e. a factor of N/(N-1) = 4/3. So the automatic rule is not the exact OLS slope for tiny populations; it is the slope inflated by N/(N-1). For the populations GP actually runs (100 to 10,000), that factor is 1.01 down to 1.0001 and is irrelevant, and the inflation only ever makes the pressure slightly stronger, never wrong in sign. I keep the gplearn rule as is rather than hand-tuning the ddof, since the destination is the established automatic coefficient and the discrepancy is negligible at scale — but it is worth knowing it is there.

The implementation also has to respect the negativity issue I found. Under the lower-better convention a negative c would mean rewarding length, which is the opposite of what I want, so I clamp c at zero from below. A single noisy generation can also throw up a huge slope, so I cap c with a mild ceiling to keep selection from suddenly ignoring fitness altogether. In the MLS-Bench edit surface the function receives raw lower-better fitnesses, so I compute the ratio from the current lengths and raw MSE vector, guard the degenerate zero-variance case (a population where every tree is the same size has no slope to cancel), and clamp to [0, 0.001]. The generation loop is then: keep the raw-MSE elite, build the penalised vector MSE + c*length, and use that vector only inside tournament selection while crossover, mutation, and reproduction fill the rest of the population.

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

The chain holds together, and the small example is what convinced me of each link rather than the symbols alone. I begin with bloat as a selection-caused mean-size drift, and watched it produce +1.3 nodes in one generation on four programs. The size evolution equation says mean size changes only through length-class selection probabilities, and the brute-force sum reproduced that. Fitness-proportionate selection turns the change into Price's covariance formula, and 3.25/2.5 = 1.3 matched both ways, confirming the cross-term really vanishes. A general size penalty changes the numerator from Cov(l,f) to Cov(l,f) - Cov(l,g), so no growth requires Cov(l,g) = Cov(l,f); the linear choice g = c*l forces c = Cov(l,f)/Var(l) = 0.6190 on the example, and that exact c drove E[Delta mu] to machine zero. The power and target-tracking formulas follow from the same one-unknown equation and degenerate correctly. The implementation uses the same covariance-over-variance rule — accepting the small N/(N-1) inflation from numpy's mixed ddof and clamping c to [0, 0.001] to stay sane under the lower-is-better MSE convention and the function surface I was given.
