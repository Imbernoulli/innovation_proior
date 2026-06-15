Let me start from what actually goes wrong when I run a tree-based GP on a symbolic-regression problem. I have a population of expression trees, and each generation I have to pick parents to breed. The default move is tournament selection: grab a handful of individuals at random, keep the one with the lowest error, repeat. And "error" here means the aggregate — I take each individual's vector of per-case errors `e(i) = |y − ŷ(i, x)|`, one entry for each of my `N` training points, and I crush it down to a single scalar like the mean absolute error `f(i) = (1/N) Σ_t e_t(i)`, and I select on that scalar. The thing nagging at me is that crushing step. I'm evolving programs, and the whole premise of GP is that a full solution gets assembled out of partial solutions — subprograms that each get some region of the problem right, recombined by crossover until one tree gets all of it right. But the instant I average `e(i)` into `f(i)`, I've thrown away *which* regions a given individual is good at. Two individuals can have the identical mean error while having completely different profiles: one is mediocre everywhere, the other is *the best in the whole population* on a cluster of hard cases and bad on the rest. For the purpose of harvesting a building block, the second one is gold and the first is nothing — and tournament can't tell them apart. Worse, it actively discards the specialist: its few bad cases inflate its average, so against generalists with a lower mean it almost never wins a tournament. The information I most need to guide the search — the shape of `e(i)`, not its average — is exactly the information I'm deleting before I select.

So the real question is: what if I *don't* aggregate? What if selection looks at the error vector directly? Let me think about what I actually want pressure on. I want to reward an individual for being uniquely good on *some* part of the problem — and especially on the parts that are *hard*, the cases that few others in the population get right, because those are precisely where I'm missing a building block. A generalist that's decently good everywhere is fine, but it's the specialist on the hard region that carries the piece I can't yet make. Aggregate selection gives every case equal, undifferentiated weight and rewards the average; I want something that can say "you, you're the best the population has to offer on this nasty case, you breed."

There's prior art that smells right and prior art that doesn't quite get there. Implicit fitness sharing, from McKay and from Krawiec & Nawrocki, has the right instinct: reward solving cases that few others solve. For the set `T_i` of cases `i` solves and `n(t)` the count of individuals solving case `t`, it scores `f_IFS(i) = Σ_{t∈T_i} 1/n(t)` — a case solved by only a handful is worth a lot, a case everyone solves is worth almost nothing. The non-binary version generalizes to graded per-case fitness, `f_NBIFS(i) = Σ_t f(i,t)/Σ_{i'} f(i',t)`. Historically-assessed hardness (Klein & Spector) does a similar thing — scale each case's error by the population's success rate on it. Co-solvability (Krawiec & Lichocki) goes further and rewards solving *pairs* of cases together, `f_CS(i) = Σ_{t_j,t_k} 1/n(t_j,t_k)`. All three are reaching for "reward the rare and hard." But stare at what they do at the end: they all *sum*. IFS sums the population-weighted per-case rewards into one number and then selects on that number. So I've weighted the cases more cleverly, but I've still collapsed the profile into a scalar before selecting — I've just chosen a fancier scalar. I can't reward an individual for a unique *combination* of cases this way, because the combination is gone the moment I add the terms up. Co-solvability keeps pairs, which is one step better, but only pairs, it has no clean continuous-error version, and it costs `O(PN²)` which is brutal at thousands of cases. The aggregation is the disease, and these are all treating symptoms while keeping the disease.

What would it even mean to select on the *combination* without aggregating? Here's the picture that gets me unstuck. Don't score-then-pick. Instead, *filter*. Treat each case as a gate. Start with the whole population as my candidate pool. Look at one case, and throw out everyone who isn't the best on that case — keep only the elite on that single case. Then look at another case, and among the survivors throw out everyone who isn't the best on *that* case. Keep going. Each gate is the simplest possible test: "are you the best left, on this case?" No averaging anywhere — just a sequence of elitism filters. If I run enough gates the pool shrinks to one individual, and that's my parent. And notice what kind of individual survives a long chain of these: someone elite on case A *and* elite-among-the-A-elite on case B *and* … — an individual good on a specific *conjunction* of cases. That's the combination I couldn't get from any sum.

But the *order* of the gates matters enormously. If I always apply the cases in the same fixed order, the first case becomes a permanent dictator — only ever individuals elite on case 1 can be selected, and case 1's elite then competes on case 2, and so on; I'd be selecting lexicographically by a fixed priority, which is just a weird deterministic ranking and would crush diversity instantly. The fix is to *shuffle the cases freshly for every single parent-selection event*. Now the first gate — the one with the most filtering power, because it acts on the whole population — is a different case each time. Over many selection events, every case gets its turn at being first, so selective pressure is spread across all cases rather than monopolized by one. And here's the part I like: a case's filtering strength is proportional to its *difficulty*. An easy case that almost everyone ties on as "best" barely shrinks the pool; a hard case where only one or two individuals reach the population-best error slices the pool down to those one or two. So whenever a hard case lands early in the shuffle, it hands selection straight to the specialist on that hard case. Difficult cases automatically exert more pressure, and they exert it toward exactly the specialists I want to propagate. This is lexicase selection — "global pool" (start from the whole population), "uniform random sequence" (reshuffle cases every event), "elitist" (keep only the best on each case) — and it's dead simple to implement: choose a case, filter to the elites on it, repeat until one remains or cases run out (then pick a random survivor).

And there's a diversity bonus that falls out of the mechanism for free. Because each parent is selected through a different random conjunction of cases, individuals are pressured to be good on *unique* subsets of cases. Two parents selected in the same generation likely came down different filter paths, so they're behaviorally different. The population doesn't collapse onto one family the way it does under repeated aggregate-best tournament selection — the pressure is constantly being redistributed to whoever is elite on the currently-first hard case, which spreads the population across the behavior space rather than concentrating it. That matches the original motivation: lexicase was built for *modal* problems, where the right program must do qualitatively different things on different input regions, and for *uncompromising* problems, where a solution has to be as good as possible on *every* case and no good-here-for-bad-there trade is acceptable. On those, "best on this case, no compromise" is precisely the right gate, and aggregation is precisely wrong. On discrete program-synthesis and Boolean problems this beats tournament, IFS, and co-solvability.

So I'm sold on the structure. Let me drop it onto my actual problem — continuous symbolic regression on noisy real data — and watch it work. Pool is the whole population. Shuffle the cases. First case: find the population-best error on it, keep only the individuals whose error *equals* that best… and right there I hit the wall. My errors are continuous, real-valued, and noisy. What's the chance that two *distinct* expression trees produce the *exact same* floating-point error on a given case? Essentially zero — it only happens if the two trees are literally the same model (or algebraically reduce to it). So "best error on this case" is achieved by exactly *one* individual, almost always. The very first gate slams the pool down from `P` to 1. The parent is decided by a *single* case. Every selection event uses one case and one case only. I've thrown away the entire multi-case filtering chain — the conjunction-of-cases mechanism that was the whole point — because the elitism filter is too sharp for a continuous space. And single-case selection is not just weak, it's actively *worse* than tournament on regression: I'm picking parents off one noisy case at a time, ignoring all the rest. The exact-equality pass condition, which was perfect for discrete pass/fail problems, is fatal here. Wall.

Let me diagnose precisely what broke so I fix the right thing. The gate asks "is your error on case `t` *equal* to the pool-best error `e*_t`?" In a discrete space, many individuals share the best error, so the gate keeps a meaningful subset and the chain continues. In a continuous space, "equal" is measure-zero, so the gate keeps a singleton and the chain dies on case one. The mechanism is sound; the *pass condition* is too stringent. I don't want "exactly best." I want "near-best" — close enough to the best on this case that I'm not going to disqualify a perfectly good near-elite individual just because it lost by `10⁻⁷`. So relax the gate: an individual passes case `t` if its error is *within some tolerance `ε`* of the best error on the case, rather than exactly equal to it. Now a near-elite individual survives a case instead of being culled, so a single case no longer empties the pool to one, multiple cases get to participate in the filtering again, and I recover lexicase's conjunction-of-cases behavior in continuous space. The pass condition becomes `e_t(i) ≤ e*_t + ε` (or a multiplicative version, `e_t(i) < e*_t(1+ε)`).

Now, what is `ε`? My first instinct is the dumb thing: make it a user-set constant. Either an absolute error band — `i` passes if `e_t(i) < ε_y`, meaning `ŷ` is within `±ε_y` of the true `y_t` — or a band around the elite, `e_t(i) ≤ e*_t(1+ε_e)`. Let me try this in my head and find where it hurts. The absolute band `ε_y` has an ugly failure: if no individual in the population is anywhere near the true value on a hard case, *every* individual fails the case, the gate keeps nobody (or keeps everybody, depending on how I define empty), and the case provides zero selection pressure — it can't discriminate when the whole population is far from the truth, which is exactly when I'd want the case to be picking out whoever is *least* far. The elite-relative band `ε_e` avoids that — at least the elite always passes its own band, so the pool never empties. But both share two deeper problems. First, the right value is *problem-dependent*: a band that gives sensible filtering on one dataset gives almost-no-filtering or almost-all-filtering on another, depending on the error scale and the population's spread. Second — and this is the one that really bothers me — a *fixed* `ε` is blind to how the population is doing. Early in a run the errors are huge and spread wide; late in a run, as the population converges, the errors are small and tight. A constant band that's reasonable early will, late in the run, be so wide relative to the now-tiny spread that *every* surviving individual passes every case — the gates stop filtering, selection goes random, and I've lost all pressure right when I should be fine-tuning. I'd have to hand-tune `ε` per problem *and* it would still be wrong at one end of the run or the other. I don't want a knob I have to babysit. I want `ε` to *set itself* from the data.

So `ε` should adapt to the population's actual performance on each case. The natural idea: make `ε` track the *spread* of the errors on the case. On a case where the population's errors are tightly clustered, a small `ε` is enough to separate near-elites from the rest; on a case where they're widely spread (a hard, contentious case), a larger `ε` admits the contenders. As the population improves on a case and the errors compress, the spread shrinks, so `ε` shrinks with it — the gate automatically gets more selective as the case gets "solved," which is exactly the self-scheduling behavior the constant `ε` lacked. So `ε` for case `t` should be some measure of dispersion of the per-case error vector `e_t = (e_t(1), …, e_t(P))` across the population.

Which dispersion measure? The textbook one is the standard deviation `σ(e_t)`. Let me sanity-check it against what a GP population actually looks like, and it falls apart immediately. A GP population at any moment is full of *garbage* individuals — freshly mutated junk, expressions that blow up or divide by near-zero and produce enormous errors. Those are extreme outliers in `e_t`. And `σ` is dominated by outliers: a handful of individuals with error `10⁶` while everyone real sits near `0.3` will send `σ` through the roof. So a standard-deviation `ε` would be set by the junk, not by the meaningful spread among the competitive individuals — it'd be so large that essentially everyone passes every case, and I'm back to no filtering. I need a dispersion measure that ignores the tails and reports the *typical* spread among the bulk. That's the median absolute deviation: `MAD(e_t) = median_j(|e_t(j) − median_k(e_t(k))|)`. The inner median is the typical error; the outer median is the typical distance from it. A minority of arbitrarily huge outliers can move the median by at most one rank step — MAD has a 50% breakdown point, it simply doesn't see the junk. So MAD reports the spread that the real contenders live in, which is exactly the scale at which "near-elite" should be judged. Set `ε_t = λ(e_t) := MAD(e_t)`. It's parameter-free — no knob to tune — it auto-scales to each case's difficulty, and it shrinks as the population converges. That's the `ε` I want, and the reason it's MAD and not `σ` is the outlier-heavy nature of GP populations specifically.

Let me write the relaxed gate with this `ε`, and I need to be precise about the reference set — over which individuals do I compute the best error and the MAD? The simplest answer to reach for first: the whole population, once per generation. At the start of a generation I have the whole population `P`, so for each case `t` I can form the population error column `e_t = (e_t(1), ..., e_t(P))`. The best error for the case is `e*_t = min_{j in P} e_t(j)`, and the automatic tolerance is `ε_t = MAD(e_t) = median_j(|e_t(j) - median_k(e_t(k))|)`. Then the pass condition for individual `i` on case `t` is `e_t(i) ≤ e*_t + ε_t`. The appeal is that both quantities are fixed properties of the population on that case, computed once, cheap. Let me write that down and then poke at whether the population is the right reference set.

There is a small trap if I freeze `e*_t` over the whole population. Deep in a selection event the population-best individual for case `t` may already have been removed by earlier cases, and then it is possible that no current candidate lies inside the population band. I must not let that empty the pool. The clean way to view that frozen version is as preprocessing: convert each continuous error into a binary pass/fail value for each case, where pass means `e_t(i) ≤ e*_t + ε_t`; then run ordinary lexicase over those binary values. If at least one current candidate passes the case, keep the passers. If none passes, then every current candidate has the same binary value on that case, so the case cannot discriminate among them and the pool stays as it is. Now the pool is always nonempty, the threshold is still defined by the full population, and the procedure is exactly a continuous-error-to-discrete-pass table followed by lexicase filtering. (If instead I anchor `e*_t` to the current pool, this trap never arises — the pool-best always clears its own band — which is one more reason to prefer the pool-relative form below.)

Now I can see the family around this choice, and they are different algorithms, not interchangeable word choices. The version I just wrote — both `e*_t` and `ε_t` computed once per generation over `P`, frozen into a population-level pass table — is the *static* one. But staring at it, I am bothered that by the third or fourth gate my pool is a handful of strong individuals while I am still measuring "best" against the original thousand, most already filtered out and irrelevant to this event; the bar is coarser than it could be. So sharpen it: keep `ε_t = MAD({e_t(j): j in P})` fixed over the full population (it is a stable estimate of the case's difficulty), but recompute `e*_t(S) = min_{j in S} e_t(j)` over the current pool `S` at each gate. Now the band is `pool-best + MAD`, so at least one current candidate always passes (the pool-best clears its own band, so the empty-pool worry disappears) and the bar rises toward the survivors who are actually still in contention — this behaves much more like true lexicase. That is *semi-dynamic*. Push once more: recompute *both* `e*_t(S)` and `ε_t(S) = MAD({e_t(j): j in S})` over the current pool. As the pool homogenizes — the survivors all near each other — the pool-MAD collapses, so `ε` shrinks further and the later gates keep filtering hard instead of waving everyone through. That is *dynamic*, fully relative to who is left. Of the three, anchoring to the live pool is what keeps the gates discriminating all the way down the chain; static is the cheap population-level approximation. The clean landing point for continuous regression is the pool-relative form — recompute the pool elite and the pool MAD at each gate — because it is the one that keeps filtering correctly as the pool shrinks, and the per-case error matrix I already have makes it natural to slice the current candidate set directly. Static stays as the originally-proposed parameter-free version when I want the cheapest preprocessing.

This also clarifies the multi-objective interpretation. If every training case is treated as a minimization objective, exact lexicase only selects individuals on the Pareto front defined by the cases. Once I allow a tolerance, I should not claim exact Pareto optimality anymore. The selected individual can be away from the exact front, but only inside the `ε` band induced by the case thresholds: it is `ε`-Pareto-optimal, meaning it lies within the allowed per-case slack of the Pareto-optimal boundary rather than necessarily being one of the exact nondominated points. That is the right weakening for noisy regression: I am not pretending a tiny continuous-error difference is meaningful, but I am still excluding individuals that are worse by more than the robust population scale.

The cost is still in the same worst-case class as lexicase. The pool-relative form I am landing on recomputes a minimum and a MAD over the current pool at each gate, but that is linear in the current pool size per case; selecting one parent may inspect every remaining candidate on every case in the worst case, so one event is `O(PN)` and filling a population of `P` parents is `O(P²N)` — the same worst case as plain lexicase, with the MAD adding no asymptotic term. (The static version is even cheaper to prepare — one scan of the `P × N` matrix, one minimum and one MAD per case, an `O(PN)` pass table — but it does not change the asymptotic bound either.) The real hope is not a better big-O than lexicase; it is that the relaxed gates let more than one case participate on continuous errors without requiring a hand-tuned tolerance, and in practice the pool winnows fast so the wall-clock time stays in the tournament-selection ballpark.

Now let me write it as the exposed GP functions, landing on the pool-relative form. Crossover and mutation stay as ordinary subtree operators. The scalar fitness stays for elitism and fallback tournament selection. The new information is the per-case absolute-error matrix prepared once per generation; the parent chooser slices that matrix to the current candidate set at each gate and recomputes the pool elite and the pool MAD there, so the band tracks who is actually still in contention.

```python
import numpy as np
import random


def fitness_function(tree, X, y):
    """MSE — lower is better. Used for elitism and as a selection fallback."""
    y_pred = safe_evaluate(tree, X)
    return float(np.mean((y - y_pred) ** 2))


def selection(population, fitnesses, n_select, _errors=None, _X=None, _y=None):
    """epsilon-lexicase parent selection (pool-relative form).

    The per-case error matrix has shape (P, N), with absolute error
    e_t(i) = |y_t - y_hat_t(i)|. At each gate, the pool elite e*_t and the
    tolerance epsilon_t = MAD(e_t) are recomputed over the *current* pool, so
    the band tracks the surviving candidates. Falls back to tournament if the
    per-case error matrix wasn't supplied.
    """
    selected = []
    pop_size = len(population)

    if _errors is None:                          # no behavioral matrix -> tournament
        for _ in range(n_select):
            candidates = random.sample(range(pop_size), min(7, pop_size))
            best = min(candidates, key=lambda i: fitnesses[i])
            selected.append(population[best].copy())
        return selected

    n_cases = _errors.shape[1]
    for _ in range(n_select):
        candidates = list(range(pop_size))       # global pool: whole population
        cases = list(range(n_cases))
        random.shuffle(cases)                    # uniform random sequence of cases

        for case in cases:
            if len(candidates) <= 1:             # pool reduced to one -> done
                break
            case_errors = _errors[candidates, case]
            min_err = float(np.min(case_errors))                 # pool elite e*_t
            mad = float(np.median(np.abs(                        # epsilon_t = MAD(e_t)
                case_errors - float(np.median(case_errors)))))   #   over current pool
            # relaxed elitist gate: keep individuals within epsilon of the best
            candidates = [c for c, e in zip(candidates, case_errors)
                          if e <= min_err + mad]

        winner = random.choice(candidates)       # tie -> random survivor
        selected.append(population[winner].copy())

    return selected


def crossover(parent1, parent2, n_features, max_depth=17):
    """Standard subtree crossover (unchanged from the base harness)."""
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
    if offspring.depth() > max_depth:
        return parent1.copy()
    return offspring


def mutation(parent, n_features, max_depth=17):
    """Standard subtree mutation (unchanged from the base harness)."""
    offspring = parent.copy()
    if offspring.size() <= 1:
        return generate_tree('grow', 3, n_features)
    mut_point = random.randint(1, offspring.size() - 1)
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
    """One generation with elitism + epsilon-lexicase selection."""
    new_population = []

    elite_idx = int(np.argmin(fitnesses))        # elitism: never lose the best
    new_population.append(population[elite_idx].copy())

    errors = np.empty((len(population), X_train.shape[0]))
    for i, tree in enumerate(population):
        y_pred = safe_evaluate(tree, X_train)
        errors[i] = np.abs(y_train - y_pred)     # per-case absolute error matrix

    while len(new_population) < pop_size:
        r = random.random()
        if r < crossover_rate:
            parents = selection(population, fitnesses, 2, _errors=errors)
            child = crossover(parents[0], parents[1], n_features, max_depth)
        elif r < crossover_rate + mutation_rate:
            parents = selection(population, fitnesses, 1, _errors=errors)
            child = mutation(parents[0], n_features, max_depth)
        else:
            child = selection(population, fitnesses, 1, _errors=errors)[0]
        new_population.append(child)

    return new_population[:pop_size]
```

Let me trace the causal chain back to make sure every piece earned its place. I started stuck because aggregate selection crushes each individual's per-case error vector into one scalar, which deletes the information about *which* regions it is good at and discards specialists that may carry the building blocks I need. The weighting methods still aggregate, so they cannot preserve a unique combination of cases. Lexicase gets around that by filtering, not summing: shuffle the cases for each parent, gate the population one case at a time, and keep the individuals that are elite on the current case. Exact elitism is too sharp for noisy continuous regression, so I relax the gate to `e_t(i) ≤ e*_t + ε_t`. A fixed `ε` is a brittle new knob, so I set `ε_t` from the robust population spread on that case, `MAD(e_t) = median_j(|e_t(j) - median_k(e_t(k))|)`, which ignores the extreme-error junk that would make a standard deviation threshold useless. Computing both `e*_t` and `ε_t` over the population gives a static pass table; semi-dynamic moves only `e*` to the current pool, and dynamic moves both `e*` and `ε` to the current pool, so the band keeps discriminating as the pool shrinks. I land on the pool-relative form — recompute the pool elite and the pool MAD at each gate — which never empties the pool, selects within an `ε`-Pareto band rather than at exact Pareto points, and keeps lexicase's `O(P²N)` worst-case generation cost while restoring multi-case filtering to continuous errors; the static population-referenced pass table stays as the cheaper originally-proposed parameter-free version.
