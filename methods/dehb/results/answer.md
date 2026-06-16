# DEHB, distilled

DEHB (Differential Evolution Hyperband) is a general-purpose, multi-fidelity hyperparameter
optimizer. It keeps Hyperband's bracket schedule for allocating budget across fidelities, but
replaces Hyperband's *random* configuration sampling with a *Differential Evolution* search.
The result is model-free (no Bayesian surrogate), so it scales to high-dimensional and discrete
search spaces where model-based methods struggle, has constant per-evaluation overhead, is
conceptually simple, and parallelizes naturally.

## Problem it solves

Black-box, expensive, noisy hyperparameter optimization where a single method must be a robust
default: strong anytime *and* final performance, effective parallel use, scalability with
dimensionality, and flexibility on mixed continuous/integer/ordinal/categorical spaces —
including high-dimensional discrete spaces (e.g. tabular NAS) where Bayesian-optimization-based
multi-fidelity methods lose their edge.

## Key idea

- **Hyperband skeleton, DE sampler.** Hyperband fixes the multi-fidelity budget schedule (the
  `n`-versus-`B/n` tradeoff hedged across Successive-Halving brackets, discard factor `eta`).
  Instead of sampling each configuration uniformly at random, DEHB evolves it with DE — a
  model-free learner whose difference-vector mutation self-scales to the population's spread and
  only ever compares function values, so it is indifferent to dimensionality and to discrete
  dimensions.
- **One persistent subpopulation per fidelity.** Each budget level owns a DE population that
  lives across the whole run. Its size is set by the Hyperband component (the max number of
  configurations HB ever allocates to that budget), which *removes DE's population-size
  hyperparameter*.
- **Random sampling happens only at the start.** The first full Hyperband sweep of
  `s_max + 1` SH brackets seeds the subpopulations: the very first low-budget population begins
  from random vectors, and higher rungs in the opening sweep *promote* (re-evaluate) the best
  lower-budget configurations. After that opening sweep, no random sampling occurs — every
  budget runs DE evolution.
- **Modified mutation = information flow across fidelities.** When evolving a higher budget's
  subpopulation, the DE *target* stays in that budget but the mutation *parents* are drawn from
  a **parent pool**: the top configurations of the *lower* budget (its good region). Information
  flows upward as a bias on the mutation, not as a forced population swap. The lowest rung of
  each bracket has no lower budget, so it runs *vanilla* DE (parents from its own
  subpopulation), keeping each budget's search independently alive.
- **Selection at the higher budget is the guard.** A trial is adopted only if it scores at
  least as well *at its own fidelity*. So if a low fidelity is misleading (uncorrelated with the
  full budget), the transferred trial simply fails selection and the target is retained — no
  damage, only one spent evaluation.
- **Global population pool.** When a parent pool has fewer than the minimum parents (e.g. a
  single survivor at the top budget), the union of all subpopulations supplies extra mutation
  geometry before the final rand/1 parent sampling. Fitness is irrelevant here because these
  vectors only define directions in `[0,1]^D`, and no extra function evaluations are spent.
- **Immediate update.** A winning trial enters the subpopulation at once (vs. classical DE's
  deferred update), so improvements propagate within a generation and brackets can run
  asynchronously in parallel.

## Final algorithm

```
s_max = floor(log_eta(b_max / b_min))
initialize (s_max + 1) DE subpopulations, one per fidelity, sized by the HB component, random in [0,1]^D
bracket_counter = 0
while not terminated:
    for iteration in {0, ..., s_max}:
        s = s_max - (iteration mod (s_max + 1))
        N = ceil(((s_max + 1) / (s + 1)) * eta^s)
        b_0 = b_max * eta^{-s}
        N_i = floor(N * eta^{-i}), b_i = b_0 * eta^i for i in {0, ..., s}
        n_configs, budgets = SH_bracket_under_HB(b_min, b_max, eta, iteration)
        for i in {0, ..., s}:                                  # rungs, increasing budget
            for j in {1, ..., n_configs[i]}:
                target = rolling pointer over subpopulation DE[budgets[i]]
                mutation_type = "vanilla" if i == 0 else "altered"
                if bracket_counter == 0 and i > 0:
                    config = j-th best config from DE[budgets[i-1]]   # promotion (init only)
                else:
                    config = DE_trial_generation(target, mutation_type)  # mutate + crossover + repair
                result = evaluate(config, budgets[i])
                DE selection: adopt config over target at budgets[i] iff result <= target's score
                update incumbent
    bracket_counter += 1
return incumbent
```

where `DE_trial_generation` builds a mutant `v = r1 + F*(r2 - r3)` (rand/1) with parents from
the budget's own subpopulation (`vanilla`) or from the lower budget's parent pool (`altered`);
if that pool is too small, extra mutation-pool members are generated from the global pool before
the final parent sampling. The mutant is binomial-crossed with the target and repaired back into
`[0,1]^D`.

## Defaults and why

`eta = 3`, `F = 0.5`, `p = 0.5` (crossover rate); subpopulation sizes are *not* user-set.

- `eta = 3` is Hyperband's standard near-optimal setting for the geometric `n`-versus-`B/n`
  tradeoff — aggressive triage without discarding too recklessly.
- `F = 0.5` is the robust mid-range DE scaling factor in `(0,1]`: difference-vector steps
  neither stall nor overshoot.
- `p = 0.5` balances mutant/target recombination so each trial is a meaningful mix.
- Population sizes come from the Hyperband component (large at cheap budgets, small at expensive
  ones), removing DE's most finicky hyperparameter.

## Mixed data types

Keep every subpopulation continuous in the unit hypercube `[0,1]^D` and decode to the original
space only at evaluation time: integer/float `X^i in [a_i, b_i]` as `a_i + (b_i - a_i) * u`
(integers rounded), ordinal/categorical with `n` choices by splitting `[0,1]` into `n` equal
bins. A directly-discrete population would collapse diversity (many duplicates, degenerate
difference vectors); keeping it continuous preserves DE's search power on discrete spaces.

## Working code

Faithful to the canonical `automl/DEHB` structure: a DE engine (rand/1 mutation, binomial
crossover, boundary repair, immediate selection) plus a DEHB driver owning one subpopulation per
fidelity, the Successive-Halving spacing, promotion-during-init, the parent-pool / global-pool
modified mutation, and the anytime incumbent. The code mirrors the integer `_get_next_iteration`
schedule used by that driver; the mathematical Hyperband formula is the one stated above.

```python
import numpy as np


def sh_spacing(b_min, b_max, eta, iteration):
    """Integer bracket scheduler used by the implementation."""
    max_SH_iter = int(np.floor(np.log(b_max / b_min) / np.log(eta))) + 1
    all_fidelities = b_max * np.power(
        eta, -np.linspace(start=max_SH_iter - 1, stop=0, num=max_SH_iter)
    )
    s = max_SH_iter - 1 - (iteration % max_SH_iter)
    fidelities = all_fidelities[-(s + 1):]
    n0 = int(np.floor(max_SH_iter / (s + 1)) * eta ** s)
    n_configs = [max(int(n0 * eta ** (-i)), 1) for i in range(s + 1)]
    return n_configs, fidelities


class DE:
    """Model-free DE engine; population lives in the unit hypercube [0,1]^D."""

    def __init__(self, space, pop_size, F=0.5, p=0.5, rng=None):
        self.space, self.pop_size = space, pop_size
        self.F, self.p, self.rng = F, p, rng
        self.min_parents = 3                                   # rand/1 needs 3 distinct parents
        self.population = np.array([space.encode(space.sample_uniform(rng))
                                    for _ in range(pop_size)])
        self.fitness = np.full(pop_size, np.inf)               # np.inf = unevaluated (minimizer)
        self.ptr = 0

    def next_target(self):
        idx = self.ptr
        self.ptr = (self.ptr + 1) % self.pop_size              # rolling pointer over the subpopulation
        return idx

    def sample_population(self, alt_pop=None, target=None):
        population = self.population if alt_pop is None else np.asarray(alt_pop)
        if target is not None and len(population) > 1:
            for i, row in enumerate(population):
                if np.all(row == target):
                    population = np.concatenate([population[:i], population[i + 1:]])
                    break
        if len(population) < self.min_parents:
            filler = self.rng.rand(self.min_parents - len(population), self.space.dim)
            population = np.concatenate([population, filler])
        idx = self.rng.choice(np.arange(len(population)), self.min_parents, replace=False)
        return population[idx]

    def mutation_rand1(self, r1, r2, r3):
        return r1 + self.F * (r2 - r3)                         # v = r1 + F*(r2 - r3)

    def mutation(self, current=None, alt_pop=None):
        r1, r2, r3 = self.sample_population(alt_pop=alt_pop, target=current)
        return self.mutation_rand1(r1, r2, r3)

    def crossover_bin(self, target, mutant):
        mask = self.rng.rand(self.space.dim) < self.p
        if not mask.any():
            mask[self.rng.randint(self.space.dim)] = True      # j_rand: >=1 coord from the mutant
        return np.where(mask, mutant, target)

    def boundary_check(self, vec):
        bad = (vec < 0) | (vec > 1)
        vec = vec.copy()
        vec[bad] = self.rng.rand(int(bad.sum()))               # resample out-of-range coords
        return vec

    def init_mutant_population(self, pop_size, population, target):
        return np.array([
            self.mutation(current=target, alt_pop=population)
            for _ in range(pop_size)
        ])

    def evolve(self, target_idx, mutation_pop):
        target = self.population[target_idx]
        mutant = self.mutation(current=target, alt_pop=mutation_pop)
        trial = self.crossover_bin(target, mutant)
        return self.boundary_check(trial)

    def select(self, target_idx, trial_vec, trial_score):
        if trial_score <= self.fitness[target_idx]:            # '<=' keeps exploring plateaus
            self.population[target_idx] = trial_vec            # immediate update
            self.fitness[target_idx] = trial_score


class DEHB:
    """Differential-Evolution Hyperband: HB skeleton with a DE sampler, one persistent
    subpopulation per fidelity, parent-pool information flow across fidelities."""

    def __init__(self, space, b_min, b_max, eta=3, F=0.5, p=0.5, seed=42):
        self.space, self.b_min, self.b_max, self.eta = space, b_min, b_max, eta
        self.rng = np.random.RandomState(seed)
        self.s_max = int(np.floor(np.log(b_max / b_min) / np.log(eta)))
        self.max_SH_iter = self.s_max + 1

        self.max_pop = self._get_pop_sizes()
        self.fidelities = sorted(self.max_pop)
        self.de = {f: DE(space, self.max_pop[f], F, p, self.rng) for f in self.fidelities}

        self.iteration = 0
        self.promotion = {f: [] for f in self.fidelities}      # queued promotions (init iteration)
        self.inc_score, self.inc_config = np.inf, None         # anytime incumbent

    def _get_pop_sizes(self):
        max_pop = {}
        for it in range(self.max_SH_iter):
            n_configs, fids = sh_spacing(self.b_min, self.b_max, self.eta, it)
            for n, f in zip(n_configs, fids):
                max_pop[f] = max(max_pop.get(f, 0), n)
        return max_pop

    def _concat_pops(self):
        """Global pool: union of all subpopulations; fitness is ignored here."""
        return np.array([v for f in self.fidelities for v in self.de[f].population])

    def _promotion_candidate(self, lower_fidelity, high_fidelity, n_configs):
        if not self.promotion[high_fidelity]:
            lo = self.de[lower_fidelity]
            evaluated = np.where(lo.fitness != np.inf)[0]
            order = evaluated[np.argsort(lo.fitness[evaluated])]
            queue = []
            for idx in order:
                candidate = lo.population[idx]
                already_high = np.any(np.all(candidate == self.de[high_fidelity].population, axis=1))
                if not already_high:
                    queue.append(candidate)
                if len(queue) == n_configs:
                    break
            if not queue and len(order):
                queue = [lo.population[order[0]]]
            self.promotion[high_fidelity] = queue
        return self.promotion[high_fidelity].pop(0)

    def _mutation_pool(self, lower_fidelity, fidelity, n_configs, target):
        de_lo = self.de[lower_fidelity]
        pool = de_lo.population[np.argsort(de_lo.fitness)[:n_configs]]
        if len(pool) < self.de[fidelity].min_parents:
            filler = self.de[fidelity].min_parents - len(pool) + 1
            extra = self.de[fidelity].init_mutant_population(
                pop_size=filler,
                population=self._concat_pops(),
                target=target,
            )
            pool = np.concatenate([pool, extra])
        return pool

    def _acquire_config(self, fids, n_configs, rung):
        fidelity = fids[rung]
        de = self.de[fidelity]
        target_idx = de.next_target()
        target = de.population[target_idx]
        lower_fidelity = fids[max(rung - 1, 0)]
        num_configs = n_configs[rung]

        if self.iteration < self.max_SH_iter and rung > 0:
            config_vec = self._promotion_candidate(lower_fidelity, fidelity, num_configs)
        else:
            mutation_pop = self._mutation_pool(lower_fidelity, fidelity, num_configs, target)
            config_vec = de.evolve(target_idx, mutation_pop)
        return config_vec, target_idx, fidelity

    def suggest(self):
        """Yield job records for each step of the current SH bracket."""
        n_configs, fids = sh_spacing(self.b_min, self.b_max, self.eta, self.iteration)
        for i, fidelity in enumerate(fids):
            for _ in range(n_configs[i]):
                config_vec, target_idx, fidelity = self._acquire_config(fids, n_configs, i)
                yield {
                    "config": self.space.decode(config_vec),
                    "config_vec": config_vec,
                    "fidelity": fidelity,
                    "target_idx": target_idx,
                }
        self.iteration += 1

    def tell(self, job, score):
        fidelity, target_idx, config_vec = job["fidelity"], job["target_idx"], job["config_vec"]
        self.de[fidelity].select(target_idx, config_vec, score)
        if score < self.inc_score:                             # minimizer; flip sign for maximization
            self.inc_score, self.inc_config = score, config_vec
```
