Let me start from what actually hurts when I try to tune a model. The objective `f(x)` — validation loss as a function of the hyperparameters `x` — is a black box: the only way to read it is to train a model and measure it, which is slow, and what I get back is noisy, `y(x) = f(x) + noise`, because the training itself is stochastic. So I am optimizing an expensive, noisy, derivative-free function, and I have a fixed budget of evaluations. That much is the standard story. What I keep running into, though, is that the methods I have each win on one axis and quietly lose on another, and the spaces I care about most are the ones where the losing happens. I want a *default* — something I can point at any HPO or NAS problem without thinking — and a default has to be good at all of: returning a decent answer early (anytime), getting the best possible answer with a big budget (final), using parallel machines, not falling apart as the dimension grows, and handling messy spaces with integer, ordinal, and categorical dimensions. No single thing I have does all of that robustly. So let me lay out the tools honestly, find exactly where each one breaks, and see if the breakages point at each other in a way I can exploit.

The cheapest idea is random search: sample configurations uniformly, evaluate each at full budget, keep the best. It is shockingly hard to beat when only a couple of hyperparameters matter, it parallelizes trivially, and it never gets stuck. But it is memoryless — every sample is drawn independently of everything I have already learned — so on a problem where the good region is inferable from past evaluations, it throws the entire budget at blind exploration. And it pays full price for every configuration, including the obviously hopeless ones. Two distinct kinds of waste: it doesn't learn, and it doesn't triage.

The triage problem has a clean answer: multi-fidelity. For most learning problems a *cheap approximation* of the objective exists — train for fewer epochs, on less data, for fewer MCMC steps, fewer RL trials — and these cheap scores are noisy but correlated with the real one. So I can rule out hopeless configurations at a fraction of the cost. Successive Halving makes this concrete: sample `N` configurations, evaluate all at the lowest budget, keep the top `1/eta`, multiply the budget by `eta`, repeat to the top. The survivors get exponentially more resource; the expensive full evaluation is paid for only a handful. The trouble is the input `N`. For a fixed total budget `B`, do I want many configurations each run cheaply (large `N`, tiny resource each), or few configurations each run thoroughly (small `N`)? That is the `n`-versus-`B/n` dilemma, and the right answer depends on something I do not know in advance: how well the cheap fidelity predicts the expensive one. If the correlation is strong, go wide and cheap; if it is weak, going wide and cheap will *discard at a low budget exactly the configuration that would have won at the high budget*. SH forces me to guess.

Hyperband refuses to guess and hedges instead. It runs SH at several starting budgets — `s = s_max` down to `s = 0`, where `s_max = floor(log_eta R)`, total per-bracket budget `B = (s_max+1) R`, and bracket `s` starts `n = ceil((B/R) eta^s/(s+1))` configurations at resource `r = R eta^{-s}`, halving inward `n_i = floor(n eta^{-i})`, `r_i = r eta^i`, keeping the top `floor(n_i/eta)` each rung. The most aggressive bracket throws a flood of configurations at a tiny budget; the least aggressive bracket `s=0` is just random search at full budget. By spanning the brackets it covers the whole `n`-versus-`B/n` spectrum, which buys it a guarantee — at most about `s_max+1` times slower than random search — and genuinely strong anytime behavior, because the cheap brackets surface a reasonable configuration fast. But now stare at where the configurations come from. Every single one is sampled *uniformly at random*. Hyperband is brilliant at *allocating* budget across fidelities and at *killing* weak configurations early, but it never uses the outcome of one evaluation to decide where to look next. Its quality is capped by the quality of random sampling in that space. Give it a long run and any method that learns from history walks past it.

That is the gap BOHB closes — keep Hyperband's bracket schedule, but replace the random sampling with a model. Specifically a Tree-Parzen Estimator: once there are enough observations at a budget, fit a density `l(x)` over the well-performing configurations and `g(x)` over the poorly-performing ones, and sample new points to maximize the ratio `l(x)/g(x)`. This grafts Bayesian optimization's strong final performance onto Hyperband's anytime skeleton, and it is the best off-the-shelf multi-fidelity optimizer I have. So why am I not done? Because of *where* it is weakest, and it is weakest exactly where I most need a default to be strong. The model component is BO, and BO does not love high dimensions, does not natively handle discrete and categorical dimensions, and needs roughly `d+1` observations before its model says anything useful — so in a high-dimensional space it spends a long opening phase behaving like the random search it was supposed to improve on. Its model-fitting overhead grows with the number of observations rather than staying flat. On the tabular NAS benchmarks — which are nothing but high-dimensional discrete HPO — and on high-dimensional discrete spaces generally, BOHB's edge over plain Hyperband shrinks or vanishes. The thing that was supposed to be my default is observed to flatten out precisely on the hard, discrete, high-dimensional problems.

So the diagnosis sharpens. Hyperband's failure is *random sampling never learns*. BOHB tried to fix that by bolting a *model* onto Hyperband, and the model is what fails in high-dimensional discrete spaces. The natural question is whether there is a *model-free* way to learn from history — something that biases the search toward good regions without ever fitting a probabilistic model, so that it inherits none of BO's dimensionality and discreteness pain and none of its growing overhead.

Differential Evolution is sitting right there, and it is exactly that kind of thing. It keeps a population of `N` vectors and improves them generation by generation with three operators. Mutation, the rand/1 form: pick three distinct individuals and form a mutant `v = x_{r1} + F (x_{r2} - x_{r3})` with `F in (0,1]`. Crossover, binomial: build a trial by taking each coordinate from `v` with probability `p` and from the target `x` otherwise, forcing at least one coordinate from `v` (a random index `j_rand`) so the trial is never a pure copy of the target. Selection, for a minimizer: keep the trial if `f(u) <= f(x)`, else keep the target. The part people point to as the engine of DE is the difference vector: supposedly, when the population is spread out the perturbations are large and the search explores, and as the population converges they shrink and it exploits. I want to actually see that before I lean on it, because the entire appeal of DE for me is that this self-scaling replaces a learning rate I would otherwise have to tune. So let me sample the operator on two one-dimensional populations — a spread-out one `{0.1, 0.9, 0.5, 0.2, 0.8}` and a nearly-converged one `{0.50, 0.51, 0.49, 0.50, 0.52}` — and just measure the mean magnitude of `F*(x_{r2}-x_{r3})` with `F=0.5` over many random parent draws. The spread-out population gives a mean step of about `0.219`; the converged one gives about `0.007`. That is a factor of roughly 31 between them, with no parameter changed — the step scale really does track the population's own spread, purely from the geometry of the difference vector. So DE carries its own adaptive step size, no model, no learning-rate schedule, and crucially it only ever *compares* function values, so it is indifferent to whether a coordinate is continuous, integer, or categorical. While I am at it, let me confirm the other operator does what the description claims: with `p=0.5`, target all-zeros and mutant all-ones, over 100000 binomial-crossover draws the `j_rand` forcing produces a trial identical to the target zero times — so a trial is genuinely never a pure copy, recombination always injects at least one mutant coordinate. That last property — comparison-only fitness — is huge for my hard cases: if I keep the population continuous in the unit hypercube `[0,1]^D` and only decode to the original (possibly discrete) space at evaluation time, DE searches a smooth `[0,1]^D` while the world stays discrete, and I don't collapse population diversity the way a directly-discrete population would. DE is model-free, gradient-free, conceptually trivial, and its per-step cost is a constant — it never grows with the number of evaluations the way a model fit does. It is the anti-BOHB sampler.

But DE as I have stated it is *single-fidelity*: every individual is evaluated at full budget. So it has none of Hyperband's triage and none of its anytime behavior — it pays full price for everyone, including the early garbage. And classical DE has a second wrinkle I should remember: its update is *deferred*. The standard generation loop evolves every individual against the *frozen* population of the previous generation, and only swaps in the winners once the whole generation is done. So a good offspring discovered early in a generation cannot influence the rest of that generation; its benefit is delayed by up to a full pass through the population. On an expensive objective that delay is real wall-clock time wasted.

So I have two parts whose flaws look like mirror images. Hyperband has the multi-fidelity skeleton I love — the bracket schedule, the killing of weak configs at cheap budgets, the anytime guarantee — and its one flaw is that it samples randomly. DE is a model-free learner that fixes "samples randomly" without inheriting BO's high-dimensional, discrete-space, overhead-growing weaknesses, but it has no multi-fidelity machinery. The obvious move is to keep Hyperband's bracket arithmetic exactly as is and replace its `get_hyperparameter_configuration(n)` — the random sampler — with a DE search. Whether that actually composes is the real question, and I want to find out *how* it breaks before I commit to a fix, because the cleanest wiring I can think of has at least one place I am suspicious of.

The naive wiring is tempting. Hyperband at the start of a bracket needs `n` configurations at the lowest budget; instead of sampling them at random, I run a few generations of DE and hand over its population. Then SH evaluates them, keeps the top `1/eta`, and the next rung needs configurations at the higher budget — so I run DE there too. Let me actually execute this against a concrete schedule rather than wave at it. Take `b_min=1, b_max=27, eta=3`, so `s_max = floor(log_3 27) = 3` and there are 4 SH brackets. Tracing the spacing, the first bracket (the most aggressive) issues `n_configs = [27, 9, 3, 1]` at fidelities `[1, 3, 9, 27]`; the later brackets are `[9, 3, 1]` at `[3, 9, 27]`, `[6, 2]` at `[9, 27]`, and `[4]` at `[27]`. Now walk the first bracket. The lowest rung: I have no population yet, so DE has nothing to evolve from — I have to sample randomly anyway to seed it. Fine, the very first rung is random; that is unavoidable and matches Hyperband's opening. SH evaluates those 27, keeps the top `27/3 = 9`. Next rung at budget 3 with 9 configs, keep top 3; next rung at budget 9 with 3 configs, keep top 1. And there it is, written in the schedule: the top rung wants **1** configuration. If I promote that single survivor and call it the DE population at budget 27, I have a population of one, and DE rand/1 needs three distinct parents to even form a difference vector. There is no difference vector on a population of one. So the straightforward "DE per rung on the survivors" doesn't degrade gracefully — it has nothing to run at the top of the most aggressive bracket, and the schedule guarantees that top rung is size 1. The wall is real and it is in the arithmetic, not in my imagination.

Let me also notice a subtler problem before I patch the first one, because they are related. If each rung's DE population is *just the survivors promoted from below*, then the higher budget never does any independent searching — it only ever sees configurations that were good at the *lower* budget. That is fine when the fidelities are strongly correlated, but I already know from the benchmarks that on the hard problems they are *not*, and a configuration that looked great at few epochs can be mediocre at many. If the high budget can only ever evolve the low budget's winners, it has no way to recover from a misleading low fidelity. So the naive wiring is both mechanically broken (population of one) and strategically brittle (no independent high-budget search).

Both problems point the same direction: the higher budget needs *its own standing population* that persists, not a population freshly handed to it each rung and then thrown away. Let me make that the central design choice. Keep one DE population — call it a *subpopulation* — permanently attached to each budget level. The subpopulation for budget `b` lives across the whole run; it is the DE state for searching at fidelity `b`. How big should each be? I do not want to invent a new hyperparameter, and DE's population size is its single most finicky knob. But Hyperband already tells me, for each budget, the maximum number of configurations it ever allocates to that budget across all its brackets. Let me actually compute that from the same `b_min=1, b_max=27, eta=3` schedule: take the max over the four brackets of the count assigned to each fidelity. Budget 1 only ever appears in bracket 0 with 27 configs, so 27. Budget 3 appears as 9 (bracket 0) and 9 (bracket 1) — max 9. Budget 9 appears as 3, 3, 6 across brackets — max 6. Budget 27 appears as 1, 1, 2, 4 — max 4. So the subpopulation sizes come out `{1: 27, 3: 9, 9: 6, 27: 4}`: large at the cheap budget, shrinking toward the expensive one, and entirely determined by the Hyperband component. I never chose them, and notably the top budget's subpopulation is 4, not 1 — comfortably above the three parents rand/1 needs — so persisting a standing subpopulation per budget *also* defuses the population-of-one wall for the bulk of the run. This is the first real piece: a persistent DE subpopulation per fidelity, sized by Hyperband.

With persistent subpopulations, what does a run look like? The first full Hyperband sweep — the first `s_max + 1` SH brackets — has to act as initialization. The lowest rung of the first bracket gives me the first evaluated low-budget population; higher rungs in that opening sweep *promote* the best evaluated configurations from the immediately lower budget, where "promote" means evaluate the same vector at the higher budget rather than mutate it yet. By the time this opening sweep is over, every budget level has a real, evaluated subpopulation. From then on, *no more random sampling is needed*. Each later SH bracket reuses the subpopulations already sitting at each budget and advances them by DE evolution. So the random-sampling-never-learns flaw of Hyperband is gone after the opening: the search at every budget is now a learning DE search, not fresh random draws.

That fixes "where do the populations come from" but I still have the two problems from the naive wiring to resolve, and they were both really about *how information should move between budgets*. Let me think hard about what the right cross-budget coupling is, because this is the crux.

I do not want the budgets to be independent — that would just be a bunch of separate single-fidelity DE runs, throwing away the whole point of multi-fidelity, which is that cheap evaluations should *guide* expensive ones. But I also do not want the rigid "the high budget's population *is* the low budget's survivors," because that re-imports SH's brittleness: it can't recover from an uncorrelated low fidelity, and it dies when the survivor set is too small for DE. I want something in between — the low budget should *inform* the high budget's search without *dictating* its population.

DE's mutation gives me exactly the lever. The difference vector `F(x_{r2} - x_{r3})` is added to a base individual `x_{r1}` — the mutant inherits its *location* from `x_{r1}` and its *search direction and scale* from where `r2` and `r3` are. So if I choose *where the mutation parents come from*, I choose what region of the space the new candidates are pulled toward. In vanilla DE the three parents come from the same population that is being evolved. What if, when I evolve the *higher* budget's subpopulation, I draw the mutation parents not from the higher budget's own population but from the *good region of the lower budget*? Concretely: take the top `1/eta` of the lower budget's subpopulation — the configurations that performed well *at the cheaper fidelity* — and call them the *parent pool*. When I evolve an individual at the higher budget, the target `x_i` is still a member of the higher budget's own subpopulation (so the new candidate stays anchored in the budget I am actually searching), but the mutation parents `r1, r2, r3` are sampled from this parent pool. The mutant is then `x_{r1} + F(x_{r2} - x_{r3})` with all three drawn from the lower budget's good region, and the binomial crossover mixes it with the higher-budget target.

Let me check this does what I want. The parent pool is "a good-performing *region* with respect to the lower budget," not a list of configurations to re-evaluate verbatim. Information flows up: the higher budget's new candidates are biased toward where the cheaper search found promise, which is the whole multi-fidelity premise. But it flows up as a *bias on the mutation*, not as a *replacement of the population* — the target stays in the high budget, and the difference vector still self-scales to the spread of the good region (the same geometry I just measured the factor-of-31 effect from, now operating over the cheap budget's winners). So I get cross-fidelity guidance without forcing the high budget to literally adopt the low budget's points.

Now the recovery-from-uncorrelated-fidelity worry. What stops a misleading low budget from poisoning the high budget's search? It has to be the selection step, so let me trace one step to be sure selection actually does the guarding rather than just assert it. Suppose the parent pool points me at a region that is excellent at budget 3 but poor at budget 27, and the resulting trial, evaluated at budget 27, scores `0.8`, while the high-budget target it is competing against already sits at fitness `0.4` (better, since this is a minimizer). The selection rule is "adopt the trial iff `trial_score <= target_fitness`." Here `0.8 <= 0.4` is false, so the trial is rejected and the target is retained unchanged. The only cost was the single wasted evaluation at budget 27; the high-budget subpopulation did not move toward the misleading region at all. Run the opposite case — the transfer was actually good and the trial scores `0.3` — then `0.3 <= 0.4` is true and the trial is adopted, which is exactly what I want when the low budget told the truth. So the low budget can *suggest* but never *impose*: a bad suggestion fails the inequality and evaporates, a good one passes. And there is a second guard built into the bracket structure: in every SH bracket the *first* rung is the lowest budget of that bracket, where there is no lower budget to draw a parent pool from. There I just run *vanilla* DE — parents from the budget's own subpopulation. So the lowest budget of every bracket always keeps searching on its own steam, independent of any cross-fidelity transfer. The high budgets get guided when the guidance is good and ignore it (via selection) when it is not, while the low budgets never stop making independent progress. That is the robustness I wanted, and it comes from making selection the arbiter rather than promotion.

There is still the population-of-one corner — let me make sure the parent-pool design handles the cases the per-budget subpopulation sizing did not already cover. The subpopulation sizes I computed (`27, 9, 6, 4`) are all at least 3, so a budget's *own* population never starves. But the parent pool is the top `1/eta` of the *lower* budget, and that fraction can be small: at the most expensive rung the lower budget contributes only `n_configs/eta` winners, and the schedule has rungs where that count is 1 or 2. The top `1/eta` of a one- or two-survivor set is below the three parents rand/1 needs. So the wall has moved but not vanished — it now lives in the parent pool, not the subpopulation. I need two more sources of geometry from somewhere, but I do not want to spend extra function evaluations to manufacture them, and I do not want to fabricate random points that ignore everything learned. The fix: keep a *global population pool* that is just the union of *all* the subpopulations across all budgets. When a parent pool is short of the minimum, use this global pool to create the extra mutation-pool members needed before the final rand/1 parent sampling. Is that legitimate? These extra vectors only enter mutation as *geometry* — endpoints and bases for difference vectors — and their *fitness values are never consulted* in mutation. So it does not matter that individuals from different budgets have fitness on different scales (a score at few epochs is not comparable to a score at many) — I am using them purely as *points in `[0,1]^D`* to define a search direction, not as ranked candidates. The global pool just supplies geometry. No extra evaluations are spent, and the missing geometry comes from configurations the search has actually visited rather than from thin air. The population-of-one wall is gone, and it is gone in a way that reuses information instead of wasting budget.

Two more details I want to get right because they affect convergence speed. First, which individual is the *target* of each evolution step? I sweep the subpopulation with a rolling pointer: each successive DE step advances the pointer by one to pick the next target, wrapping around at the end. So across a generation every member of the subpopulation gets a turn as the target, exactly as in classical DE, but driven by an explicit pointer because the bracket schedule, not a tidy nested loop, decides when each step happens. Second — and this is where I might be able to beat classical DE outright — the *update timing*. Classical DE is deferred: it evolves the whole generation against the frozen previous population and only swaps in winners afterward, so a good offspring found early cannot help the rest of the generation. Let me make the update *immediate*: the moment a trial wins its selection, it goes straight into the subpopulation, so the very next evolution step can already use it as a parent or target. On an expensive objective this means improvements propagate within the same generation instead of waiting a full pass. It also has a structural payoff I will want later: because each evolution step reads and writes a single shared copy of the subpopulations and is agnostic to which bracket it came from, the brackets can be run *asynchronously* — a free worker can start the next SH bracket against the current state of the subpopulations without waiting for a previous bracket's rung to finish, which is how this turns into a clean parallel method. Immediate update buys both faster sequential convergence and natural parallelism.

Let me also pin down why the constants are what they are, since I am inheriting some and choosing others. `eta = 3` comes straight from Hyperband: it controls both the fraction promoted (top `1/eta`) and the budget multiplier, and 3 is the standard near-optimal setting for the geometric `n`-versus-`B/n` tradeoff — aggressive enough to triage hard, gentle enough not to discard too recklessly. The DE knobs: the scaling factor `F = 0.5`, squarely in the robust middle of `(0,1]` — and the step-size measurement I did earlier was at exactly this `F`, where the adaptive range from converged to spread was already a factor of ~31, so I am not relying on an extreme setting to get useful step variation; and the crossover rate `p = 0.5`, a balanced split so each trial mixes roughly half mutant and half target coordinates, which keeps recombination meaningful without either copying the target or jumping wholesale to an untested mutant. These are the standard DE defaults and I take them as given rather than tuning per problem, because the entire point is a *default*. And the population sizes I have already removed as a hyperparameter — they are set by the Hyperband component, which is one of the nicest consequences of the whole design.

Now let me make sure I can write the full procedure end to end without a gap. The mathematical bracket spacing I inherit is fixed: `s_max = floor(log_eta(b_max/b_min))`; for bracket `s = s_max - (iteration mod (s_max + 1))`, start `N = ceil(((s_max + 1)/(s + 1)) * eta^s)` configurations at `b_0 = b_max * eta^{-s}`, then rung `i` uses `N_i = floor(N * eta^{-i})` configurations at `b_i = b_0 * eta^i`. Those counts and fidelities are what define the ladder — the `[27, 9, 3, 1]` over `[1, 3, 9, 27]` I traced above is exactly this formula at `iteration = 0`. I initialize `s_max + 1` DE subpopulations, one per budget level, each sized by the maximum count this ladder assigns that budget (the `{1:27, 3:9, 9:6, 27:4}` I computed), each filled with random `[0,1]^D` vectors and unevaluated. Then run the SH brackets in Hyperband order. At rung `i`, budget `b_i`, do `N_i` acquisition steps. For each step: pick the target by the rolling pointer over that budget's subpopulation; during the opening sweep, while the first `s_max + 1` brackets are seeding all fidelities, any rung above the lowest just *promotes* — take the next-best not-yet-promoted configuration from the lower budget and evaluate it here. After that opening sweep, and at the lowest rung even during the opening, generate a candidate by DE evolution — if `i == 0` use vanilla mutation (parents from this budget's own subpopulation), else use the altered mutation (parents from the parent pool = top configurations of the lower budget, topped up through the global pool if fewer than the minimum), then binomial crossover with the target, then a boundary repair back into `[0,1]^D`. Evaluate the candidate at `b_i`, run DE selection against the target (adopt iff the candidate's score is at least as good), and immediately update the subpopulation and the running incumbent. Once the opening sweep is over, the promotion branch never fires again — every later candidate is born from DE evolution. Every branch has a reason traceable to a specific failure above: promotion only to bootstrap, vanilla DE at the lowest rung to keep independent progress, altered DE above it to flow information up, selection-at-budget to guard against bad transfer (the `0.8 <= 0.4` rejection I traced), global pool to never starve the mutation, immediate update for speed and parallelism.

The code I ship has two pieces: a DE engine with the three operators and a `[0,1]`-hypercube encoding, and a driver that owns the per-fidelity subpopulations, the bracket schedule, promotion during the opening sweep, and the immediate selection update. For the driver I use the integer scheduler that fixes the same fidelity suffixes and population sizes the running engine will actually use.

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
    """Model-free differential-evolution engine. The population lives in the unit
    hypercube [0,1]^D; configurations are decoded to the real space only at eval time."""

    def __init__(self, space, pop_size, F=0.5, p=0.5, rng=None):
        self.space, self.pop_size = space, pop_size
        self.F, self.p, self.rng = F, p, rng
        self.min_parents = 3                              # rand/1 needs 3 distinct parents
        self.population = np.array([space.encode(space.sample_uniform(rng))
                                    for _ in range(pop_size)])
        self.fitness = np.full(pop_size, np.inf)          # np.inf = not yet evaluated (minimizer)
        self.ptr = 0                                       # rolling pointer over the subpopulation

    def next_target(self):
        """Rolling pointer: each call returns the next individual to serve as the target."""
        idx = self.ptr
        self.ptr = (self.ptr + 1) % self.pop_size
        return idx

    def sample_population(self, alt_pop=None, target=None):
        """Sample mutation parents, excluding the target if it is present."""
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
        """rand/1: v = r1 + F*(r2 - r3)."""
        return r1 + self.F * (r2 - r3)

    def mutation(self, current=None, alt_pop=None):
        r1, r2, r3 = self.sample_population(alt_pop=alt_pop, target=current)
        return self.mutation_rand1(r1, r2, r3)

    def crossover_bin(self, target, mutant):
        """Binomial crossover; j_rand forces at least one coordinate to come from the mutant."""
        mask = self.rng.rand(self.space.dim) < self.p
        if not mask.any():
            mask[self.rng.randint(self.space.dim)] = True
        return np.where(mask, mutant, target)

    def boundary_check(self, vec):
        """Repair out-of-range coordinates back into [0,1] (resample violators)."""
        bad = (vec < 0) | (vec > 1)
        vec = vec.copy()
        vec[bad] = self.rng.rand(int(bad.sum()))
        return vec

    def init_mutant_population(self, pop_size, population, target):
        """Generate extra mutation-pool members from a global population."""
        return np.array([
            self.mutation(current=target, alt_pop=population)
            for _ in range(pop_size)
        ])

    def evolve(self, target_idx, mutation_pop):
        """One DE step: mutate from mutation_pop, cross with the target, repair."""
        target = self.population[target_idx]
        mutant = self.mutation(current=target, alt_pop=mutation_pop)
        trial = self.crossover_bin(target, mutant)
        return self.boundary_check(trial)

    def select(self, target_idx, trial_vec, trial_score):
        """Immediate selection: adopt the trial iff it is no worse at this fidelity.
        '<=' (not '<') keeps the search exploring plateaus."""
        if trial_score <= self.fitness[target_idx]:
            self.population[target_idx] = trial_vec
            self.fitness[target_idx] = trial_score


class DEHB:
    """Differential-Evolution Hyperband. One persistent DE subpopulation per fidelity,
    sized by the Hyperband component; random sampling happens only in the first
    (initialization) iteration, DE evolution everywhere after, with information flowing
    from lower to higher fidelities through a modified (parent-pool) mutation."""

    def __init__(self, space, b_min, b_max, eta=3, F=0.5, p=0.5, seed=42):
        self.space, self.b_min, self.b_max, self.eta = space, b_min, b_max, eta
        self.rng = np.random.RandomState(seed)
        self.s_max = int(np.floor(np.log(b_max / b_min) / np.log(eta)))
        self.max_SH_iter = self.s_max + 1

        # One DE subpopulation per fidelity; pop size = max #configs HB ever puts on that fidelity.
        self.max_pop = self._get_pop_sizes()
        self.fidelities = sorted(self.max_pop)
        self.de = {f: DE(space, self.max_pop[f], F, p, self.rng) for f in self.fidelities}

        self.iteration = 0                  # which SH bracket the scheduler is issuing
        self.promotion = {f: [] for f in self.fidelities}   # queued promotions during init
        self.inc_score, self.inc_config = np.inf, None      # anytime incumbent

    def _get_pop_sizes(self):
        max_pop = {}
        for it in range(self.max_SH_iter):
            n_configs, fids = sh_spacing(self.b_min, self.b_max, self.eta, it)
            for n, f in zip(n_configs, fids):
                max_pop[f] = max(max_pop.get(f, 0), n)
        return max_pop

    def _concat_pops(self):
        """Global population pool: union of all subpopulations, used only to supply
        geometry when a lower-fidelity mutation pool is too small (fitness ignored)."""
        return np.array([v for f in self.fidelities for v in self.de[f].population])

    def _promotion_candidate(self, lower_fidelity, high_fidelity, n_configs):
        """Queue lower-fidelity winners for initialization-time promotion."""
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
                queue = [lo.population[order[0]]]          # duplicate fallback, matching the edge case
            self.promotion[high_fidelity] = queue
        return self.promotion[high_fidelity].pop(0)

    def _mutation_pool(self, lower_fidelity, fidelity, n_configs, target):
        """Top lower-fidelity region, enlarged through the global pool if rand/1 would starve."""
        de_lo = self.de[lower_fidelity]
        order = np.argsort(de_lo.fitness)[:n_configs]
        pool = de_lo.population[order]
        if len(pool) < self.de[fidelity].min_parents:
            # The +1 leaves room for AsyncDE-style target removal before sampling parents.
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
        """Propose jobs for the current SH bracket, keeping the vector needed by tell()."""
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
        """Report a result: immediate DE selection at this fidelity + incumbent update."""
        fidelity, target_idx, config_vec = job["fidelity"], job["target_idx"], job["config_vec"]
        self.de[fidelity].select(target_idx, config_vec, score)
        if score < self.inc_score:                         # anytime best-so-far (minimizer)
            self.inc_score, self.inc_config = score, config_vec
```

Let me trace the causal chain to be sure it holds together. I needed a single default HPO method strong on anytime, final, parallel, high-dimensional, and discrete problems at once. Random search never learns and never triages. Successive Halving triages via cheap fidelities but forces me to guess the `n`-versus-`B/n` tradeoff and can drop a config that would win at full budget. Hyperband hedges that tradeoff across brackets and gets anytime performance and a near-random-search guarantee, but every configuration it tries is sampled at random — it never learns. BOHB fixed the learning by grafting a Bayesian model onto Hyperband, but the model is exactly what fails in high-dimensional and discrete spaces, needs `d+1` points to start, and has growing overhead. Differential Evolution is a model-free learner whose self-scaling difference vector adapts to the population's spread — I measured that adaptive range directly at roughly a factor of 31 between a converged and a spread-out population, with no parameter changed — only ever compares function values (so it eats discrete and high-dimensional spaces when kept in `[0,1]^D`), and has constant per-step cost; but it is single-fidelity and its deferred update is slow. Wiring DE naively into each Hyperband rung broke twice, and I confirmed the first break in the schedule arithmetic itself: rand/1 starves on the population-of-one at the top of a bracket (the `[27, 9, 3, 1]` ladder ends in a literal 1), and a high budget that only evolves the low budget's survivors can't recover when fidelities are uncorrelated. The resolution was a persistent DE subpopulation per fidelity, sized by Hyperband (so no population-size hyperparameter — I computed the sizes as `{1:27, 3:9, 9:6, 27:4}`, all safely above the three parents rand/1 needs); an opening Hyperband sweep that promotes lower-budget winners just long enough to seed every budget, after which random sampling never recurs; a modified mutation that draws its parents from a parent pool — the good region of the *lower* fidelity — so information flows upward as a search bias rather than a forced population swap; selection *at the higher budget* as the guard that lets bad cross-fidelity transfers fail harmlessly (I traced a `0.8 <= 0.4` rejection that kept the misleading region from being adopted) while the lowest rung's vanilla DE keeps each budget improving independently; a global population pool to create extra mutation-pool geometry whenever a parent pool is too small, spending no extra evaluations; and an immediate update so winners propagate within a generation and the brackets parallelize. `eta = 3`, `F = 0.5`, `p = 0.5` are the standard robust defaults, and the population sizes fall out of the Hyperband component. The method is conceptually simple, model-free, cheap, and built from two well-understood parts whose flaws cancel: Hyperband supplies the multi-fidelity skeleton DE lacked, and DE supplies the model-free learning Hyperband lacked.
