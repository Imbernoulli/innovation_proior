The GA numbers landed almost exactly on the three-way bet I made, and the pattern is the whole argument
for what comes next. Rastrigin: 8.99 at 30D (mean 8.05) and 115 at 100D (mean 114) — a decisive win over
DE's 172 and 926. Put a number on "decisive": at 30D that is DE's mean 256.8 down to 8.05, a factor of
32; at 100D 925.8 down to 113.8, a factor of 8.1. The per-coordinate polynomial mutation did what I
predicted: on the separable egg-carton, ratcheting one axis at a time beats DE's coupled `CR=0.9` bundle
by an order of magnitude, and the residual fell from about `9.3` per coordinate (three pits out) to about
`1.1` (each coordinate essentially one pit from the origin) at 100D. But the two regressions I feared are
right there, and the more interesting is Rosenbrock, where the mean and the good seed disagree and the
disagreement is the lesson. GA Rosenbrock is 87 at seed 42, mean 138 — worse than DE's *seed-42* 32.2 by
a factor of 2.7, though far better than DE's *mean* of 48302. Those are two different facts, not a
conflict: DE's mutation, when its `rand` base happened to seed the valley, tracked the ribbon and reached
32; when it scattered it never entered the valley and the mean blew up. The GA never scatters like that —
it is consistent around 100 — but it is consistently *stuck*, crawling the diagonal valley with axis-
aligned SBX and polynomial moves never aligned to it. So the honest reading: DE has the right mutation
and the wrong base; the GA has a safe base and the wrong mutation. Ackley says the same from the smooth
side: GA 1.05 (mean 1.12) is more consistent than DE's noisy 10.0, but DE's best seed reached 0.488 — the
difference vector nearly solved the broad basin on the run where it entered it, whereas the GA's axis-
aligned operators leave a full unit of error on every seed. And the convergence columns confirm nobody
finished: the GA is still improving at generation ~490–500 on the 30D problems and at 1000 on 100D — it
has not converged, it has run out of budget, just as DE was still creeping at its wall.

So I have a clean, contradictory lesson, and I can state it as who owns what. DE's self-scaling
anisotropic difference vector is the right *mutation* — it owns Rosenbrock and Ackley the moment it
enters the basin — but its single global `F`/`CR` is catastrophic on Rastrigin and its fixed `rand` base
is slow and scatter-prone. The GA's per-coordinate locality owns Rastrigin but throws away the anisotropy
that owns the valley. No single fixed setting of either is right on all four landscapes: Rosenbrock wants
the coupled, high-`CR`, anisotropic move; Rastrigin wants the low-`CR`, one-axis move that the GA's
`indpb=1/n` delivered by accident. I have now hand-tuned the same underlying algorithm twice and each
time gotten a method right on some columns and wrong on others. That is the thing to kill: the parameters
must learn themselves from what is working on *this* problem, right now. The signal is sitting in the
selection step — every generation some trials beat their parents and some do not, and the winners'
`(F_i, CR_i)` are evidence of what worked on this landscape this generation. So I should sample `F` and
`CR` per individual from distributions centered on recent winners and slide those centers toward the
latest winners. That is what would let *one* algorithm drive `CR` toward 0 on Rastrigin and high on
Rosenbrock without me choosing.

Work through the pieces the way a careful adaptive DE assembles them, because each is a decision I would
otherwise get wrong. First, the mutation strategy. Plain `DE/rand/1` picks a random base — fine but slow,
the very slowness the convergence columns exposed. The greedy fix `current-to-best/1`,
`v = x_i + F·(x_best − x_i) + F·(x_r1 − x_r2)`, pulls each individual partway toward the single best plus
a random difference; fast on a unimodal bowl, but on a multimodal function *everyone* is dragged toward
the one incumbent best and the whole population funnels into its basin — premature convergence, the
Rastrigin death. The fix is to pull toward a *random one of the top few*: `x_pbest` drawn from the top
`N·p`, `v_i = x_i + F_i·(x_pbest − x_i) + F_i·(x_r1 − x_r2)`. Now `p` is a greediness dial. In the loop
`p_i` is drawn per individual from `[p_min, p_max]` with `p_min = 2/N_init = 0.01` and `p_max = 0.2`; the
pbest pool is the top `round(p_i·N)`, so at `N=200` it ranges from the top 2 individuals to the top 40. A
small-`p` individual chases essentially the incumbent best two; a large-`p` individual chases anywhere in
the top forty. Randomizing `p_i` spreads the population's attraction across several good basins at once —
the diversity Rastrigin demands, bought without giving up the guides that speed Rosenbrock and Ackley.
This `current-to-pbest/1` is strictly more flexible than current-to-best/1: setting `p` to its smallest
value recovers current-to-best, so I lose nothing and gain a diversity dial.

Second, the difference `x_r1 − x_r2`. If both come from the current population, then as the population
converges this difference shrinks toward zero along with everything else, and I lose diversification
exactly when I might be stuck in a Rastrigin pit. The parents that just *lost* selection encode where the
search recently was and chose to leave; keep them in an external archive `A` and draw `x_r2` from
`P ∪ A`. Now the difference can reach back to a recently-abandoned region, adding diversity without
enlarging the live population (which would cost evaluations the budget cannot spare). I cap the archive at
the current population size and, on overflow, drop random members — a reservoir of recent history, not a
museum that biases the search. So `r1` from `P` only, `r2` from `P ∪ A`, distinct from each other and
from `i`.

Third, what do I sample `F` and `CR` from, and how do I summarize the winners? `CR` is a probability that
should settle near a stable good value, so Normal is natural: `CR_i = N(M_CR[r], 0.1)`, clamped to
`[0,1]`. A tight Normal keeps `CR` near its learned center. `F` is different: it controls mutation
magnitude, and the failure mode I fear most — premature convergence — is precisely `F` collapsing to
small values too early and the search going quiet. So `F` wants a distribution that keeps *proposing*
large values even when the center drifts down: a heavy tail. Cauchy: `F_i = M_F[r] + 0.1·tan(π(rand −
0.5))`, same center as a Normal but fat tails. Quantify "far": how often does each sampler propose an `F`
at least `0.3` above its center? For the Cauchy that needs the standard variate to exceed `3`, and
`P(C > 3) = 0.5 − arctan(3)/π = 0.102`, about one draw in ten; for the Normal it needs `Z > 3`,
`P(Z > 3) = 0.00135`, one draw in 740. So the Cauchy throws a large, population-reviving `F` roughly 75
times more often — the concrete meaning of "never let `F` go quiet," and why `F` gets the fat tail and
`CR` does not. The truncation rules respect what each parameter is: `F > 1` truncate to 1 (huge `F` is
unstable), but `F ≤ 0` I *resample* rather than clamp, because a non-positive scaling inverts or kills the
mutation; clamping `CR` at both ends is fine because both endpoints are meaningful.

How do I summarize successful `F` into a new center? Arithmetic mean pulls toward the bulk of successes,
which on a converging population tend to be smallish, so the center drifts down, mutation weakens, and I
manufacture the exact premature convergence I am avoiding. I need a summary that *resists* that pull —
extra weight to the larger successful `F`. The Lehmer mean does this: `mean_L(S) = (Σ S_k²)/(Σ S_k)`,
larger elements pulling the ratio up, and that it is always at least the arithmetic mean is an identity,
`mean_L − mean_A = Var(S)/mean_A ≥ 0`, zero only when every success coincides. So Lehmer-averaging keeps
the center biased toward the larger successes and stops the magnitude dying. And not every success is
equally informative: weight each winner by its improvement `Δf_k = |f(x_k) − f(u_k)|`, normalize
`w_k = Δf_k/Σ Δf_l`, and use the weighted Lehmer mean for `F` and a weighted arithmetic mean for `CR`.
That asymmetry — Cauchy/Lehmer for `F`, Normal/arithmetic for `CR` — is "let `CR` settle, but never let
`F` go quiet."

Does this machinery actually deliver the landscape-dependent `CR` I have been promising? The mechanism is
the selection filter, and it runs itself. On separable Rastrigin, a high-`CR` trial overwrites most
coordinates at once, and such a bundle is almost always rejected on a survivor; the trials that *do*
improve are overwhelmingly the low-`CR` ones that touched a coordinate or two, so the winners fed into
the memory carry small `CR`, the weighted mean is small, and `M_CR` slides down. On Rosenbrock the
reverse: a trial that nudges two coordinates cannot descend a valley demanding coordinated moves, so the
improving trials are the high-`CR` coordinated ones and `M_CR` slides up. The same update rule pushes the
center to opposite ends purely because selection admits different winners. Translated back through the
donor count `1 + (D−1)·CR`: a Rastrigin-driven `M_CR ≈ 0.1` yields about `4` coordinates per trial at
30D — essentially the GA's one-axis regime — while a Rosenbrock-driven `M_CR ≈ 0.9` yields `≈ 27`, the
fully coupled valley move DE used. One algorithm occupies the exact per-coordinate regime each fixed
method needed, and gets there without me choosing, because the sampling width cooperates:
`CR ~ N(M_CR, 0.1)` keeps draws within about `±0.3` of the center, so a center near `0.1` samples a tight
cloud of small `CR` and a center near `0.9` a tight cloud of large `CR`, each self-consistent with the
winners that produced it.

Now the fragility a single-center (JADE-style) version carries: there is *one* pair `(M_F, M_CR)` steering
every individual. Selection is stochastic; some generation, by luck, the winning trials carry mediocre
`F`/`CR` for unrelated reasons, the update slides my one center toward them, and next generation the
*entire* population samples from the contaminated center. On a hard multimodal problem where success is
noisy and rare — Rastrigin — this is the normal weather, not a corner case. The cure is redundancy: keep
a *set* of `H` centers, each individual picks an index `r` uniformly and samples around that slot, and
each generation writes the winners' summary into *one* slot, cycling with a counter `k` round-robin. A
contaminated summary then lands in one of the `H` slots; next generation only about `1/H` of the
population draws from it while the other `H−1` slots keep things sane, and the bad slot is overwritten in
a few generations. Round-robin matters: overwriting every slot each generation would be a single effective
center with extra steps. Here `H = 6`, so at most one sixth of the population ever draws from a
freshly-contaminated slot. (Edge cases stay concrete: a generation with no winners updates nothing; the
weighted means use a `1e-30` denominator guard.)

That is a genuinely self-tuning DE — SHADE — but it has done nothing about the third knob, `N`, which is
still fixed, and `N` is the one whose pain the convergence columns screamed. The tension is sharp. Early I
want a *large* `N`: broad coverage of the box so I do not miss a basin, and a large, diverse pool of
successes per generation so the memory has good statistics (at `N=200` a generation can hand the memory
up to 200 winners, at `N=4` at most four, so the adaptation is only trustworthy while the population is
still large). Late, once the population has localized, a large `N` is pure waste: hundreds of evaluations
per generation for one tiny refinement step, when what I want is *many generations* of small precise
moves. With fixed `N` the budget buys `budget/N` generations, full stop. The fix: start large and *shrink
linearly*. Pinned at `N = 200` the budget buys `budget/200` generations; sliding `N` from `200` down to
`4` makes the mean population size about `(200+4)/2 = 102`, so the same budget buys roughly `budget/102`
generations — close to twice as many — and every extra generation falls in the cheap, late, small-`N`
regime, exactly the many-small-steps refinement the unconverged GA never reached. Shrinking manufactures
late generations out of budget a fixed large `N` would have burned on redundant exploration, and it
respects DE's self-scaling: a smaller converged population still has small differences, so refinement
stays fine-grained.

I want a *deterministic* monotone schedule — reactive resizing would replace one parameter with several
meta-parameters governing when and how much to resize, and I have spent two methods eliminating
hand-tuned parameters, not reintroducing a fistful. Linear is the simplest monotone decrease and needs
only the initial size:
`N_next = round(N_init + (N_min − N_init)·(gen+1)/n_generations)`, floored at `N_min`. `N_min = 4` is
pinned by the operator, not chosen — `current-to-pbest/1` needs `x_i`, a distinct `x_pbest`, `x_r1`, and
`x_r2`, four distinct individuals. When `N` must drop I delete the *worst* by fitness (elitist shrink
keeps the elites and accelerates late focusing) and trim the archive to the new size.

One adaptation to this task's budget is load-bearing and I will not gloss it. The canonical L-SHADE sets
`N_init = 18·D`, sized for the large evaluation budgets of competition benchmarks. At 100D that is `1800`
— nine times the harness's `pop_size`. But this harness runs a *fixed, small* `n_generations` with a
fixed `pop_size`, and with the matched total-evaluation budget here a starting population of 1800 would
consume the whole budget in a handful of generations, so the linear reduction never gets the runway to
reach the small-`N` refinement regime — it would starve Rastrigin-100D of exactly the late generations the
shrink exists to create. So I use `N_init = pop_size` and let the linear reduction to `N_min = 4` do its
work over the fixed generation count, keeping the budget identical to the other methods. This is the one
place the implementation deliberately departs from the textbook constant, and it is the right call for
*this* budget — but I flag it, because it is also the one place I am trading away early exploration, and
if any benchmark punishes that trade it will be the high-dimensional multimodal one.

Now the bet against the GA and DE numbers. The self-adapting `CR` should let one algorithm be right on all
four landscapes where neither fixed method was: driving `CR` low on Rastrigin (recovering the GA's
per-coordinate win) while keeping the anisotropic difference vector (recovering DE's valley and basin
power). Concretely: Ackley should drop from the GA's 1.05 and DE's 10.0 to near machine precision — the
broad smooth basin is where adaptive `F` plus difference vectors should fully solve it, and converge early
rather than burn the budget, so watch the convergence column fall well below 500 there. Rosenbrock should
beat both the GA's 87/138 and recover toward DE's anisotropy, into the low tens or better. Rastrigin-30d
should be competitive with the GA's 8. The one place I am genuinely uncertain is Rastrigin-100d: the
deliberate `N_init=pop_size` choice trades exploration for refinement, and Rastrigin at 100D is the
hardest multimodal case, so it may land *near* the GA's 114 rather than crushing it — and if the trade is
bad it could even come in above. So the column I will read first, and worry about, is Rastrigin-100d.
