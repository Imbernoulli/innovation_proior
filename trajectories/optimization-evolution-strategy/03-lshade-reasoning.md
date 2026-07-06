The GA numbers landed almost exactly on the three-way bet I made, and the pattern is the whole argument for
this rung. Rastrigin: 8.99 at 30D (mean 8.05) and 115 at 100D (mean 114) — a decisive win over DE's 172 and
926. Put a number on "decisive": at 30D that is DE's mean 256.8 down to 8.05, a factor of 32; at 100D it is
925.8 down to 113.8, a factor of 8.1. The per-coordinate polynomial mutation did what I predicted: on the
separable egg-carton, ratcheting one axis at a time beats DE's coupled `CR=0.9` bundle by an order of
magnitude, and in per-coordinate terms the residual fell from about `9.3` per coordinate (three pits out)
to about `1.1` per coordinate (each coordinate essentially one pit from the origin) at 100D. But the two
regressions I feared are right there too, and the more interesting one is Rosenbrock, because the mean and
the good seed disagree and the disagreement is the lesson. GA Rosenbrock is 87 at seed 42, mean 138 — worse
than DE's *seed-42* 32.2 by a factor of 2.7, even though it is far better than DE's *mean* of 48302. Those
two comparisons are not in conflict; they are two different facts. DE's mutation, when its `rand` base
happened to seed the valley (seed 42), tracked the curved ribbon and reached 32; when it scattered (the
other seeds) it never entered the valley and the mean blew up. The GA never scatters like that — it is
consistent around 100 — but it is consistently *stuck*, crawling the diagonal valley with axis-aligned SBX
and polynomial moves that are never aligned to it, exactly as I called. So the honest Rosenbrock reading is:
DE has the right mutation and the wrong base; the GA has a safe base and the wrong mutation. And Ackley says
the same thing from the smooth side: GA 1.05 (mean 1.12) is more consistent than DE's noisy 10.0 mean, but
DE's best seed reached 0.488 — the difference vector nearly solved the broad basin on the run where it
entered it, whereas the GA's axis-aligned operators leave a full unit of error on the table on every seed.
And the convergence columns confirm nobody has actually finished: the GA is still improving at generation
~490–500 on the 30D problems and at 1000 on 100D — it has not converged, it has run out of budget, just as
DE was still creeping at its wall.

So I now have a clean, contradictory lesson from two rungs, and I can state it as a table of who owns what.
DE's self-scaling anisotropic difference vector is the right *mutation* — it owns Rosenbrock and Ackley the
moment it enters the basin — but its single global `F`/`CR` is catastrophic on Rastrigin and its fixed
`rand` base is slow and scatter-prone. The GA's per-coordinate locality owns Rastrigin but throws away the
anisotropy that owns the valley. No single fixed setting of either method is right on all four landscapes:
Rosenbrock wants the coupled, high-`CR`, anisotropic move; Rastrigin wants the low-`CR`, one-axis move that
the GA's `indpb=1/n` delivered by accident. The three knobs `F`, `CR`, `N` are coupled and all
problem-dependent, and I have now hand-tuned the same underlying algorithm twice and gotten a method that is
right on some columns and wrong on others each time. That is the thing to kill: the parameters must learn
themselves from what is working on *this* problem, right now.

So I go back to DE's clever core and ask how to make it adapt. The mutation `v = x_r1 + F·(x_r2 − x_r3)`
never had to invent a step size — the difference vector carries the population's own scatter, big early and
small late, and anisotropic in the valley. That is why DE owned Rosenbrock-42. The pain was that `F`, `CR`,
and `N` are static and problem-dependent, and the right values genuinely change with the landscape: a
smooth bowl wants moderate `F` and high `CR`; the Rastrigin egg-carton wants a diversifying `F` and,
crucially, a *low* `CR` (the one-coordinate-at-a-time policy the GA's `indpb=1/n` mutation accidentally
delivered and DE's `CR=0.9` denied). The three knobs are coupled and all problem-dependent. I am
hand-tuning the same algorithm over and over. The signal for learning them is sitting in the selection
step: every generation some trials beat their parents and some do not. The winners' `(F_i, CR_i)` are
evidence — "these worked on this landscape this generation" — so I should sample `F` and `CR` per
individual from distributions centered on recent winners and slide those centers toward the latest winners.
This is what would let *one* algorithm drive `CR` toward 0 on Rastrigin and toward high on Rosenbrock
without me choosing.

Work through the pieces the way a careful adaptive DE assembles them, because each is a decision I would
otherwise get wrong. First, the mutation strategy. Plain `DE/rand/1` (my last rung) picks a random base —
fine but slow, the very slowness the convergence columns exposed. The greedy fix `current-to-best/1`,
`v = x_i + F·(x_best − x_i) + F·(x_r1 − x_r2)`, pulls each individual partway toward the single best plus a
random difference. Fast on a unimodal bowl, but on a multimodal function *everyone* is dragged toward the
one incumbent best, so the whole population funnels into whatever basin that best sits in — premature
convergence, the Rastrigin death. The fix is to pull toward a *random one of the top few* rather than the
single best: `x_pbest` drawn from the top `N·p` individuals,
`v_i = x_i + F_i·(x_pbest − x_i) + F_i·(x_r1 − x_r2)`. Now `p` is a greediness dial, and its range is worth
computing so I know what "top few" means in practice. In the task loop `p_i` is drawn per individual from
`[p_min, p_max]` with `p_min = 2/N_init = 2/200 = 0.01` and `p_max = 0.2`; the pbest pool is the top
`round(p_i·N)`, so at `N=200` that ranges from the top `round(0.01·200)=2` individuals to the top
`round(0.2·200)=40`. A small-`p` individual chases essentially the incumbent best two; a large-`p`
individual chases anywhere in the top forty. Randomizing `p_i` per individual instead of fixing one `p`
spreads the population's attraction across several good basins at once — some members exploit the very
best, others a broad good set — which is exactly the diversity Rastrigin demands bought without giving up
the guides that speed Rosenbrock and Ackley. This `current-to-pbest/1` is strictly more flexible than
current-to-best/1: setting `p` to its smallest value recovers current-to-best, so I lose nothing and gain a
diversity dial.

Second, the difference `x_r1 − x_r2`. If both come from the current population, then as the population
converges this difference shrinks toward zero along with everything else, and I lose diversification
exactly when I might be stuck in a Rastrigin pit. The parents that just *lost* selection encode where the
search recently was and chose to leave; keep them in an external archive `A` and draw `x_r2` from `P ∪ A`.
Now the difference can reach back to a recently-abandoned region, adding diversity without enlarging the
live population (which would cost evaluations I already showed the budget cannot spare). I cap the archive
at the current population size and, when it overflows, drop random members — a reservoir of recent history,
not a museum that biases the search. So `r1` from `P` only, `r2` from `P ∪ A`, distinct from each other and
from `i`.

Third — where the sampling distributions earn their keep — what do I sample `F` and `CR` from, and how do I
summarize the winners? `CR` is a probability in `[0,1]` that should settle near a stable good value, so
Normal is natural: `CR_i = N(M_CR[r], 0.1)`, clamp to `[0,1]`. A tight Normal keeps `CR` near its learned
center, which is what I want for a steady parameter. `F` is different: it controls mutation magnitude, and
the failure mode I fear most — premature convergence, the Rastrigin death — is precisely `F` collapsing to
small values too early and the search going quiet. So `F` wants a distribution that keeps *proposing* large
values even when the center drifts down: a heavy tail. Cauchy: `F_i = M_F[r] + 0.1·tan(π(rand − 0.5))`,
same center as a Normal but fat tails, routinely throwing an `F` far from the center. Let me quantify "far"
so the choice is earned and not folklore. Ask how often each sampler proposes an `F` at least `0.3` above
its center — a genuinely aggressive step. For the Cauchy that needs the standard Cauchy variate to exceed
`3`, and `P(C > 3) = 0.5 − arctan(3)/π = 0.5 − 0.398 = 0.102`, about one draw in ten. For the Normal it
needs `Z > 3`, `P(Z > 3) = 0.00135`, about one draw in 740. So the Cauchy sampler throws a large,
population-reviving `F` roughly 75 times more often than the Normal would — that is the concrete meaning of
"never let `F` go quiet," and it is why `F` gets the fat tail and `CR` does not. The truncation rules
respect what each parameter is: `F > 1` truncate to 1 (huge `F` is unstable), but `F ≤ 0` I *resample*
until positive rather than clamp, because a non-positive scaling inverts or kills the mutation — the lower
end is degenerate. Clamping `CR` at both ends is fine because both endpoints are meaningful.

How do I summarize successful `F` into a new center? Arithmetic mean pulls toward the bulk of successes,
which on a converging population tend to be smallish, so the center drifts down and down, mutation weakens,
and I manufacture the exact premature convergence I am avoiding. I need a summary that *resists* that
pull — extra weight to the larger successful `F`. The Lehmer mean does this: `mean_L(S) = (Σ S_k²)/(Σ S_k)`,
each term squared in the numerator but first-power in the denominator, so larger elements pull the ratio
up. That it is always at least the arithmetic mean is not a hope, it is an identity: `mean_L − mean_A =
Var(S)/mean_A ≥ 0`, zero only when every success coincides. Check it on numbers: for `S = {0.3, 0.9}` with
equal weight the arithmetic mean is `0.6`, while the Lehmer mean is `(0.09+0.81)/(0.3+0.9) = 0.90/1.2 =
0.75` — pulled up toward the larger `0.9`, and indeed `0.75 − 0.6 = 0.15 = Var/mean_A` with
`Var = 0.09`. So Lehmer-averaging successful `F` keeps the center biased toward the larger successes and
stops the magnitude dying. And not every success is equally informative: a trial that improved fitness by a
lot is stronger evidence than one that barely squeaked past. Weight each winner by its improvement
`Δf_k = |f(x_k) − f(u_k)|`, normalize `w_k = Δf_k/Σ Δf_l`, and use the weighted Lehmer mean for `F` and a
weighted arithmetic mean for `CR`. Trace it once so the update is concrete: if two winners carry
`F = {0.4, 0.8}` with improvements `Δf = {1, 3}`, the weights are `{0.25, 0.75}`, the weighted arithmetic
mean is `0.7`, and the weighted Lehmer mean is `(0.25·0.16 + 0.75·0.64)/(0.25·0.4 + 0.75·0.8) = 0.52/0.70 =
0.743` — the big-improvement winner at `F=0.8` drags the new center to `0.743`, above the arithmetic `0.7`.
That asymmetry — Cauchy/Lehmer for `F`, Normal/arithmetic for `CR` — is "let `CR` settle, but never let `F`
go quiet."

Before I go further I want to check that this machinery actually delivers the landscape-dependent `CR` I
have been promising, because the whole rung stands on it. The mechanism is the selection filter, and it
runs itself. On separable Rastrigin, a trial built with high `CR` overwrites most coordinates at once, and
I argued last rung that such a bundle is almost always rejected on a survivor; the trials that *do* improve
are overwhelmingly the low-`CR` ones that touched a coordinate or two, so the winners fed into the memory
carry small `CR`, the weighted arithmetic mean is small, and `M_CR` slides down. On Rosenbrock the reverse:
a trial that nudges two coordinates cannot descend a valley that demands the coordinates move together, so
the improving trials are the high-`CR` coordinated ones, their `CR` values are large, and `M_CR` slides up.
The same update rule pushes the center to opposite ends purely because selection admits different winners on
the two surfaces. Translate the settled centers back into coordinates moved, using last rung's donor count
`1 + (D−1)·CR`: a Rastrigin-driven `M_CR ≈ 0.1` yields about `1 + 29·0.1 ≈ 4` coordinates per trial at 30D —
essentially the GA's one-axis-at-a-time regime — while a Rosenbrock-driven `M_CR ≈ 0.9` yields
`1 + 29·0.9 ≈ 27`, the fully coupled valley move DE used. So one algorithm can occupy the exact
per-coordinate regime each fixed method needed, and it gets there without me choosing, because the sampling
width cooperates: `CR ~ N(M_CR, 0.1)` keeps draws within about `±0.3` of the center (three standard
deviations, before the `[0,1]` clamp), so a center near `0.1` samples a tight cloud of small `CR` and a
center near `0.9` a tight cloud of large `CR`, each self-consistent with the winners that produced it.

Now the fragility that the single-center version (JADE-style) carries: there is *one* pair `(M_F, M_CR)`
steering every individual. Selection is stochastic; some generation, by luck, the winning trials carry
mediocre `F`/`CR` for unrelated reasons, the update slides my one center toward them, and next generation
the *entire* population samples from the contaminated center. On a hard multimodal problem where success is
noisy and rare — Rastrigin — this is the normal weather, not a corner case. The cure is redundancy: keep a
*set* of `H` centers, `M_F = M_CR = [0.5,...,0.5]`, each individual picks an index `r` uniformly and
samples around that slot, and each generation writes the winners' summary into *one* slot, cycling with a
counter `k` round-robin. A contaminated summary then lands in exactly one of the `H` slots; next generation
only about `1/H` of the population draws from it while the other `H−1` slots keep things sane, and the bad
slot is overwritten in a few generations. Round-robin matters: overwriting every slot each generation would
be a single effective center with extra steps; one slot per generation keeps the `H` slots a diverse
buffer of what worked at different times. For this loop `H = 6`, so at most one sixth of the population ever
draws from a freshly-contaminated slot. (I keep the edge cases concrete: a generation with no winners
updates nothing; the weighted Lehmer/arithmetic means use a small `1e-30` denominator guard. The
single-center variant also locks `CR` to a terminal `CR=0` regime on slots where all successful `CR` are
zero, but this loop does not carry that sentinel — it just writes the weighted means and relies on the
Normal sampler to drive `CR` low on its own, which is enough on this panel.)

That is a genuinely self-tuning DE — SHADE — but it has done nothing about the third knob, `N`, which is
still fixed, and `N` is the one whose pain the convergence columns screamed: every method so far ran out of
budget. The tension is sharp. Early I want a *large* `N`: broad coverage of the box so I do not miss a
basin, and — specifically for an adaptive method — a large, diverse pool of successes per generation so the
memory has good statistics to learn `F`/`CR` from (at `N=200` a generation can hand the memory up to 200
winners to average, at `N=4` at most four, so the adaptation is only trustworthy while the population is
still large — another reason to spend the big `N` early). Late, once the population has localized, a large
`N` is
pure waste: hundreds of evaluations per generation to make one tiny refinement step, when what I want near
the end is *many generations* of small precise moves to polish the basin. With fixed `N`, the budget buys
`budget/N` generations, full stop — I cannot have both the broad early population and the many late
generations. The fix: start large and *shrink linearly* over the run. Put the arithmetic on it in the
canonical accounting, where the budget is a fixed number of evaluations. Pinned at `N = 200`, that budget
buys `budget/200` generations. Sliding `N` linearly from `200` down to `4` makes the mean population size
about `(200+4)/2 = 102`, so the same evaluation budget now buys roughly `budget/102` generations — close to
twice as many — and every one of the extra generations falls in the cheap, late, small-`N` regime, which is
exactly the many-small-steps refinement phase the unconverged GA never reached. Shrinking manufactures late
generations out of budget a fixed large `N` would have burned on redundant exploration, and it respects
DE's self-scaling: a smaller converged population still has small differences, so refinement stays
fine-grained.

I want a *deterministic* monotone schedule — reactive resizing would replace one parameter with several
meta-parameters governing when and how much to resize, and I have spent two rungs eliminating hand-tuned
parameters; I will not reintroduce a fistful to fix the last one. Linear is the simplest monotone decrease
and needs only the initial size. Here the schedule is by generation fraction:
`N_next = round(N_init + (N_min − N_init)·(gen+1)/n_generations)`, floored at `N_min`. `N_min = 4` is pinned
by the operator, not chosen — `current-to-pbest/1` needs `x_i`, a distinct `x_pbest`, `x_r1`, and `x_r2`,
four distinct individuals, so the population can never drop below four without the mutation becoming
ill-defined. When `N` must drop I delete the *worst* individuals by fitness (elitist shrink keeps the elites
and accelerates late focusing) and trim the archive to the new population size.

One adaptation to this task's budget is load-bearing and I will not gloss it. The canonical L-SHADE sets
`N_init = 18·D`, sized for the large evaluation budgets of competition benchmarks. At 100D that is
`18·100 = 1800` — nine times the harness's `pop_size`. But this harness runs a *fixed, small*
`n_generations` with a fixed `pop_size`, and with the matched total-evaluation budget here a starting
population of 1800 would consume the whole budget in a handful of generations, so the linear reduction never
gets the runway to reach the small-`N` refinement regime at all — it would starve Rastrigin-100D of exactly
the late generations I just argued the shrink exists to create. So I use `N_init = pop_size` as given by the
harness and let the linear reduction to `N_min = 4` do its work over the fixed generation count, keeping the
budget identical to the other rungs. This is the one place the task's implementation deliberately departs
from the textbook constant, and it is the right call for *this* budget — but I flag it, because it is also
the one place I am trading away early exploration, and if any benchmark punishes that trade it will be the
high-dimensional multimodal one.

Now the falsifiable bet against the GA and DE numbers. The self-adapting `CR` should let one algorithm be
right on all four landscapes where neither fixed method was: I expect it to drive `CR` low on Rastrigin
(recovering the GA's per-coordinate win) while keeping the anisotropic difference vector (recovering DE's
valley and basin power). Concretely: Ackley should drop from the GA's 1.05 and DE's 10.0 mean to *near
machine precision* — the broad smooth basin is where adaptive `F` plus difference vectors should fully
solve it, and I expect it to converge early rather than burn the budget, so watch the convergence column
fall well below 500 there. Rosenbrock should beat both the GA's 87/mean-138 and recover toward DE's
anisotropy, landing in the low tens or better. Rastrigin-30d should be competitive with the GA's 8 (the
low-`CR` regime plus linear reduction's refinement). The one place I am genuinely uncertain is
Rastrigin-100d: the deliberate `N_init=pop_size` (not 18·D) choice trades exploration for refinement, and
Rastrigin at 100D is the hardest multimodal case, so it may land *near* the GA's 114 rather than crushing
it — and if the trade is bad it could even come in above the GA. If the result is near-zero Ackley, sub-20
Rosenbrock, and single-digit-ish Rastrigin-30d with Rastrigin-100d roughly at the GA's level, then the
adaptive machinery has done what two fixed methods could not — one algorithm right across landscapes — and
this is the strongest rung; the one column I will read first, and worry about, is Rastrigin-100d.
