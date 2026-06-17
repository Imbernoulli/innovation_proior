The GA numbers landed almost exactly on the three-way bet I made, and the pattern is the whole argument for
this rung. Rastrigin: 8.99 at 30D (mean 8.05) and 115 at 100D (mean 114) — a decisive win over DE's 172
and 926. The per-coordinate polynomial mutation did what I predicted: on the separable egg-carton, fixing
one axis at a time beats DE's coupled `CR=0.9` move by an order of magnitude at 100D. But the two
regressions I feared are right there too. Rosenbrock: 87 at seed 42, mean **138** — far worse than DE's
seed-42 32.2. Axis-aligned SBX and polynomial mutation have no notion of the diagonal valley, so the GA
crawls down Rosenbrock's curved ribbon with moves that are never aligned to it, exactly as I called.
Ackley: 1.05 (mean 1.12) — better than DE's noisy 10.0 mean but nowhere near solved; the broad smooth
basin is where DE's anisotropic difference vectors should have shone and where the GA's axis-aligned
operators leave a full unit of error on the table. And note the convergence columns: the GA is still
improving at generation ~490–500 on the 30D problems and at 1000 on 100D — it has not converged, it has
run out of budget. So I now have a clean, contradictory lesson from two rungs. DE's self-scaling
anisotropic difference vector is the right *mutation* (it owns Rosenbrock and Ackley when it enters the
basin) but its single global `F`/`CR` is catastrophic on Rastrigin and its fixed `rand` base is slow. The
GA's per-coordinate locality fixes Rastrigin but throws away the anisotropy that owns the valley. No single
fixed setting of either method is right on all four landscapes. That is the thing to kill: the parameters
must learn themselves from what is working on *this* problem, right now.

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
`v_i = x_i + F_i·(x_pbest − x_i) + F_i·(x_r1 − x_r2)`. Now `p` is a greediness dial — small `p` is
aggressive, a moderate `p` spreads the attraction over several good basins so the population does not all
rush one hole. This `current-to-pbest/1` is strictly more flexible than current-to-best/1, and it keeps the
diversity Rastrigin demands while exploiting the good guides that speed up Rosenbrock and Ackley. In the
task loop the greediness is sampled per individual: each one draws `p_i` uniformly from `[p_min, p_max]`
with `p_min = 2/N_init` and `p_max = 0.2`, then takes its `pbest` from the top `round(p_i·N)` (at least 1).
Randomizing `p_i` per individual instead of fixing one `p` widens the spread of guides — some individuals
chase the very best, others a broader good set — which is extra diversity for free.

Second, the difference `x_r1 − x_r2`. If both come from the current population, then as the population
converges this difference shrinks toward zero along with everything else, and I lose diversification
exactly when I might be stuck in a Rastrigin pit. The parents that just *lost* selection encode where the
search recently was and chose to leave; keep them in an external archive `A` and draw `x_r2` from `P ∪ A`.
Now the difference can reach back to a recently-abandoned region, adding diversity without enlarging the
live population (which would cost evaluations). I cap the archive at the current population size and, when
it overflows, drop random members — a reservoir of recent history, not a museum that biases the search.
So `r1` from `P` only, `r2` from `P ∪ A`, distinct from each other and from `i`.

Third — where the sampling distributions earn their keep — what do I sample `F` and `CR` from, and how do I
summarize the winners? `CR` is a probability in `[0,1]` that should settle near a stable good value, so
Normal is natural: `CR_i = N(M_CR[r], 0.1)`, clamp to `[0,1]`. A tight Normal keeps `CR` near its learned
center, which is what I want for a steady parameter. `F` is different: it controls mutation magnitude, and
the failure mode I fear most — premature convergence, the Rastrigin death — is precisely `F` collapsing to
small values too early and the search going quiet. So `F` wants a distribution that keeps *proposing* large
values even when the center drifts down: a heavy tail. Cauchy: `F_i = M_F[r] + 0.1·tan(π(rand − 0.5))`,
same center as a Normal but fat tails, routinely throwing an `F` far from the center. The truncation rules
respect what each parameter is: `F > 1` truncate to 1 (huge `F` is unstable), but `F ≤ 0` I *resample*
until positive rather than clamp, because a non-positive scaling inverts or kills the mutation — the lower
end is degenerate. Clamping `CR` at both ends is fine because both endpoints are meaningful.

How do I summarize successful `F` into a new center? Arithmetic mean pulls toward the bulk of successes,
which on a converging population tend to be smallish, so the center drifts down and down, mutation weakens,
and I manufacture the exact premature convergence I am avoiding. I need a summary that *resists* that pull —
extra weight to the larger successful `F`. The Lehmer mean does this: `mean_L(S) = (Σ S_k²)/(Σ S_k)`, each
term squared in the numerator but first-power in the denominator, so larger elements pull the ratio up. The
weighted Lehmer mean minus the weighted arithmetic mean equals the weighted variance over the weighted
mean, which is nonnegative — so Lehmer is at least arithmetic, equal only when all successes coincide. So
Lehmer-averaging successful `F` keeps the center biased toward the larger successes and stops the magnitude
dying. And not every success is equally informative: a trial that improved fitness by a lot is stronger
evidence than one that barely squeaked past. Weight each winner by its improvement
`Δf_k = |f(x_k) − f(u_k)|`, normalize `w_k = Δf_k/Σ Δf_l`, and use the weighted Lehmer mean for `F` and a
weighted arithmetic mean for `CR`. That asymmetry — Cauchy/Lehmer for `F`, Normal/arithmetic for `CR` — is
"let `CR` settle, but never let `F` go quiet."

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
buffer of what worked at different times. For this loop `H = 6`. (I keep the edge cases concrete: a
generation with no winners updates nothing; the weighted Lehmer/arithmetic means use a small `1e-30`
denominator guard. The single-center variant also locks `CR` to a terminal `CR=0` regime on slots where all
successful `CR` are zero, but this loop does not carry that sentinel — it just writes the weighted means and
relies on the Normal sampler to drive `CR` low on its own, which is enough on this panel.)

That is a genuinely self-tuning DE — SHADE — but it has done nothing about the third knob, `N`, which is
still fixed, and `N` is the one whose pain the convergence columns screamed: every method so far ran out of
budget. The tension is sharp. Early I want a *large* `N`: broad coverage of the box so I do not miss a
basin, and — specifically for an adaptive method — a large, diverse pool of successes per generation so the
memory has good statistics to learn `F`/`CR` from. Late, once the population has localized, a large `N` is
pure waste: hundreds of evaluations per generation to make one tiny refinement step, when what I want near
the end is *many generations* of small precise moves to polish the basin. With fixed `N`, the budget buys
`budget/N` generations, full stop — I cannot have both the broad early population and the many late
generations. The fix: start large and *shrink linearly* over the run. Early generations cost a lot but buy
exploration and good adaptation statistics; as `N` shrinks each generation costs less, so the same
remaining budget buys *more* generations — exactly the many-small-steps regime the unconverged GA never
reached. Shrinking manufactures late generations out of budget a fixed large `N` would have burned on
redundant exploration, and it respects DE's self-scaling: a smaller converged population still has small
differences, so refinement stays fine-grained.

I want a *deterministic* monotone schedule — reactive resizing would replace one parameter with several
meta-parameters governing when and how much to resize, and I have spent two rungs eliminating hand-tuned
parameters; I will not reintroduce a fistful to fix the last one. Linear is the simplest monotone decrease
and needs only the initial size. Here the schedule is by generation fraction:
`N_next = round(N_init + (N_min − N_init)·(gen+1)/n_generations)`, floored at `N_min`. `N_min = 4` is pinned
by the operator, not chosen — `current-to-pbest/1` needs `x_i`, a distinct `x_pbest`, `x_r1`, and `x_r2`,
four distinct individuals. When `N` must drop I delete the *worst* individuals by fitness (elitist shrink
keeps the elites and accelerates late focusing) and trim the archive to the new population size.

One adaptation to this task's budget is load-bearing and I will not gloss it. The canonical L-SHADE sets
`N_init = 18·D`, sized for the large evaluation budgets of competition benchmarks. But this harness runs a
*fixed, small* `n_generations` with a fixed `pop_size`, and at 100D `18·D = 1800` would starve the search
of generations — with the matched total-evaluation budget here, that large `N_init` degrades Rastrigin-100D
badly rather than helping (the linear reduction never gets to run enough generations to refine). So I use
`N_init = pop_size` as given by the harness and let the linear reduction to `N_min = 4` do its work over the
fixed generation count, keeping the budget identical to the other rungs. This is the one place the task's
implementation deliberately departs from the textbook constant, and it is the right call for *this* budget.

Now the falsifiable bet against the GA and DE numbers. The self-adapting `CR` should let one algorithm be
right on all four landscapes where neither fixed method was: I expect it to drive `CR` low on Rastrigin
(recovering the GA's per-coordinate win) while keeping the anisotropic difference vector (recovering DE's
valley and basin power). Concretely: Ackley should drop from the GA's 1.05 and DE's 10.0 mean to *near
machine precision* (the broad smooth basin is where adaptive `F` plus difference vectors should fully
solve it). Rosenbrock should beat both the GA's 87/mean-138 and recover toward DE's anisotropy, landing in
the low tens or better. Rastrigin-30d should be competitive with the GA's 8 (the low-`CR` regime plus
linear reduction's refinement), and Rastrigin-100d should at least match the GA's 114 rather than DE's 926.
The one place I am genuinely uncertain is Rastrigin-100d: the deliberate `N_init=pop_size` (not 18·D)
choice trades exploration for refinement, and Rastrigin at 100D is the hardest multimodal case, so it may
land *near* the GA rather than crushing it. If the result is near-zero Ackley, sub-20 Rosenbrock, and
single-digit-ish Rastrigin-30d with Rastrigin-100d roughly at the GA's level, then the adaptive machinery
has done what two fixed methods could not — one algorithm right across landscapes — and this is the
strongest rung.
