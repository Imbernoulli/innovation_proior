The L-SHADE numbers came back and they confirm the rung's whole thesis: one adaptive
algorithm *is* right across landscapes where the two fixed methods each failed somewhere.
Ackley fell to 4.7e-11 at seed 42 (mean 9.4e-12) — solved, near machine precision, the broad
smooth basin that DE's noisy 10.0 mean and the GA's 1.05 both left on the table; the adaptive
`F` plus the difference vector took it all the way down, and it converged early (gen 269 at seed
42, mean 258 of the 500-generation budget) rather than burning the run. That early plateau is
worth its own line: 258/500 is barely past the halfway mark, so on the one problem the adaptation
solved outright it recovered more than 240 generations it did not need. Rosenbrock fell to 15.0
(mean 13.3) — beating DE's seed-42 32.2 by better than half and crushing the GA's mean 138 by a
factor of 10.4, because the adaptive `CR` recovered the anisotropic coordinated moves the GA's
axis-aligned operators could not make. DE's own rosenbrock mean of 48302 against its seed-42 of
32.2 is a factor of ~1500 — a sign that DE diverged outright on at least one seed; L-SHADE's
13.3 mean against its 15.0 seed-42 is by contrast tight, the adaptation holding across seeds.
Rastrigin-30d landed at 7.9 (mean 7.3), edging the GA's mean 8.05 by 0.75 — competitive, about
9% better, the low-`CR` regime plus linear-reduction refinement matching the GA's per-coordinate
win. So three of four landscapes are clearly the best of the ladder, two of them by an order of
magnitude or more. But look at the one I flagged as uncertain: Rastrigin-100d came back at 120.6
at seed 42 (mean 132.5) — and that is *worse* than the GA's mean of 114, by 18.8 absolute, about
16.5% higher. This is the only cell on the whole panel where the strongest baseline does not lead
the ladder, and it is exactly the cell I predicted would be the hardest: high dimension times
multimodality. The deliberate `N_init=pop_size` choice — forced by this fixed-small-budget
harness, where a large canonical `N_init` would starve the generation count — traded exploration
for refinement, and on the hardest multimodal case at 100D that trade did not pay: the population
reduced too aggressively before it had explored enough of the 100-dimensional egg-carton to find
the global basin. L-SHADE is the strongest rung, but its weak spot is precisely where I predicted,
and the failure mode is the exploration/exploitation balance tipping to exploitation too early.

Let me make that diagnosis precise before I act on it, because the cure depends on which of two
things went wrong. One possibility is that the *parameters* mis-adapted: the success-history memory
drove `F` small (mutation went quiet) or `CR` to a value that did not suit the separable 100-D
landscape. The other is that the *population schedule* committed too fast: linear reduction shed
members while the search was still scattered across the egg-carton, so the survivors clustered
around a non-global basin and the shrunken population could no longer climb out. The convergence
column is the tell, and it discriminates the two cleanly. At 100D L-SHADE's `convergence_gen` mean
is 994.67 of 1000 — the first generation within 1% of the final best did not arrive until
essentially the last 0.5% of the run, meaning the best value was *still improving right up to the
budget wall*. Set that against Ackley, where convergence came at gen 258 of 500, a quarter of the
way in: on the problem it solved, the curve went flat with three quarters of the budget to spare;
on Rastrigin-100d the curve never went flat at all. So it is not that the search converged to a
wrong answer and sat there — a stuck search would show a low `convergence_gen`, an early plateau
on a bad value. It is the opposite: the search was still moving at gen 995, just too slowly, having
spent its early generations on a population too small (`N_init=pop_size`, not a large multiple of
`D`) to cover 100 dimensions, and then having thinned that already-small population by linear
reduction. The seed spread reinforces this reading. Seed 42 gave 120.6 but the mean is 132.5, so
the other two aggregated seeds average about 138.5 — a spread of nearly 18 across seeds, the
signature of a search whose final answer depends heavily on which basin the too-small early
population happened to sample. Both readings — parameters going quiet, or schedule committing fast
— point the same way: I need more effective exploration in the first half of the run *without*
enlarging the population budget, because the budget is fixed by the harness. That single constraint
is what makes this hard, and it is worth pinning down what it forbids before I decide what to do.

The obvious fix is a bigger initial population, and I have to rule it out with arithmetic rather
than wave it away. The canonical initial size for this family of adaptive DE is on the order of
`25*ln(D)*sqrt(D)`. For D=100 that is `25 * 4.605 * 10 ≈ 1151`, about 5.75x the harness `pop_size`
of 200; for D=30 it is `25 * 3.401 * 5.48 ≈ 466`, about 2.3x. Now, the harness fixes both
`pop_size` and `n_generations`, and the linear-reduction scheme spends roughly
`n_generations * (N_init + N_min)/2` evaluations. With `N_init=200`, `N_min=4`, `n_generations=1000`
that is about `1000 * 102 = 102000` evaluations on the 100D problem. If I instead start at 1151 and
keep `n_generations` fixed, I spend `1000 * (1151+4)/2 ≈ 577500` evaluations — 5.7x over budget,
which the fixed harness will not grant. If instead I hold the evaluation budget fixed and let the
big `N_init` eat generations, the generation count collapses by the same factor, and a search that
was already *still improving at gen 995* under 1000 generations would be cut off far earlier — the
exact starvation the L-SHADE rung already measured directly when it noted that the canonical large
initial size degraded Rastrigin-100D from 128 to 313. So a larger `N_init` is not merely
unattractive here; it makes the measured failure strictly worse, because the failure *is* running
out of generations while still improving. That closes the obvious door and forces the subtler one:
extract more exploration from the same population by changing *how* each individual moves early.

Before I settle on that, two other doors are worth walking through, because each is tempting and
each fails for a reason I can name from the numbers. The first is a restart or diversity-injection
scheme: detect stagnation and re-seed part of the population into fresh random points. But the
convergence column just told me the 100D search never stagnated — `convergence_gen` 994.67 means
the best was still falling at the wall. A stagnation trigger keyed on a flat curve would essentially
never fire on the problem I am trying to fix, and where it did fire it would discard a trajectory
that was still descending, spending evaluations I cannot spare to re-explore ground the difference
vector was already covering. So restart is the right tool for the wrong failure: it treats
premature convergence, and what I have is premature *commitment under a still-moving curve*, which
is not the same thing. The second door is a global regime switch — run a more exploratory base
vector such as `rand/1` early and switch to the elite-guided `current-to-pbest` late. This at least
targets the balance, but it throws away the elite guidance during the early phase entirely, and the
elite guidance is exactly what owns Rosenbrock (10.4x better than the GA) and Ackley (eleven orders
of magnitude better than the GA's 1.05). A hard early switch to `rand/1` would risk those two wins
to help the third, and there is no reason the same schedule that helps 100D Rastrigin should be
right for the smooth Ackley basin where the elite pull is already converged by gen 258. A binary
switch is too blunt: it changes the base vector wholesale rather than re-weighting the two forces
already present in it.

That last observation points straight at the one place L-SHADE's mutation is structurally crude, and
re-weighting rather than switching is the natural move. Look at the donor:
`v_i = x_i + F_i*(x_pbest - x_i) + F_i*(x_r1 - x_r2)`. The *same* `F_i` multiplies two structurally
different terms. The first, `x_pbest - x_i`, is a directed pull toward the elite — pure
exploitation, dragging the individual toward a good solution already found. The second,
`x_r1 - x_r2`, is the self-scaling random difference — the exploration, the perturbation carrying
the population's scatter. Tying both to one `F_i` forces the elite-pull strength and the
random-perturbation strength to move in lockstep. But the Rastrigin-100d failure is precisely a
balance problem: the search committed to the elite (and shrank the population toward it) before it
had perturbed enough to find the right basin. If I could make the elite pull *weaker than* the
perturbation early — explore without committing — and *stronger than* it late — refine hard once the
basin is found — I would directly attack the failure I just measured. A single `F_i` cannot express
that phase dependence; a base-vector switch expresses only two coarse states. What I want is a
continuous re-weighting. So the move is to give the elite-pull term its own factor: a weighted `Fw`
on `(x_pbest - x_i)`, with `F_i` left on the random difference. Early, `Fw < F_i` (perturb more than
pull); late, `Fw > F_i` (pull more than perturb). A three-step schedule on the budget fraction
realizes it: `Fw = 0.7*F_i` for the first 0.2 of the run, `0.8*F_i` for the first 0.4, `1.2*F_i`
thereafter. The donor becomes `v_i = x_i + Fw*(x_pbest - x_i) + F_i*(x_r1 - x_r2)` —
current-to-pbest *weighted*.

Let me trace what that actually does to a coordinate, because the numbers make the mechanism
concrete rather than a slogan. Take a representative `F_i = 0.5`. In L-SHADE the donor coordinate is
`x_i + 0.5*(pbest - x_i) + 0.5*(diff)`: the pull coefficient equals the perturbation coefficient,
ratio 1.0 — the elite pull and the random kick are always exactly balanced, whatever the phase.
Under the weighting, early (`frac < 0.2`) the pull coefficient becomes `0.7*0.5 = 0.35` while the
perturbation stays at `0.5`: the individual moves only 35% of the way toward the elite but still
takes a full 0.5x random difference, a pull/perturb ratio of 0.7 — the elite pull is cut by 30%
relative to L-SHADE while the scatter is untouched. Late (`frac >= 0.4`) the pull coefficient
becomes `1.2*0.5 = 0.6` against the same `0.5` perturbation, a ratio of 1.2 — now the pull is 20%
stronger than the kick. Across the run the pull/perturb ratio therefore swings from 0.7 to 1.2, a
factor of 1.71: the balance the single `F_i` pinned at a flat 1.0 is now slid by 71% from
exploration-heavy to exploitation-heavy. That is exactly the temporal correction the convergence
data asked for — keep the 100D population scattering while it hunts for the basin, then let it drive
hard once it has one. And it costs nothing in budget: the donor is still one pass over `dim`
coordinates per individual, `O(N*dim)` per generation, identical to L-SHADE; the `Fw` selection is
`O(1)` per individual and the extra scalar multiply per coordinate is negligible against the one
function evaluation each trial already pays, which is the term that actually sets the budget. The
shapes check out too: `x_i, x_pbest, x_r1, x_r2` are all `D`-vectors, `Fw` and `F_i` scalars, so the
donor is a `D`-vector, and the binomial crossover with a guaranteed `j_rand` coordinate keeps at
least one donor gene, so every trial differs from its parent in at least one coordinate and no
evaluation is wasted on a clone. This is the heart of the improvement.

Now a real checkpoint, because I could be tempted to think the pbest-pool change alone carries the
day and skip the weighting. Suppose I only made the elite pool broader and drew `pbest` from a
larger, more diverse top slice early — does that fix 100D on its own? It changes the *direction* of
the elite pull (a pbest sampled from a wider pool points at a more varied set of good basins) but
not its *magnitude*: the coefficient on `(pbest - x_i)` is still `F_i`, still equal to the
perturbation. The measured failure was commitment magnitude — the population collapsing toward one
basin and shrinking — so a broader pool diversifies which basin it commits to without slowing the
commitment. Necessary help, but not sufficient. That confirms the weighting has to carry the
temporal balance and the pool change is a complement, not a substitute; I keep both, but I do not
lean the whole fix on the pool.

Several smaller refinements reinforce the same exploration-early / exploitation-late arc, and each
recovers budget L-SHADE spent re-learning what I already know. First, the memory start. L-SHADE
initializes every slot neutral at `0.5`, so the early generations re-discover on every single
problem that a high `CR` helps while exploring — coordinated multi-coordinate moves are what a
spread-out population needs to make progress, and it costs generations to relearn that each time. I
already believe high `CR` is right early, so I initialize the `CR` slots high, at `0.8`, and leave
`F` at `0.5` — a free head start that the adaptation will override within a few dozen generations if
a near-separable problem prefers low `CR`. Second, a permanent aggressive reservoir: *freeze* one
memory slot (the last) at `M_F = M_CR = 0.9` and never update it. Because each individual picks its
memory index uniformly over the `H` slots, a fixed `1/H` fraction of the population always samples
around `(0.9, 0.9)` regardless of what the rest of the memory decides. With `H = 5` that fraction is
`1/5 = 20%`: at any generation about one individual in five fires an aggressive high-`F`, high-`CR`
move — cheap, always-on insurance against the whole population going quiet, which is one exact
description of the Rastrigin-100d collapse. Third, phase rails: for the first quarter of the budget
floor `CR` at `0.7`; for the first half, floor at `0.6`; for the first 0.6, cap `F` at `0.7`. These
are not adapted — they are hard rails keeping early sampling in the explore-coordinated regime and
relaxing as the run proceeds so the adaptation takes over for refinement. The `F<=0.7` early cap
matters specifically because I am about to bias the memory and freeze a slot upward; without a cap
the frozen 0.9 plus a high-drifting memory could let `F_i` truncate at 1 too often early and destroy
structure faster than the population can rebuild it. Fourth, the memory update: instead of
overwriting a slot with one noisy generation's summary, *blend* it,
`M[k] <- (mean_WL(S) + M[k]_old)/2`, so a single generation moves a slot at most halfway to its
summary and the slot retains its history. It is worth checking this does not cripple adaptation
speed. The blend has a half-life of one update — after `g` consecutive updates toward a fixed target
the residual gap is `0.5^g`, so three updates leave 12.5% of the original error. With `H-1 = 4`
adaptable slots and round-robin writing, each slot is touched roughly once every four generations,
so over 1000 generations a slot receives about 250 updates: far more than enough for a half-per-step
geometric convergence to fully settle. The blend costs essentially nothing in final adaptation while
buffering any one unlucky generation from throwing a slot to an extreme — a pure stabilizer.

The pbest greediness `p` gets the same phase treatment, and here I can put numbers to the "broader
early pool" I just argued for. L-SHADE samples `p_i` per individual from a fixed interval up to
`p_max = 0.2`; here I want `p` to *decrease* deterministically over the run, from a broad elite pool
early to a sharp one late: `p = p_max - (p_max - p_min)*nfes/max_nfes`, with `p_max = 0.25`,
`p_min = 0.125`. At the start, with the full `N = 200`, the elite pool is `0.25 * 200 = 50`
candidates — the top quarter — against L-SHADE's ceiling of `0.2 * 200 = 40`, a 25% wider pool, so
the early `pbest` guide is drawn from a genuinely more diverse set of basins on exactly the high-D
multimodal case that failed. As the run proceeds two things shrink the pool together: `p` slides
toward 0.125 and `N` slides toward `N_min` under linear reduction, so `n_pbest = round(p*N)` falls
from 50 toward the floor of 2 (`max(2, ...)`), a sharp single-elite pull at the end. This is the
same exploration-to-exploitation arc as the weighted mutation, applied to *where the elite guide
comes from* rather than how hard it pulls, and the two arcs reinforce: early, a weak pull toward a
diverse elite; late, a strong pull toward the single best.

The constants and edge cases carry over from the adaptive-DE substrate, and each has a reason I can
state. Memory size `H = 5`, with the last slot frozen, so four slots adapt and the round-robin
counter cycles over those four; the `1/H = 20%` aggressive fraction above is why `H` is kept small
rather than the 6 the plain adaptive baseline used — a larger `H` would dilute the frozen reservoir
below one-in-five. `N_min = 4`, pinned because the weighted `current-to-pbest/1` donor references
four roles that want to be distinct: `x_i`, the `pbest`, `x_r1` (drawn `!= i`), and `x_r2` (drawn
from population-plus-archive, `!= i, != r1`); at `N = 4` the indices `{0,1,2,3}` can furnish
`i, r1, r2` distinct with the fourth free for `pbest`, so four is the floor below which the donor
degenerates. The archive of beaten parents caps at the current population size with random deletion
on overflow, rescaling as the population shrinks so it never dwarfs the live population. Cauchy on
`F` (resample if `<= 0`, truncate to 1) keeps a heavy tail in the mutation strength so the occasional
large step survives; Normal on `CR` (clamp to `[0,1]`) keeps the crossover rate stable. The
terminal-`CR = 0` rule is the one edge worth dwelling on: when a slot's successful `CR` values are
all zero I lock that slot to `CR = 0`, which in the binomial crossover means only the guaranteed
`j_rand` coordinate takes the donor value — a single-coordinate move. On a separable landscape like
Rastrigin the coordinates are independent, so optimizing one axis at a time without disturbing the
others is precisely the per-coordinate behaviour that let the GA edge DE and that L-SHADE's low-`CR`
regime matched at 7.3; the terminal lock makes that regime an absorbing state on the rugged separable
problems that prefer it. And — the one place this implementation must depart from the canonical
constant, exactly as the L-SHADE rung did — `N_init` is the harness's `pop_size`, not the canonical
`25*ln(D)*sqrt(D)`, because as the arithmetic above showed that large initial population (1151 at
D=100, 5.75x pop_size) would starve the generation count on a search that is still improving at the
budget wall. The weighted mutation and the phase schedule have to do their work *within* the same
population budget as every other rung, not by enlarging it. The full scaffold module — the literal
fill of `run_evolution`, with the three operator stubs left as no-ops — is in the answer.

Now the bar this finale must clear, stated against L-SHADE's real numbers, with no result invented.
The three landscapes L-SHADE already won should hold or improve. Ackley should stay at machine
precision: it was already solved to 9.4e-12 with the curve flat by gen 258, so the late-run stronger
elite pull cannot hurt a basin that is already collapsed, and I expect `best_fitness` to remain in
the 1e-11 range with `convergence_gen` still well inside the first half. Rosenbrock should stay in
the low tens or better than L-SHADE's 13.3 mean, because the phase schedule's late `Fw = 1.2*F_i`
sharpens the valley descent exactly when the population has found the valley — if anything I expect a
small improvement here, not a regression. Rastrigin-30d should stay around or below L-SHADE's 7.3
mean, since the terminal-`CR` lock and the low-`CR` regime that owned it are preserved and reinforced.
The decisive test is Rastrigin-100d, the one L-SHADE lost to the GA by 18.8 (132.5 vs 114): the
weaker early elite pull (ratio 0.7 vs L-SHADE's flat 1.0), the broader early `p`-pool (50 vs 40
candidates), the high-biased `CR` memory, and the always-on 20% aggressive reservoir should together
keep the population exploring the 100-D egg-carton longer before it commits — directly the correction
the still-improving 994.67 convergence and the 18-point seed spread called for. So I expect
Rastrigin-100d `best_fitness` to fall *below* L-SHADE's 132.5 mean and, the real target, below the
GA's 114 — recovering the ladder's lead on the one benchmark where the strongest baseline slipped. If
Rastrigin-100d does *not* improve, the weighted-mutation thesis is falsified for this harness's
budget: it would mean the early-exploration bias cannot overcome the forced small `N_init` on the
hardest case, and the honest conclusion would be that on *this* fixed-budget panel the weighted
donor matches rather than beats L-SHADE. The falsifiable claim is sharp: same-or-better `best_fitness`
on the three L-SHADE already owns, and strictly below 132.5 on Rastrigin-100d, with its
`convergence_gen` no earlier than L-SHADE's — a search that keeps exploring longer should, if
anything, still be improving near the wall, not plateau early on a worse value.
