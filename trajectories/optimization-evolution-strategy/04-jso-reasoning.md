The L-SHADE numbers came back and they confirm the thesis: one adaptive algorithm *is* right across
landscapes where the two fixed methods each failed somewhere. Ackley fell to 4.7e-11 at seed 42 (mean
9.4e-12) — solved, near machine precision, the broad smooth basin that DE's noisy 10.0 mean and the GA's
1.05 both left on the table; the adaptive `F` plus the difference vector took it all the way down, and it
converged early (gen 269 at seed 42, mean 258 of 500) rather than burning the budget. Rosenbrock fell to
15.0 (mean 13.3) — beating DE's seed-42 32.2 by better than half and crushing the GA's mean 138 by a
factor of 10.4, because the adaptive `CR` recovered the anisotropic coordinated moves the GA's
axis-aligned operators could not make; and where DE's Rosenbrock mean of 48302 against its seed-42 32.2
signalled outright divergence on some seed, L-SHADE's 13.3 mean against 15.0 is tight, the adaptation
holding across seeds. Rastrigin-30d landed at 7.9 (mean 7.3), edging the GA's 8.05 — competitive, the
low-`CR` regime plus linear-reduction refinement matching the GA's per-coordinate win. So three of four
landscapes are clearly the best so far, two by an order of magnitude or more. But look at the one I
flagged as uncertain: Rastrigin-100d came back at 120.6 at seed 42 (mean 132.5) — *worse* than the GA's
mean of 114, by 18.8 absolute. This is the only cell where the strongest method does not lead, and it is
exactly the cell I predicted would be hardest: high dimension times multimodality. The deliberate
`N_init=pop_size` choice — forced by this fixed-small-budget harness, where a large canonical `N_init`
would starve the generation count — traded exploration for refinement, and on the hardest multimodal case
at 100D that trade did not pay: the population reduced too aggressively before it had explored enough of
the 100-dimensional egg-carton to find the global basin.

Make that diagnosis precise before acting on it, because the cure depends on which of two things went
wrong. One possibility is that the *parameters* mis-adapted: the memory drove `F` small (mutation went
quiet) or `CR` to a value unsuited to the separable 100-D landscape. The other is that the *population
schedule* committed too fast: linear reduction shed members while the search was still scattered, so the
survivors clustered around a non-global basin and the shrunken population could no longer climb out. The
convergence column discriminates them cleanly. At 100D L-SHADE's `convergence_gen` mean is 994.67 of 1000
— the first generation within 1% of the final best did not arrive until essentially the last 0.5%, so the
best value was *still improving at the budget wall*. Set that against Ackley, where convergence came at
gen 258 of 500, a quarter of the way in: on the problem it solved the curve went flat with three quarters
of the budget to spare; on Rastrigin-100d it never went flat at all. So it is not that the search
converged to a wrong answer and sat there — a stuck search shows a *low* `convergence_gen`, an early
plateau on a bad value. It is the opposite: still moving at gen 995, just too slowly, having spent its
early generations on a population too small (`N_init=pop_size`, not a large multiple of `D`) to cover 100
dimensions, then thinned that already-small population by linear reduction. The seed spread reinforces it:
seed 42 gave 120.6 but the mean is 132.5, so the other two seeds average about 138.5 — a spread of nearly
18, the signature of a search whose answer depends heavily on which basin the too-small early population
happened to sample. Both readings point the same way: I need more effective exploration in the first half
*without* enlarging the population budget, because the budget is fixed.

The obvious fix is a bigger initial population, and I have to rule it out with arithmetic. The canonical
initial size for this family is on the order of `25·ln(D)·√D`. For D=100 that is `25·4.605·10 ≈ 1151`,
about 5.75× the harness `pop_size` of 200; for D=30, `25·3.401·5.48 ≈ 466`, about 2.3×. The
linear-reduction scheme spends roughly `n_generations·(N_init + N_min)/2` evaluations, so with `N_init=200`,
`N_min=4`, `n_generations=1000` that is about `1000·102 = 102000` on the 100D problem. Starting at 1151 and
keeping `n_generations` fixed spends `1000·(1151+4)/2 ≈ 577500` — 5.7× over budget, which the fixed harness
will not grant. If instead I hold the evaluation budget fixed and let the big `N_init` eat generations, the
generation count collapses by the same factor, and a search *already still improving at gen 995* would be
cut off far earlier — the exact starvation the L-SHADE run already measured when the canonical large
initial size degraded Rastrigin-100D from 128 to 313. So a larger `N_init` is not merely unattractive;
it makes the measured failure strictly worse, because the failure *is* running out of generations while
still improving. That forces the subtler door: extract more exploration from the same population by
changing *how* each individual moves early.

Two other doors are worth walking through first, because each is tempting and each fails for a reason I
can name from the numbers. A restart or diversity-injection scheme detects stagnation and re-seeds part of
the population into fresh random points — but the convergence column just said the 100D search never
stagnated (`convergence_gen` 994.67, still falling at the wall). A stagnation trigger keyed on a flat
curve would essentially never fire on the problem I am trying to fix, and where it did fire it would
discard a trajectory still descending. Restart is the right tool for the wrong failure: it treats
premature convergence, and what I have is premature *commitment under a still-moving curve*. The second
door is a global regime switch — run an exploratory base such as `rand/1` early and switch to
`current-to-pbest` late. It targets the balance, but it throws away the elite guidance entirely in the
early phase, and that guidance is exactly what owns Rosenbrock (10.4× the GA) and Ackley (eleven orders
below the GA's 1.05). A hard early switch would risk those two wins to help the third, and there is no
reason the schedule that helps 100D Rastrigin should suit the smooth Ackley basin already converged by gen
258. A binary switch is too blunt: it changes the base vector wholesale rather than re-weighting the two
forces already present in it.

That last observation points straight at the one place L-SHADE's mutation is structurally crude, and
re-weighting rather than switching is the natural move. Look at the donor:
`v_i = x_i + F_i·(x_pbest − x_i) + F_i·(x_r1 − x_r2)`. The *same* `F_i` multiplies two structurally
different terms. The first, `x_pbest − x_i`, is a directed pull toward the elite — pure exploitation. The
second, `x_r1 − x_r2`, is the self-scaling random difference — the exploration, the perturbation carrying
the population's scatter. Tying both to one `F_i` forces the elite-pull strength and the perturbation
strength to move in lockstep. But the Rastrigin-100d failure is precisely a balance problem: the search
committed to the elite (and shrank the population toward it) before it had perturbed enough to find the
right basin. If I could make the elite pull *weaker than* the perturbation early — explore without
committing — and *stronger than* it late — refine hard once the basin is found — I would directly attack
the failure I measured. A single `F_i` cannot express that phase dependence; a base-vector switch
expresses only two coarse states. So give the elite-pull term its own factor: a weighted `Fw` on
`(x_pbest − x_i)`, with `F_i` left on the random difference. Early, `Fw < F_i`; late, `Fw > F_i`. A
three-step schedule on the budget fraction realizes it: `Fw = 0.7·F_i` for the first 0.2 of the run,
`0.8·F_i` for the first 0.4, `1.2·F_i` thereafter. The donor becomes
`v_i = x_i + Fw·(x_pbest − x_i) + F_i·(x_r1 − x_r2)` — current-to-pbest *weighted*.

The numbers make the mechanism concrete. Take a representative `F_i = 0.5`. In L-SHADE the pull
coefficient equals the perturbation coefficient, ratio 1.0 — elite pull and random kick always exactly
balanced, whatever the phase. Under the weighting, early (`frac < 0.2`) the pull coefficient becomes
`0.7·0.5 = 0.35` while the perturbation stays at `0.5`, a pull/perturb ratio of 0.7 — the pull cut by 30%
while the scatter is untouched. Late (`frac ≥ 0.4`) the pull is `1.2·0.5 = 0.6` against `0.5`, a ratio of
1.2 — the pull 20% stronger than the kick. Across the run the ratio swings from 0.7 to 1.2, a factor of
1.71: the balance the single `F_i` pinned at a flat 1.0 is now slid from exploration-heavy to
exploitation-heavy. That is exactly the temporal correction the convergence data asked for — keep the
100D population scattering while it hunts for the basin, then let it drive hard once it has one — and it
costs nothing in budget, still one pass over `dim` coordinates per individual. This is the heart of the
improvement.

Before leaning the whole fix on the weighting, one checkpoint: does broadening the elite pool alone carry
the day? A `pbest` sampled from a wider top slice changes the *direction* of the elite pull (it points at
a more varied set of good basins) but not its *magnitude*: the coefficient on `(x_pbest − x_i)` is still
`F_i`, still equal to the perturbation. The measured failure was commitment magnitude — the population
collapsing toward one basin and shrinking — so a broader pool diversifies *which* basin it commits to
without slowing the commitment. Necessary help, not sufficient; I keep both the pool change and the
weighting, but the weighting carries the temporal balance.

Several smaller refinements reinforce the same exploration-early / exploitation-late arc, each recovering
budget L-SHADE spent re-learning what I already know. First, the memory start. L-SHADE initializes every
slot neutral at `0.5`, so early generations re-discover on every problem that a high `CR` helps while
exploring — coordinated multi-coordinate moves are what a spread-out population needs. I already believe
that, so I initialize the `CR` slots high at `0.8` and leave `F` at `0.5` — a head start the adaptation
will override within a few dozen generations if a near-separable problem prefers low `CR`. Second, a
permanent aggressive reservoir: *freeze* one memory slot (the last) at `M_F = M_CR = 0.9` and never update
it. Because each individual picks its memory index uniformly over the `H` slots, a fixed `1/H` fraction
always samples around `(0.9, 0.9)` regardless of the rest of the memory. With `H = 5` that is 20%: at any
generation about one individual in five fires an aggressive high-`F`, high-`CR` move — cheap, always-on
insurance against the whole population going quiet, one exact description of the Rastrigin-100d collapse.
Third, phase rails: for the first quarter of the budget floor `CR` at `0.7`, for the first half floor at
`0.6`, for the first 0.6 cap `F` at `0.7`. These are not adapted — hard rails keeping early sampling in
the explore-coordinated regime and relaxing as the run proceeds. The `F ≤ 0.7` early cap matters
specifically because I am about to bias the memory and freeze a slot upward; without it the frozen 0.9
plus a high-drifting memory could truncate `F_i` at 1 too often early and destroy structure faster than
the population rebuilds it. Fourth, the memory update: instead of overwriting a slot with one noisy
generation's summary, *blend* it, `M[k] ← (mean_WL(S) + M[k]_old)/2`, so a single generation moves a slot
at most halfway and the slot retains its history — a pure stabilizer, and with each adaptable slot touched
roughly once every four generations over 1000 generations it still settles fully.

The pbest greediness `p` gets the same phase treatment, and here I can put numbers to the "broader early
pool." L-SHADE samples `p_i` per individual up to `p_max = 0.2`; here I want `p` to *decrease*
deterministically, `p = p_max − (p_max − p_min)·nfes/max_nfes`, with `p_max = 0.25`, `p_min = 0.125`. At
the start, with the full `N = 200`, the elite pool is `0.25·200 = 50` candidates against L-SHADE's ceiling
of `0.2·200 = 40`, a 25% wider pool, so the early `pbest` guide is drawn from a more diverse set of basins
on exactly the high-D multimodal case that failed. As the run proceeds two things shrink the pool
together: `p` slides toward 0.125 and `N` slides toward `N_min`, so `n_pbest = round(p·N)` falls from 50
toward the floor of 2, a sharp single-elite pull at the end. Same exploration-to-exploitation arc as the
weighted mutation, applied to *where* the elite guide comes from rather than how hard it pulls, and the
two arcs reinforce: early, a weak pull toward a diverse elite; late, a strong pull toward the single best.

The constants and edge cases carry over from the adaptive-DE substrate. Memory size `H = 5`, with the last
slot frozen so four slots adapt and the round-robin counter cycles over those four — kept small rather
than the 6 the plain adaptive baseline used, because a larger `H` would dilute the `1/H` aggressive
fraction below one in five. `N_min = 4`, pinned because the weighted `current-to-pbest/1` donor needs
`x_i`, `x_pbest`, `x_r1`, and `x_r2` distinct. The archive of beaten parents caps at the current
population size with random deletion on overflow, rescaling as the population shrinks. Cauchy on `F`
(resample if `≤ 0`, truncate to 1) keeps a heavy tail; Normal on `CR` (clamp to `[0,1]`) keeps it stable.
The terminal-`CR = 0` rule is the one edge worth dwelling on: when a slot's successful `CR` values are all
zero I lock that slot to `CR = 0`, which in binomial crossover means only the guaranteed `j_rand`
coordinate takes the donor value — a single-coordinate move. On a separable landscape like Rastrigin,
optimizing one axis at a time is precisely the per-coordinate behaviour that let the GA edge DE and that
L-SHADE's low-`CR` regime matched at 7.3; the terminal lock makes that regime an absorbing state on the
rugged separable problems that prefer it. And `N_init` is the harness's `pop_size`, not the canonical
`25·ln(D)·√D`, for the reason the arithmetic above already fixed: the large initial population would
starve the generation count on a search still improving at the wall. The weighted mutation and phase
schedule must do their work *within* the same population budget as every other method. The full module is
in the answer.

Now the bar this must clear, against L-SHADE's real numbers. The three landscapes L-SHADE already won
should hold or improve. Ackley should stay at machine precision — already solved to 9.4e-12 with the curve
flat by gen 258, so the late-run stronger elite pull cannot hurt a basin already collapsed, and I expect
`best_fitness` to remain in the 1e-11 range with convergence still inside the first half. Rosenbrock
should stay in the low tens or better than 13.3, because the late `Fw = 1.2·F_i` sharpens the valley
descent exactly when the population has found the valley — if anything a small improvement. Rastrigin-30d
should stay around or below 7.3, since the terminal-`CR` lock and low-`CR` regime that owned it are
preserved. The decisive test is Rastrigin-100d, the cell L-SHADE lost to the GA by 18.8 (132.5 vs 114):
the weaker early elite pull, the broader early `p`-pool, the high-biased `CR` memory, and the always-on
20% aggressive reservoir should together keep the population exploring the 100-D egg-carton longer before
it commits — directly the correction the still-improving 994.67 convergence and the 18-point seed spread
called for. So I expect Rastrigin-100d `best_fitness` below L-SHADE's 132.5 mean and, the real target,
below the GA's 114. If it does *not* improve, the weighted-mutation thesis is falsified for this budget:
the early-exploration bias could not overcome the forced small `N_init` on the hardest case, and the
honest conclusion would be that on this fixed-budget panel the weighted donor matches rather than beats
L-SHADE. The claim is sharp: same-or-better on the three L-SHADE owns, strictly below 132.5 on
Rastrigin-100d, with its convergence no earlier than L-SHADE's — a search that keeps exploring longer
should, if anything, still be improving near the wall, not plateau early on a worse value.
