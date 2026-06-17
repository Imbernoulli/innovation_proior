The L-SHADE numbers came back and they confirm the rung's whole thesis: one adaptive algorithm *is*
right across landscapes where the two fixed methods each failed somewhere. Ackley fell to 4.7e-11 at
seed 42 (mean 9.4e-12) — solved, near machine precision, the broad smooth basin that DE's noisy 10.0
mean and the GA's 1.05 both left on the table; the adaptive `F` plus the difference vector took it all
the way down, and it converged early (gen 269 at seed 42, mean 258) rather than burning the budget.
Rosenbrock fell to 15.0 (mean 13.3) — beating both DE's seed-42 32.2 and crushing the GA's mean 138,
because the adaptive `CR` recovered the anisotropic coordinated moves the GA's axis-aligned operators
could not make. Rastrigin-30d landed at 7.9 (mean 7.3), edging the GA's 8.0 — competitive, the low-`CR`
regime plus linear-reduction refinement matching the GA's per-coordinate win. So three of four
landscapes are clearly the best of the ladder. But look at the one I flagged as uncertain:
Rastrigin-100d came back at 120.6 at seed 42 (mean 132.5) — and that is *worse* than the GA's mean of
114. The deliberate `N_init=pop_size` choice (forced by this fixed-small-budget harness, where the
canonical large `N_init` would starve the generation count) traded exploration for refinement, and on
the hardest multimodal case at 100D that trade did not pay: the population reduced too aggressively
before it had explored enough of the 100-dimensional egg-carton to find the global basin. L-SHADE is
the strongest rung, but its weak spot is exactly where I predicted — high-dimensional multimodal — and
the failure mode is the exploration/exploitation balance going to exploitation too early.

Let me make that diagnosis precise before I act on it, because the cure depends on which of two things
went wrong. One possibility is that the *parameters* mis-adapted: the success-history memory drove `F`
small (mutation went quiet) or `CR` to a value that did not suit the separable 100-D landscape. The
other is that the *population schedule* committed too fast: linear reduction shed members while the
search was still scattered across the egg-carton, so the survivors clustered around a non-global basin
and the shrunken population could no longer climb out. The convergence column is the tell — at 100D
L-SHADE was still improving at generation ~995 of 1000 (it never plateaued), so it is not that the
search converged to a wrong answer and sat there; it is that the search was *still moving but too
slowly*, having spent its early generations on a population too small (`N_init=pop_size` rather than a
large multiple of `D`) to cover 100 dimensions. Both readings point the same way: I need more effective
exploration in the first half of the run *without* enlarging the population budget, because the budget
is fixed by the harness. That rules out the obvious fix (bigger `N_init`) and forces the subtler one —
extract more exploration from the same population by changing *how* each individual moves early.

That diagnosis points straight at the one place L-SHADE's mutation is structurally crude, and fixing it
is the natural next move. Look at the donor: `v_i = x_i + F_i*(x_pbest - x_i) + F_i*(x_r1 - x_r2)`. The
*same* `F_i` multiplies two structurally different terms. The first, `x_pbest - x_i`, is a directed
pull toward the elite — pure exploitation, dragging the individual toward a good solution already
found. The second, `x_r1 - x_r2`, is the self-scaling random difference — the exploration, the
perturbation carrying the population's scatter. Tying both to one `F_i` forces the elite-pull strength
and the random-perturbation strength to move in lockstep. But the Rastrigin-100d failure is precisely a
balance problem: the search committed to the elite (and shrank the population toward it) before it had
perturbed enough to find the right basin. If I could make the elite pull *weaker than* the perturbation
early — explore without committing — and *stronger than* it late — refine hard once the basin is found —
I would directly attack the failure I just measured. A single `F_i` cannot express that phase
dependence. So the move is to give the elite-pull term its own factor: a weighted `Fw` on
`(x_pbest - x_i)`, with `F_i` left on the random difference. Early, `Fw < F_i` (perturb more than pull);
late, `Fw > F_i` (pull more than perturb). A three-step schedule on the budget fraction realizes it:
`Fw = 0.7*F_i` for the first 0.2 of the run, `0.8*F_i` for the first 0.4, `1.2*F_i` thereafter. The
donor becomes `v_i = x_i + Fw*(x_pbest - x_i) + F_i*(x_r1 - x_r2)` — current-to-pbest *weighted*. This
is the heart of the improvement: it decouples the two roles `F` was forced to play and slides their
balance from exploration-heavy early to exploitation-heavy late, which is exactly the correction
Rastrigin-100d needed.

Several smaller refinements reinforce the same exploration-early/exploitation-late arc, and each saves
budget L-SHADE wasted. First, the memory start. L-SHADE initializes every slot neutral at `0.5`, so the
early generations re-discover on every problem that a high `CR` helps while exploring. I already believe
high `CR` is right early (coordinated multi-coordinate moves on a spread population), so I initialize
the `CR` slots high, at `0.8`, and leave `F` at `0.5` — a free head start that the adaptation will
override if a near-separable problem prefers low `CR`. Second, a permanent aggressive reservoir:
*freeze* one memory slot (the last) at `M_F = M_CR = 0.9` and never update it. Because each individual
picks its memory index uniformly, about `1/H` of the population always samples around `(0.9, 0.9)`
regardless of what the rest of the memory decides — cheap insurance against the whole population going
quiet, which is one description of the Rastrigin-100d collapse. Third, phase rails: for the first
quarter of the budget floor `CR` at `0.7`; for the first half, floor at `0.6`; for the first 0.6, cap
`F` at `0.7`. These are not adapted — they are hard rails keeping early sampling in the
explore-coordinated regime, relaxing as the run proceeds so the adaptation takes over for refinement.
Fourth, the memory update: instead of overwriting a slot with one noisy generation's summary, *blend*
it, `M[k] <- (mean_WL(S) + M[k]_old)/2`, so a single generation moves a slot at most halfway to its
summary and the slot retains its history — a stabilizer on the adaptation trajectory.

The pbest greediness `p` gets the same phase treatment. L-SHADE samples `p_i` per individual from a
fixed interval; here I want `p` to *decrease* over the run, from a broad elite pool early to a sharp one
late: `p = p_max - (p_max - p_min)*nfes/max_nfes`, with `p_max = 0.25`, `p_min = 0.125`. Early, the
elite guide is drawn from the top quarter (diverse, several good basins represented); late, from the
top eighth (sharp convergence). This is the same exploration-to-exploitation arc as the weighted
mutation, applied to where the elite guide comes from — and it directly widens early exploration on the
high-D multimodal case that failed.

The constants and edge cases carry over from the adaptive-DE substrate. Memory size `H = 5`, with the
last slot frozen, so four slots adapt and the round-robin counter cycles over those four. `N_min = 4`,
pinned because current-to-pbest/1 needs four distinct individuals. The archive caps at the current
population size, random deletion on overflow, rescaled as the population shrinks. Cauchy on `F`
(resample if `<= 0`, truncate to 1) for diversity in the mutation strength; Normal on `CR` (clamp to
`[0,1]`) for stability; the terminal-`CR = 0` rule for slots whose successful `CR` are all zero, locking
the one-coordinate regime that helps on rugged separable landscapes. And — the one place this task's
implementation must depart from the canonical constant, exactly as the L-SHADE rung did — `N_init` is
the harness's `pop_size`, not the canonical `round(25*log(D)*sqrt(D))`, because this fixed-small-budget
harness would have its generation count starved by a large initial population. The weighted mutation
and the phase schedule have to do their work *within* the same population budget as every other rung,
not by enlarging it. The full scaffold module — the literal fill of `run_evolution`, with the three
operator stubs left as no-ops — is in the answer.

Now the bar this finale must clear, stated against L-SHADE's real numbers, with no result invented. The
three landscapes L-SHADE already won should hold or improve: Ackley should stay at machine precision
(the weighted pull cannot hurt a problem already solved to 1e-11); Rosenbrock should stay in the low
tens or better (the late-run stronger elite pull should sharpen the valley descent past L-SHADE's
13.3 mean); Rastrigin-30d should stay around or below L-SHADE's 7.3 mean. The decisive test is
Rastrigin-100d, the one L-SHADE lost to the GA: the weaker early elite pull plus the broader early
`p`-pool plus the high-biased `CR` memory should keep the population exploring the 100-D egg-carton
longer before it commits, so I expect Rastrigin-100d to fall *below* L-SHADE's 132.5 mean and, the real
target, below the GA's 114 — recovering the ladder's lead on the one benchmark where the strongest
baseline slipped. If Rastrigin-100d does *not* improve, the weighted mutation thesis is falsified for
this harness's budget: it would mean the early-exploration bias cannot overcome the forced small
`N_init` on the hardest case, and the honest conclusion would be that on *this* fixed-budget panel jSO
matches rather than beats L-SHADE. The falsifiable claim is sharp: same-or-better on the three L-SHADE
already owns, and strictly better than 132.5 on Rastrigin-100d.
