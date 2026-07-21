iCEM's numbers came back, and they confirmed the bet *and* exposed the risk I flagged when I built it.
The bet paid off at the short horizon exactly as predicted: CEM's weakest benchmark, horizon 30, went
from 0.70 to 0.80 and the residual `mean_dist` dropped from 9.5 to 5.7 — the colored, far-ranging
excursions plus the across-step warm start did reach the door inside a single `plan()` call, which is
precisely what white CEM could not do cold. At horizon 60 iCEM held respectably, 0.90 with residual 4.1,
up from CEM's 0.85 / 6.5. But at horizon 90 iCEM scored 0.85 — *below* CEM's 0.95 there — with residual
4.8 against CEM's 3.4. Line the three horizons up against CEM as differences: `+0.10` at 30, `+0.05` at
60, `−0.10` at 90. The improvement shrinks as the horizon grows and then goes *negative*, the regression I
named in advance. It shows in the average too: CEM's mean success across horizons is `0.833`, iCEM's
`0.850` — a `0.017` edge, a wash, not the clean dominance I wanted. iCEM traded long-horizon robustness for
short-horizon reach.

The third column sharpens the diagnosis rather than softening it. CEM reached the goal in `23.79 / 23.47
/ 46.37` steps; iCEM took `24.75 / 25.22 / 38.53`. At the short and middle horizons the step counts
barely move, but at horizon 90 iCEM's successes arrive nearly eight steps *sooner*. That is not in
tension with the regression — it is the same mechanism from the other side. The colored draws and the
persistent shifted mean make iCEM *commit* quickly, so on the seeds where it commits to the right side of
the door it gets there faster than white CEM's meander; but that same fast commitment dooms the seeds it
loses — once iCEM has spent its spread on one route it has no residual exploration to re-cross, so a wrong
initial commitment fails outright inside the 200-step cap rather than wandering to a slow late success the
way CEM's broader white search sometimes did. Faster successes and more outright failures are two readings
of one behavior, over-commitment, and horizon 90 shows both at once: the successes that survive are
quicker (`38.53`), but fewer survive (`0.85`). So whatever replaces the hard cut has to keep the
commitment speed while restoring the ability to *abandon* a route the rollouts score badly — the precise
property the hard top-k destroys, when the kept elites and the persistent shifted mean all agree on a
wrong-side route and the search has less spread than CEM to escape.

The root is the *hard* elite selection. Both CEM and iCEM refit by taking the top-k lowest-cost sequences
and computing their plain mean and variance, which throws away two things. First, *how much* better the
best elite was than the worst: a dramatically-better sequence and one that barely cleared the cut count
exactly the same, each getting weight `1/k`. Second, the hard cut is brittle near the decision boundary
between the two routes. Make the drag concrete, because it is the exact geometry of the horizon-90 miss.
Suppose an elite set of 20 splits near the boundary: 14 on the good route (first-step heading `(+0.6,
0)`) and 6 on the wrong-side route (`(−0.6, 0)`). CEM's uniform elite mean is `(14·(+0.6) + 6·(−0.6))/20
= (+0.24, 0)` — pulled 30% toward the wrong route, a heading that points at the wall *between* the doors.
The hard mean is dragged by the wrong-route elites in proportion to their *count*, `6/20`, with no regard
to their scoring worse. A soft, cost-weighted mean is dragged instead by their *weight* — count
discounted by how much worse they rolled out — so if that route really is worse its 30% population share
shrinks to a much smaller weight share and the mean stays on the good route. That is the difference I
need: an update where every sampled sequence contributes weighted by how good it is, so a markedly-better
rollout pulls the mean and tightens the variance far more than a marginal one, and no information is
discarded at a hard threshold. And I want the *principled* soft update, not an ad-hoc reweighting — the
same demand I made moving from random shooting to CEM.

So go back to the object. I have sampled action sequences and their costs and want the new control
distribution that best concentrates on low cost. Frame it as the free-energy object: assign each
trajectory `τ` a Boltzmann weight exponential in its negative cost, `w(τ) ∝ exp(−S(τ)/λ)`, with `S(τ)`
the rollout cost and `λ` a temperature. This is not arbitrary: define the free energy `F = log
E_P[exp(−S/λ)]` over control sequences from a base distribution; by Jensen on the concave log, `−λF ≤
E_Q[S] + λ·D_KL(Q‖P)` for any sampling distribution `Q`, and the right side is exactly the control
objective — expected cost plus a KL penalty pulling `Q` toward the base — with the bound tight precisely
when `Q` is the base tilted by `exp(−S/λ)`. So the optimal sampling distribution is the Gibbs
distribution over trajectories, and the optimal control is the weighted average of the sampled controls,
`u_i ← Σ_k w_k · a_{i,k}` with `w_k = exp(−S(τ_k)/λ) / Σ_j exp(−S(τ_j)/λ)`. This is the soft,
information-preserving analogue of CEM's hard elite mean: each sequence votes with a weight that decays
smoothly in its cost, so the dramatically-better wrong-route-avoiding sequence dominates without a hard
line, and a cluster of mediocre wrong-route sequences gets exponentially suppressed rather than averaged
in at full weight — exactly the long-horizon failure iCEM showed. The temperature sets the two limits:
`λ → 0` puts all mass on the single lowest-cost sequence (a hard argmax), `λ → ∞` weights everything
equally (plain averaging). CEM's uniform elite mean is literally the `λ → ∞` corner of this family, and
the soft update interpolates between that and the greedy argmax — the middle is where a clearly better
route dominates while near-best sequences still hold the variance open, the property iCEM's hard reuse
destroyed.

Now the honest form the harness gives me. The full path-integral derivation assumes a control-affine
system with a particular noise-cost coupling and yields a Girsanov likelihood-ratio correction for
shifting both the mean and the covariance. I do not have a control-affine analytic model — I have a
learned JEPA world model I can only roll forward and a black-box cost — so I am not importing the
covariance-scaling Girsanov machinery. What I land is the discrete, sampling-based core as it is actually
used on top of learned models: keep the CEM skeleton — sample a batch around the current mean, roll
through the model, take the lowest-cost elites — but replace the *update* with the exponential
cost-weighting. After scoring I still select the top elites (here `num_elites = max(20, num_samples//10)
= 20`, the same top 10% CEM used), then compute weights over those elites, `score_k = exp(temperature ·
(min_cost − cost_k))`, normalized to sum to 1, and set the new mean to the weighted average of the elite
sequences and the new std to the weighted standard deviation. Subtracting `min_cost` inside the
exponential is the standard numerical stabilization — it shifts the largest exponent to zero so nothing
overflows, and because the weights are normalized the shift cancels and leaves the Boltzmann weighting
unchanged. Restricting to the top-k before soft-weighting is deliberate: with a finite population the
far-out high-cost samples still carry a tiny nonzero weight that contributes noise to the variance fit,
so keeping the elites cleans the fit while the exponential weighting preserves the graded "how much
better" information among the survivors. So this is not CEM with a different elite count — it is the same
elite truncation with a soft, cost-weighted refit inside the elite set. The temperature is small, `0.005`,
keeping the weighting gentle relative to the cost scale — sharp enough that a clearly better rollout
dominates, soft enough that the near-best sequences still contribute and the variance does not collapse
onto one route the way iCEM's hard reuse did. A small temperature is a *large* `λ`, deliberately toward
the robust, broad-averaging end, the right choice when the model rollouts themselves are noisy and I do
not want a single lucky sample to dictate the plan.

I keep the sampling *white* here — plain `torch.randn` perturbations, not colored noise. That is
deliberate given what iCEM's numbers showed: the colored over-smoothing is part of what made iCEM
over-commit at the long horizon, and the soft weighting is supposed to do the reach-and-route work
through the *update*, not through a far-ranging noise prior. White sampling with a soft cost-weighted mean
lets the distribution be pulled toward whichever route the rollouts actually score well, without a
low-frequency prior baking in a committed direction before the cost is read. I also widen the initial
spread to `max_std = 2.0` (above CEM's 1.5), because the soft update is more forgiving of a wide start —
it will not be dragged by a few wide outliers the way a hard elite mean can, since those outliers carry
exponentially small weight — so I can explore both routes more aggressively at the first iteration and let
the Boltzmann weighting concentrate from there. At `σ = 2.0` the per-step action has RMS norm `√2·2.0 ≈
2.83`, above the `max_norm = 2.45` cap, so on the order of half the sampled actions overshoot the feasible
ball. Unlike CEM and iCEM, this rung does *not* re-project them — it follows the upstream MPPI convention
of leaving the samples unclipped — and that is tolerable precisely because of the soft weighting: an
over-long action the model rolls out to a bad latent scores high and is exponentially suppressed, so the
wide, occasionally infeasible tail costs almost nothing while the wide *center* buys aggressive early
exploration. The exponential does the feasibility triage that CEM did with a hard clip.

The std refit is where the soft update earns its robustness over iCEM's collapse. CEM and iCEM set the new
spread to the plain (or momentum-smoothed) elite std, which contracts hard on the coordinates the top-k
agree about — the anisotropic collapse that let iCEM lock onto a wrong route with no spread to escape.
Here the new std is the *weighted* spread of the elites, `√(Σ_k score_k·(a_k − mean)²)`, so it contracts
only in proportion to how much the *low-cost* elites agree. When both routes are still live — good and
wrong-side elites both present with comparable weights — the weighted spread stays wide, and the search
keeps both open for another iteration; it tightens only once the weights have concentrated on one route,
i.e. once the cost actually distinguishes them. That is a strictly gentler collapse than the hard elite
std, and it is the mechanism by which the soft update avoids iCEM's premature commitment: the variance
shrinks only after the evidence, not on the first iteration's lucky cluster.

One more place the soft framing changes the *output*, and it matters for the maze. CEM and iCEM return
the distribution's center. But the mean of a soft-weighted distribution, in a bimodal cost surface with
two routes, can land *between* the modes — in the wall — which is worse than either route: averaging a
plan that goes left of the door with one that goes right gives a plan that walks into the wall between
them. So instead of executing the mean, I select the executed sequence by *sampling* from the final elite
weights: draw one elite with probability proportional to its score and return that actual evaluated
sequence. This is the path-integral controller's stochastic action selection, the right thing in a
multi-modal surface — it commits to one real, rolled-out route rather than averaging two incompatible ones
into an infeasible compromise. The chosen sequence is one I actually scored, so it is feasible-as-rolled
and known-good, and because the draw is weighted by the Boltzmann score it almost always lands on the
dominant low-cost route while retaining a small chance of the other. That small chance is itself a hedge
across the whole receding-horizon loop: if the dominant route is genuinely right, the occasional off-route
first action is corrected on the very next re-plan and costs almost nothing; but if the dominant route is
*systematically* wrong — heading into the wall as the agent approaches — its rollout cost climbs step by
step, the weights shift toward the alternative, and the loop reallocates its commitment without any
explicit escape mechanism. It is a built-in, cost-driven hedge against the exact wrong-route lock-in that
a deterministic argmin-elite (or iCEM's persistent shifted mean) would ride all the way into the wall.
Nothing persists across env steps in this rung — each call re-optimizes from the zero mean and wide std —
so `t0` is irrelevant again, unlike iCEM where it was load-bearing; I am deliberately dropping iCEM's
across-step machinery along with its colored prior, because the soft update is meant to earn its
robustness within a single call. The whole loop runs under `no_grad`, reading only function values; the
differentiable model's gradient is still sitting unused, the lever a method beyond this ladder would pull.

So the falsifiable expectations against iCEM's numbers. The soft cost-weighting is aimed squarely at
iCEM's long-horizon regression, so the cleanest prediction is at horizon 90: where iCEM slipped to 0.85 by
over-committing, the Boltzmann weighting should recover the long-horizon performance CEM had — back toward
0.95 with the residual back down toward CEM's 3.4 — because the soft update plus stochastic-elite selection
stops a few wrong-route elites from collapsing the search and stops the planner from executing an
in-the-wall mean. At horizon 60 I expect a clear lift over iCEM's 0.90, the middle regime where both
routes are live and a hard cut is most brittle. At horizon 30 the open question is whether dropping iCEM's
colored noise costs me the short-horizon reach: I expect to roughly *hold* iCEM's 0.80 rather than beat
it, because white sampling reaches the door less reliably than colored excursions inside one short call —
and if horizon 30 instead drops below 0.75, that is the signal that the short horizon genuinely needs the
colored prior and the right method would be this soft update *over* colored noise. The headline bet is
that the soft, information-preserving update gives the first planner strong at *all three* horizons at
once — short reach near iCEM's, long-horizon robustness back at CEM's level — so the mean across horizons
finally clears the `0.85` wash both hard-cut baselines were stuck at, rather than trading one horizon for
another. The full scaffold module for MPPI is in the answer.
