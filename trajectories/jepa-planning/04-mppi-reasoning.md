iCEM's numbers came back, and they confirmed the bet *and* exposed the risk I flagged when I built it.
The bet paid off at the short horizon exactly as predicted: CEM's weakest benchmark, horizon 30, went
from 0.70 to 0.80 and the residual `mean_dist` dropped from 9.5 to 5.7 — the colored, far-ranging
excursions plus the across-step warm start did reach the door inside a single `plan()` call, which is
precisely what white CEM could not do cold. At horizon 60 iCEM held respectably, 0.90 with residual
4.1, up from CEM's 0.85 / 6.5. But look at horizon 90: iCEM scored 0.85 — *below* CEM's 0.95 there —
with residual 4.8 against CEM's 3.4. Line the three horizons up against CEM as differences: `+0.10` at
30, `+0.05` at 60, `−0.10` at 90. The improvement shrinks as the horizon grows and then goes *negative*,
which is the regression I named in advance. And it shows in the average: CEM's mean success across the
three horizons is `(0.70+0.85+0.95)/3 = 0.833`, iCEM's is `(0.80+0.90+0.85)/3 = 0.850`. That is a
`0.017` edge — a wash, not the clean dominance I wanted. iCEM traded long-horizon robustness for
short-horizon reach and came out barely ahead on average. I need a rung that keeps the short-horizon
gain *and* recovers the long horizon, and to build it I have to understand precisely why the long
horizon regressed.

There is a third column I have not read yet, and it sharpens the diagnosis rather than softening it:
`mean_steps_to_success`. CEM reached the goal in `23.79 / 23.47 / 46.37` steps at the three horizons;
iCEM took `24.75 / 25.22 / 38.53`. At the short and middle horizons the step counts barely move — `+0.96`
and `+1.75` — but at horizon 90 iCEM's successes arrive *faster*, `38.53` against CEM's `46.37`, nearly
eight steps sooner. That is not in tension with the regression; it is the same mechanism seen from the
other side. The colored draws and the persistent shifted mean make iCEM *commit* to a route quickly, so
on the seeds where it commits to the right side of the door it gets there in fewer steps than white CEM's
meander. But that same fast commitment is exactly what dooms the seeds it loses: once iCEM has spent its
spread on one route, it has no residual exploration to re-cross to the other, so a wrong initial
commitment fails outright inside the 200-step cap rather than wandering to a slow, late success the way
CEM's broader white search sometimes did. Faster successes and more outright failures are two readings of
one behavior — over-commitment — and horizon 90 shows both faces at once: the successes that survive are
quicker (`38.53`), but fewer of them survive (`0.85`). So whatever replaces the hard cut has to keep that
commitment speed while restoring the ability to *abandon* a route the rollouts score badly, which is the
precise property the hard top-k destroys.

The mechanism is the one I predicted: at the long horizon, where CEM's white sampling plus the
re-planning loop had plenty of room to converge on a clean route, iCEM's aggressive elite reuse and
smooth low-frequency colored draws *over-commit*. When the kept elites and the persistent shifted mean
all agree on a route, and that route is the wrong side of the door for a given start/goal, the search
has even less spread to escape than CEM did — the hard top-k elite cut plus the colored over-smoothing
leaves no candidate that disagrees enough to pull the distribution onto the other route. The residual
tells the same story quantitatively: `4.8` at horizon 90 is not a small miss near the `4.5` threshold,
it is a committed route that ended up on the wrong side, so the agent finished a real distance out and
never recovered inside the 200-step episode.

The root is the *hard* elite selection, and I should be precise about why, because that is what the next
move has to replace. Both CEM and iCEM refit the distribution by taking the top-k lowest-cost sequences
and computing their plain mean and variance. That throws away two things. First, it throws away *how
much* better the best elite was than the worst: a sequence that is dramatically better and one that
barely cleared the elite cut count exactly the same in the mean — each of the k elites gets weight
`1/k`, whether its cost was the minimum or sat right at the `10%` boundary. Second, the hard cut is
brittle near the decision boundary between the two routes through the door: a handful of mediocre
elites on the wrong route can drag the mean and, worse, keep the variance from shrinking onto the right
route — or, in iCEM's over-committed case, the elites all land on one route and the variance collapses
there with no soft pressure from the near-elite samples on the other route. Let me make the drag concrete, because it is the exact geometry of the horizon-90 miss. Suppose an
elite set of 20 splits near the door decision boundary: 14 sequences on the good route (say a first-step
heading of `(+0.6, 0)`) and 6 on the wrong-side route (`(−0.6, 0)`). CEM's uniform elite mean is
`(14·(+0.6) + 6·(−0.6))/20 = (+0.24, 0)` — pulled `30%` of the way toward the wrong route, a heading
that points at the wall *between* the two doors rather than at either. The hard mean is dragged by the
wrong-route elites in exact proportion to their *count*, `6/20`, with no regard to the fact that they
scored worse. A soft, cost-weighted mean is dragged instead by their *weight*, which is their count
discounted by how much worse they rolled out — so if the wrong-route route really is worse, its `30%`
population share shrinks to a much smaller weight share and the mean stays on the good route instead of
splitting the difference into the wall. That is the difference I need: an update that is *soft*, where
every sampled sequence contributes weighted by how good it is, so a markedly-better rollout pulls the
mean and tightens the variance far more than a marginal one, and no information is discarded at a hard
threshold. And I want it to be the *principled* soft update, not an ad-hoc reweighting — the
same demand I made when I moved from random shooting to CEM.

So go back to the object and ask what the right weighting is. I have sampled action sequences and their
costs, and I want the new control distribution that best concentrates on low cost. Frame it as the
free-energy / partition-function object: assign each trajectory `τ` a Boltzmann weight that is
exponential in its negative cost, `w(τ) ∝ exp(−S(τ)/λ)`, where `S(τ)` is the cost of rolling that
sequence through the model and `λ` is a temperature. Low-cost trajectories get large weight, high-cost
ones are suppressed, and `λ` controls how sharply. There is a clean way to see this is not arbitrary:
define the free energy `F = log E_P[exp(−S/λ)]` over control sequences drawn from a base distribution.
By Jensen's inequality on the concave log, `−λF ≤ E_Q[S] + λ·D_KL(Q‖P)` for any sampling distribution
`Q` — the right side is "expected cost plus a KL penalty pulling `Q` toward the base," which is the
control objective, and the bound is tight exactly when `Q` is the base tilted by `exp(−S/λ)`. I have
been burned by Jensen-tightness claims before, so let me check that the tilted distribution really is the
minimizer rather than trust it. On a six-trajectory toy with a random base `p`, random costs in `[0,4]`,
and `λ = 0.7`, computing `−λF` directly from `F = log Σ_τ p(τ)exp(−S(τ)/λ)` gives `1.12233`; forming
`q*` by tilting `p` and renormalizing, then evaluating the right-hand side `E_{q*}[S] + λ·D_KL(q*‖p)`,
gives `1.12233` — equal to five figures. Evaluating the same right-hand side at the base `p` itself gives
`1.78`, and at three other random distributions gives `1.62`, `1.71`, `1.82` — every one strictly above
the floor, and only `q*` touches it. So the Gibbs tilt is genuinely the optimum, not a plausible
candidate, and the exponential weighting is the exact minimizer of the control objective, not a
heuristic reweighting. So the
optimal sampling distribution is the Gibbs distribution over trajectories, weighted by `exp(−cost/λ)`,
and the optimal control is the weighted average of the sampled controls under those weights —
`u_i ← Σ_k w_k · a_{i,k}` with `w_k = exp(−S(τ_k)/λ) / Σ_j exp(−S(τ_j)/λ)`. This is the soft,
information-preserving analogue of CEM's hard elite mean: each sequence votes with a weight that decays
smoothly in its cost, so the dramatically-better wrong-route-avoiding sequence dominates without a hard
line, and a cluster of mediocre wrong-route sequences gets exponentially suppressed rather than averaged
in at full weight — which is exactly the long-horizon failure iCEM showed.

Let me actually watch the temperature do its job on a small example, because the two limits are the
whole reason to prefer this over a hard cut and I would rather compute them than assert them. Take three
elites with costs `[10, 11.5, 14]` and weight `w_k = exp(temperature·(min_cost − cost_k))` with
`temperature = 1/λ`. At an intermediate `temperature = 0.8` the normalized weights come out `[0.745,
0.224, 0.030]` — graded by how much better each sequence is, the best getting three-quarters of the mass
and the worst almost none, but none discarded. Now push the temperature to `10⁶` (`λ → 0`): the weights
become `[1, 0, 0]`, all mass on the single lowest-cost sequence — a hard argmax. Push it to `10⁻⁹`
(`λ → ∞`): the weights become `[1/3, 1/3, 1/3]`, a plain unweighted average that ignores cost entirely —
which is exactly CEM's hard-mean behavior *if* CEM kept all samples, or its uniform-over-elites behavior
on the elite set. So CEM's uniform elite mean is literally the `λ → ∞` corner of this family, and the
soft update interpolates between that and the greedy argmax. The temperature is the knob that says how
much to trust the cost magnitudes, and the middle of the range is where a clearly better route dominates
while near-best sequences still hold the variance open — the property iCEM's hard reuse destroyed.

Now I have to be honest about the specific form the task's harness gives me, because the full
path-integral derivation assumes a control-affine system with a particular noise-cost coupling and
yields a Girsanov likelihood-ratio correction for shifting both the mean and the covariance of the
sampling distribution. I do not have a control-affine analytic model here — I have a learned JEPA world
model I can only roll forward and a black-box cost — so I am not going to import the covariance-scaling
Girsanov machinery. What I land instead is the discrete, sampling-based core of the method as it is
actually used on top of learned models: keep the CEM skeleton — sample a batch around the current mean,
roll through the model, take the lowest-cost elites — but replace the *update* with the exponential
cost-weighting. Concretely, after scoring I still select the top elites (here `num_elites = max(20,
num_samples//10) = 20`, the same top 10% CEM used), then compute weights over those elites, `score_k =
exp(temperature · (min_cost − cost_k))`, normalized to sum to 1, and set the new mean to the weighted
average of the elite sequences and the new std to the weighted standard deviation. Restricting to the
top-k before soft-weighting is deliberate: with a finite population the far-out high-cost samples still
carry a tiny nonzero weight, which contributes noise to the variance fit especially; keeping the elites
cleans the fit while the exponential weighting preserves the graded "how much better" information among
the survivors. So this is not CEM with a different elite count — it is the same elite truncation with a
soft, cost-weighted refit inside the elite set instead of a uniform one.

I should check that the weight I actually compute is numerically safe and unchanged by the shift I put
in it, because subtracting `min_cost` inside the exponential is doing two jobs and I want to be sure it
does not quietly do a third. It is there for numerical stabilization — it shifts the largest exponent to
exactly zero (the best elite gets `exp(0) = 1`) so nothing overflows, and every other exponent is
non-positive so nothing blows up. And because the weights are normalized, the shift should cancel and
leave the Boltzmann weighting unchanged. Let me confirm it really is invariant rather than approximately
so: take the same three costs `[10, 11.5, 14]` at `temperature = 0.8`, compute the weighted mean and std,
then add `+1000` to *every* cost and recompute. The weights come out `[0.745, 0.224, 0.030]` both times,
and the weighted mean and std are identical to floating-point roundoff — the `+1000` shift changed
nothing, as it must, because it is multiplying numerator and denominator by the same `exp(1000/λ)`. So
the stabilization is a true no-op on the answer and a genuine guard on the arithmetic; I compute
`exp(temperature·(min_cost − cost))`, the exponentiated cost gap from the best sample. The temperature
here is small, `0.005`, which keeps the weighting gentle relative to the cost scale — sharp enough that
a clearly better rollout dominates, soft enough that the near-best sequences still contribute and the
variance does not collapse onto one route the way iCEM's hard reuse did. A small temperature is a *large*
`λ`, i.e. deliberately toward the robust, broad-averaging end of the family, which is the right choice
when the model rollouts themselves are noisy and I do not want a single lucky sample to dictate the plan.

I keep the sampling *white* here — plain `torch.randn` perturbations, not colored noise. That is a
deliberate choice given what iCEM's numbers showed: the colored over-smoothing is part of what made iCEM
over-commit at the long horizon, and the soft weighting is supposed to do the reach-and-route work
through the *update*, not through a far-ranging noise prior. White sampling with a soft cost-weighted
mean lets the distribution be pulled toward whichever route the rollouts actually score well, without a
low-frequency prior baking in a committed direction before the cost is read. I also widen the initial
spread to `max_std = 2.0` (above CEM's 1.5), because the soft update is more forgiving of a wide start —
it will not be dragged around by a few wide outliers the way a hard elite mean can be, since those
outliers carry exponentially small weight — so I can afford to explore both routes more aggressively at
the first iteration and let the Boltzmann weighting concentrate from there. At `σ = 2.0` the per-step
action has RMS norm `√2·2.0 ≈ 2.83`, already above the `max_norm = 2.45` cap, so on the order of half
the sampled actions overshoot the feasible ball. Unlike CEM and iCEM, this rung does *not* re-project
them to the boundary — it follows the upstream MPPI convention of leaving the samples unclipped — and
that is tolerable precisely because of the soft weighting: an over-long action that the model rolls out
to a bad latent scores high and is exponentially suppressed in the weights, so the wide, occasionally
infeasible tail costs me almost nothing while the wide *center* buys aggressive early exploration of
both routes. The exponential is doing the feasibility triage that CEM did with a hard clip.

The std refit deserves the same attention, because it is where the soft update earns its robustness over
iCEM's collapse. CEM and iCEM set the new spread to the plain (or momentum-smoothed) elite std, which
contracts hard on the coordinates the top-k agree about — the anisotropic collapse I traced two rungs
back, and the very thing that let iCEM lock onto a wrong route with no spread to escape. Here the new
std is the *weighted* spread of the elites, `√(Σ_k score_k·(a_k − mean)²)`, so it contracts only in
proportion to how much the *low-cost* elites agree, not the whole top-k. When both routes are still live
— the good and the wrong-side elites both present with comparable weights — the weighted spread stays
wide, because the weighted samples are genuinely spread across the two routes, and the search keeps both
open for another iteration. It tightens only once the weights have concentrated on one route, i.e. once
the cost actually distinguishes them. That is a strictly gentler collapse than the hard elite std, and
it is the mechanism by which the soft update avoids iCEM's premature commitment: the variance is allowed
to shrink only after the evidence, not on the first iteration's lucky cluster.

There is one more place the soft framing changes the *output*, and it matters for the maze. CEM and
iCEM return the distribution's center (the mean, or iCEM's best-so-far fallback). But the mean of a
soft-weighted distribution, in a bimodal cost surface with two routes, can land *between* the two
modes — in the wall — which is worse than either route: averaging a plan that goes left of the door with
a plan that goes right of it gives a plan that walks into the wall between them. So instead of executing
the mean, I select the executed sequence by *sampling* from the final elite weights: draw one elite with
probability proportional to its score, and return that actual evaluated sequence. This is the
path-integral controller's stochastic action selection, and it is the right thing in a multi-modal
surface — it commits to one real, rolled-out route rather than averaging two incompatible routes into an
infeasible compromise. The chosen sequence is one I actually scored, so it is feasible-as-rolled and
known-good, not an untested centroid, and because the draw is weighted by the Boltzmann score it will
almost always land on the dominant low-cost route while retaining a small chance of the other — which is
itself a hedge against committing to the wrong side. The stochastic draw also does something useful across the whole receding-horizon loop, not just within
one call. With weights like the `[0.745, 0.224, 0.030]` example, a single `plan()` returns the dominant
route about three-quarters of the time and one of the alternatives the rest — so over the many control
steps of an episode the planner usually commits to the best route but occasionally executes a first
action from a different one. If the dominant route is genuinely right, this costs almost nothing (the
occasional off-route first action is corrected on the very next re-plan). But if the dominant route is
*systematically* wrong — heading into the wall as the agent approaches — its rollout cost climbs step by
step, the weights shift toward the alternative, and the loop reallocates its commitment without any
explicit escape mechanism. The stochastic selection is thus a built-in, cost-driven hedge against the
exact wrong-route lock-in that a deterministic argmin-elite (or iCEM's persistent shifted mean) would
ride all the way into the wall. Nothing persists across env steps in this rung —
each `plan()` re-optimizes from the zero mean and wide std — so `t0` is irrelevant again, unlike iCEM
where it was load-bearing; I am deliberately dropping iCEM's across-step machinery along with its
colored prior, because the soft update is meant to earn its robustness within a single call. And the
whole loop runs under `no_grad` because, like every rung before it, MPPI reads only function values of
the cost and never touches the gradient the differentiable model could hand it. That gradient is still
sitting unused; it is the lever a method beyond this ladder would pull.

So the falsifiable expectations against iCEM's numbers. The soft cost-weighting is aimed squarely at
iCEM's long-horizon regression, so the cleanest prediction is at horizon 90: where iCEM slipped to 0.85
by over-committing to one route, the Boltzmann weighting should recover the long-horizon performance CEM
had — I expect horizon 90 back up to `~0.95` with the residual `mean_dist` back down toward CEM's 3.4,
because the soft update plus stochastic-elite action selection stops a few wrong-route elites from
collapsing the search and stops the planner from executing an in-the-wall mean. At horizon 60 I expect a
clear lift over iCEM's 0.90 — the soft update is most valuable in the middle regime where both routes are
live and a hard cut is brittle, so this is where I would most expect to see it clear 0.90. At horizon 30
the open question is whether dropping iCEM's colored noise costs me the short-horizon reach: I expect to
roughly *hold* iCEM's 0.80 rather than beat it, because white sampling reaches the door less reliably
than colored excursions inside one short call, and if horizon 30 instead *drops* below 0.75, that is the
signal that the short horizon genuinely needs the colored prior and the right method would be MPPI's
soft update *over* colored noise, combining this rung's robustness with iCEM's reach. But the headline
bet is that the soft, information-preserving update gives the first planner that is strong at *all three*
horizons at once — short reach near iCEM's, and long-horizon robustness back at CEM's level — so that the
mean across horizons finally clears the `0.85` wash both hard-cut baselines were stuck at, rather than
trading one horizon for another. The full scaffold module — the literal `CustomPlanner` fill for MPPI —
is in the answer.
