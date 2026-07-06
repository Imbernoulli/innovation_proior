KL-UCB's numbers split exactly along the fault line I predicted, and reading the three columns against
UCB1's tells me the next move is forced. Take them one difference at a time so I don't fool myself. On
the **non-stationary** setting KL-UCB dropped to 0.0349 (per seed 0.0354 / 0.0408 / 0.0284) from UCB1's
0.0829 — that is a fall of 0.048 in normalized regret, a ratio of 0.42, less than half the cumulative
shortfall UCB1 was paying, and the winner of that column outright. The mechanism I bet on paid off: the
tighter KL index relearns each of the four changepoints faster on full history even with no forgetting at
all, so "sharper geometry, not windowing" was the right diagnosis there. On the **contextual** setting it
was 0.1785 (0.185 / 0.163 / 0.188) against UCB1's 0.1793 — a difference of 0.0008, which is a twentieth of
the seed-to-seed spread within either rung, i.e. statistically nothing. That flatness is itself a result:
two different index geometries produce the same contextual number because neither one reads the context,
so both are estimating the same collapsed marginal means and neither can converge to a best arm that does
not exist as a fixed object. The contextual column has now not moved across two rungs, and its
immovability is the strongest signal I have about what is actually wrong there.

But on the **stochastic MAB** KL-UCB got *worse*: 0.0612 (0.0602 / 0.0622 / 0.0612) versus UCB1's 0.0378 —
an increase of 0.0234, a ratio of 1.62, more than half again the regret in the home regime. That is the
falsifiable risk I flagged materializing, and the per-seed structure confirms the reading: KL-UCB's three
seeds sit in a band of width 0.002, extraordinarily tight, the signature of a rule whose *early* behaviour
is dominated by a deterministic exploration schedule that fires the same way regardless of seed. With the
practical `c = 1` log-budget, KL-UCB explores more conservatively per arm early, and at `T = 10000` with
arm means clustered between 0.10 and 0.80 the asymptotic constant advantage never arrives — the heavier
early exploration costs more than the tighter constant saves. So I now hold a rule that wins one world,
loses another relative to where I started, and has left the contextual world untouched across two full
rungs. Two conclusions are forced out of this arithmetic. First, an index policy — Hoeffding or KL — *cannot*
use the context, so the 0.18 will sit there until I change *what is modeled*, not how tightly I bound it.
Second, I want an exploration mechanism that is not tied to a single deviation constant that has to be
tuned against the horizon, because that constant is exactly what traded me the stochastic column for the
non-stationary one. Both conclusions point the same way: away from deterministic confidence bonuses and
toward a Bayesian, randomized rule.

Before committing, let me actually walk the design space I am standing in, because there are three or four
live options and I want to reject the tempting ones with arithmetic rather than taste. The first is to keep
the index and just retune the budget — run KL-UCB with a smaller `c`, say `c = 0.5`, to cut the early
over-exploration on the stochastic world. But this is a zero-sum knob against the two columns it touches:
shrinking `c` narrows the confidence radius everywhere, which would pull the stochastic column back down
toward UCB1's 0.0378 but would *also* narrow the radius on the non-stationary world, where a slightly wider
radius is precisely what let KL-UCB keep sampling the newly-good arm fast enough to win 0.0349 — so I would
be handing back the one column I won. Worse, retuning `c` does exactly nothing to the contextual 0.18,
because no scalar constant makes a marginal-mean estimator see `x`. This is the horizon-tuned deviation
constant I explicitly said I want to escape; retuning it is moving along the wrong axis. Reject.

The second option is an adversarial-style rule — EXP3 with exponential weights and importance-weighted
rewards, which is the standard hammer for non-stationary and adversarial rewards. It would forget by
construction. But its regret order is the wrong shape for this task: EXP3's bound is `O(sqrt(T K log K))`,
and plugging the stochastic world's numbers, `sqrt(10000 · 10 · log 10) = sqrt(10000 · 10 · 2.30) =
sqrt(230000) ≈ 480` cumulative, i.e. a normalized `~0.048` — already worse than UCB1's 0.0378 on the home
regime before I even worry about the contextual column, which EXP3 also cannot read because it too works on
marginal per-arm weights. A rule that is born worse than my rung-1 baseline on the stationary world and
still blind on the contextual world is not a universal answer. Reject. The third option is a deterministic
contextual index — LinUCB, ridge-regress `x·theta_a` per arm and add a confidence bonus in context space.
This one would genuinely attack the 0.18, and I hold that thought, because it is the first idea in two
rungs that even *models* the context. But it reintroduces exactly the object I am trying to leave behind:
a deterministic exploration bonus with a width constant that has to be calibrated against the horizon and
the noise scale, the same front-loaded-exploration disease that just cost KL-UCB the stochastic column. If
I am going to switch to a contextual model anyway, I would rather get the exploration from the posterior's
own geometry than bolt a second tunable radius onto it. That leaves the fourth option, and it is the one
that subsumes the useful half of every rejected idea: put a posterior on each arm's unknown parameter and
sample from it. It reads context if the parameter is a vector, it forgets if the posterior is discounted,
and its exploration needs no horizon-tuned constant because the posterior's own width supplies it. That is
Thompson Sampling, and I will build it per regime.

Let me reconstruct why posterior sampling is the right object, because it is genuinely not "another index."
The deterministic UCB family — UCB1, KL-UCB — adds an exploration term that is a fixed function of the
counts. Thompson's idea is to make exploration come from the *posterior's own uncertainty* instead. Put a
prior on each arm's unknown success probability, update it to a posterior as rewards arrive, and at each
round draw one plausible value from each arm's posterior and play the arm whose draw is largest. The
exploration is then automatically targeted: an arm I am unsure about has a wide posterior and its draw can
come out high, so I sample it; an arm I am confident is bad has a tight posterior near its low mean and its
draw essentially never wins, so I leave it alone; and as the posteriors concentrate the exploration
extinguishes itself with no schedule to tune. The equivalence that makes this precise is that the
probability I play arm `k` equals the posterior probability that arm `k` is optimal — probability-matching
realized by sampling. For two arms this is literally `P(p_1 > p_2 | data)`, which for Beta posteriors has a
closed form; for `K` arms the sample-and-argmax procedure is the practical realization of the same integral.
And I sample the *parameter* — the plausible mean — not a 0/1 outcome, because the comparison I need is
about which arm *could be best*, which is a question about means, not about a single Bernoulli draw whose
variance would swamp the mean gaps I am trying to resolve.

For the Bernoulli arms — the stochastic MAB and the non-stationary settings — the posterior is conjugate
and trivial. The uniform prior on `[0,1]` is `Beta(1,1)`; after `r` successes and `s` failures the
posterior is `Beta(r+1, s+1)` — the data enter only through the counts, the prior contributes the `+1`s,
and an observation just bumps one parameter. So I maintain `alpha_a` (successes + 1) and `beta_a`
(failures + 1) per arm, draw `theta_a ~ Beta(alpha_a, beta_a)`, and play `argmax_a theta_a`. Let me trace a
small concrete state to convince myself the targeting works the way I claim. Suppose early on some arm has
3 successes in 5 pulls: its posterior is `Beta(4, 3)`, mean `4/7 ≈ 0.571`, and standard deviation
`sqrt(alpha·beta / ((alpha+beta)^2 (alpha+beta+1))) = sqrt(12 / (49·8)) ≈ 0.175` — very wide, so a single
draw can easily land anywhere from 0.4 to 0.75 and this arm will keep getting sampled. Now suppose the true
best arm has been pulled 1000 times at its 0.80 mean: `Beta(801, 201)`, mean `≈ 0.799`, std
`≈ sqrt(0.80·0.20/1000) ≈ 0.0126` — a tight spike. A draw from the spike almost always beats a draw from
any arm whose mean is below `0.799 − a few·0.0126 ≈ 0.76`, so the low arms stop winning as soon as their
posteriors separate, while genuinely-close arms keep getting resampled until the data settle them. That is
exactly the self-extinguishing, gap-adaptive exploration I want. The regret analysis (Agrawal–Goyal) makes
it rigorous: `E[N_a(T)] = O(log T / Delta_a^2 + 1/Delta_a^4)` and hence `O(log T / Delta_a)` regret,
matching the Lai–Robbins `log T` order that UCB1 and KL-UCB also achieve — so I lose nothing in asymptotic
rate, and at finite horizons the randomization avoids the front-loaded deterministic exploration that, on
this very task, inflated KL-UCB's `c = 1` budget to 0.0612 on the stochastic MAB. So on the stochastic
world I expect Beta-Bernoulli TS to recover past KL-UCB and land at or below UCB1's 0.0378: the
front-loaded-exploration disease that hurt KL-UCB is precisely what a posterior-sampling rule does not have.

Now the contextual setting, the gap that has refused to move for two rungs. Here the reward is `x·theta_a`
for a `d = 10` parameter vector `theta_a` per arm, and the context `x` changes every round, so there is no
fixed best arm — the best arm depends on `x`. A per-arm scalar posterior cannot represent this: it would
average over all contexts and collapse to exactly the marginal mean the index rungs already estimate, which
is why both landed at 0.18. I need a posterior over each arm's *parameter vector* `theta_a`, and I need to
sample a plausible `theta_a` and score it against the *current* context. This is linear Thompson sampling.
Model arm `a`'s rewards as `x·theta_a + noise` with a Gaussian likelihood and a Gaussian prior on
`theta_a`; the posterior is then also Gaussian, `theta_a ~ N(theta_hat_a, v^2 · B_a^{-1})`, where
`B_a = lambda·I + sum x x^T` is the regularized design (Gram) matrix over the contexts on which arm `a` was
played, `theta_hat_a = B_a^{-1} f_a` is the ridge estimate with `f_a = sum reward · x`, and `v^2` scales the
sampling covariance. At each round I draw `theta_tilde_a ~ N(theta_hat_a, v^2 B_a^{-1})` per arm and play
`argmax_a x·theta_tilde_a`. The posterior width `v^2 B_a^{-1}` is large in directions the arm has not been
exercised, so exploration is targeted in *context space*, which is precisely the structure UCB1 and KL-UCB
could not see. Let me dimension-check the pieces so a silent shape bug does not eat the whole gain: `B_a` is
`d×d = 10×10`; `B_a^{-1} f_a` is `[d×d][d] = [d]`, so `theta_hat_a` is a length-10 vector, good; the draw
`theta_hat_a + L z` with `L` a `10×10` factor and `z` length-10 is length-10, good; and `x·theta_tilde_a`
contracts a length-10 with a length-10 to a scalar score, good. The argmax over the five scalars is the arm.

Two implementation choices make LinTS run inside this harness without blowing the compute budget, and both
survive a back-of-envelope cost count. First, I never form `B_a` and invert it from scratch each round; I
maintain `B_a^{-1}` directly and update it incrementally with the Sherman–Morrison identity: when arm `a`
is played on context `x`, `B_a^{-1} <- B_a^{-1} − (B_a^{-1} x x^T B_a^{-1}) / (1 + x^T B_a^{-1} x)`. That is
one matrix-vector product `B_a^{-1} x` at `O(d^2) = 100` flops, one outer product at `O(d^2)`, one scalar
denominator — call it a few hundred flops per update, times `T = 10000` updates, is order `10^6` flops
total, nothing. No `O(d^3)` inversion ever runs on the update path, and `theta_hat_a <- B_a^{-1} f_a` after
bumping `f_a += reward · x` is another `O(d^2)`. Second, to draw `theta_tilde_a ~ N(theta_hat_a, v^2 B_a^{-1})`
exactly I take a Cholesky factor `L` of `v^2 B_a^{-1}` and set `theta_tilde_a = theta_hat_a + L z` with `z`
standard normal. Cholesky is `O(d^3) = 1000` flops per arm, times `K = 5` arms times `T = 10000` rounds is
`5·10^7` flops — a few tens of milliseconds of numpy, entirely affordable at `d = 10`; the `d^3` term would
only bite if `d` were in the hundreds. If the Cholesky fails numerically (a `B_a^{-1}` that has drifted
slightly non-PSD from floating-point accumulation) I fall back to the isotropic `theta_hat_a + sqrt(v^2) z`,
which is a safe over-approximation of the width rather than a crash. The regularizer is `lambda = 1` — a
unit ridge prior, well-conditioned at `d = 10`, which also means `B_a^{-1}` starts at exactly `I` so the
very first draws are standard-normal explorations around zero, the correct uninformed prior. The
sampling-variance scale is `v^2 = 0.25`, and I want to pin down that number rather than guess it. The
contextual reward is clipped to `[0,1]`, and a `[0,1]`-bounded variable has variance at most `0.25`, attained
at the fair-coin point — so `v^2 = 0.25` is the conservative *upper* proxy for the reward variance that the
Gaussian likelihood wants. The actual additive noise std is only `0.1` (variance `0.01`), so `0.25` is if
anything generous, which is the right side to err on early: a too-small `v^2` would under-explore and lock
onto a wrong `theta_hat_a` before the design matrix has seen enough directions, whereas a modest surplus of
sampling width just costs a little early exploration that the posterior concentration will retire. The
harness routes all of this automatically: `select_arm` checks `context is not None and context_dim > 0` and
dispatches to the LinTS branch, so the *same* `BanditPolicy` object is per-arm Beta on the two MABs and
LinTS on the contextual world. This is the rung that finally attacks the 0.18.

Now the non-stationary setting, where I want to do at least as well as KL-UCB's 0.0349 without breaking the
stochastic world. The danger with any posterior method on a piecewise-stationary world is exactly the
disease that hurt UCB1: after a changepoint the posterior is dominated by thousands of stale observations
from the old segment, so it is sharply — and now wrongly — concentrated, and it takes a long time to
overcome that mass with fresh data. The fix that does not require detecting changepoints is a *discounted*
posterior: before each Bernoulli update, decay both posterior parameters toward the prior by a factor
`gamma < 1`, `alpha <- gamma·alpha`, `beta <- gamma·beta`, then add the new observation. Geometrically this
gives an observation `k` rounds ago weight `gamma^k`, so the total effective sample size is the geometric
sum `sum_{k>=0} gamma^k = 1/(1 − gamma)`, and the effective memory is `~1/(1−gamma)` rounds. With
`gamma = 0.999` that is `1000` rounds — comfortably shorter than the `2000`-round segments between the
changepoints at `{2000, 4000, 6000, 8000}`, so each segment's true means are learned and the previous
segment's are forgotten before the next change, yet long enough that within a segment the posterior still
concentrates well. Let me check the forgetting rate against the segment length quantitatively: stale mass
from just before a changepoint decays like `gamma^k = e^{k ln gamma} ≈ e^{−k/1000}`, so by the end of a
2000-round segment the oldest observations retain `e^{−2} ≈ 0.135` of their weight and by 1000 rounds into
the *next* segment they retain `e^{−3} ≈ 0.05` — effectively gone well before that segment's midpoint. So a
changepoint costs me roughly a memory-length worth of relearning and no more, which is the timescale I want.

The worry I have to actually resolve is what this discount does to the *stationary* MAB, because I intend to
run it on every non-contextual update, stochastic world included. Compute the steady state: if an arm with
true mean `p` is pulled steadily, its `alpha` obeys `alpha <- gamma·alpha + p` at equilibrium, so
`alpha* = p/(1−gamma) = 1000p`, and likewise `beta* = 1000(1−p)`, for a total pseudo-count of `1000`. The
posterior std at that steady state is `~sqrt(p(1−p)/1000)`; for the `p = 0.80` best arm that is
`sqrt(0.16/1000) ≈ 0.0126`, and the gaps between adjacent arm means here are `~0.05–0.10`, several times
that width — so the discounted posterior still separates the arms cleanly on the stationary world. The
discount caps the effective sample size at 1000 instead of letting it grow to 10000, which mildly inflates
the posterior width versus undiscounted TS, but a posterior-sampling rule tolerates a little extra width far
better than KL-UCB tolerated its `c = 1` budget, because the extra width only occasionally lets a
near-optimal arm win a draw, not systematically front-loads exploration across all arms. I also clamp
`alpha, beta >= 1` after decaying, so the posterior can never collapse below the uniform prior; for an arm
that is rarely pulled the decay would otherwise shrink both parameters toward zero, and the clamp instead
parks it at `Beta(1,1)`, keeping a floor of exploration alive on that arm. That clamp is the safety that
lets me run the discount globally on the Bernoulli branch: on the stationary MAB its only effect is the mild,
clamped width inflation just computed, which the steady-state check says the arm gaps absorb. The contextual
branch gets *no* discount — LinTS's ridge accumulation already has its own `lambda·I` regularization, and the
contextual world is stationary, so discounting there would only add noise and throw away hard-won design
directions. As a sanity limit, note `gamma -> 1` recovers `alpha <- alpha`, i.e. exactly the vanilla
undiscounted Beta update, so the discount is a strict one-parameter generalization that reduces to plain TS
in the no-forgetting limit — which is what I want it to be.

One more harness detail I have to get right: the RNG. The arm draws have to be genuinely random for
probability-matching to hold — if two runs shared a deterministic stream the `P(play k) = P(k optimal)`
equivalence would degrade into a fixed tie-break — so I seed a private `np.random.default_rng` in `__init__`
(seeded from `np.random.randint`), rather than leaning on the global `np.random` the placeholder used.
`reset` restores `alpha, beta` to 1, zeros the counts, and re-initializes the LinTS `B_inv`, `f`, `theta_hat`
to the prior on each fresh run so the three seeds are genuinely independent restarts. That is the whole
literal edit — one `BanditPolicy` with three branches the harness selects by `context_dim` and reward type
(the full module is in the answer).

Now the falsifiable expectations against the two prior rungs' measured numbers, using only the three
reported columns. On the **stochastic MAB** I expect Beta-Bernoulli TS to come in at or slightly below
UCB1's 0.0378 and well below KL-UCB's 0.0612 — because randomized posterior exploration avoids the
front-loaded deterministic exploration that inflated KL-UCB's `c = 1` budget here; the small clamped
discount is the only thing that could nudge it up, and the steady-state computation says any such nudge is
marginal. On the **contextual** setting I expect a large drop from the stuck ~0.18 of both index rungs —
more than an order of magnitude, since a tenth of 0.18 is already below 0.02 — because LinTS is the first
rule that actually models `x·theta_a` and explores in context space; if it does *not* collapse the
contextual regret, that falsifies the whole premise that the contextual loss was a modeling gap rather than
an exploration gap, and would say the `x·theta_a` structure is not learnable inside this budget. On the
**non-stationary** setting the comparison is the close one: KL-UCB already won it at 0.0349, and the
discounted Beta posterior attacks it with a different mechanism (forgetting versus a tighter full-history
index). I expect TS to land in the same neighborhood but plausibly a touch *worse* than KL-UCB — somewhat
above its 0.0349 — because a 1000-round effective memory still carries some stale mass into the early part
of each new segment, whereas KL-UCB's tight full-history index, once it relearns, is very sharp. So my expectation is that TS wins the stochastic and
contextual worlds decisively and is competitive-but-not-dominant on non-stationary. If that holds, TS is the
strongest single rule across the three settings — best on two of three, and the only baseline that closes the
contextual gap — which is exactly the bar a strongest-rung method has to clear: not winning every column, but
being the rule whose total regret across the three heterogeneous regimes is lowest, because it is the only
one that adapts its *mechanism* per regime (Beta posterior, LinTS, discounted Beta) instead of forcing one
index everywhere. The remaining weakness is the non-stationary column, where a method that combined
change-point detection or a sharper forgetting schedule with this posterior could plausibly do better — but
that is past the strongest single rule this task carries.
