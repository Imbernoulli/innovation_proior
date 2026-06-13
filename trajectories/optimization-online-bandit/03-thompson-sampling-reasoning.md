KL-UCB's numbers split exactly along the fault line I predicted, and they make the next move forced. On
the **non-stationary** setting it dropped to 0.0349 (per seed 0.0354 / 0.0408 / 0.0284) from UCB1's
0.083 — a clean win, confirming that the tighter KL index relearns each changepoint faster on full
history even without any forgetting, so the "sharper geometry, not windowing" bet paid off there. On the
**contextual** setting it was 0.179 (0.185 / 0.163 / 0.188), essentially identical to UCB1's 0.179 — no
surprise, the modeling gap was untouched, context still ignored. But on the **stochastic MAB** it got
*worse*: 0.0612 (0.0602 / 0.0622 / 0.0612) versus UCB1's 0.0378. That is the falsifiable risk I flagged
materializing: with the practical `c = 1` log-budget, KL-UCB explores more conservatively per arm early,
and at `T = 10000` with arm means clustered between 0.1 and 0.8 the asymptotic constant advantage never
arrives — the heavier early exploration costs more than the tighter constant saves. So I now have a rule
that wins one world and loses another relative to where I started, and the *contextual* world has not
moved at all across two rungs. Two things are clear: an index policy — Hoeffding or KL — cannot use the
context, so the 0.18 will sit there until I change *what is modeled*; and I want an exploration mechanism
that is not tied to a single deviation constant that has to be tuned against the horizon. Both point at
abandoning deterministic confidence bonuses for a *Bayesian, randomized* rule.

Let me reconstruct why posterior sampling is the right object, because it is not "another index." The
deterministic UCB family — UCB1, KL-UCB — adds an exploration term that is a fixed function of the
counts. Thompson's idea is to make exploration come from the *posterior's own uncertainty* instead. Put
a prior on each arm's unknown success probability, update it to a posterior as rewards arrive, and at
each round draw one plausible value from each arm's posterior and play the arm whose draw is largest.
The exploration is then automatically targeted: an arm I am unsure about has a wide posterior and its
draw can come out high, so I sample it; an arm I am confident is bad has a tight posterior near its low
mean and its draw essentially never wins, so I leave it alone; and as the posteriors concentrate the
exploration extinguishes itself with no schedule to tune. The key equivalence is that the probability I
play arm `k` equals the posterior probability that arm `k` is optimal — probability-matching realized by
sampling. For two arms this is exactly `P(p_1 > p_2)`, which has a closed form, but for `K` arms the
sample-and-argmax procedure is the practical realization, and crucially I sample the *parameter* (the
plausible mean), not a 0/1 outcome — the comparison structure I need is about which arm could be best,
which is a question about means.

For the Bernoulli arms (the stochastic MAB and the non-stationary settings) the posterior is conjugate
and trivial. The uniform prior on `[0,1]` is `Beta(1,1)`; after `r` successes and `s` failures the
posterior is `Beta(r+1, s+1)` — the data enter only through the counts, the prior contributes the `+1`s,
and an observation just bumps one parameter. So I maintain `alpha_a` (successes + 1) and `beta_a`
(failures + 1) per arm, draw `theta_a ~ Beta(alpha_a, beta_a)`, and play `argmax_a theta_a`. The regret
analysis (Agrawal–Goyal) gives `E[N_a(T)] = O(log T / Delta_a^2 + 1/Delta_a^4)` and hence
`O(log T / Delta_a)` regret, matching the Lai–Robbins `log T` order; in practice Thompson Sampling
typically beats UCB-style indices at finite horizons because its randomization avoids the front-loaded
deterministic exploration that, on this very task, made KL-UCB's `c = 1` budget over-explore the
stochastic MAB to 0.0612. So on the stochastic world I expect Beta-Bernoulli TS to recover and beat
*both* prior rungs — the front-loaded-exploration disease that hurt KL-UCB is exactly what a posterior-
sampling rule does not have.

Now the contextual setting, which is the gap that has refused to move. Here the reward is `x·theta_a`
for a `d = 10` parameter vector `theta_a` per arm, and the context `x` changes every round, so there is
no fixed best arm — the best arm depends on `x`. A per-arm scalar posterior cannot represent this; I need
a posterior over each arm's *parameter vector* `theta_a`, and I need to sample a plausible `theta_a` and
score it against the current context. This is linear Thompson sampling. Model arm `a`'s rewards as
`x·theta_a + noise` with a Gaussian likelihood and a Gaussian prior on `theta_a`; the posterior is then
also Gaussian, `theta_a ~ N(theta_hat_a, v^2 * B_a^{-1})`, where `B_a = lambda*I + sum x x^T` is the
regularized design (Gram) matrix over the contexts on which arm `a` was played, `theta_hat_a = B_a^{-1}
f_a` is the ridge estimate with `f_a = sum reward * x`, and `v^2` scales the sampling covariance. At each
round I draw `theta_tilde_a ~ N(theta_hat_a, v^2 B_a^{-1})` per arm and play `argmax_a x·theta_tilde_a` —
the same probability-matching idea, lifted to vector posteriors. The posterior width `v^2 B_a^{-1}` is
large in directions the arm has not been exercised, so exploration is targeted in *context space*, which
is precisely the structure UCB1 and KL-UCB could not see.

Two implementation choices make LinTS run inside this harness without blowing the budget. First, I never
form `B_a` and invert it; I maintain `B_a^{-1}` directly and update it incrementally with the
Sherman–Morrison identity: when arm `a` is played on context `x`, `B_a^{-1} <- B_a^{-1} - (B_a^{-1} x
x^T B_a^{-1}) / (1 + x^T B_a^{-1} x)`, an `O(d^2)` rank-one update with no matrix inversion, and
`theta_hat_a <- B_a^{-1} f_a` after bumping `f_a += reward * x`. Second, to draw `theta_tilde_a ~
N(theta_hat_a, v^2 B_a^{-1})` I take a Cholesky factor `L` of `v^2 B_a^{-1}` and set `theta_tilde_a =
theta_hat_a + L z` with `z` standard normal — exact Gaussian sampling, `O(d^3)` but `d = 10` so it is
cheap; if the Cholesky fails numerically I fall back to the isotropic `theta_hat_a + sqrt(v^2) z`. The
regularizer is `lambda = 1` (a unit ridge prior, standard and well-conditioned at `d = 10`) and the
sampling-variance scale is `v^2 = 0.25` — deliberately modest, because the contextual noise std is only
0.1 and the rewards are clipped to `[0,1]`, so a large `v^2` would over-explore and a too-small one would
under-explore; `0.25` is the conservative middle that matches the `[0,1]`-bounded-reward variance proxy.
The harness routes this automatically: `select_arm` checks `context is not None and context_dim > 0` and
dispatches to the LinTS branch, so the *same* `BanditPolicy` is per-arm Beta on the two MABs and LinTS on
the contextual world. This is the rung that finally attacks the 0.18.

Now the non-stationary setting, where I want to do better than KL-UCB's 0.0349 without breaking the
stochastic world. The danger with any posterior method on a piecewise-stationary world is exactly the
disease that hurt UCB1: after a changepoint the posterior is dominated by thousands of stale observations
from the old segment, so it is sharply (and wrongly) concentrated, and it takes a long time to overcome
that mass with fresh data. The fix that does not require detecting changepoints is a *discounted*
posterior: before each Bernoulli update, decay both posterior parameters toward the prior by a factor
`gamma < 1`, `alpha <- gamma*alpha`, `beta <- gamma*beta`, then add the new observation. Geometrically
this gives recent observations exponentially more weight than old ones, so the effective memory is
`~1/(1-gamma)` rounds and a changepoint's stale mass decays away on that timescale rather than persisting
for the whole run. With `gamma = 0.999` the effective window is `~1000` rounds — comfortably shorter than
the 2000-round segments between changepoints, so each segment's true means are learned and the previous
segment's are forgotten before the next change, yet long enough that within a segment the posterior still
concentrates well. I clamp `alpha, beta >= 1` after decaying so the posterior can never collapse below
the uniform prior, which keeps a floor of exploration alive even on long stationary stretches — important
because the discount is applied on *every* update, including on the stationary MAB. That clamp is what
lets me run the discount globally on the Bernoulli branch without wrecking the stochastic world: on the
stationary MAB the means never change, so the only effect of the discount is a mild, clamped inflation of
posterior width, which a posterior-sampling rule tolerates far better than KL-UCB tolerated its `c = 1`
budget. So I apply the discounted-Beta update uniformly to the non-contextual branch, accepting a tiny
cost on the stochastic world in exchange for robustness on the non-stationary one. The contextual branch
gets no discount — LinTS's ridge accumulation has its own regularization, and the contextual world is
stationary, so discounting there would only add noise.

One more harness detail I have to get right: the RNG. The arm draws have to be genuinely random for
probability-matching to work, so I seed a private `np.random.default_rng` in `__init__` (drawn from
`np.random.randint`), rather than relying on the global `np.random` the placeholder used. `reset`
restores `alpha, beta` to 1, zeros the counts, and re-initializes the LinTS `B_inv`, `f`, `theta_hat` to
the prior on each fresh run. That is the whole literal edit — one `BanditPolicy` with three branches the
harness selects by `context_dim` and reward type (the full module is in the answer).

Now the falsifiable expectations against the two prior rungs' measured numbers. On the **stochastic MAB**
I expect Beta-Bernoulli TS to come in around 0.035 — at or slightly below UCB1's 0.0378 and well below
KL-UCB's 0.0612 — because randomized posterior exploration avoids the front-loaded deterministic
exploration that inflated KL-UCB's `c = 1` budget here; the small clamped discount is the only thing that
could nudge it up, and if it does, it should be marginal. On the **contextual** setting I expect a large
drop from the stuck ~0.18 of both index rungs to roughly 0.02 — more than an order of magnitude — because
LinTS is the first rule that actually models `x·theta_a` and explores in context space; if it does *not*
collapse the contextual regret, that falsifies the whole premise that the contextual loss was a modeling
gap rather than an exploration gap. On the **non-stationary** setting the comparison is the close one: KL-
UCB already won it at 0.0349, and the discounted Beta posterior has a different mechanism (forgetting vs
tighter index). I expect TS to land in the same neighborhood but plausibly a touch *worse* than KL-UCB —
maybe ~0.06 — because a 1000-round effective memory still carries some stale mass into the early part of
each new segment, whereas KL-UCB's tight full-history index, once it relearns, is very sharp. So my
expectation is that TS wins the stochastic and contextual worlds decisively and is competitive-but-not-
dominant on non-stationary. If that holds, TS is the strongest single rule across the three settings —
best on two of three, and the only baseline that closes the contextual gap — which is exactly the bar a
strongest-rung method has to clear: not winning every column, but being the rule whose total regret
across the three heterogeneous regimes is the lowest, because it is the only one that adapts its
mechanism per regime (Beta posterior, LinTS, discounted Beta) instead of forcing one index everywhere.
The remaining weakness is the non-stationary column, where a method that combined change-point detection
or a sharper forgetting schedule with this posterior could plausibly do better — but that is past the
strongest rule the task carries.
