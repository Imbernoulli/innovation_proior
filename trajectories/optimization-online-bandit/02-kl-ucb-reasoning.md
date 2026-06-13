UCB1's numbers told me exactly where the optimism principle is right and where its *geometry* is wrong,
and they told me in three very different places. On the stochastic MAB it landed at normalized regret
0.0378 (per seed 0.0397 / 0.0348 / 0.0389) — solid, tight across seeds, the home regime working as the
theory promised. On the contextual setting it was 0.179 (0.183 / 0.162 / 0.192) — bad, but bad for a
reason I already named: it ignores the context, so it is estimating marginal arm means that are close
and there is no fixed best arm to converge to. And on the non-stationary setting it was 0.083 (0.077 /
0.083 / 0.089) — its worst world, more than double its stochastic number, exactly the slow-unlearning
failure I predicted: the full-history index is anchored by thousands of stale samples after each of the
four changepoints, so the formerly-best arm keeps a tiny confidence radius and the agent relearns the
new best slowly, four times over. So I have two distinct problems, and I should be honest that one rung
will not solve both. The contextual loss is a *modeling* gap (no use of `x`); the stochastic and
non-stationary numbers are both about the *confidence geometry* of the index. Let me attack the geometry
first, because it is the same defect underneath both: UCB1's bonus is the wrong shape.

Look hard at that bonus, `sqrt(2 log t / N_a)`. It depends only on `N_a` and the range `[0,1]`; it never
looks at *where* `mu_hat_a` sits. So an arm with `mu_hat = 0.5` and an arm with `mu_hat = 0.05` get the
*same* confidence half-width at equal `N_a`. That cannot be right. A coin that comes up heads five times
in a hundred is intrinsically far less variable than a fair coin — its sample mean concentrates much
harder — so my interval around 0.05 should be much narrower than my interval around 0.5. UCB1 throws
that away: it uses a *symmetric*, range-only width, which is exactly the inverse of the quadratic
Pinsker bound, and Pinsker is loosest precisely near the boundaries 0 and 1. On the stochastic MAB the
arm means run from 0.10 up to 0.80, so several arms sit well away from 0.5 where this slackness bites; on
the non-stationary world the segments include means like 0.1, 0.2, 0.85, 0.9 — again boundary-heavy. So
the symmetric width over-explores exactly the low-mean and high-mean arms these settings are full of,
which is leaking regret on both. The disease is the variance-blind, symmetric confidence radius.

Is there a floor I am trying to reach, so I know how far off the constant `8` is? Yes — Lai and Robbins
proved one, generalized by Burnetas and Katehakis: for any uniformly-good policy,
`liminf E[N_a(T)]/log T >= 1/Kinf(nu_a, mu*)`, where `Kinf` is the smallest KL divergence to a
distribution whose mean exceeds `mu*`. For Bernoulli arms this collapses to the Bernoulli KL
`d(mu_a, mu*)`, with `d(p,q) = p log(p/q) + (1-p) log((1-p)/(1-q))`. So the per-arm floor is
`1/d(mu_a, mu*)`. Now compare. Pinsker says `d(mu_a, mu*) >= 2 Delta_a^2`, so `1/d <= 1/(2 Delta_a^2)`;
UCB1 pays `8/Delta_a^2`, sixteen times the Pinsker-level constant, and worse against the true `d`. And
the gap between `d` and the Pinsker parabola *grows* as means approach 0 or 1 — the local curvature of
`d` is `1/(2 p(1-p))`, which blows up near the boundary while Pinsker keeps the flat `2`. That is the
quantitative version of "the boundary arm should be cheap to rule out, and UCB1 doesn't notice." On the
stochastic MAB this is exactly why UCB1's 0.0378 is "good but not tight" — it is paying the inflated
constant on every off-center arm.

So the fix I want is concrete: a confidence width built from `d` itself, not from its Pinsker lower
bound. Where does that come from? From the actual large-deviation tail of the sample mean, not the
Hoeffding one. For i.i.d. Bernoulli with mean `mu`, Chernoff's method gives
`P(mu_hat >= mu + eps) <= exp(-n d(mu+eps, mu))` — the *exact* exponential rate, derived by minimizing
the moment generating function over the tilt: for `lambda > 0`,
`P(mu_hat >= mu+eps) <= (mu exp(lambda(1-mu-eps)) + (1-mu) exp(-lambda(mu+eps)))^n`, and the minimizing
`lambda = log((mu+eps)(1-mu)/(mu(1-mu-eps)))` collapses the base to exactly `exp(-d(mu+eps, mu))`.
Hoeffding (Pinsker) only delivers `exp(-2 n eps^2)`, the same near `mu = 1/2` but enormously weaker near
the boundaries. So `n d(mu_hat, mu)` is the natural "nats of surprise" in an observed deviation, and I
should build my interval by thresholding *that*.

Concretely, invert the tail. Fix a deviation budget and define the upper bound as the largest mean `q`
whose KL distance from `mu_hat_a` costs at most that budget:
`U_a(t) = max{ q in [0,1] : N_a * d(mu_hat_a, q) <= budget }`. Because `d(mu_hat_a, .)` is zero at `q =
mu_hat_a`, strictly convex, and strictly increasing on `[mu_hat_a, 1]`, this `U_a(t)` is well defined —
the unique root where `d(mu_hat_a, q) = budget/N_a`, or `1` if the whole right branch is feasible. And —
this is the payoff — as `mu_hat_a` rises toward 1, the function `d(mu_hat_a, .)` gets steeper, so the
width `U_a(t) - mu_hat_a` *shrinks on its own*. The interval is asymmetric and self-tightening near the
boundary, for free, no variance estimate to plug in. UCB1 is literally the special case where I replace
`d(p,q)` by its Pinsker lower bound `2(p-q)^2`: then `N_a * 2(mu_hat_a - q)^2 <= budget` solves to `q =
mu_hat_a + sqrt(budget/(2 N_a))`, the additive sqrt bonus. So I am not inventing a new family — I am
using the *tight* member of the one UCB1 already lives in.

What budget? To make the failure risk decay like `1/t` I want `exp(-N_a * budget/N_a) ~ 1/t`, i.e.
`budget ~ log t`. The proven exploration function is `f(t) = log t + 3 log log t` (the `c = 3` correction
is what makes the optimal-arm under-estimation term provably negligible — lower `c` lets the relevant
series diverge), but the `3 log log t` term is a theorem artifact: for it to matter against a clean
`(1+eps) log t` you would need `t` past `~10^51`, far beyond `T = 10000`. So the practical budget is the
pure `c_impl * log t` with `c_impl = 1`, the theorem-tight constant. That is the literal choice I land:
`budget = log(max(t,1))`, no `log log` correction. The full KL-UCB regret proof — a self-normalized
supermartingale `W_t = exp(lambda S(t) - N(t) psi_mu(lambda))` to handle the *random* number of pulls,
peeled over geometric slices of `N(t)`, giving `P(under-estimate) <= e * ceil(budget log T) * exp(-budget)`,
then a decomposition of suboptimal pulls into best-arm under-estimates and an over-estimate tail —
delivers `E[N_a(T)] <= (1+eps) log T / d(mu_a, mu*) + O(log log T)`, hence
`limsup R_T/log T <= sum_a Delta_a / d(mu_a, mu*)`, which is the Lai–Robbins floor. So for the Bernoulli
worlds (stochastic_mab and nonstationary are both Bernoulli) this index is asymptotically optimal, a
strict leading-constant improvement over UCB1. And it is distribution-free over `[0,1]`: the convexity
lemma `E[exp(lambda X)] <= 1 - mu + mu exp(lambda)` shows the Bernoulli is the least-concentrated bounded
law, so its `d` dominates everyone's deviations — the proof never needed the rewards to be binary.

Now the harness-specific decisions, because the contract is the same `select_arm/update`, and there are
two temptations from UCB1's failure that I have to resolve honestly. The first: UCB1's worst world was
non-stationary, and at the close of the last rung I hoped a tighter index "in particular, would forget
stale history." I have to be careful — KL-UCB does not, by itself, forget. The natural non-stationary
move is a sliding window (recompute the index over only the last `W` pulls). But I already saw, when I
considered it for UCB1, that windowing inflates the *stationary* regret badly: discarding history keeps
every confidence radius permanently wide, so the index never tightens, and the stochastic MAB is exactly
where UCB1 was already strong (0.0378). Since one rule is graded on all three settings and I cannot see
which world I am in, paying a certain loss on the stochastic world to chase the non-stationary one is the
wrong trade — and the whole reason to move to KL-UCB is the *opposite*, to make the stochastic index
tighter, not looser. So I keep KL-UCB on the *full* history with no window. The non-stationary
improvement, if any, has to come purely from the tighter index relearning a changed arm faster than
UCB1's range-only bound does — not from forgetting. The second temptation is the contextual setting:
KL-UCB, like UCB1, has no contextual machinery, so on the contextual world it again runs per-arm KL-UCB
ignoring `x`. I expect it to be no better than UCB1 there, because the modeling gap is untouched; that
loss is real and I am not pretending this rung closes it.

For the implementation I need to actually compute `U_a(t) = max{ q : N_a d(mu_hat_a, q) <= log t }`.
There is no closed form for the inverse of `d` in its second argument, but the geometry is friendly: for
fixed `p = mu_hat_a`, `q -> d(p,q)` is strictly convex, zero at `p`, increasing on `[p,1]`, so the
equation `d(p,q) = (log t)/N_a` has a unique root and `U_a(t)` is that root. I find it by bisection on
`[p, 1]`. The harness already exposes `kl_ucb_bound(mu_hat, n, t, c)` via `scipy.optimize.brentq`, but I
will hand-write a fixed 32-iteration binary search instead — 32 halvings give `~1e-10` precision,
deterministic cost, and avoid the per-call `brentq` overhead, which matters because this runs `K` times
per round for `T = 10000` rounds across many seeds and settings. Each iteration computes `d(p, mid)`
directly: `p log(p/mid) + (1-p) log((1-p)/(1-mid))`, with `p` clipped off the exact 0/1 endpoints (where
`d = +inf`) to a tiny epsilon so the logarithm never sees zero. I seed the search on `[p, 1-1e-10]`
rather than the Pinsker bracket — with a fixed 32-iteration budget the wider bracket still converges to
full precision, and it removes a dependence on the seed formula. The policy structure is then: play each
arm once (`t < K` round-robin), then for each arm compute the KL-UCB index from its running `mu_hat_a`
and `N_a`, and `argmax`. `update` just accumulates counts and reward sums — no buffer, no window, no
context state. That is the whole literal edit (the full module is in the answer).

Now the falsifiable expectations against UCB1's measured numbers. On the **stochastic MAB** the tighter
KL geometry should reduce regret on the off-center arms relative to UCB1's symmetric bound — but I
should be honest that with the practical `c = 1` budget (vs UCB1's effective `c = 2`), KL-UCB explores
*more conservatively per arm* early on, and on a horizon of only `T = 10000` with arm means clustered
between 0.1 and 0.8 the asymptotic constant advantage may not have fully materialized. So I would not be
shocked if KL-UCB's stochastic number is *worse* than UCB1's 0.0378 here — the asymptotic optimality is a
`T -> infinity` statement, and at `T = 10000` the heavier early exploration of `c = 1` log-budget can
cost more than the tighter constant saves. On the **non-stationary** setting I expect KL-UCB to *beat*
UCB1's 0.083: the tighter index relearns a changed arm faster because its confidence radius near the new
best's empirical mean is sharper, so even on full history it should recover from each of the four
changepoints quicker than UCB1's range-only bound, pulling the normalized regret down below 0.083. On the
**contextual** setting I expect essentially no change from UCB1's 0.179 — same modeling gap, context
still ignored — so it should land in the same high-teens neighborhood, around 0.18. If KL-UCB does *not*
improve the non-stationary number, that falsifies the "tighter index relearns faster on full history"
hypothesis and would say the only real cure for non-stationarity is forgetting, which this rung
deliberately refused. And whatever happens on stochastic_mab and contextual, the conclusion is already
pointed at the next rung: an index — optimistic or KL — cannot use the context, and the contextual 0.18
will not move until I switch to a rule that actually models `x·theta_a`. That is a Bayesian, posterior-
sampling rule, which also gives a single clean mechanism for adapting to non-stationarity via a
discounted posterior — closing both gaps at once.
