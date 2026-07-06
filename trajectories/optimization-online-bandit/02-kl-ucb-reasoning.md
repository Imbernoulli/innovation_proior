UCB1's numbers told me exactly where the optimism principle is right and where its *geometry* is wrong,
and they told me in three very different places. On the stochastic MAB it landed at normalized regret
0.0378 (per seed 0.0397 / 0.0348 / 0.0389) — solid, tight across seeds (the spread is `0.0049`, about a
seventh of the mean), the home regime working as the theory promised. Read the cumulative form: that is
`378` regret against my gap-free envelope of `~960` for this world, so UCB1 is realizing regret well
inside its own worst-case bound, which is exactly what "home regime" should look like. On the contextual
setting it was 0.179 (0.183 / 0.162 / 0.192, spread `0.030`, a noticeably wider sixth of the mean, the
signature of a rule with almost no signal to lock onto) — bad, but bad for a reason I already named: it
ignores the context, so it is estimating marginal arm means that are close and there is no fixed best arm
to converge to. And on the non-stationary setting it was 0.083 (0.077 / 0.083 / 0.089) — its worst
world, `2.2×` its stochastic number, exactly the slow-unlearning failure I predicted: the full-history
index is anchored by thousands of stale samples after each of the four changepoints, so the formerly-best
arm keeps a tiny confidence radius and the agent relearns the new best slowly, four times over. In
cumulative terms that world cost `829` regret, and spread across four changepoints that is roughly `207`
of regret burned relearning at each one — the per-changepoint cost I am trying to cut. So I have
two distinct problems, and I should be honest that one rung will not solve both. The contextual loss is a
*modeling* gap (no use of `x`); the stochastic and non-stationary numbers are both about the *confidence
geometry* of the index. The contextual gap needs a new model and I will not touch it here; let me attack
the geometry first, because it is the same defect underneath both Bernoulli worlds: UCB1's bonus is the
wrong shape.

Look hard at that bonus, `sqrt(2 log t / N_a)`. It depends only on `N_a` and the range `[0,1]`; it never
looks at *where* `mu_hat_a` sits. So an arm with `mu_hat = 0.5` and an arm with `mu_hat = 0.05` get the
*same* confidence half-width at equal `N_a`. That cannot be right. A coin that comes up heads five times
in a hundred is intrinsically far less variable than a fair coin — its sample mean concentrates much
harder — so my interval around 0.05 should be much narrower than my interval around 0.5. UCB1 throws
that away: it uses a *symmetric*, range-only width, which is exactly the inverse of the quadratic
Pinsker bound, and Pinsker is loosest precisely near the boundaries 0 and 1. On the stochastic MAB the
arm means run from 0.10 up to 0.80, so several arms sit well away from 0.5 where this slackness bites; on
the non-stationary world the segments are piecewise-Bernoulli and, as such benchmarks usually are to make
their changepoints detectable, plausibly boundary-leaning — again a place where the symmetric width is
slack. So the symmetric width over-explores exactly the low-mean and high-mean arms these settings are
full of, which is leaking regret on both. The disease is the variance-blind, symmetric confidence radius.

Is there a floor I am trying to reach, so I know how far off the constant `8` is? Yes — Lai and Robbins
proved one, generalized by Burnetas and Katehakis: for any uniformly-good policy,
`liminf E[N_a(T)]/log T >= 1/Kinf(nu_a, mu*)`, where `Kinf` is the smallest KL divergence to a
distribution whose mean exceeds `mu*`. For Bernoulli arms this collapses to the Bernoulli KL
`d(mu_a, mu*)`, with `d(p,q) = p log(p/q) + (1-p) log((1-p)/(1-q))`. So the per-arm floor is
`1/d(mu_a, mu*)`. Now compare, and compare on the actual arms. Pinsker says `d(mu_a, mu*) >= 2 Delta_a^2`,
so `1/d <= 1/(2 Delta_a^2)`; UCB1 pays `8/Delta_a^2`, sixteen times the Pinsker-level constant, and worse
against the true `d`. Take the near-optimal arm at `mu = 0.70` (gap `0.10`): the true rate is `d(0.70,
0.80) = 0.0282`, the Pinsker surrogate is `2 * 0.10^2 = 0.0200`, so the floor per this arm is `1/0.0282 =
35.5` in units of `log T`, while UCB1's bound is `8/0.10^2 = 800` — a factor of `~22`. And take the
far arm at `mu = 0.10` (gap `0.70`): `d(0.10, 0.80) = 1.146` versus Pinsker `2 * 0.49 = 0.98`, so `d` is
`1.17×` the parabola there. The gap between `d` and the Pinsker parabola *grows* as means approach 0 or
1 — the local curvature of `d` in its first argument is `1/(2 p(1-p))`, which blows up near the boundary
while Pinsker keeps the flat `2`. That is the quantitative version of "the boundary arm should be cheap
to rule out, and UCB1 doesn't notice."

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

Let me put a number on how much the geometry actually buys, because that number is going to control my
expectation for the stochastic world, and I do not want to fool myself into predicting a win that the
arm configuration will not deliver. Take a common budget `b = budget/N_a = 0.01` and compare the KL width
to the Pinsker width at three arms, solving `d(p, p+w) = b` for the KL width `w` and `sqrt(b/2)` for the
Pinsker one. At `p = 0.10` the KL width is `0.048` against Pinsker's `0.071` — the KL ball is `2/3` the
size, a real tightening. At `p = 0.50` they are `0.070` versus `0.071` — essentially identical, as they
must be, since Pinsker is tight at the center. But at `p = 0.70` the KL width is `0.062` against
Pinsker's `0.071` — only `12%` tighter. And here is the uncomfortable part: `0.70` is exactly the arm
that *drives* the stochastic MAB's regret. The regret-heavy arms are the small-gap ones clustered just
below the optimum — `0.55`, `0.60`, `0.70` — and their KL curvatures `1/(2 p(1-p))` are `2.02`, `2.08`,
`2.38`, barely above Pinsker's flat `2.0`. The big curvature lives at the far low arms (`p = 0.10` has
curvature `5.56`), but those have large gaps and get ruled out after a handful of pulls regardless of
which geometry I use, so tightening them saves almost nothing. So the tighter constant is real, but on
*this* arm profile it barely activates where the regret is, and it is an asymptotic statement besides.

Let me trace the two actual indices at a mid-run round to see the effect operationally, since `select_arm`
will compute the literal KL-UCB index against UCB1's literal radius, budgets and all. Take `t = 5000`.
A far low arm sitting at `mu_hat = 0.10` with `N = 100` pulls gets KL-UCB index `0.265` versus UCB1's
`0.10 + sqrt(2 log 5000 / 100) = 0.513`: KL-UCB has already crushed that arm's optimism down near the
truth while UCB1 still floats it above `0.5`, so KL-UCB stops wasting pulls on it far sooner. Now the
regret-driving near-top arm at `mu_hat = 0.70` with `N = 500`: KL-UCB index `0.779` versus UCB1's `0.885`.
Both indices drop under KL, but look at *how much* — the low arm's index fell by `0.25`, the near-top
arm's by only `0.11`. The tightening is concentrated on the arms that were never the problem. That is the
same story the curvature computation told, now in operational numbers: KL-UCB clamps hard on the far arms
I could already rule out and only gently on the clustered top arms where the regret actually lives. As a
sanity check on the other end of the range, a freshly-seen arm with `mu_hat = 0` after `N = 1` pull gets
KL index `max{ q : d(0,q) <= log t } = 1 - 1/t`, which at `t = 100` is `0.99` — near the top, exactly as
it should be, matching UCB1's saturated optimism for an unknown arm; the two rules agree that a
once-pulled arm must stay maximally optimistic, and only diverge as evidence accumulates. So the index is
well-behaved at both extremes; the question is entirely about the crowded middle, and there the numbers
say the win is modest.

What budget? To make the failure risk decay like `1/t` I want `exp(-N_a * budget/N_a) ~ 1/t`, i.e.
`budget ~ log t`. The proven exploration function is `f(t) = log t + 3 log log t` (the `c = 3` correction
is what makes the optimal-arm under-estimation term provably negligible — lower `c` lets the relevant
series diverge), but the `3 log log t` term is a theorem artifact: for it to matter against a clean
`(1+eps) log t` you would need `t` past `~10^51`, far beyond `T = 10000`, where `3 log log t = 3 * log
log 10000 ≈ 6.8` is a rounding error on `log t ≈ 9.2`. So the practical budget is the pure `c_impl * log
t` with `c_impl = 1`, the theorem-tight constant. That is the literal choice I land: `budget =
log(max(t,1))`, no `log log` correction. Note what this does to the exploration relative to UCB1: UCB1's
radius carries `2 log t` in its square (its effective `c = 2`), while my KL budget is a bare `log t` (`c
= 1`), so on top of the geometry I am also running a *smaller* deviation budget. Both effects push my
index below UCB1's — more exploitation, less exploration — which is the right asymptotic direction, but
at `T = 10000` a smaller budget on the crowded top arms could just as easily under-explore and delay
pinning the best among `0.5 / 0.55 / 0.6 / 0.7 / 0.8`. This is the second reason the stochastic outcome
is genuinely uncertain, not a guaranteed improvement.

The full KL-UCB regret proof — a self-normalized supermartingale `W_t = exp(lambda S(t) - N(t)
psi_mu(lambda))` to handle the *random* number of pulls, peeled over geometric slices of `N(t)`, giving
`P(under-estimate) <= e * ceil(budget log T) * exp(-budget)`, then a decomposition of suboptimal pulls
into best-arm under-estimates and an over-estimate tail — delivers `E[N_a(T)] <= (1+eps) log T /
d(mu_a, mu*) + O(log log T)`, hence `limsup R_T/log T <= sum_a Delta_a / d(mu_a, mu*)`, which is the
Lai–Robbins floor. So for the Bernoulli worlds (stochastic_mab and nonstationary are both Bernoulli)
this index is asymptotically optimal, a strict leading-constant improvement over UCB1. And it is
distribution-free over `[0,1]`: the convexity lemma `E[exp(lambda X)] <= 1 - mu + mu exp(lambda)` shows
the Bernoulli is the least-concentrated bounded law, so its `d` dominates everyone's deviations — the
proof never needed the rewards to be binary.

Now the harness-specific decisions, because the contract is the same `select_arm/update`, and there are
two temptations from UCB1's failure that I have to resolve honestly. The first: UCB1's worst world was
non-stationary, and at the close of the last rung I hoped a tighter index "in particular, would forget
stale history." I have to be careful — KL-UCB does not, by itself, forget. The natural non-stationary
move is a sliding window (recompute the index over only the last `W` pulls). But I already saw, when I
considered it for UCB1, why windowing wrecks the *stationary* regret: because UCB1-style rules concentrate
almost all pulls on the best arm, a window keeps every suboptimal arm's in-window count tiny, so its
radius stays permanently wide, and every `~W` rounds a confirmed-bad arm ages out of the window and
gets re-explored from scratch — enough forced re-exploration to inflate the stationary cumulative regret
from `~960` toward `~1450`. The stochastic MAB is exactly where UCB1 was already strong (0.0378), and the
whole reason to move to KL-UCB is the *opposite* of windowing — to make the stochastic index tighter, not
looser. Since one rule is graded on all three settings and I cannot see which world I am in, paying a
certain loss on the stochastic world to chase the non-stationary one is the wrong trade. So I keep KL-UCB
on the *full* history with no window. The non-stationary improvement, if any, has to come purely from the
tighter index relearning a changed arm faster than UCB1's range-only bound does — not from forgetting.
The second temptation is the contextual setting: KL-UCB, like UCB1, has no contextual machinery, so on
the contextual world it again runs per-arm KL-UCB ignoring `x`. The marginal means collapse together
there exactly as they did for UCB1, so I expect no better than UCB1's 0.179; that loss is real and I am
not pretending this rung closes it.

For the implementation I need to actually compute `U_a(t) = max{ q : N_a d(mu_hat_a, q) <= log t }`.
There is no closed form for the inverse of `d` in its second argument, but the geometry is friendly: for
fixed `p = mu_hat_a`, `q -> d(p,q)` is strictly convex, zero at `p`, increasing on `[p,1]`, so the
equation `d(p,q) = (log t)/N_a` has a unique root and `U_a(t)` is that root. I find it by bisection on
`[p, 1]`. The harness already exposes `kl_ucb_bound(mu_hat, n, t, c)` via `scipy.optimize.brentq`, but I
will hand-write a fixed 32-iteration binary search instead — the bracket starts with width at most `1`,
and 32 halvings shrink that to `2^{-32} ≈ 2.3e-10`, so I get roughly `1e-10` precision at a deterministic
cost, avoiding the per-call `brentq` overhead, which matters because this runs `K` times per round for `T
= 10000` rounds across three seeds and three settings. Each iteration computes `d(p, mid)` directly:
`p log(p/mid) + (1-p) log((1-p)/(1-mid))`, with `p` clipped off the exact 0/1 endpoints (where `d =
+inf`) to a tiny epsilon so the logarithm never sees zero. I seed the search on `[p, 1-1e-10]` rather
than the Pinsker bracket — with a fixed 32-iteration budget the wider bracket still converges to full
precision, and it removes a dependence on the seed formula. The policy structure is then: play each arm
once (`t < K` round-robin), then for each arm compute the KL-UCB index from its running `mu_hat_a` and
`N_a`, and `argmax`. `update` just accumulates counts and reward sums — no buffer, no window, no context
state. That is the whole literal edit (the full module is in the answer).

Before I write down expectations, let me reason carefully about the non-stationary dynamics, because the
claim "a tighter index relearns faster" is not automatically true and I want to know which way it actually
cuts. Split a changepoint into two arms. The *formerly*-best arm carries a high stale `mu_hat`, a huge
`N`, and therefore a tiny radius under either geometry — and that persistence is the dominant cost:
because `N` is enormous, each fresh contrary sample barely moves `mu_hat`, so this arm keeps a high index
and keeps getting pulled until its estimate finally decays, and that decay is governed by `N`, not by the
radius shape. So on the fallen arm KL-UCB and UCB1 suffer almost identically; the tighter geometry buys
me little there. The leverage, if any, is on the *newly*-best arm, which was suboptimal before the change
(low stale `mu_hat`, moderate `N`). As its fresh rewards start arriving high, two effects fight. On one
side, a tighter radius is *less* optimistic, which could slow the initial re-exploration that first
rediscovers the risen arm — a point against KL-UCB. On the other side, once that arm's `mu_hat` climbs
into a boundary-heavy region (a segment mean near 0.85–0.9, say), the KL bound there is sharp, so the
index tracks the true risen mean closely and the *separation* between a rising arm and a decaying
competitor is far crisper than under UCB1's flat, muddy width — the arm-swap resolves cleanly instead of
oscillating. My bet is that on boundary-leaning segments the crisper separation outweighs the slower
initial nudge, so KL-UCB nets out ahead — but the two effects are genuinely opposed, which is exactly why
I am treating the non-stationary improvement as a falsifiable prediction rather than a theorem.

One harness subtlety worth naming: only two of the three worlds are literally Bernoulli. The contextual
rewards are clipped Gaussians, not coin flips, so the Bernoulli-KL index is technically the wrong
likelihood there — but it does not matter, because on the contextual world I ignore the context and the
marginal reward is still bounded in `[0,1]`, and the same convexity lemma that made the proof
distribution-free says the Bernoulli `d` upper-bounds the deviations of *any* `[0,1]`-supported law. So
the index stays a valid (if slack) confidence bound on the contextual marginal; it just cannot help me
there for the modeling reason, not a validity reason.

Now the falsifiable expectations against UCB1's measured numbers. On the **stochastic MAB** I have
argued myself into genuine uncertainty, and I should record it as such rather than pretend. The tighter
KL geometry should reduce regret on the off-center arms relative to UCB1's symmetric bound — but the
computation above says the reduction is only `~12%` at the `0.70` arm that actually drives the regret,
while the `c = 1` budget (against UCB1's effective `c = 2`) explores more conservatively per arm early,
and on a horizon of only `T = 10000` with arm means clustered between 0.1 and 0.8 the asymptotic constant
advantage may not have fully materialized. So I would not be shocked if KL-UCB's stochastic number comes
in *worse* than UCB1's 0.0378 — the asymptotic optimality is a `T -> infinity` statement, and at `T =
10000` the heavier-or-lighter early exploration of the `c = 1` log-budget can cost more than the tighter
constant saves on this particular arm profile. I expect it near 0.0378, and the stochastic_mab column
will tell me the sign. On the **non-stationary** setting I expect KL-UCB to *beat* UCB1's 0.083: the
tighter index relearns a changed arm faster because, on segments whose means lean toward the boundaries,
the KL bound separates a risen arm from a fallen one far more sharply than the flat Pinsker width does —
so even on full history it should recover from each of the four changepoints quicker than UCB1's
range-only bound, pulling the normalized regret down below 0.083. That is a hypothesis, not a certainty,
because the persistence of the stale old-best arm (large `N`, tiny radius, slow-decaying `mu_hat`) hurts
KL-UCB and UCB1 alike; if KL-UCB does *not* improve the non-stationary number, that falsifies the
"tighter index relearns faster on full history" bet and would say the only real cure for non-stationarity
is forgetting, which this rung deliberately refused. On the **contextual** setting I expect essentially
no change from UCB1's 0.179 — same modeling gap, context still ignored, marginal means still collapsed —
so it should land in the same high-teens neighborhood, around 0.18.

Whatever happens on stochastic_mab and contextual, the conclusion is already pointed at the next rung: an
index — optimistic or KL — cannot use the context, and the contextual 0.18 will not move until I switch
to a rule that actually models `x·theta_a`. What I want next is a rule that carries a full uncertainty
model over each arm's parameter — rich enough to read the context on the contextual world, and with a
built-in way to down-weight stale history on the non-stationary one — so that a single change of what is
modeled attacks both remaining gaps at once.
