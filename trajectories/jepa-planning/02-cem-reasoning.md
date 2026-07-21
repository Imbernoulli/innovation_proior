The random-search floor told me exactly where it bleeds, and it told me in the step counts. At horizon
30 it cleared only 0.55 of the episodes — eleven of twenty — and finished a mean distance of 12.6 from
the goal, which in a `65×65` grid is more than half the time never getting near. At horizons 60 and 90
the success rate crept to 0.70, but look at how it got there. The `mean_dist` fell across horizons —
12.6, 9.6, 8.2 — while `mean_steps_to_success` *rose* — 64, 82, 112. Read those two trends together and
they say one thing: the longer horizon does not make the planner smarter, it makes it wander longer and
occasionally get luckier. The distances close only because a 90-step episode gives the re-planning loop
more control steps, hence more 200-sample lotteries, to stumble the agent inward; the step counts climb
because each of those successes is reached by a random walk, not a route. Put numbers on the climb:
`82/64 = 1.28` and `112/82 = 1.37`, so the steps-to-success grows *faster* than linearly in the horizon
ratio (`60/30 = 2`, `90/60 = 1.5`), which is exactly the Brownian signature — a white-noise action
sequence integrates to a random walk whose net displacement grows only like the square root of its
length, so more steps buy proportionally less reach.

There is a second thing hiding in the success column that sharpens the diagnosis. Random scores `0.55`
at horizon 30 and `0.70` at *both* 60 and 90 — the success rate rose once and then plateaued, even
though the horizon kept growing and the step count kept climbing (`82 → 112`). So the extra thirty steps
from horizon 60 to 90 bought *more wandering* but *no more successes*. That is the re-plan lottery
hitting its ceiling: past some point, handing the loop more control steps just lets the agent
random-walk around longer without raising the fraction of episodes where the white cloud ever happens to
point a plan through the door. The `0.70` plateau is the ceiling of pure luck in this maze, and it says
the deficit is not "too few re-plans" — more re-plans stopped helping — it is that no single `plan()`
call ever *commits* to the door. That is precisely the thing an adaptive refit inserts, and it is why I
expect refitting to break the plateau rather than just nudge it.

The leak is confirmed too: the planner learns nothing within a control step, spending the same
200-sample batch the iterative methods will spend *per iteration*, once, and throwing every score away.
Each of those 200 costs is essentially "how near did this plan's endpoint land," and the floor reads all
200 only to keep one and re-draw the identical white cloud. The fix is not a better noise source yet — it
is to stop drawing from a frozen distribution and instead let the sampling distribution *move toward* the
sequences that scored well, batch after batch, so within a single `plan()` call the search concentrates
on the route through the door.

I want that move to be principled, not a nudge, because "move toward the good points" can be done a
dozen ad-hoc ways — average the best few, step by some rate, weight by score — and I have no idea which
is right or what it converges to. So let me re-read the problem. Instead of asking "where is the lowest
cost?", pick a high score threshold `γ` (recall the cost is to minimize, so a high score means a low
cost — I will work with `S = −cost` and chase its max) and ask: how probable is it that a randomly drawn
action sequence clears `γ`? That is `ℓ(γ) = P_u(S(X) ≥ γ)` under some broad baseline density. Now push
`γ` up toward the true optimum. The set `{x : S(x) ≥ γ}` shrinks toward the maximizers, and under a
broad density its probability tends to zero — it becomes a *rare* event. A density that made this rare
event likely — that put its mass on `{S(X) ≥ γ}` for `γ` near the optimum — is exactly a density that
samples almost only the best action sequences. Finding the optimal plan and making the near-optimum
event common are the same problem, so I can borrow whatever the rare-event people use to hunt a good
sampling distribution.

Their tool is importance sampling: don't sample from the broad `f(·;u)`, sample from a cleverer `g` and
undo the change of measure with the likelihood ratio, `ℓ = E_g[I{S(X) ≥ γ} f(X;u)/g(X)]`. And there is
a *best* `g` — the one giving the estimator zero variance. The estimator is an average of i.i.d. terms
`Y = I{S(X) ≥ γ} f(X;u)/g(X)`, and its variance vanishes exactly when `Y` is almost surely constant in
`x`; an a.s.-constant variable equals its own mean `ℓ`, so setting the summand to `ℓ` forces
`g*(x) = I{S(x) ≥ γ} f(x;u)/ℓ`. Read that: `g*` is just the baseline density *restricted to the elite
event* `{S(x) ≥ γ}` and renormalized — precisely the "throw away everything below the bar, keep the mass
on the good region" distribution I was hand-waving about. The catch is the one that motivated all this:
`g*` carries the unknown `ℓ` and is some arbitrary restricted shape, so I cannot sample it directly.

So I restrict to a tractable parametric family `f(·;v)` and find the member closest to the restricted
target in KL divergence. At iteration `t` the target is `g*_t(x) = I{S(x) ≥ γ} f(x;v_{t−1})/ℓ_t`, and
minimizing `D(g*_t, f(·;v))` over `v` drops the `v`-free term and leaves `max_v E_{v_{t−1}}[ I{S(X) ≥ γ}
ln f(X;v) ]`. Replace the expectation by its Monte-Carlo counterpart over the batch I actually drew, and
read what the indicator does: it is 1 for the sequences that beat the bar and 0 for the rest, so the
objective is the *log-likelihood of the threshold-beating samples under `f(·;v)`* — maximizing it is
fitting `f(·;v)` by maximum likelihood to those samples alone. That is the answer to "how do I move the
distribution toward the good sequences," and it is not a nudge: take the sequences that beat the bar, fit
your sampling distribution to them by maximum likelihood.

But a fixed high `γ` is a dead end — exactly the rarity that motivated importance sampling. Under the
broad initial distribution almost no sequence clears a high `γ`, every indicator is zero, and the
maximum-likelihood program has no data. So I stop fixing `γ` and let it track the sample: each iteration,
sort the costs and take the top fraction as elites — the `(1−ρ)` sample quantile. Then there are always
exactly that many elites, never zero, and it is self-tuning: early, when the distribution is broad, the
cutoff is a modest score; as the distribution concentrates on good plans, the same top fraction is a
*higher* score, so the bar ratchets upward toward the optimum on its own. The loop becomes: sample from
the current distribution, score by rolling through the model, take the lowest-cost elites, refit by
maximum likelihood, repeat. This is what random shooting structurally could not do.

Now the family. The fit is a maximum-likelihood estimation I run *every* iteration, so I want a closed
form and cheap sampling — both point at a diagonal Gaussian over every (timestep, coordinate). For a
Gaussian the elite-restricted MLE is exactly the elite sample mean and variance, coordinate by
coordinate (the σ cancels out of the mean stationarity condition; the variance is the MLE `1/N_e`
version, not the unbiased `N_e−1`). So the whole update is: take the elite sequences, compute their
per-(timestep,coordinate) mean and spread, and those *are* next iteration's sampling parameters. The
mean re-centers the search on where the good action sequences clustered; the variance re-sizes the
exploration — automatically tighter on the coordinates the elites agreed about, wider where they did
not. That second part is the thing the fixed-spread random floor never had: the optimizer sets its own
exploration width from the data instead of me freezing it at unit spread. In the code the spread is
`torch.std` over the elite dimension.

The limit this drives toward tells me both the right fixed point and the danger. As the distribution
concentrates the elites cluster tighter and their variance shrinks toward zero; in the limit `σ → 0` the
Gaussian collapses to a point mass at `μ` — a sampler fully concentrated on the best region found, the
*right* fixed point, so the dynamics are self-terminating in spirit. But that same collapse can bite: if
an early batch's elites happen to land in a small cluster (by luck, or because the broad initial sampling
under-covered the real optimum's basin) the variance update slams `σ` small right there, and now I am
sampling a tiny region around a possibly-wrong plan with no spread to escape. In this maze that is a real
risk with a specific shape — the cost has two routes through the door, and the elites are the top 10% by
a cost dominated by reach, so on the coordinates that most determine whether a plan reaches the door the
elites agree strongly and the variance tightens *hard*, while on coordinates that barely affect the
endpoint it stays wide. The refit is an anisotropic contraction that locks in the reach-relevant
directions within a few iterations — exactly the behavior I want against the floor's isotropic waste —
but "the direction that carries the plan to the door" is bimodal, so if the first elite set agrees on the
wrong side, that same fast contraction locks the search onto the wrong route. This rung accepts the risk
for the commitment; if it shows up as a horizon where CEM underperforms, it will demand a less brittle
selection than a hard top-k.

The standard guard is to start the spread deliberately *wide* so the first iterations explore
aggressively; I set the initial standard deviation to `var_scale = 1.5` everywhere, larger than the
floor's unit spread. That number does double duty against the reach problem the floor exposed: at
`σ = 1.5` an action's squared norm has mean `2σ² = 4.5`, an RMS per-step displacement of `2.12` right up
against the `max_norm = 2.45` cap, so roughly a quarter of the initial actions saturate the cap (against
about 5% at unit spread). The early CEM population is composed of near-maximal strides — precisely what a
plan needs to reach a door tens of cells away inside one horizon — so the wide init both delays the
variance collapse and puts the first batch's energy where reach demands it. (A momentum blend of the
fresh fit with the previous parameters is the generic CEM damping knob I would reach for if premature
collapse proved to be the bottleneck; this rung uses the raw mean/std update.)

The continuous action space forces the same feasibility projection the floor used — draw from the
Gaussian, then radially rescale any action whose 2-D norm exceeds `max_norm = 2.45` to the boundary,
direction preserved. The one thing this rung does that the floor could not is fit the elites on the
*projected* sequences, so the mean and variance CEM marches toward are statistics of feasible plans.

The bookkeeping, from the harness: the constructor receives the shared `num_samples = 200`, `n_iters =
20` defaults, and unlike the floor I *use* `n_iters` — 20 refinement passes, 200 sequences each, so 4000
rollouts per `plan()` call against the floor's 200. That is a twentyfold compute increase and the
comparison to random is therefore not compute-matched, but it is the right comparison, because the floor
held the *first batch* equal so any win here is attributable to the iteration and refit rather than a
bigger draw. The elite count is `max(10, num_samples // 10)` — the top 20 with 200 samples, a floor of
10 so the mean and variance stay estimable if the budget shrinks; 20 elites is enough to estimate a
`2·plan_length`-dimensional mean and diagonal variance without a single sequence dominating, and 10% is
tight enough that the implied `γ`-bar sits well into the good tail. The initial mean is the zero action
sequence — no directional prior, the floor's neutral start — and because my objective is a cost to
minimize while the derivation maximized `S = −cost`, the elites are the samples with the *lowest* cost,
picked as the top-k of `−cost`.

What I return is the distribution's center: after the final refit the mean sequence is the Gaussian's
point estimate of the optimum, and in the `σ → 0` limit it *is* the optimum. I also pass back the
per-iteration min cost and elite-cost mean/std as diagnostics — they do not affect control, but they are
exactly the trace that would *catch* a premature collapse: if the elite-cost std falls toward zero while
the min cost stays high and flat, the elites all agree yet none is good, the signature of the variance
collapsing onto a mediocre basin, and that is the number that would send me to the momentum guard.
Nothing persists across env steps — each call re-optimizes from the zero-mean, wide-`σ` start — so `t0`
is irrelevant, as it was for the floor, and the whole loop runs under `no_grad` because CEM only ever
reads function values; it never touches the gradient the differentiable model could hand it, a lever a
later rung will pull.

So the falsifiable expectation against random's numbers. The iteration-and-refit is aimed straight at
the wandering the step counts exposed, so the cleanest single prediction is the step count: if refitting
toward the elites is the right fix, `mean_steps_to_success` must *collapse* relative to random's 64/82/112,
because the planner now commits to a coherent route within a single call instead of re-rolling white noise
every control step until it stumbles in. I expect success to rise at every horizon, most clearly at the
longer ones where the floor's Brownian sequences fell shortest, and `mean_dist` to drop substantially
from 12.6/9.6/8.2 — a refit distribution centered on the door is a better place to fail from than a white
cloud. If instead the step count does *not* collapse, that points at premature convergence onto the wrong
route and would send me to the momentum guard or make me question the white noise source itself. The full
scaffold module for CEM is in the answer.
