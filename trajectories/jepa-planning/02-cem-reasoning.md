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
length, so more steps buy proportionally less reach and the agent has to take ever more of them to
cover the same ground.

There is a second thing hiding in the success column that sharpens the diagnosis. Random scores `0.55`
at horizon 30 and `0.70` at *both* 60 and 90 — the success rate rose once and then plateaued, even
though the horizon kept growing and the step count kept climbing (`82 → 112`). So the extra thirty steps
from horizon 60 to 90 bought *more wandering* (`112` steps to succeed instead of `82`) but *no more
successes*. That is the re-plan lottery hitting its ceiling: past some point, handing the loop more
control steps just lets the agent random-walk around longer without raising the fraction of episodes
where the white cloud ever happens to point a plan through the door. The `0.70` plateau is the ceiling
of pure luck in this maze, and it says the deficit is not "too few re-plans" — more re-plans stopped
helping — it is that no single `plan()` call ever *commits* to the door. That is precisely the thing an
adaptive refit inserts, and it is why I expect refitting to break the plateau rather than just nudge it.

The mechanism I predicted at the floor is confirmed in the numbers, then, but so is the leak. The
planner learns nothing within a control step: it spends the same 200-sample batch the iterative methods
will spend *per iteration*, once, and throws every score away. Every one of those 200 rollouts is a
measurement of where the low-cost sequences live — the objective scores the predicted final latent's
closeness to the goal, so each cost is essentially "how near did this plan's endpoint land" — and the
floor reads all 200 of those measurements only to keep one and discard the rest, then re-draws the
identical white cloud on the next call. The re-planning loop is doing the real work, and it is doing it
by brute lottery. That is the leak to plug, and the fix is not a better noise source yet — it is to stop
drawing from a frozen distribution and instead let the sampling distribution *move toward* the sequences
that scored well, batch after batch, so within a single `plan()` call the search concentrates on the
route through the door instead of re-rolling white noise.

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
event common are the same problem. So I can borrow whatever the rare-event people use to hunt down a
good sampling distribution.

Their tool is importance sampling: don't sample from the broad `f(·;u)`, sample from a cleverer `g` and
undo the change of measure with the likelihood ratio, `ℓ = E_g[I{S(X) ≥ γ} f(X;u)/g(X)]`. And there is
a *best* `g` — the one giving the estimator zero variance. The estimator is an average of i.i.d. terms
`Y = I{S(X) ≥ γ} f(X;u)/g(X)`, so its variance is `Var_g(Y)/N`, and that vanishes exactly when `Y` is
almost surely constant in `x`. An almost-surely-constant random variable equals its own mean `ℓ`, so
set the summand to `ℓ`: that forces `g*(x) = I{S(x) ≥ γ} f(x;u)/ℓ`. Read that: `g*` is just the
baseline density *restricted to the elite event* `{S(x) ≥ γ}` and renormalized. It is precisely the
"throw away everything below the bar, keep the mass on the good region" distribution I was hand-waving
about when I said "move toward the good sequences." Before I lean on `g*` as the target for the whole
method, I want to actually confirm the zero-variance claim rather than trust the algebra, because it is
load-bearing. Take the toy `f = N(0,1)`, `S(x) = x`, `γ = 2`. Then `ℓ = P(X ≥ 2) = 1 − Φ(2) ≈ 0.02275`,
and `g*` is `f` truncated to `[2, ∞)` and renormalized. If I draw a hundred thousand points from that
truncated normal and form the summand `I{x ≥ 2} f(x)/g*(x)`, its sample mean comes out `0.02275` and its
sample standard deviation `≈ 5×10⁻¹⁸` — the summand is the constant `ℓ` to floating-point roundoff, so
the estimator is exact from a single sample. `g*` really is the variance-minimizing density, not just a
plausible candidate. The catch is the one that motivated all this: `g*` carries the unknown `ℓ` and is
some arbitrary restricted shape, so I cannot sample it directly.

I cannot sample `g*` directly, so I restrict to a tractable parametric family `f(·;v)` and find the
member closest to the restricted target. At iteration `t` the target is built from the law I am actually
using now, `g*_t(x) = I{S(x) ≥ γ} f(x;v_{t−1})/ℓ_t`. Closest in what sense? The KL divergence,
`D(g,h) = ∫ g ln(g/h)`. Minimizing `D(g*_t, f(·;v))` over `v` drops the `v`-free first term and leaves
`max_v ∫ g*_t(x) ln f(x;v) dx = max_v E_{v_{t−1}}[ I{S(X) ≥ γ} ln f(X;v) ]` (the positive constant
`1/ℓ_t` does not move the argmax). Replace the expectation by its Monte-Carlo counterpart over the batch
I actually drew from `f(·;v_{t−1})`, and read what the indicator does: `I{S(X_k) ≥ γ}` is 1 for the
sequences that beat the bar and 0 for the rest, so the objective is the *log-likelihood of the
threshold-beating samples under `f(·;v)`* — maximizing it is fitting `f(·;v)` by maximum likelihood to
those samples alone. That is the answer to "how do I move the distribution toward the good sequences,"
and it is not a nudge: take the sequences that beat the bar, fit your sampling distribution to them by
maximum likelihood.

But a fixed high `γ` is a dead end — exactly the rarity that motivated importance sampling. Under the
broad initial distribution almost no sequence clears a high `γ`, every indicator is zero, and the
maximum-likelihood program has no data. So I stop fixing `γ` and let it track the sample: each
iteration, sort the costs and take the top fraction as elites — equivalently the `(1−ρ)` sample
quantile. Then there are always exactly that many elites, never zero, no matter how broad the
distribution. And it is self-tuning: early, when the distribution is broad, the elite cutoff is a
modest score; as the distribution concentrates on good plans, the same top fraction is a *higher* score,
so the bar ratchets upward toward the optimum on its own. The loop becomes: sample from the current
distribution, score by rolling through the model, take the lowest-cost elites, refit the distribution to
them by maximum likelihood, repeat. This is what random shooting structurally could not do — it is the
iteration the floor discarded.

Now the family. The fit is a maximum-likelihood estimation I run *every* iteration, so I want a closed
form (no inner optimization) and cheap sampling. Both point at the Gaussian. Here the candidate is an
action sequence of shape `[plan_length, action_dim]`, so I take a diagonal Gaussian over every
(timestep, coordinate) — per entry, `x ~ N(μ, σ²)`. I should actually derive the elite-restricted MLE
rather than quote it, because the update *is* the algorithm and I want to see it drop out. The Gaussian
log-density of one elite with independent coordinates is `Σ_j [−½ln(2π) − ln σ_j − (x_j−μ_j)²/(2σ_j²)]`;
sum over the elite set `I` and maximize. The partial in `μ_j` is `Σ_{k∈I}(x_{kj} − μ_j)/σ_j²`; set it to
zero, and since `σ_j²` is a common positive factor it cancels, leaving `Σ_{k∈I}(x_{kj} − μ_j) = 0`, so
`μ_j` is the **elite sample mean**, coordinate by coordinate. The partial in `σ_j` of the `j`-term is
`−1/σ_j + (x_{kj}−μ_j)²/σ_j³`; summing over the elites and setting to zero gives `−N_e/σ_j +
(1/σ_j³)Σ_{k∈I}(x_{kj}−μ_j)² = 0`, i.e. `σ_j²` is the **elite sample variance** (divided by `N_e`, the
MLE, not the unbiased `N_e−1`). So the whole update is: take the elite sequences, compute their
per-(timestep,coordinate) mean and spread, and those *are* next iteration's sampling parameters. The
mean re-centers the search on where the good action sequences clustered; the variance re-sizes the
exploration — automatically tighter on the coordinates the elites agreed about, wider where they did
not. That second part is the thing the fixed-spread random floor never had: the optimizer sets its own
exploration width from the data instead of me freezing it at unit spread. In the code I store the spread
as `torch.std` over the elite dimension — the same lowest-cost-elite refit loop.

Let me read what limit this drives toward, because it tells me both the right fixed point and the
danger. As iterations proceed and the distribution concentrates, the elites cluster tighter and their
variance shrinks toward zero; in the limit `σ → 0` the Gaussian collapses to a point mass at `μ` — a
sampler fully concentrated on the best region found, which is the *right* fixed point, so the dynamics
are self-terminating in spirit. But that same collapse can bite: if an early batch's elites happen to
land in a small cluster (by luck, or because the broad initial sampling under-covered the real optimum's
basin) the variance update slams `σ` small right there, and now I am sampling a tiny region around a
possibly-wrong plan with no spread to escape — the spread collapsed before the search found the global
basin. In this maze that is a real risk, and I can name exactly where it comes from: the cost surface has
two routes through the door, and if the first elites all cluster on the wrong-side route the variance can
commit before the right route is even sampled. The standard guard is to start the spread deliberately
*wide* so the first iterations explore aggressively; here I set the initial standard deviation to
`var_scale = 1.5` everywhere, larger than the unit spread the floor used, exactly so the early batches
cover both routes before any collapse.

It is worth being concrete about *how* the variance collapse actually commits the search to a route,
because it is not uniform across coordinates and that anisotropy is the whole trick. The elites are the
top 10% by cost, and the cost is dominated by reach toward the goal — so on the coordinates that most
determine whether a plan reaches the door (the early timesteps' direction, the sustained heading), the
20 elites agree strongly, their per-coordinate spread is far below the sampling spread, and the variance
update tightens those coordinates hard. On coordinates that barely affect the endpoint (fine jitter late
in the horizon, motion orthogonal to the goal bearing once through the door), the elites disagree about
as much as the full population, so their variance stays wide and CEM keeps exploring them. The refit is
therefore an *anisotropic* contraction: it locks in the reach-relevant directions within a few
iterations while leaving the irrelevant ones loose. That is exactly the behavior I want against the
floor's failure — the floor spread its energy isotropically and uselessly; CEM concentrates it on the
directions that carry the plan to the door — and it is also where the two-route danger lives, because
"the direction that carries the plan to the door" is bimodal, and if the first elite set happens to
agree on the wrong side, the same fast contraction that is a virtue will lock the search onto the wrong
route. This rung accepts that risk in exchange for the commitment; I am flagging it because it is the
kind of thing that, if it shows up as a horizon where CEM underperforms, will demand a less brittle
selection than a hard top-k.

That `1.5` is not a free number — I can check what it does to the sampled steps against the reach problem
the floor exposed. With `σ = 1.5` per coordinate, an action's squared norm has mean `2σ² = 4.5`, so the
RMS per-step displacement is `2.12`, right up against the `max_norm = 2.45` cap, and the fraction of
actions that saturate the cap jumps to `P(‖a‖ > 2.45) = exp(−2.45²/(2·1.5²)) ≈ 0.26`. Compare the unit
spread of the floor, where only about `5%` of actions saturated and the RMS step was `1.41`. So widening
to `1.5` pushes roughly a quarter of the initial actions to the boundary and lifts the typical step by
half — the early CEM population is composed of near-maximal strides, which is precisely what a plan needs
to reach a door tens of cells away inside one horizon. The wide init is doing double duty: it delays the
variance collapse *and* it puts the first batch's energy where the reach problem demands it. (A second
guard — blending the fresh fit with the previous parameters by a momentum factor, `μ ← α μ_elite +
(1−α) μ_prev` — exists and is a generic CEM damping knob, but this rung uses the raw mean/std update;
momentum is the knob I would turn only if premature collapse proved to be the bottleneck rather than
budget.)

The continuous action space forces one more thing: the per-step action norm is capped at `2.45`, but a
Gaussian has unbounded support, so some sampled actions are infeasible, and handing the cost an
infeasible candidate scores a fantasy. I take the same radial projection the floor used — draw from the
plain Gaussian, then for any action whose 2-D norm exceeds `max_norm = 2.45` rescale it to the boundary,
leaving shorter actions untouched. Concretely with `n = ‖a‖`, multiply by `clamp(min(max(n,0),
max_norm))/(n + ε)`: inside the ball unchanged, outside pulled to the boundary, direction preserved.
The one thing this rung does that the floor could not is that the elites are then fit on the *projected*
sequences, so the refit respects the constraint too — the mean and variance CEM marches toward are
statistics of feasible plans, not of the unbounded Gaussian.

Now the bookkeeping numbers, from the harness. The constructor receives `num_samples = 200` and
`n_iters = 20` — the shared defaults — and unlike the floor I *use* `n_iters`: 20 refinement passes,
each drawing 200 sequences, so `4000` rollouts per `plan()` call against the floor's `200`. That is a
twentyfold compute increase, and I want to be honest that the comparison to random is therefore not a
compute-matched one; but it is the right comparison, because the floor was built to hold the *first
batch* equal so that any win here is attributable to the iteration and refit rather than a bigger draw.
The elite count I take as `max(10, num_samples // 10)`, i.e. the top 10% (a floor of 10 so the mean and
variance are estimable even if the budget shrinks); with 200 samples that is the 20 lowest-cost
sequences per iteration. Twenty elites is enough to estimate a `2·plan_length`-dimensional mean and
diagonal variance without the fit being dominated by a single sequence, and 10% is tight enough that the
`γ`-bar it implies sits well into the good tail. The initial mean is the zero action sequence — no prior
bias toward any direction, the same neutral start the floor used — and the initial standard deviation is
the wide `var_scale = 1.5`. Because my objective is a cost to minimize while the derivation maximized
`S = −cost`, the elites are the samples with the **lowest** cost, which I pick as the top-k of `−cost`.

What do I return? The distribution's center. After the final refit, the mean sequence is the Gaussian's
point estimate of the optimum, and in the degenerate `σ → 0` limit it *is* the optimum, so I return
`mean` as the plan (the loop takes its first action). I also pass back the per-iteration min cost and
elite-cost mean/std as diagnostics, but those do not affect control — though they are exactly the trace
I would read to *catch* a premature collapse: if the elite-cost std falls toward zero over the 20
iterations while the min cost stays high and flat, that is the signature of the variance collapsing onto
a mediocre basin — the elites all agree (low spread) yet none of them is good (high min) — and it would
be the number that sends me to the momentum guard. If instead the min cost keeps dropping while the
elite-cost std shrinks, that is healthy convergence onto a genuinely improving route. Nothing needs to persist across env
steps — each `plan()` call re-optimizes from the zero-mean, wide-`σ` start — so `t0`, the first-call
marker, is irrelevant here, exactly as it was for the floor. And the whole loop runs under `no_grad`
because, like the floor, CEM only ever reads function values of the cost. It never touches the gradient
the differentiable model could hand it — a fact I am noting now because it is the lever a later rung
will pull.

So the falsifiable expectations against random's numbers. The iteration-and-refit is supposed to fix
exactly the wandering the step counts exposed, so the cleanest single prediction is the step count: if
refitting toward the elites is the right fix, `mean_steps_to_success` must *collapse* relative to
random's 64/82/112, because the planner now commits to a direction within a single call — the refit
marches the distribution onto a coherent route through the door within a few iterations — instead of
re-rolling white noise every control step until it stumbles in. I expect the success rate to rise at
every horizon, most clearly at the longer ones where the floor's Brownian sequences fell shortest
(random's 0.70 at horizons 60/90 should climb, since the refit lets a single 60- or 90-step plan
actually reach the door rather than relying on the re-plan lottery). And I expect `mean_dist` to drop
substantially from random's 12.6/9.6/8.2, because even on the episodes CEM fails to clear the `4.5`
threshold it should finish much closer — a refit distribution centered on the door is a better place to
fail from than a white cloud. If instead the success rate barely moves, or it moves but the step count
does *not* collapse, that points at premature convergence — the variance collapsing onto the wrong route
before the search covers both — and would send me to the momentum guard or make me question the white
noise source itself. But the headline bet is the step count: it is the number that most directly encodes
"does the planner commit or wander," and refitting toward the elites is a commitment mechanism. The full
scaffold module — the literal `CustomPlanner` fill for CEM — is in the answer.
