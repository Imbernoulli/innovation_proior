The random-search floor told me exactly where it bleeds, and it told me in the step counts. At horizon
30 it cleared only 0.55 of the episodes and finished a mean distance of 12.6 from the goal — more than
half the time it never got near. At horizons 60 and 90 the success rate crept to 0.70, but look at how
it got there: `mean_steps_to_success` ran 64 → 82 → 112, climbing with the horizon. When random
shooting succeeds, it succeeds *slowly* — it wanders. That is the Brownian signature I predicted: a
white-noise action sequence integrates to a random walk whose net displacement grows only like the
square root of its length, so most single-pass batches never produce a sequence that actually reaches
the door, and the ones that do meander there. The re-planning loop is doing the real work — re-drawing
200 fresh white sequences every control step and occasionally catching a lucky one — which is why the
longer horizons, with more re-plans, edge upward at all. But the planner itself learns nothing within a
step: it spends the same 200-sample batch the iterative methods will spend *per iteration*, once, and
throws every score away. Every one of those 200 rollouts told me something about where the low-cost
sequences live, and the floor discards all of it. That is the leak to plug, and the fix is not a better
noise source yet — it is to stop drawing from a frozen distribution and instead let the sampling
distribution *move toward* the sequences that scored well, batch after batch.

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
a *best* `g` — the one giving the estimator zero variance. Zero variance means the summand
`I{S(x) ≥ γ} f(x;u)/g(x)` is constant in `x`; setting it equal to its own mean `ℓ` forces
`g*(x) = I{S(x) ≥ γ} f(x;u)/ℓ`. Read that: `g*` is just the baseline density *restricted to the elite
event* `{S(x) ≥ γ}` and renormalized. It is precisely the "throw away everything below the bar, keep the
mass on the good region" distribution I was hand-waving about when I said "move toward the good
sequences." The rare-event framing handed me the ideal target for my adaptive sampler, and it is not a
heuristic — it is the variance-minimizing density.

I cannot sample `g*` directly — it carries the unknown `ℓ` and is some arbitrary restricted shape — so I
restrict to a tractable parametric family `f(·;v)` and find the member closest to the restricted target.
At iteration `t` the target is built from the law I am actually using now,
`g*_t(x) = I{S(x) ≥ γ} f(x;v_{t−1})/ℓ_t`. Closest in what sense? The KL divergence,
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
iteration, sort the costs, and take the top fraction as elites — equivalently the `(1−ρ)` sample
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
(timestep, coordinate) — per entry, `x ~ N(μ, σ²)`. Its elite-restricted MLE drops out cleanly. The
Gaussian log-density of one elite with independent coordinates is
`Σ [−½ln(2π) − ln σ − (x−μ)²/(2σ²)]`; summing over the elite set and setting `∂/∂μ = 0` gives
`Σ(x − μ)/σ² = 0`, and since `σ²` is a common positive factor it cancels, leaving `μ` = the **elite
sample mean**, coordinate by coordinate. Setting `∂/∂σ = 0` gives `−N_e/σ + (1/σ³)Σ(x−μ)² = 0`, i.e.
`σ²` = the **elite sample variance**. So the whole update is: take the elite sequences, compute their
per-(timestep,coordinate) mean and spread, and those *are* next iteration's sampling parameters. The
mean re-centers the search on where the good action sequences clustered; the variance re-sizes the
exploration — automatically tighter on the coordinates the elites agreed about, wider where they did
not. That second part is the thing the fixed-spread random floor never had: the optimizer sets its own
exploration width from the data instead of me freezing it. In the code I store the spread as
`torch.std` over the elite dimension — the same lowest-cost-elite refit loop.

Let me read what limit this drives toward, because it tells me both the right fixed point and the
danger. As iterations proceed and the distribution concentrates, the elites cluster tighter and their
variance shrinks toward zero; in the limit `σ → 0` the Gaussian collapses to a point mass at `μ` — a
sampler fully concentrated on the best region found, which is the *right* fixed point. But that same
collapse can bite: if an early batch's elites happen to land in a small cluster (by luck, or because the
broad initial sampling under-covered the real optimum's basin) the variance update slams `σ` small right
there, and now I am sampling a tiny region around a possibly-wrong plan with no spread to escape — the
spread collapsed before the search found the global basin. In this maze that is a real risk: the cost
surface has two routes through the door, and if the first elites all cluster on the wrong-side route the
variance can commit before the right route is even sampled. The standard guard is to start the spread
deliberately *wide* so the first iterations explore aggressively; here I set the initial standard
deviation to `var_scale = 1.5` everywhere, larger than the unit spread, exactly so the early batches
cover both routes before any collapse. (A second guard — blending the fresh fit with the previous
parameters by a momentum factor — exists, but this rung uses the raw mean/std update; momentum is the
knob I would turn only if premature collapse proved to be the bottleneck.)

The continuous action space forces one more thing: the per-step action norm is capped at `2.45`, but a
Gaussian has unbounded support, so some sampled actions are infeasible. I take the same projection the
floor used — draw from the plain Gaussian, then for any action whose 2-D norm exceeds `max_norm = 2.45`
rescale it to the boundary, leaving shorter actions untouched. Concretely with `n = ‖a‖`, multiply by
`clamp(min(max(n,0), max_norm))/(n + ε)`: inside the ball unchanged, outside pulled to the boundary.
Feasibility is enforced before the cost ever sees a candidate, and the elites are then fit on the
*projected* sequences, so the refit respects the constraint too.

Now the bookkeeping numbers, from the harness. The constructor receives `num_samples = 200` and
`n_iters = 20` — the shared defaults — and unlike the floor I *use* `n_iters`: 20 refinement passes,
each drawing 200 sequences. The elite count I take as `max(10, num_samples // 10)`, i.e. the top 10% (a
floor of 10 so the mean and variance are estimable even if the budget shrinks); with 200 samples that is
the 20 lowest-cost sequences per iteration. The initial mean is the zero action sequence — no prior bias
toward any direction, the same neutral start the floor used — and the initial standard deviation is the
wide `var_scale = 1.5`. Because my objective is a cost to minimize while the derivation maximized
`S = −cost`, the elites are the samples with the **lowest** cost, which I pick as the top-k of `−cost`.

What do I return? The distribution's center. After the final refit, the mean sequence is the Gaussian's
point estimate of the optimum, and in the degenerate `σ → 0` limit it *is* the optimum, so I return
`mean` as the plan (the loop takes its first action). I also pass back the per-iteration min cost and
elite-cost mean/std as diagnostics, but those do not affect control. Nothing needs to persist across env
steps — each `plan()` call re-optimizes from the zero-mean, wide-`σ` start — so `t0` is irrelevant here,
and the whole loop runs under `no_grad` because, like the floor, CEM only ever reads function values of
the cost. It never touches the gradient the differentiable model could hand it — a fact I am noting now
because it is the lever a later rung will pull.

So the falsifiable expectations against random's numbers. The iteration-and-refit is supposed to fix
exactly the wandering the step counts exposed. I expect `mean_steps_to_success` to *collapse* relative
to random's 64/82/112 — the refit should march the distribution onto a coherent route through the door
within a few iterations, so successes arrive far faster — and I expect the success rate to rise at every
horizon, most clearly at the longer ones where the floor's Brownian sequences fell shortest (random's
0.70 at horizons 60/90 should climb). The `mean_dist` should drop substantially from random's 12.6 / 9.6
/ 8.2, because even on the episodes CEM fails to reach the threshold it should finish much closer. If
instead the success rate barely moves or the variance collapses onto the wrong route, that points at
premature convergence and would send me to the momentum guard or a non-white noise source. But the
cleanest single prediction is the step count: if refitting toward the elites is the right fix for
random's wandering, the steps-to-success must fall sharply, because the planner now commits to a
direction instead of re-rolling white noise every call. The full scaffold module — the literal
`CustomPlanner` fill for CEM — is in the answer.
