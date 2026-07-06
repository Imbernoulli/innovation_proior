The SPOT numbers tell a sharp story, and I want to read them carefully before deciding what to change.
Pen-cloned landed at a mean of 56.4, but the three seeds were 61.5, 23.1, 84.6 — a spread of over 60
points, seed 123 falling to 23 while seed 456 nearly hit 85. That is not a method with a stable operating
point; it is a method whose outcome depends heavily on the draw. Hammer-cloned averaged 19.7, and the
average is a lie: the seeds were 1.9, 1.3, 55.8. Two runs were essentially dead and one lucky run at 55.8
carried the mean single-handedly — the 456 seed is roughly thirty to forty times the other two, so
"19.7" describes no run that actually happened. And the held-out expert variant, hammer-expert,
*collapsed* to a mean of 2.5 across a tight 3.0 / 3.0 / 1.4 — near random on every seed, no lucky
exception. That last column is the one that diagnoses the method, precisely because it is tight: this is
not variance, it is a systematic failure. On expert data the VAE's support is *tight*, so the density
penalty should keep the deterministic TD3 actor exactly on the demonstrated trajectory — and instead the
policy went to nearly zero return on all three seeds. The contrast is the tell: on the noisy `cloned`
mixtures the method is erratic (wide spread, one lucky hammer seed), but on the *cleanest* data it fails
*consistently*. If the binding problem were the policy's support, clean expert data — where the support
is a thin, unambiguous tube — should be the easy case, not the guaranteed-failure case. So the support
constraint is not the binding problem. What breaks is the mechanism I flagged at the close of the last
rung: a deterministic actor maximizing `Q` rides whatever OOD action the critic over-values, and on
expert data — where the support is a thin tube and the cooled `λ` opens it during the online phase — once
the actor steps a little off that tube the critic has no real feedback to correct it and the value runs
away. The collapse on the cleanest data, uniform across seeds, is the signature that the binding problem
is the *deterministic maximizer + bootstrapped critic* exposing itself to OOD values during improvement,
not the policy's support per se. So the move is to keep the implicit-constraint idea but stop letting the
actor's improvement step ever query `Q` at a self-proposed action at all.

Let me restate what I actually need, because SPOT got two of three requirements right and I want to keep
those. I need an off-policy critic (data efficiency — the online budget is a real 1M steps but the
offline data must be reused throughout, which SPOT's TD3 critic does). I need a policy constraint for
offline stability (SPOT had one). And I need a constraint that does *not* require keeping an explicit
behavior model accurate during the shifting online phase — and here is where SPOT is brittle: it leans
on a VAE whose support, frozen, is either too tight (expert: any deviation is unguarded once `λ` cools,
which is the 2.5 collapse) or too broad (cloned: it admits the noise, which is the erratic 1.9 / 1.3 /
55.8). Notice that both failure faces trace to the *same* separately-fit density model being wrong in
opposite directions on different data. I want the constraint to come for free from the *update math*, not
from a separately-fit density model that I have to get calibrated per-dataset. That is the gap the next
rung closes.

Before I derive the fix, quantify how bad SPOT's instability really is, because it sets the bar I have to
clear and tells me which task is the true test. Take pen-cloned: seeds 61.5, 23.1, 84.6 have a mean of
56.4 and a spread from lowest to highest of 61.5 points — the range is larger than the mean itself, so
the coefficient of variation is over 50%. A method whose run-to-run range exceeds its own mean is not
yet solving the task; it is sometimes stumbling into a good policy. Hammer-cloned is worse in a different
way: with 55.8 against 1.9 and 1.3, the good seed is about `55.8/1.6 ≈ 35×` the typical bad seed, so the
"19.7" mean is an artifact of one draw and the honest summary is "usually near zero, occasionally works."
Hammer-expert is the clean diagnostic: 3.0 / 3.0 / 1.4, range 1.6, essentially no variance — a flat,
reproducible failure. So the three columns are three different signatures — high-variance-mediocre,
one-lucky-seed, and flat-collapse — and the flat-collapse on the *easiest* data is the one that isolates
the mechanism, because variance can hide a lot but a tight failure on clean demos cannot be explained by
"unlucky seed." The bar for the next rung is therefore not just a higher mean; it is *tight*
reproducibility on hammer-expert, which only a method with no OOD-query at improvement time can promise.

Write the constrained improvement problem and solve it exactly, and watch whether the behavior model can
be made to disappear. The advantage `A(s,a) = Q(s,a) − V(s)` is what improvement pushes up (maximizing
`E_π[Q]` equals maximizing `E_π[A]` since `V` is action-independent), so at iteration `k`:
`π_{k+1} = argmax_π E_{a~π}[A^{π_k}(s,a)]` subject to `KL(π(·|s) ‖ π_β(·|s)) ≤ ε` and normalization.
Form the Lagrangian with multiplier `λ` on the KL and `α` on normalization, write the KL as
`∫ π(log π − log π_β)`, and differentiate with respect to `π(a|s)` at a single action: the
`E_π[A]` term gives `A(s,a)`, the `−λ·KL` term gives `−λ(log π − log π_β + 1)` (the `+1` from
differentiating `π log π`), the normalization gives `−α`. Setting the sum to zero,
`A − λ(log π − log π_β + 1) − α = 0`, and solving for `log π` yields
`log π(a|s) = (1/λ)A(s,a) + log π_β(a|s) − 1 − α/λ`; the `−1 − α/λ` is action-independent, so
exponentiating and folding it into a per-state normalizer `Z(s)` gives
`π*(a|s) = (1/Z(s))·π_β(a|s)·exp(A(s,a)/λ)`. The optimal constrained policy is just the behavior policy
reweighted by the exponentiated advantage. `λ` is the temperature: small `λ` sharpens toward the
highest-advantage actions, large `λ` flattens toward `π_β`. This is worth pausing on — it says the
Lagrangian solution *already* respects support automatically, because `π*` is `π_β` times a positive
weight, so wherever `π_β` is zero `π*` is zero, no matter how large the advantage. The support constraint
falls out of the KL for free. The behavior policy is still sitting in the formula, though; the projection
step is where it has to die if I want the model gone.

Project this non-parametric `π*` onto my parametric policy `π_θ`. The direction of the projection makes
or breaks everything, so I check both. Minimize the *forward* KL `KL(π* ‖ π_θ)` over data states:
`argmin_θ E_{a~π*}[−log π_θ(a|s)]` (only that term depends on `θ`). Now the key move — I cannot sample
`π*` directly, but `π*` is just `π_β` reweighted, so importance-sample from the buffer:
`E_{a~π*}[−log π_θ] = E_{a~π_β}[(π*/π_β)(−log π_θ)] = E_{a~π_β}[(1/Z(s))exp(A/λ)(−log π_θ)]`. The `π_β`
factor *cancels* — `π*/π_β = (1/Z)exp(A/λ)`, no behavior model left. So the actor update becomes a
weighted maximum likelihood on samples drawn straight from the buffer:
`θ_{k+1} = argmax_θ E_{(s,a)~β}[log π_θ(a|s)·exp(A^{π_k}(s,a)/λ)]`. This is supervised learning — each
observed `(s,a)` weighted by its exponentiated advantage. No behavior model anywhere, and the constraint
is enforced *implicitly*: by reweighting the buffer's own actions, the update can never put mass on an
action the data did not contain, yet it concentrates mass on high-advantage ones. This is the third
requirement met — and crucially, it is the fix for the SPOT collapse, because the improvement step now
*reweights logged actions* instead of *maximizing `Q` at a self-proposed action*. The actor never asks
the critic "what is the value of this new action I invented?" during improvement; it only asks "of the
actions I actually saw, which were good?" That is the OOD-query the deterministic maximizer could not
avoid, and it is gone.

Contrast the *reverse* KL `KL(π_θ ‖ π*)` to be sure forward was right, because if reverse also removed
the model I would prefer it for its mode-seeking sharpness. Reverse KL =
`E_{a~π_θ}[log π_θ − log π_β − A/λ + log Z]`, which needs two things I am fleeing: it evaluates
`log π_β` (a density model — exactly the brittle VAE SPOT had to keep, the thing that was too tight on
expert and too broad on cloned) and it samples actions from `π_θ`, which offline are the possibly-OOD
actions that make `Q` extrapolate (exactly the hammer-expert collapse mechanism). Forward KL needs
neither: it samples from the buffer, not from `π_θ`, and the `π_β` factor cancels instead of appearing as
`log π_β`. So the direction that lets me sample from the buffer and cancel `π_β` is *also* the direction
that removes both failure mechanisms I just saw in the SPOT numbers. That is not a coincidence I am
comfortable leaving unremarked — the two properties I need (no density model, no OOD query) are two faces
of the same choice of projection direction, which is why forward KL is not a tuning preference but the
load-bearing decision of this rung.

The per-state normalizer `Z(s) = E_{a~π_β}[exp(A/λ)]` still sits in the weight. Do I need it? Estimating
it — say `K = 10` sampled actions per state to form `(1/K)Σ exp(A/λ)` — empirically *hurts*, because that
denominator is a small-sample estimate of an exponential-of-advantage expectation, which is exactly the
high-variance regime: a single large-advantage sample dominates the sum and the reciprocal swings
wildly, so dividing by it injects the variance of a degenerate importance weight into every gradient. And
there is a clean argument it is benign to drop: `Z(s)` is a per-*state* factor, so it only reweights how
much different states count, not how actions compete *within* a state, and the action competition is the
whole point of the update. The buffer's state distribution is already off from what `π_θ` will visit, so
faithfully preserving a per-state scale is low value anyway. Drop `Z(s)`; normalize the weights across
the minibatch instead, which controls the gradient scale without the reciprocal-variance pathology. The
trade is honest: I give up the exact per-state scaling of the constrained solution in exchange for a
gradient whose variance does not blow up on the rare high-advantage transitions, and since those rare
transitions are precisely the ones the sharp temperature already leans on, keeping their gradient
well-behaved matters more than preserving a per-state constant I could never estimate cleanly anyway.

Put a number on the temperature to be sure it does something sane. I set `awac_lambda = 0.1` and clamp
the weight `exp(adv/λ)` at `100`. The clamp saturates when `adv/0.1 = ln 100 = 4.6`, i.e. at an advantage
of about `0.46`: any transition whose advantage exceeds ~0.46 gets the same maximal weight of 100, and
transitions with advantage near zero get weight near 1. So `λ = 0.1` is a *sharp* temperature — it treats
"clearly above the state's value" as an all-or-nothing signal and concentrates the maximum-likelihood
pull onto the top slice of logged actions. That sharpness is deliberate on `cloned` data, where I want the
update to ignore the many mediocre noisy actions and clone hard onto the good ones — but I flag it as the
place this could go wrong, and I will come back to it in the expectations.

It is worth being explicit about what the batch normalization of weights does to the saturation I just
computed, because the two interact. Suppose in a 256-wide minibatch a handful of transitions have
advantage above 0.46 (weight clamped to 100) and the rest sit near zero (weight near 1). Normalizing the
weights to sum to the batch size hands almost the entire gradient budget to that handful — if ten
transitions saturate at 100 and 246 sit at 1, the saturated ten carry `1000/(1000+246) ≈ 80%` of the
pull. So the effective behavior of `λ = 0.1` plus the clamp plus batch normalization is a *top-k style*
update: clone hard onto the best few logged actions in each batch. On clean expert data every logged
action is good, so the advantages are flat and the update spreads evenly — near behavior cloning, which
is exactly right on expert demos. On noisy `cloned` data the advantages are spread out and the update
becomes genuinely selective. The single failure mode of this design is now sharp to name: it depends
entirely on the critic's ranking of logged actions being correct, because the update trusts the top slice
absolutely.

That dependence is why the critic — the policy-evaluation half — is the second deliberate departure from
the SARSA/AWR lineage. I want efficiency, so I bootstrap `Q^π` of the *current* policy off-policy
(twin-Q with a `min` target and a Polyak target network to control overestimation), not a Monte-Carlo
`V^{π_β}` of the behavior policy. The Monte-Carlo route only supports one step of improvement away from
`π_β` and is slow; the bootstrapped `Q^π` reuses off-policy data and improves iteratively, which is what
lets the online phase keep climbing rather than stalling near the offline policy. The advantage the actor
needs is `A(s,a) = Q(s,a) − V(s)`, and since `V(s) = E_{a~π}[Q(s,a)]` I estimate it by evaluating the
(min of the twin) critic at an action *sampled from the current policy*. Note this `V` estimate does
sample `π`, but it is only used to compute a *baseline subtraction inside the weight* — it never feeds an
optimization that pulls the policy toward a high-`Q` invented action, so it does not reopen the SPOT
failure. Trace it: the actor gradient is `∇_θ E[log π_θ(a|s)·w]` with `w` detached and `a` from the
buffer, so `V`'s only influence is on the scalar `w`; even if `V` is a little wrong, it rescales the
weight on a *logged* action, it never proposes a new one. The improvement direction stays pure
weighted-MLE on logged actions.

The twin-Q `min` target and the Polyak averaging both feed the same goal of keeping that ranking honest,
and they matter more here than in the deterministic case because AWAC's actor never fights the critic —
so a mis-ranked critic is never corrected by the policy stepping somewhere and getting punished; the
policy just faithfully clones the critic's mistake. The `min` of the twin critics on the bootstrap
target biases the value estimate downward, which counters the standard positive bias of a `max`-style
backup; even though AWAC's backup is a policy-action expectation rather than a hard max, the target still
evaluates `Q` at `π(s')`, a self-sampled action that can drift OOD, so the `min` is the guard on exactly
that drift. The Polyak coefficient `tau = 5e-3` means the target network is a slow exponential average
with an effective horizon of about `1/tau = 200` updates, so the bootstrap target moves two orders of
magnitude slower than the online critic — slow enough that the regression has a near-stationary target
and the ranking does not thrash from step to step. These are the same values the deterministic rung used;
what changed is that the actor no longer probes the critic, so the critic's own overestimation controls
are now the *only* thing standing between a mis-valued action and the cloned policy.

One more property of the Adroit rewards shapes how much I can trust the advantage. These manipulation
tasks give most of their return near task completion — the informative signal is concentrated late in a
trajectory, and much of the per-step reward is small shaping. That means the advantage `A(s,a)` I
estimate is a difference of two noisy critic outputs, `Q(s,a) − V(s)`, both of which are dominated by the
bootstrapped tail; on a good transition early in a successful trajectory the immediate reward is
uninformative and the whole advantage rests on the critic having correctly propagated the late payoff
backward. This is fine when the critic *can* propagate — expert data, dense good trajectories — and it is
exactly where I worry on hammer-cloned, because there the good late payoffs are rare and the backward
propagation has to survive many noisy transitions. It reinforces the boundary I am drawing: AWAC's
advantage is only as good as the critic's ability to carry a sparse late reward back to the states where
the update needs it, and the update's `λ = 0.1` sharpness then bets everything on that carried signal.

In this task's harness the AWAC fill is specific and I want it exact. The actor is a 3×256 Gaussian with
a *state-independent* `log_std` (an `nn.Parameter`, clamped to `[−20, 2]`), using a plain `Normal`
distribution with a hard action clamp — not the template's `TanhTransform` Gaussian. The state-independent
`log_std` is a small but real choice: the update is a weighted `log_prob` on dataset actions, and a
state-dependent variance head would give the network a cheap way to lower the loss by inflating variance
on hard-to-fit states rather than by moving the mean toward the good actions, so a single global
`log_std` keeps the pressure on the mean where the improvement signal lives. The critics are 3×256 and
return `(batch, 1)` *unsqueezed* (the template squeezes), each with its own Adam optimizer. The advantage
weight is `clamp_max(exp(adv/λ), 100)` with `awac_lambda = 0.1`, and the actor loss is
`(−log_prob · weight).mean()` on the *dataset* actions. The target critics get the standard Polyak
`soft_update` each step. And the decisive structural point for this task: `on_online_start` is a *no-op*.
AWAC needs no special handling at the transition, because the same advantage-weighted update runs
unchanged offline and online — the buffer simply grows with online data and the identical
`log_prob·exp(adv/λ)` update keeps running. That is exactly the seamless property I wanted, and it is
*more* seamless than SPOT's: SPOT achieved a seam-free transition by engineering it — cooling `λ`,
resetting optimizers, freezing a VAE — whereas AWAC has no transition machinery at all. There is no
schedule to mis-tune at the handoff, which removes an entire class of failure. Given that SPOT's cooling
schedule was one of the suspects in the expert collapse (the cooled `λ` opening the tube), removing the
schedule outright is itself a hypothesis I am testing.

What do I expect against the SPOT numbers, and what would falsify the premise? My falsifiable claim is
that removing the deterministic-maximizer OOD query stops the hammer-expert collapse: AWAC should
*retain* the clean expert competence rather than fall to the 2.5 I saw, because its actor never invents
an action to maximize `Q` against — I expect hammer-expert to recover into the strong (100+) range, and
because the failure there was systematic-not-variance, I expect the recovery to be tight across seeds
too. On Pen-cloned I expect to roughly match SPOT's mid-range (the implicit constraint should hold, but
`cloned` data is noisy so I do not expect a blowout), and I would be glad to shrink SPOT's 23-to-85
spread even if the mean moves little. The real risk I am taking — and the place AWAC could lose to SPOT —
is on *hammer-cloned*: recall `λ = 0.1` saturates the weight at an advantage of ~0.46, so the
advantage-weighted update concentrates hard on a tiny top slice of logged actions. On a dataset where the
good demonstrations are rare and buried in noise, if the critic cannot reliably separate the rare good
actions from the noise, those saturated weights land on the wrong transitions or smear uselessly, and the
policy never finds the manipulation behavior at all — worse, potentially, than SPOT's one-lucky-seed
55.8. If hammer-cloned comes back near zero on every seed, that is the signal: the implicit constraint is
*too* conservative on heavily-noisy mixtures, and the next rung must keep the OOD-safe value training but
give the critic a stronger improvement signal — a value function that does in-support dynamic programming
across the noisy trajectories rather than only reweighting the behavior policy's own logged actions. That
is the falsifiable boundary I am drawing between this rung and the next. The full scaffold module is in
the answer.
