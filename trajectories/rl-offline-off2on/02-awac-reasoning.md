The SPOT numbers tell a sharp story, and I want to read them carefully before deciding what to change.
Pen-cloned landed at a mean of 56.4 (61.5 / 23.1 / 84.6 across seeds) — respectable but seed 123 fell to
23, so the support constraint is not protecting every run equally. Hammer-cloned came in at 19.7, driven
almost entirely by one lucky seed (55.8) with the other two near zero (1.9, 1.3). And the held-out
expert variant — hammer-expert — *collapsed* to 2.5, near random, despite the offline data there being
clean expert demonstrations. That last number is the one that diagnoses the method. On expert data the
VAE's support is *tight*, so the density penalty should keep the deterministic TD3 actor exactly on the
demonstrated trajectory — and instead the policy went to nearly zero return. This is precisely the
failure mode I flagged: a deterministic actor maximizing `Q` rides whatever OOD action the critic
over-values, and on expert data — where the support is a thin tube and the cooled `λ` opens it during the
online phase — once the actor steps a little off that tube the critic has no real feedback to correct it
and the value runs away. The collapse on the *cleanest* data is the tell that the binding problem is not
the policy's support per se but the *deterministic maximizer + bootstrapped critic* exposing itself to
OOD values during improvement. So the move is to keep the implicit-constraint idea but stop letting the
actor's improvement step ever query `Q` at a self-proposed action at all.

Let me restate what I actually need, because SPOT got two of three requirements right and I want to keep
those. I need an off-policy critic (data efficiency — the online budget is real but the offline data must
be reused throughout, which SPOT's TD3 critic does). I need a policy constraint for offline stability
(SPOT had one). And I need a constraint that does *not* require keeping an explicit behavior model
accurate during the shifting online phase — and here is where SPOT is brittle: it leans on a VAE whose
support, frozen, is either too tight (expert: any deviation is unguarded once `λ` cools) or too broad
(cloned: it admits the noise). I want the constraint to come for free from the *update math*, not from a
separately-fit density model. That is the gap the next rung closes.

Write the constrained improvement problem and solve it exactly, and watch whether the behavior model can
be made to disappear. The advantage `A(s,a) = Q(s,a) − V(s)` is what improvement pushes up (maximizing
`E_π[Q]` equals maximizing `E_π[A]` since `V` is action-independent), so at iteration `k`:
`π_{k+1} = argmax_π E_{a~π}[A^{π_k}(s,a)]` subject to `KL(π(·|s) ‖ π_β(·|s)) ≤ ε` and normalization.
Form the Lagrangian with multiplier `λ` on the KL and `α` on normalization, write the KL as
`∫ π(log π − log π_β)`, and differentiate with respect to `π(a|s)` at a single action: the
`E_π[A]` term gives `A(s,a)`, the `−λ·KL` term gives `−λ(log π − log π_β + 1)` (the `+1` from
differentiating `π log π`), the normalization gives `−α`. Setting the sum to zero and solving for
`log π` yields `log π(a|s) = (1/λ)A(s,a) + log π_β(a|s) − 1 − α/λ`; exponentiating and folding the
action-independent constants into a per-state normalizer gives
`π*(a|s) = (1/Z(s))·π_β(a|s)·exp(A(s,a)/λ)`. The optimal constrained policy is just the behavior policy
reweighted by the exponentiated advantage. `λ` is the temperature: small `λ` sharpens toward the
highest-advantage actions, large `λ` flattens toward `π_β`. The behavior policy is still in there — the
projection step is where it has to die.

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

Contrast the *reverse* KL `KL(π_θ ‖ π*)` to be sure forward was right. Reverse KL =
`E_{a~π_θ}[log π_θ − log π_β − A/λ + log Z]`, which needs two things I am fleeing: it evaluates
`log π_β` (a density model — exactly the brittle VAE SPOT had to keep) and it samples actions from `π_θ`,
which offline are the possibly-OOD actions that make `Q` extrapolate (exactly the hammer-expert
collapse). Forward KL needs neither. So the direction that lets me sample from the buffer and cancel
`π_β` is also the direction that removes both failure mechanisms I just saw.

The per-state normalizer `Z(s) = E_{a~π_θ}[exp(A/λ)]` sits in the weight. Do I need it? Estimating it
(say `K=10` samples per element) empirically *hurts* — the estimation error injects variance that
behaves like a degenerate importance weight. And there is a clean argument it is benign to drop: `Z(s)`
is a per-*state* factor, so it only reweights how much different states count, not how actions compete
within a state, and the buffer's state distribution is already off from what `π_θ` will visit, so
faithfully preserving it is low value. Drop `Z(s)`; normalize the weights across the minibatch instead.

That leaves the critic — the policy-evaluation half — and here is the second deliberate departure from
the SARSA/AWR lineage. I want efficiency, so I bootstrap `Q^π` of the *current* policy off-policy
(twin-Q with a `min` target and a Polyak target network to control overestimation), not a Monte-Carlo
`V^{π_β}` of the behavior policy. The Monte-Carlo route only supports one step of improvement away from
`π_β` and is slow; the bootstrapped `Q^π` reuses off-policy data and improves iteratively, which is what
lets the online phase keep climbing rather than stalling near the offline policy. The advantage the actor
needs is `A(s,a) = Q(s,a) − V(s)`, and since `V(s) = E_{a~π}[Q(s,a)]` I estimate it by evaluating the
(min of the twin) critic at an action *sampled from the current policy*. Note this `V` estimate does
sample `π`, but it is only used to compute a *baseline subtraction inside the weight* — it never feeds an
optimization that pulls the policy toward a high-`Q` invented action, so it does not reopen the SPOT
failure; the improvement direction is still pure weighted-MLE on logged actions.

In this task's harness the AWAC fill is specific and I want it exact. The actor is a 3×256 Gaussian with
a *state-independent* `log_std` (an `nn.Parameter`, clamped to `[−20, 2]`), using a plain `Normal`
distribution with a hard action clamp — not the template's `TanhTransform` Gaussian. The critics are
3×256 and return `(batch, 1)` *unsqueezed* (the template squeezes), each with its own Adam optimizer.
The advantage weight is `clamp_max(exp(adv/λ), 100)` with `awac_lambda = 0.1` (a sharper temperature
than IQL's `β`), and the actor loss is `(−log_prob · weight).mean()` on the *dataset* actions. The
target critics get the standard Polyak `soft_update` each step. And the decisive structural point for
this task: `on_online_start` is a *no-op* — AWAC needs no special handling at the transition, because the
same advantage-weighted update runs unchanged offline and online (the buffer simply grows with online
data). That is exactly the seamless property I wanted and the thing SPOT had to engineer with its
`λ`-cooling and optimizer resets. AWAC removes the transition-handling machinery entirely, which is its
own kind of robustness — there is no schedule to mis-tune at the handoff.

What do I expect against the SPOT numbers, and what would falsify the premise? My falsifiable claim is
that removing the deterministic-maximizer OOD query stops the hammer-expert collapse: AWAC should
*retain* the clean expert competence rather than fall to 2.5, because its actor never invents an action
to maximize `Q` against — I expect hammer-expert to recover into the strong (100+) range. On Pen-cloned I
expect to roughly match SPOT's mid-range (the implicit constraint should hold, but `cloned` data is noisy
so I do not expect a blowout). The real risk I am taking — and the place AWAC could lose to SPOT — is on
*hammer-cloned*: the advantage-weighted update reweights only logged actions, so on a dataset where the
good demonstrations are rare and buried in noise, the weights may concentrate on too few transitions and
the policy may fail to find the manipulation behavior at all. If hammer-cloned comes back near zero, that
is the signal: the implicit constraint is *too* conservative on heavily-noisy mixtures, and the next rung
must keep the OOD-safe value training but give the critic a stronger improvement signal — a value
function that does in-support dynamic programming rather than only reweighting the behavior policy's own
actions. That is the falsifiable boundary I am drawing between this rung and the next. The full scaffold
module is in the answer.
