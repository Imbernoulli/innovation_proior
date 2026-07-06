I am starting the ladder at the rung that takes the cleanest stance on the one failure that dominates
this whole task: the offline→online Q-collapse. Before I reach for anything cleverer, I want to nail down
*why* a plain off-policy actor-critic blows up on a fixed batch, and then pick the most direct cure,
because the directness is exactly what makes this a good first measurement — it isolates one mechanism so
the numbers tell me whether that mechanism is the binding constraint here.

Write down the disease precisely. The scaffold default is a twin-critic actor-critic: critics fit the
Bellman target `r + γ Q_target(s', π(s'))`, and a deterministic actor maximizes `Q(s, π(s))`. The
poison is in that target. `π(s')` is the action the *actor* chose, not one the dataset contains. On the
Adroit `cloned-v1` mixtures — expert demos diluted with noisy ones — the behavior policy covers only a
thin sliver of the action cube. Concretely the action dimensions here run 24 (Pen), 26 (Hammer), 28
(Door); even the smallest is a 24-dimensional box, and a mixture of a few thousand demonstrations cannot
begin to fill it. If each action coordinate were resolved to a coarse ten bins, the cube has `10^24`
cells and the data touches a vanishing fraction, so `π(s')` routinely lands in a cell the batch never
visited. There the critic is a neural net extrapolating off its training manifold, and that extrapolation
is essentially unbounded; worse, it skews *upward*, because the actor is a maximizer and is actively
attracted to whichever out-of-distribution action the critic happens to over-value. Those inflated values
get bootstrapped back through the backup into neighboring states, the next actor update chases them
harder, and the value function runs away. Offline there is no environment to punish the silly action and
correct the critic, so nothing stops the divergence. The first-principles conclusion: the cure is not
more capacity or a better optimizer — it is *forbidding the actor from leaving the region where the
behavior policy actually put mass.*

What is that region? The principled answer is the **support** of `π_β`: in state `s`, allow only actions
with `π_β(a|s) > ε`. There is a clean reason to trust this rather than treat it as a hack. If I restrict
the maximization in the Bellman operator to the support set and call that backup `T_ε`, its fixed point
`Q*_ε` is the supported optimal value, and I can bound how far it sits from the true `Q*`. Writing
`α(ε) = ‖T Q* − T_ε Q*‖_∞` for the worst-case one-step gap and using that `T_ε` is a γ-contraction in
sup-norm, the triangle inequality gives `‖Q* − Q*_ε‖_∞ ≤ γ‖Q* − Q*_ε‖_∞ + α(ε)`, hence
`‖Q* − Q*_ε‖_∞ ≤ α(ε)/(1−γ)`. Put the number in: `γ = 0.99`, so `1/(1−γ) = 100`. The one-step
restriction cost `α(ε)` is amplified a hundredfold at the fixed point. That factor of 100 is not a reason
to abandon the constraint — it is a reason to keep `α(ε)` genuinely small, i.e. to make the support set
lose as little of the truly-optimal action as possible. It also tells me the dial's shape: a tighter
support (bigger `ε`) shrinks extrapolation risk but raises `α(ε)` and hence a value gap that gets
multiplied by 100; a looser one does the reverse. `ε` is the single knob trading "stay where the critic
is trustworthy" against "keep enough room to be good," and because of that ×100 amplification I want the
loosest support that still excludes the actions the critic cannot be trusted on — not an aggressive
clamp. Confinement is safe; the only question is how to enforce it.

Now the design fork that decides which method this rung is. Two camps enforce support. The
parameterization camp (BCQ and descendants) bakes it into the architecture — fit a generative model of
`π_β`, sample candidate actions, argmax `Q` over them — which respects density by construction. Walk that
one a few steps before I reject it, because it is genuinely tempting: it never even *forms* an OOD action,
so the collapse I just diagnosed cannot happen. But cost it out against this harness. Selecting an action
means sampling `N` candidates from the generative model and running `N` critic evaluations to argmax; the
online phase is `10^6` environment steps, each of which calls `select_action`, so with a typical `N` of
ten to a hundred I pay ten-to-a-hundred generative decodes plus that many critic forward passes *per env
step* — a one-to-two-order-of-magnitude inflation of the online inner loop, on top of the offline
argmax. And the offline→online swap becomes awkward: the "policy" is an argmax procedure, not a network I
can smoothly relax, so there is no single coefficient that turns the offline-conservative version into a
well-grounded online one. The regularization camp keeps the policy a plain network and adds a penalty
pulling it toward `π_β` — pluggable, one forward pass at inference, and — the property I care about most —
a single coefficient whose value defines "how offline-conservative am I." I want that. But here is the
wall every regularizer hits: it penalizes a *divergence* from `π_β` (BEAR's MMD, TD3+BC's `(π(s)−a)²`
cloning term), and a divergence measures distributional *closeness*, not *support*. The support condition
only asks "put your action where `π_β` has density above `ε`" — it says nothing about matching `π_β`'s
shape. On the `cloned` mixtures, where `π_β` is broad and multimodal, forcing `π` close to it in
distribution drags the policy toward the noisy demonstrations even when the data plainly supports a
sharper, stronger policy. So I am cornered: parameterization gets the right notion but is intrusive and
slow online; regularization is pluggable but enforces the wrong notion.

The mismatch is the clue. The reason every regularizer is "indirect" is that it measures a distance
between *distributions*, but the constraint I actually wrote is a statement about the *value of the
behavior density at the single action the policy takes*: `π_β(π(s)|s) > ε`. So I should stop measuring a
divergence and instead just evaluate the behavior density at `π(s)` and require it be large. As an
optimization over the policy, with a log so the density enters additively,
`max_φ E_{s~D}[Q(s, π_φ(s))]` subject to `min_s log π_β(π_φ(s)|s) > log ε`. The per-state `min` is an
infinite family of constraints over a continuous state space, so I relax it the standard way (as TRPO
and AWR do for their constraints) to an average over the data states,
`E_{s~D}[log π_β(π_φ(s)|s)] > ε̂`, and Lagrangianize it into a single penalty on the actor loss:
`J_π(φ) = E_{s~D}[ −Q(s, π_φ(s)) − λ·log π_β(π_φ(s)|s) ]`. This is pluggable — an extra term on the
standard actor loss, policy still a plain network — yet the penalty *is* the behavior log-density, a
direct support constraint rather than a divergence. `λ` plays the role of `ε`: large `λ` pins the policy
onto high-density actions, small `λ` lets the value term lead. And it has the property parameterization
could not give — set `λ` to zero and the penalty vanishes, so a single scalar interpolates between
"fully offline-conservative" and "plain value-maximizing," which is precisely the axis the offline→online
handoff lives on.

Check the coefficient at the other extreme too, because a knob is only trustworthy if I understand both
ends. As `λ → ∞` the value term becomes negligible and the actor minimizes `neg_log_beta` alone — it
drives `π(s)` to the action that maximizes the behavior log-density, i.e. the mode of the VAE at `s`.
That is behavior cloning onto the densest mode. So the single coefficient sweeps a genuine spectrum: at
`λ = 0` plain value-maximizing TD3, at `λ = ∞` mode-seeking behavior cloning, and in between a policy
that maximizes value *subject to* staying where `π_β` had mass. The offline→online handoff is a walk
along that spectrum from conservative toward value-driven, which is exactly the shape the cooling
schedule implements. Knowing both endpoints reassures me the interior is well-behaved rather than a
region where the two gradients fight incoherently — at any fixed `λ` the loss is a fixed convex-ish
combination, not a schedule-dependent surface.

I do not have `π_β` in closed form, only samples (the dataset actions), so I must *estimate* the density
at arbitrary `(s, a)`, including the off-data actions the policy probes. The estimator has to capture
multimodal behavior — a single Gaussian would smear the `cloned` mixture's modes into one blob and call
the valley between them "in support," which is exactly the failure I am trying to avoid: an admitted
mid-valley action is an OOD action the estimator wrongly blesses. A conditional VAE is the natural fit:
`π_β(a|s) ≈ p_ψ(a|s)` with a fixed `N(0,I)` prior, the standard flexible behavior model in offline RL.
The marginal is intractable, so I use the evidence lower bound: introduce an approximate posterior
`q_φ(z|a,s)` and, by Jensen,
`log p_ψ(a|s) ≥ E_{q_φ}[log p_ψ(a|z,s)] − KL(q_φ(z|a,s) ‖ p(z|s)) =: −L_ELBO`. Two things make this the
right object. First, `−L_ELBO` is a genuine *lower* bound (its gap is a nonnegative KL), so substituting
it for `log π_β` in a constraint I want *above* a threshold is conservative in the right direction — if
the bound is high, the true density is at least that high; I never falsely admit an action because the
bound over-states its density. Second, I can keep the KL analytic. With a Gaussian posterior
`q = N(μ, σ²)` and unit prior the KL is the closed form `−½ Σ_j (1 + log σ_j² − μ_j² − σ_j²)`, and if I
draw a single latent sample for the reconstruction term the whole ELBO is a low-variance, one-sample
Monte-Carlo estimate whose only stochastic part is the reconstruction. So the density penalty is
`neg_log_beta = L_ELBO(s, π(s)) = recon + β·KL`, with reconstruction `mean((decode(s,z) − a)²)` and
`β = 0.5` down-weighting the KL so the VAE spends capacity on faithful reconstruction rather than on
squeezing the latent to the prior. This is both what I train the VAE on and what I evaluate at `π(s)`.

Trace the shape through `_elbo_loss` once to be sure it does what I think. Inputs are a state batch
`(B, s_dim)` and the policy action `π(s)` of shape `(B, a_dim)`. Encoding gives `mean, std` of shape
`(B, z_dim)` with `z_dim = 2·a_dim`. I tile by `num_samples = N` and permute to `(B, N, z_dim)`, sample
`z` of the same shape, tile the state and target action to `(B, N, ·)`, decode to `u` of `(B, N, a_dim)`,
and reduce the squared reconstruction error over the `(N, a_dim)` axes to a per-element vector of shape
`(B,)`. The analytic KL reduces over `z_dim` to `(B,)`. So `neg_log_beta` is `(B,)` — one scalar penalty
per state — exactly the object I average in the actor loss. With `N = 1` this is the cheapest honest
estimate; more samples would only shave the reconstruction variance, and I do not need it because the
penalty is averaged over a 256-wide minibatch anyway.

Two VAE sizing details follow from the same "faithful density" requirement. The latent dimension is
`2·action_dim` — twice the action width rather than equal to it — because the `cloned` behavior is a
mixture, and a latent wide enough to carry a few modes' worth of structure keeps the decoder from
averaging distinct demonstration styles into one blurred action; a latent equal to the action dimension
would be a tight bottleneck that forces exactly that averaging. And I clamp the encoder `log_std` to a
wide `[−4, 15]` rather than the policy's `[−20, 2]`: the encoder posterior can legitimately be very
sharp (`log_std → −4`, a nearly deterministic latent for a cleanly reconstructed action) or quite broad,
and clamping it too tightly at the low end would inject spurious latent noise into the reconstruction and
inflate `neg_log_beta` on in-support actions — corrupting the very quantity the actor reads. The VAE
optimizer runs at `1e-3`, an order of magnitude above the actor's `1e-4`, which is fine because the VAE
is trained to convergence *before* Phase 1 and then frozen; there is no interaction to destabilize.

The base off-policy algorithm: TD3, not SAC. My enemy is OOD overestimation, and TD3 was built to
suppress it — the `min` of twin critics caps the bootstrap, target policy smoothing keeps the critic
from latching onto a sharp spurious peak, and delayed (every-`policy_freq`) actor updates let the value
settle before the actor chases it. Make those concrete against this task. The twin `min` turns a
symmetric critic error into a systematically *negative* bias on the target, which directly opposes the
upward OOD skew that drives the runaway. Target smoothing adds clipped noise to `π(s')`: with
`policy_noise = 0.2·max_action` clipped to `±0.5·max_action`, the backup averages `Q` over a small ball
around the target action rather than reading a single point, so a needle-thin over-valued spike gets
blurred out instead of bootstrapped. Delaying the actor to every `policy_freq = 2` critic steps means the
critic takes two gradient steps toward a target for every one step the actor takes toward the critic,
which keeps the maximizer from racing ahead of a value estimate that has not yet stabilized. SAC is the
opposite of everything I just listed: its stochastic actor *samples* actions whose tails reach OOD and
pull in erroneous values, and its entropy bonus actively rewards spreading toward the support's edge —
exactly the behavior I am forbidding. So: deterministic actor, twin unsqueezed critics, target smoothing,
delayed updates.

One scale issue remains. The actor loss `−Q + λ·neg_log_beta` mixes two terms on different scales — `Q`
lives on the scale of returns (task-dependent; the Adroit reward scales differ across Pen, Door, Hammer),
`neg_log_beta` on the scale of a log-density — so a single `λ` would need re-tuning per task. I borrow
TD3+BC's normalization: divide the value term by the detached mean `|Q|` over the minibatch,
`norm_q = 1/|Q|.mean().detach()`, so the actor loss becomes `−norm_q·Q.mean() + λ·neg_log_beta.mean()`.
Because `norm_q` is detached it changes the *magnitude* of the value gradient — pinning it to order 1
regardless of reward scale — without touching the *direction*, so within a batch the action the actor
prefers is unchanged; only the trade-off weight against the density penalty is stabilized. That is what
lets one `λ` transfer across the three tasks despite their different return scales.

Now the two structural facts the offline→online setting needs, and I want to verify the first by hand
rather than assert it. Set `λ = 0`. The actor loss collapses to `−norm_q·Q.mean()`, whose gradient is
`−norm_q·∇_φ Q(s, π_φ(s))` — precisely the deterministic policy-gradient ascent of ordinary TD3, up to
the harmless positive scalar `norm_q`. The critic loss, target smoothing, twin `min`, delayed updates
are all already plain TD3. So at `λ = 0` this rung *is* online TD3, with no architectural seam between
the offline algorithm and a well-grounded online one — that is the seamless-transition property a
parameterization could never give, and it is not a hope but an identity I just checked. **Second**, how I
use it: offline I pretrain with a strong constraint (large `λ`) to keep the critic honest on the fixed
data, but once fresh online interactions arrive, the data distribution shifts toward the policy's own
actions — exactly the actions the critic can now learn about from real feedback — so the reason for the
constraint erodes. Holding `λ` fixed would pin the policy near the *offline* support and cap online
improvement. So I *cool* `λ` linearly over the online phase toward a small floor `λ_end`,
`λ_t = λ·max(λ_end, 1 − t_online/10⁶)`. Read the schedule literally: starting from `λ = 1.0` with
`λ_end = 0.5`, the linear term `1 − t/10⁶` hits `0.5` at `t = 5·10⁵`, so the constraint relaxes from full
strength to half strength over the first 500k online steps and then holds at the floor for the second
500k. I keep the floor at half rather than going to zero because on the hardest tasks bootstrap error
stays dangerous even online — a residual constraint guards the critic mid-finetune — and because the
×100 fixed-point amplification I computed earlier means even a small residual OOD leak can still hurt, so
I do not want the support notion to vanish entirely while the online buffer is still mostly old offline
data. And I *freeze the VAE* during online fine-tuning: behavior models chase a moving, policy-dependent
target online, so re-fitting would destroy the stable "where the offline data was" notion; the decaying
`λ` alone controls how much that notion still binds. The VAE is trained once, before Phase 1, in the
harness's optional `pretrain` hook.

In this task's harness specifically, a few details differ from the generic recipe and I want them
right. The VAE trains in `pretrain(replay_buffer, batch_size)` for `10⁵` steps and is then frozen
(`self.vae.eval()`), so the constraint is stationary from the first policy step. The actor and critic
head weights are initialized small (uniform ±0.001 on the actor head, ±0.003 on the critic heads,
matching the reference) so the policy starts near-zero-action and the critics start near-zero-value,
which keeps the early offline backups gentle — a critic that starts near zero cannot over-value an OOD
action on step one, buying the constraint time to take hold. At `on_online_start` I reset all three
optimizers (Adam moment estimates accumulated over 1M offline steps would otherwise inject stale momentum
into the first online updates) and set the online discount. The parameter budget is worth doing on paper,
because the context caps total trainable parameters at ~1.2× the largest baseline and I want to know how
much room the VAE eats. The VAE is 750-wide (3-layer encoder/decoder, latent `2·action_dim`). For Hammer
(`s_dim = 46`, `a_dim = 26`, `z_dim = 52`) the encoder shared trunk is `Linear(72, 750)` plus
`Linear(750, 750)` ≈ `54.7k + 563k`, the two heads `Linear(750, 52)` add ≈ `39k` each, and the decoder
`Linear(98, 750) + Linear(750, 750) + Linear(750, 26)` ≈ `74k + 563k + 20k`, totalling roughly `1.35M`
parameters. By contrast a single 3×256 critic is about `Linear(72,256)+2·Linear(256,256)+Linear(256,1)`
≈ `150k`, and the deterministic actor ≈ `85k`, so the two trainable critics plus actor come to ~`0.39M`.
The VAE alone is over three times the entire policy-plus-critic stack and is by a wide margin the largest
network in the whole task — which is precisely why the budget is set to 1.2× a conservative-value plus
VAE architecture. I am sitting at the edge of the cap, so there is deliberately no room to add capacity;
whatever this rung buys has to come from the algorithm, not from width. The actor LR is dropped to `1e-4`
(slower, more conservative actor than the critic's `3e-4`, so the value estimate leads the policy) and
`normalize = False` is set in `CONFIG_OVERRIDES`, matching the reference SPOT config for Adroit.

What do I expect this rung to deliver, and what would falsify the premise? If the support constraint is
the binding fix, I expect stable offline pretraining and a *monotone* (not collapsing) online curve on
all three tasks — the whole point is no early Q-collapse. The risk I am explicitly taking is the
constraint's conservatism on `cloned` data: the VAE's support of a noisy expert+random mixture is broad,
so the density penalty may admit too much and the critic may still drift, *or* the cooling schedule may
relax `λ` faster than the critic stabilizes on the held-out expert variant where there is no recovery
margin. Concretely I expect Pen-cloned to land in a respectable but not dominant range and Hammer-cloned
to register meaningfully (the support constraint should let the policy find the few good demonstrations),
but I am genuinely worried about the expert held-out variant: if the VAE's support is too permissive,
the deterministic TD3 actor could ride an over-valued OOD action straight off the manifold there and
collapse to near-zero — and that is exactly the kind of measurement I want first, because if the
*directest* support method collapses on one variant, it tells me the next rung must attack the value
function's *over-estimation directly*, not just fence the policy's support. The full scaffold module is
in the answer.
