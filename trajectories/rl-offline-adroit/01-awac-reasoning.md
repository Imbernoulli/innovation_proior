Let me start from the exact shape of the problem this harness hands me, because the data is unlike the
locomotion offline benchmarks where most of these algorithms were tuned. The buffer holds the D4RL
`human-v1` dataset: roughly twenty-five human teleoperation trajectories on a 24-DoF hand, in an action
space of 24-to-30 dimensions. Twenty-five trajectories at a couple hundred steps each is on the order of
five thousand transitions total — a thin, near-expert tube of states and actions embedded in an enormous
continuous action space. I get one million gradient steps at batch 256, I never touch the environment,
and I am judged by the D4RL normalized score on Pen, Hammer and Door. Put the two numbers next to each
other and the regime is already stark: five thousand distinct transitions, and I will pass over each of
them roughly `1e6·256 / 5e3 ≈ 5·10^4` times. Every sample is seen fifty-thousand times. Whatever I fit,
I will fit to saturation, so the entire question is not "can the networks represent the demos" but "what
does the update do at every state-action the demos never visited," because that is where a hand with
twenty-four joints spends almost all of its reachable volume.

The two failure modes I have to navigate are sharp and opposite. If I lean on value-based RL the way I
would online, the critic's bootstrap will query actions the twenty-five demos never contain at a state,
the network will extrapolate there — almost always upward, because the actor is trained to maximize Q —
and the inflated value backs up through the Bellman recursion until the policy is chasing pure
extrapolation off the demo tube. If instead I lean entirely on imitation, I am capped at the
demonstrators and I throw away any chance of improving on them; on twenty-five trajectories that cap is
real but not catastrophic, which is exactly why behavior cloning is a serious competitor here and the
thing I have to beat. So the target is the cheapest algorithm that does *some* value-based improvement
over the demos while staying provably inside their support.

Before I commit to that, let me actually walk the alternatives that are in circulation, because two of
them are tempting and I want to eliminate them on arithmetic rather than taste. The prior offline fixes
— BCQ, BEAR, BRAC — all add an explicit constraint `D(π, π_β) ≤ ε`, but they buy it by *fitting a
behavior model* `π̂_β`: a conditional VAE that generates candidate actions (BCQ), or a density I take an
MMD/KL divergence against (BEAR/BRAC). Cost that out here. `π̂_β` has to model a distribution over a
~28-dim action conditioned on state, and I have five thousand samples to fit it. A conditional VAE with
its own encoder/decoder is itself a couple hundred thousand parameters trained on the same five thousand
transitions, and a constraint that pins my policy to a *bad* estimate of the data is strictly worse than
no constraint — I would be regularizing toward noise. Worse, BEAR's MMD and BRAC's KL both require
*sampling* from `π̂_β` and from `π_θ` and evaluating the kernel/divergence between the samples, which on
a 28-dim action needs many samples per state to be anything but noise. This is a hard density-estimation
problem nested inside the RL problem, and it is the exact thing the thinness of the data makes
untrustworthy. I want the stay-near-data behavior **without ever fitting a behavior model**.

The other temptation is CQL: skip the behavior model and instead add a conservative penalty that pushes
`Q` down on out-of-distribution actions, via a `logsumexp_a Q(s,a)` term estimated by sampling actions.
Walk that one step too. The logsumexp is an integral over the 28-dim action box; to estimate it I sample
`N` actions per state and forward the critic on each, so a batch of 256 becomes `256·N` critic
evaluations per step, and I still need a Lagrange multiplier `α` tuned to hold the penalty at a target
gap. It triples the critic compute, adds a tuning knob, and — the decisive objection — it *still queries
`Q` at off-support actions*, merely penalizing them afterward rather than never touching them. On data
this thin the penalty estimate is itself a high-variance sampled quantity. I want an update where the
actor improvement step never evaluates `Q` at a policy-proposed off-support action at all. That rules out
CQL and DDPG-style `∇_a Q` ascent together. So the design has narrowed to: a constraint that comes for
free from the data itself, no behavior model, no OOD sampling.

Let me write the constrained improvement problem and solve it exactly, to see whether the behavior model
can be made to disappear. I want to push up the advantage `A^{π_k}(s,a) = Q^{π_k}(s,a) - V^{π_k}(s)`
(maximizing `E_π[Q]` equals maximizing `E_π[A]`, since `V` does not depend on the action), subject to a
KL trust region around the behavior policy and a normalization constraint:

  `π_{k+1} = argmax_π E_{a~π}[ A^{π_k}(s,a) ]  s.t.  KL(π(·|s) ‖ π_β(·|s)) ≤ ε,  ∫ π(a|s) da = 1`.

The Lagrangian with multiplier `λ` on the KL and `α` on normalization, differentiated with respect to
the value `π(a|s)` at a single action and set to zero, gives `A(s,a) - λ(log π - log π_β + 1) - α = 0`.
Solving for `log π` and folding the action-independent constants into a per-state normalizer `Z(s)`:

  `π*(a|s) = (1/Z(s)) · π_β(a|s) · exp( A^{π_k}(s,a) / λ )`.

The optimal constrained policy is the behavior policy reweighted by the exponentiated advantage. `λ`,
the multiplier on the KL, is a temperature: small `λ` sharpens toward the highest-advantage actions
(aggressive improvement), large `λ` flattens toward `π_β` (cautious, BC-like). Let me sanity-check the
two limits before I trust the form. As `λ → ∞` the exponent goes to zero, `exp(A/λ) → 1` uniformly, and
`π* → π_β` — the constraint is so tight the policy just clones the data, which is exactly behavior
cloning. As `λ → 0` the exponentiation sharpens without bound and all mass concentrates on the single
highest-advantage action in the support, which is the greedy in-support improvement. So `λ` genuinely
interpolates BC (the safe competitor I must beat) and greedy improvement (the thing that would step off
support if it could), and any finite `λ` is a controlled step from cloning toward improvement. That is
precisely the dial I was looking for, and it confirms the derivation lands where the mechanics should.

The behavior model is still sitting in `π*`, though. The decisive step is the projection onto my
parametric actor `π_θ`, and the *direction* of the KL in that projection is what makes `π_β` cancel or
not. Project by minimizing the *forward* KL, averaged over the data states: `argmin_θ E_ρ[ KL(π* ‖ π_θ) ]
= argmin_θ E_ρ E_{a~π*}[ -log π_θ(a|s) ]`, since only the `-log π_θ` term depends on `θ`. I cannot sample
`π*` directly, but `π*` is just `π_β` reweighted, so importance-sample from the buffer instead:

  `E_{a~π*}[ -log π_θ ] = E_{a~π_β}[ (π*/π_β)·(-log π_θ) ] = E_{a~π_β}[ (1/Z(s)) exp(A/λ)·(-log π_θ) ]`.

The `π_β` factor cancels — `π*/π_β = (1/Z) exp(A/λ)`, no behavior model left. The actor update becomes a
*weighted maximum likelihood* on samples drawn straight from the buffer:

  `θ ← argmax_θ E_{(s,a)~buffer}[ exp( A^{π_k}(s,a) / λ ) · log π_θ(a|s) ]`.

This is supervised learning on the dataset's own actions, each `(s,a)` weighted by its exponentiated
advantage. The constraint is enforced *implicitly*: reweighting the buffer's actions can never put mass
on an action the data did not contain, yet it concentrates that mass on the high-advantage actions. No
behavior model anywhere, and — crucially for narrow human data — the actor never queries Q at a
policy-proposed off-support action during improvement. Let me verify the choice of KL direction is
load-bearing and not cosmetic, by writing the reverse one out. The reverse KL is `KL(π_θ ‖ π*) =
E_{a~π_θ}[ log π_θ - log π_β - A/λ + log Z ]`. That expectation is over `a ~ π_θ` — a policy-proposed
action, exactly the off-support query I am trying to avoid — and it contains an explicit `log π_β` term,
which is the behavior model I refuse to fit. So the reverse KL drags both diseases back in; the forward
KL is the only projection that lets me sample from the buffer and cancel `π_β`. This is the whole reason
the method exists in this form, and the check makes it concrete rather than asserted.

The per-state `Z(s) = E_{a~π_β}[exp(A/λ)]` in the weight I can drop and simply normalize weights across
the minibatch, or not normalize at all: it is a per-*state* factor, so it only reweights how much
different states count, not how actions compete within a state, and estimating it from the few samples of
each state in a minibatch injects variance like a degenerate importance weight. Dropping it leaves the
relative weighting of actions at a given state — the thing that actually drives the policy — untouched.

That leaves the critic. I want the advantage `A = Q - V` from an off-policy bootstrapped `Q^π` of the
*current* policy, because that is what makes the method improve past a single step: `Q^π` propagates
value along transitions rather than reading it off Monte-Carlo returns of the demos. I bootstrap a twin-Q
TD target with the `min` of the targets and a Polyak target update to keep overestimation in check: `y =
r + γ·min_i Q̄_i(s', a')` with `a' ~ π(·|s')`. The `min` of two critics biases the target toward
underestimation, which is self-correcting offline — the actor simply avoids actions it underrates instead
of chasing ones it overrates. And since `V(s) = E_{a~π}[Q(s,a)]`, I estimate it by evaluating the critics
at an action sampled from the current policy and taking the same `min`, so `A(s,a) = Q(s,a) - min_i
Q_i(s, a_π)` with `a_π ~ π(·|s)`. I should be honest that this `V` is a *single* stochastic sample, and
trace what that does to the weight. Suppose the true advantage of a demo action is zero but the
one-sample `V` estimate is off by `±0.1` because the critic's action-landscape at that state is bumpy on
thin data. Then `A` swings by `±0.1`, and at the temperature I am about to choose the weight
`exp(A/λ)` swings by a factor `exp(±0.1/λ)`. That amplification is the crux of the variance story, so let
me pin the temperature down before finishing the analysis.

Now I have to be honest about how this harness differs from the generic recipe, because the constraints
here are tighter than where the method was tuned and they cut against its strengths. First, this is a
*purely offline* run — one million gradient steps on a static buffer, no online phase at all. The
recipe's headline strength is that the same update flows from offline pre-training into online
fine-tuning without changing anything; that strength is simply unused here, and what is left is the
offline half, which on narrow human data is the part most exposed to the implicit-constraint being only
as good as the advantage estimate. Second, the harness fixes batch size at 256 and the hidden width at
256 across three layers; I cannot reach for the larger batch or the wider nets that smoothed the
advantage-weighting variance elsewhere. Let me count what that width actually buys me so I know the
regime: the actor is a 3×256 trunk, whose two interior `256×256` matrices are `65,536` parameters each,
plus an input map of order `state_dim·256 ≈ 45·256 ≈ 1.2·10^4` and an output map of order `256·28`, so
about `1.5·10^5` parameters; each critic is the same order, and with two critics the whole apparatus is
roughly `4.5·10^5` parameters trained on five thousand transitions — near ninety parameters per sample.
That over-parameterization is exactly why the implicit in-support constraint has to do the regularizing;
there is no capacity headroom to spend and, per the harness, none to be bought.

Third, and most consequential, the temperature. I set `awac_lambda = 0.1` to match the reference's
manipulation configuration, and I should read what `λ=0.1` means mechanically together with the weight
clip. I clip the exponentiated weight at 100 so a handful of huge advantages cannot dominate the loss;
solving `exp(A/0.1) = 100` gives `A = 0.1·ln 100 = 0.1·4.605 ≈ 0.46`, so any demo action whose advantage
exceeds about `0.46` is pinned at the same maximum weight of 100 — the weighting saturates there. Below
that, the dynamic range is enormous: `A=0.1 → e^1 ≈ 2.7`, `A=0.2 → e^2 ≈ 7.4`, `A=0.46 → 100`, while a
below-average action `A=−0.46 → e^{−4.6} ≈ 0.01`. So the weight ratio between a top-rated and a
bottom-rated demo action is on the order of `10^4`. Coupled with the single-sample `V`, the earlier
`±0.1` error in `A` becomes an `exp(±1) ≈ ×2.7` swing in a sample's weight — a small critic
miscalibration is exponentially magnified into which demos the actor imitates. That is a sharp,
high-variance reweighting on twenty-five trajectories, and I am choosing it with eyes open because it is
also what gives the method its improvement bite on the near-expert Pen data.

I also have to fix the policy parameterization to what this harness's actor needs, and it is *not* the
squashed Tanh-Gaussian the scaffold ships by default. Advantage-weighted MLE evaluates `log π_θ(a|s)` on
dataset actions; a TanhTransform distribution makes that log-prob awkward near the action-box boundary
(the demos of a dexterous hand sit near the limits of its joint range, and the Tanh log-det-Jacobian
blows up as the pre-squash argument grows). So I use a plain Gaussian over a 3×256 trunk with a
*state-independent* `log_std` parameter, clamped to `[-20, 2]`, and compute the log-prob directly with no
Tanh correction — the actor clamps its samples into `[-1,1]` but the density is a clean Normal. Making
`log_std` a single shared vector rather than a state-dependent head is itself a deliberate
data-frugality choice: a state-dependent head would add another `256·action_dim ≈ 7·10^3` parameters that
five thousand transitions cannot constrain, so I spend `action_dim ≈ 28` parameters on a global spread
instead and let the mean carry all the state dependence. The critic here returns its output *un-squeezed*,
shape `(batch, 1)`, because the advantage and TD arithmetic in this version keep the trailing dimension;
that is a small but real divergence from the squeeze-to-scalar pattern other rungs use, and it means the
MSE and the `min` all broadcast on `(batch,1)` consistently. I keep separate optimizers for the two
critics and turn on state normalization (`CONFIG_OVERRIDES = {"normalize": True}`), since whitening the
hand state — whose raw joint-angle and velocity channels live on very different scales — helps the MLPs
condition and matches how this baseline was run.

It is worth tracing what the weighted-MLE update actually does to the actor mean, because with a Gaussian
of fixed spread the objective collapses to something I can read off exactly. For a Normal with a shared
`σ`, `log π_θ(a|s) = -‖a - μ_θ(s)‖² / (2σ²) + const`, so maximizing `E[ w · log π_θ ]` with weights
`w = exp(A/λ)` is minimizing a *weighted squared regression* of the mean onto the demo actions:
`min_θ E[ w·‖a - μ_θ(s)‖² ]`. At a state `s` seen with several demo actions `a_i` and weights `w_i`, the
minimizer of the mean is the weighted average `μ*(s) = Σ_i w_i a_i / Σ_i w_i`. So the actor at each state
converges to a *convex combination of the demonstrated actions there*, tilted toward the high-advantage
ones — which is a second, independent confirmation that the policy can never leave the demo tube: a
convex mixture of in-support actions is in the convex hull of the data, and the weights only slide where
inside that hull the mean sits. It also makes the variance story concrete once more: it is the `w_i` that
move under seed noise, and they move exponentially, so the same weighted mean can shift materially between
seeds even though every action it averages is a fixed demo point.

One more harness detail I should fix now is the effective horizon, because it bounds how far value can
propagate and thus how much the bootstrapped `Q^π` can help the hard tasks. With `γ = 0.99` the geometric
horizon is `1/(1-γ) = 100` steps. Pen's task is essentially a stabilization that pays off quickly, well
inside that horizon; Hammer's nailing is a long ordered contact sequence where the reward for a good
early motion only materializes many steps later, near or beyond the `100`-step scale, so the value at the
*start* of the sequence depends on a long chain of bootstraps each of which passes through a
policy-proposed `a' ~ π(s')`. Every link in that chain is an off-support query the twin-`min` can only
damp, not eliminate. That is the structural reason I expect Pen to be reachable and Hammer not: the same
bootstrap that is one safe hop on Pen is a hundred compounding hops on Hammer.

Per step, then: sample `(s, a, r, s', done, â')` — the harness hands me the dataset's own next action
`â'`, but this method has no use for it, so I ignore it and bootstrap from `a' ~ π(·|s')` instead. Critic update every step: bootstrap `q_next = min(Q̄_1(s', a'),
Q̄_2(s', a'))` at `a' ~ π(·|s')`, target `y = r + γ(1-done) q_next`, MSE on both critics, step both. Actor
update every step: compute the advantage with detached critics, `weight = clamp(exp(A/λ), max=100)`, and
minimize `-(log π_θ(a|s)·weight).mean()`. Then soft-update both target critics at `tau = 5e-3`. The whole
thing is a standard actor-critic where the only non-standard piece is the advantage-weighted MLE actor,
which is what keeps it on the demo tube without a behavior model.

Two small implementation commitments follow from the shapes and matter enough to state. I keep a separate
Adam optimizer for each critic rather than one over the union of parameters: the two critics feed a single
summed MSE, but stepping them under distinct optimizers keeps their Adam moment estimates independent, so
the `min` genuinely averages two *decorrelated* value estimates rather than two heads that drift together
— decorrelation is the entire reason the twin-`min` damps overestimation, and sharing an optimizer would
quietly erode it. And at evaluation the actor must act deterministically: `act` returns the policy *mean*
when the module is in eval mode and only samples in train mode, because the ten evaluation rollouts should
report the mode of the learned Gaussian, not a fresh draw whose exploration noise would inject spurious
seed-to-seed spread on top of the training variance I already expect. The reported D4RL score is thus the
score of `μ_θ(s)`, the weighted-mean-of-demos policy the fixed-point analysis picked out.

What do I expect this to do on the three tasks, and where do I think it will leave value on the table —
the question the next rung will have to answer. Pen (`pen-human-v1`) is the most reachable task and the
one where advantage weighting over near-expert demos should do best, so I expect a respectable Pen score,
but with high seed-to-seed variance, because at `λ=0.1` on twenty-five trajectories the weights are a
sharp function of a noisy single-sample advantage — a seed whose critic happens to calibrate the demo
advantages will concentrate on the right actions and Pen flies; a seed whose critic is mis-calibrated
points the same `10^4`-range weights at the wrong demos and the policy is dragged off. Hammer
(`hammer-human-v1`) and Door (`door-cloned-v1`) I am pessimistic about: the advantage signal there has to
flow through a long, precise contact sequence, the implicit constraint keeps the policy welded to
twenty-five demonstrations of it, and with no online correction and no separate in-support optimism the
method has little room to improve past the average demo action. So my falsifiable expectation, stated
against the three metrics, is: a decent-but-volatile `pen-human-v1` mean with a wide per-seed spread,
near-floor `hammer-human-v1`, and near-zero `door-cloned-v1`. The weak spot is precise: the advantage here
is `Q - V` with `V` read off a single policy sample, so the "how good is this action relative to
alternatives" signal is exactly as noisy as one stochastic critic query, and that noise is exponentiated
by a factor that can hit `10^4`. The actor is pinned to demo actions by an MLE that has no explicit
machinery for *staying calibrated* when the data is this thin. So the honest limit of this rung is that
its steadiness rides entirely on how well one stochastic critic query happens to calibrate on a given
seed — which is exactly the fragility I expect the Pen spread to expose, and the first thing I would want
to attack if the numbers come back as volatile as I fear.
