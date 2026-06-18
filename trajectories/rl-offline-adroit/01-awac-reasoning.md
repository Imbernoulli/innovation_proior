Let me start from the exact shape of the problem this harness hands me, because the data is unlike the
locomotion offline benchmarks where most of these algorithms were tuned. The buffer holds the D4RL
`human-v1` dataset: roughly twenty-five human teleoperation trajectories on a 24-DoF hand, in an action
space of 24-to-30 dimensions. That is a thin, near-expert tube of states and actions embedded in an
enormous continuous action space. I get one million gradient steps at batch 256, I never touch the
environment, and I am judged by the D4RL normalized score on Pen, Hammer and Door. The two failure
modes I have to navigate are sharp and opposite. If I lean on value-based RL the way I would online, the
critic's bootstrap will query actions the twenty-five demos never contain at a state, the network will
extrapolate there — almost always upward, because the actor is trained to maximize Q — and the inflated
value backs up through the Bellman recursion until the policy is chasing pure extrapolation off the
demo tube. If instead I lean entirely on imitation, I am capped at the demonstrators and I throw away
any chance of improving on them; on twenty-five trajectories that cap is real but not catastrophic,
which is exactly why behavior cloning is a serious competitor here and the thing I have to beat.

So I want the cheapest algorithm that does *some* value-based improvement over the demos while staying
provably inside their support. Let me reason about what "inside the support" can mean mechanically,
because that is the whole design. The starting point everyone reaches for is an off-policy
actor-critic: train a critic by TD to estimate Q^π, and improve the actor by ascending Q. On this data
that improvement step is the poison — `argmax_π E[Q(s, π(s))]` and the bootstrap `a' ~ π(·|s')` both
evaluate Q at policy-proposed actions, and on a narrow human dataset those are precisely the
out-of-distribution actions with no data to correct them. The prior fixes (BCQ, BEAR, BRAC) all add an
explicit constraint `D(π, π_β) ≤ ε`, but they do it by *fitting a behavior model* `π̂_β` — a generative
model or a density I take a divergence against. On twenty-five narrow trajectories, fitting an accurate
`π̂_β` over a 30-dim action space is itself a hard density-estimation problem, and a constraint that
pins the policy to a *bad* estimate of the data is worse than no constraint at all. I want the
stay-near-data behavior **without ever fitting a behavior model**.

Let me write the constrained improvement problem and solve it exactly, to see whether the behavior
model can be made to disappear. I want to push up the advantage `A^{π_k}(s,a) = Q^{π_k}(s,a) - V^{π_k}(s)`
(maximizing `E_π[Q]` equals maximizing `E_π[A]`, since `V` does not depend on the action), subject to a
KL trust region around the behavior policy and a normalization constraint:

  `π_{k+1} = argmax_π E_{a~π}[ A^{π_k}(s,a) ]  s.t.  KL(π(·|s) ‖ π_β(·|s)) ≤ ε,  ∫ π(a|s) da = 1`.

The Lagrangian with multiplier `λ` on the KL and `α` on normalization, differentiated with respect to
the value `π(a|s)` at a single action and set to zero, gives `A(s,a) - λ(log π - log π_β + 1) - α = 0`.
Solving for `log π` and folding the action-independent constants into a per-state normalizer `Z(s)`:

  `π*(a|s) = (1/Z(s)) · π_β(a|s) · exp( A^{π_k}(s,a) / λ )`.

The optimal constrained policy is the behavior policy reweighted by the exponentiated advantage. `λ`,
the multiplier on the KL, is a temperature: small `λ` sharpens toward the highest-advantage actions
(aggressive improvement), large `λ` flattens toward `π_β` (cautious, BC-like). The behavior model is
still sitting in `π*`, though. The decisive step is the projection onto my parametric actor `π_θ`, and
the *direction* of the KL in that projection is what makes `π_β` cancel or not. Project by minimizing
the *forward* KL, averaged over the data states: `argmin_θ E_ρ[ KL(π* ‖ π_θ) ] = argmin_θ E_ρ
E_{a~π*}[ -log π_θ(a|s) ]`, since only the `-log π_θ` term depends on `θ`. I cannot sample `π*`
directly, but `π*` is just `π_β` reweighted, so importance-sample from the buffer instead:

  `E_{a~π*}[ -log π_θ ] = E_{a~π_β}[ (π*/π_β)·(-log π_θ) ] = E_{a~π_β}[ (1/Z(s)) exp(A/λ)·(-log π_θ) ]`.

The `π_β` factor cancels — `π*/π_β = (1/Z) exp(A/λ)`, no behavior model left. The actor update becomes
a *weighted maximum likelihood* on samples drawn straight from the buffer:

  `θ ← argmax_θ E_{(s,a)~buffer}[ exp( A^{π_k}(s,a) / λ ) · log π_θ(a|s) ]`.

This is supervised learning on the dataset's own actions, each `(s,a)` weighted by its exponentiated
advantage. The constraint is enforced *implicitly*: reweighting the buffer's actions can never put mass
on an action the data did not contain, yet it concentrates that mass on the high-advantage actions. No
behavior model anywhere, and — crucially for narrow human data — the actor never queries Q at a
policy-proposed off-support action during improvement. The reverse KL would drag both the behavior
model and the OOD-Q query back in (it evaluates `log π_β` and samples `a ~ π_θ`), so forward KL is the
right call precisely because it lets me sample from the buffer and cancel `π_β`.

The per-state `Z(s) = E_{a~π_θ}[exp(A/λ)]` in the weight I can drop and simply normalize weights across
the minibatch: it is a per-*state* factor, so it only reweights how much different states count, not
how actions compete within a state, and estimating it injects variance like a degenerate importance
weight. That leaves the critic. I want the advantage `A = Q - V` from an off-policy bootstrapped Q^π of
the *current* policy (this is what makes the method improve past a single step, unlike Monte-Carlo
behavior-value methods). I bootstrap a twin-Q TD target with the `min` of the targets and a Polyak
target update to keep overestimation in check: `y = r + γ·min_i Q̄_i(s', a')` with `a' ~ π(·|s')`. And
since `V(s) = E_{a~π}[Q(s,a)]`, I estimate it by evaluating the critics at an action sampled from the
current policy and taking the same `min`, so `A(s,a) = Q(s,a) - min_i Q_i(s, a_π)` with `a_π ~ π(·|s)`.

Now I have to be honest about how this harness differs from the generic recipe, because the constraints
here are tighter than where the method was tuned and they cut against its strengths. First, this is a
*purely offline* run — one million gradient steps on a static buffer, no online phase at all. The
recipe's headline strength is that the same update flows from offline pre-training into online
fine-tuning without changing anything; that strength is simply unused here, and what is left is the
offline half, which on narrow human data is the part most exposed to the implicit-constraint being only
as good as the advantage estimate. Second, the harness fixes batch size at 256 and the hidden width at
256 across three layers; I cannot reach for the larger batch or the wider nets that smoothed the
advantage-weighting variance elsewhere. Third, and most consequential, the temperature: on dexterous
manipulation a small `λ` sharpens the weights toward the few highest-advantage demo actions, which is
aggressive on data this thin. I set `awac_lambda = 0.1` to match the reference's manipulation
configuration, knowing it is a sharp, high-variance reweighting on twenty-five trajectories. I clip the
exponentiated weight at 100 so a handful of huge advantages cannot dominate the loss.

I also have to fix the policy parameterization to what this harness's actor needs, and it is *not* the
squashed Tanh-Gaussian the scaffold ships by default. Advantage-weighted MLE evaluates `log π_θ(a|s)`
on dataset actions; a TanhTransform distribution makes that log-prob awkward near the action-box
boundary (the demos sit near the limits of a dexterous hand's range). So I use a plain Gaussian over a
3×256 trunk with a *state-independent* `log_std` parameter, clamped to `[-20, 2]`, and compute the
log-prob directly with no Tanh correction — the actor clamps its samples into `[-1,1]` but the density
is a clean Normal. The critic here returns its output *un-squeezed*, shape `(batch, 1)`, because the
advantage and TD arithmetic in this version keep the trailing dimension; that is a small but real
divergence from the squeeze-to-scalar pattern other rungs use. I keep separate optimizers for the two
critics and turn on state normalization (`CONFIG_OVERRIDES = {"normalize": True}`), since whitening the
24-to-30-dim hand state helps the MLPs and matches how this baseline was run.

Per step, then: sample `(s, a, r, s', done, â')` (I ignore `â'` — that is ReBRAC's hook, not mine).
Critic update every step: bootstrap `q_next = min(Q̄_1(s', a'), Q̄_2(s', a'))` at `a' ~ π(·|s')`, target
`y = r + γ(1-done) q_next`, MSE on both critics, step both. Actor update every step: compute the
advantage with detached critics, `weight = clamp(exp(A/λ), max=100)`, and minimize `-(log π_θ(a|s)·
weight).mean()`. Then soft-update both target critics at `tau = 5e-3`. The whole thing is a standard
actor-critic where the only non-standard piece is the advantage-weighted MLE actor, which is what keeps
it on the demo tube without a behavior model.

What do I expect this to do on the three tasks, and where do I think it will leave value on the table —
the question the next rung will have to answer. Pen is the most reachable task and the one where
advantage weighting over near-expert demos should do best, so I expect a respectable Pen score, but
with high seed-to-seed variance, because at `λ=0.1` on twenty-five trajectories the weights are a sharp
function of a noisy advantage estimate — a seed whose critic happens to inflate a few demo actions will
do well, a seed whose critic is mis-calibrated will collapse. Hammer and Door I am pessimistic about:
the advantage signal there has to flow through a long, precise contact sequence, the implicit
constraint keeps the policy welded to twenty-five demonstrations of it, and with no online correction
and no separate in-support optimism the method has little room to improve past the average demo action.
So my falsifiable expectation is: a decent-but-volatile Pen mean, near-floor Hammer, and near-zero Door
— and if that is what the numbers show, the diagnosis for the next rung is already legible. The weak
spot is that the advantage here is `Q - V` with `V` read off a single policy sample, so the "how good
is this action relative to alternatives" signal is exactly as noisy as one stochastic critic query, and
the actor is pinned to demo actions by an MLE that has no explicit machinery for *staying calibrated*
when the data is this thin. A method that builds the constraint into the *target construction* — so the
critic itself never trusts an off-support next action — should be steadier across seeds, and that is the
direction the failure points.
