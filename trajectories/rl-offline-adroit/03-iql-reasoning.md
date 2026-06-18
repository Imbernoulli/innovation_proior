The decoupled-penalty rung sharpened my picture of the ceiling, and again the numbers are exact about
where it landed. Pen came in at a mean of 74.6, and — as I hoped — it *tightened* relative to the
advantage-weighted rung: per-seed 95.5, 55.0, 73.2, a spread of forty points instead of seventy. So
making the policy deterministic and putting an explicit penalty in did remove some of the sampled-`V`
variance, and the mean even crept up. But Hammer fell to a mean of 0.35 (0.34, 0.28, 0.44) — *below* the
previous rung's 1.05, and rock-steady at the floor — and Door collapsed to 0.016, indistinguishable from
zero. That Hammer number is the tell I have to read carefully: the per-seed values are tight and tiny,
which means this is not high-variance failure, it is *systematic* — the method is reliably learning a
policy that does almost nothing on the nailing task. The tiny actor penalty I set on Hammer (`0.01`) was
supposed to let the policy improve off the demos; instead the deterministic TD3 actor, doing one-step
ascent on a critic whose target is still a single bootstrapped `min`, simply has no in-support
improvement signal to climb on the long contact sequence, and the per-env penalty knob cannot
manufacture one. So the explicit, decoupled, deterministic constraint fixed the *variance* on Pen but it
is still fundamentally a one-step improvement off a TD3 critic, and on the tasks that need genuine
multi-step value propagation — stitching the long Hammer sequence, exploiting the broad cloned Door data
— it has nothing.

So I want to step back from "constrain the policy" entirely and attack the thing both previous rungs
worked around: the bootstrap's out-of-distribution query. Both rungs left a learned `Q` being evaluated,
at some point, at an action the demos never contained — the advantage-weighting rung at `a' ~ π(s')` in
its critic target and at `a_π ~ π(s)` in its `V` estimate; the decoupled-penalty rung at the smoothed
`π̄(s')` in its target, merely *penalized* toward `â'` rather than forbidden. What if the value training
never touches an unseen action at all, and yet still does real multi-step dynamic programming? That is
the move that could give Hammer the propagated signal it lacks.

Start from what actually breaks. Standard Q-learning regresses `Q(s,a)` onto `r + γ max_{a'} Q̄(s',a')`,
and that max ranges over *all* actions, including ones absent from the data at `s'`, where the network
extrapolates upward and the policy chases it. The safe alternative is SARSA: bootstrap with the
*dataset's* next action, `r + γ Q̄(s', a')` with `a'` from `D`. Now no off-support action is ever
touched — but MSE fits `Q` to the *mean* of those targets, so the fixed point is `Q^{π_β}`, the value of
the behavior policy. That is pure policy evaluation, one improvement step, no iteration — exactly the
"one-step" ceiling that floors the long-horizon tasks, because to get a good value at the start of the
Hammer sequence I need value to flow backward across transitions, stitching fragments, which a single
step cannot do. So I am caught: the max improves but queries off-support; SARSA's mean is safe but does
not improve.

Look hard at what SARSA's mean is missing. Reading `Q(s,a)` over the behavior action distribution as a
per-state random variable (randomness from the action `a ~ π_β(·|s)`), SARSA's MSE recovers its *mean*.
What improvement needs is the *maximum over in-support actions* — a max restricted to actions the
behavior policy could actually produce at `s`. That restriction is the whole game: it is a max (so it
improves, and iterating it does real dynamic programming) but it never reaches an off-support action (so
it stays safe). The obstacle is that I cannot *compute* that restricted max by sampling actions and
querying `Q` at each — the moment I sample and query, I am back to evaluating `Q` off-support. I need the
in-support max *without ever querying `Q` at any specific `a'`*.

The statistic that gives the upper tail of a random variable from its samples, without naming any point,
is the expectile. The τ-expectile of `X` minimizes an *asymmetric* squared loss,
`m_τ = argmin_m E[ |τ - 1(u<0)|·u² ]` with `u = x - m`: a positive residual (a sample above my estimate)
is weighted `τ`, a negative one `1-τ`. At `τ = 0.5` both weights are `½` and `m_{0.5}` is the mean
(SARSA). For `τ > 0.5` the samples *above* the estimate dominate, so the estimate is pushed up; as
`τ → 1` the expectile climbs to the supremum of the support. The expectiles are monotone non-decreasing
in `τ` and bounded by the support, so the limit exists and equals the in-support max. So the upper
expectile of `Q` over the behavior actions, estimated by an asymmetric-L2 regression on in-sample data
only, *is* the in-support max I wanted — exactly the improvement operator, expressed as a regression I
can run with SGD and zero off-support queries.

But I must be careful where I apply the expectile, and the reason is precisely the kind of optimism that
could re-poison the long Hammer bootstrap. If I take the expectile of the raw TD residual
`r + γ Q̄(s',a') - Q(s,a)`, the target carries *two* sources of randomness: the action `a' ~ π_β(·|s')`
(which I *want* to be optimistic over — the best in-support action is the improvement signal) and the
stochastic transition `s' ~ p(·|s,a)` (which I emphatically do *not* want to be optimistic over). An
upper expectile rewards high targets indiscriminately, so it would reward a target that is high merely
because the dynamics happened to land in a lucky next state — conflating "there is a better action here"
with "I got lucky with the dice." Compounded over the Hammer horizon, that dynamics-optimism produces a
wildly overoptimistic value. So I split the estimate. A value network `V_ψ(s)` takes the upper expectile
over actions with the transition held fixed:

  `L_V(ψ) = E_{(s,a)~D}[ L_2^τ( Q̄(s,a) - V_ψ(s) ) ]`,

where both `s` and `a` come from `D`, so the only randomness for a given `s` is the action, and `V` becomes
the τ-expectile of `Q` over dataset actions — optimistic over actions, no dynamics in it. Then back this
into `Q` with an *honest* MSE that averages over the transition:

  `L_Q(θ) = E_{(s,a,s')~D}[ ( r + γ V_ψ(s') - Q_θ(s,a) )² ]`.

The MSE is correct here precisely because `V_ψ(s')` already did the optimistic action selection; what
remains is to average `γ V_ψ(s')` over `s' ~ p`, and a mean is the right way to average dynamics. The
division of labor is the whole method: `V` takes the upper expectile over actions, `Q` takes the mean
over transitions, and both losses touch only dataset `(s,a,s')`. No policy appears in value training, no
off-support action is ever queried. And this *is* multi-step dynamic programming: the resulting value is
monotone in `τ`, bounded by the in-support optimum, and converges to it as `τ → 1`, spanning SARSA
(`τ=0.5`, the floor both previous rungs effectively sat near on the hard tasks) to in-support Q-learning
(`τ → 1`, the propagation Hammer needs). I stabilize with clipped double-Q (a single twin-Q module, take
the `min`) and a Polyak target critic so `V` chases a stable `Q̄`.

Now I have a near-optimal in-support `Q` and `V`, but no policy — value training was deliberately
policy-free. Extraction must obey the same commandment: never query `Q` at an unseen action. So no
`argmax_a Q` (searches off-support) and no DDPG-style `∇_a Q` ascent (evaluates `Q` at the policy's
possibly-off-support actions — exactly the one-step move that floored the previous rung on Hammer). What I
*can* do is reweight the dataset's own actions: advantage-weighted regression. The KL-constrained
improvement `max_π E_{a~π}[A] s.t. KL(π‖π_β) ≤ ε` has the closed form `π* ∝ π_β·exp(A/β_temp)`, and
projecting it onto the parametric policy by weighted maximum likelihood gives

  `L_π(φ) = E_{(s,a)~D}[ -exp( β_temp·(Q̄(s,a) - V_ψ(s)) )·log π_φ(a|s) ]`,

with advantage `A = Q - V` and inverse temperature `β_temp`. This only ever evaluates dataset actions —
it reweights observed `(s,a)` by how advantaged they are — so it queries nothing unseen, inherits an
implicit stay-near-`π_β` constraint, and crucially decouples from value training. I clip the weight
`exp(β_temp·A)` at 100 so a few huge advantages cannot dominate.

Let me now ground this exactly in the harness, because the way this rung fills the edit surface differs
from the previous two and from the generic recipe in ways I should state. The critic becomes a single
twin-Q module (a `Critic` holding `q1`, `q2`, with a `.both()` returning both heads squeezed to scalars
and `forward` returning their `min`) plus a `ValueFunction` for `V`. The actor is the place this task is
specific: it is *not* the scaffold's TanhTransform Gaussian and *not* the previous rung's plain Normal —
it is a 2×256 MLP with **Dropout 0.1** after each hidden ReLU and a **Tanh on the mean output**, with a
state-independent `log_std` and a plain `Normal` (no Tanh transform on the density). The dropout is real
regularization for the AWR objective on twenty-five trajectories — it fights the actor memorizing the
tiny dataset — and the Tanh-bounded mean keeps the predicted mean inside the action box where the demos
live. I set `iql_tau = 0.8` (the expectile; high enough to do meaningful in-support improvement, not so
high the asymmetric loss becomes unstable on thin data), `beta = 3.0` (the AWR temperature),
`exp_adv_max = 100`. The actor optimizer gets a `CosineAnnealingLR` decayed over the full `1e6` steps,
which matters on this little data — it anneals the AWR updates so late training does not thrash the
policy. I turn on state normalization (`CONFIG_OVERRIDES = {"normalize": True}`). Per step: update `V` by
expectile regression against the *target* critic, update `Q` by MSE onto `r + γ(1-done) V(s')` and
soft-update the target, then update the actor by clipped advantage-weighted log-likelihood and step the
cosine schedule. I ignore `â'` in the batch — that was the previous rung's hook, not mine.

So the delta from the previous rung is sharp and it targets exactly the systematic Hammer/Door floor the
numbers exposed. Where the decoupled-penalty rung did one-step ascent off a TD3 critic whose target was a
single bootstrapped `min` — and so had no propagated value to climb on the long sequence — I now (1)
never query an unseen action anywhere in value training; (2) replace the mean bootstrap with an upper
*expectile over actions* that is genuine multi-step in-support dynamic programming, so value can flow
backward across the Hammer contact sequence and stitch the cloned Door data; (3) split the expectile
(over actions) from an honest MSE (over dynamics) so I am not optimistic about lucky transitions; and (4)
extract the policy by advantage-weighted regression over dataset actions only, regularized with dropout
against memorizing twenty-five trajectories. My falsifiable expectations against the previous numbers:
Pen should hold high and stay tight — I expect a mean comfortably above the 74.6 of the previous rung,
with a seed spread no worse, because expectile DP plus AWR is the most reliable extractor on near-expert
data. Hammer is the real test of the whole thesis: if in-support multi-step propagation is what was
missing, Hammer should finally lift clearly off the ~0.35-1.0 floor the two previous rungs sat at — I
expect a small but unambiguous, above-noise mean, the first sign the long sequence is being valued. And
Door (cloned) should move off zero into low-but-positive territory, because the broad cloned data is
exactly what stitching can exploit and neither previous rung could. If Pen rises, Hammer clears the
floor, and Door turns positive across seeds, the in-support-DP hypothesis is confirmed and this is the
strongest rung.
