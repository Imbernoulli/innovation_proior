The decoupled-penalty rung sharpened my picture of the ceiling, and again the numbers are exact about
where it landed. Pen came in at a mean of 74.6, and — as I hoped — it *tightened* relative to the
advantage-weighted rung: per-seed 95.5, 55.0, 73.2, a spread of about forty points instead of seventy. So
making the policy deterministic and putting an explicit penalty in did remove some of the sampled-`V`
variance, and the mean even crept up from 67.1 to 74.6. But Hammer fell to a mean of 0.35 (0.34, 0.28,
0.44) — *below* the previous rung's 1.05, and rock-steady at the floor — and Door collapsed to 0.016,
indistinguishable from zero. That Hammer number is the tell I have to read carefully. The per-seed values
are `0.34, 0.28, 0.44`: a spread of `0.16` around a mean of `0.35`, a coefficient of variation under a
half. Contrast the advantage-weighted rung's Hammer, `0.57, 0.97, 1.62`, whose spread was larger than its
own mean. So the previous rung's Hammer was *high-variance* near-zero — occasionally something caught —
whereas this rung's Hammer is *tight* near-zero. Tight and tiny means this is not bad luck; it is
*systematic*: the method reliably learns a policy that does almost nothing on the nailing task. The tiny
actor penalty I set on Hammer (`0.01`) was supposed to let the policy improve off the demos, and the
gradient balance says the spring is soft enough to allow motion — but the deterministic TD3 actor doing
one-step ascent needs a value gradient to move *along*, and on a critic whose target is still a single
bootstrapped `min`, there is no propagated in-support improvement signal on the long contact sequence to
climb. Softening the spring let the policy move and gave it nowhere good to move to. The per-env penalty
knob cannot manufacture a signal that the target construction never produced.

So I want to step back from "constrain the policy" entirely, because I have now tried both ways of doing
it — an implicit exponential-weight constraint and an explicit squared-penalty constraint — and both left
the same Hammer/Door floor. Let me name the alternatives at this rung honestly before I pick. One option
is to keep pushing on the constraint: raise the critic penalty further, add a Monte-Carlo return floor,
anneal the coefficients. But every one of these still sits on a target that is a single bootstrapped `min`
over a policy-proposed next action; they change *how hard I hold the policy near data*, not *whether the
value propagates*, and propagation is what the tight Hammer floor says is missing. A second option is
CQL-style pessimism — push `Q` down on sampled off-support actions — but that still evaluates `Q` at
policy-proposed actions (merely penalizing them) and needs many action samples per state on a ~28-dim box
to estimate the conservative term, which is both expensive and high-variance on five thousand
transitions. Both of these are variations on constraining the policy or the critic *after* it has queried
an unseen action. The move I have not tried is to make the value training *never touch an unseen action at
all*, and yet still do genuine multi-step dynamic programming so value can flow backward across the Hammer
sequence. That is the one that could give Hammer the propagated signal it lacks, so that is the one I
build.

Start from what actually breaks. Standard Q-learning regresses `Q(s,a)` onto `r + γ max_{a'} Q̄(s',a')`,
and that max ranges over *all* actions, including ones absent from the data at `s'`, where the network
extrapolates upward and the policy chases it. The safe alternative is SARSA: bootstrap with the
*dataset's* next action, `r + γ Q̄(s', a')` with `a'` from `D`. Now no off-support action is ever touched
— but MSE fits `Q` to the *mean* of those targets, so the fixed point is `Q^{π_β}`, the value of the
behavior policy. Let me confirm that MSE gives the mean, because the whole design hinges on replacing this
mean with something better: minimizing `E[(X - m)²]` over `m` has derivative `-2E[X - m] = 0`, so `m =
E[X]`, the mean. That is pure policy evaluation, one improvement step, no iteration — exactly the
"one-step" ceiling that floors the long-horizon tasks, because to get a good value at the start of the
Hammer sequence I need value to flow backward across transitions, stitching fragments, which a single step
of evaluation cannot do. So I am caught: the max improves but queries off-support; SARSA's mean is safe
but does not improve.

Look hard at what SARSA's mean is missing. Reading `Q(s,a)` over the behavior action distribution as a
per-state random variable (randomness from the action `a ~ π_β(·|s)`), SARSA's MSE recovers its *mean*.
What improvement needs is the *maximum over in-support actions* — a max restricted to actions the behavior
policy could actually produce at `s`. That restriction is the whole game: it is a max (so it improves, and
iterating it does real dynamic programming) but it never reaches an off-support action (so it stays safe).
The obstacle is that I cannot *compute* that restricted max by sampling actions and querying `Q` at each —
the moment I sample and query, I am back to evaluating `Q` off-support. I need the in-support max *without
ever querying `Q` at any specific `a'`*.

The statistic that gives the upper tail of a random variable from its samples, without naming any point,
is the expectile. The τ-expectile of `X` minimizes an *asymmetric* squared loss, `m_τ = argmin_m E[ |τ -
1(u<0)|·u² ]` with `u = x - m`: a positive residual (a sample above my estimate) is weighted `τ`, a
negative one `1-τ`. Let me differentiate to see exactly what it computes. Setting `d/dm E[|τ -
1(u<0)|·u²] = 0` gives `E[|τ - 1(u<0)|·(x - m)] = 0`, i.e. `τ·E[(x-m)_+] = (1-τ)·E[(m-x)_+]` — the
`τ`-weighted mass of samples above `m` balances the `(1-τ)`-weighted mass below. At `τ = 0.5` both weights
are `½`, the condition is `E[x-m]=0`, and `m_{0.5}` is the mean — SARSA exactly. For `τ > 0.5` the samples
*above* the estimate are weighted more, so the balance point `m` is pushed up; as `τ → 1` the below-mass
weight `(1-τ)` vanishes and the only way to balance is `m` at the top of the support, so `m_τ →` the
supremum of `X`. The expectiles are monotone non-decreasing in `τ` and bounded by the support, so the
limit exists and equals the in-support max. So the upper expectile of `Q` over the behavior actions,
estimated by an asymmetric-L2 regression on in-sample data only, *is* the in-support max I wanted — exactly
the improvement operator, expressed as a regression I can run with SGD and zero off-support queries. And
because it is monotone in `τ`, the single knob `τ` slides continuously from SARSA (`τ=0.5`, the very floor
both previous rungs effectively sat near on the hard tasks) to in-support Q-learning (`τ→1`, the
propagation Hammer needs). That is the dial the previous rungs did not have.

But I must be careful where I apply the expectile, and the reason is precisely the kind of optimism that
could re-poison the long Hammer bootstrap. If I take the expectile of the raw TD residual `r + γ
Q̄(s',a') - Q(s,a)`, the target carries *two* sources of randomness: the action `a' ~ π_β(·|s')` (which I
*want* to be optimistic over — the best in-support action is the improvement signal) and the stochastic
transition `s' ~ p(·|s,a)` (which I emphatically do *not* want to be optimistic over). An upper expectile
rewards high targets indiscriminately, so it would reward a target that is high merely because the
dynamics happened to land in a lucky next state — conflating "there is a better action here" with "I got
lucky with the dice." Compounded over the Hammer horizon of order `1/(1-γ) = 100` steps, that
dynamics-optimism produces a wildly overoptimistic value, the exact overestimation disease in a new dress.
So I split the estimate. A value network `V_ψ(s)` takes the upper expectile over actions with the
transition held fixed:

  `L_V(ψ) = E_{(s,a)~D}[ L_2^τ( Q̄(s,a) - V_ψ(s) ) ]`,

where both `s` and `a` come from `D`, so the only randomness for a given `s` is the action, and `V`
becomes the τ-expectile of `Q` over dataset actions — optimistic over actions, no dynamics in it. Then
back this into `Q` with an *honest* MSE that averages over the transition:

  `L_Q(θ) = E_{(s,a,s')~D}[ ( r + γ V_ψ(s') - Q_θ(s,a) )² ]`.

The MSE is correct here precisely because `V_ψ(s')` already did the optimistic action selection; what
remains is to average `γ V_ψ(s')` over `s' ~ p`, and a mean is the right way to average dynamics. The
division of labor is the whole method: `V` takes the upper expectile over actions, `Q` takes the mean over
transitions, and both losses touch only dataset `(s,a,s')`. No policy appears in value training, no
off-support action is ever queried. And this *is* multi-step dynamic programming: the resulting value is
monotone in `τ`, bounded by the in-support optimum, and converges to it as `τ → 1`, spanning SARSA to
in-support Q-learning. I stabilize with clipped double-Q (a single twin-Q module, take the `min`) and a
Polyak target critic so `V` chases a stable `Q̄` rather than a moving one.

Let me trace one backup to convince myself the value genuinely propagates, because "multi-step DP" is the
claim the whole rung rests on. Take three consecutive transitions on the Hammer sequence, `s_0 → s_1 →
s_2`, all in the data. `V_ψ(s_2)` is the `0.8`-expectile of `Q̄(s_2, ·)` over the demonstrated actions at
`s_2` — an optimistic-over-actions value. `Q_θ(s_1, a_1)` regresses onto `r_1 + γ V_ψ(s_2)`, so it
inherits that optimism one step back. Then `V_ψ(s_1)` is the `0.8`-expectile of `Q̄(s_1, ·)`, which now
sits on top of the backed-up `V_ψ(s_2)`, and `Q_θ(s_0, a_0)` regresses onto `r_0 + γ V_ψ(s_1)`. So after
two sweeps the value at `s_0` already reflects an optimistic-in-support choice made two steps downstream,
and iterating the two losses to convergence backs the in-support-best value all the way up the chain —
this is exactly the stitching that a single SARSA evaluation could not do, and it is why I expect Hammer,
whose reward is realized only at the end of a long ordered motion, to be the task that moves most if the
thesis is right. Nothing in that trace ever evaluated `Q` or `V` at an action outside the demonstrations
at each state, which is the safety half of the same claim.

There is a second, quieter reason the split matters that the variance bookkeeping makes precise. The raw
TD target's variance decomposes as variance over the action plus variance over the transition,
`Var(target) = Var_a[·] + E_a Var_{s'}[·]` by the law of total variance. Taking an upper expectile of the
*whole* thing rewards both variance components, so it would inflate the value by the transition noise as
well as the action tail. By routing the action-optimism through `V` (expectile, `Var_a` only) and the
transition-averaging through `Q` (MSE, `E Var_{s'}` handled as a mean), I apply optimism to exactly the
first term and honest averaging to exactly the second. That is not a heuristic; it is the reason the split
into two networks exists rather than a single expectile regression on the TD residual.

Now set `τ`. The `τ → 1` limit is the pure in-support max, but the asymmetric loss at `τ` near `1` weights
below-residuals by `1-τ`, which near zero makes the regression cling to a handful of the very largest
targets — on five thousand transitions that is a high-variance estimator that can lock onto an outlier
demo action. So I want `τ` high enough to do meaningful in-support improvement but not so high the
expectile becomes a near-max over a tiny sample. `iql_tau = 0.8` weights above-residuals `0.8` and
below-residuals `0.2`, a `4:1` tilt — firmly optimistic, propagating value up the sequence, while still
averaging over enough of the action distribution to stay stable on thin data. That is the balance the
long-horizon tasks need: at `τ=0.5` I would be back at the SARSA floor, and near `τ=1` I would be back at
an overestimation-prone max.

Now I have a near-optimal in-support `Q` and `V`, but no policy — value training was deliberately
policy-free. Extraction must obey the same commandment: never query `Q` at an unseen action. So no
`argmax_a Q` (searches off-support) and no DDPG-style `∇_a Q` ascent (evaluates `Q` at the policy's
possibly-off-support actions — exactly the one-step move that floored the previous rung on Hammer). What I
*can* do is reweight the dataset's own actions: advantage-weighted regression. The KL-constrained
improvement `max_π E_{a~π}[A] s.t. KL(π‖π_β) ≤ ε` has the closed form `π* ∝ π_β·exp(A/β_temp)`, and
projecting it onto the parametric policy by weighted maximum likelihood gives

  `L_π(φ) = E_{(s,a)~D}[ -exp( β_temp·(Q̄(s,a) - V_ψ(s)) )·log π_φ(a|s) ]`,

with advantage `A = Q - V` and inverse temperature `β_temp`. This only ever evaluates dataset actions — it
reweights observed `(s,a)` by how advantaged they are — so it queries nothing unseen, inherits an implicit
stay-near-`π_β` constraint, and crucially decouples from value training: the value functions are learned
by expectile DP without any reference to the actor, and the actor is a pure downstream weighted regression
onto them. I set `beta = 3.0` and clip the weight `exp(β·A)` at 100 so a few huge advantages cannot
dominate. Put the clip point on the number line: `exp(3·A) = 100` at `A = ln(100)/3 = 4.605/3 ≈ 1.53`, so
any advantage above `1.53` is pinned at the maximum weight. Note the crucial difference from the earlier
exponential weighting: there the advantage was `Q` minus a *single-sample* `V`, so the weight's argument
was itself a noisy stochastic draw; here `A = Q̄ - V_ψ` is a difference of two *deterministic* learned
networks — no sampled `V` — so the same exponentiation no longer amplifies per-seed sampling noise. That
is the second reason to expect this rung to be steadier than the advantage-weighted one, on top of the
value now propagating.

Let me now ground this exactly in the harness, because the way this rung fills the edit surface differs
from the previous two and from the generic recipe in ways I should state. The critic becomes a single
twin-Q module (a `Critic` holding `q1`, `q2`, with a `.both()` returning both heads squeezed to scalars
and `forward` returning their `min`) plus a `ValueFunction` for `V`, so value training now runs three
networks — twin-Q, `V`, and a Polyak target of the twin-Q — where the previous rungs ran two critics and
a target. The actor is the place this task is specific: it is *not* the scaffold's TanhTransform Gaussian
and *not* the previous rung's plain Normal — it is a *2×256* MLP (one hidden layer fewer than the critics)
with **Dropout 0.1** after each hidden ReLU and a **Tanh on the mean output**, with a state-independent
`log_std` and a plain `Normal` (no Tanh transform on the density). Each of those is a data-frugality choice
for the AWR objective on twenty-five trajectories. The shallower 2×256 trunk has fewer parameters to
memorize five thousand transitions with; the dropout is genuine stochastic regularization that fights the
actor collapsing onto the tiny dataset — at rate `0.1` it zeros a tenth of the hidden units each forward
pass, so the weighted-MLE fit is forced to spread across redundant features rather than pin each demo to a
single path; and the Tanh-bounded mean keeps the predicted mean inside the `[-1,1]` action box where the
demos of a dexterous hand live, so the log-prob never has to explain a mean that has wandered outside the
box. Put a number on the capacity choice: the 2×256 actor is an input map `state_dim·256 ≈ 45·256 ≈ 1.2·10^4`,
one interior `256·256 = 65,536`, and an output `256·action_dim ≈ 256·28 ≈ 7·10^3`, about `8.5·10^4`
parameters — roughly half the `~1.5·10^5` of each three-layer critic. Deliberately: the value functions
need capacity to represent the propagated `Q` and `V` surfaces, but the actor only has to fit a weighted
regression onto demo actions, and the less capacity it has, the less it can memorize the five thousand
transitions verbatim. The dropout compounds this at train time in a way worth being explicit about: the
AWR log-likelihood `log π_φ(a|s)` is computed with the actor in *train* mode, so a tenth of the hidden
units are zeroed on each pass and the fit must survive that masking, whereas at evaluation `act` switches
to eval mode, disables dropout, and returns the policy *mean* — so the reported D4RL score is the clean
mode of a policy that was trained under noise, which is the standard way dropout buys generalization on a
tiny dataset. I keep `exp_adv_max = 100`, `discount = 0.99`, `tau = 5e-3`, `lr = 3e-4`. The actor optimizer
gets a `CosineAnnealingLR` decayed over the full `1e6` steps, which matters on this little data: it anneals the
AWR learning rate smoothly to zero so late training, when the value functions have converged and the
weights are their sharpest, does not thrash the policy with large steps on a fixed dataset it has already
fit. I turn on state normalization (`CONFIG_OVERRIDES = {"normalize": True}`) to whiten the hand state's
differently-scaled joint channels. Per step: update `V` by expectile regression against the *target*
critic `Q̄`, update `Q` by MSE onto `r + γ(1-done) V(s')` and soft-update the target, then update the
actor by clipped advantage-weighted log-likelihood and step the cosine schedule. Two ordering details are
load-bearing. `V` regresses against the *target* critic `min(Q1̄, Q2̄)`, not the live one: `V` and `Q`
otherwise chase each other within a step and the expectile sits on a moving estimate, so the Polyak target
at `tau = 5e-3` gives `V` a slow-moving `Q̄` to take the expectile of, and the `min` of the twin heads
keeps that expectile from being taken over an already-overestimated value. And the `Q` bootstrap uses
`V(s')` from the *live* value network rather than a separate target-`V`; `V` is bounded by construction —
it is an expectile of the twin-`min`, so it inherits the same conservatism and needs no target of its own.
The cosine schedule runs a single half-period from `3e-4` to `0` across the million steps, so the actor
learning rate is still near `3e-4` through early training when the advantages are settling and decays to
zero by the end when they are sharp. I ignore `â'` in the
batch — that next-action anchor was the previous rung's hook, not mine, since my value training never needs
a next action at all.

So the delta from the previous rung is sharp and it targets exactly the systematic Hammer/Door floor the
numbers exposed. Where the decoupled-penalty rung did one-step ascent off a TD3 critic whose target was a
single bootstrapped `min` — and so had no propagated value to climb on the long sequence — I now (1) never
query an unseen action anywhere in value training; (2) replace the mean bootstrap with an upper *expectile
over actions* at `τ=0.8` that is genuine multi-step in-support dynamic programming, so value can flow
backward across the Hammer contact sequence and stitch the cloned Door data; (3) split the expectile (over
actions) from an honest MSE (over dynamics) so I am not optimistic about lucky transitions; and (4)
extract the policy by advantage-weighted regression over dataset actions only, off a *deterministic*
learned advantage rather than a sampled one, regularized with dropout against memorizing twenty-five
trajectories. My falsifiable expectations against the previous numbers, on the three metrics:
`pen-human-v1` should hold high and stay tight — I expect a mean comfortably above the 74.6 of the
previous rung, with a seed spread no worse than its forty points, because expectile DP plus a
deterministic-advantage AWR is the most reliable extractor on near-expert data. `hammer-human-v1` is the
real test of the whole thesis: if in-support multi-step propagation is what was missing, Hammer should
finally lift clearly off the `0.35`–`1.05` floor the two previous rungs sat at — I expect a small but
unambiguous, above-noise mean, the first sign the long sequence is being valued. And `door-cloned-v1`
should move off zero into low-but-positive territory across seeds, because the broad cloned data is exactly
what stitching can exploit and neither previous rung could. If Pen rises, Hammer clears the floor, and Door
turns positive across seeds, the in-support-DP hypothesis is confirmed and this is the strongest rung.
