The advantage-weighted rung told me exactly what I feared it would, and the numbers are precise about
the failure. Pen came in at a mean of 67.1, but the per-seed spread is the tell: 106.7, 32.8, 61.8 — one
seed near-expert, one barely above a third of expert, a spread of about seventy-four points end to end.
Read that against the mechanism I built: at `λ=0.1` the exponentiated-advantage weight has a dynamic
range of order `10^4` across demo actions and saturates at a weight of 100 once an advantage clears
`0.1·ln 100 ≈ 0.46`, and the advantage itself is `Q` minus a *single-sample* `V(s)`. So a per-seed
critic that miscalibrates the demo advantages by even a tenth of a unit reweights the actor's imitation
targets by a factor `exp(±1) ≈ 2.7`, and the actor's mean — a weighted average of demo actions — slides
to a different corner of the demo hull. That is precisely a seventy-point Pen swing: when the critic
calibrates, the weighted MLE concentrates on the right actions and Pen flies to 106.7; when it does not,
the same enormous weights point at the wrong demos and Pen collapses to 32.8. Hammer sat at a mean of
1.05 (0.57, 0.97, 1.62) — essentially the floor, the policy never assembling the long precise contact
sequence — and Door at 0.59, which is 0.0 to within noise once you see the per-seed 1.86, −0.01, −0.07.
So the diagnosis is concrete: the implicit constraint kept the actor welded to the demos, and the
advantage signal, read off a single stochastic policy-sample of `V(s)`, was too noisy to do steady
improvement on data this thin. The constraint lived only in the *actor*, and the calibration it depended
on was exactly as noisy as one critic query.

Before I redesign, let me make sure the cheapest patch is not just retuning what I already have, because
if it were I would take it. The obvious lever is the temperature `λ`. Push it up and the weights flatten
toward uniform, the `exp(±1)` amplification shrinks, and Pen's variance would come down — but flattening
the weights *is* moving back toward plain behavior cloning, which caps the policy at the demonstrators
and would surrender the improvement bite that got a seed to 106.7 in the first place; and it does nothing
for Hammer and Door, which are already at the imitation floor, not above it. Push `λ` down instead and I
sharpen further, buying more improvement on the lucky Pen seed at the cost of even wilder variance. So the
temperature is a one-dimensional trade between Pen's mean and Pen's spread, and neither end touches the
Hammer/Door failure at all. The problem is not the *value* of the knob; it is that there is only one
constraint, it lives in the actor, and the signal driving it is a sampled `V`. I want two separate
changes: move the regularization into the *target construction* so the critic itself refuses to trust an
off-support next action, and replace the stochastic sampled advantage with a deterministic, lower-variance
signal. That is a redesign, not a retune.

Let me build it from the bottom. The strongest deterministic base I trust in continuous control is TD3,
and every piece of it is aimed at overestimation — the same disease, in its online form. The root fact is
that a greedy bootstrap is upward-biased under noise: for zero-mean estimation error `ε`, Jensen on the
convex `max` gives `E[max_a (Q+ε)] ≥ max_a E[Q+ε] = max_a Q`, the max hunts the luckiest positive error.
TD3's three answers each attack a different limb of this. Twin critics with a `min` target, `y = r +
γ·min_i Q̄_i(s', a')`, replace the upward-biased max with a downward-biased `min`, which is
self-correcting offline: an underrated action is simply avoided by the policy, whereas an overrated one is
actively chased. Clipped noise on the target action, `a' = π̄(s') + clip(N(0,0.2), -0.5, 0.5)`, smooths
the target over a small ball so it cannot overfit a razor-thin `Q` spike at one action — the clip at
`0.5` bounds the perturbation so smoothing never wanders far. And delayed actor and target updates every
two steps let the critic error settle before each policy move, so the actor ascends a value estimate that
has had two gradient steps to stop moving. These are good and I keep them. But none of them keep the
policy near the data: offline, the deterministic actor walks off the demo tube, and `min` of two
*unconstrained* off-support values is still garbage. TD3 is necessary overestimation machinery, not the
offline fix.

The cheapest offline fix is the minimalist one: add a behavior-cloning penalty to the actor so it pays a
squared cost for deviating from the logged action, `π = argmax_π E[ Q(s, π(s)) - (π(s)-a)² ]`. One line
on top of TD3. There is one scale subtlety I have to get right, and it is worth doing the dimensional
arithmetic. The BC term is *bounded*: each action coordinate lives in `[-1,1]`, so a per-coordinate
squared error is at most `(2)² = 4`, and the sum over the ~28 action dimensions is bounded by `4·28 ≈
112` — a fixed, task-independent scale. But `Q` scales with the *arbitrary* reward magnitude of the task:
on a high-reward task `Q` is numerically large and dwarfs the bounded BC term, so the policy ignores the
demos and chases value; on a low-reward task `Q` is small and the BC term dominates, so the policy just
clones. The fix is to normalize the value term by the average magnitude of `Q` itself, `λ = 1/mean(|Q|)`,
used as a *stop-gradient* scalar. Check the dimensions: if `Q ~ O(R)` for reward scale `R`, then `λ·Q ~
O(R)/O(R) = O(1)`, which now sits at the same fixed `O(1)` scale as the BC term, so the RL-versus-imitation
balance `β` means the same thing on all three tasks regardless of their reward units. Because `λ` is a
stop-gradient scalar it only rescales the loss magnitude; it does not change the *direction* of the `Q`
gradient. With this, `λ·Q - β·(π-a)²` is TD3+BC, and it is my floor. Note the contrast with the previous
rung already: instead of an implicit constraint via exponentiated-advantage weighting on a *stochastic*
policy, I have an *explicit* squared penalty on a *deterministic* policy. Deterministic removes the
sampled-`V` variance that swung Pen seventy points; explicit lets me control conservativeness directly per
task.

Now find where TD3+BC leaves value on the table, because one BC term on the actor is not the end — and it
is the same leak that hurt the previous rung, just relocated. I regularized the actor at the training
states `s`: at `s` the policy is pulled toward `a`. But the critic target is `r + γ·min Q̄(s', a')` with
`a' = π̄(s') + noise`, generated at the *next* state `s'`, and nothing in the actor's BC term at `s`
guarantees `π(s')` is in-distribution at `s'`. So the bootstrap can still pick an off-support `a'`, the
critic can still overrate it, and the overestimation loop reopens one bootstrap step downstream from where
I patched it. Trace this along the Hammer horizon to see why the floor sticks. With `γ = 0.99` the value
at the start of the nailing sequence is a chain of order `1/(1-γ) = 100` bootstraps, each of which passes
through a next-action `π̄(s')` that the actor BC never constrained; a single off-support overestimate
anywhere in that chain backs up multiplicatively toward the start. The actor penalty fixes "what action
do I take at states I've seen"; it does nothing about "what action does the target assume I take next."
Those are different leaks, and on a hundred-step task the second one dominates.

So I want a penalty *inside the critic target* too. The behavior-regularized actor-critic framing names
exactly two places to inject a divergence `D(π(·|s), π_β(·|s))`: in the actor objective (a policy
regularization — the BC term I have) or in the critic target (a value penalty, subtracting `α·D` from the
bootstrap so it is pessimistic exactly where the policy departs from behavior). The actor penalty keeps
the *acting* policy near data; the value penalty keeps the *bootstrap* from trusting a next-action that
has drifted off data. They are complementary, and TD3+BC simply never took the value-penalty half. The
general framework wrote `D` as KL/MMD/Wasserstein and needed a learned `π_β` — the very behavior model
that was too hard to fit on twenty-five narrow trajectories and that I am deliberately avoiding. But my
policy is *deterministic*: `π(s)` is a point, not a distribution. The natural divergence between the
policy's action and the data's action is just the squared Euclidean distance — and the harness already
hands me what I need. Its dataset converter preserves the dataset's *own next action* `â'` in the batch
(`s, a, r, s', done, â'`). So the value penalty needs no behavior model at all: take the bootstrapped `min
Q̄(s', a')`, and subtract a squared penalty for how far the policy's next action `a'` is from the
dataset's recorded next action `â'`. One subtraction, no extra network — and it costs nothing against the
256-width parameter budget, which matters because that budget forbids me from buying capacity my way out
of the problem.

Let me put a number on the smoothing so I know it is a gentle regularizer and not a second source of
off-support drift. The target noise is `N(0, 0.2)` clipped to `[-0.5, 0.5]`, i.e. clipped at `2.5σ`, so
the clip bites on only about `1.2%` of coordinate draws and the smoothing is effectively a small
`0.2`-standard-deviation ball around `π̄(s')`. That is small next to the action range `[-1,1]`, so it
blurs a razor-thin `Q` spike without itself launching the target action off support — which matters here,
because a large smoothing radius would manufacture exactly the off-support next-action query I am adding
the critic penalty to suppress. The two mechanisms have to be sized consistently, and a `0.2` ball inside
a `2`-wide box is consistent.

There is a subtlety in *what* I anchor the critic penalty to that is easy to get wrong. I penalize
`Σ(a' - â')²` — the policy's smoothed next action against the *dataset's recorded* next action `â'` — not
against the target policy's own `π̄(s')`. This matters: `â'` is the action the demonstrator actually took
at `s'`, a fixed per-sample supervised anchor that needs no model, whereas anchoring to `π̄(s')` would
compare the policy to itself and penalize nothing at all. Using `â'` is what turns "the harness preserved
the next action in the batch" into a free behavior-regularizer: every transition carries its own
in-support next-action target, and the penalty measures precisely how far the bootstrap's chosen
next-action has drifted from the trajectory's real continuation.

I should also fix the initialization arithmetic, because with the value-normalization scalar in play the
starting scale of `Q` is not innocuous. CORL-style init sets hidden weights uniform in `±√(1/fan_in)`;
for a `256`-wide layer that is `±1/16 ≈ ±0.0625`, a standard fan-in scaling that keeps activations
`O(1)`. The output heads start tiny — actor output uniform in `±1e-3`, critic output in `±3e-3` — so both
networks begin producing near-zero outputs. That near-zero start is deliberate: a critic that begins
outputting values around zero, together with LayerNorm's hard cap, means the runaway-extrapolation engine
has no large initial values to amplify, and the value normalization `λ = 1/mean(|Q|)` only takes over the
actor's RL-vs-BC balance once `Q` has grown to a genuine scale. The delayed-update schedule interacts with
the million-step budget too: at `policy_freq = 2` the critic takes all `1e6` gradient steps while the
actor and the three target networks take `5·10^5`, so the value estimate always leads the policy it is
being ascended, which is the ordering delayed updates exist to guarantee.

The target, written out: smoothed next action `a' = clip(π̄(s') + clip(N(0,0.2),-0.5,0.5), -1, 1)`;
bootstrap `q = min_i Q̄_i(s', a')`; value penalty `q ← q - β_critic·Σ(a' - â')²`; target `y = r +
γ(1-done)·q`. The actor keeps its own penalty, `λ·Q(s,π(s)) - β_actor·Σ(π(s)-a)²`. Now I am using a value
penalty *and* a policy regularization at once, both as cheap squared distances for a deterministic policy.
Notice what the critic penalty does mechanically at exactly the leak I diagnosed: when the policy's next
action `a'` agrees with the demonstrated `â'` the penalty is zero and the bootstrap is trusted in full;
the moment `a'` drifts off the demos the penalty grows quadratically and the target is pulled *down*, so
the critic is taught to distrust its own value precisely where the next-action left the data. That is the
downstream half of the Hammer chain being closed one link at a time.

The piece the framework left coupled is whether `β_actor` and `β_critic` should be the same number. They
do different jobs. The actor penalty controls how conservative the policy is *when it acts* — how willing
it is to leave the logged action to chase value. The critic penalty controls how distrustful the
*bootstrap* is of off-support next-actions. On a task where the dataset is broad I might want a small
actor penalty (let the policy improve) but a meatier critic penalty (the bootstrap still needs guarding);
on a narrow task the reverse. Forcing one coefficient onto both jobs gives every task a single point on a
one-dimensional trade-off when the real trade-off is two-dimensional. So decouple them, `β_actor` and
`β_critic`, tuned per task. This matters acutely here because the three Adroit datasets are *not* alike:
Pen `human` is a tight near-expert tube; Hammer `human` is a long, narrow, contact-heavy sequence; Door
`cloned` is a behavior-cloned mixture, broader and noisier. A single coefficient cannot serve all three,
which is part of why the previous rung's single global temperature left Hammer and Door at the floor. I
set them per environment, matching the harness's hardcoded values: Pen `(β_actor, β_critic) = (0.1,
0.5)`, Hammer `(0.01, 0.5)`, Door (the `door-cloned` dataset) `(0.01, 0.1)`. Read the pattern
quantitatively: the actor penalty on Pen is ten times the actor penalty on Hammer and Door, because Pen's
tube is tight enough that pinning the policy near the demos costs nothing and stabilizes it, whereas
welding the policy to the demos is exactly what floored Hammer and Door before, so they get a *tenth* the
actor pull to let the policy move. Meanwhile the critic penalty stays large on Pen and Hammer (`0.5`) —
both `human` datasets, where the bootstrap must not trust a next-action off twenty-five demonstrations —
and drops to `0.1` on the broader `door-cloned` data, where there is more support around each next-action
so the bootstrap can be trusted farther.

One architectural choice I will not inherit blindly and that I expect to be load-bearing for *this*
disease: LayerNorm in the critic. The whole problem is the critic extrapolating wildly on off-support
actions. If the last hidden feature `ψ` feeding the output head `w` is layer-normalized, its norm is a
bounded constant for *any* input: LayerNorm forces `ψ` to zero mean and unit variance across its `d = 256`
features, so `‖ψ‖ ≈ √d = √256 = 16` regardless of the action fed in. Then by Cauchy-Schwarz `|Q(s,a)| =
|wᵀ relu(ψ)| ≤ ‖w‖·‖relu(ψ)‖ ≤ ‖w‖·‖ψ‖ ≈ 16·‖w‖` — a hard, action-independent cap on the value of *any*
action, including ones the twenty-five demos never contain. Without LayerNorm the feature norm can grow
without bound as the action leaves the data, and it is exactly that unbounded growth that lets `Q`
extrapolate to a runaway value the actor then chases. Capping the feature norm at `16` kills that engine,
and it does so without telling the policy anything about which actions are good — it just refuses to let
any action look arbitrarily valuable. So post-activation LayerNorm goes in the critic, between every
hidden layer, and *not* in the actor — the actor is bounded into `[-1,1]` by a tanh and pulled to data by
`β_actor`; it is not the surface that extrapolates dangerous values. Asymmetric on purpose. I keep the
TD3+BC value normalization `λ = 1/mean(|Q|)` on the actor (so the RL-vs-imitation balance transfers across
the three reward scales), keep `policy_freq = 2` (two critic updates per actor update, so the value has
settled before each policy step), `tau = 5e-3`, use CORL-style init (uniform fan-in hidden, small-uniform
output), and run `actor_lr = critic_lr = 3e-4`. I do *not* turn on state normalization here, and I do
*not* use a Monte-Carlo return floor — both are side roads this baseline leaves off, and adding either
would confound the clean read on whether the two-place penalty is what moves the numbers.

It is worth writing the actor gradient to confirm the two terms pull the way I intend. The actor loss is
`β_actor·Σ(π(s)-a)² - λ·Q(s,π(s))`, so `∇_θ` is `2·β_actor·(π-a)·∂π/∂θ - λ·∂Q/∂a·∂π/∂θ`. The first term
is a spring of stiffness `2β_actor` pulling `π(s)` straight toward the demo action `a`; the second is the
value ascent `∇_a Q` scaled to `O(1)` by `λ`. At the fixed point these balance, `π(s)` settling where the
value gradient equals `2β_actor·(π-a)` — i.e. the policy leaves the demo action only as far as the value
gradient can overcome the spring. On Pen with `β_actor = 0.1` the spring is stiff and the policy stays
tucked against the near-expert demos; on Hammer and Door with `β_actor = 0.01` the spring is ten times
softer and the policy is *allowed* to move, which is the point — welding it to the demos is what floored
it. But softening the actor spring only helps if there is a trustworthy value gradient to move along, and
that is exactly the doubt I have about the single-`min` target on the long tasks, so I expect the softer
spring to buy Hammer and Door little unless the propagated value is there to reward the motion.

The LayerNorm asymmetry deserves one more look, because it is a real design commitment and not a default I
am copying. The critic is the only surface that maps an *arbitrary* action to a scalar value, so it is the
only place unbounded extrapolation can manufacture a target the policy will chase; capping its feature
norm at `16` bounds that. The actor maps a state to an action already confined to `[-1,1]` by a final
tanh, so its output is bounded by construction and there is nothing for a LayerNorm to protect against; a
LayerNorm inside the actor would only whiten intermediate features and constrain the policy's expressivity
for no safety gain. So LayerNorm goes in the critic and nowhere else — asymmetric because the two networks
face opposite risks.

So the delta from the previous rung is precise and it targets the exact failures the numbers showed. Where
advantage-weighting put the constraint only in a stochastic actor and read improvement off a noisy
single-sample `V`, I now (1) make the policy deterministic, removing the sampled-`V` variance that swung
Pen from 32.8 to 106.7; (2) put a second squared behavior penalty *inside the critic target* against the
dataset's own next action, closing the downstream leak that floored Hammer; (3) decouple the actor and
critic penalties and set them per task, so the tight Pen tube, the long Hammer sequence and the broad
cloned Door data each get their own conservativeness; and (4) cap critic extrapolation with LayerNorm at a
feature norm of `16`. My falsifiable expectations against the previous numbers, stated on the three
metrics: `pen-human-v1` should *tighten* — I want the seed spread well under the seventy-point swing, even
if the mean lands near or modestly above 67, because the deterministic policy removes the exponentiated
single-sample variance that produced the swing. `hammer-human-v1` I expect to stay low; a tiny actor
penalty plus a guarded bootstrap may lift it off the absolute floor, but the long contact sequence is the
hardest thing here and a target that is still a single bootstrapped `min` gives the deterministic actor
only one-step ascent to climb, so I would not bet on a large jump — it could even sit flat or dip if the
critic penalty makes the already-pessimistic `min` target so conservative that the long chain backs up
almost no value. `door-cloned-v1` I expect to stay near zero. If Pen tightens but Hammer and Door barely
move, that is the signal that the *explicit* TD3-style constraint, even decoupled and even with the
critic-side penalty, is still doing only one-step improvement off a deterministic policy — and I would
read that as evidence that no amount of tuning a one-step constraint, in the actor or in the critic, can
manufacture the propagated value the long tasks need. The ceiling would then be structural, a property of
the single bootstrapped target, not a matter of coefficients I have left un-tuned.
