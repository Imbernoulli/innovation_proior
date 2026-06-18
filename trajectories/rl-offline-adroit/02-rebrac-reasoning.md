The advantage-weighted rung told me exactly what I feared it would, and the numbers are precise about
the failure. Pen came in at a mean of 67.1, but the per-seed spread is the tell: 106.7, 32.8, 61.8 —
one seed near-expert, one barely above a third of expert, a spread of seventy points. That is the sharp
`λ=0.1` reweighting reacting to a noisy advantage on twenty-five trajectories, exactly the volatility I
expected: when the critic happens to calibrate the demo advantages on a given seed, the weighted MLE
concentrates on the right actions and Pen flies; when it does not, the same weights point at the wrong
demos and the policy is dragged off. Hammer sat at a mean of 1.05 (0.57, 0.97, 1.62) — essentially the
floor, the policy never assembling the long precise contact sequence — and Door at 0.59, which is
0.0 to within noise once you see the per-seed 1.86, −0.01, −0.07. So the diagnosis is concrete: the
implicit constraint kept the actor welded to the demos and the advantage signal, read off a single
stochastic policy-sample of `V(s)`, was too noisy to do steady improvement on data this thin. The
constraint lived only in the *actor*, and the calibration it depended on was exactly as noisy as one
critic query. I want the regularization moved into the *target construction* — make the critic itself
refuse to trust an off-support next action — and I want a deterministic, lower-variance signal in place
of the sampled advantage that produced the seventy-point Pen swing.

Let me build that from the bottom. The strongest deterministic base I trust in continuous control is
TD3, and every piece of it is aimed at overestimation — the same disease, in its online form. The root
fact is that a greedy bootstrap is upward-biased under noise: for zero-mean error, `E[max(Q+ε)] ≥ max Q`,
the max hunts the luckiest positive error. TD3's answers: twin critics with a `min` target,
`y = r + γ·min_i Q̄_i(s', a')`, which biases toward *under*estimation (self-correcting, because the
policy just avoids actions it underrates instead of chasing ones it overrates); clipped noise on the
target action, `a' = π̄(s') + clip(N(0,0.2), -0.5, 0.5)`, so the target cannot overfit a razor-thin Q
spike; and delayed actor/target updates every 2 steps, so critic error settles before each policy move.
These are good and I keep them. But none of them keep the policy near the data — offline, the actor
walks off the demo tube and `min` of two unconstrained off-support values is still garbage. So TD3 is
necessary machinery, not the offline fix.

The cheapest offline fix is the minimalist one: add a behavior-cloning penalty to the actor so it pays a
squared cost for deviating from the logged action, `π = argmax_π E[ Q(s, π(s)) - (π(s)-a)² ]`. One line
on top of TD3. There is one scale subtlety I have to get right: the BC term is bounded (actions in
`[-1,1]`, so at most ~4) but `Q` scales with the arbitrary reward magnitude, so on a high-reward task `Q`
dwarfs the BC term and on a low-reward task the BC term dominates. The fix is to normalize the value
term by the average magnitude of `Q` itself, `λ = 1/mean(|Q|)`, used as a *stop-gradient* scalar — it
rescales the loss, it does not change the direction of the `Q` gradient. With this, `λ·Q - β·(π-a)²` is
TD3+BC, and it is my floor. Note the contrast with the previous rung already: instead of an implicit
constraint via exponentiated-advantage weighting on a stochastic policy, I have an *explicit* squared
penalty on a *deterministic* policy. Deterministic removes the sampled-`V` variance that swung Pen
seventy points; explicit lets me control conservativeness directly per task.

Now find where TD3+BC leaves value on the table, because one BC term on the actor is not the end — and
it is the same leak that hurt the previous rung, just relocated. I regularized the actor at the training
states `s`: at `s` the policy is pulled toward `a`. But the critic target is `r + γ·min Q̄(s', a')` with
`a' = π̄(s') + noise`, generated at the *next* state `s'`, and nothing in the actor's BC term at `s`
guarantees `π(s')` is in-distribution at `s'`. So the bootstrap can still pick an off-support `a'`, the
critic can still overrate it, and the overestimation loop reopens one bootstrap step downstream from
where I patched it. On Hammer this is exactly why the floor sticks: the long contact sequence means the
target at every step depends on a next-action the actor BC never constrained, so the value backed up
along the sequence is unreliable everywhere the policy's next-action drifts off the twenty-five demos.
The actor penalty fixes "what action do I take at states I've seen"; it does nothing about "what action
does the target assume I take next." Those are different leaks.

So I want a penalty *inside the critic target* too. The behavior-regularized actor-critic framing names
exactly two places to inject a divergence `D(π(·|s), π_β(·|s))`: in the actor objective (a policy
regularization — the BC term I have) or in the critic target (a value penalty, subtracting `α·D` from
the bootstrap so it is pessimistic exactly where the policy departs from behavior). The actor penalty
keeps the *acting* policy near data; the value penalty keeps the *bootstrap* from trusting a
next-action that has drifted off data. They are complementary, and TD3+BC simply never took the
value-penalty half. The general framework wrote `D` as KL/MMD/Wasserstein and needed a learned `π_β` —
the very behavior model that was too hard to fit on twenty-five narrow trajectories and that I am
deliberately avoiding. But my policy is *deterministic*: `π(s)` is a point, not a distribution. The
natural divergence between the policy's action and the data's action is just the squared Euclidean
distance — and the harness already hands me what I need. Its dataset converter preserves the dataset's
*own next action* `â'` in the batch (`s, a, r, s', done, â'`). So the value penalty needs no behavior
model at all: take the bootstrapped `min Q̄(s', a')`, and subtract a squared penalty for how far the
policy's next action `a'` is from the dataset's recorded next action `â'`. One subtraction, no extra
network — and it costs nothing against the 256-width parameter budget, which matters because that budget
forbids me from buying capacity my way out of the problem.

The target, written out: smoothed next action `a' = clip(π̄(s') + clip(N(0,0.2),-0.5,0.5), -1, 1)`;
bootstrap `q = min_i Q̄_i(s', a')`; value penalty `q ← q - β_critic·Σ(a' - â')²`; target
`y = r + γ(1-done)·q`. The actor keeps its own penalty, `λ·Q(s,π(s)) - β_actor·Σ(π(s)-a)²`. Now I am
using a value penalty *and* a policy regularization at once, both as cheap squared distances for a
deterministic policy.

The piece the framework left coupled is whether `β_actor` and `β_critic` should be the same number. They
do different jobs. The actor penalty controls how conservative the policy is *when it acts* — how
willing it is to leave the logged action to chase value. The critic penalty controls how distrustful the
*bootstrap* is of off-support next-actions. On a task where the dataset is broad I might want a small
actor penalty (let the policy improve) but a meatier critic penalty (the bootstrap still needs
guarding); on a narrow task the reverse. Forcing one coefficient onto both jobs gives every task a
single point on a one-dimensional trade-off when the real trade-off is two-dimensional. So decouple
them, `β_actor` and `β_critic`, tuned per task. This matters acutely here because the three Adroit
datasets are *not* alike: Pen `human` is a tight near-expert tube; Hammer `human` is a long, narrow,
contact-heavy sequence; Door `cloned` is a behavior-cloned mixture, broader and noisier. A single
coefficient cannot serve all three, which is part of why the previous rung's single global temperature
left Hammer and Door at the floor. I set them per environment, matching the harness's hardcoded values:
Pen `(β_actor, β_critic) = (0.1, 0.5)`, Hammer `(0.01, 0.5)`, Door (the `door-cloned` dataset)
`(0.01, 0.1)`. The pattern is legible — Hammer and Door take a *tiny* actor penalty (let the policy move,
because welding it to the demos was exactly what floored it before) while keeping a real critic penalty
on Hammer (guard the long bootstrap) and relaxing both on the broader cloned Door data.

One architectural choice I will not inherit blindly and that I expect to be load-bearing for *this*
disease: LayerNorm in the critic. The whole problem is the critic extrapolating wildly on off-support
actions. If the last hidden feature feeding the output head `w` is layer-normalized, its norm is a
bounded constant for *any* input, so by Cauchy-Schwarz `|Q(s,a)| = |wᵀ relu(ψ)| ≤ ‖w‖·‖ψ‖ ≤ ‖w‖` — a
hard cap on the value of *any* action, including ones the twenty-five demos never contain. That kills
the runaway-extrapolation engine that drove the previous rung's seed-to-seed instability, and it does
so without telling the policy anything. So post-activation LayerNorm goes in the critic, between every
hidden layer, and *not* in the actor — the actor is bounded into `[-1,1]` by a tanh and pulled to data by
`β_actor`; it is not the surface that extrapolates dangerous values. Asymmetric on purpose. I keep the
TD3+BC value normalization `λ = 1/mean(|Q|)` on the actor (so the RL-vs-imitation balance transfers
across the three reward scales), keep `policy_freq = 2`, `tau = 5e-3`, use CORL-style init (uniform fan-in
hidden, small-uniform output), and run `actor_lr = critic_lr = 3e-4`. I do *not* turn on state
normalization here, and I do *not* use a Monte-Carlo return floor — both are side roads this baseline
leaves off.

So the delta from the previous rung is precise and it targets the exact failures the numbers showed.
Where advantage-weighting put the constraint only in a stochastic actor and read improvement off a
noisy single-sample `V`, I now (1) make the policy deterministic, removing the sampled-`V` variance that
swung Pen from 32.8 to 106.7; (2) put a second squared behavior penalty *inside the critic target*
against the dataset's own next action, closing the downstream leak that floored Hammer; (3) decouple the
actor and critic penalties and set them per task, so the tight Pen tube, the long Hammer sequence and the
broad cloned Door data each get their own conservativeness; and (4) cap critic extrapolation with
LayerNorm. My falsifiable expectations against the previous numbers: Pen should *tighten* — I want the
seed spread well under the seventy-point swing, even if the mean lands near or modestly above 67. Hammer
I expect to stay low; a tiny actor penalty plus a guarded bootstrap may lift it off the absolute floor
but the long contact sequence is the hardest thing here and I would not bet on a large jump. Door I
expect to stay near zero on the cloned dataset. If Pen tightens but Hammer and Door barely move, that is
the signal that the *explicit* TD3-style constraint, even decoupled, is still doing only one-step
improvement off a deterministic policy — and the next move is a method that does genuine multi-step
in-support dynamic programming without ever querying an unseen action at all.
