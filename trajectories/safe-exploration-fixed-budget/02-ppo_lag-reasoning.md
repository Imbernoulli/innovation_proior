The naive run told me exactly what I expected, in numbers, and the numbers are unambiguous: it is a
specification failure, not a learning failure. The policy learns the task perfectly well — reward is
the highest I will see on this whole ladder (mean 25.5 on PointGoal, 32.8 on CarGoal, 19.7 on Button)
— but `budget_success_rate` is zero on every single seed of every single environment, because nothing
in the objective the policy ascended carried the cost signal. And the *shape* of the violation is the
tell. On PointGoal cost lands around 51 against a budget of 25, a clean 2× overshoot; on CarGoal ~61;
and on Button it is catastrophic — 140 to 164, more than six times the limit, with one seed (123)
whose reward also collapsed to 8.8 because the arena's hazard density is so high that even greedy
reward-seeking gets tangled. The ordering across environments is exactly the geometry argument: the
densest-hazard task overshoots worst. So the diagnosis is clean. The policy felt no cost pressure, so
it took the shortest reward-seeking path through whatever hazards lay on it, and the cost is simply
whatever that path collects. To get any of these under 25 I have to put the constraint *into* the
thing the policy optimizes.

Let me read the overshoot as ratios rather than raw numbers, because the ratios are what a controller
has to close and they are not the same job in each environment. PointGoal's 51.42 against 25 is
`51.42/25 = 2.06×`; CarGoal's 60.72 is `2.43×`; Button's 152.43 is `6.10×`. So the three environments
do not present one problem at three sizes — they present a spread of nearly threefold in how far cost
has to fall (`6.10/2.06 ≈ 2.96`). And the reward these buy is the ceiling I am now working under:
25.54, 32.82, 19.69 as means. The per-seed structure sharpens it further. PointGoal's costs are
45.17 / 52.28 / 56.80, a spread of about 11.6 around the mean — the plant is visibly noisy epoch to
epoch and seed to seed, which is a fact I will have to respect when I choose how fast to move the
controller's weight, because a controller that lurches on a single noisy cost estimate will chase that
noise. Button's seed 123 is the loudest single data point: cost 163.78, the highest anywhere, *and*
reward collapsed to 8.79 while the other two Button seeds held ~25 reward — exactly the congestion
failure I flagged at the floor, where a greedy policy in the densest arena gets so tangled it loses
reward too. That one cell tells me Button is not merely over budget; it is a regime where reward and
cost are both going wrong, so the controller I build has to drag a 6× overshoot down without any help
from a well-behaved reward signal. To get any of these under 25 I have to put the constraint *into* the
thing the policy optimizes.

The reflex is to mash the two signals into one scalar and hand it to the PPO backbone I already
trust — ascend `E[Σ(r_t − β c_t)]` for some penalty weight β. And every time I reach for this I hit the
same wall: what is β? There is no formula that takes my budget `d = 25` and returns the β that achieves
it; the map from penalty coefficient to resulting cost return depends on the whole environment and the
policy class in a way I cannot invert ahead of time. Worse, the naive numbers show me the trade-off
*moves across environments* — PointGoal needs the cost halved, Button needs it cut by a factor of six.
A single β that pulls Button's 152 down under 25 would be enormous, and on PointGoal that same β would
terrify the policy off the task entirely. I can even put a number on the mismatch: to first order the
penalty needed to close a violation scales with the violation, and Button's is `2.96×` PointGoal's, so
a β tuned to just barely rescue Button would be roughly three times too strong on PointGoal — and I
already saw at the floor that PointGoal's greedy reward is only twice its budget, a fragile margin that
a 3×-too-strong penalty would obliterate, driving reward to zero for no safety benefit it needed. And
even within one run, the one β has to do two incompatible jobs: stay small early so the
still-incompetent policy can learn to navigate at all, then push hard once it has found the
rewarding-but-risky shortcuts the naive run lived on. No single number does both. A fixed penalty bakes
a trade-off I don't know how to set into a constant I have to set before I see anything — and the naive
table has just shown me that the *right* constant differs by threefold across environments and shifts
within a run, so there is no constant to find. The weight has to be produced by something that reads
the measured violation and adapts, per environment and over training. That is the whole content of the
next move.

So I stop trying to fold safety into the objective and keep it as what it is — a constraint. The clean
statement is the constrained problem `max_θ J_r(θ)` subject to `J_c(θ) ≤ d`, where `J_r` and `J_c` are
the expected returns of reward and of cost, defined identically except for which signal they sum.
This is a constrained MDP in Altman's sense, and the reason I prefer it over the penalty framing isn't
aesthetic. A constraint with a threshold encodes "get cost under `d` and then stop caring; spend
everything else on reward," which is the actual shape of the task's binary `budget_success_rate` — the
metric rewards crossing under 25 and is indifferent to how far under. A penalty keeps trading reward
against cost forever even deep inside the safe region. The threshold has a saturation built in that
matches the metric; the penalty does not.

How do I optimize a constrained objective when `J_r` and `J_c` are both nonconvex functions of a few
hundred thousand weights? I cannot project onto the feasible set — I have no closed form for it. The
standard tool for "maximize subject to an inequality" is the Lagrangian. Write the constraint in
standard form `g(θ) = J_c(θ) − d ≤ 0`, introduce a multiplier `λ ≥ 0`, and form
`L(θ, λ) = J_r(θ) − λ(J_c(θ) − d)`. The classical result is that the constrained problem equals
`max_θ min_{λ≥0} L`. Let me check the inner min by behavior before building on it. Fix θ. If the
constraint is violated, `g > 0`, then `min_{λ≥0} −λ·(positive)` drives `λ → ∞` and `L → −∞`, so the
outer max never picks such a θ — infeasible policies are punished. If the constraint is satisfied with
slack, `g < 0`, then `−λ·(negative)` increases in λ, so the min takes `λ = 0` and the penalty switches
off, giving pure reward. On the boundary the λ term vanishes. So the max-min reconstructs exactly
"reward if feasible, −∞ if not" — the Lagrangian's saddle *is* the constrained problem, and at the
saddle complementary slackness `λ·g = 0` holds: λ is positive only when the budget binds.

That is the convex textbook story, and `J_r, J_c` are nonconvex in θ, so I should worry about a duality
gap. But the nonconvexity is in the parameterization `θ ↦ π_θ`, not in the problem: in occupancy-measure
space both returns are *linear* in the occupancy measure over the convex set cut out by the Bellman
flow equations — a linear program, which has zero duality gap (Altman's LP view of CMDPs; Paternain et
al. 2019). The clean duality lives in occupancy space while I take gradient steps on neural parameters,
so the honest reading is: dual descent aims at the correct saddle structure of the true problem and my
neural updates approximate it. That is enough license to build the algorithm.

So I have a saddle-point problem and the natural solver is to alternate: ascend θ to push `L` up,
descend λ to push it down, and let them chase the saddle. The λ update first, because it is where the
adaptivity the naive run lacked actually lives. I want to descend `L` in λ; `∂L/∂λ = −g(θ)`, so
descent is `λ ← [λ + η(J_c − d)]_+`, projected to stay nonnegative. Check it by behavior, because a
flipped sign would silently invert the whole safety mechanism. Over budget (`J_c > d`): the bracket
increases, λ climbs, the next policy update weighs cost more heavily — correct, and this is exactly
the response the naive run never had to its 152 on Button. Under budget (`J_c < d`): the bracket
decreases, λ falls toward zero, the policy is freed to chase reward — correct. So λ tunes the trade-off
*for* me, reading the measured violation each epoch. Structurally λ is the running integral of the
violation signal, so it settles to whatever weight drives the average violation to zero. Let me verify
it has a fixed point where I want one. The dual update `λ ← [λ + η(Jc − d)]_+` stops moving exactly
when `Jc = d` (for λ > 0), because the increment is then zero; and if that standing λ holds `Jc` at
`d`, the pair is stationary. That is the boundary equilibrium complementary slackness predicts — a
positive λ pinning cost precisely at the budget with the constraint active — so the algorithm's rest
point is "cost equal to 25 with a standing safety weight," exactly the feasible boundary the metric
cares about. If the dynamics converge, they converge to a run that satisfies the budget with no margin
to spare. The open question the arithmetic already raised is not *where* the fixed point sits but
*whether* the slow integrator reaches it inside 1M steps. One subtlety:
the constraint is on the *episodic* cost return, so I update λ once per epoch from the logged mean
episode cost `Jc = self._logger.get_stats('Metrics/EpCost')[0]`, not from any per-step cost, and I
assert it is not NaN first — if the cost statistic ever went bad I would corrupt λ silently and the
safety mechanism would die quietly.

Now the θ update. I ascend θ on `L = J_r − λ J_c` (the `λd` term is constant in θ). The gradient is
`∇_θ J_r − λ ∇_θ J_c`, and I already have a machine that ascends an expected return given an advantage:
the PPO surrogate, with `∇_θ J = E[∇log π · A]`. The substrate already runs GAE twice — a reward
critic gives `adv_r ∝ ∇_θ J_r`, a cost critic gives `adv_c ∝ ∇_θ J_c`. So the combined gradient
corresponds to a single combined advantage `A = adv_r − λ·adv_c`, fed into exactly the PPO clip I
already trust; the clip keeps each update in a trust region and is agnostic to what `A` means. This is
the whole appeal — a safe-RL rule that is PPO plus one scalar and one line mixing two advantages, and
it is precisely the two editable hooks the task exposes: λ lives in `_update`, the mix lives in
`_compute_adv_surrogate`.

But shipping `A = adv_r − λ·adv_c` raw breaks, and the naive Button numbers tell me why it will break
*here specifically*. Button violated by 6×, so λ will not stay small — to make the agent feel a 6×
overshoot the dual ascent must drive λ to 5, 10, 50. At λ = 50 the cost term dominates and the *scale*
of the combined advantage is ~50× a normal advantage. PPO's clip range and learning rate were
calibrated assuming the advantage has roughly its usual magnitude; an advantage whose scale grows
linearly in λ silently inflates the effective step size exactly when λ is large — exactly when I most
need the update careful. The clip will not save me, because it clips the probability *ratio*, not the
advantage. So λ has leaked into two places at once: the relative weight of reward versus cost, which I
want, and the numerical scale of the gradient, which I do not.

I do not want to change λ's meaning to fix this — the ratio of reward weight to cost weight, `1 : λ`,
is what the dual is carefully tuning. I want to peel apart the two effects: keep the *direction* of the
combined advantage but normalize its *scale*. The cleanest normalization that is monotone in λ and
equals 1 at λ = 0 is dividing by `(1 + λ)`: `A = (adv_r − λ·adv_c)/(1 + λ)`. Set `u = λ/(1+λ)`; then
`A = (1−u)·adv_r − u·adv_c` with `u ∈ [0,1)`, a convex blend. At λ = 0, `u = 0`, pure reward — the
unconstrained update, right because slack means no safety tax. As λ → ∞, `u → 1`, pure cost-avoidance,
right because a huge violation should sacrifice reward. In between, the weights sum to 1, so `|A|` stays
on the order of an ordinary advantage no matter how large λ grows, and PPO's tuned step size behaves
identically throughout. Put the worst case through it: Button's 6× overshoot may push λ to something
like 10, and at λ = 10 the raw blend `adv_r − 10·adv_c` has magnitude on the order of ten normal
advantages, which — since the clip bounds the ratio, not `|A|` — would silently multiply the effective
gradient step by about ten exactly on the run that can least afford an unstable update. Under the
normalized form, λ = 10 gives `u = 10/11 = 0.909`: the update is 90.9% cost-avoidance and 9.1% reward,
which is the aggressive-but-sane mix a 6× violation should get, and `|A| = |0.091·adv_r − 0.909·adv_c|`
is still order one, so PPO's clip and learning rate see the magnitude they were tuned for. The *ratio*
of the weights is still `(1−u):u = 1:λ = 1:10`, so the gradient direction and the saddle point are
unchanged; the `(1+λ)` factor is a step-size normalization, not a new optimum. This matters most
precisely where I most need it — the densest arena, where λ must go largest — which is a reassuring
sign the normalization is aimed at the real failure mode rather than a cosmetic one.

For the λ step itself I use the OmniSafe `Lagrange` helper the harness exposes: it stores λ as a
learnable scalar and realizes the projected ascent through an Adam step on the loss
`−λ·(Jc − d)`, whose gradient is `−(Jc − d)`, giving `λ ← λ + η·(Jc − d)` followed by a nonnegative
clamp. Wrapping the single dual scalar in Adam (rather than a bare fixed-step update) smooths it against
the epoch-to-epoch Monte-Carlo noise in the estimate of `J_c`, with `lambda_lr` the dual learning rate.
The noise is not hypothetical: the naive PointGoal costs spanned about 11.6 across seeds, so a single
epoch's `Jc` estimate can be off from the true cost return by something of that order, and the raw dual
step `λ ← λ + η(Jc − d)` would inject that full swing straight into λ. Adam's running normalization and
momentum average successive gradients, so a one-epoch estimation blip does not yank λ; only a
*persistent* violation moves it. The learning rate the harness exposes is `lambda_lr = 0.035`, which is
deliberately small — with an error of order ten and a step of 0.035, λ climbs on the order of `0.035 ×
(Jc − d)` per epoch, a fraction of a unit at a time, so reaching a λ near 10 to rescue Button takes many
epochs of sustained overshoot. That slowness is a feature for stability and, I will argue below, the
liability for the metric. So `_init` constructs `Lagrange(**lagrange_cfgs)`, `_update` calls
`update_lagrange_multiplier(Jc)` then `super()._update()`, and `_compute_adv_surrogate` reads
`self._lagrange.lagrangian_multiplier` into the normalized blend.

Before I lock this in, one more alternative deserves a fair walk: instead of driving λ by dual ascent
on the measured violation, I could put λ on a fixed increasing schedule — start near zero to let the
policy learn, then anneal λ up on a fixed clock. This is tempting because it directly targets the
early-versus-late tension I named against fixed β. But it fails the same test for a subtler reason: a
schedule is open-loop. It moves λ on a clock that knows nothing about whether *this* environment's cost
is already under 25 or still at 150, so on PointGoal it would over-push a nearly-feasible policy while
on Button it would under-push a wildly infeasible one — the exact threefold environment mismatch the
naive table exposed, now baked into a schedule instead of a constant. Dual ascent, by contrast, is
closed-loop: λ is driven by the measured `Jc − d`, so it rises fast where the violation is large and
settles where it is small, adapting per environment without my setting anything. The schedule trades
one un-settable constant for one un-settable clock; the feedback law removes both by reading the plant.
That is the decisive reason to prefer measured dual ascent, and it is the same reason the constraint
belongs in a controller rather than a hyperparameter.

There is one structural worry I can already name from the arithmetic, and it will shape every
prediction. λ is the running integral of the violation, and the environment that needs the largest λ is
also the one starting furthest from feasible: Button must both climb λ to ~10 *and* drag cost down from
152, while PointGoal needs only a modest λ to close a 2× gap. So the controller's hardest lifting and
its slowest-to-build weight coincide on the same environment. With `lambda_lr = 0.035` moving λ a
fraction of a unit per epoch of sustained overshoot, the integral simply may not have finished climbing
by the time the 1M-step budget runs out on Button — and even PointGoal, whose λ has less far to go,
is being pushed by an integrator that only reacts after violation has accumulated. That coincidence is
the seed of the failure mode I test for next.

Now the falsifiable expectations against the naive numbers. The mechanism is dual *ascent* on a
single scalar driven by the running average violation, so it is slow: λ integrates the error and only
settles after many epochs, and on a 1M-step budget it may simply not have climbed high enough by the
end. So I expect cost to come *down* from naive everywhere — the policy now feels some cost pressure —
but I am not confident any environment crosses under 25. Concretely: PointGoal should fall from ~51
toward the 40s, CarGoal from ~61 toward the 40s, Button most dramatically from ~152 toward the 50s,
because the bigger the violation the harder λ pushes. Reward should drop in lockstep, because every
unit of cost pressure is reward I am trading away — I expect PointGoal's 25.5 to fall toward the teens
and Button's reward to collapse toward single digits. The clean test of the slow-integrator
hypothesis: if `budget_success_rate` is *still zero everywhere* while cost has roughly halved, then the
direction is right but the controller is too sluggish — it lowers cost without ever crossing the
budget, which is the only thing the metric counts. That failure would confirm the integrator is
structurally too late on this horizon: it only pushes hard after the violation has already piled up, so
the next rung would have to attack the *timing* of the response, not merely its eventual magnitude. The
distilled scaffold fill is in the answer.
