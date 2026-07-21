The naive run told me what I expected, in numbers, and the numbers are unambiguous: this is a
specification failure, not a learning failure. The policy learns the task fine — reward is the highest
I will see on this ladder (mean 25.5 on PointGoal, 32.8 on CarGoal, 19.7 on Button) — but
`budget_success_rate` is zero on every seed of every environment, because nothing in the objective it
ascended carried the cost signal. Read as ratios, the three environments are not one problem at three
sizes: PointGoal's 51.42 is `2.06×` the budget, CarGoal's 60.72 is `2.43×`, Button's 152.43 is `6.10×`
— a nearly threefold spread (`6.10/2.06 ≈ 2.96`) in how far cost has to fall. The per-seed structure
adds two facts I will need. PointGoal's costs are `45.17 / 52.28 / 56.80`, a spread of about 11.6 — the
plant is visibly noisy seed to seed, so a controller that lurches on a single cost estimate will chase
that noise. And Button's seed 123 is the loudest cell: cost 163.78 *and* reward collapsed to 8.79 while
its siblings held ~25, the congestion failure I flagged at the floor, where a greedy policy in the
densest arena gets so tangled it loses reward too. So the controller I build has to drag a 6× overshoot
down without help from a well-behaved reward signal. To get any of these under 25 I have to put the
constraint *into* the thing the policy optimizes.

The reflex is to mash the two signals into one scalar and hand it to the PPO backbone I already trust —
ascend `E[Σ(r_t − β c_t)]` for some penalty weight β. And every time I reach for it I hit the same wall:
what is β? There is no formula that takes my budget `d = 25` and returns the β that achieves it; the map
from penalty coefficient to resulting cost return depends on the whole environment and policy class in a
way I cannot invert ahead of time. Worse, the naive table shows the trade-off *moves across
environments*. To first order the penalty needed to close a violation scales with the violation, so a β
tuned to just barely rescue Button's 6× would be about three times too strong on PointGoal — and
PointGoal's greedy reward is only twice its budget, a fragile margin a 3×-too-strong penalty would
obliterate, driving reward to zero for no safety benefit. Even within one run the single β has two
incompatible jobs: stay small early so the incompetent policy can learn to navigate at all, then push
hard once it has found the rewarding-but-risky shortcuts. No constant does both, and the naive table has
shown me the right constant differs threefold across environments and shifts within a run. The weight
has to be *produced* by something that reads the measured violation and adapts.

So I stop trying to fold safety into the objective and keep it as a constraint: `max_θ J_r(θ)` subject
to `J_c(θ) ≤ d`. I prefer this over the penalty framing for a concrete reason. A threshold constraint
encodes "get cost under `d`, then stop caring and spend everything on reward," which is exactly the
shape of the binary `budget_success_rate` — it rewards crossing under 25 and is indifferent to how far
under. A penalty keeps trading reward against cost forever, even deep in the safe region. The threshold
has a saturation the metric shares; the penalty does not.

To optimize a constrained objective when `J_r` and `J_c` are both nonconvex in a few hundred thousand
weights, I cannot project onto the feasible set — I have no closed form for it. The standard tool is the
Lagrangian: write `g(θ) = J_c(θ) − d ≤ 0`, introduce `λ ≥ 0`, form `L(θ, λ) = J_r(θ) − λ(J_c(θ) − d)`,
and the constrained problem equals `max_θ min_{λ≥0} L`. The inner min behaves as the constraint should:
fix θ; if violated (`g > 0`) then `min_{λ≥0} −λg` drives `λ → ∞`, `L → −∞`, so the outer max never picks
an infeasible θ; if satisfied with slack (`g < 0`) the min takes `λ = 0` and the penalty switches off,
giving pure reward; on the boundary the λ term vanishes. So the saddle reconstructs "reward if feasible,
−∞ if not," with complementary slackness `λ·g = 0` — λ positive only when the budget binds.

That is the convex story, and `J_r, J_c` are nonconvex in θ, so I should worry about a duality gap. But
the nonconvexity is in the parameterization `θ ↦ π_θ`, not the problem: in occupancy-measure space both
returns are *linear* over the convex set cut out by the Bellman flow equations — a linear program with
zero duality gap. The clean duality lives in occupancy space while I take gradient steps on neural
parameters, so the honest reading is that dual descent aims at the correct saddle of the true problem
and my neural updates approximate it. Enough license to build the algorithm.

The natural solver alternates: ascend θ to push `L` up, descend λ to push it down, chasing the saddle.
The λ update is where the adaptivity the naive run lacked lives. Descending `L` in λ with `∂L/∂λ = −g`
gives `λ ← [λ + η(J_c − d)]_+`, projected nonnegative. Check it by behavior, since a flipped sign would
silently invert the whole safety mechanism: over budget, the bracket grows, λ climbs, the next policy
update weighs cost more — correct; under budget, λ falls toward zero and the policy is freed to chase
reward — correct. Structurally λ is the running integral of the violation, so it settles wherever the
average violation is driven to zero: the update stops moving exactly when `J_c = d` (for λ > 0), the
boundary equilibrium complementary slackness predicts. If the dynamics converge, they converge to a run
sitting at cost 25 with a standing safety weight. The open question is not *where* that fixed point sits
but *whether* the slow integrator reaches it inside 1M steps. One subtlety: the constraint is on the
*episodic* cost return, so I update λ once per epoch from the logged mean episode cost
`Jc = self._logger.get_stats('Metrics/EpCost')[0]`, asserting it is not NaN first — a bad cost statistic
would corrupt λ silently and kill the safety mechanism.

The θ update ascends `L = J_r − λ J_c` (the `λd` term is constant in θ), gradient `∇_θ J_r − λ ∇_θ J_c`.
I already have a machine that ascends an expected return given an advantage — the PPO surrogate,
`∇_θ J = E[∇log π · A]` — and the substrate runs GAE twice, so `adv_r ∝ ∇_θ J_r` and `adv_c ∝ ∇_θ J_c`.
The combined gradient is a single combined advantage `A = adv_r − λ·adv_c` fed into the PPO clip I
already trust. That is the whole appeal: a safe-RL rule that is PPO plus one scalar and one line mixing
two advantages, landing exactly on the two editable hooks — λ in `_update`, the mix in
`_compute_adv_surrogate`.

But shipping `A = adv_r − λ·adv_c` raw breaks, and the Button numbers say why here specifically. To make
the agent feel a 6× overshoot, dual ascent must drive λ large — 5, 10, 50. At λ = 50 the cost term
dominates and the *scale* of the combined advantage is ~50× a normal advantage. PPO's clip and learning
rate were calibrated for advantages of ordinary magnitude, and — since the clip bounds the ratio, not
`|A|` — an advantage whose scale grows linearly in λ silently inflates the effective step size exactly
when λ is large, exactly when I most need the update careful. So λ has leaked into two places: the
relative weight of reward versus cost, which I want, and the numerical scale of the gradient, which I do
not.

I do not want to touch λ's meaning — the ratio `1 : λ` is what the dual is tuning — so I peel the two
apart: keep the *direction* of the combined advantage, normalize its *scale*. Dividing by `(1 + λ)` is
the cleanest normalization monotone in λ and equal to 1 at λ = 0: `A = (adv_r − λ·adv_c)/(1 + λ)`. With
`u = λ/(1+λ)`, `A = (1−u)·adv_r − u·adv_c`, a convex blend with `u ∈ [0,1)`. At λ = 0, `u = 0`, pure
reward; as λ → ∞, `u → 1`, pure cost-avoidance; in between the weights sum to 1 so `|A|` stays order one
no matter how large λ grows. Button's 6× may push λ to ~10, giving `u = 10/11 = 0.909`: 90.9%
cost-avoidance, 9.1% reward, `|A|` still order one, and PPO sees the magnitude it was tuned for — while
the *ratio* `(1−u):u = 1:λ` and hence the gradient direction and saddle point are unchanged. The
`(1+λ)` factor is a step-size normalization, not a new optimum, and it matters most exactly on the
densest arena where λ must go largest.

For the λ step I use the OmniSafe `Lagrange` helper: it stores λ as a learnable scalar and realizes the
projected ascent through an Adam step on `−λ·(Jc − d)`, i.e. `λ ← λ + η(Jc − d)` then a nonnegative
clamp. Wrapping the single dual scalar in Adam rather than a bare fixed step smooths it against the
epoch-to-epoch Monte-Carlo noise in `J_c` — and that noise is not hypothetical, since the naive
PointGoal costs spanned about 11.6 across seeds, so one epoch's estimate can be off by that order and a
raw step would inject the full swing straight into λ. Adam's running normalization and momentum mean
only a *persistent* violation moves λ. The exposed `lambda_lr = 0.035` is deliberately small: λ climbs
on the order of `0.035 × (Jc − d)` per epoch, a fraction of a unit at a time, so reaching λ near 10 to
rescue Button takes many epochs of sustained overshoot. That slowness is a feature for stability and, I
suspect, the liability for the metric. So `_init` constructs `Lagrange(**lagrange_cfgs)`, `_update`
calls `update_lagrange_multiplier(Jc)` then `super()._update()`, and `_compute_adv_surrogate` reads
`self._lagrange.lagrangian_multiplier` into the normalized blend.

One alternative I should dismiss cleanly: driving λ on a fixed increasing schedule instead of by
measured dual ascent. A schedule is open-loop — it moves λ on a clock blind to whether *this*
environment is already under 25 or still at 150, so it over-pushes a nearly-feasible PointGoal and
under-pushes a wildly infeasible Button, rebaking the threefold mismatch the naive table exposed. Dual
ascent is closed-loop: λ rises fast where the violation is large and settles where it is small. The
schedule trades one un-settable constant for one un-settable clock; the feedback law removes both.

There is one structural worry I can already name, and it shapes the prediction. λ is the running
integral of the violation, and the environment needing the largest λ is also the one starting furthest
from feasible: Button must both climb λ to ~10 *and* drag cost down from 152, while PointGoal needs a
modest λ to close a 2× gap. The hardest lifting and the slowest-to-build weight coincide on the same
environment, and with `lambda_lr = 0.035` the integral may simply not finish climbing before the
1M-step budget runs out. That is the failure mode I test for.

So the expectations against the naive numbers. The mechanism is dual *ascent* on a single scalar driven
by the running-average violation, so it is slow — λ only settles after many epochs, and on this horizon
it may not climb high enough. I expect cost to come *down* everywhere now that the policy feels some
pressure, with the largest reduction where the violation and hence λ is largest (Button), but I am not
confident any environment actually crosses under 25. Reward should drop in lockstep, since every unit of
cost pressure is reward traded away — sharpest on Button, where the largest λ costs the most return. The
clean test of the slow-integrator hypothesis: if `budget_success_rate` is *still zero everywhere* while
cost has fallen substantially, the direction is right but the controller is structurally too late — it
lowers cost without ever crossing the line, the only thing the metric counts — and the next rung must
attack the *timing* of the response, not merely its eventual magnitude. The distilled scaffold fill is
in the answer.
