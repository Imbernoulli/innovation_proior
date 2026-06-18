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

The reflex is to mash the two signals into one scalar and hand it to the PPO backbone I already
trust — ascend `E[Σ(r_t − β c_t)]` for some penalty weight β. And every time I reach for this I hit the
same wall: what is β? There is no formula that takes my budget `d = 25` and returns the β that achieves
it; the map from penalty coefficient to resulting cost return depends on the whole environment and the
policy class in a way I cannot invert ahead of time. Worse, the naive numbers show me the trade-off
*moves across environments* — PointGoal needs the cost halved, Button needs it cut by a factor of six.
A single β that pulls Button's 152 down under 25 would be enormous, and on PointGoal that same β would
terrify the policy off the task entirely. And even within one run, the one β has to do two
incompatible jobs: stay small early so the still-incompetent policy can learn to navigate at all, then
push hard once it has found the rewarding-but-risky shortcuts the naive run lived on. No single number
does both. A fixed penalty bakes a trade-off I don't know how to set into a constant I have to set
before I see anything.

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
violation signal, so it settles to whatever weight drives the average violation to zero. One subtlety:
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
identically throughout. The *ratio* of the weights is still `(1−u):u = 1:λ`, so the gradient direction
and the saddle point are unchanged; the `(1+λ)` factor is a step-size normalization, not a new optimum.

For the λ step itself I use the OmniSafe `Lagrange` helper the harness exposes: it stores λ as a
learnable scalar and realizes the projected ascent through an Adam step on the loss
`−λ·(Jc − d)`, whose gradient is `−(Jc − d)`, giving `λ ← λ + η·(Jc − d)` followed by a nonnegative
clamp. Wrapping the single dual scalar in Adam (rather than a bare fixed-step update) smooths it against
the epoch-to-epoch Monte-Carlo noise in the estimate of `J_c`, with `lambda_lr` the dual learning rate.
So `_init` constructs `Lagrange(**lagrange_cfgs)`, `_update` calls `update_lagrange_multiplier(Jc)` then
`super()._update()`, and `_compute_adv_surrogate` reads `self._lagrange.lagrangian_multiplier` into the
normalized blend.

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
budget, which is the only thing the metric counts. That failure is what would force a controller that
reacts to the *rate* of violation and not just its accumulated integral. The distilled scaffold fill
is in the answer.
