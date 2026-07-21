The naive run told me exactly what's broken, and it told me in numbers. Return is strong everywhere —
25.5 on SafetyPointGoal1, 32.8 on SafetyCarGoal1, 19.7 on SafetyPointButton1 — which confirms the
base learner optimizes reward just fine; that was never the problem. The problem is the cost column.
Against a budget of `d = 25`, the naive agent sits at 51 on PointGoal, 61 on CarGoal, and a staggering
152 on PointButton — six times over budget on the hazard-dense one, two-to-three times over on the
others, on every single seed (PointGoal 45–57, CarGoal 53–65, PointButton 140–164). There is no lucky
safe seed; the violation is systematic. That is the signature of dropping `adv_c`: the reward gradient
points straight through the hazards, and with nothing paid for cost the agent plows through them
whenever that is the fast route to a goal. So the diagnosis is sharp — not a learning problem, the
agent learns the task; a *specification* problem. The cost is a real, separate signal the loop is
already measuring, and I am simply not using it. I need to turn the cost stream back on and make the
agent pay for violations in a way that drives 51/61/152 down toward 25 without throwing return away.

The reflex fix is to mash the two signals into one scalar and hand it to the optimizer I already
trust: ascend `adv_r - beta*adv_c` for some penalty `beta`. And every time I reach for this I hit the
same wall. What is `beta`? There is no formula that takes my budget `d = 25` and returns the `beta`
that lands cost at 25 — the map from penalty weight to resulting cost return depends on the whole
environment and policy class in a way I cannot invert ahead of time. And the naive numbers prove it:
the *same* penalty (here, zero) produced over-budget ratios of about 2.0, 2.4, and 6.1 across the
three environments, so the weight that drags PointButton from 152 to 25 has to overcome roughly three
times the excess PointGoal's 51 presents. A single constant cannot be environment-dependent; even
within PointButton the 140–164 seed spread says the "right" constant would be wrong seed to seed. And
even where some `beta` lands a safe policy at convergence, the one number would have to do two
incompatible jobs across a run — stay small early so the still-incompetent policy can learn to reach
goals, then push hard once it finds rewarding-but-risky shortcuts. The object I need is not a better
constant; it is something that *reads* the realized cost and sets its own weight from the measured gap.

So I stop trying to fold safety into the objective and keep it as what it is — a constraint:

  max over theta of J_r(theta), subject to J_c(theta) <= d,

with J_r, J_c the expected episodic returns of reward and cost and d = 25. I prefer the constraint
framing over the penalty framing for a concrete reason: a threshold encodes "get cost under 25, then
stop caring about it; spend everything else on reward," which is the actual shape of the requirement.
A penalty keeps trading reward against cost forever, even far inside the safe region. The threshold
has a saturation the penalty lacks, and saturation is exactly what I want — once I am at cost 25 I
should be free to chase reward, not still paying tax.

To optimize a constrained objective when J_r and J_c are horrible nonconvex functions of a few
hundred thousand weights, I cannot project onto the feasible set — I have no closed form for it. The
standard tool for "maximize subject to an inequality" is the Lagrangian: write the constraint as
g(theta) = J_c(theta) - d <= 0, introduce lambda >= 0, and form

  L(theta, lambda) = J_r(theta) - lambda*(J_c(theta) - d).

The constrained problem is the max-min of L, max over theta of min over lambda>=0. The inner min does
what I want: if the constraint is violated it drives lambda up and L down, so the outer max never
picks an infeasible policy; if it is satisfied with slack the min picks lambda = 0 and L = J_r — the
penalty switches off, correct once safely under budget. At the saddle, complementary slackness
lambda*g = 0 holds: lambda is positive only when the constraint binds. The one worry is that L is
nonconvex in theta, so I should not assume duality works — but the nonconvexity lives in the
parameterization, not the problem: the context's occupancy-measure view makes the returns linear and
the feasible set convex with zero duality gap. So dual descent is aimed at the correct saddle-point
structure and my neural steps only approximate it. Enough license to build the algorithm.

The natural algorithm for a saddle point is primal-dual: ascend theta to push L up, descend lambda to
push it down. Take the lambda update first, because it is a single scalar and it is where the
adaptivity the fixed penalty lacked lives. Descending L = J_r - lambda*g in lambda has gradient
-g = -(J_c - d), so

  lambda <- [ lambda + eta*(J_c - d) ]_+,

projected nonnegative. Read by behavior: over budget (the naive 51/61/152) means J_c > d, lambda
climbs, and the next policy update weighs cost more heavily; under budget means lambda falls toward
zero, freeing the policy to chase reward with its slack. lambda does the trade-off tuning for me, from
the actual measured violation each epoch. Structurally it is the *integral* of the violation —
accumulating the running sum of (J_c - d), settling to whatever penalty weight drives the average
violation to zero — and the projection is not cosmetic: an inequality multiplier must be nonnegative,
since a negative lambda would pay the agent to incur cost. The constraint is on *expected episodic*
cost, so I update lambda not from a per-timestep cost but from the mean episodic cost over the epoch,
which the substrate hands me as `self._logger.get_stats('Metrics/EpCost')[0]` — the same 51/61/152 the
naive feedback reported.

Now the theta update. Ascending theta on L = J_r - lambda*J_c (the constant lambda*d drops), the
gradient is grad J_r - lambda*grad J_c. The PPO surrogate already ascends an expected return given an
advantage, gradient E[grad log pi * A], so grad J_r corresponds to adv_r and grad J_c to adv_c — the
two advantages the substrate runs GAE twice to produce, which is exactly why the loop keeps two value
functions — and the combined gradient corresponds to a single combined advantage
A = adv_r - lambda*adv_c, fed through `_compute_adv_surrogate`. Almost nothing about the base
optimizer changes; I have only redefined the advantage it consumes.

But shipping A = adv_r - lambda*adv_c raw breaks on the deep violators. On PointButton, where naive
cost was 152, the agent keeps blowing the budget so lambda keeps incrementing — it does not stay
small, it can climb to 50 or whatever it takes to make a 152-vs-25 violation felt. At lambda = 50 the
cost term dominates and the *scale* of A is roughly 50x a normal advantage. PPO's clip range, learning
rate, and step-size calibration all assume the advantage has its usual magnitude, so an A whose
magnitude grows linearly in lambda silently inflates the effective policy-gradient step exactly when
lambda is large — exactly when I most need the update careful. The clip does not save me: it clips the
probability *ratio*, not the advantage, and a giant A still scales the per-sample gradient the ratio
multiplies. lambda has leaked into two places — the relative weight of reward versus cost, which I
want, and the numerical scale of the gradient, which I do not.

I do not want to change lambda's *meaning* to fix this; the ratio 1:lambda of reward to cost weight is
what the dual is tuning. I want to keep the *direction* of A but normalize its *scale* so PPO sees a
roughly unit-scale advantage regardless of lambda. The cleanest normalization that is monotone in
lambda and equals 1 at lambda = 0 is dividing by (1 + lambda):

  A = (adv_r - lambda*adv_c) / (1 + lambda).

Setting u = lambda/(1+lambda), this is A = (1 - u)*adv_r - u*adv_c with u in [0, 1) — a convex blend.
At lambda = 0, u = 0, A = adv_r (pure reward, the naive update, right because the constraint is slack);
as lambda grows, u -> 1 and A -> -adv_c (pure cost-avoidance, right because a huge lambda means the
budget is badly violated). In between the two weights sum to one, so |A| stays on the order of a single
advantage no matter how large lambda gets and PPO's tuned step size and clip behave the same
throughout. The reward-to-cost ratio (1-u):u = 1:lambda is identical to the un-normalized version, so
the gradient direction and dual stationary point are unchanged: (1+lambda) is a learning-rate
normalization, not a change of target. That the factor is (1 + lambda) rather than lambda is what keeps
lambda = 0 mapping to pure reward; a bare 1/lambda would blow up there.

For the lambda step I keep lambda as a `torch.nn.Parameter` and define loss_lambda = -lambda*(J_c - d),
whose gradient in lambda is -(J_c - d) so an Adam step gives lambda <- lambda + eta*(J_c - d), the dual
ascent; then I clamp lambda.data nonnegative and the hook reads lambda via `.item()`. The dual step
size eta is the substrate's `lambda_lr = 0.035`. Order within the epoch matters: read J_c, update
lambda from it *first*, then run `super()._update()` with the freshly-updated lambda in the combined
advantage, then log. Reversing it would have the policy step consume the *previous* epoch's lambda
even though this epoch's J_c is in hand, injecting a one-epoch staleness into the exact quantity the
mechanism is trying to make responsive. Wrapping the scalar in Adam rather than a bare fixed step also
smooths it against the Monte-Carlo noise in J_c — the naive PointButton cost ranged 140–164 — so a
persistently one-signed over-budget pressure becomes a steady climb while the zero-mean jitter averages
down.

The one weakness of this loop is intrinsic to lambda's integral character: lambda responds only to
*accumulated* over-budget, so it lags. Deep over budget it has to integrate up over several epochs,
and by the time it is large enough to rein the agent in it has overshot; then once cost drops below
budget it stays large a while, over-corrects toward timidity, cost undershoots, lambda falls, cost
rises — the generic behavior of a pure integrator on a delayed system. The integral term is exactly
what makes lambda converge to the right steady-state penalty with zero steady-state violation, which
is why I keep it, but the oscillation is what a richer controller with faster-reacting terms would
attack.

So, against the naive numbers: the dual loop should pull cost *down* from 51/61/152 toward 25 on all
three environments — if it does not, the sign or the blend is wrong — while return *drops* below the
naive 25.5/32.8/19.7, since every unit of cost removed is paid in reward forgone. But I do *not*
expect cost to land cleanly at 25: because lambda reacts only to accumulated violation, I expect it
reduced but still *over* budget, especially on PointButton where the violation it integrates against
is largest, and oscillating rather than settling. Cost coming down substantially but staying above 25
is the integral-only lag showing, and it is exactly the diagnosis that would motivate replacing the
pure integrator with a rule that reacts faster than accumulation allows. The full module is in the
answer.
