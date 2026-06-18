The naive run told me exactly what's broken, and it told me in numbers. Return is strong everywhere —
25.5 on SafetyPointGoal1, 32.8 on SafetyCarGoal1, 19.7 on SafetyPointButton1 — which confirms the
base learner optimizes reward just fine; that was never the problem. The problem is the cost column.
Against a budget of `d = 25`, the naive agent sits at 51 on PointGoal, 61 on CarGoal, and a staggering
152 on PointButton — six times over budget on the hazard-dense one, and roughly two-to-three times over
on the others, on every single seed (PointGoal 45–57, CarGoal 53–65, PointButton 140–164). There is no
lucky safe seed; the violation is systematic. That is the unmistakable signature of dropping `adv_c`:
the reward gradient points straight through the hazards, and with nothing paid for cost the agent plows
through them whenever that is the fast route to a goal. So the diagnosis is sharp — this is not a
learning problem, the agent learns the task; it is a *specification* problem. The cost is a real,
separate signal the loop is already measuring and valuing, and I am simply not using it. I need to turn
the cost stream back on and make the agent pay for violations in a way that drives that 51/61/152 down
toward 25 without throwing the return away.

The reflex fix is to mash the two signals into one scalar and hand it to the optimizer I already trust:
ascend `adv_r - beta*adv_c` for some penalty `beta`. And every time I reach for this I hit the same
wall. What is `beta`? There is no formula that takes my budget `d = 25` and returns the `beta` that
lands cost at 25. The map from penalty weight to resulting cost return depends on the whole environment
and the policy class in a way I cannot invert ahead of time — and the naive numbers prove it, because
the *same* "penalty" (here, zero) produced wildly different over-budget amounts across the three
environments (51 vs 61 vs 152). A single constant cannot possibly be the right tax for all three. If I
guess `beta` too small, the agent decides a few hazards are worth the reward and sits above budget like
the naive run; too big and it becomes so frightened of cost it never learns to navigate at all. And
even in the lucky case where some `beta` lands a safe policy at convergence, that constant tells me
nothing about the cost *during* training. So a fixed penalty is structurally the wrong object: it bakes
a trade-off I do not know how to set into a number I must commit to before seeing any data. Worse, the
one `beta` would have to do two incompatible jobs across a run — stay small early so the still-incompetent
policy can learn to reach goals at all, then push hard once the policy has found rewarding-but-risky
shortcuts. No single number does both.

So I should stop trying to fold safety into the objective and keep it as what it is — a constraint. The
clean statement is the constrained problem

  max over theta of J_r(theta), subject to J_c(theta) <= d,

where J_r and J_c are the expected episodic returns of reward and of cost, defined identically except
for which signal they sum, and d = 25. This is exactly a constrained MDP: an MDP with an extra cost
function and a threshold, whose feasible set is the policies sitting under the budget. The reason I
prefer the constraint framing over the penalty framing is not aesthetic. A threshold encodes "get cost
under 25, then stop caring about it; spend everything else on reward," which is the actual shape of the
safety requirement here. A penalty keeps trading reward against cost forever, even far inside the safe
region. The threshold has a saturation built in that the penalty lacks — and saturation is precisely
what I want, because once I am at cost 25 I should be free to chase reward, not still paying tax.

Now, how do I optimize a constrained objective when J_r and J_c are both horrible nonconvex functions
of a few hundred thousand neural-net weights? I cannot project onto the feasible set — I have no closed
form for it. The standard tool for "maximize subject to an inequality" is the Lagrangian. Write the
constraint in standard form g(theta) = J_c(theta) - d <= 0, introduce a multiplier lambda >= 0, and
form

  L(theta, lambda) = J_r(theta) - lambda*(J_c(theta) - d).

The classical result is that the constrained problem equals the max-min of this Lagrangian: I want
max over theta of min over lambda>=0 of L. Let me verify the inner min before building on it. Fix
theta. If the constraint is *violated*, g > 0, then minimizing -lambda*(positive) over lambda >= 0
drives lambda to infinity and L to minus infinity — so the outer max never picks such a theta;
infeasible policies (exactly the naive agent's situation) are punished to minus infinity. If the
constraint is satisfied with slack, g < 0, then -lambda*(negative) increases in lambda, so the min
picks lambda = 0 and L = J_r — the penalty switches off and I get pure reward, which is the naive
behavior, *correct* once I am safely under budget. On the boundary g = 0 the lambda term vanishes for
every lambda. So the inner min reconstructs exactly "reward if feasible, minus infinity if not," and at
the saddle the multiplier obeys complementary slackness, lambda*g = 0: lambda is positive only when the
constraint binds. The Lagrangian is not a heuristic stand-in; its max-min *is* the constrained problem.

That is the textbook story for convex problems, and J_r, J_c are nonconvex in theta, so I should not
assume duality works here. For a generic nonconvex program there is a duality gap, and dual methods then
chase the wrong target. But there is a reason it works for this problem, specific to RL. The
nonconvexity lives in the parameterization theta -> pi_theta, not in the problem itself: viewed in the
space of state-action occupancy measures, the reward and cost returns are both *linear* in the occupancy
measure, and the set of valid occupancy measures is convex (cut out by the Bellman flow equations). A
linear objective with linear constraints over a convex set is a linear program, and an LP has no duality
gap. So the underlying constrained-RL problem, expressed over occupancy measures, has zero duality gap
under the usual feasibility assumptions. I want to be careful: that clean duality lives in
occupancy-measure space, while I will actually take gradient steps on the neural parameters, where each
policy update is only an approximate primal step. So the honest reading is that dual descent is aimed at
the correct saddle-point structure of the true problem; my neural implementation approximates it. That
is enough license to build the algorithm — I am not chasing a phantom.

Now I have a saddle-point problem and the natural algorithm is primal-dual: ascend theta to push L up,
descend lambda to push L down, and let them chase each other. Take the lambda update first, because it
is a single scalar and it is where the adaptivity the fixed penalty lacked finally lives. I want to move
lambda to *decrease* L = J_r - lambda*g, i.e. gradient descent on lambda. The gradient of L with respect
to lambda is -g = -(J_c - d). Descent steps in the negative gradient direction:

  lambda <- lambda - eta*(-(J_c - d)) = lambda + eta*(J_c - d),

and projecting to keep lambda >= 0:

  lambda <- [ lambda + eta*(J_c - d) ]_+.

Let me re-check the sign by behavior, not just algebra, because a flipped sign would silently invert the
whole safety mechanism. If the current policy is over budget — exactly the naive 51/61/152 situation —
then J_c > d, g > 0, the bracket increases, lambda climbs, and the next policy update weighs cost more
heavily and gets pushed toward safety. Correct. If the policy is under budget, J_c < d, g < 0, the
bracket decreases, lambda falls toward zero, and the policy is freed to chase reward with the slack it
has. Also correct. So lambda does the trade-off tuning *for* me, reading the actual measured violation
each epoch — precisely the adaptive coefficient the fixed penalty could never produce. Structurally
lambda is the *integral* of the violation: it accumulates the running sum of (J_c - d), so it settles to
whatever penalty weight is needed to drive the average violation to zero. The projection is not
cosmetic: an inequality multiplier must be nonnegative (the KKT condition, the complementary-slackness
story again — lambda measures how much reward I would gain by loosening a binding budget, which cannot be
negative), and a negative lambda would mean *paying* the agent to incur cost, exactly backwards.

One subtlety on what J_c is. The constraint is on the *expected episodic* cost return, not on any single
step, so I must not update lambda from a per-timestep cost; I update from an estimate of J_c over the
epoch — the mean episodic cost across the trajectories just collected. The substrate already hands me
this: `self._logger.get_stats('Metrics/EpCost')[0]` is the mean episodic cost averaged across workers,
which is the same number the naive feedback reported (51, 61, 152). So once per `_update` epoch I read
that `Jc`, assert it is not NaN (a bad cost statistic would corrupt lambda silently and kill the
mechanism), and take one dual step from it.

Now the theta update. I am ascending theta on L = J_r - lambda*J_c (the constant lambda*d drops). The
gradient is grad J_r - lambda*grad J_c. I already have a machine that ascends an expected return given an
advantage: the PPO surrogate, where the gradient is E[grad log pi * A]. So grad J_r corresponds to the
reward advantage adv_r and grad J_c to the cost advantage adv_c — and the substrate already runs GAE
twice to produce both, which is exactly why the loop keeps two value functions. The combined gradient
grad(J_r - lambda*J_c) then corresponds to a single combined advantage

  A = adv_r - lambda*adv_c,

and I feed *that* into the PPO step I already trust through the `_compute_adv_surrogate` hook. Almost
nothing about the base optimizer changes; I have only redefined the advantage it consumes. That is the
appeal: a safe-RL fix that is the naive loop plus one scalar and one line that mixes the two advantage
streams the loop was already computing and the naive fill was throwing away.

Let me try shipping it raw — A = adv_r - lambda*adv_c — and see where it breaks. Picture training on
PointButton, where the naive cost was 152: the agent keeps blowing the budget, so the dual update keeps
incrementing lambda epoch after epoch. lambda does not stay small; it can climb to 5, 10, 50, whatever
it takes to make the agent feel a 152-vs-25 violation. Now look at the magnitude of A = adv_r -
lambda*adv_c when lambda = 50: the cost term dominates and the *scale* of the combined advantage is
roughly 50x a normal advantage. PPO's clip range, learning rate, and step-size calibration were tuned
assuming the advantage has roughly its usual magnitude. A combined advantage whose magnitude grows
linearly in lambda silently inflates the effective policy-gradient step size — the policy takes enormous,
badly-scaled steps exactly when lambda is large, which is exactly when I most need the update to be
careful. And the clip will not save me, because it clips the probability *ratio* r_t, not the advantage;
a giant A still scales the per-sample gradient the ratio multiplies. So lambda has leaked into two places
at once — the relative weight of reward versus cost, which I want, and the numerical scale of the
gradient, which I do not. Wall.

I do not want to change lambda's *meaning* to fix this — the ratio of reward weight to cost weight,
1:lambda, is the thing the dual is carefully tuning. What I want is to peel apart the two effects: keep
the *direction* of the combined advantage but normalize its *scale* so PPO sees a roughly unit-scale
advantage regardless of how large lambda grows. The cleanest normalization that is monotone in lambda
and equals 1 at lambda = 0 is dividing by (1 + lambda):

  A = (adv_r - lambda*adv_c) / (1 + lambda).

Check it does what I want. Set u = lambda/(1+lambda). Then 1/(1+lambda) = 1 - u and lambda/(1+lambda) =
u, so A = (1 - u)*adv_r - u*adv_c with u in [0, 1). Now it reads as a convex blend of the reward
advantage and the negated cost advantage. At lambda = 0: u = 0, A = adv_r — pure reward, the naive
update, which is right because lambda = 0 means the constraint is slack and I should pay no safety tax.
As lambda grows large: u -> 1, A -> -adv_c — pure cost-avoidance, right because a huge lambda means the
budget is being badly violated (the PointButton case) and reward should be sacrificed to fix it. In
between, since the two weights 1-u and u are in [0,1] and sum to 1, |A| stays on the order of the
individual advantages no matter how large lambda gets, so PPO's tuned step size and clip behave the same
throughout. And critically, the *ratio* of reward weight to cost weight is (1-u):u = 1:lambda, identical
to the un-normalized version, so the gradient *direction* in policy space and the dual stationary point
are unchanged. The (1+lambda) factor is a learning-rate normalization that keeps the first-order update
sane; it does not move the target. I have rescaled the step, not the optimum — forced by the scale
problem, not decorative.

There is an implementation choice in how I take the lambda step. Plain projected gradient ascent is
lambda <- [lambda + eta*(J_c - d)]_+, and I can realize it cleanly by keeping lambda as a learnable
scalar `torch.nn.Parameter` and defining a loss whose gradient is exactly the descent direction:
loss_lambda = -lambda*(J_c - d). Then the gradient with respect to lambda is -(J_c - d), so an Adam step
gives lambda <- lambda + eta*(J_c - d), exactly the dual ascent. Wrapping it in Adam (rather than a bare
fixed step) also smooths the single dual scalar against the epoch-to-epoch noise in the Monte-Carlo
estimate of J_c — and there *is* noise here, since the naive PointButton cost ranged 140–164 across
seeds. After the optimizer step I clamp lambda.data to the nonnegative range, which is the projection.
The advantage hook reads lambda directly via `.item()`, so a directly-stored, directly-clamped scalar is
the simplest thing that hook can consume. The dual step size eta is the substrate's `lambda_lr`, default
0.035. Order within an epoch: read J_c, update lambda from it *first*, then run `super()._update()` with
the freshly-updated lambda inside the combined advantage, then log lambda so I can watch it climb when
the agent is unsafe and relax when it is safe.

I should name the one weakness of this loop up front, because it is real and intrinsic to the
integral-only character of lambda, and it predicts where the *next* step will have to go. lambda is the
integral of the violation: it responds only to *accumulated* over-budget, so it lags. When the agent is
deep over budget — the 152 on PointButton — lambda has to integrate up from wherever it was, several
epochs, and by the time it is large enough to rein the agent in it has overshot; then once cost drops
below budget, lambda stays large for a while (it integrates down equally slowly), over-corrects toward
timidity, cost undershoots, lambda falls, cost rises again. The result is oscillation of cost around the
budget rather than a clean settle. This is the generic behavior of a pure-integral controller on a
delayed system, and it is the flip side of the appeal: a tiny change to a strong optimizer, but one that
reacts only after observed cost. The integral term is exactly what makes lambda converge to the right
steady-state penalty with zero steady-state violation, which is why I keep it — but the oscillation is
the obvious thing the next mechanism will attack by adding faster-reacting terms.

So my falsifiable expectations against the naive numbers. The dual loop should pull cost *down* from
51/61/152 toward 25 on all three environments — that is the whole point, and if it does not, the sign or
the blend is wrong. In exchange I expect return to *drop* below the naive 25.5/32.8/19.7, because every
unit of cost removed is paid for in reward forgone; the question the result answers is how much. But I do
*not* expect cost to land cleanly at 25. Because lambda only reacts to accumulated violation, I expect it
to still sit *over* budget — reduced from the naive level but not controlled, especially on PointButton
where the violation it has to integrate against is largest — with cost oscillating rather than settling.
If cost comes down substantially but remains above 25, that is the integral-only lag showing, and it is
exactly the diagnosis that motivates replacing the pure integrator with a controller that also reacts to
the present and anticipates the trend. The full scaffold module is in the answer.
