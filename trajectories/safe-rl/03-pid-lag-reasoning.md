The dual loop did the thing it was supposed to, and it did it incompletely, and the numbers say exactly
where it fell short. Cost came down everywhere relative to naive — PointGoal from 51 to 45.6, CarGoal
from 61 to 46.7, PointButton from 152 to 56.2 — so the sign is right, the blend is right, lambda is
reading the violation and paying for it, and on PointButton it removed almost a hundred cost-units. And
the price was steep on return, as expected: PointGoal fell from 25.5 to 15.1, CarGoal from 32.8 to 18.6,
PointButton from 19.7 to 4.0. So far this is the constrained trade-off doing its job. But look at the
cost column against the budget: 45.6, 46.7, 56.2 — *still nearly twice over 25 on every environment, on
every seed* (PointGoal 42–49, CarGoal 41–50, PointButton 51–60). It paid heavily in reward and *still
did not satisfy the constraint*. That is the worst of both worlds: it gave up the reward of a safe method
without delivering the safety. The mechanism is converging toward feasibility too slowly to get there
inside the training budget.

This is exactly the integral-only lag I flagged when I built the dual loop, and now I can see it in the
data rather than just predict it. lambda is the integral of the violation: it responds only to
accumulated over-budget, so it climbs slowly from where it started, and on these runs it simply has not
integrated up to the value that would push cost to 25 by the time training ends. The PointButton seeds
are the tell — cost pinned in the 51–60 band with return crushed to 3–5 means lambda is large enough to
wreck the reward but still not large enough to enforce the budget, and it is stuck there because the
integrator cannot react any faster than its accumulation rate allows. I do not want to fix this by simply
cranking the dual step size `lambda_lr`, because a faster integrator on a delayed plant does not track
better — it oscillates harder. So the right move is not to tune the one knob; it is to ask what *kind* of
update rule lambda should obey, and a century of control theory has opinions about that.

Let me reframe the whole training loop as the feedback control system it actually is, because that
reframe is what opens the space of update rules. The cost limit d = 25 is a setpoint. The measured mean
episodic cost J_c — the 45.6/46.7/56.2 I am reading off the logger — is the output I want to hold at the
setpoint. The multiplier lambda is the control input I get to set. And the policy-optimization step, PPO
grinding the policy forward to chase reward under the current lambda, is the *plant*: a nonlinear, frankly
unknown map from lambda to next epoch's cost. Write it as a discrete system: the policy advances under
lambda, the output is the resulting episodic cost, and lambda is some rule applied to the history of that
output relative to d. The dual update I shipped is one specific, impoverished choice of that rule:
lambda <- [lambda + K_I*(J_c - d)]_+ is *integral-only* control. Once I see it that way, the question
stops being "how do I tune the Lagrangian update" and becomes "what controller should I use."

Before reaching for a richer controller I want to be sure I understand mechanically why integral-only
control lags and overshoots, because the cure has to attack the cause. Integral control has memory but no
reflexes. It reacts only to *accumulated* error — lambda rises only after violation has been piling up.
So the response is intrinsically late. By the time lambda has integrated up to a value large enough to
push cost down, cost has already been over budget for a long stretch (the 45–60 plateau I am seeing is
that stretch, just not yet resolved within the budget). And the continuous-time picture Platt and Barr
wrote down, lambda_dot = alpha*(J_c - d), makes it precise: the multiplier is literally the time-integral
of the constraint error. A pure integrator turns a sinusoid at its input into one shifted by ninety
degrees at its output — so when the system does start to cycle, violation and lambda are offset by a
quarter cycle, the textbook fingerprint of an integral controller. My runs have not even reached the
clean limit-cycle regime; they are stuck in the slow approach, which is the same disease seen earlier in
the trajectory: the integrator is too slow to track.

What does a controls person reach for when an integral loop is too slow and rings? The other two terms of
the most-used controller there is: proportional and derivative. Let me bring them in one at a time and
check, in this same dynamical-systems picture, that they actually do what I hope.

Proportional first. The idea is to add to lambda a term proportional to the *current* error,
K_P*(J_c - d) — an instantaneous reaction, no waiting for accumulation. To see its effect on the
dynamics I express it in continuous time. The integral term was lambda_dot = alpha*g, with g = J_c - d.
A proportional term enters lambda_dot as a contribution that tracks g's rate, i.e. as the time-derivative
of the constraint: lambda_dot = alpha*g + beta*g_dot. Redoing Platt and Barr's collapse of the coupled
first-order system into one second-order equation per coordinate, the integral-only case gives a damped
oscillator x_ddot + A*x_dot + alpha*g*grad_g = 0 with damping matrix A. Adding the beta*g_dot piece and
regathering terms changes only the damping matrix: it gains beta*(grad_g)(grad_g)^T. That term is the
outer product of grad_g with itself, positive semidefinite by construction — v^T (grad_g grad_g^T) v =
(grad_g . v)^2 >= 0 for any v. So it can only *add* nonnegative damping in the grad_g direction. The
restoring force alpha*g*grad_g is untouched, so the equilibrium does not move — the solution set is still
"velocity zero, g = 0." The proportional term provides pure damping, exactly what kills an oscillation,
and it falls straight out of the same energy argument: the dissipation term gains -beta*(grad_g . x_dot)^2.

There is something satisfying here, because this beta*(grad_g)(grad_g)^T term is what the quadratic
penalty method also produces — adding (c/2)*g^2 to the objective hands you the same outer-product damping.
But the penalty pays for it twice: it has to modify the *primal* update (the penalty force lives in the
policy dynamics, changing the objective the agent optimizes), and its damping comes bundled with an extra
c*g*grad^2_g term carrying the constraint Hessian, which need not be positive semidefinite — so the
penalty helps and hurts on a single coefficient I cannot split. The proportional control term gives the
clean positive-semidefinite damping with none of that baggage: it touches only the lambda update, not the
policy objective. So proportional control is not just a heuristic transplant; in this system it is a
cleaner route to the penalty method's known benefit — and it directly addresses my ppo_lag plateau,
because a proportional term reacts to the *current* 45-vs-25 gap immediately instead of waiting for the
integrator to accumulate it.

Now derivative. Proportional reacts to the present; the derivative term should let me react to the
*future* — anticipate. The intuition: if cost is rising fast toward the limit, raise lambda now, before
cost crosses d. A term on the *trend* of the constraint enters lambda_dot one derivative higher, as the
second derivative of the constraint: lambda_dot = alpha*g + gamma*g_ddot. Grinding out g_ddot, it
contains x_ddot — the acceleration appears on both sides — and pushing lambda_dot into the per-coordinate
equation and collecting, the x_ddot terms couple through B = I + gamma*(grad_g)(grad_g)^T, identity plus a
positive-semidefinite outer product, hence positive definite and invertible. Left-multiply by B^{-1} to
decouple, and two effects appear. First, B^{-1} has eigenvalues no greater than one (strictly smaller in
the grad_g direction), so it rescales the restoring direction and reshapes the damping operator to
B^{-1}*A. Second — the predictive part — a scalar quadratic in the velocity and modulated by the
constraint curvature along the direction of motion multiplies B^{-1}*grad_g; in the acceleration equation
its sign is such that when g is already positive and the motion is into a region where g curves upward
(violation about to get worse), this anticipatory force *adds* to the restoring force and brakes harder
before overshooting. So the derivative term reads the trend and acts ahead of the violation — exactly the
anticipation the slow integrator lacks. It is also the most delicate term, the one most prone to
amplifying noise, because second differences of a noisy signal are wild; I keep that flag for estimation.

One asymmetry I bake in for the cost case specifically. Derivative control on cost should *brake*
increases — that is the dangerous direction, toward violation. When cost is *decreasing*, I do not want
the derivative term fighting that; a falling cost is good. So I rectify the derivative term: use the
positive part of the cost-increase rate, so it acts against increases and stays silent on decreases. A
one-sided derivative for a one-sided (inequality) constraint.

So the full controller is the combination: proportional damps, derivative anticipates, integral does the
one thing neither can. Why keep integral at all, given it is what lagged? Because at convergence I need
*zero* steady-state violation, and only the integral term supplies the standing lambda that holds cost
exactly at 25. When the system settles, the error J_c - d is about zero, so the proportional term vanishes
and the derivative term (cost not changing) vanishes too — if those were my only terms, lambda would
collapse and cost would drift back up. The integral term has *remembered* the accumulated history, so it
holds a nonzero lambda even at zero instantaneous error. Integral eliminates steady-state offset;
proportional and derivative shape the *transient* — the very thing the ppo_lag plateau is failing at.
And nicely, setting K_P = K_D = 0 recovers the integral-only dual update exactly, so this is a strict
generalization: I am widening the update rule from a one-knob family to a three-knob one, with the prior
baseline sitting at the origin of the two new axes.

Now I move to the discrete per-epoch rule I will actually run, against the task's exact implementation.
Each epoch I receive the mean episodic cost J_c from the logger — the same `get_stats('Metrics/EpCost')`
the dual loop read. Define the error delta = J_c - d. The integral is the running accumulation of the
error with its gain folded in, projected nonnegative for anti-windup (I do not want the integrator banking
a negative reservoir during a feasible stretch — that would delay its reaction, the exact disease I am
curing): I <- max(0, I + K_I*delta). The proportional input is the error, but smoothed against the noisy
minibatch estimate — and there *is* noise, the ppo_lag PointButton cost ranged 51–60 across seeds. The
task smooths it with an exponential moving average, delta_p <- 0.95*delta_p + 0.05*delta, a long window so
one bad epoch cannot jerk the controller. The derivative works off a smoothed cost, cost_d <-
0.95*cost_d + 0.05*J_c, and rather than a jittery one-step difference it takes a *delayed* difference: keep
a short queue of past smoothed costs, and compute pid_d = max(0, cost_d - cost_ds[oldest]), the rectified
finite difference over the delay window. The task fixes that delay at 10 epochs and the EMA coefficient at
0.95 inline — these are the CPPOPID configuration, with gains K_P = 0.1, K_I = 0.01, K_D = 0.01. The
output is pid_o = K_P*delta_p + I + K_D*pid_d, with I already carrying its K_I gain, and lambda =
max(0, pid_o) for the nonnegativity the KKT condition demands. Then push cost_d onto the queue to advance
the delay.

I want to be careful to derive against *this* harness and not the fuller controller it descends from. The
general controller has a sum-norm and a diff-norm mode and a penalty_max cap, and configurable EMA alphas;
this task exposes none of those — no normalization branches, no upper cap, EMA fixed at 0.95, delay fixed
at 10, integral seeded at 0.0 (so lambda starts at zero, like the dual loop did). So the update I actually
ship is the clean PID core: integral with anti-windup, EMA-smoothed proportional, rectified delayed
derivative, summed with the three gains and clamped at zero. The gains are small — 0.1, 0.01, 0.01 — which
is the right regime for a deep-RL plant: the integral as the slow memory that fixes the steady-state, and
proportional and derivative giving fast, *shaped* responses on top, so I get responsiveness without having
to push K_I into the regime that ruins reward.

The advantage blend is unchanged from the dual loop, and deliberately so. The large-lambda step-size
blowup is the same problem regardless of how lambda is computed, and the same fix applies: arg max of
(J_r - lambda*J_c) equals arg max of (J_r - lambda*J_c)/(1 + lambda), since dividing by the positive
constant 1 + lambda does not move the argmax but does normalize the gradient magnitude as lambda grows. So
`_compute_adv_surrogate` returns (adv_r - lambda*adv_c)/(1 + lambda) exactly as before — and using the
identical blend is what makes this a clean comparison: the only thing that changed between this step and
the last is the *controller* on lambda, not the step-size handling, so any difference in the result is
attributable to PID versus integral-only. The two separate value functions the substrate keeps are also
exactly right here and more important than before: because lambda now changes *rapidly* under the
proportional and derivative terms, a single critic trained on a moving r + lambda*c target would be
perpetually stale, whereas the substrate's fixed-target reward and cost critics stay valid and are blended
only at the policy-gradient stage by the current lambda.

So my falsifiable expectations against the ppo_lag numbers. The dual loop left cost at 45.6/46.7/56.2 —
roughly twice over budget on every environment, despite a heavy reward cost. The PID controller should
*actually reach the budget*: I expect cost to land at or below 25 on all three environments, because the
proportional and derivative terms react to the current gap and the rising trend immediately instead of
waiting for the integrator to crawl up, while the integral term still guarantees it does not overshoot
into permanent timidity. Concretely, I expect PointButton to come down from 56 into the mid-20s or below
— the case where the slow integrator was most visibly stuck — and the other two from the mid-40s to under
25. The return story is the honest cost of finally satisfying the constraint: I do *not* expect return to
recover toward the naive level, and it may well drop *below* the ppo_lag 15/19/4, because actually holding
cost at 25 (rather than hovering at 46) forces the agent to detour around hazards it was previously cutting
through. If cost lands controlled at the budget across all three environments — the thing neither naive nor
the integral-only dual loop achieved — then on the actual safe-RL objective (high return *only when* the
constraint is controlled) this is the first method on the ladder that succeeds, and that is the bar it has
to clear. The full scaffold module is in the answer.
