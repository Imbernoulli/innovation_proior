The dual loop did the thing it was supposed to, and it did it incompletely, and the numbers say exactly
where it fell short. Cost came down everywhere relative to naive — PointGoal from 51 to 45.6, CarGoal
from 61 to 46.7, PointButton from 152 to 56.2 — so the sign is right, the blend is right, lambda is
reading the violation and paying for it, and on PointButton it removed almost a hundred cost-units. And
the price was steep on return: PointGoal fell from 25.5 to 15.1, CarGoal from 32.8 to 18.6, PointButton
from 19.7 to 4.0. But against the budget the cost column reads 45.6, 46.7, 56.2 — still nearly twice
over 25 on every environment and every seed (PointGoal 42–49, CarGoal 41–50, PointButton 51–60). It
paid heavily in reward and *still did not satisfy the constraint*. The over-budget ratios are 1.82,
1.87, 2.25 — essentially unchanged in character despite very different naive starting points — while
the return outlay was about −41%, −43%, and −80%. That is the worst of both worlds: the integrator
bought almost no feasibility with a large reward outlay. A method that paid 40–80% of its return should
be at or near the budget; this one is nowhere close. The mismatch is the signature that the *rate* of
the controller, not its eventual target, is broken — aimed at the right place and crawling there too
slowly to arrive inside the training budget.

This is exactly the integral-only lag I flagged when I built the dual loop, now visible in data. lambda
is the integral of the violation: it climbs only as accumulated over-budget piles up, and on these runs
it has not integrated up to the value that would push cost to 25 before training ends. The PointButton
seeds are the tell — cost pinned in the 51–60 band with return crushed to 3–5 means lambda is large
enough to wreck reward but still not large enough to enforce the budget, stuck because the integrator
cannot react faster than its accumulation rate. I do not want to fix this by cranking `lambda_lr`,
because a faster integrator on a delayed plant does not track better — it oscillates harder. The right
move is to ask what *kind* of update rule lambda should obey.

Reframe the training loop as the feedback control system it is. The cost limit d = 25 is a setpoint.
The measured mean episodic cost J_c — the 45.6/46.7/56.2 off the logger — is the output I want to hold
at the setpoint. The multiplier lambda is the control input. And the policy-optimization step, PPO
grinding the policy forward under the current lambda, is the *plant*: a nonlinear, unknown map from
lambda to next epoch's cost. The dual update lambda <- [lambda + K_I*(J_c - d)]_+ is then one
impoverished choice of controller — *integral-only*. Once I see it that way, the question stops being
"how do I tune the Lagrangian update" and becomes "what controller should I use."

Integral control has memory but no reflexes: it reacts only to *accumulated* error, so lambda rises
only after violation has been piling up and the response is intrinsically late — the 45–60 plateau is
that stretch, not yet resolved. Platt and Barr's continuous picture, lambda_dot = alpha*(J_c - d),
makes it precise: the multiplier is the time-integral of the constraint error, and a pure integrator
shifts a sinusoid ninety degrees, so once the system cycles, violation and lambda sit a quarter cycle
apart. My runs are stuck in the slow approach, the same disease just before the limit cycle. What a
controls person reaches for when an integral loop is too slow and rings is the other two terms:
proportional and derivative.

Proportional adds a term on the *current* error, K_P*(J_c - d), an instantaneous reaction with no
waiting for accumulation. In Platt and Barr's second-order collapse of the coupled system, the
integral-only case is a damped oscillator x_ddot + A*x_dot + alpha*g*grad_g = 0, and a proportional
term enters as beta*g_dot, adding beta*(grad_g)(grad_g)^T to the damping matrix — an outer product,
positive semidefinite (v^T (grad_g grad_g^T) v = (grad_g . v)^2 >= 0). So it can only add nonnegative
damping; the restoring force and equilibrium are untouched. Proportional control is pure damping,
exactly what kills an oscillation. And this same beta*(grad_g)(grad_g)^T is what the quadratic-penalty
method produces — but the penalty pays for it twice: its force lives in the *primal* update, changing
the objective the agent optimizes, and its damping comes bundled with an extra c*g*grad^2_g term
carrying the constraint Hessian, which need not be positive semidefinite, so it helps and hurts on a
single coefficient I cannot split. Proportional control gives the clean PSD damping with none of that,
touching only the lambda update, and it reacts to the current 45-vs-25 gap immediately.

Derivative reacts to the *future*: if cost is rising fast toward the limit, raise lambda now, before it
crosses d. A term on the trend enters one derivative higher, lambda_dot = alpha*g + gamma*g_ddot. The
x_ddot terms then couple through B = I + gamma*(grad_g)(grad_g)^T, identity plus a PSD outer product,
hence positive definite and invertible; left-multiplying by B^{-1} decouples and produces a
velocity-quadratic force modulated by the constraint curvature along the direction of motion, so when g
is already positive and the motion is into a region where g curves upward — violation about to worsen —
this anticipatory force adds to the restoring force and brakes harder before overshooting. It is the
most delicate term, most prone to amplifying noise, because second differences of a noisy signal are
wild. One asymmetry I bake in for cost specifically: derivative control should *brake* increases toward
violation but stay silent when cost is *decreasing*, since falling cost is good, so I rectify to the
positive part of the cost-increase rate — a one-sided derivative for a one-sided constraint.

So the full controller: proportional damps, derivative anticipates, integral does the one thing neither
can. Why keep integral at all, given it is what lagged? Because at convergence I need *zero*
steady-state violation, and only the integral supplies the standing lambda that holds cost exactly at
25. When the system settles the error is about zero, so the proportional term vanishes and the
derivative term (cost not changing) vanishes too — if those were my only terms lambda would collapse
and cost would drift back up. The integral has remembered the accumulated history and holds a nonzero
lambda at zero instantaneous error. And setting K_P = K_D = 0 recovers the integral-only dual update
exactly, so this is a strict generalization: the same one-knob family widened to three, with the prior
baseline at the origin of the two new axes.

Now the discrete per-epoch rule. Each epoch I receive J_c and set delta = J_c - d. The integral is the
running accumulation with its gain folded in, projected nonnegative for anti-windup:
I <- max(0, I + K_I*delta). The proportional input is the error smoothed against the noisy minibatch
estimate — the ppo_lag PointButton cost ranged 51–60 — with a long-window EMA,
delta_p <- 0.95*delta_p + 0.05*delta, so one bad epoch cannot jerk the controller. The derivative works
off a smoothed cost, cost_d <- 0.95*cost_d + 0.05*J_c, and rather than a jittery one-step difference it
takes a delayed difference over a short queue, pid_d = max(0, cost_d - cost_ds[oldest]), rectified over
the delay window. I fix the delay at 10 epochs and the EMA coefficient at 0.95, with gains K_P = 0.1,
K_I = 0.01, K_D = 0.01. The output is pid_o = K_P*delta_p + I + K_D*pid_d, with I already carrying its
K_I gain, and lambda = max(0, pid_o) for the nonnegativity KKT demands. The advantage blend is
unchanged from the dual loop — `(adv_r - lambda*adv_c)/(1 + lambda)` — since the large-lambda
step-size blowup is the same problem regardless of how lambda is computed, and keeping the identical
blend makes this a clean comparison in which the only thing that changed is the *controller* on lambda.

The gains are not arbitrary; K_P = 0.1 is chosen to cure the lag K_I = 0.01 suffered. Take the ppo_lag
PointGoal plateau, cost around 45.6, delta ≈ 20. The proportional contribution is K_P*delta ≈ 2,
delivered *this epoch*; the integral contribution per epoch is K_I*delta ≈ 0.2. So the integrator needs
about ten epochs of sustained 20-over-budget to accumulate what proportional supplies in one — the
factor of ten is exactly K_P/K_I. That is the lag made quantitative: the pure integrator was always
ten-ish epochs behind the multiplier the current gap already justified, and over a finite training
budget those lost epochs are why cost never reached 25. The proportional term front-loads that standing
value immediately, and because it acts on the *smoothed* error the size-2 kick is against the trend,
not a single noisy epoch.

The smoothing constants are time scales. An EMA with coefficient 0.95 has an effective window of about
1/(1 - 0.95) = 20 epochs — long enough to average down the 51–60 seed jitter before it reaches the
proportional or derivative terms, short enough to still move within a run. The derivative's delayed
difference over a 10-epoch queue measures the cost trend across a real span, so the difference of two
smoothed values is a genuine trend and not quantization noise: the derivative is doubly protected,
differencing an already-EMA-smoothed cost over a ten-epoch baseline, the only way a
second-difference-like quantity is usable on a signal with 16%-scale jitter. The anti-windup
I <- max(0, I + K_I*delta) is aimed at the same disease: during a feasible stretch delta < 0, so
without the clamp the integrator would bank a *negative* reservoir, and when cost later rose it would
first have to climb out of that hole before producing any positive lambda — re-introducing the delayed
reaction I built this controller to remove. The max(0, .) floors the reservoir at zero, so the instant
cost crosses back over budget the integrator reacts from zero rather than from a debt. Seeding the
integral at 0.0 (lambda starts at zero, as in the dual loop) is consistent: no standing penalty until
violation is observed.

So, against the ppo_lag numbers: the dual loop left cost at 45.6/46.7/56.2, roughly twice over budget
despite a heavy reward cost. The PID controller should *actually reach the budget* — cost at or below
25 on all three environments — because the proportional and derivative terms react to the current gap
and the rising trend immediately while the integral still guarantees no permanent-timidity overshoot.
PointButton, whose densest hazards and worst naive violation make it hardest to push down, has the
least margin, so it should settle nearest the limit while the goal tasks come in under it. Return is
the honest price: I do *not* expect it to recover toward naive, and it may drop *below* the ppo_lag
15/19/4, because holding cost at 25 rather than hovering at 46 forces the agent to detour around
hazards it was cutting through. If cost lands controlled at the budget across all three environments —
the thing neither naive nor the integral-only dual loop achieved — then on the actual safe-RL
objective, high return *only when* cost is controlled, this is the first method on the ladder that
succeeds, and that is the bar it has to clear. The full module is in the answer.
