The Lagrangian run did exactly what I predicted, and the prediction being right is itself the
diagnosis. Cost came down everywhere — PointGoal from naive's 51 to 46, CarGoal from 61 to 47, Button
from a catastrophic 152 to 56 — so the dual ascent is pulling in the right direction, and the
reward collapse that came with it (PointGoal 25.5→15, Button 19.7→4) confirms the trade-off is real:
every unit of cost pressure is reward traded away. But the metric that the task actually counts,
`budget_success_rate`, is *still zero on every seed of every environment*. The mechanism lowers cost
without ever crossing under 25. That is the precise failure my closing test was built to detect: the
direction is right and the controller is too sluggish. Button is the loudest evidence — λ had to climb
enormously to drag 152 down to 56, and even that enormous λ left it 2× over budget, because by the
time the integral had accumulated enough to push hard, cost had already been over for a long stretch
and the policy had baked the unsafe path in.

So I need to understand *why* the integrator is too slow rather than just turning its gain up. Look at
the exact update PPO-Lag used: `λ_{k+1} = (λ_k + K_I(J_C − d))_+`. Stare at it. λ is a running
accumulation of the violation `J_C − d`: every epoch I add the current error to a stored quantity. The
multiplier is literally the time-integral of the constraint error. And that is the whole problem in one
line — integral control has memory but no reflexes. It reacts only to *accumulated* error, so λ rises
only after violation has been piling up for a while; the response is intrinsically late. By the time λ
has integrated up to a value large enough to push cost back toward 25, cost has already overshot, and
that is exactly the pattern in the PPO-Lag numbers: a large standing λ that arrived too late to ever
get cost under the line on a 1M-step budget. Pushing `K_I` higher does not fix this — a stiffer pure
integrator rings faster and longer, trading a late response for an oscillating one, and the reward
degrades further as a side effect. One knob, bad trade.

The fix is to stop thinking of this as "tune the Lagrangian update" and recognize what the loop
actually is: a feedback control system. The cost limit `d = 25` is a setpoint. The measured episodic
cost `J_C` is the output I want held at the setpoint. The multiplier λ is the control input I get to
set. And the PPO step grinding θ forward to chase reward is the plant — a nonlinear, unknown map from
λ to next epoch's cost. The traditional Lagrangian method is one impoverished choice of the control
rule: integral-only. Once I see it that way the question becomes "what controller should drive λ," and
the answer the field reaches for when an integral loop is late and rings is the other two terms of the
standard three-term controller — proportional and derivative. Let me bring each in and check it does
what I hope.

Proportional first. Add to λ a term proportional to the *current* error, `K_P·(J_C − d)` — an
instantaneous reaction, no waiting for accumulation. In the continuous-time picture the integral term
was `λ̇ = α·g`; the proportional contribution enters `λ̇` as a term tracking `g`'s rate, i.e. as `β·ġ`.
Redoing the second-order collapse of the coupled primal-dual dynamics with this extra piece, the
restoring force `α·g·∇g` is untouched — so proportional does not move the equilibrium, the solution set
is still feasible-and-stationary — but the damping matrix gains `β·∇g·∇gᵀ`, the outer product of `∇g`
with itself, which is positive semidefinite by construction (`vᵀ(∇g∇gᵀ)v = (∇g·v)² ≥ 0`). More damping
is exactly what kills an oscillation. So proportional control supplies damping, and it does so cleanly:
it touches only the λ update, not the policy objective, unlike the quadratic-penalty method which
injects the *same* outer-product damping but bundled with an indefinite constraint-Hessian term and a
modification of the primal step. Proportional gives me the penalty method's damping benefit with none
of the baggage. For my failure this is the missing reflex: when cost is over the line *right now*,
proportional reacts this epoch instead of waiting for the integral to build.

Now derivative — anticipation. Proportional reacts to the present; I want to react to the *trend*: if
cost is rising fast toward 25, raise λ now, before cost crosses. A term on the rate of cost change
enters λ's dynamics one derivative higher; grinding through the collapse, the derivative term scales the
restoring direction by `B⁻¹ = (I + γ·∇g∇gᵀ)⁻¹` (eigenvalues ≤ 1) and adds a force
`γ·(ẋᵀ∇²g·ẋ)·B⁻¹∇g` quadratic in velocity and modulated by the constraint curvature along the motion.
When I am moving along a direction where cost curves upward and I am already in violation, this
anticipatory force adds to the restoring force — I brake harder before overshooting. That is precisely
the overshoot the Button run suffered: cost ran up and the integral could not anticipate it. One
asymmetry I bake in for the cost case: derivative control should *brake increases* in cost (the
dangerous direction) but not fight *decreases* — a falling cost is good. So I rectify the derivative
to the positive part, `(rate of cost increase)_+`, a one-sided derivative for a one-sided constraint.
Derivative is also the term most prone to amplifying noise, because the change of a noisy cost estimate
is wild — I will need to smooth it.

Why keep integral at all? Because at convergence I need *zero* steady-state violation, and only the
integral can supply the standing λ that holds cost exactly at 25. When the system settles, the error
`J_C − d ≈ 0`, so the proportional term vanishes and the derivative term (cost not changing) vanishes
too; if those were my only terms λ would collapse and cost would drift back up — exactly the drift the
budget cannot tolerate. The integral has *remembered* the accumulated history and holds a nonzero λ
even when the instantaneous error is zero. So the division of labor is clean: proportional and
derivative shape the *transient* — the very thing PPO-Lag failed at, getting under the line in time —
while integral guarantees the *asymptote*, staying under it. And setting `K_P = K_D = 0` recovers
PPO-Lag's integral-only rule exactly, so this is a strict generalization: I have widened a
one-parameter family to a three-parameter one with the old method at the origin of the two new axes.

Now the discrete rule I will actually run, matching what the harness exposes. Each epoch I read
`J_C = self._logger.get_stats('Metrics/EpCost')[0]`. The error is `Δ = J_C − d`. To tame the noise the
derivative term would amplify, I smooth two signals with an exponential moving average at 0.95:
`delta_p ← 0.95·delta_p + 0.05·Δ` (a smoothed proportional error) and `cost_d ← 0.95·cost_d + 0.05·J_C`
(a smoothed cost level). The integral accumulates with anti-windup, `I ← (I + K_I·Δ)_+`: the inner
positive part stops the integrator from banking a *negative* reservoir during a long feasible stretch,
which would otherwise delay the next response by forcing λ to climb out of a hole first — the exact
disease I am curing, so I will not reintroduce it. The derivative is the rectified change of the
smoothed cost over a short delay window, `pid_d = (cost_d − cost_d[t−w])_+`, with `w = 10` epochs of
delay (a deque holding the smoothed-cost history); the delay-and-rectify gives a clean one-sided trend
estimate instead of a single-step difference. The multiplier is the nonnegative combination
`λ = clip(K_P·delta_p + I + K_D·pid_d, 0, 100)`, with the upper clamp at 100 a guard against a runaway
controller in the densest-hazard arena. Gains `K_P = 0.1`, `K_I = 0.01`, `K_D = 0.01`: integral small
because it accumulates, proportional an order of magnitude larger to supply the fast reflex, derivative
matched to the integral. Crucially the policy hook is *unchanged* — I reuse the same scale-normalized
blend `A = (adv_r − λ·adv_c)/(1 + λ)`, because the `(1+λ)` normalization that kept PPO's step size sane
under a large integral λ is needed even more here, where an aggressive PID controller can drive λ large
fast. The only change from PPO-Lag is the rule that *produces* λ.

Now the falsifiable expectations against the PPO-Lag numbers. The whole claim is that proportional plus
derivative get cost under the line *in time*, where the pure integrator could not. So I expect this to
be the first rung to post nonzero `budget_success_rate` — concretely, cost should fall from PPO-Lag's
40s–50s to at or below 25 on most runs, with PointGoal (the gentlest geometry) crossing under most
reliably and Button and CarGoal closer to the line and seed-dependent. The cost of buying that safety
is reward: because the controller now reacts hard and early, it suppresses the policy's reward-seeking
much more than the sluggish integral did, so I expect reward to fall further still from PPO-Lag's
already-reduced levels, plausibly toward zero on the goal tasks — a feasible-but-timid policy. That is
the explicit trade I am making: this rung is judged on the budget metric the previous two failed, and
if it crosses under 25 on most runs it is the strongest baseline on this task even though its reward is
the lowest, because a policy that violates the budget on 100% of runs cannot be called safer than one
that satisfies it, whatever its return. The clean test: if `budget_success_rate` is still zero
everywhere, the PID hypothesis is wrong and the lateness was not the cause; if it is one on the gentle
geometry and fractional on the dense ones, the transient-shaping story holds and the residual misses
are an aggressiveness-versus-reward problem, not a timing one. The distilled scaffold fill is in the
answer.
