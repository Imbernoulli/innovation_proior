The Lagrangian run did exactly what I predicted, and the prediction being right is the diagnosis. Cost
came down everywhere — PointGoal 51→46, CarGoal 61→47, Button from a catastrophic 152 to 56 — so dual
ascent pulls in the right direction, and the reward collapse that came with it (PointGoal 25.5→15,
Button 19.7→4) confirms the trade is real. But `budget_success_rate` is *still zero on every seed of
every environment*. The mechanism lowers cost without ever crossing under 25 — the precise failure my
closing test was built to detect: right direction, too sluggish.

The pattern of the reductions confirms it is a timing failure, not a direction failure. Cost fell `−11%`
on PointGoal, `−23%` on CarGoal, `−63%` on Button: the controller worked hardest exactly where the
violation was worst, which is what an integral of the violation should do — λ built largest on Button
because Button fed it the largest error for the longest time. But look where "hardest" left me: Button
at 56.22 is still `2.25×` the budget and PointGoal, needing the least work, only reached `45.58 = 1.82×`.
The controller closed the biggest gap by the largest fraction and *still* landed everyone above 25 — a
mechanism that pulls in the right direction but never in time. The reward it charged is steep and
monotone in the cost removed (×0.59, ×0.57, ×0.20), Button paying the most reward for the most cost cut,
the densest arena at the steepest exchange rate. And one cell is diagnostic: PointGoal's per-seed rewards
`11.83 / 12.47 / 21.12`, with seed 456 holding reward 21 at cost 48.52 — a spread saying λ had not
settled to a common value across seeds by the end, the signature of an integrator still climbing when the
budget ran out.

So I need to understand *why* the integrator is slow rather than just turning its gain up. The PPO-Lag
update was `λ_{k+1} = (λ_k + K_I(J_C − d))_+`: λ is a running accumulation of the violation, literally
the time-integral of the constraint error. That is the whole problem in one line — integral control has
memory but no reflexes. It reacts only to *accumulated* error, so λ rises only after violation has been
piling up, and by the time it has integrated up to a rescuing value cost has already overshot — exactly
the PPO-Lag pattern, a large standing λ that arrived too late. Pushing `K_I` higher does not fix this: a
stiffer pure integrator rings faster and longer, trading a late response for an oscillating one.

Two one-knob fixes tempt me, and each fails on its own terms. Raise the dual learning rate so the
integrator climbs faster? `lambda_lr` scales the *whole* accumulation, so a rate fast enough to reach
λ ≈ 10 on Button in time would overshoot PointGoal's much smaller target and have to unwind — the
ringing again — and PointGoal's per-seed reward spread already says λ is barely settling. One
first-order accumulator with one time constant cannot be both quick where the gap is huge and gentle
where it is small. Tighten the *target* instead, driving the integrator toward an internal limit of 20
for margin? That addresses "settles above 25" but not the lateness: the same slow integral still arrives
after cost has run up, so on Button it settles above 20 by the same 2× factor, buying nothing, while on
PointGoal it sacrifices reward to hold a tighter line I did not need. A lower setpoint fixes steady-state
offset; my failure is transient. Both rejections point one way: the missing thing is not more gain or a
lower target on the *same* rule, it is a *different kind of response* — one that reacts to the present
error and the trend, not only the accumulated past.

The reframe is to stop tuning the Lagrangian update and see what the loop actually is: a feedback control
system. The cost limit `d = 25` is a setpoint, the measured episodic cost `J_C` the output I want held
there, the multiplier λ the control input I set, and the PPO step chasing reward is the plant — a
nonlinear, unknown map from λ to next epoch's cost. The traditional Lagrangian method is one impoverished
choice of control rule: integral-only. So the question becomes what controller should drive λ, and the
answer when an integral loop is late and rings is the other two terms of the standard three-term
controller — proportional and derivative.

Proportional adds a term on the *current* error, `K_P·(J_C − d)`: an instantaneous reaction, no waiting
for accumulation. In the continuous-time picture the integral term was `λ̇ = α·g`, and the proportional
contribution enters `λ̇` tracking `g`'s rate, `β·ġ`; redoing the second-order collapse of the coupled
primal-dual dynamics, the restoring force `α·g·∇g` is untouched — so proportional does not move the
equilibrium — but the damping matrix gains `β·∇g·∇gᵀ`, positive semidefinite by construction
(`vᵀ(∇g∇gᵀ)v = (∇g·v)² ≥ 0`). More damping is exactly what kills an oscillation, and it does so touching
only the λ update, not the policy objective — the penalty method's damping benefit without its indefinite
constraint-Hessian baggage. For my failure this is the missing reflex: when cost is over the line *right
now*, proportional reacts this epoch instead of waiting for the integral to build.

Derivative adds anticipation. I want to react to the *trend* — if cost is rising fast toward 25, raise λ
before it crosses. A term on the rate of cost change enters λ's dynamics one derivative higher; through
the same collapse it adds a force quadratic in velocity and modulated by the constraint curvature along
the motion, so when I move along a direction where cost curves upward and I am already in violation, it
brakes harder before overshoot — precisely the overshoot Button suffered. One asymmetry I bake in:
derivative should *brake increases* in cost (the dangerous direction) but not fight *decreases*, so I
rectify it to the positive part, `(rate of cost increase)_+` — a one-sided derivative for a one-sided
constraint. Derivative is also the term most prone to amplifying noise, so I will smooth it.

Why keep integral at all? Because at convergence I need *zero* steady-state violation, and only the
integral supplies the standing λ that holds cost exactly at 25. When the system settles, `J_C − d ≈ 0`,
so proportional vanishes and derivative (cost not changing) vanishes; if those were my only terms λ would
collapse and cost drift back up. The integral has *remembered* the accumulated history and holds a
nonzero λ at zero instantaneous error. So the division of labor is clean: proportional and derivative
shape the *transient* — the thing PPO-Lag failed at — while integral guarantees the *asymptote*. And
`K_P = K_D = 0` recovers PPO-Lag exactly, so this is a strict generalization with the old method at the
origin of the two new axes.

Now the discrete rule, matching what the harness exposes. Each epoch I read `J_C`, error `Δ = J_C − d`.
To tame the noise derivative amplifies I smooth two signals with an EMA at 0.95:
`delta_p ← 0.95·delta_p + 0.05·Δ` and `cost_d ← 0.95·cost_d + 0.05·J_C`. The 0.95 is not free —
retention `a = 0.95` gives an effective window `1/(1−a) = 20` epochs, so each smoothed signal is a
20-epoch running average that damps the ±11-cost single-epoch swings I measured on the naive runs by
about `√20 ≈ 4.5×` — long enough to ignore a one-epoch blip, short enough to track a genuine multi-epoch
trend. The integral accumulates with anti-windup, `I ← (I + K_I·Δ)_+`: the inner positive part stops the
integrator banking a *negative* reservoir during a long feasible stretch, which would delay the next
response by forcing λ to climb out of a hole first — the exact disease I am curing. The derivative is the
rectified change of the smoothed cost over a short delay, `pid_d = (cost_d − cost_d[t−w])_+`, `w = 10`
epochs (a deque of smoothed-cost history), the delay-and-rectify giving a clean one-sided trend estimate.
The multiplier is `λ = clip(K_P·delta_p + I + K_D·pid_d, 0, 100)`, the upper clamp a guard against
runaway on the densest arena — and 100 is an *effective* saturation, since at λ = 100 the blend weight
`u = 100/101 = 0.990` is already 99% cost-avoidance, so capping there also stops the integral banking a
meaningless five-hundred that would take forever to unwind once Button finally comes under budget. Gains
`K_P = 0.1`, `K_I = 0.01`, `K_D = 0.01`: the integral is smallest because it accumulates every epoch — at
`K_I = 0.01` a sustained unit of violation adds only 0.01 to λ per epoch, so it takes ~a thousand
epoch-units of standing error to build the λ ≈ 10 Button needs, which is exactly why I am not relying on
it for the transient. Proportional is an order of magnitude larger so the *current* error can contribute
comparably in a single epoch: a 20-over violation gives `0.1 × 20 = 2` of immediate λ, a reflex the
integrator would need two hundred epochs to match. Derivative matches the integral at 0.01 because it
multiplies a cost *rate*, which on the dense arena can itself be large, so a small gain is enough
anticipation without whipping λ around.

Trace a Button-like moment to see the reflex fire while the integrator is still asleep. Cost near
`J_C = 60`, so `delta_p` tracks `Δ = 35` and the smoothed cost has risen ~10 over ten epochs,
`pid_d ≈ 10`. Proportional contributes `0.1 × 35 = 3.5` to λ this epoch from the current error alone;
derivative adds `0.01 × 10 = 0.1`; the integral, still where its slow accumulation left it — say 2 —
gives `λ = 3.5 + 2 + 0.1 = 5.6`, most of the way to the λ ≈ 10 Button needs, and the bulk of it from
proportional, which the integrator would need hundreds of epochs to reach. When the system settles at
`Δ → 0`, proportional and derivative fall to 0 and only the integral's standing value holds λ. And on the
other side, when the controller is winning and cost is dropping, `cost_d − cost_d[t−10] < 0` and the
rectifier sends the derivative to exactly 0 — the anticipatory brake releases the moment the danger
recedes and never props λ up on a good trend, which an unrectified derivative would do by reading a fast
decrease as a signal to keep pushing. That asymmetry is the point of the one-sided constraint.

The policy hook is *unchanged* — the same scale-normalized blend `A = (adv_r − λ·adv_c)/(1 + λ)`, needed
even more here where an aggressive PID can drive λ large fast. The only change from PPO-Lag is the rule
that *produces* λ.

The expectations against the PPO-Lag numbers. The claim is that proportional plus derivative get cost
under the line *in time* where the pure integrator could not, so I expect this to be the first rung to
post nonzero `budget_success_rate` — cost falling from PPO-Lag's 40s–50s to at or below 25 on most runs,
with PointGoal (the gentlest, starting only `1.82×` over) crossing under most reliably, and Button
(`2.25×` over, the arena where cost rises fastest and the derivative reads the noisiest signal) and
CarGoal closer to the line and seed-dependent. I would not be surprised to see one or two seeds on the
dense arenas land just over 25 while siblings land just under, a controller tuned to sit near the
boundary being sensitive to the ±3-cost seed noise I measured at the floor. The cost of that safety is
reward: reacting hard and early suppresses reward-seeking more than the sluggish integral did, so I
expect reward to fall further still from PPO-Lag's already-reduced levels, plausibly toward zero on the
goal tasks — a feasible-but-timid policy. That is the explicit trade: this rung is judged on the budget
metric the previous two failed, and if it crosses under 25 on most runs it is the strongest baseline here
even at the lowest reward, because a policy that violates on 100% of runs cannot be called safer than one
that satisfies. The clean test: still zero everywhere → the PID hypothesis is wrong and lateness was not
the cause; one on the gentle geometry and fractional on the dense ones → the transient-shaping story
holds and the residual misses are an aggressiveness-versus-reward problem, not a timing one. The
distilled scaffold fill is in the answer.
