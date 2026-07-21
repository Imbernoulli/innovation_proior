PID-Lag finally crossed the line, and in doing so told me precisely what is still wrong. It is the first
rung to post nonzero `budget_success_rate` — PointGoal 1.000 across all three seeds, CarGoal and Button
both 0.667 — with cost sitting right where it should, 19.3 on the goal tasks and 24.97 on Button, a
whisker under 25. As a *constraint solver* it works: the PID transient shaping got cost under budget in
time. But the reward column is brutal — PointGoal 0.20, CarGoal 1.996, Button 0.46 — the policies are
feasible and essentially *do nothing*. Against the naive ceiling PID-Lag kept `0.20/25.54 = 0.8%` of
PointGoal's reward, `1.996/32.82 = 6.1%` of CarGoal's, `0.46/19.69 = 2.3%` of Button's: feasibility
bought by destroying 94–99% of the attainable return. And it is *more* timid than the sluggish
integrator it replaced — PointGoal reward fell from PPO-Lag's 15.14 to 0.20, another ~75× down, even
though PPO-Lag was already over budget. The two missed seeds are the same story from the other side:
CarGoal 42 at 25.22 and Button 456 at 28.51, both barely over, both with reward near zero — the
controller rode the limit so tightly that ordinary estimation noise tipped them past it.

Both failures share one root. The reward is dead because the continuous penalty never switches off: even
safely under budget, the blend `A = (adv_r − λ·adv_c)/(1+λ)` keeps a positive λ — the integral term holds
a standing value precisely so cost does not drift back up — so the policy is *always* paying a safety
tax, splitting its gradient between reward and cost even deep in the feasible region where it has slack to
spend on reward. That is structural to every Lagrangian method, integral or PID: the multiplier
interpolates a trade-off it applies at *all* times. The naive ceiling says the reward is *there* — 25 on
PointGoal exists — so the question is whether the collapse is necessary or an artifact of *how* the
Lagrangian spends its gradient. That makes the reward column, not the budget column, the thing to attack.

Question the premise all three previous rungs share. Ask what the agent should do at a given moment.
Feasible, with slack? Spend the whole update on reward — there is no reason to sacrifice return while
budget is spare. Infeasible? The only thing that matters is getting back under; reward is irrelevant
until it is safe. So the "trade-off" the multiplier interpolates is, at the level of what-to-do-this-
iteration, not a trade-off at all — it is a *switch*. Feasible → improve reward. Infeasible → reduce
cost. The Lagrange multiplier, integral or full PID, is a smooth lagged approximation to a decision that
wants to be made sharply, and PID-Lag's numbers are what the smoothing costs: a residual penalty left on
in the feasible region (killing reward) and a boundary ridden so tightly noise tips seeds over.

Before throwing the multiplier out, test whether a smaller repair recovers the reward. Patch one: keep
the PID but force λ to zero whenever feasible. Walk it a step — the moment λ zeroes on crossing under,
the standing integral holding cost at the line is gone, cost drifts back up, crosses 25, and λ rebuilds
from zero; I have reinvented a switch, implemented clumsily through multiplier plumbing with an integral
that keeps resetting. Patch two: lower the internal target for margin. But a lower setpoint means a
*larger* standing λ, a *heavier* permanent tax, *less* reward — the opposite of what I need. Both
converge from opposite directions: the reward collapse is caused specifically by the penalty being
nonzero in the feasible region, and any fix that leaves a tax on while feasible cannot recover reward.
The first patch already shows the destination, so I build the switch directly.

Replace the dual-weighted blend with a direct rectification of the optimization target. Each update I
read `Jc = self._logger.get_stats('Metrics/EpCost')[0]` and use it not to drive a multiplier but to
*choose which objective the step optimizes*: within budget, feed `adv_r` to the surrogate unmodified (the
naive reward update); over budget, feed `−adv_c` so the surrogate ascends on the negative cost advantage,
lowering the cost objective. No λ anywhere — no dual LR, no integral, no PID gains, no `(1+λ)`
normalization. The constraint enters only through the `if`.

Sign-check the cost branch, because getting it backwards would silently make the "safety" step *raise*
cost. The surrogate raises the probability of positive-advantage actions and lowers negative ones; `adv_c`
is positive for actions whose expected future cost is *above* the value baseline, so feeding `+adv_c`
would raise high-cost actions — the wrong way. Feeding `−adv_c` raises below-baseline-cost actions and
suppresses above-baseline ones, lowering the expected cost return `J_c` — the direction that pulls an
infeasible run back toward budget. And `adv_c` is a normal-scale GAE advantage, the same scale `adv_r`
carries, so `−adv_c` needs no `(1+λ)` rescaling; that normalization existed only to tame `adv_r − λ·adv_c`
at large λ, and with λ gone the surrogate always sees a single unmodified advantage of ordinary magnitude
on either branch.

Now defend against the two ways a naive switch breaks, since PID-Lag's near-miss seeds show how fragile
the boundary is. First, chattering. Switch the instant `Jc` crosses 25 and, near the boundary, the noisy
Monte-Carlo *estimate* fluctuates just above and below 25, flipping the algorithm between "improve reward"
(pushing cost up) and "reduce cost" (pushing it down), oscillating without reward progress — exactly where
CarGoal 42 and Button 456 were stuck. The cure is a deadband: switch to cost-reduction only when the
estimate crosses `25 + η`. The tolerance absorbs the estimation noise and gives the reward step a band of
width η above the limit to make real progress instead of being yanked back every epoch. The price is that
the converged policy can sit up to η above the true limit, so η must be small relative to the budget. I
set `η = 2.0` — an 8% slack — and can check it against the two seeds that actually missed. CarGoal 42 at
25.22 is 0.22 over: with the threshold at `25 + 2 = 27` that run is now inside the band, reads as
feasible, and is no longer pushed to cost-reduction — the miss erased directly. Button 456 at 28.51 is
1.51 past 27, so cost-reduction still engages; but that seed's problem was riding 24.97 with no margin,
and a switch that runs pure reward up to 27 then decisively descends cost does not park on the boundary
the way a standing λ does, so it should settle comfortably below 27 rather than hugging it. So η = 2
disposes of the smaller miss and gives the larger one room, while 2 out of 25 still means genuinely safe:
larger η lets genuinely-over runs read as feasible, smaller fails to swallow the ±3-cost seed noise the
floor showed me. At the limits, `η → ∞` never fires the cost branch so CRPO becomes the naive rung
(budget ignored) and `η → −25` makes `Jc > 0` always true so the policy only ever descends cost
(maximally timid), so η slides between reckless and paralyzed and I want the near-reckless-but-guarded
point. The design axis has changed: the first three rungs all lived on the λ line — naive at λ = 0,
PPO-Lag and PID-Lag moving λ off zero by different control laws — and this rung leaves it entirely. There
is no λ to place; the only continuous knob left is the deadband, which is not a reward/cost exchange rate
but a noise tolerance. The previous rungs asked "how much should I weigh cost"; this one answers "weigh
whichever objective is binding, fully, and let feasibility decide which."

The second worry is whether alternating between *different* objectives converges at all. Each epoch
optimizes a different function depending on the switch, so the iterates descend no single fixed objective
— but under accurate cost estimates the switch makes the *right* decision each epoch (improve reward only
when genuinely feasible, reduce cost only when genuinely violated), and the per-epoch progress guarantees
of the underlying PPO step then bound both the reward optimality gap and the constraint violation over a
run, giving an `O(1/√T)` rate to a globally optimal feasible policy with `O(1/√T)` violation. The
tolerance η is exactly the deadband that analysis shrinks like `1/√T`; I hold it fixed at 2.0 because the
1M-step budget is finite and the noise floor, not the asymptotic rate, sets the right slack.

There is a third thing the switch fixes that the blend could never reach, and PID-Lag's reward column is
the proof. Under any Lagrangian rule the feasible-region gradient is `(1−u)·adv_r − u·adv_c` with
`u = λ/(1+λ) > 0`, so even when safe the reward direction is *contaminated* by a cost-avoidance component
pulling orthogonally to it — the policy climbs a compromise between reward and not-cost, and on these
navigation tasks "not-cost" and "more reward" genuinely conflict since the rewarding path runs near
hazards. I can floor how large `u` had to be: PID-Lag held cost right at the line — 19.3 on the goal
tasks, 24.97 on Button — so the standing λ was doing real work every epoch, not sitting near zero; a λ
that had decayed small would have let cost drift back toward the naive 40s–50s, and it plainly did not.
Even a modest `λ = 1` gives `u = 1/2`, meaning *half* of every feasible-region gradient was spent pulling
away from cost; `λ = 3` gives `u = 0.75`, three-quarters. With that fraction misdirected a policy cannot
climb to the naive reward however long it trains — exactly the 0.20–2.0 freeze. The switch removes the
component entirely: in the feasible region the gradient is `adv_r` and nothing else, the *same* update the
naive rung used to earn the highest reward on the whole ladder, so the policy should reclaim a large
fraction of naive's return while staying inside the band. The blend cannot do this — a multiplier exactly
zero in the feasible region would let cost drift up — so only the switch, re-checking feasibility every
epoch and re-engaging cost-reduction the instant it is violated, carries a true zero penalty while
feasible without losing the budget. The feasibility check *is* the standing guarantee the integral was
faking with a nonzero λ.

Concretely in the edit surface: `_init` stores `cost_limit = 25` and `distance = 2.0` and a violation
flag; `_update` reads `Jc`, sets `_in_violation = (Jc > cost_limit + distance)`, then calls
`super()._update()` so the PPO epochs run with the flag fixed for this update (Jc does not change inside
the epochs), and logs. `_compute_adv_surrogate` is the switch: return `−adv_c` when in violation, else
`adv_r`. For logging continuity I report the multiplier slot as 1.0 in violation and 0.0 otherwise — a
faithful readout of the binary switch, not a learned dual.

The bar this must clear is PID-Lag's real numbers, and the prediction is sharp. The whole point is to not
penalize reward in the feasible region, so I expect the headline to be a large reward recovery *at no
safety cost*: `budget_success_rate` at least as good as PID-Lag's (PointGoal 1.0; CarGoal and Button at or
above 0.667) while reward climbs well above the near-zero 0.20 / 2.0 / 0.46 — because once feasible the
policy runs the naive reward update at full strength, so its return should approach what an
unconstrained-but-feasible policy can earn. The two near-miss seeds are the direct test of the deadband:
with an 8% band and no boundary-riding, CarGoal and Button success rates should *rise* toward 1.0, not
just hold. The clean falsification: if reward stays pinned near zero the switch hypothesis is wrong and
the always-on penalty was not the cause; if `budget_success_rate` *drops* below PID-Lag's, the deadband is
too small and the switch is chattering back into violation. What I am betting on is that PID-Lag solved the
*timing* of safety at the cost of *permanent* timidity, and that a method spending nothing on safety while
feasible and everything on it while violated recovers the reward PID-Lag threw away without giving back
the budget it won.
