PID-Lag finally crossed the line, and in doing so it told me precisely what is still wrong. It is the
first rung to post nonzero `budget_success_rate` — PointGoal 1.000 across all three seeds, CarGoal and
Button both 0.667 — and cost sits right where it should, 19.3 on the goal tasks and 24.97 on Button, a
whisker under 25. So as a *constraint solver* it works: the PID transient shaping got cost under the
budget in time, exactly the timing fix I designed it for. But look at the reward column and the cost of
that safety is brutal. PointGoal reward collapsed to 0.20, CarGoal to 1.996, Button to 0.46 — the
policies are feasible and essentially *do nothing*. PID-Lag bought the budget by making the agent so
afraid of cost that it abandoned the task. And the two missed seeds are the same story from the other
side: CarGoal seed 42 landed at 25.22 and Button seed 456 at 28.51, both barely over, both with reward
near zero — the controller was riding the limit so tightly that ordinary estimation noise tipped them
just past it. So I have two coupled problems, and they share one root. The reward is dead because the
*continuous penalty never switches off*: even when the policy is safely under budget, the blend
`A = (adv_r − λ·adv_c)/(1+λ)` keeps a positive λ (the integral term holds a standing value precisely so
cost does not drift back up), so the policy is *always* paying a safety tax, always splitting its
gradient between reward and cost, even deep in the feasible region where it has slack to spend on
reward. That is structural to every Lagrangian method, integral or PID: the multiplier interpolates a
trade-off that it applies at *all* times.

Let me quantify how bad the reward collapse is, because the number is the argument. Against the naive
ceiling — the highest reward on the ladder, where the policy paid no safety tax at all — PID-Lag kept
`0.20/25.54 = 0.8%` of PointGoal's reward, `1.996/32.82 = 6.1%` of CarGoal's, and `0.46/19.69 = 2.3%`
of Button's. So feasibility was bought by destroying 94–99% of the attainable return. And it is not
just that safety is expensive; PID-Lag is *more* timid than the sluggish integrator it replaced:
PointGoal reward fell from PPO-Lag's 15.14 to 0.20, another factor of `~75×` down, even though PPO-Lag
was already violating the budget. The PID bought the budget crossing precisely by reacting so hard and
early that it froze the policy — the same aggressiveness that fixed the timing killed the reward. Two
readings are possible: either safety on these tasks simply costs ~99% of the reward and there is
nothing to recover, or the collapse is an artifact of *how* Lagrangian methods spend their gradient,
and a different rule could keep the budget while giving most of the reward back. The naive ceiling says
the reward is *there* to be had — 25 on PointGoal exists — so the question is whether I can reach it
without re-inflating cost. That is what makes the reward column, not the budget column, the thing to
attack now: the budget is solved, the reward is thrown away, and I need to find out whether it was
thrown away *necessarily*.

Let me question that premise, because it is the thing all three previous rungs share. Ask what the
agent should actually do at a given moment. If it is feasible — cost under 25 with slack — I want the
whole update spent on reward; there is no reason to sacrifice return while I have budget to spare. If it
is infeasible — cost over 25 — the only thing that matters is getting back under; reward is irrelevant
until it is safe. So the "trade-off" the multiplier so carefully interpolates is, at the level of
what-to-do-this-iteration, not a trade-off at all — it is a *switch*. Feasible → improve reward.
Infeasible → reduce cost. The Lagrange multiplier, whether driven by an integral or a full PID, is a
smooth, lagged approximation to a decision that wants to be made sharply. And PID-Lag's numbers are
exactly what the smooth approximation costs: it leaves a residual penalty on in the feasible region
(killing reward) and it has to ride the boundary tightly (so noise tips seeds over). A sharp switch
would do neither: in the feasible region it would carry *zero* cost penalty and let the policy chase
reward freely, and it would react to a violation at full strength immediately rather than riding the
edge.

Before I commit to throwing the multiplier out entirely, let me test whether a smaller repair to the
Lagrangian family could recover the reward, because if it could I would keep the machinery I already
trust. Two patches suggest themselves. The first: keep the PID controller but *force λ to zero whenever
the run is feasible*, so the feasible-region tax vanishes and only re-engages when cost crosses 25.
Walk it a step — the moment I zero λ on crossing under, the standing integral value that was holding
cost at the line is gone, so cost drifts back up, crosses 25 again, and λ has to rebuild from zero; I
have reinvented a switch, but implemented through the multiplier plumbing with an integral that keeps
getting reset and rebuilt, strictly clumsier than a switch that simply re-checks feasibility each
epoch. The second patch: lower the internal target below 25 so the feasible region carries margin and
the always-on penalty at least parks the policy somewhere safe with slack. But this does nothing for the
reward — a lower setpoint means a *larger* standing λ, hence a *heavier* permanent tax, hence *less*
reward, the opposite of what I need. Both patches converge on the same conclusion from opposite
directions: the reward collapse is caused specifically by the penalty being nonzero in the feasible
region, and any fix that leaves a Lagrangian tax on while feasible cannot recover the reward. The first
patch already shows the destination — a feasibility switch — so I should build that directly and
cleanly rather than smuggling it through a dual variable.

So replace the dual-weighted blend with a direct rectification of the optimization target. At each
update I already read the mean episode cost `Jc = self._logger.get_stats('Metrics/EpCost')[0]`. Use it
not to drive a multiplier but to *choose which objective the policy step optimizes*: if `Jc` is within
budget, take the step to maximize reward — feed the reward advantage `adv_r` to the PPO surrogate,
unmodified, exactly the naive update; if `Jc` is over budget, take the step to minimize cost — feed
`−adv_c`, so the surrogate ascends on the negative cost advantage, which lowers the cost objective.
There is no λ anywhere: no dual learning rate, no integral, no PID gains, no `(1+λ)` normalization
(unneeded, because the advantage is now always a single unmodified stream of normal scale). The
constraint enters only through the `if`. This is the constraint *rectified* — the policy is pushed back
toward feasibility exactly when, and only when, it strays out, and left entirely free otherwise.

Let me sign-check the cost branch, because getting it backwards would silently make the "safety" step
*raise* cost. The PPO surrogate ascends whatever advantage I feed it: it increases the probability of
actions with positive advantage and decreases those with negative advantage. The cost advantage `adv_c`
is positive for actions whose expected future cost is *above* the value baseline. If I fed `+adv_c`,
the policy would raise the probability of high-cost actions — the wrong way. Feeding `−adv_c` flips the
sign, so the surrogate now raises the probability of actions with *below-baseline* cost and suppresses
above-baseline ones, which lowers the expected cost return `J_c`. So ascending `−adv_c` is descending
`J_c`, the direction that pulls an infeasible run back toward the budget — the sign is right. And the
magnitudes line up too: `adv_c` is a normal-scale GAE advantage, the same scale `adv_r` carries, so
`−adv_c` needs no `(1+λ)` rescaling; that normalization existed only to tame `adv_r − λ·adv_c` when λ
was large, and with λ gone the surrogate is always a single unmodified advantage of ordinary magnitude,
so PPO's clip and step size see exactly what they were tuned for on either branch of the switch.

Now I have to defend this against the two ways a naive switch breaks, because PID-Lag's near-miss seeds
already show me how fragile the boundary is. The first is chattering. If I switch the instant `Jc` crosses
25, then near the boundary the *estimate* of `Jc` — a noisy Monte-Carlo average, not the true value —
fluctuates just above and just below 25 from epoch to epoch, and the algorithm flips between "improve
reward" (which pushes cost back up toward 25) and "reduce cost" (which pushes it down), oscillating at
the edge without making reward progress. That is exactly the regime CarGoal seed 42 and Button seed 456
were stuck in under PID-Lag. The cure is a deadband: do not switch to cost-reduction the moment the
estimate crosses 25, switch only when it crosses `25 + η` for a positive tolerance `η`. The tolerance
absorbs the estimation noise so the switch stops chattering, and it gives the reward step a band of
width `η` above the limit in which it is allowed to operate — so the policy can make real reward
progress instead of being yanked back to the boundary every epoch. The price is that the converged
policy can sit up to `η` above the true limit, so `η` must be small relative to the budget. I set
`η = 2.0` against a limit of 25 — an 8% slack — and I can check that number against the two seeds that
actually missed. CarGoal seed 42 landed at 25.22, a mere 0.22 over: with a switch threshold at
`25 + 2 = 27`, that run is now *inside* the tolerance band and reads as feasible, so it is no longer
pushed back to cost-reduction at all — the deadband erases that miss directly. Button seed 456 at 28.51
is 1.51 past the 27 threshold, so the switch would still engage cost-reduction there; but that seed's
problem under PID-Lag was that the controller *rode* 24.97 so tightly it had no margin, and a switch
that runs pure reward up to 27 and then decisively descends cost does not park on the boundary the way a
standing λ does, so it should settle comfortably below 27 rather than hugging it. So η = 2.0 disposes of
the smaller miss outright and gives the larger one room it did not have, while 2 out of 25 is small
enough that "feasible within tolerance" still means genuinely safe. Pushing η much larger would start to
let genuinely-over runs read as feasible; much smaller would fail to swallow the ±3-cost seed noise the
floor already showed me. Two is the compromise the observed noise picks. And the two limits of η
confirm it is the right axis: as `η → ∞` the switch never fires the cost branch, so CRPO becomes the
naive rung — maximal reward, budget ignored — while at `η → −25` the condition `Jc > 0` is always true
and the policy only ever descends cost, a maximally timid degenerate. So η slides between the reckless
and the paralyzed, and I want to sit near the reckless end but with the feasibility guard intact: small
enough that crossing it means genuinely unsafe, large enough that noise near the true limit does not
trip it. η = 2 is that near-reckless-but-guarded point. It is worth naming what just happened to the
design axis: the first three rungs all lived on the λ line — naive at λ = 0, PPO-Lag and PID-Lag moving
λ off zero by different control laws — and this rung leaves that line entirely. There is no λ to place;
the decision is discrete, and the only continuous knob left is the deadband, which is not a
reward/cost exchange rate but a noise tolerance. That is the structural break: the previous rungs asked
"how much should I weigh cost," and this one answers "that was the wrong question — weigh whichever
objective is binding, fully, and let feasibility decide which."

The second worry is whether alternating between *different* objectives converges at all. Each epoch
optimizes a different function depending on the switch, so the iterates are not descending any single
fixed objective — but that is fine here for the same reason it is fine for the method's analysis: under
accurate cost estimates the switch makes the *right* decision each epoch (improve reward only when
genuinely feasible, reduce cost only when genuinely violated), and the per-epoch progress guarantees of
the underlying PPO step then bound both the reward optimality gap and the constraint violation over a
run, giving an `O(1/√T)` rate to a globally optimal feasible policy with `O(1/√T)` violation. The
tolerance `η` is exactly the deadband that the theory shrinks like `1/√T`; I hold it fixed at 2.0 here
because the 1M-step budget is finite and the noise floor, not the asymptotic rate, sets the right slack.

There is a third thing the switch fixes that the blend could never reach, and PID-Lag's reward column
is the proof. Under any Lagrangian rule the policy gradient in the feasible region is
`(1−u)·adv_r − u·adv_c` with `u = λ/(1+λ) > 0`, so even when safe the reward direction is *contaminated*
by a cost-avoidance component pulling orthogonally to it — the policy is not climbing reward, it is
climbing a compromise between reward and not-cost, and on these navigation tasks "not-cost" and "more
reward" genuinely conflict (the rewarding path runs near hazards). I can put a floor under how large
`u` had to be. PID-Lag held cost right at the line — 19.3 on the goal tasks, 24.97 on Button — which
means the standing λ was doing real work every epoch, not sitting near zero; a λ that had decayed small
would have let cost drift back toward the naive 40s–50s, and it plainly did not. Even a modest standing
`λ = 1` gives `u = 1/2`, meaning *half* of every feasible-region gradient was spent pulling away from
cost rather than toward reward; a `λ = 3` gives `u = 0.75`, three-quarters. With that large a fraction
of every step misdirected, a policy cannot climb to the naive reward no matter how long it trains —
which is exactly the 0.20–2.0 freeze. That contamination is why PID-Lag's feasible policies froze: a
permanently-on cost component, and not a small one given where cost settled, biases every
feasible-region step away from the goal. The switch removes the component entirely. In the feasible
region the gradient is `adv_r` and nothing else — the *same* update the naive rung used to earn the
highest reward on the whole ladder — so the policy should reclaim a large fraction of naive's return
while staying inside the band. I am not trading reward for safety in the feasible region at all; I am
spending the whole step on whichever objective is currently binding, and feasibility means reward is the
only binding objective. The blend, by construction, can never do this: a multiplier that is exactly zero
in the feasible region would let cost drift back up (the reason the integral holds a standing value), so
the Lagrangian family is structurally forced to keep paying the tax. Only the switch, which re-checks
feasibility every epoch and re-engages cost-reduction the instant it is violated, can carry a true zero
penalty while feasible without losing the budget — the feasibility check *is* the standing guarantee
that the integral was faking with a nonzero λ.

Concretely in the edit surface: `_init` stores `cost_limit = 25` and `distance = 2.0` and a violation
flag; `_update` reads `Jc`, sets `_in_violation = (Jc > cost_limit + distance)`, then calls
`super()._update()` so the PPO epochs run with the flag fixed for this update (Jc does not change inside
the epochs), and logs. `_compute_adv_surrogate` is the switch: return `−adv_c` when in violation, else
`adv_r`. For logging continuity I report the multiplier slot as 1.0 in violation and 0.0 otherwise — a
faithful readout of the binary switch, not a learned dual. This matches the canonical primal CRPO update
line for line: `if Jc <= cost_limit + distance: return adv_r; else: return −adv_c`.

The bar this must clear is PID-Lag's real numbers, and the prediction is sharp and falsifiable. The
whole point of the switch is to *not* penalize reward in the feasible region, so I expect the headline
result to be a large reward recovery *at no safety cost*: `budget_success_rate` should be at least as
good as PID-Lag's (PointGoal 1.0; CarGoal and Button at or above 0.667) while reward climbs well above
PID-Lag's near-zero values — because once feasible, the policy runs the naive reward update with full
strength, so its return should approach what an unconstrained-but-feasible policy can earn, far above
0.20 / 2.0 / 0.46. The two PID-Lag near-miss seeds (CarGoal 42 at 25.22, Button 456 at 28.51) are the
direct test of the deadband: with an 8% tolerance band and no boundary-riding, those should now land
comfortably inside the band rather than tipping just over, so I expect the CarGoal and Button success
rates to *rise* toward 1.0, not just hold. The clean falsification: if reward stays pinned near zero,
the switch hypothesis is wrong and the reward collapse was not caused by the always-on penalty; if
`budget_success_rate` *drops* below PID-Lag's, the deadband is too small and the switch is chattering
back into violation. What I am betting on is that PID-Lag solved the *timing* of safety but at the cost
of *permanent* timidity, and that a method which spends nothing on safety while feasible and everything
on it while violated recovers the reward PID-Lag threw away without giving back the budget it won.
