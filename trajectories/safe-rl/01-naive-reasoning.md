The mechanism is the whole point, but it bolts onto a base learner, and with no mechanism that learner
is the floor — so the thing to establish first is what the agent does when safety is ignored
entirely. The substrate hands me a constrained-PPO loop with two value functions already wired: one
GAE pass on the reward signal producing `adv_r`, a second on the cost signal producing `adv_c`. The
loop is built to blend them, but at the start nothing forces it to. So the weakest possible fill is
the one that throws the cost stream away: leave the multiplier at zero and have the advantage hook
return `adv_r` untouched. That is plain reward-maximizing PPO running inside a safe-RL harness with
the safety apparatus disconnected.

Let me be precise about why this is the floor and not just *a* baseline. The two slots I can edit are
the multiplier update in `_update` and the advantage combination in `_compute_adv_surrogate`. In the
default fill, `_update` reads the measured mean episodic cost `Jc` (only to assert it is not NaN, a
sanity check on the logging) and then calls `super()._update()` without ever moving `lambda` off its
initial `0.0`. And `_compute_adv_surrogate(adv_r, adv_c)` returns `adv_r`, discarding `adv_c`
completely. So the policy gradient the inner PPO ascends is built from the reward advantage alone.
The cost critic still trains — the loop computes `adv_c` regardless — but its output never reaches the
policy. There is, by construction, no channel through which a safety violation can change the agent's
behavior. This is the constraint-unaware control: it tells me the *upper bound on reward* obtainable
when nothing is paid for cost, and the *uncontrolled cost* the environment produces when the agent
optimizes reward freely.

Why start here rather than at a real constraint method? Because the entire ladder is about the
trade-off between reward and cost, and I cannot read that trade-off without an anchor at each end. The
naive fill anchors the reward-greedy end. Every later mechanism will trade some of this reward away to
pull cost down toward the budget `d = 25`, and the only way to judge whether a mechanism is buying
safety cheaply or expensively is against this point: how much return did it give up, and how much cost
did it remove. Without the naive number I would have no idea whether a method that lands cost at 20 is
doing well (because the unconstrained agent sat at 60) or doing nothing (because the unconstrained
agent was already near 20). So the floor is not a throwaway; it is the denominator for everything that
follows.

Now I should reason about what this floor will actually do, because that prediction is what the next
step diagnoses against. PPO with `adv = adv_r` is a competent reward optimizer. On these
Safety-Gymnasium navigation tasks the reward is reasonably dense — the agent gets shaped signal for
moving toward goals — so PPO should learn the task well and post the *highest* returns of anything on
the ladder. There is nothing holding it back from the reward. The cost is the other half. The hazards
in these environments are placed exactly where the shortest, highest-reward paths run; the reward
gradient points straight through them. With no penalty on `adv_c`, the agent has no reason to detour,
so it will plow through hazards whenever that is the fast way to a goal, and the episodic cost should
sit far above the budget — many times `25` on the environments where the hazard density is high. The
expectation is sharp and asymmetric: strong on return, badly out of bounds on cost. That is the
signature of ignoring the constraint, and it is the failure the first real mechanism has to fix.

One subtlety worth naming, because it sets up the comparison. The cost will not be uniform across the
three environments. SafetyPointButton1-v0 packs the most hazards around its buttons, so I expect its
naive cost to be the largest by a wide margin; SafetyPointGoal1-v0 and SafetyCarGoal1-v0 should be
high but less extreme. The return ordering will roughly track task difficulty — the car's harder
dynamics should depress its return relative to the point robot, and the button task's longer
interaction should make its return the noisiest. None of this changes the diagnosis: wherever cost
lands, it lands without any mechanism pulling it down, so it is the worst-case cost, and the job of
step 2 is to introduce a multiplier that reads that violation and starts paying for it.

So the edit is the trivial one: keep `lambda` pinned at zero, register the multiplier key so the
logging contract is satisfied, and return `adv_r` from the advantage hook so the cost stream is
dropped (the distilled module is in the answer). What I will watch in the result is the gap between
return and a controlled-cost method: high return is expected and is not the point; the point is how
far over `25` the cost sits, because that distance is exactly the safety debt the next mechanism must
pay down, and the amount of return it has to surrender to do so is the price I will then be measuring.
The naive run is the constraint-unaware ceiling on reward and the uncontrolled floor on safety; from
here every step trades down the first to fix the second.
