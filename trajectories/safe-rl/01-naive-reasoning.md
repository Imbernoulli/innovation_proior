The mechanism is the whole point, but it bolts onto a base learner, and with no mechanism that learner
is the floor — so the thing to establish first is what the agent does when safety is ignored
entirely. The substrate hands me a constrained-PPO loop with two value functions already wired: one
GAE pass on the reward signal producing `adv_r`, a second on the cost signal producing `adv_c`. The
loop is built to blend them, but at the start nothing forces it to. So the weakest possible fill is
the one that throws the cost stream away: leave the multiplier at zero and have the advantage hook
return `adv_r` untouched. That is plain reward-maximizing PPO running inside a safe-RL harness with
the safety apparatus disconnected.

The two slots I can edit are the multiplier update in `_update` and the advantage combination in
`_compute_adv_surrogate`. In the default fill `_update` reads the mean episodic cost `Jc` only to
assert it is not NaN and never moves `lambda` off `0.0`; `_compute_adv_surrogate` returns `adv_r`,
discarding `adv_c`. PPO's policy gradient is `E[grad log pi * A]`, so with `A = adv_r` the cost
advantage appears nowhere in the update. The cost critic still trains, regressing onto the empirical
cost returns, and `adv_c` is computed every epoch and dropped on the floor. Both channels through
which safety could enter — the aggregate `Jc` that would drive a multiplier and the per-action
`adv_c` that would tilt the gradient — are open and measured yet unused. And because returning
`adv_r` leaves the tensor's shape, dtype, and *statistics* exactly what the substrate's advantage
normalization, clip range, and learning rate were tuned for, this floor is numerically the untouched
reward-only PPO update — which is what lets me read its return as a genuine ceiling rather than
something contaminated by an odd advantage scale. It fixes the upper bound on reward when nothing is
paid for cost, and the uncontrolled cost when reward is optimized freely.

Why two separate critics rather than a single critic on a blended target `r - beta*c`? Because the
blend weight is exactly the thing meant to be adaptive. A single critic on `r - beta*c` would need
retraining every time the trade-off moved, its target shifting underneath it; keeping reward and cost
critics separate lets each fit a fixed target — `V_r` on reward returns, `V_c` on cost returns,
neither depending on any trade-off parameter — and defers the blend all the way to the
policy-gradient stage, where `_compute_adv_surrogate` combines the two *advantages* with the current
weight. That is why the substrate exposes an advantage hook at all; at the floor both critics train
correctly and I route only the reward advantage into the policy step.

The exactly-zero fill is forced, not merely convenient. Baking even a small fixed `beta` into the
hook would already be a mechanism — reward shaping — and would smuggle it into the denominator every
later method is judged against, so I would no longer know the *uncontrolled* cost. Initializing
`lambda` positive and freezing it is the same fixed-`beta` case. And an actual multiplier update in
`_update` is the mechanism I am supposed to design next. Only `lambda = 0` with `A = adv_r` leaves a
clean, mechanism-free denominator.

In the constrained-optimization lens this floor is the `lambda = 0` vertex of the dual
`L = J_r - lambda*(J_c - d)`, `L = J_r`. Complementary slackness says `lambda = 0` is correct only if
the constraint is slack at the optimum — only if the reward-greedy policy sits under budget on its
own. The context's occupancy-measure argument (linear returns, convex feasible set, zero duality gap)
guarantees there is a genuine saddle to find, so the debt I am about to measure is debt against a
reachable feasible optimum. My expectation is that the reward-greedy policy sits well over `d`, the
constraint binds, and `lambda = 0` is the wrong vertex.

The "reward gradient points through the hazards" claim is mechanistic. In these Goal tasks the reward
is dense distance-reduction shaping plus a goal bonus, and hazards emit cost when the agent is inside
them but do *not* physically block movement: you can drive straight through a hazard, you just pay for
it. So the reward-maximizing trajectory is close to the geometrically shortest path, and wherever a
hazard disk sits on that line the shortest path clips through it; detouring costs distance, which
costs reward. That is why I expect cost not just above budget but many times over it — cutting
through hazards is the reward-optimal behavior the floor is optimizing for. The button task compounds
this: the agent must thread specific buttons through a denser hazard field, so more of its
reward-optimal path lies inside cost-emitting regions.

So the floor's behavior: PPO with `A = adv_r` is a competent reward optimizer, and on these
dense-reward navigation tasks it should learn well and post the highest returns on the ladder —
nothing holds it back, and every later mechanism can only trade some away. The cost is the asymmetric
other half: with no penalty on `adv_c` the agent has no reason to detour, so episodic cost should sit
far above budget, many times `d = 25` where hazard density is high, worst on SafetyPointButton1-v0
and with no lucky safe seed because there is no channel to make one. Because the over-budget amount
differs environment by environment, the denominator has to be a per-environment, per-seed table:
"how much cost did the mechanism remove" is only meaningful against the specific excess of *that*
environment.

One implementation point that would break the run if skipped: even with the multiplier frozen at
`0.0`, I still register `Metrics/LagrangeMultiplier` in `_init_log` and store it every `_update`,
because the base loop's logging contract expects that key — a missing registration faults the logger,
a registered-but-never-stored key leaves a gap the aggregation chokes on. Logging a constant zero
also makes the multiplier column comparable across the ladder, flat here and rising once a mechanism
switches on. The `assert not np.isnan(Jc)` is the same contract-keeping: I read and sanity-check the
cost statistic even though I do not yet act on it, so a corrupted signal surfaces loudly here rather
than silently poisoning the first real multiplier update.

So the edit is the trivial one, and its triviality is the point: keep `lambda` pinned at zero,
register the multiplier key, read and NaN-check `Jc` without acting on it, and return `adv_r` so the
cost stream is dropped (the module is in the answer). What I watch is not the high return — that is
expected and is not the achievement — but the *distance* the cost sits above `25` on all three
environments and every seed. That distance is the safety debt the next mechanism must pay down, and
the return it surrenders to do so is the price I will then measure.
