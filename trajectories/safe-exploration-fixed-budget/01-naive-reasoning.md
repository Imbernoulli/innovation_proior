The constraint controller is the whole point of this task, but it bolts onto a policy optimizer, and
with no controller at all that optimizer is the floor — so the thing to establish first is just
training a PPO policy on these Safety-Gymnasium tasks while the cost signal sits unused. I start here
deliberately: the task asks what update rule keeps cost under a fixed budget, and the only honest way
to read any controller's contribution is against a run that has no controller, so I can see how far
cost drifts when nothing pulls it back and how much reward the backbone can extract when it is free to
ignore safety entirely.

Let me write down what the backbone is actually optimizing, because the constraint enters precisely
where this stops. PPO maximizes a clipped surrogate
`L^CLIP = Ê_t[min(r_t Â_t, clip(r_t, 1−ε, 1+ε) Â_t)]` with `r_t = π_θ/π_old` the importance ratio and
`Â_t` an advantage; the clip is a first-order trust region — a flat spot in the loss once the ratio
leaves the band — so one batch of on-policy data is reused for several gradient epochs without the
policy walking out of the region where the surrogate is honest. The advantage is generalized advantage
estimation off a learned value function. None of that is mine to edit: the rollout, the two critics
(one fit to reward, one to cost), the GAE that produces `adv_r` and `adv_c`, the optimizers, the
training loop are all fixed substrate. What is mine is the single decision of *which* advantage the
surrogate ascends, and at this rung I make the null decision.

The substrate hands me a cost stream for free — there is a second value head fit to the per-step cost,
and `adv_c` is its GAE advantage, exactly parallel to `adv_r`. A CMDP solver would combine the two so
that the policy is penalized for actions that raise cost. The naive rung does not: `_compute_adv_surrogate`
returns `adv_r` and nothing else, so the policy gradient sees only the reward advantage and the cost
advantage is computed and then discarded. Correspondingly the controller state is inert — the
Lagrange multiplier is initialized to `0.0` and never updated in `_update`; `_update` reads the mean
episode cost only to assert it is not NaN (a substrate sanity check), then calls `super()._update()`
to run the ordinary PPO epochs and logs the still-zero multiplier. There is no `Lagrange` object, no
PID memory, no moving window — the edit is the default fill verbatim, which is why this is the floor by
construction.

Now reason about what this floor must do, because that is the entire reason to run it. With the cost
advantage discarded, the policy is an unconstrained reward maximizer. Whatever behavior earns reward in
these navigation tasks — driving straight to the goal, pressing buttons, taking the shortest path
through the arena — it will learn, and it will learn it without any pressure to route around hazards,
because entering a hazard costs nothing in the objective it sees. So I expect two things to be true at
once. First, reward should be as high as the backbone can make it, plausibly the highest of any rung
on this task: the policy spends its whole capacity on return with no safety tax. Second, cost should
run far over the budget — not marginally over, but multiples of 25, because the shortest reward-seeking
path through a hazard-dense arena collects hazard contacts at whatever rate the geometry imposes, and
nothing in the update opposes that. The binary `budget_success_rate` should therefore be zero across
the board: if the policy has no incentive to keep cost under 25, the only way it lands under 25 is by
accident of geometry, and a reward-greedy policy in a hazard field is the least likely configuration
to do so.

The three environments should separate on how hazard-dense the reward-seeking path is. `SafetyPointGoal1`
is the gentlest geometry, so its overshoot should be the smallest of the three but still well over
budget. `SafetyCarGoal1` adds harder car dynamics on the same goal task — the policy may take wider,
longer trajectories, accumulating more contacts. `SafetyPointButton1` is the densest-cost geometry
(buttons scattered among hazards), so I expect the largest overshoot there by a wide margin — this is
the environment where ignoring cost should look most catastrophic. Reward, conversely, should be
highest where the task is easiest to solve greedily.

The diagnosis this rung is built to produce is already pointed at the next one. If naive PPO posts
high reward and a `budget_success_rate` of zero with cost at multiples of the limit, the problem it
exposes is not a learning problem — the policy learns the task fine — it is a *specification* problem:
the objective the policy ascends contains no term for the constraint, so of course the constraint is
violated. The fix is to put the constraint into the objective. The substrate already exposes `adv_c`
and a multiplier slot precisely so a future rung can subtract a cost penalty from the reward advantage
and drive that penalty's weight from the measured cost. That is the move the next rung makes: turn the
inert multiplier into a learned one and combine the two advantage streams, converting the discarded
cost signal into pressure the policy actually feels. The distilled scaffold fill is in the answer.
