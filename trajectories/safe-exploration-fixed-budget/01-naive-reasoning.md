The constraint controller is the whole point of this task, but it bolts onto a policy optimizer, and
with no controller at all that optimizer is the floor — so the thing to establish first is just
training a PPO policy on these Safety-Gymnasium tasks while the cost signal sits unused. The task asks
what update rule keeps cost under a fixed budget, and the only honest way to read any controller's
contribution is against a run that has none: I need to see how far cost drifts when nothing pulls it
back and how much reward the backbone extracts when it is free to ignore safety.

Nothing stops me from wiring a Lagrangian penalty into this very first edit and skipping straight to a
controller. I reject that because a constrained first run tangles two unknowns into one number. Suppose
it posts a cost of 40 on PointGoal — I learn almost nothing, because I do not know the *uncontrolled*
cost of the environment, so I cannot say whether the controller pulled cost down by five or by a
hundred. The framing I want to lean on later — reading the loop as a plant driven toward a setpoint —
needs the plant's open-loop behavior, where cost sits with the control input pinned at zero. If the
natural drift is 30 the controller has a gentle job; if it is 150 it must push six times harder, and a
rule tuned for the gentle case is hopelessly late for the violent one. Only a no-controller run
measures that number, so the floor is the calibration reading that sizes the problem each later rung
must solve.

Let me write down what the backbone optimizes, because the constraint enters exactly where this stops.
PPO maximizes a clipped surrogate `L^CLIP = Ê_t[min(r_t Â_t, clip(r_t, 1−ε, 1+ε) Â_t)]` with
`r_t = π_θ(a_t|s_t)/π_old(a_t|s_t)` the importance ratio and `Â_t` an advantage. The clip is a
first-order trust region: once the ratio leaves `[1−ε, 1+ε]` in the improving direction the `min`
selects the clipped branch, whose θ-gradient is zero — a flat spot that lets one batch of on-policy
data be reused for several gradient epochs without the policy walking out of the region where the
surrogate is honest. One property of the clip matters far downstream: it bounds the probability
*ratio*, not the advantage's magnitude — it never looks at `|Â_t|`. So a controller that inflates the
advantage's scale is not held in check by the clip at all. The advantage `Â_t` is generalized advantage
estimation off a learned value function, and the rollout, the two critics, the GAE that produces
`adv_r` and `adv_c`, the optimizers, and the training loop are all fixed substrate. What is mine is the
single decision of *which* advantage the surrogate ascends, and at this rung I make the null one.

The substrate hands me a cost stream for free: a second value head fit to the per-step cost, whose GAE
advantage `adv_c` is built from cost residuals by the same recursion that builds `adv_r` from reward
residuals. So `adv_r` and `adv_c` are structurally identical estimators of two different return
streams, each a `[batch]` tensor of comparable scale after the rollout is flattened — a fact I will
lean on when I combine them, since it means neither dominates the other before any weight is applied.
The surrogate multiplies the advantage elementwise against the per-transition log-ratio gradient, so
whatever `_compute_adv_surrogate` returns must also be a `[batch]` tensor — the contract is a shape
contract. A CMDP solver would combine the two into `A = adv_r − λ·adv_c` so the policy is penalized for
actions that raise cost; the naive rung refuses even that, returning `adv_r` and dropping the cost
tensor on the floor. Correspondingly the controller state is inert: the multiplier is initialized to
`0.0` and never updated in `_update`, which reads the mean episode cost only to assert it is not NaN
before calling `super()._update()` and logging the still-zero multiplier. One detail in `_init_log` is
instrumentation for later: the multiplier is registered with `min_and_max=True`, so once a controller
starts moving `λ` the spread between its epoch min and max becomes a diagnostic of how hard and how
erratically the controller pushes. At this rung that range is flat at `0.0`, on purpose.

That verbatim fill has a clean interpretation. Set `λ = 0` in `A = adv_r − λ·adv_c` and it collapses to
`A = adv_r`, exactly the tensor this rung returns. So the naive update is not a different algorithm from
the constrained ones; it is the `λ = 0` point of the whole family — the origin every controller will
move away from. That reframes what I am designing: the transferable component is *how far, and how fast,
λ should leave zero* in response to measured cost, and the floor fixes the origin so later displacement
is measurable.

One temptation before I commit to `λ = 0`: seed even the floor with a small fixed positive λ, a mild
constant safety nudge. I reject it for the same measurement reason, sharpened. A fixed positive λ bakes
an arbitrary reward-versus-cost exchange rate into a run that has told me nothing about what rate this
environment implies, and a constant is exactly what cannot work across these three environments — the
small λ that barely dents a gently-over arena leaves a six-times-over arena wildly unsafe, while a λ
large enough to rescue the worst cripples the gentlest. So a fixed nonzero penalty is neither a clean
floor nor a viable controller, and establishing that here already tells me the eventual controller must
*adapt* its weight, not merely carry a nonzero one. The only uncontaminated open-loop reading is
`λ = 0`.

Now what this floor must do. With the cost advantage discarded the policy is an unconstrained reward
maximizer: whatever earns reward in these navigation tasks it learns, without any pressure to route
around hazards, because entering a hazard costs nothing in the objective it sees. So I expect two things
at once. Reward should be as high as the backbone can make it, plausibly the highest of any rung — the
policy spends its whole capacity on return with no safety tax, setting the ceiling the safe rungs are
measured against. And cost should run far over budget, multiples of 25, because the shortest
reward-seeking path through a hazard-dense arena collects contacts at whatever rate the geometry imposes
and nothing opposes it. `budget_success_rate` should therefore be zero across the board.

That zero is load-bearing, not a boring negative. For a naive run to satisfy the budget its
reward-optimal trajectory would have to pass through under 25 units of hazard — the geometry would have
to place the rewarding path almost entirely in safe space. In a task built with hazards strewn between
the agent and its goals the reward gradient points *through* the hazard field, so I expect cost to scale
with how densely the rewarding path threads hazards. A naive run landing under 25 would be evidence the
arena's reward and cost geometries are decoupled — which would undercut the premise that this task needs
a controller at all.

It helps to translate the budget into the quantity it constrains. Cost is a per-step hazard signal
accumulated over a fixed-horizon episode (on the order of a thousand control steps), each contact-step
contributing roughly a unit, so `ep_cost ≤ 25` is to first order a bound on the *number of steps per
episode the agent may spend in a hazard* — a couple dozen out of a thousand, a few percent of the
trajectory. That is demanding, and it tells me the later controllers are not asking for a small
correction: halving cost from 50 to 25 means roughly halving the time the agent spends in hazards while
still solving the task — the kind of behavioral change a reward-only policy will resist.

The three environments should separate on how hazard-dense the reward-seeking path is. `SafetyPointGoal1`
is the gentlest geometry, so its overshoot should be smallest but still well over budget. `SafetyCarGoal1`
keeps the goal task but swaps in harder car dynamics, so the policy takes wider, less precise
trajectories and every extra unit of path length in a hazard field is extra contact — cost above
PointGoal's. `SafetyPointButton1` is the densest, buttons scattered among hazards so the rewarding
behavior forces the agent repeatedly through the thick of the field, so I expect the largest overshoot
there by a wide margin. Reward should be highest where the task is easiest to solve greedily; on the
densest arena I would not be surprised to see reward itself suffer on some seeds, where a policy
careening through a crowded arena gets tangled and fails to solve the task cleanly even as its cost runs
away.

Each environment runs three seeds, and I will read the controllers per seed, not just in the mean. The
returns are Monte-Carlo averages over stochastic rollouts from a random initialization, and with no
controller nothing damps seed-to-seed divergence — there is no λ pulling every seed toward a common
boundary — so I expect visible spread in both reward and cost, largest on the densest arena where small
path differences compound into large differences in hazards threaded. A wide naive spread tells any
later feedback controller it must smooth its input rather than react to a single epoch's noisy estimate.

The diagnosis this floor produces already points at the next rung. High reward with `budget_success_rate`
zero and cost at multiples of the limit is not a learning failure — the policy learns the task fine — it
is a *specification* failure: the objective contains no term for the constraint. The fix is to put the
constraint into the objective, turning the inert multiplier into a learned one and combining the two
advantage streams so the discarded cost signal becomes pressure the policy feels. The open-loop numbers
this floor produces tell that controller how hard it must push, per environment. The distilled scaffold
fill is in the answer.
