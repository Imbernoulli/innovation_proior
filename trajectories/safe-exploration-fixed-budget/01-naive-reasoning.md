The constraint controller is the whole point of this task, but it bolts onto a policy optimizer, and
with no controller at all that optimizer is the floor — so the thing to establish first is just
training a PPO policy on these Safety-Gymnasium tasks while the cost signal sits unused. I start here
deliberately: the task asks what update rule keeps cost under a fixed budget, and the only honest way
to read any controller's contribution is against a run that has no controller, so I can see how far
cost drifts when nothing pulls it back and how much reward the backbone can extract when it is free to
ignore safety entirely.

Before I write the null fill, let me argue that starting here is not a wasted rung but a measurement I
actually need for everything downstream. The editable surface already exposes four hooks and a
multiplier slot; nothing stops me from wiring in a Lagrangian penalty on this very first edit and
skipping straight to a controller. I walk that alternative a couple of steps to see why I reject it.
Suppose the first run I ever do is already a constrained one, and it posts, say, a cost of 40 on
PointGoal. What do I learn? Almost nothing, because I have two unknowns tangled into one number: I do
not know the *uncontrolled* cost of this environment, so I cannot say whether the controller pulled
cost down by five or by a hundred. The whole framing I am going to lean on later — reading the loop as
a plant driven toward a setpoint — needs to know the plant's open-loop behavior, i.e. where cost sits
with the control input pinned at zero. If the natural drift is 30, a controller has a gentle job; if it
is 150, the controller must push six times harder, and a rule tuned for the gentle case will be
hopelessly late for the violent one. That open-loop number is exactly what a no-controller run
measures and nothing else can. So the floor is not throat-clearing; it is the calibration reading that
tells me the magnitude of the problem each later rung has to solve. That is a real reason to spend a
rung on it, and it settles the decision: run the null controller first.

Let me write down what the backbone is actually optimizing, because the constraint enters precisely
where this stops. PPO maximizes a clipped surrogate
`L^CLIP = Ê_t[min(r_t Â_t, clip(r_t, 1−ε, 1+ε) Â_t)]` with `r_t = π_θ(a_t|s_t)/π_old(a_t|s_t)` the
importance ratio and `Â_t` an advantage estimate. The clip is a first-order trust region: once the
ratio leaves the band `[1−ε, 1+ε]` in the direction that would keep improving the surrogate, the `min`
selects the clipped branch, whose gradient in `θ` is zero — a flat spot in the loss. That flat spot is
what lets one batch of on-policy data be reused for several gradient epochs without the policy walking
out of the region where the surrogate is an honest local model of the true return; the ratio simply
stops being rewarded for moving further. The advantage `Â_t` is generalized advantage estimation off a
learned value function: with the temporal-difference residual `δ_t = r_t + γ V(s_{t+1}) − V(s_t)`, GAE
sets `Â_t = Σ_{l≥0} (γλ)^l δ_{t+l}`, a geometrically-weighted sum of residuals that trades bias for
variance through `λ`. None of that is mine to edit: the rollout, the two critics, the GAE that
produces the advantages, the optimizers, the training loop are all fixed substrate. What is mine is the
single decision of *which* advantage the surrogate ascends, and at this rung I make the null decision.

Two limit checks on this machinery keep me honest about what the fixed backbone is and is not doing,
because the later rungs inherit both. First the clip. Take a single transition with a positive
advantage `Â_t > 0` and watch the ratio move as `θ` updates: at `r_t = 1` the surrogate is `r_t Â_t`
and its gradient pushes the ratio up; as `r_t` climbs past `1+ε = 1.2` the `clip` branch caps the term
at `1.2 Â_t`, whose `θ`-gradient is zero, so further increase of `π_θ(a_t|s_t)` earns nothing — the
update on that transition stalls exactly at the trust-region edge. For a negative advantage the mirror
holds at `1−ε`: the policy is discouraged from the action but only down to the clip floor, past which
the gradient again vanishes. So the clip does not bound the *advantage's magnitude*; it bounds how far
each transition can move the policy per update, which is precisely why I can reuse one rollout for
several epochs. That distinction bites later — a controller that inflates the advantage's scale is not
held in check by the clip, because the clip never looks at `|Â_t|`. Second the estimator. GAE has two
limits: `λ_GAE = 0` gives the one-step advantage `Â_t = δ_t`, low variance but biased by whatever error
the value function carries; `λ_GAE = 1` gives the full Monte-Carlo advantage `Σ γ^l δ_{t+l}`, unbiased
but high variance. The substrate runs both `adv_r` and `adv_c` at the same intermediate `λ_GAE`, so the
two streams carry the same bias/variance character — a fact I will lean on when I combine them, since
it means their scales are comparable before any multiplier touches them.

The editable interface also quietly narrows my design space to exactly one lever, and it is worth
seeing why. The obvious ways to enforce a safety budget in RL — shaping the reward with a hazard
penalty, terminating episodes early on cost, or reshaping the environment — are all off the table here:
the reward function, the rollout, and the environments are fixed substrate. The two critics and their
GAE outputs arrive already computed. So the *only* place the cost signal can enter the policy is the
advantage `_compute_adv_surrogate` returns and the multiplier state `_update` maintains. That is not a
limitation to work around; it is the task's claim that the constraint-handling rule is a single
transferable object — the map from measured cost to the combined advantage — and everything else is
held constant so that map is what gets measured. It also means my floor is genuinely a floor: with the
reward stream fixed and unshaped, returning `adv_r` is the maximal-reward, zero-safety corner of the
one lever I have. There is no configuration of the editable region that earns *more* reward than this
one, because any nonzero use of `adv_c` can only pull the update away from pure reward ascent.

The substrate hands me a cost stream for free, and it is worth being precise about its shape because
the parallelism is what makes every later rung a one-liner. There is a second value head fit not to the
reward return but to the per-step cost, so it estimates `V_c(s) ≈ E[Σ γ^k c_{t+k}]`, and its GAE
advantage `adv_c` is built from cost residuals `δ^c_t = c_t + γ V_c(s_{t+1}) − V_c(s_t)` by the exact
same recursion that builds `adv_r` from reward residuals. So `adv_r` and `adv_c` are two structurally
identical estimators of two different return streams, each a per-transition tensor of shape `[batch]`
after the rollout is flattened. The PPO surrogate multiplies the advantage elementwise against the
per-transition log-ratio gradient and averages, so whatever `_compute_adv_surrogate` returns must also
be a `[batch]` tensor — the contract is a shape contract. A CMDP solver would combine the two streams
into a single such tensor so the policy is penalized for actions that raise cost; the standard
combination from the Lagrangian-duality background is `A = adv_r − λ·adv_c` for a nonnegative weight
`λ`. The naive rung refuses even that: `_compute_adv_surrogate` returns `adv_r` and nothing else, so
the cost tensor is allocated by the substrate's cost-GAE and then dropped on the floor — I am paying to
compute a signal I discard. Correspondingly the controller state is inert: the multiplier is
initialized to `0.0` and never updated in `_update`, which reads the mean episode cost only to assert
it is not NaN (a substrate sanity check on the cost statistic) before calling `super()._update()` to
run the ordinary PPO epochs and then logging the still-zero multiplier so the harness's metrics line
keeps its key. There is no `Lagrange` object, no controller memory, no moving window — the edit is the default
fill verbatim, which is why this is the floor by construction. One detail in `_init_log` is worth
noticing because it is instrumentation for the rungs to come: the multiplier is registered with
`min_and_max=True`, so the logger will track the epoch range of `λ`, not just its mean. At this rung
that range is degenerate — min and max are both `0.0` on every epoch, a perfectly flat trace — but the
key is already wired so that the moment a controller starts moving `λ`, the spread between its min and
max over an epoch becomes a readable diagnostic of how hard and how erratically the controller is
pushing. The floor leaves that instrument reading zero on purpose.

That verbatim fill has a clean interpretation I can pin down with one substitution, and pinning it down
tells me what the entire task is really parameterizing. Take the generic Lagrangian combination
`A = adv_r − λ·adv_c` and set `λ = 0`: it collapses to `A = adv_r`, exactly the tensor this rung
returns. So the naive update is not a different algorithm from the constrained ones; it is the `λ = 0`
point of the whole family — the origin every controller will move away from. Reading it that way
reframes the task precisely: the transferable component I am designing is *how far, and how fast, λ
should leave zero* in response to measured cost. The floor fixes the origin so that later I can measure
displacement from it. This is a real check, not decoration — it verifies that the null fill is the
degenerate member of the family I am about to build, so nothing about the floor is off to one side of
the design space; it is dead center at the zero of the one knob that matters.

One more temptation to dispatch before I commit to `λ = 0`: since the Lagrangian background already
hands me the penalty form `A = adv_r − λ·adv_c`, why not seed even the floor with a small fixed
positive `λ` — a mild constant safety nudge — so the run is not quite so reckless? I reject it, and the
reasoning is the same measurement argument sharpened. A fixed positive `λ` bakes an arbitrary
reward-versus-cost exchange rate into the floor, and it is arbitrary precisely because I have no run
yet that tells me what exchange rate this environment implies. Worse, a fixed constant is the one thing
I already know cannot work across these three environments: the same small `λ` that barely dents a
gently-over-budget arena would still leave a six-times-over arena wildly unsafe, and a `λ` large enough
to rescue the worst arena would cripple the gentlest. So a fixed nonzero penalty is neither a clean
floor (it contaminates the open-loop reading) nor a viable controller (a constant cannot serve three
different violation magnitudes), which is a genuinely useful thing to have established here rather than
discovered later — it tells me the controller I eventually build must *adapt* its weight, not merely
carry a nonzero one. The only value that gives me an uncontaminated open-loop reading is exactly zero,
so `λ = 0` it is.

Now reason about what this floor must do, because that is the entire reason to run it. With the cost
advantage discarded, the policy is an unconstrained reward maximizer. Whatever behavior earns reward in
these navigation tasks — driving straight to the goal, pressing buttons, taking the shortest path
through the arena — it will learn, and it will learn it without any pressure to route around hazards,
because entering a hazard costs nothing in the objective it sees. So I expect two things to be true at
once. First, reward should be as high as the backbone can make it, plausibly the highest of any rung
on this task: the policy spends its whole capacity on return with no safety tax, so this run sets the
reward ceiling the safe rungs will be measured against. Second, cost should run far over the budget —
not marginally over, but multiples of 25, because the shortest reward-seeking path through a
hazard-dense arena collects hazard contacts at whatever rate the geometry imposes, and nothing in the
update opposes that. The binary `budget_success_rate` should therefore be zero across the board: if the
policy has no incentive to keep cost under 25, the only way it lands under 25 is by accident of
geometry, and a reward-greedy policy in a hazard field is the least likely configuration to do so.

Let me sharpen that last claim into something falsifiable rather than leaving it as a hunch. The budget
is `d = 25` accumulated cost over an episode. For a naive run to satisfy it, the reward-optimal
trajectory would have to happen to pass through fewer than 25 units of hazard contact — that is, the
geometry would have to place the rewarding path almost entirely in safe space. In a task explicitly
built with hazards strewn between the agent and its goals, the reward gradient points *through* the
hazard field, not around it, so the reward-optimal path is close to the most-cost path, not the least.
The prediction that follows is concrete: not only should `budget_success_rate` be zero, but cost should
scale with how densely the rewarding path threads hazards. If instead I ever saw a naive run land under
25, that would be evidence the arena's reward and cost geometries are decoupled — the rewarding path
avoids hazards on its own — which would undercut the whole premise that this task needs a controller at
all. So the zero I expect is not a boring negative; it is the load-bearing observation that reward and
safety genuinely conflict here.

It helps to translate the budget into the physical quantity it constrains, because that fixes how tight
25 actually is and therefore how much work the controller must do. Cost here is a per-step hazard
signal accumulated over a fixed-length episode; in these Safety-Gymnasium tasks an episode runs a fixed
horizon (on the order of a thousand control steps), and each step spent in contact with a hazard
contributes roughly a unit of cost. So `ep_cost ≤ 25` is, to first order, a bound on the *number of
steps per episode the agent is allowed to be in a hazard* — a couple dozen out of a thousand, a few
percent of the episode. That is a demanding budget: the agent must complete the navigation task while
touching hazards for only a small fraction of its trajectory. A naive reward maximizer has no term
that counts those contact-steps, so it will spend whatever fraction the shortest rewarding path
happens to require, and the multiples-of-25 overshoot I expect is just that fraction running to ten or
twenty percent of the episode instead of two. Reading the budget this way also tells me the later
controllers are not asking for a small correction — halving cost from 50 to 25 means roughly halving
the time the agent spends in hazards while still solving the task, which is exactly the kind of
behavioral change a reward-only policy will resist.

The three environments should separate on how hazard-dense the reward-seeking path is.
`SafetyPointGoal1` is the gentlest geometry — a point robot with simple dynamics navigating to goals —
so its overshoot should be the smallest of the three but still well over budget. `SafetyCarGoal1`
keeps the goal task but swaps in car dynamics, which are harder to control: the policy may take wider,
longer, less precise trajectories, and every extra unit of path length in a hazard field is extra
expected contact, so I expect its cost above PointGoal's. `SafetyPointButton1` is the densest-cost
geometry — buttons scattered among hazards, so the rewarding behavior (reach and press the right
button) forces the agent through the thick of the hazard field repeatedly — so I expect the largest
overshoot there by a wide margin, the environment where ignoring cost looks most catastrophic. Reward,
conversely, should be highest where the task is easiest to solve greedily; on the densest arena I would
not be surprised to see reward itself suffer on some seeds, because a policy careening greedily through
a crowded arena can get tangled and fail to solve the task cleanly even as its cost runs away — the one
place where ignoring safety costs *reward* too, through sheer congestion.

I should also set an expectation for the seed spread, since each environment is run at three seeds and
I will read the controllers against this floor seed by seed, not just in the mean. The reward and cost
returns are Monte-Carlo averages over stochastic rollouts, and the policy starts from a randomly
initialized network, so different seeds explore different paths early and can settle into different
reward-seeking routes. With no controller, nothing damps that divergence — there is no `λ` pulling
every seed toward the same feasible boundary — so I expect visible seed-to-seed spread in both reward
and cost, and I expect it largest on the densest-hazard arena, where small differences in the chosen
path compound into large differences in how many hazards it threads. That matters for how I read later
rungs: a controller that satisfies the budget on some seeds and misses on others is not necessarily
inconsistent, because the floor itself is variable; the honest comparison is per seed against this
uncontrolled baseline. A wide naive spread would tell me the plant is noisy, which in turn tells any
feedback controller it must smooth its input rather than react to a single epoch's estimate.

The diagnosis this rung is built to produce is already pointed at the next one. If naive PPO posts
high reward and a `budget_success_rate` of zero with cost at multiples of the limit, the problem it
exposes is not a learning problem — the policy learns the task fine — it is a *specification* problem:
the objective the policy ascends contains no term for the constraint, so of course the constraint is
violated. The fix is to put the constraint into the objective. The substrate already exposes `adv_c`
and a multiplier slot precisely so a future rung can subtract a cost penalty from the reward advantage
and drive that penalty's weight from the measured cost — moving `λ` off the zero this rung pins it to.
That is the move the next rung makes: turn the inert multiplier into a learned one and combine the two
advantage streams, converting the discarded cost signal into pressure the policy actually feels. The
open-loop cost numbers this floor produces — how far over 25 each environment drifts — are what tell
that controller how hard it has to push, per environment. The distilled scaffold fill is in the answer.
