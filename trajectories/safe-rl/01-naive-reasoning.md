The mechanism is the whole point, but it bolts onto a base learner, and with no mechanism that learner
is the floor — so the thing to establish first is what the agent does when safety is ignored
entirely. The substrate hands me a constrained-PPO loop with two value functions already wired: one
GAE pass on the reward signal producing `adv_r`, a second on the cost signal producing `adv_c`. The
loop is built to blend them, but at the start nothing forces it to. So the weakest possible fill is
the one that throws the cost stream away: leave the multiplier at zero and have the advantage hook
return `adv_r` untouched. That is plain reward-maximizing PPO running inside a safe-RL harness with
the safety apparatus disconnected.

Let me be precise about why this is the floor and not just *a* baseline, because the whole ladder
hangs off getting this anchor right. The two slots I can edit are the multiplier update in `_update`
and the advantage combination in `_compute_adv_surrogate`. In the default fill, `_update` reads the
measured mean episodic cost `Jc` (only to assert it is not NaN, a sanity check on the logging) and
then calls `super()._update()` without ever moving `lambda` off its initial `0.0`. And
`_compute_adv_surrogate(adv_r, adv_c)` returns `adv_r`, discarding `adv_c` completely. I want to
trace the gradient path explicitly, because "ignores safety" is easy to say and easy to get subtly
wrong. PPO's policy loss is `E[min(rho_t * A_t, clip(rho_t) * A_t)]` with `rho_t = pi_theta/pi_old`
the probability ratio and `A_t` the advantage that `_compute_adv_surrogate` returns. The gradient of
that loss with respect to the policy parameters `theta` is proportional to `E[grad log pi * A]` — so
it only ever contains whatever advantage I hand it. With `A = adv_r`, the term `adv_c` appears
nowhere in `grad_theta`. Meanwhile the cost critic is a *separate* value head with its own parameters;
it keeps training, regressing its predictions onto the empirical cost returns, and `adv_c = GAE(cost
signal, V_c)` is computed every epoch — but that number is then dropped on the floor of
`_compute_adv_surrogate`. So the cost apparatus spins with the clutch fully disengaged: the cost
critic learns an accurate cost value function that the policy never once consults. There is, by
construction, no channel through which a safety violation can change the agent's behavior. That is
what makes this the constraint-unaware control, and it is why it is *the* floor rather than one
baseline among many: it tells me the upper bound on reward obtainable when nothing is paid for cost,
and the uncontrolled cost the environment produces when the agent optimizes reward freely.

It is worth being concrete about the substrate I am plugging into, because the shape of what it hands
me is what makes the zero fill so clean and what the next mechanism will exploit. The loop keeps two
value functions, not one, and runs GAE twice — once on the reward stream, once on the cost stream —
to produce two advantage tensors per batch: `adv_r` and `adv_c`, each of shape `[batch]`, one scalar
advantage per collected transition. Why two separate critics rather than a single critic trained on a
blended target `r - beta*c`? Because the blend weight is exactly the thing that is supposed to be
adaptive. A single critic regressed onto `r - beta*c` would have to be *retrained* every time the
trade-off weight moved, since its regression target would shift underneath it; its learned values
would be perpetually stale relative to the current weight. Keeping reward and cost critics separate
lets each learn a fixed-target value function — `V_r` fits reward returns, `V_c` fits cost returns,
neither depending on any trade-off parameter — and defers the blending all the way down to the
policy-gradient stage, where `_compute_adv_surrogate` combines the two *advantages* with the current
weight. That architecture is the reason the substrate exposes an advantage hook at all, and at the
floor it means the reward critic and cost critic both train correctly and independently; I simply
route only the reward advantage into the policy step.

Let me trace the shapes through the hook to be sure the floor leaves the standard PPO update
byte-for-byte intact. `_compute_adv_surrogate` takes two `[batch]` tensors and must return one
`[batch]` tensor — the single advantage the clipped surrogate consumes. At the floor that map is the
identity on its first argument: `return adv_r`. So the tensor that flows into the PPO step has exactly
the shape, dtype, and (crucially) the *statistics* of a plain reward advantage — the substrate's usual
advantage normalization, clip range, and learning rate all see precisely what they were tuned for. The
floor is therefore not just "PPO ignoring cost" in spirit; it is numerically the untouched PPO update,
which is what lets me read its return as a genuine ceiling rather than something contaminated by an
odd advantage scale. The moment `adv_c` enters with a nonzero weight, that scale question becomes live
— but that is a problem for the step that turns the weight on, not for the floor.

There is a semantic point about what `adv_c` even means that clarifies why dropping it is precisely
"ignoring safety" and not something milder. GAE on the cost stream accumulates discounted cost
temporal-difference errors, so `adv_c` for a transition is how much *more* episodic cost this action
is expected to incur relative to the cost critic's baseline from that state — a per-action estimate of
marginal danger. A positive `adv_c` marks an action that leads into more cost than the state's average;
a negative one marks a safer-than-average action. The constraint itself, though, is not on any single
action's danger — it is on the *expected episodic* cost `J_c`, the sum of per-step costs over a
trajectory, which is why `_update` reads `get_stats('Metrics/EpCost')`, the mean episodic cost across
workers, rather than any per-step quantity. At the floor I read that episodic number and do nothing
with it, and I compute the per-action `adv_c` and discard it. So both channels through which safety
could enter — the aggregate episodic signal that would drive a multiplier, and the per-action marginal
signal that would tilt the policy gradient — are open and measured yet deliberately unused. That is the
fullest sense in which this is the constraint-unaware control.

One parameter I am deliberately reading past: the substrate exposes a dual step size defaulting to
`0.035` and the budget `cost_limit = 25.0`. At the floor I touch neither the step size (there is no
dual update to take) nor act on the budget (I read `Jc` but do not compare it to `25`). I note them
because they are the knobs the mechanism will engage, and leaving them idle here is what keeps this a
true zero-mechanism baseline. The broader reassurance from the context is that the constrained problem
is well-posed: over state-action occupancy measures the reward and cost returns are linear and the
feasible set is convex, so the constrained problem has zero duality gap. That matters even at the floor
because it means the frontier I am anchoring the reward-greedy end of is a real, well-defined object —
the naive point is the `lambda = 0` vertex of a dual that genuinely has a saddle to find, so the debt I
am about to measure is debt against a reachable feasible optimum, not against a mirage.

I should convince myself that the exactly-zero fill is forced, not merely convenient, by walking the
handful of other things I could put in these two slots and seeing each one disqualify itself. The
first alternative is a small fixed penalty: return `adv_r - beta*adv_c` for some tiny `beta > 0`.
That is already a mechanism — reward shaping — and it poisons the very thing the floor is supposed to
measure. If I bake even a small `beta` into the fill, I no longer know the *uncontrolled* cost,
because I have already started controlling it; the denominator every later method is judged against
would have a mechanism smuggled into it. The second alternative is to initialize `lambda` at some
positive constant and never update it, which is algebraically identical to the fixed-`beta` case with
`beta = lambda`, and dies for the same reason. The third alternative is to leave `_compute_adv_surrogate`
returning `adv_r` but add an actual multiplier update in `_update` — but a multiplier that updates
*is* the mechanism I am supposed to design at the next step; putting it here would mean the floor is
no longer a floor. And a fourth, sillier alternative — anchoring against a random policy or a
zero-reward agent — fails a different way: the anchor I need is specifically "reward-greedy, safety
ignored," because the questions I will ask of every later mechanism are "how much return did it give
up" and "how much cost did it remove," and both are only meaningful relative to the reward-greedy,
cost-ignored corner. A random policy anchors neither quantity. So among all the fills available in
these two slots, only `lambda = 0` with `A = adv_r` gives a clean, mechanism-free denominator, and it
is forced.

And because the floor cost will differ environment by environment, that denominator has to be a
per-environment, per-seed object, not a single averaged number. "How much cost did the mechanism
remove" is only meaningful against the specific over-budget amount of *that* environment: pulling the
dense button task down to the budget is a far larger reduction than pulling a sparser goal task down,
if the button task started much higher. Averaging across environments would blur exactly the
differences I most need to see — which environment a later mechanism saved cheaply and which it could
barely touch. So I record the floor as the full per-environment, per-seed table it is, and every later
comparison reads column by column against it. That is also why the naive point being an *endpoint* of
the frontier matters more than its being merely high or low: an endpoint gives a fixed reference the
whole trade-off curve can be measured from, one environment at a time.

There is a tidy way to see the same thing through the constrained-optimization lens the context lays
out, which also tells me *why* this floor is going to be unsatisfactory and thus why a mechanism is
needed at all. The safe problem is `max_theta J_r(theta)` subject to `J_c(theta) <= d`, dualized as
`L = J_r - lambda*(J_c - d)` with `lambda >= 0`. The naive fill is exactly the `lambda = 0` point of
that Lagrangian: `L = J_r`, the unconstrained reward maximizer. Complementary slackness says
`lambda = 0` is the correct multiplier *only if the constraint is slack* at the optimum — only if the
reward-greedy policy happens to sit under budget on its own. My whole expectation is that it does not:
navigation reward on these tasks points straight through the hazards, so the unconstrained optimum
will sit well over `d`, the constraint will bind, and `lambda = 0` will be the wrong vertex. That is
not a flaw in the floor; it is the floor telling me, in the language of the dual, that a positive
multiplier is required — which is precisely the mechanism the next step introduces. So the floor is
simultaneously the reward-greedy anchor and the diagnosis that a multiplier is needed.

I want to make the "reward gradient points through the hazards" claim mechanistic rather than a slogan,
because the whole prediction rests on it. In these Goal tasks the reward is dense distance-reduction
shaping — the agent is rewarded roughly in proportion to how much closer to the goal it got this step —
plus a bonus for reaching the goal. The hazards are regions that emit cost when the agent is inside
them but do *not* physically block movement: you can drive straight through a hazard, you just pay for
it. So the reward-maximizing trajectory is close to the geometrically shortest path to the goal, and
whenever a hazard disk sits on or near that straight line, the shortest path clips through it. Detouring
around the hazard costs distance, which costs reward, so a pure reward maximizer with no cost term will
prefer the through-the-hazard route unless the detour is trivially short. This is why I expect cost to
land not just above budget but *many times* over it: it is not that the agent stumbles into hazards, it
is that cutting through them is the reward-optimal behavior the floor is explicitly optimizing for. The
button task compounds this — the agent must approach specific buttons threaded through a denser hazard
field, so more of its reward-optimal path lies inside cost-emitting regions, which is the concrete
reason I expect its cost to dominate the other two.

Now I should reason about what this floor will actually do, because that prediction is what the next
step diagnoses against, and I want it sharp enough to be falsifiable. PPO with `A = adv_r` is a
competent reward optimizer. On these Safety-Gymnasium navigation tasks the reward is reasonably dense
— the agent gets shaped signal for moving toward goals — so PPO should learn the task well and post
the *highest* returns of anything on the ladder. There is nothing holding it back from the reward;
every later mechanism can only trade some of this away. The cost is the other half of the picture. The
hazards in these environments are placed exactly where the shortest, highest-reward paths run; the
reward gradient points straight through them. With no penalty on `adv_c`, the agent has no reason to
detour, so it will plow through hazards whenever that is the fast way to a goal, and the episodic cost
should sit far above the budget — many times `d = 25` on the environments where the hazard density is
high. The expectation is asymmetric and clean: strong on return, badly out of bounds on cost. That is
the signature of ignoring the constraint, and it is the failure the first real mechanism has to fix.

Let me verify that reading against a limiting case, by mentally running the *opposite* extreme fill
and checking the floor sits where I claim. Suppose instead of `lambda = 0` I drove `lambda` enormous —
equivalently returned `-adv_c` from the hook. Then the policy would minimize cost while ignoring
reward: it would learn to sit still or circle in open space, driving cost toward whatever irreducible
floor the environment imposes and collapsing return toward zero. That is the far corner of the
achievable region. The naive fill is the exact opposite corner: maximal return, maximal (uncontrolled)
cost. So on the (return, cost) plane the two constant-`lambda` extremes mark the two ends of the
frontier, and every mechanism I build later lives somewhere on the interior curve between them, buying
cost reduction with return. This confirms the naive point is a genuine endpoint of the achievable set
rather than an arbitrary interior sample — which is exactly the property an anchor needs.

One subtlety worth naming, because it sets up the per-environment comparison the next step will read
seed by seed. The cost will not be uniform across the three environments, and neither will the return.
SafetyPointButton1-v0 packs the most hazards around its buttons and asks the agent to touch a
sequence of correct buttons while avoiding wrong ones, so I expect its naive cost to be the largest by
a wide margin — the reward-greedy path there threads through the densest hazard field on the ladder.
SafetyPointGoal1-v0 and SafetyCarGoal1-v0 are single-goal navigation with sparser hazards, so their
naive cost should be high but far less extreme than the button task. The return ordering should
roughly track control difficulty rather than hazard density: the point robot has simple, near-holonomic
dynamics and should reach its goals cleanly, so PointGoal return should be the healthiest; the car has
harder second-order dynamics — it has to manage momentum and steering — so CarGoal return should be
depressed relative to the point robot even though its task is nominally the same "reach the goal"; and
the button task's longer, multi-stage interaction should make its return both lower and the *noisiest*
across seeds, because a run that mislearns which button to hit tanks that seed's return while a run
that gets it produces a much higher one. None of this changes the diagnosis: wherever cost lands, it
lands with nothing pulling it down, so it is the worst-case cost, and the job of step 2 is to
introduce a multiplier that reads that violation and starts paying for it.

A last implementation point that is easy to overlook but will break the run if I skip it: even though
the multiplier is a frozen `0.0`, I still register `Metrics/LagrangeMultiplier` in `_init_log` and
store it every `_update`. The base loop's logging contract expects that key to exist — the substrate
prints and aggregates it — so a missing registration would fault the logger, and a registered-but-never-stored
key would leave a gap the aggregation chokes on. Logging a constant zero is not decorative here; it is
what keeps the frozen substrate's bookkeeping consistent, and it also means the multiplier column in the
output is directly comparable across every step of the ladder, reading a flat zero at the floor and
rising once a mechanism switches on. The `assert not np.isnan(Jc)` in `_update` is the same kind of
contract-keeping: I read the cost statistic and sanity-check it even though I do not yet act on it, so
that a corrupted cost signal surfaces loudly here rather than silently poisoning the multiplier update
the moment one is added.

I want the prediction stated so that a result could actually falsify it, using only the two metrics the
harness reports. First: return should be the highest this ladder ever posts, on every environment,
because nothing here trades reward away — if a later, safety-handling method matched or beat the naive
return while also controlling cost, the naive point would not be a genuine reward ceiling and my whole
denominator argument would be suspect. Second: cost should exceed `25` on all three environments and on
every seed, with the button task's cost the largest — if any environment came in *under* budget with no
mechanism, the premise that the reward-optimal path crosses hazards would be false there, and no
constraint handling would be needed for that environment at all. Third, on across-seed spread: because
there is no mechanism to create a lucky safe run, every seed should converge into the same reward-greedy
basin, so I expect the cost over-budget to be *systematic* — low relative spread on the goal tasks,
with the button task noisiest in return (a seed that mislearns which button to press tanks its return)
but still uniformly over budget in cost. If I saw one clean, safe seed sitting under `25`, that would
contradict the claim that the floor has no channel to safety and would demand I re-examine the
gradient-path trace above. Those three are the concrete ways the floor reading could be wrong, and I do
not expect any of them to fire.

So the edit is the trivial one, and its triviality is the point: keep `lambda` pinned at zero, register
the multiplier key so the logging contract is satisfied, read and NaN-check `Jc` without acting on it,
and return `adv_r` from the advantage hook so the cost stream is dropped (the distilled module is in
the answer). What I will watch in the result, using only the two metrics the harness reports —
episodic return and episodic cost, per environment and per seed — is the gap between this run and a
controlled-cost method. High return is expected and is not the achievement; the achievement I am setting
up is the *distance* the cost sits above `25`, because that distance is exactly the safety debt the next
mechanism must pay down, and the amount of return it has to surrender to do so is the price I will then be
measuring. I expect that debt to be largest on SafetyPointButton1-v0 and present on all three
environments and every seed, with no lucky safe run, because there is no mechanism to make one. The naive
run is the constraint-unaware ceiling on reward and the uncontrolled floor on safety; from here every step
trades down the first to fix the second.
