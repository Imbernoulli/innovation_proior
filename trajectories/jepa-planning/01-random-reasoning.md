I want the reference rung — the planner whose behavior I understand completely and whose only job is to
say what "doing nothing clever" scores, so every later method can be measured against it. The contract
is fixed and narrow: from the current encoded observation, produce an action sequence of length up to
`plan_length`, the loop will execute its first action, re-observe, and ask me again. I have a frozen
JEPA world model I can roll action sequences through (`unroll`), a fixed cost on the predicted latents
(`objective`), and the convenience `cost_function` that composes them and returns one scalar cost per
candidate sequence (lower is better). The action is 2-D with a per-step norm cap of `2.45`. So the
whole design space for the planner is: choose a set of candidate action sequences, score them with the
black-box cost, and return one. The reference rung makes the crudest possible choice at every one of
those decisions, and I want to be deliberate about *why* each crude choice is the right floor rather
than an accident.

Start from what the cost actually is. It is a black box over action sequences: hand it a batch, get
back a number each. I have no gradient I'm willing to trust yet (I haven't earned the right to assume
the model rollout is smooth enough to descend), the cost surface over action sequences in a
walls-and-doorway maze is plainly multi-modal — there are at least two qualitatively different routes
to most goals, one through the door from each side — and each evaluation is a full horizon-length
rollout of a neural world model, so it is not free. The only operation available to me is *querying*
the cost at points I pick. The question is how to pick them, and the reference answer is: pick them
once, at random, from a fixed distribution, and keep the best. No refitting, no iteration that learns
from the scores, no gradient. This is random shooting, and its single defining virtue is that it has
nothing in it that can go subtly wrong in a data-dependent way — there is no distribution being
updated from possibly-misleading early scores, no step size to mistune, no assumption about the cost's
smoothness. That is exactly what makes it the honest yardstick: whatever a cleverer planner buys, it
buys *on top of* this, and it has to beat this curve to justify the machinery it adds.

What distribution do I draw the candidate sequences from? I have no prior reason to prefer moving in
any particular direction before I have looked at the cost, so the mean of the sampling distribution
should be zero in every action coordinate at every timestep, and the coordinates should be independent
and identically spread — an isotropic Gaussian over the whole `[plan_length, num_samples, action_dim]`
tensor. A standard normal is the natural choice: zero mean, unit spread, no built-in bias toward any
route. Crucially the per-timestep actions are drawn *independently*, which means a sampled sequence is
a white-noise action signal. I should be honest that this is the weakest part of the design — white
actions integrate to a Brownian walk in position, and a Brownian walk does not go anywhere: its
expected squared displacement grows only linearly in the number of steps because the independent
increments cancel, so a fixed budget of action "energy" buys only a small net excursion from the
start. In a maze where the goal is on the *other side* of a wall, most of these jittery sequences will
never reach the door at all, let alone pass through it. But that is precisely the failure I want the
reference rung to *exhibit*, not hide — it is the gap a smarter noise source or an iterative refit will
later have to close, and the cleanest way to make that gap legible is to leave the noise white and
single-pass here.

Before I commit to that white-noise draw, let me be explicit about why I am refusing the two things a
cleverer planner would reach for, because the refusals are what define the floor. The first is the
gradient. The whole pipeline — `unroll` through the JEPA model, then `objective` on the predicted
latents — is differentiable end to end, so in principle I could backpropagate the cost to the action
sequence and descend it directly. I am deliberately not doing that yet, and the reason is honesty about
what I have earned: I have not verified that the learned model's cost surface over action sequences is
smooth enough that its gradient points anywhere useful. A world-model rollout composes many nonlinear
predictor steps, and the gradient of a deep composition can be dominated by sharp, locally-misleading
directions that send a descent straight into the wall; until I have seen the zeroth-order methods'
behavior I have no basis for trusting it. The second refusal is *adaptation* — folding the scores back
into where I sample next. That is exactly what CEM will do, and the cleanest way to attribute CEM's
eventual win to its refit rather than to some incidental advantage is to make this rung spend the same
per-batch budget but refuse to learn from it. So the floor is defined by two abstentions: no gradient,
no adaptation. Everything it does is the minimum that still produces a feasible plan.

There is one more property of the cost surface I should name, because it governs *why* even the
adaptive methods will have to work hard. In a walls-and-doorway maze most goals admit at least two
qualitatively different routes — through the door from either side — so the cost over action sequences
is genuinely multi-modal, with separated low-cost basins. A single-pass random draw treats those basins
even-handedly (it has no state to prefer one), which is the one thing the floor does *right*: it cannot
get trapped in a basin the way a hill-climber would, because it never climbs. That even-handedness is
the floor's only structural virtue, and it is worth stating so the later methods' premature-collapse
risk — committing to the wrong basin — reads as a real danger they take on in exchange for adaptivity.

Now the action constraint. A standard normal has unbounded support, so some sampled per-step actions
will exceed the `max_norm = 2.45` cap the environment enforces. I do not want to hand the cost a
candidate the agent could never actually execute, because then the rollout scores a fantasy. The
cheapest faithful fix is to project each sampled action into the feasible ball before scoring: compute
the norm of each 2-D action, and if it exceeds `max_norm`, rescale that action down to norm
`max_norm`, leaving shorter actions untouched. Concretely, multiply each action by
`(max_norm / norm).clamp(max=1.0)` (with a tiny `eps` in the denominator so a zero action does not
divide by zero) — actions already inside the ball get a factor of 1 and are unchanged, actions outside
get pulled exactly to the boundary. Feasibility is enforced before the black-box cost ever sees a
candidate, and it costs one elementwise operation.

How many candidates, and how many iterations? The harness hands the constructor `num_samples=200` and
`n_iters=20`, the shared defaults all the ladder's planners receive. The reference rung deliberately
ignores `n_iters` entirely — it does *one* pass, no refinement — and uses the `num_samples=200` budget
as a single batch of 200 candidate sequences. That is the whole point of the floor: it spends the same
per-batch sample budget the iterative methods will spend per iteration, but only once and without ever
folding the scores back into the sampling distribution. So if CEM, which reuses essentially this batch
twenty times while marching the distribution toward the elites, beats this rung, the win is
attributable cleanly to the *iteration and the refit*, not to a larger sample count on the first pass.

The selection rule is then trivial and forced: roll all 200 candidates through the model, read the 200
costs, and return the single sequence with the lowest cost — `argmin` over the batch. The whole
sequence is returned (the loop will take its first action), and there is no distribution to carry to
the next call, so nothing needs to persist across env steps and the `t0` flag is irrelevant here. The
plan length is `min(plan_length, steps_left)` so the planner never proposes actions past the end of
the episode. The entire method runs under `torch.no_grad()` because nothing here needs a gradient — I
am only ever reading function values, which is the defining limitation of the whole zeroth-order family
this rung opens.

Let me be precise about what I expect this floor to do across the three horizon benchmarks, because
the *shape* of its failure is what the next rung must read. At the short horizon (30 steps) the maze is
most forgiving — many random sequences will, by luck, drift toward the door and through it, and the
re-planning loop gets to re-sample every step, so even a dumb planner that occasionally points the
right way can stumble to the goal; I expect a modest but nonzero success rate here. At the longer
horizons (60, 90) two competing effects fight: more steps means more chances for the re-planning loop
to correct course, but each individual plan is longer, so a white-noise sequence has to stay
coherent over more timesteps to make real progress, and Brownian cancellation gets worse the longer
the sequence. I expect the success rate to be capped well below the iterative methods at every horizon,
and the `mean_steps_to_success` to be large — when random search does succeed it succeeds slowly,
because it wanders. That large step count and the depressed success rate are the two numbers the next
rung is on the hook to improve: the obvious lever is to stop throwing the scored batch away and instead
*refit the sampling distribution toward the low-cost sequences*, iteration after iteration, so the
search concentrates on the route through the door instead of re-drawing the same white noise twenty
times. That is exactly the move from this single-pass floor to CEM. The full scaffold module — the
literal `CustomPlanner` fill for this rung — is in the answer.
