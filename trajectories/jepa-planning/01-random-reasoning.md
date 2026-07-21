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
any particular direction before I have looked at the cost, so the mean should be zero in every action
coordinate at every timestep and the coordinates independent and identically spread — an isotropic
standard normal over the whole `[plan_length, num_samples, action_dim]` tensor. Crucially the
per-timestep actions are drawn *independently*, which makes a sampled sequence a white-noise action
signal, and I should follow that white signal through the model to position, because that is where its
weakness lives. To first order the model drives the agent's location by the running integral of the
commanded action, so a white action sequence integrates to a discrete random walk in position — and a
random walk does not go anywhere: the independent zero-mean increments cancel, so expected net
displacement is zero and expected *squared* displacement grows only linearly in the number of steps.
The reach of a white plan therefore scales like √T, while a plan that commits to one direction reaches
like T, and the gap that ratio opens is the whole story. At unit per-step spread a white plan's
endpoint RMS is about `√(1.9·T)` cells — roughly 7.6 at horizon 30, 13 at horizon 90 — while the same
energy spent pointing one way reaches `2.45·T`, i.e. 73 and 220. So the committed-over-white reach
ratio is about 10× at horizon 30 and *widens* toward 17× at horizon 90, because the numerator grows
like T and the denominator only like √T. In a 65×65 grid where the goal sits tens of cells away and,
worse, on the far side of a wall so the plan must first reach a doorway near the middle, a typical white
plan simply does not travel far enough in a committed direction to get through the door. That is the
structural failure I want the reference rung to *exhibit*, not hide — the gap a smarter noise source or
an iterative refit will later have to close — and the cleanest way to make it legible is to leave the
noise white and single-pass here.

And this reach deficit is exactly what the cost penalizes, not some tangential property. The objective
scores how close the predicted *final* latent is to the goal latent, so the argmin over the batch picks
the sequence whose endpoint the model believes lands nearest the goal. Reach is almost the whole of what
the cost measures, and every one of the 200 candidates shares the same reach ceiling — the white cloud
is being asked to produce an endpoint near a goal it structurally cannot travel far enough to approach.

The floor is not simply zero, though, because I do not return a typical white plan — I return the best
of 200. The best endpoint among 200 draws reaches meaningfully farther than a typical one, enough to
occasionally catch a doorway when the goal is not too far, and the receding-horizon loop then re-draws
a fresh batch every control step and gets another 200 lottery tickets. That is the whole mechanism by
which a planner that learns nothing still solves some episodes: best-of-batch reach, repeated across up
to 200 env steps. But it is a *lottery*, not a search — the argmin needs the winning sequence to reach
far *and* point toward the door, and the floor cannot steer the 200 draws toward the door, it can only
hope enough of them happen to aim there and pick the longest. Every rollout returns a cost the floor
uses only for a single argmin and then throws away, which is the precise sense in which it wastes
information: it pays full price for the measurements and discards all but one each call.

Before I commit to that white-noise draw, let me be explicit about why I am refusing the two things a
cleverer planner would reach for, because the refusals are what define the floor. The first is the
gradient. The whole pipeline — `unroll` through the JEPA model, then `objective` on the predicted
latents — is differentiable end to end, so in principle I could backpropagate the cost to the action
sequence and descend it directly. I am deliberately not doing that yet: I have not verified that the
learned model's cost surface over action sequences is smooth enough that its gradient points anywhere
useful, and a world-model rollout composes many nonlinear predictor steps whose gradient can be
dominated by sharp, locally-misleading directions that send a descent straight into the wall. Until I
have seen the zeroth-order methods' behavior I have no basis for trusting it. The second refusal is
*adaptation* — folding the scores back into where I sample next. That is exactly what CEM will do, and
the cleanest way to attribute CEM's eventual win to its refit rather than to some incidental advantage
is to make this rung spend the same per-batch budget but refuse to learn from it. So the floor is two
abstentions: no gradient, no adaptation.

There is one more property of the cost surface worth naming, because it governs why even the adaptive
methods will have to work hard and because it is the one thing the floor does *right*. In a
walls-and-doorway maze most goals admit two qualitatively different routes — through the door from
either side — so the cost is genuinely multi-modal, with separated low-cost basins. A single-pass random
draw treats those basins even-handedly: it draws the identical isotropic cloud every call, so the
through-the-wall basin and the around-the-door basin get equal coverage on every control step, and it
cannot get trapped the way a hill-climber would, because it never climbs. A gradient descent
initialized near the wall on the wrong side of the door would follow the local cost downhill *into* the
wall — near the wall the straight-line-to-goal direction the cost rewards points through it — and,
having committed, has no mechanism to back out. The floor cannot make that mistake. It pays for its
route-blindness with reach; the adaptive methods will pay for their reach with route-commitment risk,
and this rung is where that trade is defined.

Now the action constraint. A standard normal has unbounded support, so some sampled per-step actions
exceed the `max_norm = 2.45` cap the environment enforces, and I do not want to hand the cost a
candidate the agent could never execute — the rollout would score a fantasy stride and the argmin could
select it. The cheapest faithful fix is a radial projection into the feasible ball: multiply each action
by `(max_norm / norm).clamp(max=1.0)` (with a tiny `eps` in the denominator so a zero action does not
divide by zero), so actions already inside the ball are unchanged and actions outside get pulled exactly
to the boundary. This preserves each action's direction and only caps its length — the minimal edit that
makes every scored candidate executable, touching only the few percent of over-long draws.

How many candidates, and how many iterations? The harness hands the constructor `num_samples=200` and
`n_iters=20`, the shared defaults all the ladder's planners receive. The reference rung deliberately
ignores `n_iters` — it does one pass — and uses the 200 budget as a single batch. That is the whole
point: it spends the same per-batch sample budget the iterative methods will spend per iteration, but
only once and without folding the scores back into the distribution. This rung fires 200 rollouts per
`plan()` call; an iterative method that runs 20 refit passes fires 4000 — twenty times the compute. I am
not trying to match its compute, I am trying to isolate its mechanism: if CEM beats this rung, I want the
win attributable cleanly to the iteration and the refit, not to a larger first batch. Holding the
first-batch size equal is what makes that attribution honest.

And sampling-and-keep-the-best is not an arbitrary choice of floor but the least-adaptive member of the
only viable family. A single sample is too weak to be informative; exhaustive gridding is unavailable,
because the candidate lives in `2·plan_length` real dimensions — 60 at horizon 30, 180 at horizon 90 —
and even a coarse three-points-per-axis grid is `3^60` points, astronomically off the table. That is
precisely why the whole planner family is sampling-based: in tens of dimensions, drawing candidates and
scoring them is the only thing that fits a finite budget.

The selection rule is then forced: roll all 200 candidates through the model, read the 200 costs, and
return the single lowest-cost sequence — argmin over the batch. The whole sequence is returned (the loop
takes its first action), and nothing needs to persist across env steps, so the `t0` flag — the marker of
an episode's first call, the natural place to reset carried state — is simply irrelevant here, because
there is no carried state; every call re-draws from the identical fixed Gaussian. The plan length is
`min(plan_length, steps_left)` so the planner never proposes actions past the episode end, and the whole
method runs under `torch.no_grad()` because it only ever reads function values — the defining limitation
of the whole zeroth-order family this rung opens, which every later rung inherits until one is willing to
pay to lift it.

What I expect across the three horizons follows from the reach deficit, and its *shape* is what the next
rung must read. Success should be modest but clearly nonzero, with a large residual distance on the
episodes it misses — a white plan that fails to reach the door leaves the agent stranded on the wrong
side. At the longer horizons two effects fight: more env steps mean more re-plans and more lottery
tickets, but each individual plan is longer so a white sequence has to stay coherent over more timesteps,
and the Brownian cancellation gets *worse* the longer the sequence. My honest expectation is that the
extra re-plans win enough to lift success somewhat past the short horizon while the reach handicap caps
it well below an adaptive method at every horizon, and that `mean_steps_to_success` is large and climbing
with the horizon — when random search succeeds it succeeds slowly, because it wanders rather than
commits. I will not pretend to know the exact numbers; what I commit to is the mechanism: depressed
success and a large, horizon-climbing step count are two faces of the same Brownian reach deficit, and
both are what the next rung is on the hook to improve.

And the lever is already visible. Every one of the 200 rollouts returns a cost — a real measurement of
where the low-cost sequences live — and the floor reads all 200 only to keep one and re-draw the
identical white cloud next call. That discarded structure is exactly what an adaptive planner would
exploit: stop drawing from a frozen distribution and let it *move toward* the sequences that scored well,
iteration after iteration, so the search concentrates on the route through the door instead of re-rolling
white noise twenty times. That is the move from this single-pass floor to CEM. The full scaffold module
for this rung is in the answer.
