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
a white-noise action signal, and I should follow that white signal through the model to position,
because that is where its weakness lives. To first order the model drives the agent's location by the
running integral of the commanded action — position is the cumulative sum of the per-step displacements
— so a white action sequence integrates to a discrete random walk in position. A random walk does not
go anywhere: because the increments are independent and zero-mean, the expected *net* displacement is
zero, and the expected *squared* displacement grows only linearly in the number of steps, since the
independent increments have no cross-correlations to reinforce each other. So the reach of a
white-noise plan scales like the square root of the horizon, while a plan that commits to one direction
reaches like the horizon itself.

Let me put actual numbers on that gap, because it governs how badly the floor bleeds and I would rather
compute it than assert it. A single 2-D standard-normal action has squared norm distributed as a
chi-squared with two degrees of freedom (mean 2), i.e. an exponential with mean 2, so the raw
per-step energy is `E‖a‖² = 2`. The `max_norm = 2.45` clip removes the tail: `P(‖a‖ > 2.45) =
exp(−2.45²/2) = exp(−3.00) ≈ 0.050`, so only about five percent of sampled actions are pulled in, and
after the clip `E‖a‖²` drops only from `2.0` to about `1.90` — an RMS per-step displacement of about
`1.38`. Carry that to the endpoint of a `T`-step white plan: per coordinate the endpoint variance is
about `0.95·T`, so the net displacement magnitude is Rayleigh with RMS `√(1.90·T)`. That is `7.6` cells
at `T = 30`, `10.7` at `T = 60`, `13.1` at `T = 90`. Now compare a plan that spends the *same*
per-step energy but points every step the same way: it reaches `2.45·T`, i.e. `73`, `147`, `220` cells.
The ratio — committed reach over white reach — is `9.7×` at horizon 30 and *widens* to `16.9×` at
horizon 90, because the numerator grows like `T` and the denominator only like `√T`. In a `65×65` grid
where the goal typically sits tens of cells away and, worse, on the *far* side of a wall so the plan
must first reach a doorway near the middle of the grid, a typical white plan simply does not travel far
enough in a committed direction to get through the door. That is the structural failure I want the
reference rung to *exhibit*, not hide — it is the gap a smarter noise source or an iterative refit will
later have to close, and the cleanest way to make that gap legible is to leave the noise white and
single-pass here.

It matters here exactly *what* the cost rewards, because that decides whether the reach deficit I just
computed is the thing being penalized. The objective scores how close the predicted *final* latent is to
the goal latent — the cost is essentially the distance-to-goal at the end of the horizon-length rollout.
So the argmin over the batch is picking the sequence whose endpoint the model believes lands nearest the
goal, and reach is not some tangential property: it is almost the whole of what the cost measures. A
white plan that only travels `7.6` cells when the goal is thirty cells away and behind a wall will be
scored as bad *because* it did not reach, and every one of the 200 candidates shares that same reach
ceiling. The clip-projected white cloud is being asked to produce an endpoint near a goal it structurally
cannot travel far enough to approach. This is why I trust the Brownian arithmetic to translate into the
score rather than being an academic aside — the metric is reach, and reach is what white noise is worst
at.

There is one saving grace worth quantifying, because it explains why the floor is not simply zero. I do
not return a typical white plan — I return the *best of 200*. The best endpoint magnitude among 200
Rayleigh draws is roughly `σ·√(2 ln 200)` with `σ = √(0.95·T)`, which comes out around `17` cells at
horizon 30, `25` at horizon 60, `30` at horizon 90. So the argmin over a batch does reach meaningfully
farther than a typical draw, enough to occasionally catch a doorway when the goal is not too far — and
the receding-horizon loop then re-draws a fresh batch every control step and gets another 200 lottery
tickets. That is the whole mechanism by which a planner that learns nothing still solves some episodes:
best-of-batch reach, repeated across up to 200 env steps. It is worth naming the compute this spends to
buy that lottery: 200 rollouts per `plan()` call, and up to 200 control steps per episode, so as many as
`40,000` world-model rollouts to solve one 20-episode benchmark row per episode — the floor is not cheap
in absolute terms, it is cheap only in *machinery*. Every one of those 40,000 rollouts returns a cost it
never uses beyond a single argmin, which is the precise sense in which the floor wastes information: it
pays full price for the measurements and then throws all but one away each call.

But that `17`-cell figure is optimistic, because the argmin needs *two* things to line up at once, not
just magnitude. A sequence that reaches far in the wrong direction scores no better than one that barely
moves — it has to reach far *and* point toward the door. The endpoint bearing of a white walk is uniform
on the circle (the sampling is isotropic, it has no directional prior), so if the doorway subtends,
generously, a `±30°` window from the agent's position, only about a sixth of the 200 sequences —
roughly `33` of them — are even aimed usefully. The best-reaching of those `~33` correctly-aimed
sequences is `σ·√(2 ln 33) ≈ σ·2.65`, which at horizon 30 is about `14` cells, not `17`. So the joint
magnitude-and-bearing lottery cuts the effective reach down further, and it is a *lottery*, not a search:
the floor cannot steer the 200 draws toward the door, it can only hope enough of them happen to point
there and pick the longest. The reach ceiling above is exactly why I expect the floor to leave a large
fraction of episodes unsolved and to solve the ones it does slowly, wandering in by re-drawing the cloud
step after step until it stumbles close enough.

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
get trapped in a basin the way a hill-climber would, because it never climbs. Every batch is drawn from
the same fixed isotropic Gaussian, so the coverage of the two routes is identical on the hundredth
control step and on the first — no early score can bias later draws toward the wrong side of the door.
That even-handedness is the floor's only structural virtue, and it is worth stating so the later
methods' premature-collapse risk — committing to the wrong basin because an early elite set landed
there — reads as a real danger they take on in exchange for adaptivity. Make the contrast concrete: a
gradient descent or a hill-climber initialized near the wall on the wrong side of the door would follow
the local cost downhill *into* the wall — the cost decreases as the predicted endpoint nears the goal in
straight-line latent distance, and near the wall that straight line points through it — and having
committed, it has no mechanism to back out and route around. The floor cannot make that mistake, not
because it is clever but because it never commits: it draws the identical isotropic cloud every call, so
the through-the-wall basin and the around-the-door basin get equal coverage on every single control
step, and the argmin picks whichever *evaluated* endpoint actually scored best rather than whichever
direction looked locally downhill. The floor pays for its
route-blindness with reach; the adaptive methods will pay for their reach with route-commitment risk,
and this rung is the place that trade is defined.

Now the action constraint, made faithful. A standard normal has unbounded support, so some sampled
per-step actions will exceed the `max_norm = 2.45` cap the environment enforces. I do not want to hand
the cost a candidate the agent could never actually execute, because then the rollout scores a fantasy:
the model would predict latents for a step the agent physically cannot take, and the argmin could
select a sequence whose apparent low cost comes from an infeasible stride. The cheapest faithful fix is
to project each sampled action into the feasible ball before scoring: compute the norm of each 2-D
action, and if it exceeds `max_norm`, rescale that action down to norm `max_norm`, leaving shorter
actions untouched. Concretely, multiply each action by `(max_norm / norm).clamp(max=1.0)` (with a tiny
`eps` in the denominator so a zero action does not divide by zero) — actions already inside the ball get
a factor of exactly 1 and are unchanged, actions outside get pulled precisely to the boundary. This is
a radial projection, so it preserves each action's *direction* and only caps its *length*; it is the
minimal edit that makes every scored candidate executable, and it costs one elementwise operation. The
five percent of actions it touches are exactly the over-long tail, which is why it barely dents the
`E‖a‖²` I computed above.

How many candidates, and how many iterations? The harness hands the constructor `num_samples=200` and
`n_iters=20`, the shared defaults all the ladder's planners receive. The reference rung deliberately
ignores `n_iters` entirely — it does *one* pass, no refinement — and uses the `num_samples=200` budget
as a single batch of 200 candidate sequences. That is the whole point of the floor: it spends the same
per-batch sample budget the iterative methods will spend per iteration, but only once and without ever
folding the scores back into the sampling distribution. Concretely, this rung fires 200 world-model
rollouts per `plan()` call; an iterative method that runs 20 refit passes over the same 200-sample
batches will fire 4000 rollouts per call — twenty times the compute. I am *not* trying to match its
compute; I am trying to isolate its *mechanism*. If CEM, which reuses essentially this batch twenty
times while marching the distribution toward the elites, beats this rung, I want the win to be
attributable cleanly to the iteration and the refit — to the fact that it learns within a call — and
not to a larger first batch. Holding the first-batch size equal is what makes that attribution honest.

I should check that "sample and keep the best" is even the right *kind* of floor, by looking at what
else I could do with zero adaptation and no trusted gradient. A single sample is too weak to be
informative — it would report one random walk's cost and nothing about the distribution, and its
variance across episodes would swamp any signal I want to read. The opposite extreme, exhaustively
gridding the action-sequence space, is not available at all: the candidate lives in `[plan_length,
action_dim]`, which is `2T` real dimensions — `60` of them at horizon 30, `180` at horizon 90 — and even
a coarse three-points-per-axis grid is `3^60` points, so systematic enumeration is off the table by an
astronomical margin. That is precisely why the whole planner family is *sampling*-based rather than
grid-based: in tens of dimensions, drawing candidates from a distribution and scoring them is the only
thing that fits a finite budget. So random shooting with a batch of a couple hundred is not an arbitrary
choice of floor — it is the least-adaptive member of the only viable family, which is exactly what a
reference rung should be. And `200` is a defensible batch: large enough that the best-of-batch order
statistic I computed above is meaningfully above the median (the `√(2 ln 200)` factor is `3.26`), small
enough to match the per-iteration budget the adaptive methods will spend, so the comparison stays a
comparison of mechanism.

The selection rule is then trivial and forced: roll all 200 candidates through the model, read the 200
costs, and return the single sequence with the lowest cost — `argmin` over the batch. The whole
sequence is returned (the loop will take its first action), and there is no distribution to carry to
the next call, so nothing needs to persist across env steps and the `t0` flag — the marker of an
episode's first call, the natural place to reset carried state — is simply irrelevant here, because
there *is* no carried state. Every call re-draws from the identical fixed Gaussian. The plan length is
`min(plan_length, steps_left)` so the planner never proposes actions past the end of the episode. The
entire method runs under `torch.no_grad()` because nothing here needs a gradient — I am only ever
reading function values, which is the defining limitation of the whole zeroth-order family this rung
opens, and a limitation every rung after it will inherit and I will keep flagging until something is
willing to pay to lift it.

Let me be precise about what I expect this floor to do across the three horizon benchmarks, because the
*shape* of its failure is what the next rung must read. At the short horizon (30 steps) the maze is most
forgiving in one respect — the door is often within the `~17`-cell best-of-200 reach I computed, so
when the start and goal happen to share a corridor to the door the lottery catches it — but least
forgiving in another: each individual plan is only 30 steps, and the re-planning loop, capped at 200 env
steps, gets fewer independent corrections relative to how far the agent must travel per plan. I expect a
modest but clearly-nonzero success rate here, and a large residual distance on the episodes it misses,
because a white plan that fails to reach the door leaves the agent stranded on the wrong side. At the
longer horizons (60, 90) two effects fight. More env steps means more re-plans, more lottery tickets, so
more chances for the loop to stumble the agent through; but each individual plan is longer, so a white
sequence has to stay coherent over more timesteps to make real progress, and the Brownian cancellation I
quantified gets *worse* the longer the sequence — the reach-to-committed ratio went from `9.7×` to
`16.9×`. My honest expectation is that the extra re-plans win out enough that success rate rises
somewhat from horizon 30 to 60, but the reach handicap caps it well below what an adaptive method will
reach, at every horizon; the `mean_dist` on the longer horizons should be lower than at horizon 30
simply because more re-plans give more chances to close the last gap; and the `mean_steps_to_success`
should be large and *climbing* with the horizon — when random search does succeed it succeeds slowly,
because it wanders rather than commits. I will not pretend to know the exact success numbers; the
feedback table will tell me. What I am willing to commit to is the mechanism: the depressed success and
the large, horizon-climbing step count are two faces of the same Brownian reach deficit, and both are
the numbers the next rung is on the hook to improve.

And the obvious lever is already visible from the leak. Every one of the 200 rollouts this rung fires
returns a cost — a real measurement of where the low-cost sequences live — and the floor reads all 200
only to keep one and throw the other 199 measurements away, then re-draws the identical white cloud on
the next call. That discarded structure is exactly what an adaptive planner would exploit: stop drawing
from a frozen distribution and instead let the sampling distribution *move toward* the sequences that
scored well, iteration after iteration, so the search concentrates on the route through the door instead
of re-rolling the same white noise twenty times. That is precisely the move from this single-pass floor
to CEM. The full scaffold module — the literal `CustomPlanner` fill for this rung — is in the answer.
