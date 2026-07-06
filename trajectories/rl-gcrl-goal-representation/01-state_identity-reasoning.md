The representation is the whole point, but it bolts onto a downstream agent, and with no learned
representation that agent is the floor — so the place to start is the trivial fill: hand the GCIVL
learner the raw goal observation and let it sort out everything itself. The scaffold is already built
for this. The value `V(s, phi(g))`, the actor `pi(a | s, phi(g))`, and the target networks all consume
whatever `encode_goal` returns; if I make `encode_goal` the identity, they consume the raw goal, and
the agent is the standard "concatenate `[s, g]` and learn" recipe that almost all goal-conditioned RL
defaults to. There is no auxiliary representation loss to add, so `compute_rep_loss` returns zero and
the agent's objective is just its own IVL value loss plus the AWR actor loss. This is the baseline by
construction: the floor that any learned `phi(g)` has to beat.

Let me make that reduction exact, because it is what pins down the interpretation of every later rung.
Each training step sums three terms: the IVL value loss at expectile `0.9`, the AWR actor loss at
`alpha = 10`, and the module's `rep_loss`. With the identity fill the third term is a literal `0.0`
returned by `compute_rep_loss`, and — this is the load-bearing part — it carries an empty info dict and
no parameters, so its gradient with respect to everything is zero. The module owns nothing to
differentiate. So the total objective collapses to exactly the two fixed GCIVL losses over raw goals,
and the parameter set being trained is precisely the downstream value and actor and their targets;
nothing in the goal channel updates. That is what "floor" should mean concretely: not "a weak learned
representation" but "the same learner with the representation slot short-circuited," so any number this
rung produces is attributable entirely to GCIVL confronting raw coordinates. If a later `phi(g)` beats
it, the delta is the value of the representation and nothing else, because I have held the rest of the
pipeline byte-for-byte fixed.

There is exactly one wrinkle that the harness forces on me, and it is worth stating precisely because
it is the only non-obvious line in the whole fill. The downstream value and actor networks are
initialized once, up front, for a fixed feature width `rep_dim` (default 256). They do not adapt their
input dimension to whatever `encode_goal` happens to return; they were built expecting a `rep_dim`
vector. But the raw goal observation has width `obs_dim`, which is not 256 in general — in
antmaze-large the proprioceptive state is on the order of 29 dimensions, in cube-single it is a few
dozen, in pointmaze it is tiny, a handful of position and velocity terms. So a literal `return goals`
would feed an `obs_dim`-wide vector into networks expecting `rep_dim`, and the shapes would not match.
The honest identity here is therefore "raw goal, deterministically reshaped to `rep_dim` with no
learned parameters": if `obs_dim == rep_dim` return the goal as-is; if `obs_dim < rep_dim` pad the last
axis with zeros up to `rep_dim`; if `obs_dim > rep_dim` truncate to the first `rep_dim` coordinates.

Let me actually count what each branch does on the three environments, because the arithmetic tells me
which branch is real plumbing and which is dead code I am carrying defensively. All three observation
widths sit far below 256: antmaze around 29, pointmaze a handful, cube a few dozen. So on every
environment here the active branch is the zero-pad, and the truncation branch never fires — it is a
guard for a hypothetical wide-observation environment, not something these three exercise. The pad
widths are large: antmaze pads `256 - 29 = 227` zeros, so `227 / 256 = 88.7%` of the vector I hand
downstream is a constant; pointmaze pads on the order of `252` of `256`, over 98% constant; cube
somewhere in between, comfortably above 80% constant. Most of the "goal code" the downstream net sees
is, quite literally, zeros.

I want to be sure this padding is genuinely inert and not smuggling in a distortion, so let me trace a
first-layer preactivation by hand. Take a two-dimensional observation `[a, b]` padded to width four:
the code is `[a, b, 0, 0]`. A hidden unit computes `h = w1 a + w2 b + w3 * 0 + w4 * 0 + bias = w1 a +
w2 b + bias`. The weights `w3, w4` on the padded coordinates multiply zero, so they never affect the
output — and their gradient is `dL/dw = (dL/dh) * x = (dL/dh) * 0 = 0`, so they never move from their
initialization either. The padded coordinates are frozen and irrelevant simultaneously: they add no
information (they are constant), they cost no capacity (their weights are inert), and they cannot even
drift during training. So the pad/truncate really is plumbing to satisfy the fixed network width, not a
representation; padding with zeros adds nothing and truncation only ever drops coordinates, which is
the worst case I am trying to expose anyway. The point stands that this is still "raw state as goal":
no `phi`, no auxiliary loss, no learning in the goal channel.

Before I accept the bare pad/truncate I should ask whether there is a cheap, still-assumption-free
reshaping that would be less lossy, because if one existed the "floor" would be a slightly unfair
strawman. Three candidates are on the table. First, a fixed random Gaussian projection `R` of shape
`(obs_dim, 256)` mapping the raw goal into the fixed width: by Johnson–Lindenstrauss this preserves
pairwise Euclidean distances up to a small distortion, and unlike truncation it never discards a
coordinate outright. But it buys nothing for the floor's purpose and quietly changes what the floor
measures: it rotates and mixes the raw axes, so it is no longer "the raw goal handed through," and its
distances are still raw-coordinate distances — the very geometry that has nothing to do with
reachability — merely re-embedded. It also introduces a seed (which random matrix) that the identity is
supposed not to have. So a random projection cannot help on the thing that matters, reachability
geometry, while muddying the clean diagnostic role of the floor; I drop it. Second, a single learned
linear map with no auxiliary loss: but with `rep_loss = 0` there is no gradient into it except through
the fixed downstream losses, which the harness explicitly stops from reaching the module — so a learned
map here would sit frozen at its random init, i.e. exactly the random projection I just rejected, with
extra machinery. Third, per-coordinate standardization of the raw goal (subtract mean, divide by std):
this is a monotone rescaling of each axis and changes nothing about which axes exist or how proximity
relates to reachability; it is a preprocessing convenience, not a representation, and it would blur the
"raw" in "raw-state goal." All three roads lead back to the same place: any nontrivial improvement has
to come from a *learned, reachability-aware* map trained by a real `rep_loss`, and none of these cheap
reshapings is that. The bare zero-pad is the honest, minimal, assumption-free choice, and it is what
makes the floor a floor rather than a lightly disguised representation.

Two limit checks convince me the fill is correct rather than merely plausible. First, the degenerate
case `obs_dim == rep_dim`: then no branch pads or truncates, `encode_goal` returns `goals` unchanged, so
`phi` is the exact mathematical identity and `V(s, phi(g)) = V(s, g)` is literally the standard
`[s, g]`-concatenation GCIVL value — the recipe I claimed to be reducing to, recovered exactly with no
reshaping at all. The pad and truncate branches are only the width-matching that a non-256 observation
forces; they do not change the map's identity character, they extend it to the fixed slot. Second, the
parameter-tree check: `setup()` is `pass`, so the module registers no `flax` variables, and its slice
of the optimizer's parameter tree is empty. There is therefore nothing for the optimizer to step, which
is the same conclusion I reached from the loss side (`rep_loss = 0`, empty info) arrived at from the
architecture side — a consistency I like, because it means the "no learning in the goal channel" claim
holds whether I read it off the objective or off the parameter registry. The geometric future sampling
is the last piece worth placing: it draws trajgoal offsets with a geometric law, so the reachable
`d` seen in training concentrates on small-to-moderate step counts where the value curve
`-100(1 - 0.99^d)` is steep and informative, with the curgoal spike at `d = 0` and the randomgoal tail
out in the flat plateau. The identity has to fit that whole curve, densely near zero and sparsely far
out, over coordinates in which `d` is scrambled — which is why I expect it to do least badly on
short-horizon, low-exogenous pointmaze and worst where far goals and nuisance coordinates pile up.

I should be clear about why this is the *weakest* rung and not just *a* rung, because the diagnosis it
sets up is what drives step 2. When `phi(g) = g`, the downstream value and actor are handed a
coordinate system in which proximity has nothing to do with reachability. Two goal observations that
look almost identical in raw coordinates can be a long detour apart in the maze; two that look very
different can be one step apart. On top of that, every exogenous coordinate — nuisance pose terms,
sensor jitter, in the manipulation case parts of the scene the agent does not control — rides straight
into the value and actor as if it were task-relevant. So the networks have to do three jobs at once
from finite offline data: learn the environment's reachability geometry from scratch, learn control on
top of it, and learn to suppress the irrelevant coordinates. Nothing in the input pre-sorts any of
this for them. That is precisely the burden a learned goal representation is supposed to lift, and the
identity baseline lifts none of it. By construction it carries the most noise and the least structure
of anything on the ladder.

It helps to name concretely what surface the downstream value is being asked to fit over these raw
axes, because it quantifies how unforgiving the floor is. The reward is the OGBench shifted indicator
`r = I(s = g) - 1`, i.e. `-1` on every step until the goal absorbs, with `discount = 0.99`. So the
optimal goal-conditioned value at reaching distance `d` is `V*(s, g) = -(1 - 0.99^d) / (1 - 0.99) =
-100 (1 - 0.99^d)`. Reading off a few points: `d = 1` gives `-1.0`, `d = 10` gives `-9.56`, `d = 50`
gives `-39.5`, `d = 100` gives `-63.4`, and as `d -> inf` it saturates at `-100`. The surface is
roughly linear in `d` up close and then flattens toward `-100` far away — in antmaze-large, where a
distant goal can be a couple hundred environment steps off, most far goals sit in the saturated tail
where the value barely distinguishes them. The identity forces the downstream net to learn this
nonlinear-in-`d` surface as a function of raw coordinates in which `d` itself is a scrambled, high-
detour quantity, with no feature pre-computing "how far, in steps, is this goal." That is the concrete
shape of the burden.

The relabeling distribution the harness feeds in sharpens where that burden bites. Goals are drawn
`value_p_curgoal = 0.2`, `value_p_trajgoal = 0.5`, `value_p_randomgoal = 0.3`, with geometric future
sampling. So one in five targets is the current state itself, `d = 0`, `V* = 0` — the easy, degenerate
end of the surface. Half are future states along the same trajectory, a genuinely reachable and finite
`d` sampled geometrically, so they land in the informative, roughly-linear region of the value curve.
But three in ten are random goals drawn from elsewhere in the dataset, which are typically far or even
effectively unreachable within horizon — exactly the saturated `-100` tail where a `1%`-per-step
discount has already crushed the value into a nearly flat plateau and where raw-coordinate distance is
most divorced from true reaching cost. So 30% of every value batch stresses the precise region the
identity handles worst: far goals whose raw `(x, y, pose)` gap says almost nothing about how many steps
away they really are. That is not an edge case I can wave off; it is nearly a third of the value target
distribution by design.

The two fixed losses then amplify small value errors in ways worth spelling out, because it explains
why a merely-mediocre value surface translates into a poor policy rather than a slightly-worse one. The
IVL value loss uses expectile `0.9`: an upper expectile weights positive residuals nine times as
heavily as negative ones (`0.9 / 0.1 = 9`), so it deliberately chases the top of the in-sample target
distribution — an in-support max over the neighbors that actually occur in the data. Over scrambled raw
axes, "the neighbors that occur" is a noisy set whose best member need not be the truly closest-in-
steps state, so the max the expectile latches onto is itself corrupted by the coordinate scramble. The
AWR actor then reweights behavior actions by `exp(alpha * A)` with `alpha = 10`. That exponent is
steep: an advantage error of just `0.1` moves the action weight by `exp(10 * 0.1) = e ≈ 2.7`, and an
error of `0.5` moves it by `exp(5) ≈ 148`. So the actor is exponentially sensitive to exactly the
advantages the value is struggling to estimate over raw goals — a modest value distortion becomes a
large, wrong reweighting of which actions to imitate. The identity does not just hand the value a hard
surface; it hands a steep, max-seeking value loss and an exponential actor loss a coordinate system in
which their errors compound.

A tiny worked case makes the coordinate scramble concrete. Picture a maze with a thin wall separating
two rooms, and two points `A` and `B` on opposite sides of that wall, a hair apart in raw `(x, y)` —
say Euclidean distance `0.2` — but the only opening is at the far end, so the optimal path from `A` to
`B` is a long detour, `d* ≈ 80` steps. The true value is `V*(A, B) = -100(1 - 0.99^{80}) ≈ -55`.
Raw-coordinate proximity says these are essentially the same place; the reaching structure says they
are two-thirds of the way to the saturated tail apart. Under the identity the value net sees only the
`0.2` gap and must, from data alone, learn to output `-55` there while outputting something near `0`
for a genuinely adjacent state that happens to sit the same `0.2` away on the *same* side of the wall.
Same raw input geometry, opposite correct answers — that is the reachability-versus-proximity conflict
the floor cannot pre-resolve, and it is the single fact a learned `phi(g)` exists to fix: bend the
coordinate system so that across-wall pairs are pushed apart and along-corridor pairs pulled together
before the value ever sees them.

What do I expect this to do across the three environments, and why does the expectation differ by
environment? The split should fall on how much exogenous structure the raw goal carries and how much
the reachability geometry diverges from raw-coordinate proximity. In **pointmaze-large** the
observation is essentially the point's position, which is low-dimensional and already close to the
quantity that matters; raw goals are a relatively benign coordinate system there, so the identity
should be least disadvantaged and the gap to a learned representation smallest — even though, by the
padding count above, the code I hand downstream is over 98% constant, the few live coordinates are the
ones that matter. In **antmaze-large** the goal observation mixes the x-y target the policy actually
needs to reach with a pile of joint angles and velocities that are exogenous to *which goal* this is;
the raw goal is a noisy, high-detour coordinate system, so the identity should leave a lot of success
on the table — exactly the regime a reachability-aware `phi(g)` is built for, and the 88.7%-padded
code here still carries all 29 live-but-unsorted coordinates straight into the value. In **cube-single**
manipulation, the goal configuration is entangled with arm and scene detail, the reachability
structure is highly non-Euclidean in raw coordinates, and this is the setting where structured goal
codes have historically paid off the most; the identity should be weakest here relative to what a
learned representation can do.

Two more pieces of the fixed machinery go dormant under the identity, and noting them keeps the "floor
= plain GCIVL" claim airtight. The loop EMA-updates a `target_rep_critic` *if the module exposes one*
— the identity exposes no critic and no target, so that update is a no-op and the only EMA in play is
the downstream value's own target copy at `tau = 0.005`. That rate is worth a number: moving `0.5%`
toward the online value each step is an exponential average with an effective window of about
`1 / 0.005 = 200` steps, slow enough to stabilize the bootstrapped value target and entirely unrelated
to any representation learning, because there is none. And the first downstream value layer is
being handed mostly inert width: it concatenates `[s, phi(g)]`, so for antmaze its input is `29 + 256 =
285` wide into a `512`-unit hidden layer, `285 * 512 + 512 = 146,432` parameters — of which the `227`
constant-zero goal coordinates freeze `227 * 512 = 116,224` weights at initialization by the
gradient-is-zero argument above. Roughly `80%` of the goal-side first-layer weights never move. None of
this is a bug to fix at the floor; it is simply the price of routing a `29`-wide raw goal through a
`256`-wide slot with no representation. It confirms, from the parameter side, that the identity spends
its capacity on plumbing and its learning entirely on the two fixed GCIVL losses over raw axes — which
is exactly the floor I want.

So the falsifiable expectation for this floor is asymmetric across environments: roughly competitive on
pointmaze-large (where raw position is already a decent code), clearly leaving room on antmaze-large
(where the goal-relevant signal is buried in exogenous pose), and weakest on cube-single (where the raw
coordinate geometry is furthest from reachability). Whatever the precise levels, the diagnosis is
already pointed at the next step: this is a *coordinate-system* problem, not a learning-capacity
problem. The agent is competent — I showed its objective is exactly standard GCIVL with a zeroed goal
slot — it is being handed the wrong axes. The fix is to stop feeding raw goals and instead learn a
`phi(g)` whose geometry already encodes "how hard is this goal to reach," turning the empty
`compute_rep_loss` into a real auxiliary objective and the pad/truncate identity into a trained
encoder.

Stated against the one metric this task reports — success rate on antmaze-large-navigate-v0,
cube-single-play-v0, and pointmaze-large-navigate-v0 — the prediction is a specific ordering of the
*gap* a learned code will later have to close, not of the raw levels. I expect the identity's shortfall
to be smallest on pointmaze-large-navigate (few live coordinates, all task-relevant, short reversible
paths), intermediate on antmaze-large-navigate (the x-y target buried under joints and velocities, and
long detours that push far goals into the saturated `-100` tail), and largest on cube-single-play
(non-Euclidean reaching in an entangled arm-and-cube configuration). If instead a learned representation
were to help *uniformly* across the three, or most on pointmaze, that would falsify the coordinate-
system diagnosis and point back at raw learning capacity as the bottleneck — but the whole construction
of this floor, plain GCIVL over scrambled axes, says capacity is not the issue and the axes are. That
is exactly what step 2 takes up — the distilled module for this floor is in the answer.
