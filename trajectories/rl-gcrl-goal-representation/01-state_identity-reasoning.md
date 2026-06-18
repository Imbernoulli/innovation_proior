The representation is the whole point, but it bolts onto a downstream agent, and with no learned
representation that agent is the floor — so the place to start is the trivial fill: hand the GCIVL
learner the raw goal observation and let it sort out everything itself. The scaffold is already built
for this. The value `V(s, phi(g))`, the actor `pi(a | s, phi(g))`, and the target networks all consume
whatever `encode_goal` returns; if I make `encode_goal` the identity, they consume the raw goal, and
the agent is the standard "concatenate `[s, g]` and learn" recipe that almost all goal-conditioned RL
defaults to. There is no auxiliary representation loss to add, so `compute_rep_loss` returns zero and
the agent's objective is just its own IVL value loss plus the AWR actor loss. This is the baseline by
construction: the floor that any learned `phi(g)` has to beat.

There is exactly one wrinkle that the harness forces on me, and it is worth stating precisely because
it is the only non-obvious line in the whole fill. The downstream value and actor networks are
initialized once, up front, for a fixed feature width `rep_dim` (default 256). They do not adapt their
input dimension to whatever `encode_goal` happens to return; they were built expecting a `rep_dim`
vector. But the raw goal observation has width `obs_dim`, which is not 256 in general — in
antmaze-large the proprioceptive state is on the order of 29 dimensions, in cube-single it is larger,
in pointmaze it is tiny. So a literal `return goals` would feed an `obs_dim`-wide vector into networks
expecting `rep_dim`, and the shapes would not match. The honest identity here is therefore "raw goal,
deterministically reshaped to `rep_dim` with no learned parameters": if `obs_dim == rep_dim` return the
goal as-is; if `obs_dim < rep_dim` pad the last axis with zeros up to `rep_dim`; if `obs_dim > rep_dim`
truncate to the first `rep_dim` coordinates. Padding with zeros adds no information and no parameters —
the appended coordinates are constant, so the downstream net simply learns to ignore them — and
truncation only ever drops coordinates, which is the worst case I am trying to expose anyway. The point
is that this is still "raw state as goal": no `phi`, no auxiliary loss, no learning in the goal channel.
The pad/truncate is plumbing to satisfy the fixed network width, not a representation.

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

What do I expect this to do across the three environments, and why does the expectation differ by
environment? The split should fall on how much exogenous structure the raw goal carries and how much
the reachability geometry diverges from raw-coordinate proximity. In **pointmaze-large** the
observation is essentially the point's position, which is low-dimensional and already close to the
quantity that matters; raw goals are a relatively benign coordinate system there, so the identity
should be least disadvantaged and the gap to a learned representation smallest. In **antmaze-large**
the goal observation mixes the x-y target the policy actually needs to reach with a pile of joint
angles and velocities that are exogenous to *which goal* this is; the raw goal is a noisy, high-detour
coordinate system, so the identity should leave a lot of success on the table — exactly the regime a
reachability-aware `phi(g)` is built for. In **cube-single** manipulation, the goal configuration is
entangled with arm and scene detail, the reachability structure is highly non-Euclidean in raw
coordinates, and this is the setting where structured goal codes have historically paid off the most;
the identity should be weakest here relative to what a learned representation can do.

So the falsifiable expectation for this floor is asymmetric across environments: roughly competitive on
pointmaze-large (where raw position is already a decent code), clearly leaving room on antmaze-large
(where the goal-relevant signal is buried in exogenous pose), and weakest on cube-single (where the raw
coordinate geometry is furthest from reachability). Whatever the precise levels, the diagnosis is
already pointed at the next step: this is a *coordinate-system* problem, not a learning-capacity
problem. The agent is competent; it is being handed the wrong axes. The fix is to stop feeding raw
goals and instead learn a `phi(g)` whose geometry already encodes "how hard is this goal to reach,"
turning the empty `compute_rep_loss` into a real auxiliary objective and the pad/truncate identity into
a trained encoder. That is exactly what step 2 takes up — the distilled module for this floor is in the
answer.
