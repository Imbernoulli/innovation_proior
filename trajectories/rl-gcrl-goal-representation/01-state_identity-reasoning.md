The representation is the whole point, but it bolts onto a downstream agent, and with no learned
representation that agent is the floor — so the place to start is the trivial fill: hand the GCIVL
learner the raw goal observation and let it sort out everything itself. The harness is already built
for this. The value `V(s, phi(g))`, the actor `pi(a | s, phi(g))`, and the target networks all consume
whatever `encode_goal` returns; if I make `encode_goal` the identity, they consume the raw goal, and
the agent is the standard "concatenate `[s, g]` and learn" recipe that almost all goal-conditioned RL
defaults to. There is no auxiliary representation loss to add, so `compute_rep_loss` returns zero and
the agent's objective is just its own IVL value loss plus the AWR actor loss. This is the baseline by
construction: the floor that any learned `phi(g)` has to beat.

What "floor" means here fixes the interpretation of every later step. With the identity fill the module's `rep_loss` term is a literal `0.0` with an empty info
dict and no parameters — `setup()` is `pass`, so it registers nothing for the optimizer to step. The
total objective collapses to precisely the two fixed GCIVL losses over raw goals, and the only
parameters that move are the downstream value and actor and their targets. So this is not "a weak
learned representation" but "the same learner with the representation slot short-circuited," and any
number it produces is attributable entirely to GCIVL confronting raw coordinates. When a later `phi(g)`
beats it, the delta is the value of the representation and nothing else, because the rest of the
pipeline is held byte-for-byte fixed.

There is exactly one wrinkle the harness forces on me, and it is the only non-obvious line in the fill.
The downstream networks are initialized once for a fixed feature width `rep_dim` (default 256); they do
not adapt to whatever `encode_goal` returns. But the raw goal has width `obs_dim`, which is far below
256 on all three environments — antmaze-large is around 29, pointmaze a handful, cube a few dozen. So a
literal `return goals` would mismatch shapes. The honest identity is therefore "raw goal
deterministically reshaped to `rep_dim`, no learned parameters": return as-is if `obs_dim == rep_dim`,
zero-pad the last axis if smaller, truncate if larger. On these three the active branch is always the
zero-pad; the truncate branch is a guard that never fires. And the pad is genuinely inert — a hidden
unit's weight on a constant-zero coordinate multiplies zero in the forward pass and receives zero
gradient, so those weights never move from init. Zero-padding adds no information, costs no capacity,
and cannot drift; it is plumbing to satisfy the fixed width, not a representation.

No cheaper reshaping would make the floor less of a strawman without ceasing to be the floor. A random
Gaussian projection preserves pairwise distances (Johnson–Lindenstrauss) but they are still
raw-coordinate distances — the geometry divorced from reachability, merely rotated — and it smuggles in
a seed the identity should not have. A single learned linear map is worse: with `rep_loss = 0` and the
harness stopping the downstream loss from reaching the module, it would sit frozen at random init, the
random projection with extra machinery. Per-coordinate standardization is a monotone rescale that
changes no axis. Any nontrivial gain must come from a *learned, reachability-aware* map trained by a
real `rep_loss`, and none of these is that — which is what makes the bare zero-pad the honest floor.

Now the diagnosis this step is really here to set up. When `phi(g) = g`, the downstream value and actor
are handed a coordinate system where proximity has nothing to do with reachability: two goal
observations that look nearly identical in raw coordinates can be a long detour apart in the maze, and
every exogenous coordinate — nuisance pose terms, sensor jitter, uncontrolled scene detail in
manipulation — rides straight into the value and actor as if task-relevant. So from finite offline data
the networks must do three jobs at once: learn the environment's reachability geometry from scratch,
learn control on top of it, and learn to suppress the irrelevant coordinates. Nothing in the input
pre-sorts any of this. That is exactly the burden a learned goal representation lifts, and the identity
lifts none of it.

The value surface makes the burden quantitative. The reward is the OGBench shifted indicator
`r = I(s = g) - 1` with `discount = 0.99`, so the optimal goal-conditioned value at reaching distance
`d` is `V*(s, g) = -100 (1 - 0.99^d)`: `-1.0` at `d = 1`, `-9.56` at `d = 10`, `-39.5` at `d = 50`,
`-63.4` at `d = 100`, saturating at `-100`. Roughly linear up close, then crushed flat far out — in
antmaze-large a distant goal a couple hundred steps off sits in the saturated tail where the value
barely distinguishes goals. The relabeling distribution the harness feeds sharpens where that bites:
`value_p_curgoal = 0.2` (the current state, `d = 0`, the easy end), `value_p_trajgoal = 0.5` (future
states on the same trajectory, finite `d` in the informative region), and `value_p_randomgoal = 0.3`
(goals from elsewhere, typically far or unreachable — the saturated tail). So nearly a third of every
value batch stresses precisely the region the identity handles worst: far goals whose raw `(x, y, pose)`
gap says almost nothing about how many steps away they are. The two fixed losses then amplify small
errors. The IVL value loss uses expectile `0.9`, weighting positive residuals `9:1` to chase an
in-support max — but over scrambled raw axes "the neighbors that occur" is a noisy set whose best member
need not be the closest in steps, so the max is itself corrupted. The AWR actor reweights actions by
`exp(alpha * A)` with `alpha = 10`: an advantage error of `0.1` moves the weight by `e ≈ 2.7`, one of
`0.5` by `exp(5) ≈ 148`. So the actor is exponentially sensitive to exactly the advantages the value is
struggling to estimate over raw goals; modest value distortion becomes a large, wrong reweighting.

A tiny worked case makes the coordinate scramble concrete. Picture a thin wall separating two rooms,
with points `A` and `B` a hair apart in raw `(x, y)` — Euclidean distance `0.2` — but the only opening
is at the far end, so the optimal path is a long detour, `d* ≈ 80`, and `V*(A, B) = -100(1 - 0.99^{80})
≈ -55`. Under the identity the value net sees only the `0.2` gap and must, from data alone, output `-55`
there while outputting near `0` for a genuinely adjacent state the same `0.2` away on the *same* side of
the wall. Same raw input geometry, opposite correct answers — that is the reachability-versus-proximity
conflict the floor cannot pre-resolve, and it is the single fact a learned `phi(g)` exists to fix.

So the expectation splits by how much exogenous structure the raw goal carries and how far reachability
diverges from raw proximity. In **pointmaze-large** the observation is essentially the point's position,
already close to what matters, so the identity is least disadvantaged and the gap to a learned code
smallest. In **antmaze-large** the goal-relevant x-y target is buried under joint angles and velocities
exogenous to *which* goal this is, and long detours push far goals into the saturated tail — a lot of
success left on the table. In **cube-single** the goal configuration is entangled with arm and scene
detail and the reaching structure is most non-Euclidean, so the identity should be weakest relative to
what a learned representation can do. The one piece of fixed machinery that goes dormant here confirms
the reading: the loop only EMA-updates `target_rep_critic` if the module exposes one, and the identity
exposes none, so the sole learning is the two GCIVL losses over raw axes.

Against the one metric this task reports — success rate on the three environments — the prediction is
an ordering of the *gap* a learned code will later have to close: smallest on pointmaze-large,
intermediate on antmaze-large, largest on cube-single. If a learned representation instead helped
uniformly, or most on pointmaze, that would falsify the coordinate-system diagnosis and point back at
raw learning capacity — but plain GCIVL over scrambled axes says the axes are the issue, not capacity.
Step 2 turns the empty `compute_rep_loss` into a real auxiliary objective and the pad/truncate identity
into a learned encoder whose geometry encodes how hard a goal is to reach.
