The identity floor told me exactly what I expected. Raw goals are roughly competitive only where the
raw coordinate is already close to the quantity that matters — the point's position in pointmaze-large
— and they leave the most on the table where reachability geometry diverges most from raw proximity:
cube-single manipulation, with antmaze-large in between, its goal-relevant x-y target buried under joint
angles and velocities the policy must ignore. That is not a learning-capacity failure. The GCIVL agent
is competent; it is being handed the wrong axes and forced to reconstruct reachability structure from
scratch while also learning control and suppressing exogenous coordinates. So the fix is to stop feeding
raw goals and learn a map `phi(g)` whose *geometry already encodes reachability*, slotting it into the
same `encode_goal` hook so the value and actor see `phi(g)` everywhere they used to see `g`. The empty
`compute_rep_loss` becomes a real auxiliary objective; everything downstream stays fixed. I should be
disciplined about the evidence, though: this task carries no leaderboard and no per-seed numbers, so
what I have is the qualitative ordering, not a gap in success-rate points — enough to justify the
direction of the fix, not enough to invent precision.

Before committing I should rule out the other reachability-aware routes on concrete grounds. Regressing
`phi` against temporal distance directly fails offline: I only see the behavior policy's gaps, which are
upper bounds on `d*`, not `d*` itself, and a plain regression has no in-sample max to recover the
optimal shortcut. A stochastic information bottleneck compresses toward a prior, not toward reachability
— its objective has no term that knows about steps-to-goal, so it could strip the x-y target as readily
as the noise. A behavioral contrastive code is closer, but its geometry is an inner-product logit tuned
for classification, not a calibrated distance the downstream value can reason over. The parsimonious
move exploits an identity I half-see already: reaching cost *is* a value function, and value functions
are exactly what offline RL knows how to fit with an in-sample max. So I parameterize a value as a
distance and let value learning inject `d*`, starting from the strongest structural prior available —
that reaching cost is a metric — and spending complexity only if the environments prove it too rigid.

So what should "close in representation space" mean? The only closeness that matters for goal reaching
is temporal: the number of steps an optimal policy needs. Write `d*(s, g)` for that. I want a `phi` such
that ordinary Euclidean distance between `phi(s)` and `phi(g)` *is* that temporal distance — but I have
no `d*` labels, only trajectories. The hinge is a classical identity: with reward `-1` until `s = g` and
an absorbing goal, the return is the negative count of steps to reach `g`, so `V*(s, g) = -d*(s, g)`.
Distance-to-goal and goal-conditioned value are the same object, and value functions I *can* learn
offline. So instead of regressing `d*` I learn a value and force its functional form to be a distance
between two embeddings,

`V(s, g) = -||phi(s) - phi(g)||`,

train `V` to satisfy the goal Bellman equations, and at convergence `||phi(s) - phi(g)|| ≈ d*(s, g)` —
the isometry I wanted, with no distance label ever needed.

This parameterization buys structure before any training. It is symmetric in its two arguments, so a
single shared encoder `phi` suffices for both states and goals. It gives `V(s, s) = 0` and `V <= 0` for
free, two facts the value would otherwise have to discover. And it is a genuine metric by construction:
`||phi(s) - phi(g)||` obeys the triangle inequality for any `phi`, so shortcuts through a waypoint always
compose — exactly the right inductive bias in a maze, where reachability really does compose along paths.
The goal code handed downstream is simply `phi(g)`, which directly answers the floor's failure: where
the identity dumped joint angles and texture into the value, `phi` is trained so only the
reachability-relevant content survives into the distance.

But temporal distance is *not* symmetric — climbing a cliff costs more than descending it, a one-way
passage is one step one way and infinity the other — so `d*` is a quasimetric, and `||phi(s) - phi(g)||`
is rigidly symmetric. Am I dead? Not quite: I never drive the policy with this parameterized `V`. I want
`phi` only as a coordinate system for the downstream GCIVL agent, which keeps its own value and actor.
So the question is not "is the symmetric metric exactly `d*`" but "is its best symmetric approximation a
good enough coordinate system." There is even a deeper obstruction independent of asymmetry: not every
metric embeds isometrically in `l2`. The 4-cycle `C4` is the witness — four states on a loop, adjacent
pairs at distance `1`, opposite pairs at `2`; making both opposite pairs `2` apart forces all four
points colinear, which destroys the unit edges, so no Euclidean placement satisfies all six constraints.
So I stop hoping for exact isometry and reframe the target as the best *approximate* symmetric Hilbert
embedding of the MDP's temporal structure — honest, and exactly what the value objective optimizes.
This matters for the environments at hand: antmaze and pointmaze are largely reversible, so a symmetric
metric should fit; cube manipulation has irreversible structure where the symmetry starts to bite, a
tension I return to at the close.

Why `l2` specifically? Because it is induced by an inner product, so the space is a Hilbert space
carrying directions and projections, not just distances (`l1`/`l∞` give metrics but no consistent inner
product). I only strictly need a metric for reaching distance, but the inner product is structure I bank
for later: a direction `phi(g) - phi(s)` becomes an actionable control signal. Fixing `R^D` with `l2`
costs nothing in the objective, so I take the stronger geometry.

Now the real work: training `V(s, g) = -||phi(s) - phi(g)||` toward `V* = -d*` from a fixed dataset of
suboptimal trajectories. The optimal Bellman backup is a `max` over reachable next states, and offline I
cannot take that max without querying transitions absent from the data — the classic route to
catastrophic overestimation. The tool for an in-sample max is expectile regression (the IQL move): the
asymmetric squared error `L_kappa(u) = |kappa - 1(u < 0)| u^2` with `kappa > 1/2` weights positive
residuals more, so fitting a scalar to a distribution of targets pulls toward the top of the support,
approaching the max in the limit. Regressing `V` toward TD targets over the dataset's own transitions
with an upper expectile pulls it toward the best backup among transitions that actually occur — an
optimal-style max that never invents a transition. The behavior policy can be terrible on average; as
long as the data occasionally contains the good next step, the upper expectile latches onto it.

Here I have to respect *this* harness, which does not give me a single distance-parameterized head to
read `d*` off. The fixed agent owns its own IVL value and computes it so it does *not* backprop into
`phi`; only `rep_loss` trains the representation. So `compute_rep_loss` runs a self-contained
value-learning loop whose sole purpose is to shape `phi`, using a private twin `rep_critic` with an EMA
`target_rep_critic` the loop maintains. The loss is two parts. The **representation value** is the
Hilbert distance, `v = -sqrt(max(||phi_s - phi_g||^2, eps))`, fit by expectile regression toward the
target critic: `adv = q_t - v` with `q_t = min(q1_t, q2_t)` from the frozen `target_rep_critic`, loss
`|kappa - 1(adv < 0)| adv^2` at `kappa = rep_expectile`. The **critic** is an ordinary TD fit:
`td = r + gamma * mask * v(next_s, g)` (stop-gradient), both online heads regressed with squared error.
The pieces interlock: the critic learns a bootstrapped reaching value from data, the expectile loss
drags the *distance* up toward an in-support max of that estimate, and because the distance is
`-||phi_s - phi_g||`, that drag injects temporal structure into `phi`. Keeping the representation value
strictly separate from the downstream control value is the whole reason the harness exposes a private
`rep_critic` — the distance head shapes the code, the GCIVL value extracts the policy.

Two load-bearing details. First the discount: `d*` is an undiscounted step count, but I carry `gamma`
into the backup because the undiscounted value grows with horizon and is nasty to fit with bootstrapped
TD; `gamma` keeps the backup contractive at the cost that I converge to a *discounted* approximation.
With the shifted indicator reward that fixed point is `V*(s, g) = -100(1 - 0.99^d)`, monotone in `d` but
strongly compressive far out, so the Hilbert distance resolves near and mid-range goals well and squashes
far ones together — acceptable here, where control mostly needs the local gradient of reaching cost.
Second the square-root floor: `||phi_s - phi_g||` has derivative `1/(2 sqrt(.))`, which blows up as the
squared distance goes to zero (at init, or `s` near `g`), so I compute `sqrt(max(squared_dist, 1e-6))`,
capping the norm-gradient factor at `1/(2·1e-3) = 500` instead of `+inf`. That single `max` is the
difference between training and a first-batch NaN. Its one honest side effect is that `V(s, s) =
-1e-3` rather than an exact `0`, an offset far below any reward scale here. For `kappa` I use the harness
`rep_expectile = 0.7`, a `7:3` upper expectile — an in-sample max softened just enough to survive
suboptimal offline data rather than chasing the single luckiest transition. The `encode_goal` hook
returns `phi(goals)` averaged over the 2-member ensemble, and the twin critic's `min` is the standard
clipped-double-Q guard against target overestimation under bootstrapping.

Environment by environment, then. The mechanism — replace raw axes with axes where distance is
reachability — should help most where the floor hurt most. On **antmaze-large** the Hilbert code is
trained to keep only what moves the reaching distance, so it should collapse the nuisance pose and lift
success clearly above the identity. On **cube-single** the metric code should help too, but this is
where I expect the symmetry assumption to leave the most on the table, because manipulation has
irreversible transitions a symmetric `||.||` cannot represent. On **pointmaze-large**, where raw
position was already a decent code, I expect Hilbert to tie or only modestly beat the floor. That is the
ceiling I am accepting: I bought the shared encoder, the free `V(s,s)=0`, and the triangle inequality by
assuming reaching cost is a symmetric metric, and where that assumption is false the code cannot follow.
If manipulation stays stubbornly below navigation, the symmetry assumption is the first place I would
look next.
