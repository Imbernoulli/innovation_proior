The identity floor told me exactly what I expected it to, and the pattern is the whole motivation for
this step. Raw goals are roughly competitive only where the raw coordinate is already close to the
quantity that matters — the point's position in pointmaze-large — and they leave the most on the table
exactly where the reachability geometry diverges most from raw-coordinate proximity: cube-single
manipulation, with antmaze-large in between, its goal-relevant x-y target buried under joint angles and
velocities the policy must ignore. That is not a learning-capacity failure. The GCIVL agent is
competent; it is being handed the wrong axes, and it has to reconstruct the environment's reachability
structure from scratch while also learning control and suppressing exogenous coordinates. So the fix is
to stop feeding raw goals and learn a map `phi(g)` whose *geometry already encodes reachability*, then
slot it into the same `encode_goal` hook so the value and actor see `phi(g)` everywhere they used to see
`g`. The empty `compute_rep_loss` from the floor becomes a real auxiliary objective that trains `phi`,
and the pad/truncate identity becomes a trained encoder. Everything downstream stays fixed.

So what should "close in representation space" mean? The only notion of closeness that matters for goal
reaching is temporal: how many steps an optimal policy needs to get from one state to another. Write
`d*(s, g)` for that optimal temporal distance. I would love a `phi` such that ordinary Euclidean
distance between `phi(s)` and `phi(g)` *is* that temporal distance. But that immediately raises the
question of how I would ever learn such a `phi` from offline data — I have no `d*` labels, only
trajectories. The hinge is a classical identity. Set up the goal-reaching reward as `r(s, g) = -1`
unless `s = g`, terminating at the goal; then the return is the negative count of steps until reaching
`g`, so the optimal value is exactly the negative optimal temporal distance, `V*(s, g) = -d*(s, g)`.
"Distance to goal" and "goal-conditioned value function" are the same object — and value functions I
*can* learn offline. So instead of regressing `d*` directly, I learn a goal-conditioned value `V(s, g)`
and *force its functional form to be a distance between two embeddings*:

`V(s, g) = -||phi(s) - phi(g)||`.

Train `V` to satisfy the goal-conditioned Bellman equations and, at convergence,
`||phi(s) - phi(g)|| ≈ -V(s, g) ≈ d*(s, g)` — exactly the isometry I wanted, with the value-learning
machinery doing the work of injecting `d*` into `phi` and no distance label ever needed.

Notice what this parameterization buys me before any training. It is *symmetric in its two arguments*,
so a single shared encoder `phi` suffices for both states and goals — I encode a state and a goal with
the same map and take the distance between the codes. That is economical and natural: I am modeling a
distance, and a distance takes two points from the same space. It also gives `V(s, s) = 0` for free
(being already at the goal costs nothing) and `V <= 0` always (temporal distance is nonnegative), two
facts the value function would otherwise have to discover. And the goal code I hand downstream is simply
`phi(g)` — encode the raw goal with `phi`, and that vector replaces `g` in `encode_goal`. This directly
answers the floor's failure: where the identity dumped joint angles and texture straight into the value,
`phi` is trained so that only the reachability-relevant content survives into the distance.

I have to confront the symmetry I just praised, because temporal distance is *not* symmetric. Climbing
a cliff costs more than descending it; a one-way passage is one step one way and infinity the other. So
`d*(s, g) != d*(g, s)` in general — it is a quasimetric, with the triangle inequality and `d*(s,s)=0`
but not symmetry — while `||phi(s) - phi(g)||` is rigidly symmetric. Am I dead? Think about what I will
actually *use* `phi` for. I am not going to drive the policy directly with this possibly-wrong
parameterized `V`. I only want `phi` as a *representation* — a coordinate system handed to the
downstream GCIVL agent, which keeps its own separate value and actor. So the question is not "is the
symmetric metric exactly `d*`" but "is the best symmetric approximation of `d*` a good enough coordinate
system." There is even a deeper obstruction: not every metric, even a symmetric one, embeds
isometrically into a Euclidean space (there are finite metric spaces with no isometric `l2` embedding).
So I stop hoping for an exact isometry and reframe the target as the best *approximate* symmetric
Hilbert embedding of the MDP's temporal structure. That is honest, and it is exactly what the
value-learning objective optimizes. I keep the symmetric form, accept it is an approximation in
asymmetric environments, and let the downstream success rate adjudicate whether the approximation is
useful. This matters for the environments at hand: antmaze and pointmaze navigation are largely
reversible, so a symmetric metric should fit them well; cube manipulation has more irreversible
structure, where the symmetry assumption starts to bite — a tension I will return to at the close.

Why insist the embedding space be Euclidean with the `l2` norm specifically? Because `l2` is induced by
an inner product, so the space is a Hilbert space — it carries an algebra, not just distances:
directions, projections, midpoints. `l1` and `l∞` give metric spaces but no consistent inner product. I
only strictly need a metric to talk about reaching distance, so the inner product is structure I am
banking for later: a direction `phi(g) - phi(s)` in this geometry is itself an actionable control
signal, so the downstream agent can be steered by elementary algebra on goal codes rather than
re-learning geometry on frozen features. Since fixing the space to `R^D` with `l2` costs nothing in the
representation objective, I take the stronger geometry.

Now the real work: how to train `V(s, g) = -||phi(s) - phi(g)||` toward `V* = -d*` from a fixed offline
dataset of suboptimal trajectories. The optimal Bellman backup is a `max` over reachable next states,
and offline I cannot take that max without evaluating transitions absent from the data — querying
out-of-distribution actions in an offline value is the classic route to catastrophic overestimation.
The tool for an in-sample max is expectile regression, the move behind IQL. The expectile loss is an
asymmetric squared error, `L_kappa(u) = |kappa - 1(u < 0)| u^2` with `kappa in (0, 1)`. For
`kappa > 1/2` it weights positive residuals more than negative ones, so fitting a scalar to a
distribution of targets pulls the fit toward the top of the support; in the limit it approaches the
maximum. Regress `V(s, g)` toward TD targets computed over the dataset's own transitions with an upper
expectile and `V` is pulled toward the best achievable backup among transitions that actually occur — an
optimal-style max that never invents a transition. The behavior policy can be terrible on average; as
long as the dataset occasionally contains the good next step out of `s`, the upper expectile latches
onto it. That is what makes recovering *optimal* `d*` from suboptimal offline data possible.

Here is where I have to be precise about *this task's harness*, because it does not give me a single
distance-parameterized value head and let me read `d*` off it directly. The fixed agent already owns its
own downstream IVL value, and — critically — it computes that value loss so it does *not* backpropagate
into `phi`. Only the module's `rep_loss` trains the representation. So inside `compute_rep_loss` I must
run a self-contained value-learning loop whose *only* purpose is to shape `phi`. The harness hands me a
clean way to do it: a separate twin MLP critic, `rep_critic`, with an EMA target copy
`target_rep_critic` that the loop maintains for me. The design that fits this contract is a two-part
loss. First, the **representation value** is the Hilbert distance itself, `v = -sqrt(max(||phi_s -
phi_g||^2, eps))`, and I fit it by expectile regression toward the target critic: the advantage is
`adv = q_t - v` where `q_t = min(q1_t, q2_t)` from the frozen `target_rep_critic`, and the loss is
`|kappa - 1(adv < 0)| adv^2` with `kappa = rep_expectile`. This is the in-sample-max pull on the
distance value, keyed by the target critic so it is stable under bootstrapping. Second, the **critic**
is an ordinary TD fit: build the TD target from the Hilbert value at the next state,
`td = r + gamma * mask * v(next_s, g)` (stop-gradient), and regress both online critic heads to it with
plain squared error. The total `rep_loss` is the sum, and `encode_goal` returns the mean of the shared
`phi` ensemble. The two pieces interlock: the critic learns a bootstrapped estimate of the reaching
value from data, the expectile loss drags the *distance* parameterization up toward an in-support max of
that estimate, and because the distance is `-||phi_s - phi_g||`, that drag is exactly what injects
temporal structure into `phi`. I keep the representation value strictly separate from the downstream
control value, which is the whole reason the harness exposes a private `rep_critic` — the distance head
shapes the code, the GCIVL value extracts the policy.

A few load-bearing details. The discount: `d*` is undiscounted (a raw step count), yet I carry `gamma`
into the backup, because the undiscounted goal value grows with horizon and is numerically nasty to fit
with bootstrapped TD; `gamma` keeps the backup contractive and the magnitudes bounded, at the cost that
I converge to a *discounted* approximation of the temporal structure. Combined with the symmetry
approximation and the embeddability obstruction, the honest description of the fixed point is "the best
discounted symmetric Hilbert approximation of the MDP's temporal distances." The square-root floor is
not optional: `||phi_s - phi_g|| = sqrt(sum (phi_s - phi_g)^2)` has derivative `1/(2 sqrt(.))`, which
blows up as the squared distance goes to zero — at init, or for `s` near `g` — so I compute
`sqrt(max(squared_dist, 1e-6))`. That bounds the gradient near zero and removes the NaN, while being
negligible where the distance is healthy. It is a small constant but it is the difference between
training and diverging on the first batch. And I use the harness `rep_expectile` (0.7) for `kappa` — an
upper expectile, aggressive enough to chase the good in-support backup without becoming brittle and
chasing the single luckiest transition.

Let me close on what this should do against the identity floor, environment by environment, because the
expectations are falsifiable. The mechanism — replace raw goal axes with axes where distance is
reachability — should help *most* exactly where the floor hurt most. On **antmaze-large**, the floor
buried the goal-relevant x-y target in exogenous pose; the Hilbert code is trained to keep only what
moves the reaching distance, so `phi(g)` should collapse the nuisance coordinates and lift success
clearly above the identity. On **cube-single**, the floor was weakest because raw geometry is furthest
from reachability; the metric code should help here too — but this is also where I expect the symmetry
assumption to leave the most on the table, because manipulation has irreversible transitions the
symmetric `||.||` cannot represent. On **pointmaze-large**, where raw position was already a decent
code, I expect the Hilbert representation to be roughly level with the floor or only modestly above it —
there is little reachability structure to recover that the raw coordinate did not already supply. So the
falsifiable claim is: Hilbert beats identity decisively on antmaze and cube, ties or barely improves on
pointmaze, and its residual weakness — relative to whatever a *directed*, non-symmetric aggregator could
do — should show up precisely on the manipulation environment, where reachability is most asymmetric.
That residual is exactly the gap the next step attacks by dropping the symmetric metric for an
inner-product value. The distilled module — shared `phi`, the floored Hilbert distance, the
expectile-plus-TD `rep_loss` — is in the answer.
