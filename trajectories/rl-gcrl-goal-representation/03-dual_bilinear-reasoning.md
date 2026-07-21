The Hilbert rep did what I asked and stopped exactly where I predicted. Replacing raw goal axes with
axes where Euclidean distance is reachability lifted success clearly above the identity floor on
antmaze-large and cube-single, barely moved on pointmaze-large (raw position was already serviceable),
and its residual weakness showed up in precisely the place I flagged: the symmetric metric leaves the
most on the table on manipulation, and trails on antmaze, while only tying on pointmaze. That is
structural, not a tuning artifact, and naming the structure drives this step. The Hilbert value
`V(s, g) = -||phi(s) - phi(g)||` is rigidly symmetric, so it must assume that reaching `g` from `s`
costs the same as reaching `s` from `g`. In reversible navigation that is roughly true; in manipulation
it is false — pushing a cube off a ledge is easy, lifting it back is hard, many transitions are one-way.
Forcing a symmetric code onto asymmetric reaching costs expressiveness exactly where the task is
hardest. (As before, this task reports no per-seed numbers, so I carry a qualitative ordering,
not a measured gap — enough to set the direction of the fix.)

There is a second, subtler limitation that sharpens the fix. Even setting asymmetry aside, a
shared-embedding metric `||phi(s) - phi(g)||` is not a universal approximator of two-argument functions
— there are finite metric spaces with no isometric `l2` embedding at all — and `d*(s, g)` is in general
neither symmetric nor metric-shaped. So the Hilbert head is doubly constrained: symmetric *and* limited
in the pairwise surfaces it can represent. To carry the full reachability relation I need an aggregator
that is (a) directed — different maps for the state and goal sides — and (b) expressive enough to
approximate an arbitrary continuous two-argument value, while keeping everything else: the same offline
value-learning machinery, the same harness contract, the same `encode_goal` hook.

Let me build the right object from the reaching problem. With the shifted indicator reward the optimal
value is a monotone transform of optimal reaching time, so the value *is* the reaching structure in
log-discount coordinates. What is a goal to a controller? Not just "far from here" — operationally, a
goal is the thing every state has some reaching relation to. So the ideal representation of `g` is the
whole function of incoming relations, `phi^vee(g) = (s |-> d*(s, g))`: identify a goal by *all* its
temporal distances from the state space. This is the familiar slogan that an object is determined by its
relations to every other object, and it is only useful if the code meets two control requirements —
sufficiency and noise invariance. Sufficiency: handed only `f = phi^vee(g)`, the one-step greedy action
needs only successor values, `argmax_a E_{s'}[gamma^{f(s')}] = argmax_a E_{s'}[V*(s', g)] = argmax_a
Q*(s, a, g)`, so the functional throws away nothing the optimal policy needs and never touches the raw
goal. Noise invariance: in a block-structured observation model where the reward is on the latent,
`r^ell(s, g) = I(p(s) = p(g))`, two goal observations with the same latent induce identical returns on
every trajectory, so `phi^vee(g_1) = phi^vee(g_2)` and exogenous observation noise disappears. Made
concrete on cube: two goals that agree on the cube's target pose but differ in an uncontrolled
coordinate share the same latent-reward indicator at every state, so their optimal values coincide
pointwise and the ideal functional collapses them — the gradient never rewards distinguishing them.
This is the floor's failure (joint angles, texture riding into the value) addressed at the level of the
target rather than patched onto the controller after the fact.

The functional is the right ideal, but I cannot store an arbitrary function per goal and I do not know
`d*`. The sufficiency argument tells me how to finitize it: the functional is always *paired with a
state and evaluated*, `f(s') = d*(s', g)`. So model the two-argument surface through two embeddings,
`V(s, g) ~= f(psi(s), phi(g))`, and export the goal-side vector as the code. The only open choice is the
aggregator `f`, and there are three directed candidates. A monolithic MLP over the concatenation is
universal but couples `s` and `g` through joint hidden layers, so there is no clean goal-side vector
whose relation to the state code *is* the value — it breaks the requirement that a goal be a vector paired
with the state, and kills the "a direction in code space is control" algebra. An explicit asymmetric
quasimetric like `sum_k ReLU(phi_k(g) - psi_k(s))` keeps the triangle inequality while dropping symmetry,
but it re-imposes metric axioms the sufficiency and noise-invariance arguments never asked for, and is a
fussier surface to fit for no gain in the representation role. The simplest aggregator that is directed,
universal, *and* keeps the value a plain function of two vector codes is the bilinear inner product:

`f(psi(s), phi(g)) = psi(s)^T phi(g)`.

It is directed — `psi` and `phi` are different maps, so `psi(s)^T phi(g)` need not equal
`psi(g)^T phi(s)`, directly attacking the manipulation gap. And it is universal — with wide enough
learned feature maps, sums of separable products approximate any continuous two-variable function on a
compact domain. I can make the capacity relation sharp: the negative squared distance expands as
`-||a - b||^2 = 2 a^T b - ||a||^2 - ||b||^2`, which is a single inner product in a lifted space (set
`psi(s) = [2 phi(s), -||phi(s)||^2, 1]`, `phi_goal(g) = [phi(g), 1, -||phi(g)||^2]`). So with three extra
coordinates the bilinear head reproduces the negative squared Hilbert value; the bilinear family strictly
contains the metric family (up to the monotone `x -> -sqrt(-x)`, irrelevant to the greedy argmax). The
inner product can be no *less* expressive than the metric, so the only thing that could go wrong is
optimization, not representational reach.

In the practical head I scale by the width, `V(s, g) = psi(s)^T phi(g) / sqrt(d)`. At init the codes are
roughly zero-mean, order-one per coordinate, so the dot product of `d = 256` near-independent terms has
standard deviation `~sqrt(256) = 16` — which would swamp the reward of magnitude `1` inside the TD target
and, worse, change with the width, silently forcing a re-tune whenever `rep_dim` moves. Dividing by
`sqrt(d)` pulls the score to order one and makes it width-invariant, the same normalization attention
uses. The inner product also buys numerical smoothness for free: there is no `sqrt(||.||)`, so no
`1/(2 sqrt(.))` singularity and no `1e-6` floor — the gradient of `psi^T phi / sqrt(d)` w.r.t. `phi(g)`
is just `psi(s)/sqrt(d)`, bounded wherever LayerNorm keeps the codes bounded. The honest cost is the
metric's freebies: `V(s, s)` is no longer pinned to `0`, `V <= 0` is not guaranteed, and the triangle
inequality no longer holds by construction. Those were real inductive biases; I relearn them from data in
exchange for direction and universality — a good trade where reaching is asymmetric, roughly neutral
where it was already metric-shaped.

The offline learning is the *exact same recipe* as the Hilbert step — only the value parameterization
changes from `-||psi - phi||` to `psi^T phi / sqrt(d)`. The Bellman `max` is unsafe offline, so expectile
regression toward the private twin `target_rep_critic` gives the in-sample max; the `rep_critic` is fit
by TD to the bilinear value at the next state; the fixed GCIVL value does not backprop into the
embeddings. The **representation value** is `v = (phi(obs) * psi(goal)).sum(-1) / sqrt(rep_dim)` — the
harness names the state branch `phi` and the goal branch `psi` for the bilinear case — fit with
`adv = q_t - v`, `q_t = min(q1_t, q2_t)`, expectile `kappa = 0.7`; the **critic** regresses both heads to
`td = r + gamma * mask * v(next_s, g)`. `encode_goal` returns `psi(goals)` averaged over the ensemble.
The discount story is unchanged (contractive TD, discounted approximation), and I keep the
greedy-sufficiency argument honest by noting the discount transform preserves rankings: `gamma^d` is
strictly decreasing in `d`, so ordering successors by `gamma^{d*}` matches ordering by `-d*`, and the
monotone value transform never reorders the greedy step.

One design point to defend, because it is the reason this is a *representation* method and not just a
different value head: I keep the bilinear representation value strictly separate from the downstream
control value, as the harness forces. The bilinear structure makes the goal code relational and compact,
but a constrained inner-product head can be too constrained to extract a policy from — in antmaze the
goal code can care mostly about the x-y target, while the controller still needs joint angles and
velocities to choose actions. So the bilinear value learns `phi(g)`, and a separate monolithic GCIVL
value/actor consumes it for control. Two value functions are not redundant: one shapes the
representation, the other extracts control — which is exactly why the edit surface gives the module its
own private `rep_critic`.

Against Hilbert, then. The bilinear change buys direction and universality, and both should pay off
most where the symmetric metric hurt most. On **cube-single**, where I attributed the Hilbert shortfall
to irreversible, asymmetric reaching, the directed inner product should be the largest improvement — the
strongest claim here; if cube does *not* improve over Hilbert, the asymmetry diagnosis is wrong. On
**antmaze-large**, where reaching is mostly reversible but the value surface is richer than a
shared-metric embedding can represent, universality should give a clear but smaller gain. On
**pointmaze-large**, where Hilbert already matched the floor, I expect the two codes within noise. The
capacity subsumption sharpens this into a two-sided test: because the inner product reproduces the
negative squared Hilbert value with three extra coordinates, if bilinear came in *below* Hilbert
anywhere that would not indict its reach but point at optimization — the harder non-metric surface, the
lost `V(s, s) = 0` and triangle-inequality priors it must relearn.
