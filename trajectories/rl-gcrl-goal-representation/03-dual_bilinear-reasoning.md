The Hilbert rep did what I asked of it and stopped exactly where I predicted it would. Replacing raw
goal axes with axes where Euclidean distance is reachability lifted success clearly above the identity
floor on antmaze-large and cube-single, and on pointmaze-large it barely moved — which is consistent
with my diagnosis there, since raw position was already a serviceable code and there was little
reachability structure left to recover. But the residual weakness I flagged showed up in precisely the
place I said it would: the symmetric metric leaves the most on the table on the manipulation
environment, and trails on antmaze too, while only tying on pointmaze. That is not a tuning artifact.
It is structural, and naming the structure is what drives this step. The Hilbert value is
`V(s, g) = -||phi(s) - phi(g)||`, and a Euclidean norm is *rigidly symmetric*:
`||phi(s) - phi(g)|| = ||phi(g) - phi(s)||`. So the representation it learns is forced to assume that
reaching `g` from `s` costs the same as reaching `s` from `g`. In reversible navigation — a point or an
ant wandering an open maze — that assumption is roughly true, which is why pointmaze and antmaze were
not punished too hard. In manipulation it is false: pushing a cube off a ledge is easy, lifting it back
is hard; many transitions are one-way. Forcing a symmetric code onto an asymmetric reaching structure
costs expressiveness exactly where the task is hardest, and that is the gap I now want to close.

I should be precise about the evidence, since this task reports no per-seed numbers and no leaderboard —
what I carry forward from the Hilbert rung is a qualitative ordering, not a measured gap. The
consensus is that the symmetric metric sits clearly above the raw-goal floor but below what a directed
aggregator reaches, with the shortfall concentrated on antmaze-large and the cube manipulation setting
and the two essentially tied on pointmaze-large. That pattern is exactly the fingerprint of a symmetry
constraint biting where reaching is directed and vanishing where it is reversible, so I will let it set
the direction of the fix without pretending I can quantify the points it will recover.

There is a second, subtler limitation worth stating because it sharpens the fix. Even setting asymmetry
aside, a shared-embedding metric `||phi(s) - phi(g)||` is *not a universal approximator* of two-argument
functions. Not every pairwise function — not even every symmetric one — can be written as a Euclidean
distance between codes from a single shared map; there are finite metric spaces with no isometric `l2`
embedding at all. So the Hilbert head is doubly constrained: symmetric *and* limited in what pairwise
value surfaces it can represent. The reaching value `d*(s, g)` is in general neither symmetric nor
metric-shaped. If I want `phi(g)` to carry the full reachability relation, I need an aggregator that is
(a) directed — different maps for the state side and the goal side — and (b) expressive enough to
approximate an arbitrary continuous two-argument value. That is the design target for this step, and it
keeps everything else: the same offline value-learning machinery, the same harness contract, the same
`encode_goal` hook into the fixed GCIVL agent.

Let me build up the right object from the reaching problem rather than guessing an architecture. With
the OGBench shifted indicator reward `r(s, g) = I(s = g) - 1` and an absorbing goal, the optimal value
is a monotone transform of the optimal reaching time, `V*(s, g) = -(1 - gamma^{d*(s, g)})/(1 - gamma)`;
with the plain `0/1` absorbing reward it is `V*(s, g) = gamma^{d*(s, g)}`. Either way the value *is* the
reaching structure in log-discount coordinates. Now ask what a goal really *is* to a controller. It is
not just "far from here." Operationally, a goal is the thing that every state has some reaching relation
to. So the ideal representation of `g` is the whole function of incoming relations,
`phi^vee(g) = (s |-> d*(s, g))` — identify a goal by *all* its temporal distances from the state space.
This matches a familiar mathematical slogan: an object is determined by its relations to every other
object (the Riesz/kernel-section/Yoneda viewpoint). The slogan is only useful if the representation
actually satisfies the control requirements, so the two things I care about are sufficiency (the code
keeps everything an optimal greedy step needs) and noise invariance (the code drops exogenous detail).

Sufficiency first, because it tells me how the code is *consumed*, which in turn tells me what finite
form to give it. Suppose I am handed only the functional `f = phi^vee(g)`, so for any successor `s'` I
can query `f(s') = d*(s', g)`. The one-step greedy action for the optimal goal-conditioned value needs
only successor values: `argmax_a E_{s'}[gamma^{f(s')}] = argmax_a E_{s'}[gamma^{d*(s', g)}] =
argmax_a E_{s'}[V*(s', g)] = argmax_a Q*(s, a, g)`, the optimal greedy step. So the functional has
thrown away nothing the optimal policy needs; it never touches the raw goal, only the table of optimal
successor values. Noise invariance second: in a block-structured observation model where each
observation maps to a latent and the reward is latent, `r^ell(s, g) = I(p^ell(s) = p^ell(g))`, two goal
observations with the same latent induce *identical* latent rewards on every trajectory, so their
optimal values agree at every state and `phi^vee(g_1) = phi^vee(g_2)`. Exogenous observation noise that
does not change the latent task disappears — which is the floor's failure (joint angles, scene texture
riding into the value) finally addressed at the level of the target, not patched after the fact. Make it
concrete on cube manipulation: take two goal observations that agree on the cube's target pose but differ
in some coordinate the agent does not control — a background detail, an unrelated arm-joint reading at
the goal. Their latent reward `I(p(s) = p(g))` is the same indicator at every state `s`, so every
trajectory earns identical returns toward `g_1` and `g_2`, their optimal values coincide pointwise, and
the ideal functional maps them to the same code. The bilinear value is trained on exactly that
latent-reward signal, so the gradient never rewards `psi` for distinguishing `g_1` from `g_2` — the two
collapse. The floor did the opposite: it fed both raw observations, differences and all, straight into
the value, so the downstream net had to *learn* to ignore the exogenous coordinate from finite data.
Here the invariance is a property of what the code is trained to represent, not a burden pushed onto the
controller.

The functional is the right ideal, but I cannot store an arbitrary function per goal and I do not know
`d*`. The sufficiency argument tells me exactly how to finitize it: the functional is *paired with a
state and evaluated*, `f(s') = d*(s', g)`. So model the two-argument value surface through two
embeddings, `V(s, g) ~= f(psi(s), phi(g))`, and export the goal-side vector `phi(g)` as the finite
code. The only open choice is the aggregator `f`. The Hilbert rung chose the metric
`f = -||psi - phi||` with `psi = phi`, and I just diagnosed why that is too rigid: symmetric and
non-universal. There are three genuinely different ways to build a directed, more expressive `f`, and I
want to eliminate two of them on concrete grounds before committing. The most flexible is a monolithic
MLP over the concatenation, `f = MLP([psi(s), phi(g)])`: universal and directed, but it fails the one
structural requirement the sufficiency argument imposed — that the goal be carried by a *vector code*
that is paired with the state and evaluated. An MLP couples `s` and `g` through joint hidden layers, so
there is no clean goal-side vector whose relationship to the state code is the value; `encode_goal` would
still have to return `phi(g)`, but the value surface would no longer be a simple function of the two
codes, and the "a direction in code space is an actionable control signal" algebra that the Hilbert rung
banked would be gone. A second option is an explicit asymmetric quasimetric — say `sum_k ReLU(phi_k(g) -
psi_k(s))` — which is directed and, unlike the Euclidean norm, keeps the triangle inequality while
dropping symmetry, so it is arguably the "correct" object for `d*`. But it re-imposes metric axioms I do
not actually need for a *representation*: the sufficiency and noise-invariance arguments ask only that
the code preserve optimal successor values and collapse latent-preserving noise, not that it obey the
triangle inequality, and the quasimetric head is a fussier, harder-to-fit surface for no gain in the
representation role. The simplest aggregator that is directed and universal *and* keeps the value a plain
function of two vector codes is the **bilinear inner product**:

`f(psi(s), phi(g)) = psi(s)^T phi(g)`.

Two things fall out immediately. It is *directed*: `psi` and `phi` are different maps, so
`psi(s)^T phi(g)` need not equal `psi(g)^T phi(s)`, and the code can represent the asymmetric reaching
costs the metric could not — directly attacking the manipulation gap. And it is *universal*: with
learned feature maps of sufficient width, sums of separable products approximate any continuous
two-variable function on a compact domain, so the bilinear value can represent reaching-value surfaces
the metric provably cannot. I can make the "cannot" sharp by checking that the inner product actually
*subsumes* the Hilbert metric rather than merely differing from it. The negative squared distance
expands as `-||a - b||^2 = 2 a^T b - ||a||^2 - ||b||^2`, and that is exactly a single inner product in a
lifted space: set `psi(s) = [2 phi(s), -||phi(s)||^2, 1]` on the state side and `phi_goal(g) = [phi(g),
1, -||phi(g)||^2]` on the goal side, and their dot product is `2 phi(s)^T phi(g) - ||phi(s)||^2 -
||phi(g)||^2 = -||phi(s) - phi(g)||^2`. So with just three extra coordinates the bilinear head reproduces
the negative squared Hilbert value; the bilinear family strictly contains the metric family (up to the
monotone `x -> -sqrt(-x)` reparameterization, which does not matter for the greedy argmax or for the
code's role as coordinates). That settles the capacity question in one direction: the inner product can
be no *less* expressive than the metric, so the only thing that could go wrong is optimization, not
representational reach — a reassuring floor under the whole rung.

In the practical head I scale the dot product by the width, `V(s, g) = psi(s)^T phi(g) / sqrt(d)`, and
the scale is worth computing rather than asserting. At initialization the two codes are roughly
zero-mean with order-one per-coordinate variance, so `psi(s)^T phi(g) = sum_{k=1}^{d} psi_k phi_k` is a
sum of `d` near-independent order-one terms: its standard deviation grows like `sqrt(d)`, which at
`d = 256` is `16`. An unscaled score of typical magnitude `16` would swamp the reward of magnitude `1`
inside the TD target `r + gamma * mask * v`, and — worse — that magnitude would *change with the width*,
so retuning `rep_dim` would silently rescale the value and its gradients and force a re-tune of the
expectile dynamics. Dividing by `sqrt(d) = 16` pulls the initial score back to order one and makes it
invariant to the width choice; it is the same square-root normalization attention uses, and it is
load-bearing here precisely because `rep_dim = 256` is large enough that the unscaled `16` would matter.
One thing the inner product buys for free relative to the metric is numerical smoothness: there is no
`sqrt(||.||)` anywhere, so no `1/(2 sqrt(.))` singularity at coincident codes, and the `1e-6` floor the
Hilbert rung needed simply disappears — the gradient of `psi^T phi / sqrt(d)` with respect to `phi(g)` is
just `psi(s) / sqrt(d)`, bounded wherever LayerNorm keeps the codes bounded, one fewer fragile constant
than before. The honest cost of the trade is that the bilinear head gives up the metric's freebies:
`V(s, s) = psi(s)^T phi(s) / sqrt(d)` is not pinned to `0`, `V <= 0` is not guaranteed, and the triangle
inequality no longer holds by construction. Those were true inductive biases, and I am choosing to relearn
them from data in exchange for direction and universality — a good trade exactly where reaching is
asymmetric, and a roughly neutral one where it was already metric-shaped.

Now the offline learning, and here I keep the *exact same recipe* as the Hilbert rung — only the value
parameterization changes from `-||psi - phi||` to `psi^T phi / sqrt(d)`. The reasoning is identical:
the optimal Bellman backup's `max` over actions is unsafe offline because it queries out-of-distribution
actions, so I use expectile regression for an in-sample max. The harness contract is the same too: the
fixed GCIVL value loss does not backprop into the embeddings, so `compute_rep_loss` runs a self-contained
loop that shapes `phi`/`psi`, using the private twin `rep_critic` and its EMA `target_rep_critic` that
the loop maintains. The two-part loss carries over verbatim in structure. The **representation value** is
now the bilinear score, `v = (psi(s) * phi(g)).sum(-1) / sqrt(d)` (here the harness names the state
branch `phi` and the goal branch `psi` for the bilinear case — the bilinear value computes
`phi(obs) * psi(goal) / sqrt(rep_dim)` summed over the last axis), and I fit it by expectile regression
toward the target critic: `adv = q_t - v`, `q_t = min(q1_t, q2_t)` from `target_rep_critic`, loss
`|kappa - 1(adv < 0)| adv^2` with `kappa = rep_expectile = 0.7`. The **critic** is the ordinary TD fit:
`td = r + gamma * mask * v(next_s, g)` (stop-gradient), both online heads regressed to it with squared
error. The total `rep_loss` is the sum; `encode_goal` returns the goal branch `psi(g)` averaged over the
ensemble. The interlock is the same as before — the critic learns a bootstrapped reaching value from
data, the expectile loss drags the bilinear value toward an in-support max of that estimate, and because
the value is now `psi^T phi / sqrt(d)`, that drag injects the *directed* reaching structure into the two
embeddings. The discount story is unchanged (`gamma` for TD stability, converging to a discounted
approximation), and I no longer need the square-root floor that the metric required — there is no
`sqrt(||.||)` singularity in an inner product, so the bilinear value is numerically smoother at init,
one fewer fragile constant than the Hilbert rung.

Two checks reassure me the head is wired correctly. First the shapes, which the ensemble and the concat
make worth tracing once: `observations` and `goals` come in `(B, obs_dim)`; the state ensemble `phi` and
goal ensemble `psi` produce `(2, B, rep_dim)` each; the elementwise product scaled by `1/sqrt(rep_dim)`
and summed over the last axis is `(2, B)`, averaged over the ensemble axis to `v_mean` of shape `(B,)`.
The critic branch is identical in shape to the Hilbert rung — `[observations, goals, actions]` concat
into the twin `target_rep_critic`, two `(B, 1)` heads squeezed and `min`'d to a `(B,)` `q_t` — so
`adv = q_t - v_mean` and `td_target = r + gamma * mask * next_v_mean` are both clean `(B,)` differences,
and `encode_goal` returns `psi(goals).mean(axis=0)` of shape `(B, rep_dim)`. Every shape matches the
Hilbert rung except the scalar the value is built from, which is the whole point: only the
parameterization changed, `-||phi_s - phi_g||` to `phi_s^T psi_g / sqrt(d)`, and the surrounding
expectile-plus-TD machinery is byte-for-byte the same recipe. Second, the greedy-sufficiency argument I
leaned on needs the discount transform to preserve rankings, and it does: `gamma in (0, 1)` makes
`gamma^d` strictly decreasing in `d`, so ordering successors by `gamma^{d*(s', g)}` is the same ordering
as by `-d*(s', g)`, hence `argmax_a E[gamma^{d*(s', g)}]` picks the same action as minimizing expected
reaching time — the monotone value transform never reorders the greedy step, which is exactly why
carrying the code in log-discount coordinates loses nothing the optimal policy needs.

One design point I want to defend explicitly, because it is the reason this is a *representation* method
and not just a different value head: I keep the bilinear representation value strictly separate from the
downstream control value, exactly as the harness forces. The bilinear structure is what makes the goal
code relational and compact, but a constrained inner-product head can be *too* constrained to extract a
good policy from — in antmaze the goal code can mostly care about the x-y target, while the controller
still needs joint angles and velocities to choose actions. So the bilinear value learns `phi(g)`, and a
separate monolithic GCIVL value/actor consumes `phi(g)` to do control. Two value functions are not
redundant: one shapes the representation, the other extracts control. That is precisely why the task's
edit surface gives the module its own private `rep_critic` rather than reusing the agent's value — and
why this rung does not touch anything downstream.

Now the falsifiable close, against the Hilbert numbers. The bilinear change buys two things the metric
lacked — direction and universality — and both should pay off most where the symmetric metric hurt
most. On **cube-single**, where I attributed the Hilbert shortfall to irreversible, asymmetric reaching
that a symmetric norm cannot encode, the directed inner product should be the largest improvement on the
ladder — this is the rung's strongest claim, and if cube does *not* improve over Hilbert, the
asymmetry diagnosis is wrong. On **antmaze-large**, where reaching is mostly reversible but the value
surface is still richer than a shared-metric embedding can represent, universality should give a clear
but smaller gain over Hilbert. On **pointmaze-large**, where Hilbert already essentially matched the
floor and reaching is benign, I expect the bilinear and Hilbert codes to be close — possibly bilinear a
touch lower or a touch higher, but within noise — because there is little asymmetric or non-metric
structure to exploit. So the sharp, falsifiable prediction is: bilinear > Hilbert most on cube, clearly
on antmaze, roughly tied on pointmaze, with the bilinear average above the Hilbert average — and the
whole improvement traceable to having dropped the symmetric-metric assumption for a directed, universal
aggregator. The capacity subsumption tightens this into a two-sided test on the one metric available,
success rate across the three environments. Because the inner-product head can reproduce the negative
squared Hilbert value with three extra coordinates, it can be no less expressive than the metric; so if
bilinear were to come in *below* Hilbert anywhere, that would not indict the representation's reach — it
would point at optimization (the harder-to-fit non-metric surface, the lost `V(s, s) = 0` and triangle-
inequality priors it now must relearn). And if it fails to improve on cube specifically, the asymmetry
diagnosis itself is wrong, because cube is where direction should matter most. Either outcome is
informative, which is what makes the prediction worth staking. The distilled module — bilinear `phi`/`psi`, the `/sqrt(d)` score, the expectile-plus-TD
`rep_loss`, goal code `psi(g)` — is in the answer.
