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
riding into the value) finally addressed at the level of the target, not patched after the fact.

The functional is the right ideal, but I cannot store an arbitrary function per goal and I do not know
`d*`. The sufficiency argument tells me exactly how to finitize it: the functional is *paired with a
state and evaluated*, `f(s') = d*(s', g)`. So model the two-argument value surface through two
embeddings, `V(s, g) ~= f(psi(s), phi(g))`, and export the goal-side vector `phi(g)` as the finite
code. The only open choice is the aggregator `f`. The Hilbert rung chose the metric
`f = -||psi - phi||` with `psi = phi`, and I just diagnosed why that is too rigid: symmetric and
non-universal. The simplest aggregator that fixes both problems is the **bilinear inner product**:

`f(psi(s), phi(g)) = psi(s)^T phi(g)`.

Two things fall out immediately. It is *directed*: `psi` and `phi` are different maps, so
`psi(s)^T phi(g)` need not equal `psi(g)^T phi(s)`, and the code can represent the asymmetric reaching
costs the metric could not — directly attacking the manipulation gap. And it is *universal*: with
learned feature maps of sufficient width, sums of separable products approximate any continuous
two-variable function on a compact domain, so the bilinear value can represent reaching-value surfaces
the metric provably cannot. In the practical head I scale the dot product by the width,
`V(s, g) = psi(s)^T phi(g) / sqrt(d)`. Without it, the dot product is a sum of `d` terms whose scale
grows with the representation width, so changing `rep_dim` would silently change the initial value and
gradient magnitude; dividing by `sqrt(d)` keeps the bilinear score order-one across widths — the same
square-root scaling attention uses, and load-bearing here for stable training at `rep_dim = 256`.

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
aggregator. The distilled module — bilinear `phi`/`psi`, the `/sqrt(d)` score, the expectile-plus-TD
`rep_loss`, goal code `psi(g)` — is in the answer.
