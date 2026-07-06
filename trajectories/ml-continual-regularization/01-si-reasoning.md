The harness hands me a default fill — diagonal Fisher importance with a quadratic penalty — and a frozen
loop that snapshots parameters at every context boundary, accumulates whatever importance I return, and,
most tellingly, maintains a per-step accumulator `model._custom_W` that adds `-grad * (p - p_old)` every
single training step. That `_custom_W` is the thing I want to start from, because it is *already*
computing the one quantity the default Fisher fill throws away, and the whole design of this rung is to
spend it instead of paying for a separate Fisher phase. So let me derive what importance that accumulator
actually measures, and why anchoring the penalty to it is a coherent place to begin the ladder rather
than the default.

Start from the failure the regularizer exists to fix, stated mechanically. Training is sequential: at
context `mu` I only have the loss `L_mu` in front of me; the data and losses of everything before are
gone. What I actually want to minimize is the sum over everything I have ever seen, `L = sum_mu L_mu`,
but I am forced to descend one summand at a time. Descending `L_mu` drags the shared weights `theta`
wherever lowering `L_mu` demands, and those same weights were carrying context 1, so `L_1` quietly climbs
back while I am not looking. The representation is shared and distributed, so any weight change that helps
the new context tends to corrupt an old one — the more I train, the more I overwrite. I cannot change the
data (no replay, the constraint forbids it), and I would rather not change the architecture, so I change
the *objective*: add a penalty to `L_mu` that discourages clobbering what the net already knew. The crude
version writes itself — anchor every weight to where it ended up, `L_mu + sum_k (theta_k - theta_tilde_k)^2`
with `theta_tilde` the boundary snapshot. But one global stiffness cannot be right, and I can see why
without running it: large enough to actually hold the old context in place and the net is too rigid to
learn the new one; small enough to learn the new one and it does not hold the old. The stiffness has to be
*per parameter* — stiff on the weights an old context relied on, slack on the ones it did not. So the real
question is two coupled ones: which parameters were important, and how do I turn "important" into a
penalty.

What does "important" mean here? The net is wildly over-parameterized — a context does not pin a single
solution, it pins a whole low-loss manifold, many weight settings all solving it equally well. That is my
opening: somewhere near the old solution there may well be a configuration that *also* solves the new
context, and the job is to steer toward that one instead of toward an arbitrary new minimum that happens
to wreck the old. To steer I need to know, per weight, how much the old context would suffer if I moved
it. Locally that is a curvature statement: expand the old loss around its minimum, the Hessian tells me
how much loss I pay for moving in each direction, stiff directions are the ones the old context cared
about. So per-parameter importance is morally per-parameter curvature, and a curvature-weighted quadratic
penalty is the natural form — which is exactly what the default Fisher fill encodes.

Given that importance is curvature, let me lay out what estimators are actually on the table at this first
rung and cost them honestly, rather than reaching for the elegant one because it is elegant. Option one is
to leave the scaffold's default in place: a diagonal Fisher recomputed at the endpoint in a separate
post-context sweep. Option two is the uniform-stiffness L2 anchor I just rejected — a single global spring
constant — which cannot have per-parameter selectivity by construction, so it is out on the argument
above. Option three is a cheap magnitude proxy: accumulate `sum |g_k|` or `sum g_k^2` along training and
call the weights with the biggest gradients important. It looks free, and it tempts because the loop is
already handling gradients, but it credits the wrong quantity. A weight whose gradient stays large while
it barely moves — because the surface is stiff exactly there and the optimizer cannot budge it — racks up
enormous "importance" without ever having changed the loss; a weight that quietly slid the loss down under
a small persistent gradient scores little. And it has the units of gradient (or gradient-squared), not of
loss, so I cannot turn it into a faithful spring without exactly the motion-bookkeeping I am about to
derive. That leaves option one and option four — the path integral `_custom_W` already holds — and since I
have argued both are curvature-flavored, the deciding question is cost: at equal faithfulness the ladder
should open with the estimator that spends nothing the loop is not already spending.

So price option one concretely. The default `estimate_importance` runs `min(len(dataset), 200)` single-
example forward passes after each context, and for *each* of those it does one backward pass per output
class — `output.shape[1]` of them, holding the graph with `retain_graph` between classes. On the binary
Split-MNIST heads that is `200 * 2 = 400` backward passes at every context boundary; on the ten-way
Permuted-MNIST and Split-CIFAR100 heads it is `200 * 10 = 2000` backward passes per boundary, repeated at
all ten boundaries. That entire sweep exists only to read curvature at the final point. Meanwhile the loop
has *already* summed `_custom_W = sum(-g * delta_theta)` step by step, out of gradients training computed
for the optimizer regardless — option four's importance is one `O(#params)` division away, with zero extra
forward or backward passes. If option four's `W` genuinely measures curvature-like importance, then the
first rung that costs the loop nothing beats the first rung that pays for a 2000-backward sweep, at equal
faithfulness. So the elegance is worthless unless the running sum is really curvature; let me check that it
is, because that is the load-bearing claim.

Make it precise. Picture the training of context `mu` as a path `theta(t)` through parameter space, from
the boundary snapshot `theta(t_{mu-1})` to the endpoint `theta(t_mu)`. An infinitesimal step `delta(t)`
changes the loss, to first order, by `sum_k g_k(t) delta_k(t)` with `g_k = dL/dtheta_k`. So each
coordinate's little move contributes `g_k(t) delta_k(t)` to the change in total loss right then. Sum these
infinitesimal contributions over the whole path. Writing `delta_k(t) = theta'_k(t) dt`, the total loss
change along the path is the line integral of the gradient field, `integral g(theta(t)) . theta'(t) dt`.
The gradient of a scalar loss is a *conservative* field — it is literally a gradient — so this integral
does not depend on the path; it equals the difference of the loss between the endpoints,
`L(theta(t_mu)) - L(theta(t_{mu-1}))`, which is negative during successful descent. That is a sanity
anchor: whatever I build from the per-step pieces, summed up it must reproduce the total loss change with
the right sign. And crucially the integral *decomposes coordinate by coordinate*, because the dot product
is a sum: each term `integral g_k theta'_k dt` is the contribution of one parameter to the whole context's
loss drop. That per-parameter line integral is what I call importance,
`omega_k^mu = - integral g_k(theta(t)) theta'_k(t) dt`. The minus sign is deliberate: I care about loss
*decreasing*, and a parameter that drove the loss down has `g_k theta'_k < 0` (it moves against its own
gradient under descent), so flipping the sign makes `omega_k^mu` a positive number measuring how much that
parameter helped. This is exactly the thing the default Fisher cannot get without its separate after-the-
fact phase — and I am getting it from quantities the loop already has at every step.

Can I accumulate it cheaply during training? The integrand is `g_k(t) . theta'_k(t)` — the gradient times
the rate of change of the parameter. At each optimizer step I have the gradient `g_k`, and the parameter
update `delta_theta_k = theta_k^new - theta_k^old` is the discrete stand-in for `theta'_k dt`. So I keep a
running sum, per parameter, of `-g_k . delta_theta_k` — and that is *exactly* what the harness's
`_custom_W[n] += -grad * (p - p_old[n])` already does. The loop's per-step hook is the discrete path
integral. For full-batch descent with an infinitesimal learning rate this running sum *is* the path
integral; in practice I run SGD, so `g_k` is a noisy minibatch gradient and the noise leaks in. Let me
make the leak precise, because it decides the one hyperparameter I get. Split the stochastic gradient into
its true and noise parts, `g_k = gbar_k + xi_k` with `E[xi_k] = 0`, and note the recorded update moves, to
first order, opposite the same minibatch gradient — the loop records the *actual* step whatever the
optimizer, so write `delta_theta_k ~ -a_k (gbar_k + xi_k)` for some positive effective step `a_k`. Then the
per-step contribution is `-g_k delta_theta_k ~ a_k (gbar_k + xi_k)^2 = a_k (gbar_k^2 + 2 gbar_k xi_k +
xi_k^2)`. Take the expectation over the minibatch: the cross term dies because `E[xi_k] = 0`, leaving
`a_k (gbar_k^2 + Var(xi_k))`. The clean path integral wanted only `a_k gbar_k^2`; the extra `a_k Var(xi_k)`
is a strictly positive, sign-definite bias present at *every* step, and it accumulates additively over the
whole context. So `W` systematically *overestimates* the true importance, by roughly the effective step
times the summed gradient-noise variance — a bias that grows with the number of steps, the noise scale,
and the hardness of the net. Hold that; it forces both a correction and a prediction.

Now turn `omega_k^mu` into a penalty. I want the penalty added while training a later context to recreate,
as far as the descent dynamics can tell, the effect of the unavailable past loss — to pull the weights as
if that loss were still present. The simplest faithful surrogate is a quadratic anchored at the boundary
weights `theta_tilde_k`, `L_mu + c sum_k Omega_k (theta_tilde_k - theta_k)^2`, with a per-parameter
strength `Omega_k` to set and a single scalar `c` trading old against new. What should `Omega_k` be? Demand
faithfulness: if I had trained on the quadratic surrogate `s_k (theta_tilde_k - theta_k)^2` *instead of*
the real old loss, it should produce the same per-parameter loss drop over the same motion. Over the
context a parameter moved by `Delta_k = theta_k(t_mu) - theta_k(t_{mu-1})`, and the quadratic's loss drop
over that motion is, up to the constant, `s_k Delta_k^2`. Set that equal to the credit `omega_k^mu` the
real loss actually accrued: `s_k Delta_k^2 = omega_k^mu`, forcing `s_k = omega_k^mu / Delta_k^2`. There is
the normalization — divide the path-integral importance by the square of how far the parameter actually
moved. That `Delta^2` in the denominator is not a fudge: it makes the surrogate quadratic yield the same
`omega` over the same distance, so descent on the surrogate mimics descent on the true past loss, and it
fixes the units — `omega` is in units of loss, `Delta^2` in parameter-squared, so `omega/Delta^2` times
`(theta_tilde - theta)^2` comes out in units of loss, matching `L_mu` exactly and making `c` a clean
dimensionless knob. I bolt on a small damping `xi` in the denominator for a concrete reason: a parameter
that barely moved has `Delta_k -> 0` and `omega/Delta^2` blows up — a weight that sat still would get
infinite importance, which is nonsense — and `xi` floors that. The harness exposes exactly this constant as
`model.epsilon` (default `0.1`). So the importance I return per parameter is `W_k / (Delta_k^2 + epsilon)`,
read straight from the loop's accumulated `W` and the net motion since the boundary snapshot.

It is worth reading what `epsilon = 0.1` actually does at the two ends of the `Delta^2` range, because the
constant is doing real work, not decoration. For a weight that moved a lot — `Delta^2 >> 0.1` — the floor
is negligible and `Omega_k -> W_k / Delta_k^2`, the clean faithfulness ratio. For a weight that barely
budged — `Delta^2 << 0.1` — the floor takes over and `Omega_k -> W_k / 0.1 = 10 W_k`, so the damping caps
the importance of a near-static weight at ten times its accumulated path credit rather than letting a
vanishing denominator send it to infinity. And a near-static weight has small `W_k` too (little motion
means little `-g delta_theta` accumulated), so numerator and denominator both shrink and the bare ratio is
ill-conditioned exactly there; `epsilon` is what makes that limit well-behaved, resolving the `~0/0` toward
`10 W_k` instead of noise. The scale `0.1` is thus a soft threshold between "moved enough that its
faithfulness ratio is trustworthy" and "sat still, so trust its raw credit scaled by a fixed cap." I take
the harness value; it is the right kind of constant and there is no reason to fight it at rung one.

Let me pressure-test whether this is really measuring curvature, the way the default Fisher claims to, on
the one case I can solve: a quadratic loss `E(theta) = 0.5 (theta-theta*)^T H (theta-theta*)`. Continuous
gradient descent gives `theta(t) = theta* + exp(-Ht/tau)(theta(0)-theta*)`, so the velocity is
`theta'(t) = -(1/tau) H exp(-Ht/tau)(theta(0)-theta*)`. The importance is `-integral g_k theta'_k dt`, and
under descent `tau dtheta/dt = -g`, so `-g_k theta'_k = tau (dtheta_k/dt)^2`; the per-parameter `omega` are
the diagonal of `Q = tau integral (dtheta/dt)(dtheta/dt)^T dt`. Diagonalize `H` with eigenpairs
`lambda^a, u^a` and let `d^a` be the initial displacement along `u^a`; the time integral of
`exp(-(lambda^a+lambda^b)t/tau)` is `tau/(lambda^a+lambda^b)`, and after the `1/tau^2` from the two
velocities and the leading `tau`, the `(a,b)` prefactor is `lambda^a lambda^b/(lambda^a+lambda^b)` — `tau`
drops out entirely, so importance does not depend on the learning rate, reassuring. Averaging over random
initial conditions with `<d^a d^b> = sigma^2 delta_{ab}`, the off-diagonals vanish, the diagonal gives
`lambda^a/2`, and `<Q> = 0.5 sigma^2 H`. So on average the path-integral matrix is *half the Hessian* up
to the displacement scale — and the half is exactly the half in the loss drop of a quadratic bowl. Dividing
by `Delta_k^2` (which averages to `sigma^2`) strips the trajectory scale and leaves `Omega_k = 0.5 H_kk`;
because I write the penalty as `Omega_k (theta-theta_tilde)^2` with no leading half, the effective
curvature is `2 Omega_k = H_kk`. So the `Delta^2` normalization I introduced for faithfulness also turns
the path integral into the right curvature convention.

Let me nail that with the smallest concrete instance, one coordinate, because the eigen-averaging above
hides the mechanics. Take `H = 2`, initial displacement `d = 1`, `tau = 1`. Then `theta(t) - theta* =
e^{-2t}`, the gradient is `g = H(theta - theta*) = 2 e^{-2t}`, and the velocity is `theta' = -2 e^{-2t}`.
The integrand `-g theta' = -(2 e^{-2t})(-2 e^{-2t}) = 4 e^{-4t}`, and `integral_0^inf 4 e^{-4t} dt = 1`,
so `omega = 1`. The net motion is `Delta = theta(inf) - theta(0) = -1`, `Delta^2 = 1`, so
`Omega = omega/Delta^2 = 1` and the effective curvature `2 Omega = 2 = H` — dead on, no averaging needed in
one dimension because there are no off-diagonals to cancel. Now put the default beside it on the same
example: the empirical Fisher is the squared score at the endpoint `theta*`, where `g = H(theta* - theta*)
= 0`, so the endpoint Fisher reads *zero* curvature. It has thrown away all the curvature by the time it
looks; the path integral accumulated curvature-flavored information along the way, while the gradients were
still nonzero, at no extra backward pass. That is the sharpest form of the case for starting the ladder
here rather than at the default — and it is not a slogan, the one-coordinate numbers say `1` versus `0`.

Before I pin the hyperparameter, trace what happens across contexts, because that is where the noise bias
comes home to roost. The loop sums whatever I return into `_custom_importance` and re-anchors
`_custom_prev_params` to the latest boundary each time, so after `K` contexts the accumulated stiffness in
each direction is `sum_t omega_t / (Delta_t^2 + epsilon)`, anchored at the most recent snapshot — a running
sum that, with the framework's default `gamma = 1`, only ever *grows* and is never bounded or re-scaled.
The cost is fine — `estimate_importance` is one `O(#params)` pass with no forward/backward, and
`compute_regularization_loss` is one `O(#params)` reduction per step; memory is a single importance dict
plus one snapshot, constant in the number of contexts. But the *magnitude* is not fine on a long sequence.
Each context dumps its `omega/Delta^2` into the pile, and I just showed each `omega` is biased upward by
the accumulated SGD noise variance; so the pile grows faster than the true curvature would, and later
contexts — especially deterministic ones on uncorrelated inputs, which move weights a long way and dump
large-norm increments — pour disproportionate stiffness onto a fixed-capacity net. There is no decay to
relieve it. This is the mechanism I have to keep in view when I read the numbers: unbounded, upward-biased,
never re-scaled.

One structural fact about the benchmarks sharpens where that stiffness lands. The networks are all multi-
head, and the loop "computes the task loss only over the classes present in the current context," so each
context is read out through its own head and the springs on head weights only ever face the one context
that trained them — the real battleground is the *shared trunk*, whose features every context reuses and
overwrites. On the two task-incremental benchmarks the separate heads insulate the readout, so forgetting
is entirely a story about trunk weights; on Permuted-MNIST the same ten classes recur under each
permutation and the uncorrelated inputs force the trunk to re-represent from scratch every context, which
is precisely why the trunk springs there are asked to hold mutually contradictory settings and why the
undecayed accumulation should hurt most on that sequence. So when I predict the failure below, I am
predicting it about the trunk, and the multi-head structure is what keeps the heads from confounding the
read.

Now the hyperparameter the noise decides. Because `W` overestimates, `c = 1` (equal weight to old and new,
if the integral were exact) over-constrains the net; the honest remedy is `c < 1`, an empirical knob
trading old memories against capacity, and on noisier or harder benchmarks it wants to come down. In this
harness `c` is the per-benchmark `reg_strength`, scalable by `CONFIG_OVERRIDES['reg_strength_scale']`; I
leave the override empty and take the harness default — `c = 1` is the honest in-the-limit value and the
baseline spends it as-is, so any over-constraint from the noise bias will show *undamped* in the numbers,
which is what I want at rung one: I would rather see the failure mode clearly than tune it away before I
understand it. The loop also handles the cross-context summation, so my `estimate_importance` returns the
raw normalized increment `W_k/(Delta_k^2 + epsilon)` per context, and the per-step penalty is the cheap
`sum_k Omega_k (p-theta_tilde)^2` with no leading half — both filling the two slots exactly (the full
module is in the answer).

So at step 1 my edit is: leave the loop and its `_custom_W` accumulator alone, replace the default
endpoint-Fisher `estimate_importance` with the trajectory normalization `W/(Delta^2 + epsilon)`, and
replace the default `0.5 * sum F (theta-theta*)^2` with the no-half `sum Omega (theta-theta*)^2`. Now reason
about what this floor should do, because that is the point of running it, and the noise mechanism above
turns into a falsifiable pattern across the three benchmarks. On a short task-incremental sequence —
Split-MNIST, 5 binary tasks, a small MLP trained to convergence each time — the path integral is
accumulated over few, clean descents, the overestimate `a_k Var(xi_k)` is mild and summed over only a
handful of contexts, and the per-parameter springs should hold the earlier binary tasks well; I expect SI
to be strong here, possibly the strongest single number on the whole ladder, because the trajectory
importance is at its most faithful when each context converges cleanly and there are only a handful of
them. Split-CIFAR100 is task-incremental with 10 contexts on a CNN — more contexts, noisier gradients, a
harder net, so more accumulated noise bias — but each task still has its own head and the contexts do not
demand mutually contradictory weights, so I expect the overestimate to bite only somewhat and the average
to land in the mid-0.5s, respectable but not dominant. The real exposure is Permuted-MNIST: domain-
incremental, 10 contexts of *uncorrelated* inputs, a larger MLP, every context demanding genuinely
different weights. Here the running `Omega` only ever adds across ten contexts, the SGD overestimate
compounds with each, and there is no decay to relieve it — so by the later permutations the accumulated
springs may have rigidified the fixed-capacity net so hard that new permutations cannot be learned *and*
the over-strong, scale-mismatched importance (deterministic later contexts dump large-norm increments) may
even fail to hold the early ones. If SI is going to break anywhere, it is on the long domain-incremental
sequence: I expect a steep decay across the ten Permuted contexts, the average dragged well below what a
bounded importance would give, even as Split-MNIST stays excellent. That asymmetry — best on the short
clean sequence, worst on the long uncorrelated one, with Split-CIFAR100 in between — is exactly the
falsifiable signature I am watching for, and if it shows up it points the next rung straight at importance
that does not over-accumulate: a curvature estimate that is bounded and re-centered rather than a path
integral that only ever grows.
