SI's numbers told me exactly what its trajectory importance does well and exactly where it breaks, and
the split is as sharp as I feared. On Split-MNIST it was excellent — 0.9852 average, the per-context line
{0.9991, 0.9812, 0.9859, 0.9754, 0.9845} barely sagging, `max - min = 0.0237`, every one of the five
binary tasks held above 0.975 — which is the regime I predicted it would own: five short binary tasks, a
small MLP trained cleanly to convergence each time, few contexts, mild SGD overestimate, so the path-
integral springs held the early tasks almost perfectly. Split-CIFAR100 landed at 0.5363, and when I sort
the per-context accuracies {0.4140, 0.5500, 0.4840, 0.5130, 0.4600, 0.6210, 0.5800, 0.5960, 0.5920,
0.5530} against the context index there is no monotone trend at all — the earliest is the lowest (0.414)
but the middle is jagged and the last context (0.553) sits *above* the first, so this line is not forgetting-
shaped, it is per-task-difficulty-shaped. That tells me something I will use: on this benchmark the
estimator is not the bottleneck; capacity and task hardness are, and swapping how I read curvature will
barely move it. But Permuted-MNIST is the collapse I was watching for, and it is worse than "dragged
below expectations": 0.4474 average, and the per-context line is a *monotone slide* — {0.7862, 0.7221,
0.6577, 0.6121, 0.5270, 0.4036, 0.2898, 0.1933, 0.1430, 0.1394}.

Read that Permuted line with a pencil, because the shape is the diagnosis. The step-to-step drops are
{0.064, 0.064, 0.046, 0.085, 0.123, 0.114, 0.097, 0.050, 0.004}: they *accelerate* through the middle,
peaking around the fifth-to-seventh contexts, then *decelerate* at the end — and the deceleration is not
recovery, it is the chance floor. A ten-way readout at chance sits near 0.10, and the last two contexts,
0.1430 and 0.1394, are about 1.4 times chance, effectively dead; the slide flattens because it has hit the
bottom, not because the springs started holding. That accelerate-then-flatten-at-the-floor fingerprint is a
system being driven into the floor. And crucially the earliest context is *also* eroding: context 1 ends at
0.7862, where the very same first-trained, most-protected task on Split-MNIST held 0.9991 — a 0.21 absolute
gap on the context that had the most protection and the least interference. So it is not simply "can't learn
the new ones." It is both failure modes at once — the accumulated, undecayed `Omega` has rigidified the
fixed-capacity net so the later permutations can't be learned, *and* the over-strong, scale-mismatched
importance has stopped holding the early ones. That is precisely the long-uncorrelated-sequence disease I
flagged: the running path integral only ever adds across ten contexts, the SGD overestimate compounds at
roughly `a_k Var(xi_k)` per step, and there is no relief. The average slope from context 1 to 10 is
`0.6468 / 9 = 0.072` per context, a steady seven-point bleed with no bounded resting point. So the
diagnosis is clean — SI's importance is *unbounded and grows*, and on the one benchmark that demands
genuinely different trunk weights from many contexts it both over-constrains and under-protects. The next
rung has to fix the *importance estimator* itself: I want a per-parameter curvature that is bounded,
anchored, and PSD by construction, not a trajectory sum that runs away.

Lay out the bounded-estimator options before committing, because "bounded" admits several answers and most
are patches. One: keep SI's running sum but clip or renormalize it per context so no single context can
dump a large-norm increment. That treats the symptom — it caps the magnitude — but leaves the sign
problem (the sum can still be negative) and, worse, does nothing about the fact that ten clipped springs
still only ever add on a fixed-capacity net; I would be rescaling a runaway, not stopping it. Two: read
importance as the squared gradient of the *true-label* loss at the endpoint — the empirical Fisher. That
is bounded and PSD, but it is evaluated at exactly the point where a well-fit context has near-zero
gradient on its own labels, so at a clean minimum it reads close to zero and tells me almost nothing; it
also uses the labels rather than the model's predictive spread, which biases it. Three: the exact Hessian
diagonal. It is the real curvature, but the raw Hessian at a minimum found by a stochastic optimizer can
have small negative entries (the point is not a perfect stationary point), so it would hand me negative
stiffnesses, and forming even its diagonal exactly is a second-derivative computation I would rather avoid.
Each patch fails for a *structural* reason, not a tuning reason, and the structure they all miss is that
importance ought to be the precision of a posterior — a genuinely PSD, genuinely bounded object with a
first-order estimator. So rather than patch, derive it.

So go back to what "importance" should be and derive it properly rather than reading it off the path. The
honest way to think about "what does the network know about a finished context A" is probabilistic.
Training to optimize parameters is, read correctly, finding the most probable `theta` given the data: by
Bayes, `log p(theta|D) = log p(D|theta) + log p(theta) - log p(D)`, and the cross-entropy loss *is* a
negative log-likelihood, so MAP estimation and loss minimization are the same act seen two ways. Split the
data into the finished context A and the current context B, assume conditional independence given `theta`,
and apply Bayes a second time to peel off B:
`log p(theta | D_A, D_B) = log p(D_B | theta) + log p(theta | D_A) - log p(D_B | D_A)`. The last term does
not depend on `theta`, so it vanishes under any gradient — set it aside. Stare at the rest. The first term
is the context-B loss, what I would minimize anyway. The middle term, `log p(theta | D_A)`, is the
*posterior over weights given A*, and notice that *all* of A's information has been absorbed into this one
object. So if I had that posterior I would be done: maximize `log p(D_B|theta) + log p(theta|D_A)` and I am
fitting B while respecting everything A taught. The posterior is the compact summary of A I want — but it
is a distribution over millions of weights, intractable. I need an approximation that is cheap to store,
cheap to add per step, and encodes per-weight importance. A Gaussian gives all three: its negative-log-
density is a quadratic in `theta` — exactly the spring penalty I already want — and its precision becomes
per-weight stiffness once I make it diagonal. This is the key difference from SI's route: SI never named a
posterior; it built a plausible running sum and *checked* it was curvature-flavored on a quadratic. Here
the curvature is derived as the precision of an approximate posterior, which is what will let me bound it.

How do I Gaussian-approximate `p(theta | D_A)` when I cannot write it down? Laplace's method, and it is
natural here. I trained A to a (local) optimum `theta*_A`; at that point the gradient of
`-log p(theta | D_A)` vanishes — that is what "optimum" means. Expand `-log p(theta|D_A)` to second order
around `theta*_A`: the constant I drop, the *first*-order term is zero because the gradient vanishes there,
and I am left with `-log p(theta|D_A) ~ const + 0.5 (theta - theta*_A)^T H (theta - theta*_A)`, where `H`
is the Hessian of the negative log posterior at `theta*_A`. That is the quadratic part of a Gaussian with
mean `theta*_A` and precision `H`. Because `theta*_A` is a minimum of `-log p`, `H` is positive semi-
definite there — good, a precision had better be PSD. So the summary of A is a quadratic bowl centered at
`theta*_A` with curvature `H`: the diagonal of `H` tells me, per weight, how sharply A's negative log
posterior rises as I move it. Large curvature means A is sensitive to that weight — move it and A degrades
fast, so it should be stiff; near-zero curvature means A does not care, leave it loose for B. This is the
crucial structural contrast with SI's collapse: this importance is evaluated *at one point* with bounded
curvature, not summed along a trajectory that compounds. There is nothing here that grows with the number
of steps in a context — which is the exact property whose absence produced the 0.072-per-context bleed.

But `H` is the full Hessian — data-likelihood part plus prior curvature — a millions-by-millions matrix I
cannot form, store, or invert. Wall. I need a first-order object that matches the likelihood curvature.
This is where the Fisher information earns its place. The Fisher is
`F = E_{y ~ p_theta(y|x)}[(grad_theta log p_theta(y|x))(grad_theta log p_theta(y|x))^T]` — the expected
outer product of the score, with the output `y` drawn from the model's *own* predictive distribution. The
load-bearing fact is that this expected outer product equals the expected Hessian of the negative log-
likelihood. Let me check it, because the whole argument leans on it. For one coordinate, write `p` for
`p_theta(y)`: the second derivative of `log p` is `(1/p) d^2p/dtheta^2 - (d log p/dtheta)^2`. Take the
expectation over `y ~ p`, i.e. multiply by `p` and sum over `y`:
`E[d^2 log p/dtheta^2] = sum_y d^2p/dtheta^2 - E[(d log p/dtheta)^2]`. The first piece is
`d^2/dtheta^2 (sum_y p) = d^2/dtheta^2 (1) = 0`, because probabilities sum to one regardless of `theta`. So
`E[-d^2 log p/dtheta^2] = E[(d log p/dtheta)^2]` — the expected Hessian of the NLL is the expected squared
score, the Fisher. That single line rescues me: under the model distribution the curvature is an *average
of squared first-order gradients*, no second derivatives. And the outer-product form is automatically PSD
— a sum of squares cannot be negative — whereas the raw empirical Hessian can be indefinite and would give
me negative "stiffnesses," meaningless as importances.

That PSD-by-construction is not a footnote; it is the second half of SI's cure. SI's running sum
`W = -sum g delta_theta` can go *negative* — a coordinate that on some steps moved with its gradient
(loss went up along it) contributes a positive `g delta_theta`, hence a negative `omega`, an importance
that says "hold this weight negatively," which is nonsense and is exactly why SI needed the `epsilon` floor
to keep the ratio from misbehaving. The Fisher, being a sum of squares, cannot be negative and needs no
floor; the estimator is well-posed by shape, not by a patched constant. So the Fisher is the usable
curvature: exact as an expected-likelihood identity, first-order computable, guaranteed PSD. It does cost
something SI did not — the separate post-context sweep of `200 * output.shape[1]` backward passes I priced
at rung one, 400 on the binary heads and 2000 on the ten-way ones per boundary — so I am deliberately
buying back the compute SI dodged in exchange for boundedness. Given that the failure was a runaway
magnitude and not a compute budget, that is a trade I take without hesitation.

I still have a full matrix `F`. In a million-dimensional weight space, storing even one full Fisher is
hopeless. Keep only the *diagonal* — treat off-diagonals as zero, asserting the posterior is a factorized
Gaussian, one independent quadratic per weight. It is lossy (weights surely covary in their effect on A),
but it is the price of linearity in the number of parameters, and the diagonal already carries the per-
weight curvature `F_i` I most need. With the diagonal-Gaussian summary, the A-term becomes
`sum_i 0.5 F_i (theta_i - theta*_{A,i})^2` up to a constant and a scalar. Assemble: minimizing
`-[log p(D_B|theta) + log p(theta|D_A)]` with the Laplace-Fisher approximation gives the loss I minimize
while training B, `L(theta) = L_B(theta) + sum_i (lambda/2) F_i (theta_i - theta*_{A,i})^2`. There is the
spring, derived rather than guessed: a quadratic anchor to `theta*_A` with per-coordinate stiffness `F_i`,
the diagonal Fisher. Important weights held nearly rigid, unimportant ones free for B. The `0.5` is the
Gaussian quadratic-form factor that fell out of the Taylor expansion — and note this is *why* this rung's
penalty carries the leading `0.5` that SI's no-half penalty deliberately dropped (SI absorbed the half via
the `Delta^2` normalization; here the half is the honest Gaussian coefficient). `lambda` trades A against
B: the clean derivation says the Fisher should be multiplied by A's sample size `N`, but that is exactly
what I must *not* take literally — `N` runs to the thousands here, and multiplying the spring by a
thousand would drown `L_B` entirely and freeze the net solid, which is not overconfidence in some abstract
sense but a concrete three-orders-of-magnitude mismatch between the derivation's nominal strength and a
usable one. The diagonal Laplace approximation is overconfident (it ignores the off-diagonal correlations
that would soften it), so rather than nail `lambda` to `N` I let it be the tunable knob — which in this
harness is the per-benchmark `reg_strength`. The scale is easy to sanity-check: the harness divides the
Fisher by `n_samples`, so each returned `F_i` is a per-example average, of order `p(1-p) <= 0.25` per
entry, while `L_B` is an order-one loss. For the spring to be *comparable* to the task loss the strength
wants to be roughly order one to order hundred, not order thousand — so the derivation's nominal `lambda =
N` overshoots a usable value by three to four orders of magnitude, and that gap is exactly why `lambda` has
to be a benchmark-level knob rather than a formula.

Make sure I can actually compute that diagonal Fisher, because the harness's default fill already does it
and I want to land exactly that. `F_i = E_x E_{y ~ p_theta(y|x)}[(d log p_theta(y|x)/dtheta_i)^2]` at
`theta*_A`. The expectation over `x` is an average over a sample of the finished context's inputs — a
hundred or two is plenty for a stable diagonal, and the harness caps it at 200 single-example passes. The
inner expectation over `y` is the subtle one, and I want it right: the Fisher averages the squared score
over `y` drawn from the *model's own* predictive distribution, NOT over the true labels. So per input I
run a forward pass to get the class distribution `p = softmax(logits)`; then for each class `k` I compute
the per-class NLL `-log p_theta(y=k|x)`, backprop it to get the gradient, square it coordinatewise, weight
by `p_k`, and sum over `k`. That weighted sum is the exact inner expectation (the expectation over a
categorical is the probability-weighted sum over outcomes). Average over the inputs and I have the diagonal
Fisher. The harness default does precisely this — `label_weights = softmax(output)`, a backward per class
weighted by `label_weights[0][label_index]`, in eval mode (so dropout/BN do not corrupt the estimate), one
example at a time, divided by the sample count. So my `estimate_importance` is essentially the *default
fill* — the EWC-shaped curvature the scaffold ships — and the penalty is the default `0.5 * sum F (theta -
theta*)^2`.

Before I trust the prediction that this under-weights Split-MNIST relative to SI, let me actually compute
the Fisher of a cleanly converged binary task and see how small it gets, because that is the whole worry.
Take the softmax over logits; the score of the per-class NLL with respect to logit `j` for target `k` is
`p_j - delta_{jk}`, so the diagonal Fisher of the logits is `F_jj = sum_k p_k (p_j - delta_{jk})^2 =
p_j - p_j^2 = p_j (1 - p_j)`. On a binary head that has solved its task confidently, say `p = [0.99, 0.01]`,
that gives `F = 0.99 * 0.01 = 0.0099` — a hundredth. The more confidently the task is solved, the closer
`p` is to one-hot, and `p(1-p) -> 0`: the endpoint Fisher of a well-separated binary classifier is nearly
zero, so its weights get held only loosely. That is the concrete mechanism behind my worry about Split-
MNIST: SI's trajectory importance accumulated real curvature *while the gradients were nonzero*, but the
endpoint Fisher, evaluated after the confident minimum is reached, has little curvature left to report, so
it under-protects exactly the cleanly solved short tasks SI held best. The number `0.0099` is why I expect
EWC to sit a hair under SI there rather than above it.

The one thing the loop does for me is cross-context accumulation: when context C arrives after A and B, it
sums my returned Fisher into `_custom_importance`, which is the right multi-context EWC because a sum of
quadratics is a quadratic, each anchored at its boundary snapshot with its stored Fisher. That matches the
EWC rule exactly — at the price of memory that grows with the number of contexts, and more to the point a
summed stiffness `sum_t F_t` that only ever grows, which I note now because it is the very thing the next
rung will have to attack. There is a wrinkle the harness forces me to be honest about. The canonical form
of this method keeps a *separate* Fisher and anchor per past context and sums explicit springs, each
anchored at its own `theta*_t`. This harness's loop instead sums my per-context Fisher returns into a
single `_custom_importance` buffer and re-snapshots `_custom_prev_params` to the *latest* boundary at each
context. So the penalty here anchors all the accumulated stiffness at the *most recent* optimum, not at
each context's own optimum. For two contexts these coincide; from the third on they differ, and that
difference is exactly what the next rung is built around — so for this EWC rung I take the harness at face
value: diagonal-Fisher importance summed by the loop, quadratic penalty anchored at the carried snapshot.

So the delta from SI is precise and surgical. Where SI read importance off the trajectory as
`W/(Delta^2 + epsilon)` — unbounded, growing, possibly-negative, scale-mismatched across contexts — EWC
reads it off the *endpoint* as a diagonal Fisher: a bounded, PSD, per-point curvature with no dependence on
how many steps a context took. Where SI dropped the half via `Delta^2`, EWC carries the honest Gaussian
`0.5`. The penalty shape is the same spring; only the stiffness changes, and that is the whole point — the
failure was the estimator, so I changed the estimator. Now the falsifiable expectations against SI's
measured line, benchmark by benchmark. On Split-MNIST I expect EWC to be *very good but probably slightly
below* SI's 0.9852: the endpoint Fisher, which I just watched fall to `~0.01` on a confidently solved
binary task, under-weights exactly the cleanly converged tasks SI's trajectory held perfectly — so I would
not be surprised to see EWC land around 0.95–0.97, a hair under SI here, and if anything it is one or two
tasks that the near-zero Fisher fails to protect that would drag the mean. The decisive test is Permuted-
MNIST. EWC's importance does not compound along the trajectory and is bounded per context, so it should
*not* exhibit SI's monotone slide to 0.14; I expect the per-context line to decay at most gently rather
than crash to the chance floor, and the average to jump well above SI's 0.4474 — into the low-to-mid 0.8s
if the bounded Fisher genuinely keeps the ten permutations alive. That single benchmark is where EWC should
win the aggregate, because SI's 0.6468 total collapse there is the biggest number on the board to recover.
On Split-CIFAR100 I expect a near-tie with SI in the low-to-mid 0.5s — the per-context line there was
difficulty-shaped, not forgetting-shaped, so the Fisher-vs-path-integral choice should barely register, and
a difference of a point or two either way would be consistent with the estimator simply not being the
bottleneck. If EWC's Permuted-MNIST does *not* clear SI's by a wide margin, my diagnosis of SI's collapse —
unbounded growing importance — was wrong, and the fix is something else; but if it does, the remaining
weakness is the one I already named: the loop sums Fishers without bound across contexts and re-anchors
only at the latest, which on a long sequence will itself start to over-constrain — the opening for the next
rung.
