The floor came back at `asr = 0.474` mean over 65 queries, and the shape of that result tells me exactly
what to fix. Read it scenario by scenario. On the easier pairs it actually worked — ResNet20/CIFAR-10
0.565, ResNet20/CIFAR-100 0.615, MobileNetV2/CIFAR-100 0.590 — where the boundary is close enough that
even unstructured noise stumbles across it within 64 timid steps. But on VGG11-BN it collapses:
VGG11-BN/CIFAR-10 0.255, VGG11-BN/CIFAR-100 0.300, less than half the ResNet number on the same dataset.
And every single row reads `avg_queries = 65`. That last fact is the whole story. The floor was handed a
budget of 1000 queries per sample and it spent 65 — it capped itself at `min(n_queries, 64)` candidate
steps and walked away from ~94% of its allowance. The images that failed did not fail because the budget
ran out; they failed because the floor stopped trying. So I have two separable diseases: an *economic*
one (the search quits at 65 queries when 1000 are available) and a *directional* one (isotropic interior
nudges in ~3000 dimensions are almost orthogonal to any descent direction, so even the steps it does take
crawl, and on the harder VGG boundary they crawl to nothing). Curing the economic disease is trivial —
run more iterations. The directional disease is the real one, and the floor's failure is a clean
statement of it: random search with no notion of *which way* to move wastes almost every proposal.

Before I move on, let me decompose that table properly, because the way the six numbers split tells me
*which* disease dominates where, and it confirms the guess I went in with. Average by architecture: ResNet20
`(0.565 + 0.615)/2 = 0.590`, MobileNetV2 `(0.520 + 0.590)/2 = 0.555`, VGG11-BN `(0.255 + 0.300)/2 = 0.2775`.
Average by dataset: CIFAR-10 `(0.565 + 0.255 + 0.520)/3 = 0.447`, CIFAR-100 `(0.615 + 0.300 + 0.590)/3 =
0.502`. Two things jump out. First, CIFAR-100 is *higher* than CIFAR-10 by `0.055`, not lower — exactly
the "more classes is more targets for an untargeted flip" effect, so the class count is not what is
hurting the floor. Second, VGG11-BN sits at `0.2775` against `~0.57` for the other two backbones, a ratio
of `0.2775/0.59 = 0.47` — it is scoring under half. Within a fixed dataset the same gap holds:
`0.255/0.565 = 0.45` on CIFAR-10, `0.300/0.615 = 0.49` on CIFAR-100. So the axis that separates success
from failure is *architecture*, not dataset: VGG11-BN presents a boundary that `~1.4%`-aligned noise
simply cannot reach in 64 steps, while ResNet and MobileNet boundaries are close enough that blind luck
lands some flips. That pins the target for this rung — whatever I build has to rescue the VGG rows, and it
has to do it by *aiming*, because the dataset decomposition already rules out "CIFAR-100 is just hard" as
the explanation.

So the question for this rung sharpens to: can I, using only the forward queries the floor was throwing
away, reconstruct an actual *descent direction* instead of guessing one? That is the one black-box
primitive the floor refused to use. PGD is the gold standard for `L_inf` attack strength — step along the
gradient of the correct-class advantage and project back into the box — but it needs the gradient, and
the oracle gives me none. The honest fix is to estimate the gradient from score queries. And I want to be
careful about *which* objective I estimate the gradient of, because the floor's choice of plain `f_y`
quietly hurt it too. Cross-entropy saturates — once the model is confident the softmax pins at 0 or 1 and
the loss goes flat, so its value barely moves as I perturb the input. Put a number on how bad that is for
a finite-difference method. On a correctly classified image the model is typically confident: say the
correct logit leads the field by `5`, so `p_y = 1/(1 + e^{-5} + ...) ~ 0.993` and cross-entropy is
`-log(0.993) ~ 0.007`. The sensitivity of CE to the correct logit there is `p_y - 1 ~ -0.007`, so a probe
that shifts logits by an order-`1` amount changes CE by only `~0.007` — right down at any readout noise
floor, which means `df` for CE is essentially noise and the estimated gradient of CE is garbage precisely
where I start. The margin, by contrast, is linear in the logits: dropping `f_y` by one logit moves
`J = f_y - max_{k!=y} f_k` by about a full `1`, roughly `140x` more signal than CE at that confidence. The
logits stay roughly linear far from the clean point, so the margin keeps that slope all the way to the
boundary. So I will descend the **margin** `J(x) = f_y(x) - max_{k!=y} f_k(x)`: it is positive while `y`
is on top, crosses zero exactly at misclassification, and keeps a usable slope long after cross-entropy has
died. I want that slope, because all I will ever have is differences in `J`, and I just showed those
differences are `~140x` larger for the margin than for the loss the confident model wants to flatten.

Now, how do I descend an objective I can only measure, never differentiate? The textbook move when you
have only function values is finite differences: for each coordinate `i`, probe to each side and
difference, `ghat_i = (J(x + c e_i) - J(x - c e_i)) / (2c)`, which is just the partial derivative made
numerical, unbiased to `O(c^2)` because the symmetric difference cancels the even Taylor term. But this
is exactly the floor's economic disease in a worse form. To assemble one full gradient I must probe every
coordinate of the image — `D = C*H*W`, which is 3072 for a CIFAR image — at two queries each, so one
gradient costs `2D = 6144` queries. That blows the entire 1000 budget six times over for a *single* step.
Coordinate-wise differencing is the right idea and the wrong economics. The real requirement is an
estimate whose cost does *not* grow with `D`.

Here is the trick that buys that. The reason FDSA costs `D` probes is that it insists on isolating each
coordinate — perturb along `e_i` alone so the difference reports purely on `partial_i J`. Drop the
isolation. Perturb *all* coordinates at once with a single random vector `v in R^D` and take the
two-sided difference along it: `df = J(x + c v) - J(x - c v)`. Taylor-expanding,
`J(x +/- c v) = J(x) +/- c (v . grad J) + (c^2/2) v^T H v +/- O(c^3)`, so the symmetric difference is
`df = 2c (v . grad J) + O(c^3)` — the quadratic terms cancel. That single scalar is the directional
derivative along `v`, mixing all `D` partials. To pull coordinate `i` back out, divide by `v_i`:
`ghat_i = df / (2c v_i) = partial_i J + sum_{j != i} (v_j / v_i) partial_j J + O(c^2)`. The `j = i` term
gives exactly `partial_i J`; every other partial appears too, but carried by a *random* coefficient
`v_j / v_i`. This is only useful if that cross-talk averages to zero. If `v` has independent, mean-zero
components, then for `j != i`, `E[v_j / v_i] = E[v_j] E[1/v_i] = 0`, *provided `E[1/v_i]` is finite*. So
`E[ghat_i] = partial_i J + O(c^2)` — almost unbiased, from exactly **two** function evaluations,
regardless of `D`. That is the economics the floor needed: two queries buy a full noisy gradient in any
dimension.

The finiteness caveat decides everything, so I cannot wave it past. I need `E[1/v_i]` finite. The
instinctive choice `v ~ N(0, I)` *fails* it — the Gaussian density is bounded away from zero near
`v_i = 0` while `1/v_i` blows up there, so the inverse moment diverges and the cross-talk has no usable
mean; in practice this shows up as occasional enormous estimates whenever some `v_i` lands near zero and
I divide by it. The cure is a mean-zero distribution whose components stay *away* from zero. The clean
one is the symmetric Bernoulli (Rademacher), `v_i in {+1, -1}` each with probability 1/2: mean zero, and
since `v_i = +/- 1`, `1/v_i = v_i`, bounded, finite moments of every order, so the cross-talk truly
cancels. And a bonus the implementation uses: because `v_i = +/-1`, *dividing* by `v_i` equals
*multiplying* by it, so `ghat = (df / 2c) . v` needs no per-coordinate division at all. The Rademacher
choice is not aesthetic; it is forced by the finite-inverse-moment condition. This is the simultaneous-
perturbation gradient estimate — SPSA.

Let me size the cross-talk to see how bad one estimate is, because the number decides how much averaging I
need. For a single Rademacher probe, `ghat_i = partial_i J + sum_{j != i} (v_j/v_i) partial_j J`, and with
`v_i = +/-1` each ratio `v_j/v_i` is itself `+/-1` with mean zero, so the cross-talk is a random-sign sum
of the other `D - 1` partials. Its variance is `sum_{j != i} (partial_j J)^2 = ||grad J||^2 -
(partial_i J)^2 ~ ||grad J||^2`. So a *single* estimate carries per-coordinate noise of order `||grad J||`
sitting on top of a signal `partial_i J`, which for a roughly isotropic gradient is only about
`||grad J|| / sqrt(D) = ||grad J|| / 55`. The single-probe per-coordinate signal-to-noise is thus about
`1/55` — hopeless coordinate by coordinate. That is exactly why I cannot use one probe.

One simultaneous estimate is correct on average but jittery: each single random-direction estimate points
partly sideways because of the surviving cross-talk, and any query noise adds to that. The standard cure is
the same averaging the asymptotics rely on, done *within* a step: draw `n` independent Rademacher vectors,
form `n` independent two-query estimates, and average, `ghat_bar = (1/n) sum (df_i / 2c) v_i`. Variance
falls like `1/n` and the cross-talk is knocked toward its zero mean. With `n = 128` the per-coordinate
noise drops to `||grad J|| / sqrt(128) = ||grad J|| / 11.3`, so the per-coordinate SNR rises to about
`(1/55)/(1/11.3) = 0.20` — *still* noise-dominated coordinate by coordinate. This is the crucial thing to
understand about the estimate and it sets up why I need Adam: I never get an accurate per-pixel gradient,
even after averaging 128 probes. What I *do* get is a vector whose inner product with the true gradient is
right on average — `E<ghat_bar, grad J> = ||grad J||^2 + O(c^2)` while the cross-talk contributes zero mean
— so the *direction* is usable in aggregate even though each coordinate is garbage. That distinction
(usable direction, unusable per-coordinate values) is what a noise-robust optimizer has to exploit. This
costs `2n` queries per step, and — crucially for the GPU and for the oracle's batch-counting — those `2n`
forward passes are one batch. In this task's fill `n = nb_sample = 128`, so a step costs `2 * 128 = 256`
queries, evaluated in chunks of `max_batch_size = 64`. That `n` is the dial between a clean, reliable
direction (large `n`, strong but expensive) and a cheap noisy one (small `n`); at `128` I have bought a
direction that is right on average but still `0.2`-SNR per pixel, so the optimizer must do the rest.

Now I have a noisy gradient and I am back in stochastic-approximation territory: descend with it and
project. The plain update `x' = x - a . ghat_bar` works, but `ghat_bar` is noisy and its coordinates are
unevenly scaled — some pixels have large derivatives, some tiny, while the noise floor is roughly uniform
across them. A single global step `a` is the wrong tool for that, the same per-coordinate-scaling problem
that plagues noisy network training. The fix there is Adam: keep per-coordinate exponential moving
averages of the estimate and of its square and step by `mhat / (sqrt(vhat) + eps)`. The first moment
smooths noise across steps (a second layer of averaging on top of the within-step averaging); the
`1/sqrt(vhat)` rescaling gives each pixel its own effective step so a few large-derivative pixels do not
dominate. The estimator is unbiased enough that Adam cannot tell it from a real gradient, and Adam's
robustness to noisy, unevenly-scaled gradients is exactly what this directional estimate needs. The fill
runs Adam on the perturbation at `lr = 0.01`. I do have to be honest about one tension this exposes: Adam's
first-moment smoothing is *across* steps, and I only get three steps, so the running average never
accumulates much history — the cross-step denoising that would normally compound over dozens of iterations
here has almost nothing to average. Most of the denoising therefore has to come from the *within*-step
`nb_sample = 128` averaging, and Adam is contributing mainly its per-coordinate `1/sqrt(vhat)` rescaling
rather than its momentum. That is fine — the rescaling is the part that matters for a `0.20`-SNR estimate
with wildly uneven coordinate scales — but it tells me the real workhorse here is the within-step average,
and the step count is the resource I am genuinely short of.

I keep the variable as the *perturbation* `dx` rather than the image, because the constraint lives on
`dx`. After each Adam step I must return to the feasible set the harness enforces: `||dx||_inf <= eps`,
and `x0 + dx in [0,1]`. Both are simple clamps — clamp `dx` to `[-eps, eps]` (Euclidean projection onto
the `L_inf` box, since a box projects coordinate-wise), then clamp `x0 + dx` to `[0,1]` and fold the
result back into `dx`. This is precisely the projection step the harness will re-check on return, so
getting it right here is what keeps the sample from being scored a failure on a validity violation rather
than on the model's prediction. It is worth confirming that the two clamps really are the exact Euclidean
projection and not an approximation, because after a noisy step the iterate can land well outside the
feasible set and I need the return to the set to be optimal, not merely feasible. The feasible set for
coordinate `i` is the intersection of `|dx_i| <= eps` with `0 <= x0_i + dx_i <= 1`, i.e. the interval
`dx_i in [max(-eps, -x0_i), min(eps, 1 - x0_i)]`. Because the constraints are separable across
coordinates, the projection of the whole vector decomposes into `D` independent one-dimensional
projections, and projecting a scalar onto an interval is just clamping to its endpoints. So clamp-to-`eps`
then clamp-to-`[0,1]` computes exactly that per-coordinate interval clamp — the true `L_2` projection onto
the intersection, in closed form, no iteration. That the constraint set is a box is what makes the projection
free; if it were an `L_2` ball I would need a rescaling, and if it were a general polytope I would need to
iterate. The `L_inf` geometry hands me the projection for the price of two clamps.

The probe radius `c = delta = 0.01` and the step `lr = 0.01` need a word, because they set the bias-noise
trade. Smaller `c` means smaller Taylor bias (`O(c^2)`), but the *signal* in `df = 2c(v . grad J)` scales
with `c` while the model's own query-noise floor is fixed, so as `c -> 0` the per-probe signal-to-noise
*degrades* like `c`. There is a sweet spot: small enough that the directional-derivative approximation is
faithful, large enough that `df` rises above the noise floor. `0.01` on `[0,1]`-scaled pixels sits in
that window — well inside the `eps = 8/255 ~ 0.031` ball, and big enough to produce a measurable `df`.
The Adam `lr = 0.01` is likewise modest so the noisy estimate does not yank the iterate out of the
productive region in one step.

That `nb_sample = 128` is not a free knob — it trades directly against the number of descent steps, and I
should price the alternatives instead of accepting `128` by default. The budget buys `nb_iter =
n_queries // (2 * nb_sample)` steps. At `nb_sample = 32` each step costs `64` queries and I get `1000 //
64 = 15` steps, but the per-coordinate SNR of the direction drops to `sqrt(32)/55 = 0.10`, half as clean —
fifteen very noisy steps. At `nb_sample = 512` each step costs `1024`, which already exceeds the `1000`
budget, so `1000 // 1024 = 0` and the `max(1, .)` floor leaves me a single, very clean step — one shot,
no ability to correct. At `nb_sample = 128` I land in between: `2 * 128 = 256` per step, `1000 // 256 = 3`
steps at the `0.20`-SNR direction I computed above. Three is uncomfortably few, but the direction is clean
enough that each step makes real progress, whereas fifteen `0.10`-SNR steps risk wandering. So `128` is a
defensible interior choice — enough probes to make each of a handful of steps trustworthy — and its
obvious weakness, that a "handful" is literally three, is the thing I will have to watch on the hardest
boundaries.

Now the economics for this harness, where the floor bled out. The budget is `n_queries` per sample; a
step costs `2 * nb_sample` queries; so the fill sets `nb_iter = max(1, n_queries // (2 * nb_sample))` —
it spends *exactly* the budget rather than capping at 64. With `n_queries = 1000` and `nb_sample = 128`
that is `1000 // 256 = 3` Adam steps; the oracle's query counter will read out near the full budget,
which is why I should expect `avg_queries = 768` (3 steps * 256) against the floor's 65. That is the
deliberate inversion of the floor's economic disease: where random search quit early, SPSA spends the
whole allowance reconstructing a real direction. And note the tension this exposes — SPSA pays `2n`
queries for *one* descent step, so even spending the full budget it gets only a handful of steps. That is
the SPSA bargain: a genuine, masking-immune descent direction, but an expensive one, and the per-step
query tax caps how many corrections it can make.

Let me verify the whole construction against limiting cases before I trust it, rather than assert it works.
Take `n -> infinity` at fixed `c`: the cross-talk term averages exactly to zero and `ghat_bar ->
(1/2c)(J(x + cv) - J(x - cv)) v` in expectation resolves to `grad J + O(c^2)`, the true gradient up to the
Taylor bias — so the estimator is consistent, which is the property Adam needs to be fed something it can
treat as a gradient. Take `c -> 0` at fixed `n`: the Taylor bias `O(c^2)` vanishes, but the measured
difference `df = 2c (v . grad J) + O(c^3)` shrinks *linearly* in `c` while any fixed additive readout noise
does not, so the signal-to-noise of each probe degrades like `c` — the two limits pull opposite ways and
force an interior optimum for `c`, which is why `c = 0.01` is a chosen value and not "as small as
possible." A units check closes it: `df` is a difference of margins (logit units), dividing by `2c` (pixel
units) gives logit-per-pixel, and multiplying by the dimensionless `v` keeps those units, so `ghat_bar` has
the units of `d J / d x` — a genuine gradient, dimensionally, feedable straight into Adam whose update is
scale-free anyway. And the degenerate case `D = 1` recovers the plain symmetric finite difference
`(J(x+c) - J(x-c))/2c` with no cross-talk to cancel — the same estimator the coordinate method would use,
which is the right consistency check that simultaneous perturbation is a strict generalization, not a
different animal.

So the falsifiable expectations against the floor's numbers. SPSA should beat `random_search` everywhere,
because it descends an estimated gradient of the margin instead of guessing isotropic directions, and
because it spends ~768 queries instead of 65. I expect the mean `asr` to jump from 0.474 well into the
0.8s. The two scenarios that should improve *most dramatically* are the ones the floor collapsed on — the
VGG11-BN pairs (0.255 and 0.300): if the floor's VGG failure was a directional failure, a real descent
direction should roughly double those, lifting both into the 0.6 range. The ResNet20 pairs, already the
floor's best, should saturate toward the low-to-mid 0.9s. The cost is the headline regression: where the
floor read `avg_queries = 65`, SPSA will read ~768, because the gradient estimate is expensive and only a
few steps fit in the budget. If SPSA does *not* clear the floor on the VGG pairs, my diagnosis was wrong
— the failure would not be directional but a budget-or-projection artifact — but I expect it to, and I
expect the residual weak spot to be exactly those VGG/CIFAR-100 rows where 3 noisy steps may still be too
few. The distilled module — the literal scaffold edit calling `torchattacks.SPSA` — is in the answer.
