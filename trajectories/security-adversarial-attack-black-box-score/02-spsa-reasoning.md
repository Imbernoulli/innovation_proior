The floor came back at `asr = 0.474` mean over 65 queries, and the shape of that result tells me what to
fix. On the easier pairs it worked — ResNet20/CIFAR-10 0.565, ResNet20/CIFAR-100 0.615,
MobileNetV2/CIFAR-100 0.590 — where the boundary is close enough that even unstructured noise stumbles
across it within 64 steps. On VGG11-BN it collapses: 0.255 on CIFAR-10, 0.300 on CIFAR-100, less than half
the ResNet number on the same dataset. And every row reads `avg_queries = 65`. That last fact is the whole
story: handed 1000 queries per sample, the floor spent 65 — it capped itself at `min(n_queries, 64)` steps
and walked away from ~94% of its allowance. The images that failed did not fail because the budget ran out;
they failed because the floor stopped trying. Two separable diseases: an *economic* one (quits at 65 when
1000 are available) and a *directional* one (isotropic interior nudges in ~3000 dimensions are almost
orthogonal to any descent direction). Curing the economic disease is trivial — run more iterations. The
directional disease is the real one.

Decomposing the table properly tells me *which* disease dominates where. By architecture: ResNet20
`(0.565 + 0.615)/2 = 0.590`, MobileNetV2 `0.555`, VGG11-BN `(0.255 + 0.300)/2 = 0.2775`. By dataset:
CIFAR-10 `0.447`, CIFAR-100 `0.502`. Two things jump out. CIFAR-100 is *higher* by `0.055`, not lower —
the "more classes is more targets for an untargeted flip" effect, so class count is not what hurts the
floor. And VGG11-BN sits at `0.2775` against `~0.57` for the other two backbones, a ratio of `0.47`;
within a fixed dataset the same gap holds (`0.255/0.565 = 0.45`, `0.300/0.615 = 0.49`). So the axis that
separates success from failure is *architecture*, not dataset: VGG11-BN presents a boundary that
`~1.4%`-aligned noise cannot reach in 64 steps, while ResNet and MobileNet boundaries are close enough that
blind luck lands some flips. That pins the target: rescue the VGG rows, and do it by *aiming*, because the
dataset decomposition already rules out "CIFAR-100 is just hard."

So the question sharpens: using only the forward queries the floor was throwing away, can I reconstruct an
actual *descent direction* instead of guessing one? PGD is the gold standard for `L_inf` attack strength —
step along the gradient of the correct-class advantage and project back — but it needs the gradient the
oracle refuses. The honest fix is to estimate the gradient from score queries. And *which* objective I
estimate the gradient of matters, because the floor's plain `f_y` hurt it. Cross-entropy saturates: once
the model is confident the softmax pins at 0 or 1 and the loss goes flat. On a correctly classified image
the correct logit typically leads by ~`5`, so `p_y ~ 0.993`, CE `= -log(0.993) ~ 0.007`, and CE's
sensitivity to the correct logit is `p_y - 1 ~ -0.007`. A probe that shifts logits by order `1` changes CE
by only `~0.007` — right down at any readout-noise floor, so a finite-difference estimate of CE's gradient
is garbage precisely where I start. The margin is linear in the logits: dropping `f_y` by one logit moves
`J = f_y - max_{k!=y} f_k` by about a full `1`, roughly `140x` more signal than CE at that confidence, and
the logits stay roughly linear far from the clean point so the margin keeps that slope to the boundary. So
I descend the **margin** `J(x) = f_y(x) - max_{k!=y} f_k(x)`: positive while `y` is on top, zero at
misclassification, usable slope long after cross-entropy has died.

How do I descend an objective I can only measure, never differentiate? The textbook move with only function
values is finite differences: `ghat_i = (J(x + c e_i) - J(x - c e_i)) / (2c)`, the partial derivative made
numerical, unbiased to `O(c^2)` because the symmetric difference cancels the even Taylor term. But this is
the floor's economic disease in a worse form: to assemble one full gradient I must probe every coordinate
at two queries each, `2D = 6144` queries — six times the whole budget for a *single* step. The right idea,
the wrong economics. I need an estimate whose cost does *not* grow with `D`.

Here is the trick. The reason coordinate-wise differencing costs `D` probes is that it isolates each
coordinate. Drop the isolation. Perturb *all* coordinates at once with a single random vector `v in R^D`
and take the two-sided difference along it: `df = J(x + c v) - J(x - c v)`. Taylor-expanding,
`J(x +/- c v) = J(x) +/- c (v . grad J) + (c^2/2) v^T H v +/- O(c^3)`, so `df = 2c (v . grad J) + O(c^3)` —
the quadratic terms cancel. That scalar is the directional derivative along `v`, mixing all `D` partials.
To pull coordinate `i` back out, divide by `v_i`: `ghat_i = df / (2c v_i) = partial_i J + sum_{j != i}
(v_j / v_i) partial_j J + O(c^2)`. The `j = i` term gives exactly `partial_i J`; every other partial appears
carried by a *random* coefficient `v_j / v_i`. This is only useful if that cross-talk averages to zero: if
`v` has independent, mean-zero components, then for `j != i`, `E[v_j / v_i] = E[v_j] E[1/v_i] = 0`,
*provided `E[1/v_i]` is finite*. So `E[ghat_i] = partial_i J + O(c^2)` — almost unbiased, from exactly
**two** function evaluations, regardless of `D`. That is the economics the floor needed.

The finiteness caveat decides everything. The instinctive `v ~ N(0, I)` *fails* it: the Gaussian density
is bounded away from zero near `v_i = 0` while `1/v_i` blows up there, so the inverse moment diverges and
the cross-talk has no usable mean — in practice, occasional enormous estimates whenever some `v_i` lands
near zero. The cure is a mean-zero distribution whose components stay away from zero: the symmetric
Bernoulli (Rademacher), `v_i in {+1, -1}` each with probability 1/2. Mean zero, and since `v_i = +/- 1`,
`1/v_i = v_i` is bounded, so the cross-talk truly cancels — and *dividing* by `v_i` equals *multiplying* by
it, so `ghat = (df / 2c) . v` needs no per-coordinate division. Rademacher is forced by the
finite-inverse-moment condition, not aesthetic. This is the simultaneous-perturbation gradient estimate,
SPSA.

How bad is one estimate? For a single Rademacher probe the cross-talk `sum_{j != i} (v_j/v_i) partial_j J`
is a random-sign sum of the other `D - 1` partials, variance `~ ||grad J||^2`. So a single estimate carries
per-coordinate noise of order `||grad J||` on a signal `partial_i J ~ ||grad J|| / sqrt(D) = ||grad J|| /
55` — SNR about `1/55`, hopeless coordinate by coordinate. The cure is averaging: draw `n` independent
Rademacher vectors, form `n` two-query estimates, average `ghat_bar = (1/n) sum (df_i / 2c) v_i`. Variance
falls like `1/n`; at `n = 128` the per-coordinate noise drops to `||grad J|| / 11.3` and the SNR rises to
about `(1/55)/(1/11.3) = 0.20` — *still* noise-dominated per coordinate. That is the crucial thing: I never
get an accurate per-pixel gradient even after 128 probes. What I do get is a vector whose inner product with
the true gradient is right on average (`E<ghat_bar, grad J> = ||grad J||^2 + O(c^2)`, cross-talk mean
zero), so the *direction* is usable in aggregate even though each coordinate is garbage. That distinction —
usable direction, unusable per-coordinate values — is what a noise-robust optimizer has to exploit. This
costs `2n` queries per step, and those `2n` forward passes are one batch. In the fill `n = nb_sample = 128`,
so a step costs `256` queries, evaluated in chunks of `max_batch_size = 64`.

Now I have a noisy gradient and I am in stochastic-approximation territory: descend and project. The plain
update `x' = x - a . ghat_bar` works, but `ghat_bar` is noisy and its coordinates are unevenly scaled —
some pixels have large derivatives, some tiny, while the noise floor is roughly uniform. A single global `a`
is the wrong tool, the same per-coordinate-scaling problem that plagues noisy network training. The fix is
Adam: keep per-coordinate EMAs of the estimate and its square and step by `mhat / (sqrt(vhat) + eps)`. The
first moment smooths noise across steps; the `1/sqrt(vhat)` rescaling gives each pixel its own effective
step so a few large-derivative pixels do not dominate. The fill runs Adam on the perturbation at
`lr = 0.01`. One honest tension: Adam's first-moment smoothing is *across* steps, and I only get three
steps, so the running average never accumulates much history — most of the denoising has to come from the
*within*-step `nb_sample = 128` averaging, and Adam contributes mainly its per-coordinate rescaling rather
than its momentum. That is fine — the rescaling is the part that matters for a `0.20`-SNR estimate with
wildly uneven coordinate scales — but it tells me the within-step average is the workhorse and the step
count is the resource I am genuinely short of.

I keep the variable as the *perturbation* `dx` rather than the image, because the constraint lives on `dx`.
After each Adam step I return to the feasible set: clamp `dx` to `[-eps, eps]`, then clamp `x0 + dx` to
`[0,1]` and fold back into `dx`. Because the constraints are separable across coordinates, this pair of
clamps *is* the exact Euclidean projection onto the feasible box — each coordinate is projected onto its own
interval `dx_i in [max(-eps, -x0_i), min(eps, 1 - x0_i)]`, which is closed-form clamping, no iteration.
That the constraint set is a box is what makes the projection free; an `L_2` ball would need rescaling. And
it is precisely what the harness re-checks on return, so getting it right keeps a sample from being scored a
failure on a validity violation rather than on the model's prediction.

The probe radius `c = delta = 0.01` and step `lr = 0.01` set the bias-noise trade. Smaller `c` means
smaller Taylor bias (`O(c^2)`), but the signal in `df = 2c(v . grad J)` scales with `c` while the model's
query-noise floor is fixed, so as `c -> 0` the per-probe SNR degrades like `c`. There is a sweet spot:
small enough that the directional-derivative approximation is faithful, large enough that `df` rises above
the noise floor. `0.01` on `[0,1]`-scaled pixels sits in that window — well inside the `eps = 0.031` ball,
big enough for a measurable `df`. `lr = 0.01` is modest so a noisy estimate does not yank the iterate out
of the productive region in one step.

`nb_sample = 128` is not a free knob — it trades directly against descent steps, `nb_iter = n_queries //
(2 * nb_sample)`. At `nb_sample = 32` each step costs 64 queries and I get `1000 // 64 = 15` steps, but the
per-coordinate SNR drops to `sqrt(32)/55 = 0.10`, half as clean — fifteen very noisy steps. At
`nb_sample = 512` each step costs 1024, already over budget, so `max(1, .)` leaves a single very clean step
with no ability to correct. At `128` I land in between: `256` per step, `1000 // 256 = 3` steps at the
`0.20`-SNR direction — uncomfortably few, but each step clean enough to make real progress, whereas fifteen
`0.10`-SNR steps risk wandering. So `128` is a defensible interior choice, and its obvious weakness — that a
"handful" is literally three — is what I will watch on the hardest boundaries.

Now the economics for this harness, where the floor bled out. A step costs `2 * nb_sample` queries, so the
fill sets `nb_iter = max(1, n_queries // (2 * nb_sample))` — it spends *exactly* the budget rather than
capping at 64. With 1000 and `nb_sample = 128` that is 3 Adam steps, and the query counter reads out near
the full budget: `avg_queries = 3 * 256 = 768` against the floor's 65. That is the deliberate inversion of
the floor's economic disease. But it exposes the SPSA bargain: SPSA pays `2n` queries for *one* descent
step, so even spending the whole allowance it gets only a handful of steps, and the per-step query tax caps
how many corrections it can make.

So the falsifiable expectations against the floor's numbers. SPSA should beat `random_search` everywhere,
because it descends an estimated gradient of the margin instead of guessing isotropic directions, and it
spends ~768 queries instead of 65 (`3 * 256`, a-priori). The dramatic gains should land where the floor
collapsed — the VGG11-BN pairs at 0.255 and 0.300: if that failure was directional, a real descent
direction should roughly double them. The ResNet pairs, already the floor's best, should saturate high. The
headline cost is the regression from `avg_queries = 65` to ~768, because the gradient estimate is expensive
and only three steps fit. If SPSA does *not* clear the floor on the VGG pairs, my diagnosis was wrong — the
failure would be a budget-or-projection artifact, not directional — and I expect the residual weak spot to
be exactly those VGG rows where three noisy steps may still be too few. The distilled module calling
`torchattacks.SPSA` is in the answer.
