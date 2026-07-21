MAGIC reached a held-out combined of ~`0.85`, and the prediction landed the way I argued: the gain
came almost entirely through the MSE term (~`0.73` → `0.82`) while the Poisson term barely moved
(~`0.87` → `0.87`). The PCA-denoised embedding and the adaptive-bandwidth soft kernel recovered the
log-normalized shape hard uniform pooling had smeared, and the `t = 2` peak with a monotone decline
confirmed the diffusion lever behaves the way the spectral picture said. So I keep the graph and the
diffusion as the backbone. The question is what is left, and the situation has flipped. After MAGIC the
MSE term at `0.82` leaves an absolute gap to perfect of only ~`(1 − 0.82)·0.55 ≈ 0.10` — the cheap,
narrow headroom is nearly spent. The Poisson term at `0.87` leaves ~`(1 − 0.87)·2.98 ≈ 0.39` —
expensive, wide, and now by far the larger reservoir of unrecovered score. So whatever I build next has
to earn most of its keep on the Poisson term, which means getting the absolute per-gene *rate* right,
not just the log-normalized shape, and doing so especially at the dropout entries where the likelihood
bleeds hardest.

That reframes the gaps MAGIC named into a targeted attack. First, I forced every gene through the same
square-root transform, whose variance stabilization is non-monotone across the dropout spectrum, so a
gene expressed almost everywhere and one that drops out nine times in ten go through a nonlinearity that
suits neither. Second, every diffusion so far ran in count or square-root space and never touched the
log-normalized space the MSE is computed in. Third, the diffusion captures only the *local* manifold
geometry, never the *global* low-rank structure underneath it. And fourth, from the last rung: the
diffusion fills every exact zero the same way, dropout or genuine biological zero, because nothing
distinguishes them. I want to attack all four while keeping MAGIC's adaptive-bandwidth diffusion graph
as the backbone, assembling the pieces — a multi-transform ensemble with low-rank and log-space
refinement — in plain numpy/scipy/sklearn so it runs anywhere rather than leaning on a specialized graph
library.

The headroom reading says where to spend first, and the obvious move is wrong. The most direct thing I
could bolt on is a smoothing pass in the log-normalized scoring space — one gap, easy, attacking the
exact space the MSE is read in. But the MSE headroom is nearly spent: only ~`0.10` absolute remains, so
even a perfect log-space pass moves the combined score by at most a couple of hundredths through that
term. The ~`0.39` on the Poisson side can only be reached by fixing the per-gene *rate*, which the
transforms, the zero-imputation, and the low-rank blend move — none of the count-space or log-space
smoothing touches it. So I lead with the transforms even though they are the most work, and treat the
log-space passes as the necessary defence of the MSE rather than the main event.

The transforms target the per-gene mis-stabilization directly. The square root is one
variance-stabilizing transform for Poisson data, not uniformly best. The Anscombe transform
`2√(x + 3/8)` is the classical low-count correction — the `3/8` shift sharpens stabilization where
counts are small — and the Freeman–Tukey transform `√x + √(x+1)` is built for the regime that is almost
all zeros and ones, the deep-dropout end. Each suits a different part of the spectrum: Anscombe for
well-expressed genes, Freeman–Tukey for high-dropout genes, the plain square root as a bridge. Rather
than pick one and re-inherit the single-transform flaw, I run the whole MAGIC-style pipeline three
times, once under each transform, and ensemble the outputs gene-wise, weighting each gene toward the
transform its dropout calls for. The weighting is a smooth interpolation of the dropout fraction `p_g`
via the binomial split: `w_ans ∝ (1 − p_g)²`, `w_ft ∝ p_g²`, `w_sqrt ∝ 2 p_g (1 − p_g)`, normalized —
the term-by-term expansion of `((1 − p_g) + p_g)² = 1`, so the three weights are a genuine partition of
unity at every dropout level. At `p_g = 0` all weight is Anscombe; at `p_g = 1` all Freeman–Tukey; the
bridge weight peaks at `p_g = 0.5`, the crossover where neither specialist is clearly right. At the
typical dropout ~`0.59` the weights land ~`0.17 / 0.35 / 0.48`, the bridge and deep-dropout transform
carrying most of it — so the ensemble genuinely moves mass across the transforms as dropout rises rather
than nominally picking one. This gene-adaptive multi-VST ensemble is the single biggest structural
change from MAGIC, aimed at the Poisson term: getting the per-gene stabilization right is getting the
per-gene *rate* right, which is what the likelihood reads.

A hard per-gene assignment — pick the single best transform for each gene by a dropout threshold — I
reject for the same reason I rejected the hard `k` and the fixed bandwidth: it puts a cliff into the
gene axis, so genes at dropout `0.49` and `0.51` land on slightly different scales, and since the
Poisson term reads the ensemble gene by gene, that discontinuity shows up as avoidable noise in the
metric. The soft blend has no such seam and costs nothing extra since I run all three passes anyway.
There is a mechanism reason the transform choice bears on the Poisson term specifically. The diffusion
averages cells in the transformed space and then inverts, so the recovered rate is `f⁻¹` of an average
of `f`-transformed counts; when `f` is nonlinear and the averaged cells' rates differ even slightly,
that composition is biased by the curvature of `f` at the relevant count level — a Jensen-type effect. A
transform whose curvature is well-matched to a gene's count regime keeps that bias small; a mismatched
one inflates it. Because the Poisson NLL reads the absolute rate directly, that curvature bias lands
squarely on the Poisson term, whereas the log-normalized MSE, caring about relative shape after its own
`log1p`, is far less sensitive. So choosing the transform per gene is the most direct handle on the bias
of the recovered rate — the quantity with all the remaining headroom.

The graph I build once, on the Anscombe embedding, and reuse across all three passes, because the
neighbour geometry is a property of the cells, not of the transform, and rebuilding it three times would
only add noise and cost. I mix a self-loop into the transition matrix, replacing `P` with a
renormalized `½ I + ½ P`, so each cell retains half its own mass at every step — because I am about to
apply the diffusion far more aggressively than MAGIC (imputation, multi-scale sums, log-space passes),
and without the self-loop those repeated applications would race toward consensus.

The fourth gap, indiscriminate zero-filling, I fix before the main diffusion in each pass. A hard zero
in a high-dropout gene is far more likely a dropout than biology, so I impute the exact zeros
specifically: take a couple of diffusion steps for a neighbour average, and fill *only* the zero entries
with what the manifold says should be there, leaving observed non-zeros alone, then renormalize. This
fills what is missing without over-writing what was measured — again pointed at the Poisson term, since
the filled holes are exactly the `X_train = 0`, `X_test > 0` entries where the raw likelihood bled
worst.

Inside each pass I want to be smarter than a single power `Pᵗ`, and here I have to be careful, because
MAGIC's global optimum was `t = 2` with larger `t` losing score monotonically. But that was one global
depth applied to *every* gene, and the genes do not want the same depth: a high-dropout gene has sparse,
noisy observations and wants deep pooling to fill and stabilize it, while a low-dropout gene is already
well-measured and wants almost none. So instead of a single power I use a gene-wise weighted multi-scale
diffusion, accumulating `X, PX, P²X, …, PᵗX` with per-gene decaying weights and a guaranteed baseline:
`X̂ = (Σᵢ bᵍᵢ Pⁱ X) / (Σᵢ bᵍᵢ)` with `b_g = 0.9 (0.2 + 0.8 p_g)`. Letting `t` run out to seven looks
like it contradicts the `t = 2` finding, so what matters is the *effective* depth the weighting
produces: for a gene that never drops out `b_g ≈ 0.18` and the weighted-mean power is about `0.2` steps
— essentially no smoothing; at the typical dropout ~`0.59` the effective depth is ~`1.4` steps, at
`p_g = 1` about `3`. So `t = 7` is a ceiling almost no gene reaches, not a flat depth — the high-dropout
genes land near the two-to-three-step depth MAGIC found optimal globally, while the low-dropout genes
are smoothed far less than a flat `t = 2` would have. A flat `Pᵗ` at `t = 7` would over-diffuse exactly
as MAGIC showed; this weighted sum is a different operator, giving each gene the depth its dropout
warrants. The baseline term guarantees every gene keeps at least some of its own diffused-once signal so
nothing is smoothed to nothing.

After diffusing I still guard against over-smoothing the genes that did not need it, blending the
diffused signal back toward the raw normalized signal per gene: `w_g = clip(p_g · vred_g · (1 − μ̃_g),
0, 0.7)`, where `vred_g` is the fraction of variance the diffusion removed for that gene and `μ̃_g` its
min-max-scaled mean. I lean on the diffused version only where all three signals agree smoothing is
appropriate — the gene drops out often, the diffusion removed a lot of variance, and the mean is low. A
well-expressed, low-dropout gene, where the diffusion mostly destroyed real per-gene variation, gets a
weight near zero and keeps its raw values; the `0.7` cap keeps even the most diffusion-friendly gene at
least thirty percent raw. This adaptive blend is the safety valve against the aggressive multi-scale
diffusion reaching too far on genes that did not want it.

On top of the ensemble sit the two global refinements. The first is low-rank, which I reconcile with my
explicit refusal at the last rung to truncate the diffused output to the PCA rank. That refusal stands —
this is not a truncation but a light *blend*. The cells live on a low-dimensional manifold, so the
denoised matrix should be approximately low-rank in the gene directions, and a truncated SVD captures
the global structure the purely local diffusion misses. I reconstruct from the top components and blend
it back in with a small weight, ~`0.10` — small for exactly the reason I refused the hard truncation: an
aggressive low-rank pull would flatten the over-dispersion and hurt the MSE. It should help the Poisson
term and is the single riskiest knob for the MSE term, which is why I keep its weight low and want the
next refinement to defend the MSE.

That refinement closes the last gap: I smooth in the log-normalized space the MSE is actually computed
in. Everything so far diffused in count or VST space, but the metric rescales each cell to a fixed
total, takes `log1p`, and measures squared error there — so I add diffusion passes in exactly that
space: rescale to the target total, `log1p`, diffuse with the same `P` for a few steps, `expm1`,
rescale back, and blend the result in. Smoothing directly in the scoring space is the most direct attack
on the MSE term and the natural counterweight to the low-rank step's risk: where the low-rank blend
might cost a little MSE for Poisson, the log-space diffusion pays it back by smoothing precisely where
the MSE is read. I do it twice — once mid-strength (a few steps, weight ~`0.6`) and once as a lighter
final polish (fewer steps, weight ~`0.3`) — so the output is settled in the scoring space before it is
returned. The inversions of Anscombe and Freeman–Tukey, unlike the plain square root, are not simple
squarings; they need a short Newton solve per pass to invert the shift and the sum-of-roots, a bounded
numerical cost that keeps each pass on the count scale the ensemble needs.

None of this is free — each piece adds a knob and the knobs interact, so I tune them jointly on the tune
set and watch the held-out gap most of all. I am now carrying on the order of a dozen hyperparameters —
the transform set, the neighbour and diffusion counts, the decay, the low-rank components and weight,
the two log-space step-and-weight pairs, the imputation depth, the blend cap — and a dozen knobs tuned
on a single seed have real capacity to memorize that seed's manifold rather than the general denoising
structure. This is why the held-out mean over seeds one through three, not the tune score, is the number
that counts. The earlier rungs earned the right to trust this: the raw calibration confirmed the scale
is stable across seeds, and kNN and MAGIC each held their held-out scores within a couple of thousandths
of their tune scores. If this endpoint instead shows a tune score noticeably above its held-out mean,
that gap is over-fit and tells me to strip knobs back — collapse the transform set, drop the second
log-space pass — until the ensemble reports the manifold and not the seed.

The prediction follows from the headroom and the mechanism together: the room left is overwhelmingly on
the Poisson side, and the gene-adaptive transforms, the zero-imputation, and the low-rank blend are all
aimed at the absolute per-gene rate the likelihood reads, so I expect the endpoint's gain to land most
on the Poisson term — the one MAGIC left almost untouched — climbing into the `0.9`s, while the log-space
passes hold the MSE term roughly flat against the low-rank step's cost. If instead the Poisson term
refuses to move and only the MSE budges, the remaining error is not the per-gene rate at all and my
whole reading of what MAGIC left behind is wrong. What none of it can do is exceed the true-rate
ceiling: the multiplicative biological over-dispersion baked into `Λ` is irreducible noise no denoiser
can recover, so every term stays strictly below one and this is the top of the ladder, not a claim of
perfection.
