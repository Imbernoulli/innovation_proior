MAGIC reached a held-out combined of about `0.85`, and the prediction I held against it landed the way
I argued: the gain came almost entirely through the MSE term, which rose from about `0.73` to `0.82`,
while the Poisson term barely moved, from about `0.87` to `0.87`. That is the signature I said a
geometry-aware denoiser should leave — the PCA-denoised embedding and the adaptive-bandwidth soft kernel
recovered the log-normalized shape that hard uniform pooling had smeared — and the `t = 2` peak with a
monotone decline afterward confirmed the diffusion lever behaves the way the spectral picture said. So
the graph and the diffusion are sound and I want to keep them as the backbone. The question is what is
left, and here I have to read the anchors again, because the situation has flipped. After MAGIC the MSE
term at `0.83` leaves an absolute gap to perfect of only about `(1 − 0.83)·0.55 ≈ 0.09` — the cheap,
narrow headroom is now nearly spent. The Poisson term at `0.887` leaves an absolute gap of about
`(1 − 0.887)·2.98 ≈ 0.34` — expensive, wide, and now by far the larger reservoir of unrecovered score.
So whatever I build next has to earn most of its keep on the Poisson term, which means it has to get the
absolute per-gene *rate* right, not just the log-normalized shape, and it has to do so especially at the
dropout entries where the likelihood bleeds hardest. That reframes the three gaps I named at the last
rung from a wish-list into a targeted attack.

The three gaps, plus one I flagged in passing, are these. First, I forced every gene through the same
square-root transform, and I already measured that the square root's variance stabilization is
non-monotone across the dropout spectrum — it over-inflates the moderate genes and treats the near-empty
and well-expressed ones differently again — so a gene expressed in almost every cell and a gene that
drops out nine times in ten are being pushed through a nonlinearity that suits neither. Second, every
diffusion so far ran in count or square-root space and never touched the log-normalized space the MSE is
actually computed in. Third, the diffusion captures only the *local* manifold geometry and never the
*global* low-rank structure sitting underneath it. And fourth, from the last rung: the diffusion fills
every exact zero the same way, whether it was a dropout of a gene the cell truly expresses or a genuine
biological zero, because nothing distinguishes them. I want to attack all four, keep MAGIC's
adaptive-bandwidth diffusion graph as the backbone, and build everything else around it. The result
converges on the ingredients of the denoiser that currently tops the OpenProblems leaderboard — a
multi-transform ensemble with low-rank and log-space refinement — which I will rebuild in plain
numpy/scipy/sklearn so it runs anywhere rather than leaning on a specialized graph library.

The headroom reading tells me where to spend first, and it is worth spelling out because the obvious
move is the wrong one. The single most direct thing I could bolt onto MAGIC is a smoothing pass in the
log-normalized scoring space — it is one gap, it is easy, and it attacks the exact space the MSE is
read in. But the MSE headroom is nearly spent: only about `0.09` in absolute terms remains, so even a
perfect log-space pass can move the combined score by at most a couple of hundredths through that term.
The `0.34` of absolute room sitting on the Poisson side can only be reached by fixing the per-gene
*rate*, and none of the count-space or log-space smoothing touches that — it is the transforms, the
zero-imputation, and the low-rank blend that move the rate. So I lead with the transforms even though
they are the most work, and I treat the log-space passes as the necessary defence of the MSE rather
than as the main event. Spending the whole rung on the easy log-space lever would be optimizing the
term that has almost nothing left to give.

Start with the transforms, because that is the cleanest win and it targets the per-gene
mis-stabilization directly. The square root is one variance-stabilizing transform for Poisson data, but
not the only one and not uniformly best. The Anscombe transform, `2√(x + 3/8)`, is the classical
low-count correction — the `3/8` shift is exactly what sharpens the stabilization where counts are
small — and the Freeman–Tukey transform, `√x + √(x+1)`, is built for the regime where counts are almost
all zeros and ones, the deep-dropout end. Each suits a different part of the spectrum: Anscombe for the
well-expressed genes, Freeman–Tukey for the high-dropout genes, and the plain square root as a bridge in
between. Rather than pick one and re-inherit the single-transform flaw, I run the *whole*
MAGIC-style pipeline three times, once under each transform, and ensemble the three denoised outputs
gene-wise, weighting each gene toward the transform that suits its dropout. The weighting has to be a
smooth interpolation of the dropout fraction `p_g`, and the natural one is the binomial split
`w_ans ∝ (1 − p_g)²`, `w_ft ∝ p_g²`, `w_sqrt ∝ 2 p_g (1 − p_g)`, normalized to sum to one. That form is
not arbitrary: it is the expansion of `((1 − p_g) + p_g)² = 1` term by term, so the three weights are a
genuine partition of unity for every gene — I checked that they sum to exactly one at every dropout
level — and they behave correctly at the extremes and the middle. At `p_g = 0` all the weight is on
Anscombe; at `p_g = 1` all of it is on Freeman–Tukey; and the square-root bridge weight `2 p_g(1 − p_g)`
peaks at exactly `p_g = 0.5`, where it takes half the total, precisely the crossover regime where
neither specialist transform is clearly right. So each gene is denoised through a smooth blend of the
three, dominated by whichever transform its dropout calls for. This gene-adaptive multi-VST ensemble is
the single biggest structural change from MAGIC, and it is aimed at the Poisson term: getting the
per-gene stabilization right is getting the per-gene *rate* right, which is what the likelihood reads.
To see the blend is doing something graded rather than nominal, take a few concrete genes. A gene that
drops out a quarter of the time gets weights of about `0.56` Anscombe, `0.06` Freeman–Tukey, `0.38`
bridge — squarely an Anscombe-dominated gene with a square-root correction. At the matrix's typical
dropout near `0.59` the weights land near `0.17` Anscombe, `0.35` Freeman–Tukey, `0.48` bridge — the
bridge and the deep-dropout transform now carrying most of the gene, which is right for a gene that is
mostly zeros. And a gene that fires in only one cell in ten sits at about `0.01`, `0.81`, `0.18` — almost
pure Freeman–Tukey. So the ensemble is genuinely moving mass across the three transforms as a gene's
dropout rises, not just rubber-stamping one of them.

I considered the cheaper alternative of a hard per-gene assignment — pick the single best transform for
each gene by a dropout threshold and denoise it through that one — and rejected it for the same reason I
rejected the hard `k` and the fixed bandwidth two rungs ago. A hard assignment puts a cliff into the
gene axis: two genes at dropout `0.49` and `0.51` would be pushed through different transforms and land
on slightly different scales, and since the ensemble is what the Poisson term reads gene by gene, that
discontinuity would show up as avoidable noise in the metric. The soft binomial blend has no such seam —
it varies continuously with `p_g` — and it costs nothing extra at ensemble time since I am running all
three passes anyway. The only real cost of the ensemble is that it triples the pipeline, three
impute-diffuse-blend passes instead of one, but the graph is built a single time and reused, each pass
is a few sparse matrix-vector products, and there is one SVD and a couple of log-space passes on top, so
the whole thing is still dominated by the one-time PCA and stays comfortably cheap.

There is a mechanism reason the transform choice bears on the Poisson term specifically, beyond the
loose "stabilization" argument, and pinning it down is what convinces me the ensemble is the right
lever rather than a decoration. The diffusion averages cells in the transformed space and then inverts,
so the recovered rate for a gene is `f⁻¹` of an average of `f`-transformed counts. When `f` is nonlinear
and the average spans cells whose rates differ even slightly, that composition is biased — by the
curvature of `f` at the relevant count level, a Jensen-type effect — so the inverted value is not an
unbiased estimate of the underlying rate. A transform whose curvature is well-matched to a gene's count
regime keeps that bias small; a mismatched transform inflates it. Because the Poisson NLL reads the
absolute rate directly, that curvature-induced bias lands squarely on the Poisson term, whereas the
log-normalized MSE, which cares about relative shape after its own `log1p`, is far less sensitive to it.
So choosing the transform per gene is not cosmetic: it is the most direct handle I have on the bias of
the recovered rate, which is the quantity the term with all the remaining headroom is measuring.

The graph itself I build once, on the Anscombe embedding, and reuse across all three passes, because
the neighbour geometry should not depend on which VST I am currently denoising through — it is a
property of the cells, not of the transform — and rebuilding it three times would only add noise and
cost. I also mix a self-loop into the transition matrix, replacing `P` with a renormalized
`½ I + ½ P`, so each cell retains half its own mass at every diffusion step. That matters because I am
about to apply the diffusion far more aggressively than MAGIC did — imputation passes, multi-scale
sums, log-space passes — and without the self-loop those repeated applications would race toward
consensus; holding half the mass on the diagonal slows the contraction and keeps every pass
well-behaved.

Now the fourth gap, the indiscriminate zero-filling, which I fix before the main diffusion in each
pass. A hard zero in a high-dropout gene is far more likely a dropout than biology, so I impute the
exact zeros specifically: take a couple of diffusion steps to get a diffusion-weighted neighbour
average, and fill *only* the zero entries with what the manifold says should be there, leaving the
observed non-zeros alone, then renormalize. This distinguishes the dropout holes from the observed
signal in a way plain diffusion never did — it fills what is missing without over-writing what was
measured — and it is again pointed at the Poisson term, because those filled holes are exactly the
`X_train = 0`, `X_test > 0` entries where the raw likelihood bled worst.

Inside each pass I also want to be smarter than the single power `Pᵗ`, and here I have to be careful,
because at the last rung I found the global optimum was `t = 2` and larger `t` lost score monotonically.
That result was for *one global smoothing depth applied to every gene*. But the genes do not all want
the same depth: a high-dropout gene has sparse, noisy observations and wants deep pooling to fill and
stabilize it, while a low-dropout gene is already well-measured and wants almost none. So instead of a
single power I use a gene-wise weighted multi-scale diffusion, accumulating `X, PX, P²X, …, PᵗX` with
per-gene decaying weights and a guaranteed baseline: `X̂ = (Σᵢ bᵍᵢ Pⁱ X) / (Σᵢ bᵍᵢ)` with
`b_g = 0.9 (0.2 + 0.8 p_g)`. This looks like it contradicts the `t = 2` finding, since I let `t` run out
to seven, so I checked what the weighting actually does to the *effective* smoothing depth per gene. For
a gene that never drops out, `b_g ≈ 0.18`, and the weighted-mean power works out to about `0.2` steps —
essentially no smoothing, the gene keeps itself. At the matrix's typical dropout near `0.59`, the
effective depth is about `1.4` steps; at `p_g = 0.9` it is about `2.5`; at `p_g = 1` about `3`. So the
maximum `t = 7` is a ceiling that almost no gene reaches, not a flat depth — the decay concentrates each
gene's weight near an effective depth that ranges from a fraction of a step for the well-measured genes
up to about three steps for the deep-dropout ones. Crucially the high-dropout genes land near the same
two-to-three-step depth MAGIC found optimal globally, while the low-dropout genes are smoothed far less
than a flat `t = 2` would have. A flat `Pᵗ` at `t = 7` would over-diffuse and lose score exactly as the
last rung showed; this weighted sum is a different operator, giving each gene the depth its dropout
warrants and reconciling the aggressive ceiling with the earlier finding. The additive baseline — the
`i = 0` term with weight one — guarantees every gene keeps at least some of its own diffused-once
signal so nothing is smoothed to nothing.

After diffusing I still have to guard against over-smoothing the genes that did not need it, so I blend
the diffused signal back toward the raw normalized signal with a per-gene weight, `w_g = clip(p_g ·
vred_g · (1 − μ̃_g), 0, 0.7)`, where `vred_g` is the fraction of variance the diffusion removed for that
gene and `μ̃_g` is its min-max-scaled mean expression. The logic reads off the factors: I lean on the
diffused version only where all three signals agree that smoothing is appropriate — the gene drops out
often, the diffusion removed a lot of variance, and the gene's mean is low. A well-expressed,
low-dropout gene, where the diffusion mostly destroyed real per-gene variation, gets a weight near zero
and keeps its raw values; the `0.7` cap ensures even the most diffusion-friendly gene retains at least
thirty percent of its raw signal. This adaptive blend is the safety valve that stops the ensemble from
smearing the genes the earlier rungs already handled well — a hedge specifically against the multi-scale
diffusion reaching too far on genes that did not want it.

On top of the ensemble sit the two global refinements that close the remaining gaps. The first is
low-rank, and I have to reconcile it with a decision I made explicitly at the last rung, where I refused
to truncate the diffused output to the PCA rank because the true rate is low-rank *plus* multiplicative
over-dispersion, so a hard truncation would throw away real per-gene variation. That refusal still
stands — what I add here is not a truncation but a light *blend*. The cells live on a low-dimensional
manifold, so the denoised matrix should be approximately low-rank in the gene directions, and a
truncated SVD captures that global structure the purely local diffusion misses. I reconstruct from the
top components and blend the reconstruction back in with a small weight, around `0.10`. The weight has
to be small for exactly the reason I refused the hard truncation: an aggressive low-rank pull would
flatten the over-dispersion and hurt the MSE, so this is a light touch that recovers the dominant global
structure and tightens the Poisson likelihood without erasing the per-gene spread. I expect this step to
help the Poisson term and to be the single riskiest knob for the MSE term, which is exactly why I keep
its weight low and why I want the next refinement to defend the MSE.

That next refinement closes the last gap the last rung named: I smooth in the log-normalized space the
MSE is actually computed in. Everything so far diffused in count or VST space, but the metric rescales
each cell to a fixed total, takes `log1p`, and measures squared error there — so I add diffusion passes
performed in exactly that space: rescale to the target total, `log1p`, diffuse with the same `P` for a
few steps, `expm1`, rescale back, and blend the result in. Smoothing directly in the scoring space is
the most direct possible attack on the MSE term, and it is the natural counterweight to the low-rank
step's risk: where the low-rank blend might cost a little MSE in exchange for Poisson, the log-space
diffusion pays that back by smoothing precisely where the MSE is read. I do it twice — once
mid-strength, a few steps at a weight around `0.6`, and once as a lighter final polish, fewer steps at a
weight around `0.3` — so the output is settled in the scoring space before it is returned. The
inversions of Anscombe and Freeman–Tukey, unlike the plain square root, are not simple squarings — they
need a short Newton solve per pass to invert the shift and the sum-of-roots — but that is a bounded
numerical cost and it keeps each pass on the count scale the ensemble needs.

So the full endpoint builds the diffusion graph once on the Anscombe embedding with a self-loop; runs
the impute-then-multiscale-diffuse-then-adaptive-blend pipeline under Anscombe, Freeman–Tukey, and
square root; ensembles the three gene-wise by dropout; applies the light truncated-SVD refinement; and
finishes with the two log-normalized-space diffusion passes. Every piece exists to close a gap MAGIC
left open, and none of them is free — each adds a knob, and the knobs interact, so I tune them jointly on
the tune set and accept whatever the held-out sets say, watching the low-rank weight most closely
because it is the one that can move the two metrics in opposite directions. That interaction is the real
methodological hazard of this rung, and I should be blunt about it: I am now carrying on the order of a
dozen hyperparameters — the transform set, the neighbour and diffusion counts, the decay, the low-rank
components and weight, the two log-space step-and-weight pairs, the imputation depth, the blend cap —
and a dozen knobs tuned jointly on a single seed have real capacity to memorize that seed's particular
manifold rather than the general denoising structure. This is why the held-out mean over seeds one
through three, not the tune score, is the number that counts, and why I want to watch the gap between
them. The earlier rungs earned the right to trust this: the raw calibration confirmed the scale is
stable across seeds, and kNN and MAGIC each held their held-out scores within a couple of thousandths
of their tune scores despite their knobs. If this endpoint instead shows a tune score noticeably above
its held-out mean, that gap is the over-fit, and it would tell me to strip knobs back — collapse the
transform set, drop the second log-space pass — until the ensemble is reporting the manifold and not the
seed. So the joint tune is done with the held-out gap, not just the tune peak, as the thing I am
reading. The falsifiable prediction
follows from the headroom reading and the mechanism together: after MAGIC the room that is left is
overwhelmingly on the Poisson side, and the gene-adaptive transforms, the zero-imputation, and the
low-rank blend are all aimed at the absolute per-gene rate the likelihood reads, so I expect the
endpoint's gain to land most on the Poisson term — the term MAGIC left almost untouched — while the
log-space passes hold the MSE term steady against the low-rank step's cost. In absolute units the prediction
is concrete: MAGIC's raw Poisson NLL sat near `−0.61` on the tune seed with the perfect anchor at about
`−0.95`, so I expect the endpoint to push that NLL down meaningfully toward the anchor — a Poisson term
climbing from `0.887` into the low-to-mid `0.9`s — while the raw MSE, near `1.15` after MAGIC, holds
roughly flat or dips only slightly, since its headroom is nearly gone. If instead the Poisson term
refuses to move and only the MSE budges, then the remaining error is not the per-gene rate at all and my
whole reading of what MAGIC left behind is wrong. What none of it can do is exceed the true-rate
ceiling: the multiplicative biological over-dispersion baked into `Λ` is irreducible noise no denoiser
can recover, so every term stays strictly below one and this is the top of the ladder, not a claim of
perfection.
