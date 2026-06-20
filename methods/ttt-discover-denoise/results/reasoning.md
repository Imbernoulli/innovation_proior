MAGIC got me a clean diffusion denoiser, and the feedback named exactly where it stops short. Two
things. First, I forced every gene through the same square-root transform — but a gene that is
expressed in almost every cell and a gene that drops out in nine cells out of ten are not stabilized
by the same nonlinearity, and treating them identically leaves variance on the table. Second, plain
diffusion smooths in the count space and never looks at the log-normalized space the MSE is actually
computed in, nor at the global low-rank structure that sits underneath the local manifold geometry.
I want to attack all three: gene-adaptive transforms, a low-rank refinement, and a final pass that
smooths in the exact space the metric scores. The diffusion graph from MAGIC stays as the backbone;
everything else is built around it.

Start with the transforms, because that is the cleanest win. The square root is one
variance-stabilizing transform for Poisson data, but it is not the only one and not uniformly the
best. The Anscombe transform, `2√(x + 3/8)`, stabilizes Poisson variance more accurately at low
counts than the bare square root. The Freeman-Tukey transform, `√x + √(x+1)`, behaves better still
in the deep-dropout regime where counts are mostly zeros and ones. Each is best for a different part
of the dropout spectrum: Anscombe for the well-expressed genes where counts are moderate, Freeman-
Tukey for the high-dropout genes that are almost all zeros, and the plain square root as a bridge in
between. So rather than pick one, I run the *whole* MAGIC-style diffusion pipeline three times, once
under each transform, and ensemble the three denoised outputs *gene-wise* — weighting each gene
toward the transform that suits its dropout. A gene that drops out rarely gets weighted toward
Anscombe, a gene that drops out constantly toward Freeman-Tukey, and the weights interpolate
smoothly with the dropout fraction. This is the gene-adaptive multi-VST ensemble: the single biggest
structural change from MAGIC, and it directly targets the per-gene mis-stabilization the feedback
flagged.

Inside each transform's pass I also want to be smarter than plain `Pᵗ`. Two refinements. One: the
hard zeros in a high-dropout gene are not real biology, they are dropouts, so before diffusing I
impute the exact zeros with a diffusion-weighted neighbor average — fill the holes with what the
manifold says should be there, then diffuse the filled matrix. Two: instead of a single power `Pᵗ`,
I use a gene-wise weighted *multi-scale* diffusion — accumulate `X, PX, P²X, …, Pᵗ X` with weights
that decay per gene, so each gene gets the amount of smoothing its dropout warrants, with a
guaranteed baseline so every gene is smoothed at least a little. And after diffusing I blend the
diffused signal back toward the raw normalized signal with a per-gene weight that depends on dropout,
variance reduction, and how correlated the diffused and raw versions are — genes that diffusion
clearly helps get more of it, genes where diffusion mostly destroyed signal keep more of their raw
values. This adaptive blend is the safety valve that stops the ensemble from over-smoothing the
genes that did not need it.

Now the two global refinements that sit on top of the ensemble. The first is low-rank. The cells
live on a low-dimensional manifold, so the denoised matrix should be approximately low-rank in the
gene directions too; a truncated SVD captures that global structure that local diffusion misses. I
take a truncated SVD of the ensembled matrix, reconstruct from the top components, and blend the
low-rank reconstruction back in with a modest weight. I have to be careful here — the true rate in
this data is deliberately *not* exactly low-rank (there is multiplicative biological over-dispersion
on top of the low-rank signal), so an aggressive low-rank weight would throw away real per-gene
variation and *hurt* the MSE. So this is a light touch: enough to recover the dominant global
structure and tighten the Poisson likelihood, not so much that it flattens the over-dispersion. The
weight is small and tuned.

The second global refinement is the one that closes the last gap the feedback named: I smooth in the
log-normalized space the MSE is actually computed in. Up to now everything diffused in count or VST
space, but the metric normalizes each cell to a fixed total, takes `log1p`, and measures squared
error there. So I add a final diffusion pass performed *in that space* — rescale each cell to the
target total, `log1p`, diffuse with the same `P` for a few steps, invert with `expm1`, and rescale
back — and blend it in. Smoothing directly in the scoring space is the most direct possible attack
on the MSE term, and I do it twice: once mid-pipeline guided by the per-gene blend weights, and once
as a final polish. This is the "extra diffusion in log-normalized space" that the count-space MAGIC
never touched.

So the full endpoint is: build the diffusion graph once on the Anscombe embedding (with a self-loop
mixed in for stability); run the impute-and-multiscale-diffuse-and-adaptive-blend pipeline under
Anscombe, Freeman-Tukey, and square root; ensemble the three gene-wise by dropout; apply a light
truncated-SVD low-rank refinement; and finish with two log-normalized-space diffusion passes. Every
piece is there to close a gap MAGIC left open, and none of them is free — each adds a knob, and the
knobs interact, so I tune them jointly on the tune set and accept whatever the held-out sets say. I
am wary of the low-rank weight in particular, because the over-dispersion in this data means low-rank
can hurt the MSE even as it helps the Poisson, so I keep it small and let the log-space polishing do
the heavy lifting on the metric I care most about.

This is an adaptation of the denoiser that test-time-training/discover (TTT-Discover, arXiv:
2601.16175) evolved to the top of the OpenProblems leaderboard, reaching 0.71 on PBMC and 0.73 on
Tabula Muris versus MAGIC's ~0.64. Its real implementation leans on graphtools and scprep for the
graph and normalization; I rebuild the same ideas — multi-VST ensembling, zero-imputation, gene-wise
multi-scale diffusion, adaptive raw/diffused blending, truncated-SVD refinement, and the final
log-space polishing — in plain numpy/scipy/sklearn so it runs anywhere. I expect it to clear MAGIC
decisively, and I expect the gain to show up most on the MSE term, because that is where the
gene-adaptive transforms and the log-space smoothing are aimed. What it cannot do is exceed the
true-rate ceiling — the over-dispersion is irreducible Poisson-plus-biological noise no denoiser can
recover — so this is the top of the ladder, not a claim of perfection.
