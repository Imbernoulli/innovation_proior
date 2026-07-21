The raw baseline came back at exactly `0.0000` on both terms every seed — the scale is wired
correctly, the perfect anchor sits below the raw anchor on both metrics, and the whole distance to one
is open. Two numbers from that calibration set the exchange rate of the whole ladder, and I will come
back to them: raw MSE around `1.5`, raw Poisson NLL around `1.7`, with a perfect anchor well below
each. First the idea that lifts a denoiser off the floor. Cells in the same biological state are
independent noisy measurements of one rate, so averaging the cells that share a state beats the Poisson
noise down while the shared signal survives. Quantitatively that tells me how much is on the table:
averaging `m` independent draws of the same rate cuts sampling variance by `1/m`, so pooling a cell
with ~ten genuine neighbours would knock per-entry Poisson variance to about a ninth. That is why even
the crudest averaging should capture most of the gap the identity left. The catch that organizes
everything below is the phrase "genuine neighbours" — the `1/m` reduction is realized only to the extent
the pooled cells actually share the rate, and every impostor I average in trades variance reduction for
bias. I want this idea in its crudest honest form first, both because it is the natural next rung and
because I want to measure what a bare neighbour average is worth before spending complexity on a
smarter neighbour relation.

The recipe: for each cell, find its `k` nearest neighbours and replace its profile with the average
over itself and them. Three decisions live in that one sentence.

The first is what space I measure distance in. Raw counts are clearly wrong: a deeper-sequenced cell
looks far from an identical shallow one purely because its counts are uniformly larger, so Euclidean
distance on raw counts mostly measures library size — exactly the nuisance I want to be blind to.
Library-normalizing each cell to a common total removes depth, but UMI counts are heteroscedastic: a
gene at mean 100 has Poisson SD 10, a gene at mean 1 has SD 1, so on the count scale a few loud genes
carry almost all the squared distance and the low-expression genes that actually distinguish states
contribute nothing. The canonical cure for Poisson data is the square root: to first order `Var(√X)` is
mean-independent, so every gene contributes comparably. It is imperfect exactly where I live, though —
the asymptotic `Var(√X) → 0.25` is only reached at large means, but a median library of ~900 over 1000
genes puts the typical per-entry rate near `0.9`, half that after the split, so I am in the `λ ≈ 0.2`
to `1` regime where the transform is least stabilized and actually *non-monotone*: `Var(√X)` runs low
near `λ = 0.2`, peaks around `λ ≈ 1`, then settles back toward `0.25`. So it over-inflates the moderate
genes relative to the near-empty and well-expressed ones — a residual gene-dependent heteroscedasticity
I carry forward as an open gap, since a single global transform treats genes at different dropout rates
inconsistently. The square root is still the right first move, vastly better than the untransformed
count. I reject `log1p` for the distance geometry: it is the natural transform for multiplicative or
lognormal noise, over-compresses the small integers that carry the dropout structure, and does not
stabilize Poisson variance — so even though the MSE is scored in log space, using it here would
reintroduce the heteroscedasticity the square root removes. That tension between the Poisson-natural
search space and the log-normalized scoring space is real, and I note it as a gap I am choosing not to
close now. So: square-root, library-normalize, measure neighbours there.

The second decision is where to look for neighbours, and here I am deliberately crude. Classical
kNN-smoothing picks neighbours directly on the observed noisy normalized profiles; it does not denoise
the space it searches. That is the character of this rung and its known weakness. In high dropout the
observed profiles are sparse and noisy, so the neighbour relation is corrupted by the same noise I am
averaging away: two same-state cells that dropped out on different genes look far apart, two
different-state cells sharing their few captured genes look close. The distance decomposes into a
between-state part and a within-state sampling-noise part, and the relation is reliable only where the
between-state signal exceeds the within-state noise. But with dropout near 60%, a substantial share of
the state-distinguishing genes are zero in one of any given pair purely by chance, inflating the
within-state floor toward the very scale of the between-state signal I am trying to resolve. When those
are comparable, the `k` nearest cells by observed distance are a noisy sample of the true
neighbourhood — some genuine, some impostors — and no tuning of `k` fixes a corrupted ranking. That is
precisely why the realized variance reduction will fall short of the ideal `1/m`, and precisely what a
denoised embedding could fix by projecting onto the shared low-dimensional structure — the lever the
next rung pulls. Curing it here would collapse two rungs into one and rob me of a clean measurement of
what the bare neighbour average is worth, so I accept the corrupted relation on purpose and measure the
cost.

The third decision is how to pool — crude again. Cell plus its `k` neighbours, uniform average, a hard
cliff at the `k`-th: a cell just inside contributes full weight, one a hair outside nothing, even at
essentially identical distance. That discontinuity is not how similarity falls off, but the uniform
average is the honest baseline and its one knob, `k`, is a genuine bias-variance lever. Small `k`
removes little variance but distorts little signal; large `k` removes more but, because the pool is
hard and uniform, inevitably reaches across the local state boundary and blends in cells that are not
quite the same, smearing real biological variation into bias. The deeper flaw is that no single global
`k` can be right everywhere: a cell in a dense region has many genuine neighbours and tolerates a large
pool, while one on a trajectory edge has only a few and a large `k` drags in strangers from the next
state over. One bandwidth and a hard pool cannot serve both — the structural weakness I name now so the
next rung has a clear target. I sweep `k` over ~5, 10, 20, 50 on the tune seed and take the peak; I
expect ~10 as the compromise, with the score still climbing at `k = 5` (variance left on the table) and
falling back by `k = 50`, where a hard pool of a sixth of the dataset reaches well past the local state
for edge cells. A curve that keeps rising to fifty would mean the states are coarser than I think; one
that peaks early and drops fast, tighter. Either way the shape of the sweep is information about the
data.

After averaging in the stabilized normalized space I invert: square back to the count scale, and
multiply each cell by its *own* original library size rather than the pool's, so a deep cell stays deep
and a shallow one shallow instead of every cell flattened to the average depth. Squaring guarantees
non-negativity with no clipping and preserves shape cell-for-gene; restoring per-cell depth matters
specifically for the Poisson term, which rescales to the test budget and would be thrown off by
collapsed depths.

Now the scale, where the rung-1 anchors earn their keep. The raw-to-perfect gap — the denominator of
each normalized score — differs sharply between the metrics. On the tune seed raw MSE was ~`1.60`,
perfect MSE ~`1.06`, headroom ~`0.55`; raw Poisson ~`2.03`, perfect ~`−0.95`, headroom ~`2.98`, more
than five times wider. That is the exchange rate I need to keep straight: a given absolute NLL
reduction buys far less normalized credit than the same absolute MSE reduction, because it is divided
by a five-times-wider headroom. Hard uniform averaging is mechanistically a variance-beater — it shrinks
the maximally over-dispersed raw prediction toward the pooled mean, and over-dispersion is exactly what
the Poisson likelihood punishes hardest — so in *absolute* terms I expect the NLL to drop a great deal.
Recovering the log-normalized *shape* is a different quantity, needing the local geometry to be right,
and my geometry is corrupted by noisy neighbour selection and blurred by the hard pool, so I expect a
more limited absolute MSE reduction. Which *normalized* term ends up higher is not obvious — the
mechanism asymmetry (large absolute NLL gain, modest MSE gain) fights the headroom asymmetry (Poisson
headroom five times wider). If the mechanism reading is right the normalized Poisson term should lead,
since the over-dispersion fix is what hard averaging does best and should clear the MSE term even
through the wide headroom. If the two come back level or inverted, my reading of the mechanism is
wrong.

Overall I expect a large jump off the floor, likely more than on real tissue because the synthetic
manifold is cleaner and the neighbour relation less corrupted, and the computation is trivial — exact
nearest-neighbour search over 900 cells in the ~1000-dimensional sqrt space is well under a billion
distance operations. On over-fitting: the held-out raw anchors clustered tightly (~`1.43`–`1.49` MSE),
so the three datasets are comparable draws rather than wildly different regimes, and with a single
coarse knob I expect the held-out combined score to track the tune score closely; a materially lower
held-out score would mean even one coarse knob found something seed-specific, and I would distrust
every richer rung above this one. The two named weaknesses — neighbours chosen on noisy profiles, a hard
uniform pool with one global `k` — are the clear gap below the next rung, which denoises the space the
neighbours are found in and replaces the hard uniform pool with an adaptive-bandwidth,
transitively-weighted average.
