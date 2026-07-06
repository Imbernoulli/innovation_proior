The raw baseline came back at exactly `0.0000` on both normalized terms, every seed, which is the
clean calibration I wanted: the scale is wired correctly, the perfect anchor sits below the raw anchor
on both metrics, and the whole distance from zero to one is sitting open in front of me. The raw MSE
landed around `1.5` and the raw Poisson NLL around `1.7` in the harness's units, with a perfect anchor
well below each — I will come back to those exact numbers, because they turn out to set the exchange
rate of the whole ladder. But first the idea that lifts a denoiser off the floor at all. Cells in the
same biological state are independent noisy measurements of one underlying rate, so if I can find, for
each cell, the other cells that share its state and average them together, the Poisson noise averages
down while the shared signal survives. That is worth making quantitative before I build anything,
because it tells me how much is even on the table: averaging `m` independent draws of the same rate
cuts the sampling variance by a factor of `1/m`, so pooling a cell with ten genuine neighbours would,
in the ideal case, knock the per-entry Poisson variance down to about a ninth of its raw value. That
is an enormous reduction, and it is why even the crudest averaging should capture most of the gap the
identity left open. The catch, which will organize everything below, is the phrase "genuine
neighbours": the `1/m` reduction is only realized to the extent the pooled cells actually share the
rate, and every impostor I average in trades variance reduction for bias. I want to start with this
idea in its crudest honest form before reaching for anything graph-theoretic, both because it is the
natural next rung and because I want to measure how much of the gap a bare neighbour average can close
before I spend complexity making the neighbour relation smarter.

So the recipe is: for each cell, find its `k` nearest neighbours, and replace its profile with the
average over itself and those `k` cells. Three decisions live inside that one sentence, and each is a
real fork I should walk rather than assume.

The first is what space I measure distance in, and it is not obvious. The naive choice is raw counts,
and it is clearly wrong: a cell that happened to be sequenced deeper looks far from an identical
shallow cell purely because its counts are uniformly larger, so a Euclidean distance on raw counts
mostly measures library size, which is exactly the nuisance variable I want to be blind to. The fix
is to remove depth by library-size-normalizing each cell to a common total — but that alone is not
enough, because UMI counts are heteroscedastic in a way that hijacks the distance. A gene expressed at
a mean of a hundred has a Poisson standard deviation of ten, while a gene at a mean of one has a
standard deviation of one; on the count scale a few high-expression genes carry almost all the
squared distance and the low-expression genes that actually distinguish cell states contribute
nothing. The classical cure is a variance-stabilizing transform, and for Poisson data the square root
is the canonical one: to first order `Var(√X)` is constant, independent of the mean, so after the
transform every gene contributes comparably to the distance instead of the loudest ones dominating. I
should be honest about how imperfect it is at the counts that actually dominate here, and I can put
numbers on it. The asymptotic target is `Var(√X) → 0.25` as the mean grows, but the median library of
about nine hundred molecules spread over a thousand genes puts the typical per-entry full-cell rate
near `0.9`, and half that after the split, so I am living in the `λ ≈ 0.2` to `1` regime — exactly
where the square root is least stabilized. On a quick Poisson check the variance of `√X` runs about
`0.16` at `λ = 0.2`, rises to roughly `0.29` at `λ = 0.45`, and peaks near `0.39` around `λ ≈ 1`
before settling back toward `0.25` at higher rates. So the transform not only fails to hit its
constant here, it is actually *non-monotone* across the dropout spectrum, over-inflating the variance
of the moderate genes relative to both the near-empty and the well-expressed ones. That residual,
gene-dependent heteroscedasticity is a real weakness I am carrying forward — a single global transform
is treating genes at different dropout rates inconsistently — but the square root is still the right
first move, vastly better than the untransformed count, and I will take it as the baseline VST and
leave the question of whether a gene-aware family of transforms would do better as an open gap rather
than something to solve at this rung. There is one more candidate space I should name and reject for now: the metric scores
its MSE in *log-normalized* space, so an argument could be made to measure neighbour distances there
too, for consistency with what I am scored on. But `log1p` is the natural transform for multiplicative
or lognormal noise, not for Poisson counts — it over-compresses the small integers that carry the
dropout structure and does not stabilize Poisson variance — so using it for the distance geometry
would reintroduce the heteroscedasticity the square root removes. The tension between "measure
neighbours in the Poisson-natural space" and "score in the log-normalized space" is real, and I am
noting it deliberately as a gap I am choosing not to close at this rung. For now: square-root the
counts, normalize each cell's `√X` vector to a common library size, and measure neighbours there.

The second decision is the one I want to be deliberately crude about: where do I look for neighbours?
The honest classical kNN-smoothing picks neighbours directly on the observed, noisy normalized
profiles — it does not first denoise the space it searches in. That is the whole character of this
rung and also its known weakness, and I want to be precise about the failure mode because it is what
motivates everything after. In a high-dropout regime the observed profiles are sparse and noisy, so
the nearest-neighbour relation is itself corrupted by the same noise I am trying to average away: a
cell's true neighbours can be pushed away and impostors pulled in, purely by which genes happened to
drop out in each cell. Two cells in the same state that dropped out on different genes look far apart;
two cells in different states that happen to share their few captured genes look close. So the very
variance I want to beat down is also corrupting my choice of what to average, and some fraction of my
pooling will simply be wrong — impostors dragged in, real neighbours left out — eating into that
ideal `1/m` variance reduction and adding bias. I could try to cure this now by denoising the space
before searching it, but that is a genuinely different and heavier construction — a geometry-aware,
adaptive-bandwidth affinity graph built on a denoised embedding — and it is exactly what the next rung
is for. Curing it here would collapse two rungs into one and rob me of the clean measurement of what
the bare neighbour average is worth. So I accept the corrupted neighbour relation on purpose: for now,
neighbours on the raw normalized profiles, eyes open to the cost.

It is worth being quantitative about why that corruption is not a minor blemish, because it sets the
ceiling on what this rung can achieve. Whether the nearest-neighbour relation is trustworthy is a
signal-to-noise question: the distance between two cells decomposes into a between-state part, the real
difference in their rates, and a within-state part, the sampling noise that separates two cells drawn
from the *same* rate. The neighbour relation is reliable only where the between-state signal exceeds
the within-state noise. But the within-state distance is not small here. Two cells with identical rates
still differ because each gene is an independent Poisson draw, and in the `λ ≈ 0.2` to `1` regime a
gene that should fire is captured in one cell and dropped in the other a large fraction of the time —
with dropout near sixty percent, a substantial share of the genes that distinguish states are
zero in one of any given pair purely by chance. Each such disagreement adds to the sqrt-normalized
distance between two same-state cells, inflating the within-state noise floor toward the very scale of
the between-state signal I am trying to resolve. When those two are comparable, the `k` nearest cells
by observed distance are a noisy sample of the true neighbourhood — some genuine, some impostors —
and no amount of tuning `k` fixes it, because the ranking itself is corrupted. This is the concrete
reason the realized variance reduction will fall short of the ideal `1/m`, and it is precisely the
quantity a denoised embedding could improve: projecting onto the directions of shared, low-dimensional
structure would suppress the per-gene sampling noise in the distance while keeping the between-state
signal, raising the ratio. That is the lever the next rung pulls; here I simply pay the cost and
measure it.

The third decision is how to pool, and here I make the crude choice again on purpose. I take the cell
plus its `k` neighbours and average their normalized profiles uniformly — a hard, uniform pool, where
every neighbour counts equally and the boundary at the `k`-th neighbour is a cliff. A cell just inside
the cutoff contributes its full weight; a cell a hair outside contributes nothing, even if its
distance is essentially identical. That discontinuity is not how similarity actually falls off, and a
smoothly weighted pool would obviously be gentler, but the uniform average is the honest baseline and
its failure is instructive. The one knob it exposes, `k`, is a genuine bias-variance lever, and its
imperfection is structural rather than incidental. Small `k` averages few cells, so it removes little
variance — the `1/m` factor is close to one — but distorts little signal, because the few cells it
pools are the most similar. Large `k` removes more variance, driving the `1/m` factor down, but
because the pool is a hard uniform average it inevitably reaches across the boundary of the local
state and blends in cells that are not quite the same, smearing real biological variation into bias.
There is a sweet spot where the marginal variance removed by adding one more neighbour equals the
marginal bias it injects, and that spot is where I want `k`. But — and this is the deeper flaw — no
single global `k` can be right everywhere: a cell in a dense region of the manifold has many genuine
neighbours and could tolerate a large pool, while a cell on the edge of a trajectory has only a few
and a large `k` drags in strangers from the next state over. One global `k` and a hard uniform pool
cannot serve both the dense interior and the sparse boundary of the same dataset. That single
bandwidth cannot adapt to local density is the structural weakness of the whole approach, and I will
name it now so the next rung has a clear target. I will tune one value on the tune set and report it
honestly; I expect something in the range of ten neighbours to be a reasonable compromise between
beating down variance and not smearing across states.

Concretely I will sweep `k` over something like five, ten, twenty, and fifty on the tune seed and read
the combined score off the tune column, because the two effects trade off in a way I can anticipate. At
`k = 5` the `1/m` variance factor is only about a sixth, so I am leaving noise on the table that a
larger pool would remove, and I expect the score to be climbing still. Somewhere around ten the
marginal neighbour is roughly the last one that is still reliably same-state for a typical cell, so the
variance still falling should be about matched by the bias starting to rise. By `k = 50` the hard
uniform pool is almost certainly reaching well past the local state for cells near trajectory
boundaries — a sixth of the whole dataset is a lot of cells to call one cell's neighbours — so I expect
the bias from smearing to dominate and the score to fall back. If the tune curve instead keeps rising
all the way to fifty, that would tell me my mental model of the state sizes is wrong and the manifold
is coarser than I think; if it peaks early and drops fast, the states are tighter than I think. Either
way the shape of that sweep is information about the data, not just a knob-fit, and I will take the `k`
at the tune-set peak and then check on the held-out seeds whether that choice generalizes.

After averaging in the stabilized normalized space I have to undo the transform, and the inversion is
worth stating carefully because it is where the output re-enters the world the metrics live in. I
square the averaged `√`-space profile to get back to the count scale, and I multiply each cell back by
its own original library size rather than by the pool's, so a deeply-sequenced cell stays deep and a
shallow one stays shallow instead of every cell being flattened to the average depth. Squaring
guarantees the output is non-negative with no clipping, the shape is preserved cell-for-gene, and the
result lands on the count scale the Poisson NLL expects — the output contract the harness enforces is
satisfied by construction. Restoring per-cell depth matters specifically for the Poisson term, which
rescales to the test budget and would be thrown off if I had collapsed the depths into one.

Now what I expect, and here the rung-1 anchors earn their keep, because they let me read the scale
rather than guess at it. The raw-to-perfect gap is the denominator of every normalized score, and it
is strikingly different between the two metrics. On the tune seed the raw MSE was about `1.60` and the
perfect MSE about `1.06`, a headroom of only about `0.55`; the raw Poisson was about `2.03` and the
perfect Poisson about `−0.95`, a headroom of nearly `2.98`, more than five times larger. That ratio is
the exchange rate I need to keep straight: a given absolute reduction in NLL buys far less normalized
credit than the same absolute reduction in MSE, because it is divided by a headroom five times as
wide. So the two normalized terms are measuring absolute progress on very different scales, and a
method that reduces NLL enormously in absolute terms can still post a modest normalized Poisson number,
while a method that nudges the MSE down a little in absolute terms can post a large normalized MSE
number. With that in hand I can make a real prediction rather than a vague one. Hard uniform averaging
is, mechanistically, a variance-beater: it takes the maximally over-dispersed raw prediction and
shrinks its variance toward the pooled mean, and over-dispersion is exactly what the Poisson
likelihood punishes hardest, so in *absolute* terms I expect the NLL to drop a great deal — this is
the term where the `1/m` variance reduction pays off most directly. The log-normalized shape is a
different quantity: recovering it well needs the local geometry to be right, and my geometry is
corrupted by noisy neighbour selection and blurred by the hard uniform pool, so I expect the absolute
MSE reduction to be more limited. What I genuinely cannot call in advance is which *normalized* term
ends up higher, because the mechanism asymmetry (huge absolute NLL gain, modest absolute MSE gain)
and the headroom asymmetry (Poisson headroom five times wider) push against each other. If the
mechanism reasoning is right, the normalized Poisson term should lead the normalized MSE term — the
over-dispersion fix is the thing hard averaging does best, and even divided by the wide headroom it
should clear the MSE term, whose absolute gain is limited by the smeared geometry. That is my
falsifiable prediction; the split in the feedback table is what confirms or breaks it, and if the two
terms come back level or inverted, my reading of the mechanism is wrong.

Overall I expect a large jump off the floor — neighbour averaging genuinely does beat down Poisson
noise, so this should close a substantial fraction of the gap to the rate on this synthetic data,
likely more than it would on real tissue because the synthetic manifold is cleaner and the neighbour
relation less corrupted. And the computation is cheap enough not to matter: an exact nearest-neighbour
search over nine hundred cells in the roughly thousand-dimensional sqrt-normalized space is well under
a billion distance operations, nothing that needs approximation. But I also expect this rung to leave
a clear gap below whatever comes next, for the two reasons I have already named and quantified. The
neighbours are chosen on noisy profiles, so some of the pooling is simply wrong and the realized
variance reduction falls short of the ideal `1/m`; and the pool is a hard uniform average with a
single global `k`, so it cannot adapt its bandwidth to the local density of the manifold. There is one more thing the rung-1 anchors let me predict and then check: stability across the
held-out seeds. The raw MSE anchors on the three held-out seeds came back tight around `1.43` to
`1.49`, close to the tune seed's `1.60`, and the raw Poisson anchors clustered similarly — which tells
me the three held-out datasets are genuinely comparable draws, not wildly different regimes. Since this
method carries exactly one tunable knob, and a coarse one at that, I expect its held-out combined
score to track its tune-set score closely, with no meaningful over-fit: a single `k` chosen on one
seed has almost no capacity to memorize a particular partition. If instead the held-out score comes
back materially below the tune score, that would mean even one coarse knob found something seed-specific
to exploit, and I would distrust every richer rung above this one. So the tune-versus-held gap on this
rung is a second, cheaper calibration — of over-fitting risk rather than of scale — riding along on top
of the first real denoiser. Curing the two structural weaknesses I have quantified — the corrupted
neighbour relation and the non-adaptive hard pool — by denoising the space the neighbours are found in
and replacing the hard uniform pool with an adaptive-bandwidth, transitively-weighted average is the
next rung.
