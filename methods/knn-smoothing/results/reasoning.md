The raw baseline sits at exactly zero because it pools nothing — every cell is denoised by
itself, so none of the Poisson noise is ever averaged away. The only thing that can lift a denoiser
off that floor is to use the fact that I have many cells, not one. Cells in the same biological
state are independent noisy measurements of one underlying rate, so if I can gather, for each cell,
the other cells that share its state and average them, the sampling noise should beat down while the
shared signal survives. Before I reach for anything graph-theoretic I want the crudest honest
version of that idea, both because it is the natural first rung and because I want to measure how
much of the gap a bare neighbor average can actually close.

How much should averaging help, in principle? If a pool holds `m` cells that really are
independent `Poisson(μ)` draws of the same rate, the mean of the pool has variance `μ/m` — the noise
falls off like `1/m`. That is worth a number rather than a hand-wave, so I draw 500k pools of `m`
independent `Poisson(5)` cells and look at the variance of the pooled estimate:

```
k= 0  pool= 1  Var[mean]=4.99   μ/m=5.00
k= 4  pool= 5  Var[mean]=1.00   μ/m=1.00
k= 9  pool=10  Var[mean]=0.50   μ/m=0.50
k=19  pool=20  Var[mean]=0.25   μ/m=0.25
```

So a pool of ten cells cuts the noise variance tenfold. That is the entire upside of the rung, and
it is real — provided the `m` cells genuinely share a rate. Every difficulty below is some way that
assumption is violated.

So the recipe is: for each cell, find its `k` nearest neighbors, and replace its profile with the
average over itself and those `k` cells. Two decisions inside that sentence matter. The first is
*what space I measure distance in*. Raw counts look like a terrible space, because a cell sequenced
deeper has larger counts everywhere and so sits far from an identical shallow cell for a reason that
has nothing to do with biology. I should test whether that intuition is real before building on it.
Take two cells with identical expression composition `[2,4,6,8]` and the same profile scaled 4×
deeper, `[8,16,24,32]`, and measure their distance raw versus after a square-root then
normalize-to-unit-sum:

```
raw L2 distance        : 32.86
sqrt + library-norm    : 0.00
```

Raw distance calls two biologically identical cells the farthest-apart pair in this little set;
after removing depth they coincide exactly. So I have to library-size-normalize each cell to a common
total before comparing, or neighbor selection will track sequencing depth instead of state.

That experiment also used a square root, and I should say why rather than smuggle it in. The counts
are heavily right-skewed and heteroscedastic — for Poisson data the variance equals the mean, so a
few high-expression genes carry far more raw variance than the rest and would dominate a Euclidean
distance. The square root is the classical fix because `Var[√X]` is supposed to be roughly constant
regardless of the mean. I have always taken that on faith; here I can just check it. Drawing 2M
`Poisson(μ)` samples at a range of means:

```
μ=  1   Var[X]=  1.00   Var[√X]=0.402
μ=  5   Var[X]=  5.00   Var[√X]=0.286
μ= 10   Var[X]=  9.99   Var[√X]=0.261
μ= 50   Var[X]= 50.02   Var[√X]=0.252
μ=100   Var[X]= 99.93   Var[√X]=0.251
```

The raw variance climbs linearly with the mean, exactly as Poisson predicts; the square-rooted
variance flattens out toward `1/4` and is essentially mean-independent by `μ≈10`. It is not perfect
at very low counts (0.40 at `μ=1`, where dropout lives), but it is the right move: it stops the
loudest genes from hijacking the distance. So the distance space is: square-root the counts,
normalize to a common library size, *then* measure neighbors.

The second decision is where to look for neighbors, and here I want to be deliberately crude. The
simplest choice is to pick neighbors directly on the observed (noisy) normalized profiles — not to
first denoise the space I am searching in. That is the whole character of this rung and also its
known weakness. In a high-dropout regime the observed profiles are sparse and noisy, so the
nearest-neighbor relation is itself corrupted: a cell's true neighbors may be pushed away and
impostors pulled in, purely by which genes happened to drop out in each. I am going to accept that
weakness on purpose, because curing it is exactly what a later graph construction would be for. For
now, neighbors on the raw normalized profiles.

Then the averaging. I take the cell plus its `k` neighbors and average their normalized profiles
uniformly — a *hard, uniform* pool, every neighbor weighted equally, with a cliff at the `k`-th
neighbor. After averaging in the stabilized normalized space I have to undo the transform so the
output lands on the count scale the Poisson metric expects: square the averaged value, and multiply
each cell back by its own library factor so it keeps its own depth rather than being flattened to the
group mean.

I should run the whole thing once on something I can read by eye, to catch the obvious mistakes —
wrong shape, negative outputs, depth collapsed, or neighbors pooled across states. Four cells, two
tight clusters (two high in gene 0, two high in gene 1), `knn=1`:

```
in  library sums : [11, 13, 10, 12]
out library sums : [10.68, 12.29, 9.69, 11.28]
out:
 [10.215  0.     0.250  0.217]   # cluster A
 [11.750  0.     0.288  0.250]   # cluster A
 [ 0.     9.221  0.215  0.250]   # cluster B
 [ 0.    10.739  0.250  0.291]]  # cluster B
```

Shape preserved, every entry non-negative, and each cell pooled with its true cluster partner rather
than across clusters — the gene-0 cells stay gene-0-dominant, the gene-1 cells stay gene-1-dominant.
The dropout zeros even get partially filled: cell 1's gene-3 count was 0 and comes back as 0.25 from
its partner. One thing I did not expect and should be honest about: the output library sums are not
exactly the inputs — 11 went to 10.68, 13 to 12.29. That is because the library factor is computed
in square-root space and then re-applied to an *averaged* sqrt-profile whose sum is no longer the
cell's own; squaring afterward does not put it back. So per-cell depth is approximately, not
exactly, preserved. For this denoising purpose that is fine — the point was to keep cells at their
own scale rather than collapse them to one mean, and they clearly do — but I would not advertise it
as exact normalization.

Now `k` itself, which is a genuine bias-variance lever. Small `k` averages few cells, so it removes
little noise (the `μ/m` table says pool of 5 still leaves a fifth of the variance) but distorts
little signal; large `k` removes more noise but, because the pool is a hard uniform average, it
eventually reaches past the boundary of the local state and blends in cells that are not quite the
same, smearing real biological variation. And there is no single `k` that is right everywhere: a
cell in a dense region has many genuine neighbors and could tolerate a large pool, while a cell on
the edge of a trajectory has few and a large `k` drags in strangers. That a single global `k` cannot
be right for both is the structural flaw of the whole approach. I will tune one value on the tune
set and report it honestly; given the `μ/m` curve — most of the variance reduction is already bought
by a pool of around ten — I would start searching near `k≈10` and not expect to need it much larger.

What I expect from the feedback, then, is a real jump off zero: the variance check says a ten-cell
pool removes ninety percent of the per-gene noise, so neighbor averaging should close a large
fraction of the gap to the rate. But I also expect a clear gap remaining below a graph-diffusion
rung, for the two reasons I built in on purpose. The neighbors are chosen on noisy profiles, so some
of the pooling is simply wrong — the depth check shows the distance becomes meaningful once I
normalize, but it says nothing about dropout corrupting *which* cell is nearest, and that corruption
is exactly what survives here. And the pool is a hard uniform average with one global `k`, so it
cannot adapt its bandwidth to the local density of the manifold the way a weighted affinity graph
can. Denoising the space neighbors are found in, and replacing the hard uniform pool with an
adaptive-bandwidth, transitive diffusion, is what the next rung would have to do.
