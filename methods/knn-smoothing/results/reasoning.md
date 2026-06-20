The raw baseline sits at exactly zero because it pools nothing. The single idea that lifts a
denoiser off the floor is that cells in the same biological state are independent noisy
measurements of one underlying rate — so if I can find, for each cell, the other cells that share
its state and average them together, the Poisson noise averages down while the shared signal
survives. This is the oldest and simplest move in the imputation literature, and I want to start
with it in its crudest honest form before reaching for anything graph-theoretic, both because it is
the natural next rung and because I want to see how much of the gap to the rate a bare neighbor
average can close.

So the recipe is: for each cell, find its `k` nearest neighbors, and replace its profile with the
average over itself and those `k` cells. Two decisions inside that sentence matter. The first is
*what space I measure distance in*. Raw counts are a terrible space for distance — a cell that
happened to be sequenced deeper looks far from an identical shallow cell purely because its counts
are larger. I have to remove the depth before comparing, so I library-size-normalize each cell to a
common total. And the counts are heavily right-skewed and heteroscedastic — a few high-expression
genes would dominate a Euclidean distance — so I want a variance-stabilizing transform first. The
square root is the canonical choice for Poisson data: it makes the variance of a count roughly
constant regardless of its mean, so the distance is not hijacked by the loudest genes. Square-root
the counts, normalize to a common library size, and *then* measure neighbors.

The second decision is the one I want to be deliberately crude about: where do I look for neighbors?
The honest, classical kNN-smoothing of Wagner and colleagues picks neighbors directly on the
observed (noisy) normalized profiles — it does not first denoise the space it searches in. That is
the whole character of this rung and also its known weakness. In a high-dropout regime the observed
profiles are sparse and noisy, so the nearest-neighbor relation is itself corrupted: a cell's true
neighbors may be pushed away and impostors pulled in, purely by which genes happened to drop out in
each. I am going to accept that weakness on purpose, because curing it is precisely what the next
rung's graph construction is for. For now, neighbors on the raw normalized profiles.

Then the averaging. I take the cell plus its `k` neighbors and average their normalized profiles
uniformly. This is a *hard, uniform* pool — every neighbor counts equally and the boundary at the
`k`-th neighbor is a cliff. After averaging in the stabilized normalized space, I undo the
transform: square the result and multiply each cell back by its own original library size, so the
output is on the count scale the Poisson metric expects and each cell keeps its own depth rather
than being flattened to the average.

I have to pick `k`, and the choice is a genuine bias-variance lever. Small `k` averages few cells,
so it removes little noise but distorts little signal; large `k` removes more noise but, because the
pool is a hard uniform average, it inevitably reaches across the boundary of the local state and
blends in cells that are not quite the same, smearing real biological variation. There is no `k`
that is right everywhere: cells in a dense region have many genuine neighbors and could tolerate a
large pool, while cells on the edge of a trajectory have few and a large `k` drags in strangers.
That a single global `k` cannot be right for both is the structural flaw of the whole approach, but
I will tune one value on the tune set and report it honestly — I expect something in the range of
ten neighbors to be a reasonable compromise.

What I expect from the feedback is a real jump off zero — neighbor averaging genuinely does beat
down Poisson noise, so this should close a large fraction of the gap to the rate. But I also expect
it to leave a clear gap below the graph-diffusion rung, for two reasons I have already named. The
neighbors are chosen on noisy profiles, so some of the pooling is simply wrong; and the pool is a
hard uniform average with a single global `k`, so it cannot adapt its bandwidth to the local density
of the manifold the way a properly weighted affinity graph can. Curing exactly those two
weaknesses — denoising the space neighbors are found in, and replacing the hard uniform pool with an
adaptive-bandwidth, transitive diffusion — is the next rung.
