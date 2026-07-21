The kNN-smoothing baseline left me with two specific complaints, and I want to fix both at once. The first is
that I chose neighbors on the raw, noisy profiles, so some of my pooling was simply wrong — impostor
cells dragged in by dropout, true neighbors pushed away. The second is that the pool was a hard
uniform average with a single global `k`: every neighbor counted equally, the boundary at the `k`-th
neighbor was a cliff, and one bandwidth had to serve both dense and sparse regions of the manifold.
Both complaints point in the same direction — I need a smoother, weighted, geometry-aware notion of
"similar cell," and I need to stop trusting a single noisy distance to define a hard set.

Start with the space. Instead of measuring distances in the full noisy gene space, I first project
the normalized cells onto their top principal components. Most of the genes are noise in any given
cell; the leading PCs capture the shared low-dimensional structure — the trajectories and branches
the cells actually live on — and discard the per-gene Poisson jitter. Distances in that denoised
embedding should be far more reliable than distances on raw profiles, which directly addresses the
first complaint. So: square-root transform, library-size normalize, PCA to a few dozen components,
and build the neighbor structure *there*.

Now the weighting, which addresses the second complaint. Rather than a hard top-`k` set, I want a
soft affinity that falls off smoothly with distance — and crucially, an affinity whose bandwidth
adapts to local density. A cell in a dense region should have a narrow kernel, because its genuine
neighbors are close; a cell in a sparse region should have a wide one, because its neighbors are
farther but still real. The obvious way to get this is to set each cell's kernel width from its own
distance to its `k`-th neighbor, so the width is automatically narrow where cells are packed and
wide where they are spread out — a kernel of the form `exp(−(d/σ_i)^α)` with `σ_i` the per-cell
distance to the `k`-th neighbor and `α` controlling how sharply the affinity decays.

Take two cells, one in a dense region whose `k`-th neighbor sits at distance `σ = 0.5`, one in a
sparse region with `σ = 3.0`, and `α = 2`. At `d = σ_i` the exponent is `−1` for both, so a neighbor
that is "the `k`-th one" gets the same vote — `exp(−1) = 0.368` — in both regions regardless of the
absolute scale: the bandwidth normalizes away the difference. But a neighbor sitting at a *fixed*
absolute distance `d = 1.0` is treated oppositely in the two regions: `exp(−(1/0.5)^2) = exp(−4) =
0.018` for the dense cell, essentially cut off, against `exp(−(1/3)^2) = exp(−0.111) = 0.895` for the
sparse cell, kept as a strong neighbor. The same physical gap is a wall for the packed cell and a
near neighbor for the spread-out one — exactly the density adaptation a single global bandwidth
cannot give.

Symmetrize the resulting affinity matrix so the relation is mutual, and row-normalize it into a
Markov transition matrix `P` — now `P[i,j]` is the probability of stepping from cell `i` to cell `j`
in one diffusion step, large for similar cells and smoothly small for dissimilar ones.

With a transition matrix in hand, I do not have to stop at immediate neighbors. One step of `P`
averages each cell with its affinity-weighted neighbors, which is already a soft version of kNN. But
nothing forces me to take a single step: I can impute by `X̂ = Pᵗ X`. The question is whether powering
`P` buys anything a one-step soft average does not. Take the smallest case that has the structure I
care about: four cells strung along a trajectory, `0–1–2–3`, each connected only to its immediate
neighbors. Row-normalizing the adjacency (with self-loops) gives

```
P    = [[.50 .50 .00 .00]
        [.33 .33 .33 .00]
        [.00 .33 .33 .33]
        [.00 .00 .33 .50]]
```

In one step, cell 0 can only see cells 0 and 1: `P[0,2] = 0`, so the soft average never touches cell
2 at all. Square it:

```
P²   = [[.417 .417 .167 .000]
        [.278 .389 .222 .111]
        [.111 .222 .389 .278]
        [.000 .167 .417 .417]]
```

Now `P²[0,2] = 0.167` — cell 0 *does* borrow from cell 2, even though they share no edge, because
the walk reaches it through the intermediate cell 1. That is the transitive pooling: a nonzero entry
that was zero one step earlier. The same example also gives the honest limit: `P²[0,3] = 0` exactly.
Two steps reach two hops and no farther, so `t` is literally the radius of pooling along the chain —
diffusion extends reach beyond direct neighbors, but the reach is controlled and finite, not an
uncontrolled blur. The hard cliff of kNN is replaced by a smooth, distance-weighted average that
pools out to a tunable number of hops along the continuum.

That immediately raises the worry on the other side: if a couple of steps already pull in
two-hop cells, what stops me from cranking `t` up for ever-more smoothing? I can answer that from the
same toy chain by pushing it to the limit. Take `P` to a high power:

```
P⁶⁴ row 0 = [.20 .30 .30 .20]
P⁶⁴ row 3 = [.20 .30 .30 .20]
```

Every row has become identical — cell 0 and cell 3 now impute to the *same* mixture — and that
mixture `[.2 .3 .3 .2]` is exactly proportional to the cells' degrees `[2 3 3 2]`, i.e. the
stationary distribution of the walk. So in the large-`t` limit every cell is replaced by one
common, geometry-weighted global average, and all the biological variation between cells is gone.
That is the over-smoothing failure made concrete: `t` too small under-pools, `t` too large collapses
toward a single profile, and the useful regime is in between. This is the same bias–variance lever
`k` was for kNN-smoothing, but better-behaved, because the affinity weighting already does most of the
denoising in a single step and the powers are there only to reach transitively, not to grind
everything flat. I therefore expect the sweet spot to be a handful of steps; I will tune `t` on the
tune set and expect it to land small. The kernel decay `α` and the neighbor count `k` setting the
bandwidth are secondary knobs I will fix to standard values (`α = 2`, a Gaussian-like decay; `k`
around ten).

The diffusion is run in the same square-root, library-normalized space I built the graph in, then
inverted — square the diffused values and restore each cell's library size — so the output lands on
the count scale.

I expect this to beat kNN-smoothing, and specifically on the MSE term — the log-normalized shape —
because the adaptive weighting and reliable embedding recover the local geometry that hard uniform
pooling smeared, and the few-hop diffusion fills dropout without the cliff artifacts. Two things I do
not expect this to fix. First, I am committing to a *single* variance-stabilizing transform, the
square root, for every gene — but genes with very different dropout rates are stabilized best by
different transforms, and a high-dropout gene and a low-dropout gene should not be forced through the
same nonlinearity. Second, plain diffusion smooths in the count space but never directly targets the
log-normalized metric the MSE is computed in, nor does it exploit the global low-rank structure that
a factorization could capture on top of the local diffusion. Gene-adaptive transforms and a
refinement that targets the scoring space directly are gaps left for a further method to close.
