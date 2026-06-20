The kNN rung left me with two specific complaints, and I want to fix both at once. The first is
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
embedding are far more reliable than distances on raw profiles, which directly addresses the first
complaint. So: square-root transform, library-size normalize, PCA to a few dozen components, and
build the neighbor structure *there*.

Now the weighting, which addresses the second complaint. Rather than a hard top-`k` set, I want a
soft affinity that falls off smoothly with distance — and crucially, an affinity whose bandwidth
adapts to local density. A cell in a dense region should have a narrow kernel, because its genuine
neighbors are close; a cell in a sparse region should have a wide one, because its neighbors are
farther but still real. The way to get this is to set each cell's kernel width from its own distance
to its `k`-th neighbor, so the kernel is automatically narrow where cells are packed and wide where
they are spread out. This is the alpha-decay kernel that MAGIC uses: `exp(−(d/σ_i)^α)` with `σ_i`
the adaptive per-cell bandwidth and `α` controlling how sharply the affinity decays. Symmetrize the
resulting affinity matrix so the relation is mutual, and row-normalize it into a Markov transition
matrix `P` — now `P[i,j]` is the probability of stepping from cell `i` to cell `j` in one diffusion
step, large for similar cells and smoothly small for dissimilar ones.

Here is the part that genuinely separates this from kNN. With a transition matrix in hand, I do not
have to stop at immediate neighbors. I diffuse: `X̂ = Pᵗ X`. One step of `P` averages each cell with
its affinity-weighted neighbors, exactly like a soft kNN. But `Pᵗ` for `t > 1` lets information flow
*transitively* — cell `i` borrows from its neighbors' neighbors, weighted by the probability of
reaching them in `t` steps along the manifold. On a smooth trajectory this is precisely right: cells
that are a few hops apart along the continuum *should* share information, and the powered transition
matrix reaches them through the chain of intermediate cells rather than requiring them to be direct
nearest neighbors. The hard cliff of kNN is gone; in its place is a smooth, distance-weighted,
transitively-pooled average whose bandwidth adapts to the data.

The diffusion is run in the same square-root, library-normalized space I built the graph in, then
inverted — square the diffused values and restore each cell's library size — so the output lands on
the count scale. The free parameter that matters most is `t`, the number of diffusion steps, and it
is the same bias-variance lever as `k` was, but better-behaved: small `t` under-smooths, large `t`
over-diffuses and eventually washes every cell toward the global mean, erasing the real biological
variation I am trying to preserve. There is a sweet spot, and I will tune it on the tune set; I
expect something small, a handful of steps, because the affinity weighting already does most of the
work in a single step and the powers are there to reach transitively, not to grind everything flat.
The kernel decay `α` and the neighbor count `k` setting the bandwidth are secondary knobs I will
fix to standard values.

I expect this to clear the kNN rung, and to clear it specifically on the MSE term — the
log-normalized shape — because the adaptive weighting and reliable embedding recover the local
geometry that hard uniform pooling smeared. What I do not expect it to fix is two things that will
become the next rung's targets. First, I am committing to a *single* variance-stabilizing transform,
the square root, for every gene — but genes with very different dropout rates are stabilized best by
different transforms, and a high-dropout gene and a low-dropout gene should not be forced through
the same nonlinearity. Second, plain diffusion smooths in the count space but never directly targets
the log-normalized metric the MSE is computed in, nor does it exploit the global low-rank structure
that a factorization could capture on top of the local diffusion. Those two gaps — gene-adaptive
transforms and a refinement that targets the actual scoring space — are what the endpoint rung is
built to close.
