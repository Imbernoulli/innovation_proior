The kNN rung closed a large fraction of the gap — held-out combined ~`0.80` — but the split between
the two terms points cleanly at what to fix. The Poisson term recovered to ~`0.87` while the
log-normalized MSE lagged at ~`0.73`, exactly the ordering the mechanism predicted: hard uniform
averaging crushes the over-dispersion the likelihood punishes, so the Poisson term jumps, but
recovering the log-normalized *shape* needs the local geometry right, and my geometry was corrupted by
choosing neighbours on noisy profiles and blurred by a hard uniform pool with one global `k`. So the
two crude choices I made on purpose are the ones the MSE term now bills me for. Both complaints point
one way: a smoother, weighted, geometry-aware notion of "similar cell," and no single noisy distance
defining a hard set.

Before building the fix, read the anchors to see where the gain actually converts. The MSE headroom
from raw to perfect is only ~`0.55`, so the kNN MSE term at `0.73` leaves an absolute gap of roughly
`(1 − 0.73)·0.55 ≈ 0.15`. The Poisson headroom is ~`2.98`, so the term at `0.87` leaves ~`(1 − 0.87)·
2.98 ≈ 0.40`. There is more raw room left on the Poisson side, but it is expensive — every unit of NLL
recovered is divided by the wide headroom and returns little normalized credit — whereas the MSE room
is small but cheap, each unit of squared error returning a large normalized move. So a shape-fixing
method converts best on the MSE term, which is precisely the term the kNN weaknesses damaged. I hold
that alignment as the prediction to test: a geometry-aware denoiser should post its gain most visibly
on the MSE term.

Start with the space, the root of the shape damage. Instead of measuring distances in the full noisy
gene space, I project the normalized cells onto their top principal components. The justification is
structural: the true rate has low-rank trajectory-and-branch structure with over-dispersion on top, so
most of the variance in any single gene is per-cell Poisson jitter and only the shared low-dimensional
directions carry biology. The leading components are those shared directions — the trajectories and
branches the cells live on — while the trailing components are where per-gene sampling noise piles up.
Truncating to a few dozen is therefore a linear denoiser of the distance itself, raising the
neighbour-relation signal-to-noise I named as the kNN rung's ceiling. Order of magnitude: if per-gene
noise were roughly isotropic across the ~1000 gene directions, keeping 50 components retains
essentially all the signal plus only the 50 largest slices of noise, discarding on the order of 95% of
the isotropic noise energy. The noise is not perfectly isotropic and PCA does greedily keep the largest
noise directions along with the signal, so the real gain is somewhat less, but the direction and rough
scale are right, and it is far more than tuning `k` on the corrupted full-space distance could buy.
Fifty sits comfortably above the handful of dimensions a few branching trajectories occupy and below
the point where I would be reconstructing individual genes' noise. The obvious cheaper alternative —
iterate the kNN smoother and search on its output — is circular and still hard-pooled, laundering the
same corrupted relation through itself; PCA is the clean linear move. So: square-root transform,
library-normalize, PCA to fifty components, build the neighbour structure there, which also makes the
search cheaper.

Now the weighting, for the hard-pool half of the complaint. Rather than a hard top-`k` set with a
cliff, I want a soft affinity that falls off smoothly with distance and — the part that fixes the
single-global-`k` flaw — whose bandwidth adapts to local density. A cell in a dense region wants a
narrow kernel, one on a sparse trajectory edge a wide one, and no single width serves both. The
per-cell width comes for free from each cell's distance to its `k`-th neighbour: small where cells are
packed, large where they are spread. With `σ_i` set that way the affinity is `exp(−(d_ij/σ_i)^α)`, an
alpha-decay kernel. At `α = 1` it is a plain exponential with a heavy tail, so distant cells still leak
in weight; as `α` grows the decay steepens toward a near-hard cutoff. `α = 2` sits between — Gaussian-
like, smooth near the origin, falling off fast enough in the tail to keep genuinely distant cells out
without the discontinuity of a hard set. That is the standard choice, which I fix rather than tune,
since the parameter that actually matters is the diffusion depth. A plain fixed-width Gaussian I reject
for the same reason as a global `k`: it reintroduces the density problem. I symmetrize the affinity so
the relation is mutual and row-normalize it into a Markov transition matrix `P`, so each row sums to
one and applying `P` replaces each cell by a convex, affinity-weighted combination of its neighbours —
creating no mass and pushing no value outside its neighbours' range.

One decision inside is easy to get wrong: I build the graph on the fifty-dimensional PCA embedding but
diffuse the *full-gene* normalized matrix, `Pᵗ X` in the thousand-gene space, not the PCA coordinates.
The PCA is only there to make the distance geometry reliable; it is not the thing I want to
reconstruct. Diffusing the coordinates and projecting back would force the output to be exactly
rank-fifty — a stronger claim than the data supports, since the true rate is low-rank *plus*
multiplicative over-dispersion, so a hard truncation would discard real per-gene variation, not just
noise. So I separate the roles: the low-rank embedding decides *who* averages with whom, the
full-resolution gene values get averaged, letting each gene keep whatever structure survives the
diffusion. Whether some gentle global low-rank structure is worth adding back on top is a real question
with its own over-dispersion risks, left for later rather than smuggled in as a truncation here. The
construction stays cheap and deterministic — search in fifty dimensions, a sparse affinity with ~30
entries per row, a handful of sparse matrix-vector products, cost dominated by the PCA, which I pin
with a fixed random state.

Here is the part that separates this from kNN. With a transition matrix in hand I diffuse, `X̂ = Pᵗ X`.
One step averages each cell with its affinity-weighted neighbours — already a soft, adaptive-bandwidth
version of the kNN pool. But `Pᵗ` for `t > 1` lets information flow *transitively*: cell `i` borrows
from its neighbours' neighbours, weighted by the probability of reaching them in `t` hops along the
manifold. On a smooth trajectory this is exactly right — two cells a few steps apart measure nearly the
same rate and should pool, even without direct adjacency. On a locally one-dimensional trajectory `P`
is approximately banded, and a `t`-th power of a banded matrix has bandwidth roughly `t` times as wide,
so `Pᵗ` pools an effective neighbourhood of order `t` hops, but with the far reaches decaying like the
product of transition probabilities along the path. That is why transitivity is not the same as
enlarging `k`: enlarging `k` pools far cells at full uniform weight and smears across states, whereas
`Pᵗ` reaches them only through the intervening chain at exponentially decaying weight, pooling along the
continuum without jumping across it. But powering `P` has a dangerous far end. `P` is row-stochastic,
its largest eigenvalue exactly one and every other mode below one, and `Pᵗ` damps a mode with
eigenvalue `λ` as `λᵗ`. Small `t` preserves the fine structure distinguishing nearby states; large `t`
drives every subdominant mode to zero and washes every cell toward one global average. So `t` is the
same bias-variance dial as `k`, but continuous and geometry-weighted, and it contracts toward consensus
fast. I expect the sweet spot small — a couple of steps — because the adaptive affinity already does
most of the pooling in one step and the powers are there for transitive reach, not to grind everything
flat. I sweep `t = 1, 2, 3, 4, 6`: at `t = 1` a soft density-adaptive neighbour average, not yet using
the transitivity that is the whole point of holding a transition matrix; at `t = 2` one hop further
along the manifold at decayed weight, at or near the peak; from `t = 3` on the subdominant modes damp
hard enough to eat real between-state structure, so I expect a monotone decline. A peak at `t = 4` or
higher would mean the manifold is smoother than I assume; a win at `t = 1` would mean transitivity buys
nothing here and the graph is effectively a soft kNN.

The diffusion runs in the same square-root, library-normalized space the graph was built in, then I
invert as before — square, restore each cell's library size — landing back on the non-negative count
scale.

Two properties of diffusing in this space I want on the record. First, because `P` averages each cell
into a convex combination of its neighbours, an exact zero — a dropout — is automatically filled by
whatever its neighbours read on that gene, so the diffusion imputes dropouts with no special
zero-handling. But it fills every zero the same way, whether a dropout of a truly-expressed gene or a
genuine biological zero, since nothing distinguishes them — a limitation I carry forward. Second, the
tension I flagged last rung, now sharper: I diffuse in the square-root space where the affinity geometry
makes sense, but the MSE is scored in log-normalized space, and diffusion in one space is not the same
operation as in the other. I choose the clean geometry on purpose, accepting that I never directly touch
the space the MSE is computed in.

So the prediction I hold against the feedback is specific: this should clear kNN, and clear it on the
MSE term in particular, because both fixes — a PCA-denoised embedding and an adaptive-bandwidth soft
kernel — are aimed at the log-normalized shape hard uniform pooling smeared, which is also the term
where recovery is cheapest. The Poisson term was already high and its wide headroom makes further
absolute gains convert poorly, so I expect it to improve only modestly. Concretely I expect the raw MSE
to fall by a few hundredths and the raw NLL to stay nearly flat, so the gain shows up almost entirely
as a rise in the normalized MSE term. If instead the gain lands mostly on the Poisson term, or the MSE
sticks at `0.73`, my reading of *why* kNN's MSE lagged is wrong. Two things this still will not fix —
the next rung's targets: a single square-root transform for every gene, when its stabilization is
non-monotone across the dropout spectrum; and diffusion that smooths in the VST space but never targets
the log-normalized scoring space, and captures only the local manifold geometry, never the global
low-rank structure a factorization on top could recover.
