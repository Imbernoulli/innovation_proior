The kNN rung closed a large fraction of the gap — a held-out combined of about `0.80` — but the split
between the two terms is the interesting part and it points cleanly at what to fix. The Poisson term
recovered to about `0.87` while the log-normalized MSE term lagged at about `0.73`, and that ordering
is exactly what I predicted from the mechanism: hard uniform averaging is a variance-beater, so it
crushes the over-dispersion the likelihood punishes and the Poisson term jumps, but recovering the
log-normalized *shape* needs the local geometry to be right, and my geometry was corrupted by choosing
neighbours on noisy profiles and blurred by a hard uniform pool with one global `k`. So the very two
crude choices I made on purpose are the ones the MSE term is now billing me for. Both complaints point
in the same direction: I need a smoother, weighted, geometry-aware notion of "similar cell," and I need
to stop trusting a single noisy distance to define a hard set.

Before I build the fix, let me read the anchors to see where the efficient gain actually is, because
the two terms are on very different exchange rates and I do not want to spend effort where it does not
convert. The MSE headroom from raw to perfect is only about `0.55`, so the kNN MSE term at `0.73`
leaves an absolute gap of roughly `(1 − 0.73)·0.55 ≈ 0.15` still to close. The Poisson headroom is
about `2.98`, five and a half times wider, so the kNN Poisson term at `0.87` leaves an absolute gap of
roughly `(1 − 0.87)·2.98 ≈ 0.40`. In raw metric units there is more room left on the Poisson side, but
it is expensive room — every unit of NLL I recover is divided by that wide headroom and returns little
normalized credit — whereas the MSE room is small but cheap, each unit of squared error returning a
large normalized move. So the place a shape-fixing method converts best is the MSE term, and that is
precisely the term the kNN weaknesses damaged. That alignment — the term with the corrupted geometry is
also the term where recovery is cheapest in normalized credit — is what tells me a geometry-aware
denoiser should post its gain most visibly on the MSE term. I will hold that as the prediction to test.

Start with the space, because the corrupted-neighbour problem is the root of the shape damage. Instead
of measuring distances in the full noisy gene space, I first project the normalized cells onto their
top principal components. The justification is structural: the true rate here has low-rank trajectory
and branch structure with over-dispersion sprinkled on top, so most of the variance in any single gene
is per-cell Poisson jitter, and only the shared, low-dimensional directions carry the biology. The
leading principal components are exactly those shared directions — they capture the trajectories and
branches the cells actually live on — while the trailing components are where the per-gene sampling
noise piles up. Truncating to a few dozen components is therefore a linear denoiser of the distance
itself: it raises the signal-to-noise ratio of the neighbour relation I quantified as the kNN rung's
ceiling, keeping the between-state signal and discarding much of the within-state noise. How many
components? The matrix is nine hundred cells by a thousand genes, so I can take up to eight hundred and
ninety-nine, but I want far fewer — enough to span the trajectory-plus-branch structure and no more,
because every extra component past the real rank is just readmitting noise I meant to drop. Fifty is a
standard and defensible choice: comfortably above the handful of dimensions a few branching
trajectories occupy, comfortably below the point where I would be reconstructing individual genes'
noise. I can put an order of magnitude on the win. If the per-gene sampling noise were roughly
isotropic across the thousand gene directions, the total noise energy would be spread over all thousand
of them, while the biological signal is concentrated in the few dozen directions the trajectories and
branches span. Keeping fifty components retains essentially all of that signal plus only the fifty
largest slices of noise, discarding on the order of ninety-five percent of the isotropic noise energy —
so the neighbour-distance signal-to-noise ratio, the exact quantity I named as the kNN rung's ceiling,
improves by something like an order of magnitude. The noise is not perfectly isotropic and PCA does
greedily keep the largest noise directions along with the signal, so the real gain is somewhat less
than that idealization, but the direction and rough scale are right, and it is far more than tuning
`k` on the corrupted full-space distance could ever have bought. I considered the obvious cheaper alternative — iterate the kNN smoother and search on its
output — but that is circular and still hard-pooled, and it would launder the same corrupted relation
through itself rather than denoise it; PCA is the clean linear move. So: square-root transform,
library-size normalize, PCA to fifty components, and build the neighbour structure there, which also
makes the search cheaper, in fifty dimensions instead of a thousand.

Now the weighting, which addresses the hard-pool half of the complaint. Rather than a hard top-`k`
set with a cliff at the boundary, I want a soft affinity that falls off smoothly with distance, and —
this is the part that fixes the single-global-`k` flaw — an affinity whose bandwidth adapts to local
density. The failure of one global bandwidth was concrete: a cell in a dense region has genuine
neighbours packed close, so it wants a narrow kernel, while a cell on a sparse trajectory edge has
real neighbours that are simply farther away, so it wants a wide one, and no single width serves both.
The way to get a per-cell width for free is to read it off each cell's own distance to its `k`-th
neighbour: where cells are packed that distance is small and the kernel is automatically narrow, where
they are spread it is large and the kernel widens to match. With `σ_i` set that way, the affinity is
`exp(−(d_ij/σ_i)^α)` — an alpha-decay kernel, with `α` controlling how sharply it falls. The exponent
is worth a moment: at `α = 1` the kernel is a plain exponential with a heavy tail, so distant cells
still leak in non-trivial weight; as `α` grows the decay steepens toward a near-hard cutoff, recovering
something like the kNN cliff in the limit. The value `α = 2` sits between these — a Gaussian-like decay
that is smooth near the origin and falls off fast enough in the tail to keep genuinely distant cells
from contributing, without the discontinuity of a hard set. That is the standard choice and I will fix
it there rather than spend tuning budget on it, because the parameter that actually matters is the
diffusion depth. I considered a plain fixed-width Gaussian and rejected it for exactly the reason I
rejected a global `k`: it reintroduces the density problem I am trying to solve. I symmetrize the resulting affinity so the
relation is mutual — if `i` reaches `j`, `j` should reach `i` — and row-normalize it into a Markov
transition matrix `P`, so `P_ij` is the probability of stepping from cell `i` to cell `j` in one
diffusion step, large for similar cells and smoothly small for dissimilar ones. Row-normalization is
what makes it a proper averaging operator: each row sums to one, so applying `P` to the data replaces
each cell by a convex, affinity-weighted combination of its neighbours, which cannot create mass out
of nothing and cannot push a value outside the range of its neighbours.

One decision inside this is easy to get wrong, and I want to be deliberate about it: I build the graph
on the fifty-dimensional PCA embedding, but I diffuse the *full-gene* normalized matrix, `Pᵗ X` in the
thousand-gene space, not the fifty PCA coordinates. The PCA is only there to make the distance geometry
reliable; it is not the thing I want to reconstruct. If I diffused the PCA coordinates and projected
back, I would be forcing the output to be exactly rank-fifty, and that is a stronger claim than I am
entitled to make about this data — the true rate is low-rank *plus* multiplicative over-dispersion, so
a hard rank-fifty truncation of the output would discard the per-gene variation that is genuinely
there, not just noise. So I separate the two roles cleanly: use the low-rank embedding to decide *who*
averages with whom, but average the full-resolution gene values, letting each gene keep whatever
structure survives the diffusion. Whether some *gentle* global low-rank structure is worth adding back
on top of the local diffusion is a real question, but it is a different move with its own risks around
that same over-dispersion, and I am leaving it for later rather than smuggling it in as a truncation
here. On the practical side the whole construction stays cheap and I need it to be deterministic: the
neighbour search now runs in fifty dimensions instead of a thousand, the affinity matrix is sparse with
about thirty entries per row, and the diffusion is a handful of sparse matrix-vector products, so the
cost is dominated by the PCA — which I pin with a fixed random state so the randomized solver returns
the same embedding every run and the harness sees a deterministic denoiser, as its contract requires.

Here is the part that genuinely separates this from kNN. With a transition matrix in hand I do not have
to stop at immediate neighbours; I diffuse, `X̂ = Pᵗ X`. One step of `P` averages each cell with its
affinity-weighted neighbours, which is already a soft, adaptive-bandwidth version of the kNN pool. But
`Pᵗ` for `t > 1` lets information flow *transitively*: cell `i` borrows from its neighbours' neighbours,
weighted by the probability of reaching them in `t` hops along the manifold. On a smooth trajectory
this is exactly right — two cells a few steps apart along the continuum are measuring nearly the same
rate and *should* pool, even though they are not direct nearest neighbours, and the powered transition
matrix reaches them through the chain of intermediate cells rather than requiring direct adjacency.
There is a clean way to see the reach: on a locally one-dimensional trajectory the transition matrix is
approximately banded, each cell connected to a small neighbourhood, and a `t`-th power of a banded
matrix has bandwidth roughly `t` times as wide, so `Pᵗ` lets each cell pool an effective neighbourhood
of order `t` hops along the manifold — but with the weight of the far reaches decaying like the product
of transition probabilities along the path, so distant cells contribute little even though they are
reachable. This is why transitivity is not the same as simply enlarging `k`: enlarging `k` would pool
far cells at full uniform weight and smear across states, whereas `Pᵗ` reaches them only through the
intervening chain and at exponentially decaying weight, pooling along the continuum without jumping
across it. The
hard cliff of kNN is gone, replaced by a smooth, distance-weighted, transitively-pooled average whose
bandwidth adapts to the data. But powering `P` is a lever with a dangerous far end, and I want to be
clear-eyed about it. `P` is row-stochastic, so its largest eigenvalue is exactly one, belonging to the
stationary distribution, and every other eigenvalue has modulus below one; `Pᵗ` damps the mode with
eigenvalue `λ` as `λᵗ`. Small `t` therefore preserves the fine, high-frequency structure that
distinguishes nearby states, while large `t` drives every subdominant mode toward zero and leaves only
the stationary consensus — which is to say, as `t` grows, every cell is washed toward one global
average and the biological variation I am trying to preserve is erased. I checked the endpoint of that
process on a small chain to make sure I had the direction right: on a six-cell toy graph with a signal
spanning a range of nine, powering a row-stochastic `P` shrank the spread from about `1.33` at one step
to `0.89` at two, `0.69` at four, `0.38` at eight, and essentially zero by thirty-two, while the mean
held fixed at the initial average throughout — so diffusion redistributes mass toward consensus rather
than creating or destroying it, and it does so fast enough that even single-digit `t` is already well
down the contraction curve. That confirms the
lever is exactly the same bias-variance dial as `k`, but better behaved because it is continuous and
geometry-weighted: too small under-smooths, too large over-diffuses to the global mean. I expect the
sweet spot to be small — a couple of steps — because the adaptive affinity already does most of the
pooling in a single step and the powers are there to reach transitively along the manifold, not to
grind everything flat. I will sweep `t` on the tune set and expect something like `t = 2`, with larger
`t` losing score monotonically as it over-diffuses; the decay `α` and the neighbour count `k` that
sets the bandwidth I fix to standard values rather than spend tuning budget on secondary knobs.

Concretely I will try `t = 1, 2, 3, 4, 6` and read the tune-set combined score, and I have a
prediction for the shape of that curve that follows from the spectral picture. At `t = 1` the diffusion
is a single soft, adaptive-bandwidth neighbour average — better than kNN because it is weighted and
density-adaptive, but it has not yet used the transitivity that is the whole point of holding a
transition matrix, so I expect it to improve on kNN but leave something on the table. At `t = 2` the
second power reaches one hop further along the manifold at decayed weight, which on a smooth trajectory
is exactly the reach that pools genuinely-similar cells the direct neighbourhood missed, so I expect
this to be at or very near the peak. From `t = 3` onward the subdominant modes are being damped hard
enough that the smoothing starts eating real between-state structure, and because the contraction
toward consensus is fast — the toy showed the spread already more than halved by four steps — I expect
the score to fall monotonically rather than plateau. So the falsifiable shape is: up from `t = 1` to a
peak at `t = 2`, then a clean monotone decline. If instead the peak sits at `t = 4` or higher, the
manifold is smoother and the states larger than I am assuming and the affinity is under-reaching in one
step; if `t = 1` already wins, the transitivity is buying nothing here and the graph is effectively a
soft kNN. Either outcome is diagnostic, which is why the sweep is worth running rather than fixing `t`
by fiat.

The diffusion runs in the same square-root, library-normalized space the graph was built in, then I
invert — square the diffused values and restore each cell's original library size — so the output lands
back on the count scale, non-negative by the squaring and shape-preserving cell-for-gene, satisfying
the harness contract exactly as the kNN rung did.

Two properties of running the diffusion in this space are worth naming now, because I want them on the
record as I read the feedback. The first is a quiet benefit: because `P` averages each cell into a
convex combination of its neighbours, an exact zero in one cell — a dropout — is automatically filled by
whatever its neighbours read on that gene, so the diffusion imputes dropouts without my writing any
special zero-handling. But it fills every zero the same way, whether it was a dropout of a gene the
cell truly expresses or a genuine biological zero the cell should keep, because nothing in the
construction distinguishes the two; that indiscriminate filling is a limitation I will carry forward
rather than fix here. The second is the tension I already flagged at the previous rung, now sharper. I
am diffusing in the square-root space — the Poisson-natural space where the affinity geometry makes
sense — but the MSE is scored in the log-normalized space, and diffusion in one space is not the same
operation as diffusion in the other. Smoothing where the geometry is clean and scoring where the metric
lives are pulling in slightly different directions, and I am choosing the clean geometry on purpose,
accepting that I am never directly touching the space the MSE is computed in. That mismatch is a second
named gap, and together with the single global transform it is what the endpoint will have to address.

So the prediction I will hold against the feedback is specific: this should clear kNN, and it should
clear it *on the MSE term* in particular, because the two fixes — a PCA-denoised embedding that raises
the neighbour signal-to-noise, and an adaptive-bandwidth soft kernel that recovers the local geometry —
are aimed squarely at the log-normalized shape that hard uniform pooling smeared, which is also the
term where normalized credit is cheapest. The Poisson term was already high after kNN, both because
any averaging fixes the gross over-dispersion the likelihood cares about and because its wide headroom
makes further absolute gains convert poorly, so I expect it to improve only modestly. I can even put
rough numbers on the prediction from the anchors: kNN's raw MSE was about `1.21` on the tune seed, and
a normalized MSE climbing from `0.73` into the low `0.8`s would mean the raw MSE falling by something
like `0.04` to `0.05`, toward roughly `1.16`, while the raw NLL — kNN's was about `−0.60` — should
barely move, since the Poisson term is near the top of its cheaply-recoverable range. So the signature
I am predicting is a raw MSE that drops by a few hundredths and a raw NLL that is nearly flat. If the
feedback instead shows the gain landing mostly on the Poisson term, or the MSE stubbornly stuck at
`0.73`, my reading of *why* kNN's MSE lagged is wrong and the fixes are not doing what I built them to. And I should be honest about the two things this construction still will not fix, because they
are what the next rung has to target. First, I am still committing to a *single* variance-stabilizing
transform, the square root, for every gene — and I already measured that the square root's stabilization
is non-monotone across the dropout spectrum, so genes at different dropout rates are being pushed
through a nonlinearity that suits none of them uniformly; a gene-aware family of transforms is left on
the table. Second, plain diffusion smooths in the count-or-VST space but never directly targets the
log-normalized space the MSE is actually computed in, and it captures only the *local* manifold
geometry, never the *global* low-rank structure that a factorization on top of the diffusion could
recover. Those two gaps — gene-adaptive transforms and a refinement that targets the actual scoring
space and the global structure — are what the endpoint rung is built to close.
