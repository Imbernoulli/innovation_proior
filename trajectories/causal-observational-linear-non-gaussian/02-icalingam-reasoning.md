The continuous-optimization floor told me, in numbers, exactly what it costs to ignore non-Gaussianity,
and the costs land precisely where the theory said they would. On ER30 it was strong but not clean —
F1 0.919, but seed 456 collapsed to F1 0.835 with SHD 35 while seeds 42 and 123 sat at SHD 9 and 4, so
the method is *unstable* on the denser small graph, one bad seed dragging the mean. The shape of the
errors is what should dictate the next engine, so read them as mechanism. Take SF100, the real
indictment: F1 0.716, SHD 136, precision 0.897 but
recall **0.597**. The scale-free graph here has about `3·(100−3) = 291` true directed edges. Recall
0.597 means I recovered roughly `0.597·291 ≈ 174` of them and simply *missed* about `117`; precision
0.897 means of the `174/0.897 ≈ 194` edges I did output, about `20` were spurious. Add the two error
piles, `117 + 20 ≈ 137`, and that is almost exactly the measured SHD of 136 — which tells me the errors
are overwhelmingly *missing* edges, not reversals and not false additions. That is a sharp, specific
fingerprint: on the hub-heavy graph the least-squares program is dropping roughly forty percent of the
true edges while keeping the ones it does emit almost all correct. Contrast the bad ER30 seed, where
SHD 35 with precision 0.802 and recall 0.871 decomposes to about `14` misses and `23` spurious edges — a
much more even split, the signature of a dense small graph confusing a few local orientations rather
than systematically failing on hubs. And ER50 (SHD 58, F1 0.872) decomposes similarly to about `36`
misses and `25` false out of `245` edges, doubling the nodes hurting more than doubling the samples
helped. So three scenarios, three error shapes, but one dominant story on the one that matters: the
missing-edge collapse on SF100.

Read that missing-edge signature as a mechanism. A method that keeps its edges mostly right but omits a
large fraction of the true ones is a method that, faced with an edge it cannot confidently orient,
prunes it rather than risk a reversal — and it cannot confidently orient the many edges incident to a
high-degree hub because it reads only second-order structure and, under the *uniform* (sub-Gaussian)
noise on SF100, it has the least higher-order signal to work with even if it wanted it. The diagnosis is
unambiguous: the ceiling on the previous method was set by its refusal to use non-Gaussianity, and the
orientation failures on the hard graph are exactly the gap. The fix is not a better optimizer; it is to
switch the engine to one whose *entire* mechanism is the non-Gaussian fingerprint.

To be sure non-Gaussianity is the right lever and not a hope, go back to the model and ask what
second-order statistics provably cannot do. The model is recursive linear,
`x_i = sum_{k(j) < k(i)} b_{ij} x_j + e_i`, disturbances `e_i` mutually independent, no hidden common
cause, acyclic. In matrix form `x = Bx + e`, and acyclicity means there is some ordering under which
`B` is *strictly* lower triangular — a variable cannot be its own parent and depends only on earlier
ones. I just do not know that ordering; in my data the rows are in arbitrary order, so `B` is permutable
to strict lower triangularity but not actually triangular as I see it. Now the wall the previous method
hit, made concrete. Suppose the `e_i` were Gaussian. Then `x` is jointly Gaussian, and a multivariate
Gaussian is *completely* described by its mean and covariance — nothing else is observable. Take two
variables. Model one: `x_1 = e_1`, `x_2 = 0.8 x_1 + e_2`, with `var(e_1)=1`, `var(e_2)=0.36`. Then
`var(x_1)=1`, `cov(x_1,x_2)=0.8·var(x_1)=0.8`, `var(x_2)=0.8^2·1 + 0.36 = 0.64 + 0.36 = 1`, so the
covariance matrix is `[[1, 0.8],[0.8, 1]]`. Model two, the reversed arrow: `x_1 = 0.8 x_2 + e_1`,
`x_2 = e_2`, with `var(e_2)=1`, `var(e_1)=0.36`. Then `var(x_2)=1`, `cov=0.8`, `var(x_1)=0.64+0.36=1` —
the *identical* covariance matrix `[[1,0.8],[0.8,1]]`. Two completely different causal claims, same mean,
same covariance, and since both are Gaussian, the *same joint density* down to the last moment. No
statistic can separate them. This is information-theoretic, not an estimator weakness: anything reading
only second-order structure recovers at best the Markov equivalence class — the skeleton plus
collider-forced orientations — and the `117` missing hub edges on SF100 are the visible cost of living
inside that class.

The covariance is a dead end *by assumption*, and the assumption is Gaussianity. Drop it. The hint that
non-Gaussianity is the right lever, not just a hope: for those same two variables, the direction *is*
recoverable once the noise is non-Gaussian — regress `x_2` on `x_1` and the residual is independent of
`x_1` in the causal direction but *dependent* in the anti-causal direction, an asymmetry that lives
entirely in the higher moments the covariance discards. That is a proof of concept that non-Gaussianity
carries the directional information second-order structure throws away. But it is two variables and gives
me neither coefficients nor an ordering for `d` variables. I need the structure that turns "non-Gaussianity
breaks the symmetry" into a general procedure over 30, 50, 100 nodes.

Stare at the model written the other way. From `x = Bx + e`, solve for `x`: `(I − B)x = e`, so
`x = (I − B)^{-1} e`. Call `A = (I − B)^{-1}`; then `x = A e`. Read it literally: my observed vector is
a fixed invertible linear mixing `A` of the disturbance vector `e`, whose components are — by my own
model assumption — mutually independent, and which I am now insisting are non-Gaussian. `x = A e` with
independent non-Gaussian `e` and invertible `A` is *exactly* the linear independent component analysis
model: observed data is an invertible linear mixture of independent non-Gaussian sources. Not a
metaphor — the same equations. The disturbances `e_i` are the independent components, `A` is the mixing
matrix, and `W = A^{-1} = I − B` is the separating matrix. And now the old Gaussian obstruction has a
precise opposite: the ICA mixing matrix is identifiable from the data when the sources are independent
and at most one is Gaussian — up to permutation, scaling, and sign of its columns, with *no* rotational
ambiguity. In this model every disturbance is non-Gaussian, so the condition holds with room to spare.
Contrast Gaussian sources, where any orthogonal rotation `A R` produces the same distribution (rotating
independent Gaussians gives independent Gaussians of the same covariance) — that rotational freedom is
precisely the `d(d−1)/2`-dimensional slack that left the previous method unable to orient. Non-Gaussianity
collapses that slack to a discrete residue, and that residue is the whole remaining problem.
Identifiability "up to permutation, scaling, sign"
means I recover `A`, hence `W = I − B`, hence `B`, once I resolve a permutation (one of `d!` row orders),
a per-row scaling (`d` continuous factors), and a per-row sign (`2^d` choices). Everything else — the
rotational continuum that blinded the covariance — is gone. So ICA hands me the raw material and the
causal content lives entirely in nailing down those `d!` orders, `d` scales, and `d` signs.

It sharpens the picture to see *where inside the ICA fit* the non-Gaussianity does its work, because that
is where the fragility will live. FastICA first *whitens* the data — a linear map `V` making
`cov(Vx) = I`, which uses up exactly the second-order structure — after which the remaining unmixing is
an *orthogonal* matrix `U`, so `W = U V` and the whole search collapses to finding the right rotation
among the `d(d−1)/2` orthogonal angles. A rotation of *Gaussian* sources is undetectable, which is
precisely why the covariance route could never pin direction: the direction lives entirely in those
rotation angles, and only non-Gaussianity distinguishes one rotation from another. So maximizing
non-Gaussianity over `U` recovers what the covariance threw away — but that objective is non-convex over
a `d(d−1)/2`-dimensional manifold, `4950` angles at `d = 100`, and that is the surface on which a global
fit can slip into a wrong basin.

Take scaling first — it is easiest and it tells me what "correct" means. In the SEM convention each
variable's own coefficient is one: the equation `x_i = (stuff) + e_i` has coefficient `+1` on `e_i`, so
`W = I − B` has an all-ones diagonal (`W_{ii} = 1 − b_{ii} = 1`, since there are no self-loops). That is
a fixed, known anchor. If the rows were in the right order, every diagonal entry of the true `W` would
be exactly one, and to remove ICA's per-row scaling I divide each row by its own diagonal entry. This
single division does double duty: if ICA returned a row scaled by some factor `c` (which may be negative),
its diagonal reads `c·1 = c`, and dividing the row by `c` restores the unit diagonal *and* flips the sign
back when `c < 0` — so scaling and sign, `d` continuous factors and `d` signs both, are resolved at once
the instant the row order is known. All the weight falls onto the permutation.

The permutation is the subtle one, and it is decidable because of the DAG structure rather than any
heuristic. The correct `W = I − B`: since `B` is permutable to strict lower triangular, `W` is permutable
(by the *same* row-and-column permutation) to lower triangular with a *nonzero* diagonal — the ones from
`I`. So the correctly ordered `W` is lower triangular with no zeros on its diagonal, and I claim exactly
one row order achieves a fully nonzero diagonal. Write the aligned `W = P_d M P_d^T` with `M` lower
triangular, nonzero diagonal, `P_d` the true causal order. ICA returns `W_ica = P_ica W = P_ica P_d M
P_d^T`; set `P_1 = P_ica P_d` (row permutation on `M`) and `P_2 = P_d` (column permutation), so
`W_ica = P_1 M P_2^T`. Let `K` be the all-ones lower-triangular matrix — the densest support `M` could
have. The number of potentially nonzero diagonal entries of `P_1 K P_2^T` is `tr(P_1 K P_2^T) =
tr(K P_2^T P_1)` by cyclicity of trace, so only the combined `Q = P_2^T P_1` matters. If `Q ≠ I`, some
column `i` must move to a position `j < i` (if none moved left, all moved right, and the finite set of
positions could not be filled); the diagonal entry landing in new column `j` comes from `K[j,i]` with
`j < i`, strictly *above* the lower-triangular support, hence zero — and since `K` is the densest
support, that diagonal slot is zero for any lower-triangular `M`. Conversely `Q = I` means `P_1 = P_2`,
the same relabeling applied to rows and columns, so the diagonal of `P_1 M P_1^T` is `M`'s nonzero
diagonal merely reordered, all nonzero. So a fully nonzero diagonal ⟺ `P_1 = P_2`, and the zero-free-diagonal
row permutation is unique and correct. The acyclicity is exactly what makes the permutation decidable —
the same structure the previous method had to *guess* at through magnitude thresholding, which is why it
reversed and missed hub edges. Here it hands me a clean criterion: find the row order making the diagonal
nonzero.

Finite data degrades that clean criterion. ICA on a sample never returns exact zeros — every entry of
`W_ica` is some small nonzero number, so *every* row permutation gives a technically nonzero diagonal,
and the exact "zero-free" test becomes vacuous. I need a soft version: prefer the permutation whose
diagonal entries are as *far from zero* as possible, since the correct alignment puts the structurally
large unit-ish entries on the diagonal and the structurally small (true-zero) entries off it. A natural
cost penalizes small diagonal magnitudes: take `sum_i 1/|W_ii|` after permutation and minimize it, so a
near-zero diagonal entry blows the cost up. And this is not ad hoc — model each disturbance with a
generalized-Gaussian density `log p(e) = −|e|^α/β + Z`; since a candidate row is later divided by its
diagonal `W_ii`, the value entering the density scales as `e/W_ii`, and the log-likelihood becomes, up
to constants, `−sum_i (1/(β|W_ii|^α)) sum_t |e_it|^α`. With identical component densities and `α = 1`
(the Laplace/double-exponential case), maximizing over the row correspondence is exactly
`min_perm sum_i 1/|W_ii|`. So the heavy penalty on small diagonals is not a trick, it is the
maximum-likelihood objective for the correspondence. Naively searching all `d!` permutations is hopeless
beyond a handful of nodes — the same `10^{32}`/`10^{157}` walls that killed order-based search — but the
objective has the shape of a *linear assignment problem*: if row `r` is placed at diagonal column `c`,
the contribution is `C_{r,c} = 1/|W_ica[r,c]|`, and minimizing the summed assignment cost is solved in
`O(d^3)` by the Hungarian algorithm. The cost magnitudes make the mechanism vivid: on the correct
ordering the diagonal holds entries near `1`, so their assignment cost is near `1`, while a true structural
zero has `|W_ica| ≈ 10^{-3}` and an assignment cost near `10^3` — so the Hungarian algorithm is strongly
repelled from ever parking a near-zero entry on the diagonal, which is exactly the correct behavior. I
build the cost matrix `C_{r,c} = 1/|W_ica[r,c]|`, run `linear_sum_assignment` to get the row-to-diagonal
matching, and scatter each row into its matched position so the matched entries land on the diagonal.
This `O(d^3)` step is cheap even at `d = 100` (`10^6` operations), so it scales straight through SF100
where the previous method's hub orientation broke down.

With the rows correctly ordered, scaling is the easy step set up earlier: take the diagonal `D` of the
permuted `W`, divide each row by its diagonal entry so the diagonal becomes all ones — now `W_estimate`
is an estimate of `I − B` in the right units, with scaling and sign resolved — and set `B = I − W`. But
for a clean directed graph I still want an explicit causal order under which `B` is strictly lower
triangular, and on finite data `B` has no exact zeros, so no permutation makes the upper triangle
exactly zero. If the estimates *were* exact, finding the order is easy: in `B` the row for variable `i`
holds its incoming coefficients, so an all-zero row is a variable with no remaining parents — a source.
A strict-lower-triangular structure always has at least one such row, so peel it off (record it, delete
its row and column) and repeat; peeling sources one at a time builds a valid topological order, and if
at some point no all-zero row exists the remainder is cyclic. For finite data I lean again on "true
zeros are small." A strict-lower-triangular `d × d` matrix has `d(d+1)/2` structural zeros — the diagonal
(`d`) plus the entire upper triangle (`d(d−1)/2`), which for `d=100` is `5050` of the `10000` entries. So
I sort all `d^2` entries of `B` by absolute value, set the `d(d+1)/2` smallest to exact zero, then walk
the remaining entries from smallest upward, zeroing one more at a time and running the peel test,
stopping at the first valid order. If in the true matrix all structural zeros are smaller than all true
nonzeros, the first valid peel is correct; in general it is a cheap, magnitude-informed greedy
approximation that gives the causal order.

But `B = I − W` is fully dense — every off-diagonal entry is some nonzero number, including all that
should be structural zeros. The order only made the upper triangle "as zero as possible"; it pruned
nothing. The honest, statistically grounded way to get the final edges is to *re-estimate* them by
regression along the discovered order with sparsity, and the harness already provides exactly this. The
edit hands the order to `causal-learn`'s `_BaseLiNGAM._estimate_adjacency_matrix`, which walks the
variables in causal order and, for each target, regresses it on its predecessors by **adaptive lasso**:
fit ordinary least squares for pilot coefficients `β̂`, form per-predictor weights `|β̂_j|^{-γ}` (with
`γ = 1`), fit a lasso whose penalty on predictor `j` is scaled by that weight and whose global strength
is chosen by BIC, and unweight the result. The weighting gives consistent selection — strong pilot
coefficients get a light penalty and survive, weak ones a heavy penalty and are zeroed, which is the
oracle property adaptive lasso is designed for — so spurious weak edges are pruned while true edges keep
good estimates. The first variable in the order has no predecessors and stays exogenous with an all-zero
row. The result fills the signed coefficients into `B[target, predecessors]`, returned with
`B[i,j] != 0` meaning `j -> i`. Note this is the same sparse re-estimation the continuous-optimization
floor approximated with a single hard threshold `ω = 0.3`; doing it as a BIC-selected lasso per node is
strictly more careful, since it adapts the penalty per predictor instead of applying one global cutoff,
which is part of why I expect the missing-edge problem to shrink.

One honest caveat I keep in front of me: the ICA step is a non-convex optimization, and it is the *only*
part of this pipeline that is. FastICA maximizes a `log cosh` negentropy contrast by a fixed-point
iteration with symmetric decorrelation, and that objective has multiple local optima; a run from a poor
initialization can settle into the wrong one, giving an unreliable `W` and hence a wrong graph. The cost
is not the worry — the fit is trivial within a 1000-iteration budget — the worry is that its `10^4`
unmixing parameters are estimated from a thin `~10:1` sample ratio on SF100, over a `100`-dimensional
landscape whose contrast is tuned for super-Gaussian sources and is weakest on that scenario's
sub-Gaussian uniform noise. I fix the random seed (from `SEED`) and give a generous budget, but I am
clear-eyed that this is the soft spot, as opposed to the permutation/scaling/order logic downstream,
which is exact given a good `W`. (The full module is in the answer.)

Now the falsifiable expectations against the previous numbers. ICA-LiNGAM *uses* non-Gaussianity, so on
the scenarios with strong higher-order signal it should clear the continuous program decisively, and
where it does not it should not improve much. ER30 has Laplace noise (heavy-tailed, strongly
super-Gaussian — ideal ICA fuel) and only 30 nodes: I expect near-perfect recovery, and specifically that
it fixes the seed-456 instability (F1 0.835, SHD 35) that dragged the previous mean down, since a FastICA
fit on heavy-tailed sources should not carry the orientation ambiguity that produced that seed's `~23`
false edges. ER50 has exponential noise (also strongly non-Gaussian) with 2000 samples — a healthy `40:1`
sample-to-parameter ratio — so I expect the SHD-58, F1-0.872 result to improve sharply as well. SF100 is
the test of whether this engine helps where the previous one most failed (F1 0.716, recall 0.597, `~117`
missing edges). Two forces fight: the assignment-plus-acyclic-order machinery should orient the hub edges
the least-squares program kept pruning, so I expect recall to rise substantially and the missing-edge
pile to shrink; but the sub-Gaussian uniform noise and thin `10:1` ratio stress the single global FastICA
fit, so I expect SF100 to improve yet remain the worst scenario by a wide margin, with the non-convexity
surfacing as high seed-to-seed variance — one seed landing in a bad basin. If that variance appears it
localizes the fault: not in non-Gaussianity, which won ER30 and ER50, but in the single global non-convex
fit I routed it through — and that is the reliance any stronger method must remove.
