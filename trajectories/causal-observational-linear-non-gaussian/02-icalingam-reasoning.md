The continuous-optimization rung told me, in numbers, exactly what it costs to ignore non-Gaussianity,
and the costs land precisely where the theory said they would. On ER30 it was strong but not clean —
F1 0.919, but seed 456 collapsed to F1 0.835 with SHD 35 while seeds 42 and 123 sat at SHD 9 and 4, so
the method is *unstable* on the denser small graph, one bad seed dragging the mean. On ER50, the larger
graph with twice the samples, F1 fell to 0.872 and SHD ballooned to a mean of 58 — every seed in the
40s–80s — so doubling the nodes hurt more than doubling the samples helped. And SF100 is the real
indictment: F1 0.716, SHD 136, with a glaring asymmetry — precision 0.897 but recall **0.597**. Read
that asymmetry: on the hub-heavy scale-free graph the least-squares program is *missing* roughly forty
percent of the true edges (low recall) while the edges it does keep are mostly right (decent precision).
That is the signature of a method that cannot orient edges into and out of high-degree hubs from
second-order structure alone — it threshold-prunes ambiguous hub edges rather than risk a reversal, and
under the uniform (sub-Gaussian) noise on SF100 it has the *least* higher-order signal to work with
even if it wanted it. The diagnosis is unambiguous: the ceiling on the previous rung was set by its
refusal to use non-Gaussianity, and the orientation failures on the hard graph are exactly the gap. The
fix is not a better optimizer; it is to switch the engine to one whose *entire* mechanism is the
non-Gaussian fingerprint.

So let me go back to the model and ask what second-order statistics provably cannot do, to be sure
non-Gaussianity is the right lever and not a hope. The model is recursive linear,
`x_i = sum_{k(j) < k(i)} b_{ij} x_j + e_i`, disturbances `e_i` mutually independent, no hidden common
cause, acyclic. In matrix form `x = Bx + e`, and acyclicity means there is some ordering under which
`B` is *strictly* lower triangular — a variable cannot be its own parent and depends only on earlier
ones. I just do not know that ordering; in my data the rows are in arbitrary order, so `B` is permutable
to strict lower triangularity but not actually triangular as I see it. Now the wall the previous rung
hit. Suppose the `e_i` were Gaussian. Then `x` is jointly Gaussian, and a multivariate Gaussian is
*completely* described by its mean and covariance — nothing else is observable. Take two variables.
Model one: `x_1 = e_1`, `x_2 = 0.8 x_1 + e_2`, with `var(e_1)=1`, `var(e_2)=0.36`. Then `var(x_1)=1`,
`var(x_2) = 0.64 + 0.36 = 1`, `cov = 0.8`. Model two, the reversed arrow: `x_1 = 0.8 x_2 + e_1`,
`x_2 = e_2`, with `var(e_2)=1`, `var(e_1)=0.36`. Same variances, same covariance — and since both are
Gaussian, the *same joint density*. Two completely different causal claims, no statistic can tell them
apart. This is information-theoretic, not an estimator weakness: anything reading only second-order
structure recovers at best the Markov equivalence class — the skeleton plus collider-forced
orientations. That is precisely the regime the least-squares program lived in, and its missing-edge
recall on SF100 is the visible cost of being stuck there.

The covariance is a dead end *by assumption*, and the assumption is Gaussianity. Drop it. The hint that
this is the right lever: for two variables, cause-effect direction *is* recoverable when the noise is
non-Gaussian — the asymmetry shows up in higher moments, in regression residuals being non-normal in a
direction-dependent way. That is a proof of concept that non-Gaussianity carries the directional
information the covariance throws away. But it is two variables and gives me neither coefficients nor an
ordering for `d` variables. I need the structure that turns "non-Gaussianity breaks the symmetry" into
a general procedure.

Stare at the model written the other way. From `x = Bx + e`, solve for `x`: `(I − B)x = e`, so
`x = (I − B)^{-1} e`. Call `A = (I − B)^{-1}`; then `x = A e`. Read it: my observed vector is a fixed
linear mixing `A` of the disturbance vector `e`, whose components are — by my own model assumption —
mutually independent. If I insist they are non-Gaussian, then `x = A e` is *exactly* the linear
independent component analysis model: observed data is an invertible linear mixture of independent
non-Gaussian sources. Not a metaphor — the same equations. The disturbances `e_i` are the independent
components, `A` is the mixing matrix, and `W = A^{-1} = I − B` is the separating matrix. And now the old
Gaussian obstruction has a precise opposite: the ICA mixing matrix is identifiable from the data when
the sources are independent and at most one is Gaussian — up to permutation, scaling, and sign of its
columns, with *no rotational ambiguity*. In this model every disturbance is non-Gaussian, so the
condition holds. Contrast Gaussian sources, where any orthogonal rotation `A R` produces the same
distribution (rotating independent Gaussians gives independent Gaussians of the same covariance) — that
rotational freedom is precisely what left the previous rung unable to orient. So the non-Gaussianity I
am now leaning on is the exact thing that collapses the ambiguity and makes `A`, hence `W = I − B`,
hence `B`, recoverable. The two-variable result generalizes because the engine underneath is ICA
identifiability. I should estimate `W` by ICA and read `B` off it.

But "up to permutation, scaling, and sign" is doing a great deal of work, and those three indeterminacies
are exactly what stand between me and `B`. ICA hands me some `W_ica` that, in the population limit,
equals `P D W` — the true separating matrix with rows permuted by an unknown permutation `P` and each
row scaled and sign-flipped by an unknown diagonal `D`. In a typical ICA application nobody cares: the
order, scale, and sign of recovered audio sources are meaningless. I cannot ignore them. The rows of `W`
correspond to the disturbances `e_i`, the columns to the observed `x_i`; a random row permutation loses
the correspondence between which recovered component is the noise of which observed variable, and the
scaling leaves the rows in the wrong units. I must *undo* the permutation and *fix* the scaling, and
nothing in ICA does that for me. ICA gives the raw material; the causal content is in resolving its
indeterminacies. Two sub-problems: find the right row permutation, and find the right row scaling.

Take scaling first — it is easier and it tells me what "correct" means. In the SEM convention each
variable's own coefficient is one: the equation `x_i = (stuff) + e_i` has coefficient `+1` on `e_i`, so
`W = I − B` has an all-ones diagonal (`W_ii = 1 − b_ii = 1`). That is a fixed, known anchor. If the rows
were in the right order, every diagonal entry of the true `W` would be exactly one, and to remove ICA's
per-row scaling I divide each row by its own diagonal entry — which pins the scale *and* the sign at
once. So scaling is solved the instant I know the row order, by normalizing the diagonal to ones. All
the weight falls onto the permutation.

The permutation is the subtle one. What distinguishes the correct row order from the `d! − 1` wrong ones?
The correct `W = I − B`: since `B` is permutable to strict lower triangular, `W` is permutable (by the
*same* row-and-column permutation) to lower triangular with a *nonzero* diagonal — the ones from `I`. So
the correctly ordered `W` is lower triangular with no zeros on its diagonal. ICA scrambled the rows. The
property I need is sharper than a heuristic: among all ways to permute the rows back, exactly one yields
a fully nonzero diagonal, and it is correct. Write the aligned `W = P_d M P_d^T` with `M` lower
triangular, nonzero diagonal, `P_d` the true causal order. ICA returns `W_ica = P_ica W = P_ica P_d M
P_d^T`; set `P_1 = P_ica P_d` (row permutation on `M`) and `P_2 = P_d` (column permutation), so
`W_ica = P_1 M P_2^T`. Let `K` be the all-ones lower-triangular matrix — the densest support `M` could
have. The number of potentially nonzero diagonal entries of `P_1 K P_2^T` is `tr(P_1 K P_2^T) =
tr(K P_2^T P_1)` by cyclicity, so only the combined `Q = P_2^T P_1` matters. If `Q ≠ I`, some column `i`
must move to a position `j < i` (if none moved left, all moved right, and the finite set of positions
could not be filled); the diagonal entry in new column `j` comes from `K[j,i]` with `j < i`, strictly
above the lower-triangular support, hence zero — and since `K` is the densest support, that diagonal slot
is zero for any lower-triangular `M`. Conversely `Q = I` means `P_1 = P_2`, the same relabeling applied
to rows and columns, so the diagonal of `P_1 M P_1^T` is `M`'s nonzero diagonal reordered, all nonzero.
So a fully nonzero diagonal ⟺ `P_1 = P_2`, and the zero-free-diagonal row permutation is unique and
correct. The DAG structure is what makes the permutation decidable — and it is exactly the structure
that the previous rung had to *guess* at through magnitude thresholding, which is why it reversed and
missed hub edges. Here the acyclicity gives me a clean criterion: find the row order making the diagonal
nonzero.

Finite data degrades that clean criterion. ICA on a sample never returns exact zeros — every entry of
`W_ica` is some small nonzero number, so *every* row permutation gives a technically nonzero diagonal. I
need a soft version: prefer the permutation whose diagonal entries are as *far from zero* as possible,
since the correct alignment puts the structurally large unit-ish entries on the diagonal and the
structurally small (true-zero) entries off it. A natural cost penalizes small diagonal magnitudes: take
`sum_i 1/|W_ii|` after permutation and minimize it; a near-zero diagonal entry blows that term up. And
this is not ad hoc — model each disturbance with a generalized-Gaussian density `log p(e) = −|e|^α/β + Z`;
since a candidate row is later divided by its diagonal `W_ii`, the value entering the density scales as
`e/W_ii`, and the log-likelihood becomes, up to constants, `−sum_i (1/(β|W_ii|^α)) sum_t |e_it|^α`. With
identical component densities and `α = 1`, maximizing over the row correspondence is exactly
`min_perm sum_i 1/|W_ii|`. So the heavy penalty on small diagonals is the maximum-likelihood objective
for the correspondence. Naively searching all `d!` permutations is hopeless beyond a handful of nodes,
but the objective has the shape of a *linear assignment problem*: if row `r` is placed at diagonal
column `c`, the contribution is `C_{r,c} = 1/|W_ica[r,c]|`, and minimizing the summed assignment cost is
solved in `O(d^3)` by the Hungarian algorithm. So I build the cost matrix `C_{r,c} = 1/|W_ica[r,c]|`,
run `linear_sum_assignment` to get the row-to-diagonal matching, and scatter each row into its matched
position so the matched entries land on the diagonal. This is the step that scales to the 100-node SF100
graph where the previous rung's hub orientation broke down — `O(d^3)` assignment is cheap at `d = 100`.

With the rows correctly ordered, scaling is the easy step set up earlier: take the diagonal `D` of the
permuted `W`, divide each row by its diagonal entry so the diagonal becomes all ones — now `W_estimate`
is an estimate of `I − B` in the right units with scaling and sign resolved — and set `B = I − W`. But
for a clean directed graph I still want an explicit causal order under which `B` is strictly lower
triangular, and on finite data `B` has no exact zeros, so no permutation makes the upper triangle
exactly zero. If the estimates *were* exact, finding the order is easy: in `B` the row for variable `i`
holds its incoming coefficients, so an all-zero row is a variable with no remaining parents — a source.
A strict-lower-triangular structure always has at least one such row, so peel it off (record it, delete
its row and column) and repeat; peeling sources one at a time builds a valid topological order, and if
at some point no all-zero row exists, the remainder is cyclic. For finite data I lean again on "true
zeros are small." A strict-lower-triangular `d × d` matrix has `d(d+1)/2` structural zeros — the
diagonal (`d`) plus the entire upper triangle (`d(d−1)/2`). So sort all `d^2` entries of `B` by absolute
value, set the `d(d+1)/2` smallest to exact zero, then walk the remaining entries from smallest upward,
zeroing one more at a time and running the peel test, stopping at the first valid order. If in the true
matrix all structural zeros are smaller than all true nonzeros, the first valid peel is correct; in
general it is a cheap, magnitude-informed greedy approximation. That gives the causal order.

But `B = I − W` is fully dense — every off-diagonal entry is some nonzero number, including all that
should be structural zeros. The order only made the upper triangle "as zero as possible"; it pruned
nothing. The honest, statistically grounded way to get the final edges is to *re-estimate* them by
regression along the discovered order with sparsity, and the harness already provides exactly this. The
edit hands the order to `causal-learn`'s `_BaseLiNGAM._estimate_adjacency_matrix`, which walks the
variables in causal order and, for each target, regresses it on its predecessors by **adaptive lasso**:
fit ordinary least squares for pilot coefficients, form per-predictor weights `|coef|^γ` (with `γ = 1`),
rescale the predictors by these weights, fit a lasso with the penalty chosen by BIC, and multiply the
coefficients back. The weighting gives consistent selection — strong pilot coefficients get a light
penalty and survive, weak ones a heavy penalty and are zeroed — so spurious weak edges are pruned while
true edges keep good estimates. The first variable in the order has no predecessors and stays exogenous
with an all-zero row. The result fills the signed coefficients into `B[target, predecessors]`, returned
with `B[i,j] != 0` meaning `j -> i`. Note this is the same sparse re-estimation the continuous-
optimization rung approximated with a single hard threshold `ω = 0.3`; doing it as a BIC-selected lasso
per node is strictly more careful, which is part of why I expect the missing-edge problem to shrink.

One honest caveat I keep in front of me: the ICA step is a non-convex optimization. Maximizing
non-Gaussianity (the negentropy contrast, log-cosh nonlinearity, fixed-point iteration) has multiple
local optima, and a run from a poor initialization can land in the wrong one, giving an unreliable `W`
and hence a wrong graph. There is no convexity guarantee; stability depends on initialization and step
size. I fix the random seed (from `SEED`) for reproducibility and give FastICA a generous 1000-iteration
budget, but I am clear-eyed that this is the soft spot — the part that rests on an optimization that can
fail, as opposed to the permutation/scaling logic, which is exact given a good `W`. (The full scaffold
fill — FastICA, the assignment, the peel-off order search, and the `_BaseLiNGAM` adaptive-lasso
re-estimation — is in the answer.)

Now the falsifiable expectations against the previous rung's numbers. The whole point is that ICA-LiNGAM
*uses* non-Gaussianity, so on the scenarios with strong higher-order signal it should clear the
continuous program decisively, and on the scenario where it does not it should not improve much. ER30
has Laplace noise (heavy-tailed, strongly super-Gaussian — ideal ICA fuel) and only 30 nodes: I expect
near-perfect recovery, F1 well above 0.95 and SHD in the low single digits, fixing the seed-456
instability (F1 0.835, SHD 35) that dragged the previous mean to 0.919. ER50 has exponential noise (also
strongly non-Gaussian, skewed) with 2000 samples: I expect the SHD-58, F1-0.872 result to improve
sharply, F1 well above 0.95 and SHD into the low single digits, because the source signal is exactly
what ICA separates and the sample size is generous. SF100 is the test of whether this engine helps where
the previous one most failed (F1 0.716, recall 0.597). Two forces fight here: the assignment + acyclic-
order machinery should orient the hub edges the least-squares program kept missing — so I expect recall
to rise — but the noise on SF100 is *uniform*, which is sub-Gaussian and light-tailed, the weakest case
for an ICA contrast tuned to super-Gaussian sources, and 100 nodes with only 1000 samples stresses the
single global FastICA fit (one non-convex optimization over a 100-dimensional unmixing). So I expect
SF100 to improve over the previous rung's F1 but to remain the worst scenario by a wide margin, with the
global-ICA non-convexity showing up as high seed-to-seed variance — the failure that, if it appears, will
point the next rung away from a single global ICA optimization and toward estimating the causal order
*directly*, one variable at a time, without ever solving a non-convex problem in unmixing space.
