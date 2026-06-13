The ICA rung confirmed the thesis on the easy scenarios and exposed exactly where the ICA *machinery*
breaks on the hard one, and the break is legible in the numbers. ER30 went from the continuous program's
unstable F1 0.919 (SHD 16, seed 456 collapsing) to a near-perfect F1 0.989, SHD 2.3, recall a clean
1.0 on every seed — the heavy-tailed Laplace noise is ideal ICA fuel and the instability vanished. ER50
went from F1 0.872, SHD 58 to F1 0.994, SHD 3.0, again recall 1.0 across all three seeds — the
exponential noise plus 2000 samples is exactly what ICA separates, and doubling the nodes no longer
hurt. So on both Erdos-Renyi scenarios the switch to a non-Gaussian engine did precisely what I
predicted: it fixed the missing-edge problem and stabilized the dense small graph. But SF100 tells the
real story, and it is *not* the story I half-expected. F1 rose only to 0.804 (from 0.716), SHD stayed
catastrophic at 120, and — crucially — the precision/recall asymmetry *flipped*. The continuous program
was high-precision (0.897) and low-recall (0.597): it missed edges. ICA-LiNGAM is now high-recall
(0.944) and low-*precision* (0.702): it finds the true edges but spews false ones. And the variance is
ugly — seed 42 SHD 107, seed 456 SHD 95, but seed 123 SHD **159** with F1 0.756. That single bad seed,
and the precision collapse, are the fingerprint of one thing: the global FastICA fit on a 100-dimensional
unmixing under *uniform* (sub-Gaussian, light-tailed) noise. ICA's contrast is tuned to super-Gaussian
sources; on sub-Gaussian sources it is at its weakest, and a single non-convex optimization over a
100-dim unmixing matrix is exactly where a poor local optimum scrambles the separating matrix, corrupts
the assignment-based permutation, and produces a wrong causal order — after which the adaptive lasso
dutifully fits dense edges into a wrong ordering, tanking precision. The diagnosis is sharp: the
Erdos-Renyi wins came from non-Gaussianity, but the SF100 failure came from routing it through a single
*global, non-convex* ICA optimization. So the next move is forced: keep the non-Gaussianity, drop the
global ICA optimization and its local-optimum fragility, and get the causal order *directly*, one
variable at a time, with no iterative search in unmixing space anywhere.

Let me be precise about what I want to avoid, because the cracks in the ICA route are what dictate the
replacement. The ICA step has three pathologies I now have measured evidence for. First, it maximizes a
non-Gaussianity contrast (negentropy / mutual-information minimization), and that objective is
*non-convex* — iterative search can settle into a local optimum, with no guarantee of reaching the right
unmixing in any finite number of steps; the seed-123 SHD-159 blowup on SF100 is that local optimum made
visible. Second, it has knobs with no principled setting: an initial guess (a bad one gives an outright
wrong answer), a step size, a convergence criterion — none with a systematic recipe, all of which I am
currently papering over with a fixed seed and a 1000-iteration budget. Third, and subtler, both the
permutation cost `1/|W_ii|` and the lower-triangularization scoring are *not scale-invariant*: they
depend on the variables' variances, even though the causal *order* cannot — multiplying a variable by a
constant cannot change whether it is upstream. An estimator whose answer can flip under innocuous
rescaling is keying on the wrong thing. I want to keep the non-Gaussianity, since that is what makes the
problem identifiable at all and what won ER30 and ER50, but discard the ICA apparatus wholesale.

So can I get the causal order *directly*, without ever solving a non-convex optimization in parameter
space? Think about what the order structurally *is*. In a DAG with no latent confounders, acyclicity
forces at least one variable to have no parents — a source of the graph; call it exogenous. For an
exogenous `x_j`, the model says `x_j = e_j`: no `b_{jh} x_h` term, so the variable simply *equals* its
own independent non-Gaussian disturbance — an observed copy of an independent source, a very clean
object. If I could *find* the exogenous variable, I would know it sits first in the order. Then, if I
peel its effect off all the others and the remaining system is again a LiNGAM with one fewer variable, I
find *its* source, and so on, building the order one variable at a time — `d − 1` rounds of regression
and an exogeneity test each, no iterative parameter search anywhere. The entire problem reduces to one
question: how do I detect, from data, which variable is exogenous?

Here the non-Gaussianity must earn its keep on a *direct test* rather than on a global optimization.
Tentatively treat `x_j` as the source and regress every other `x_i` on it by least squares, forming the
residual `r_i^{(j)} = x_i − (cov(x_i,x_j)/var(x_j)) x_j`. What does independence between `x_j` and these
residuals tell me? Work both directions, because if the equivalence holds it is the whole algorithm.
First suppose `x_j` really is exogenous, `x_j = e_j`. From `x = Ae`, write `x_i = a_{ij} x_j + ē_i^{(j)}`
where `ē_i^{(j)} = sum_{h≠j} a_{ih} e_h` collects every source other than `e_j`. Since `x_j = e_j` and
the sources are mutually independent, `x_j` is independent of `ē_i^{(j)}`. The least-squares coefficient
of `x_i` on `x_j` is `cov(x_i,x_j)/var(x_j)`, and because `x_j = e_j` is independent of `ē_i^{(j)}`,
`cov(x_i,x_j) = a_{ij} var(x_j)`, so the coefficient is exactly `a_{ij}`, and the residual
`r_i^{(j)} = x_i − a_{ij} x_j = ē_i^{(j)}` is precisely the bundle of other sources, independent of
`x_j`. So an exogenous `x_j` is independent of *every* residual. One direction.

Now the converse, where the non-Gaussianity is indispensable, via Darmois–Skitovitch. Suppose `x_j` is
*not* exogenous, with nonempty parent set `P_j`, `x_j = sum_{h∈P_j} b_{jh} x_h + e_j`. First, is there a
parent `x_i` with `cov(x_i,x_j) ≠ 0`? Collect the parents into `x_{P_j}` with weights `b_{P_j}`; then
`E(x_{P_j} x_j) = E(x_{P_j} x_{P_j}^T) b_{P_j}`, since each parent is built from sources other than `e_j`
and so is uncorrelated with `e_j`. The parent covariance is positive definite (their independent
disturbances are not collinear) and `b_{P_j} ≠ 0`, so the product is not the zero vector — some parent
`x_i` has nonzero covariance. Take it and form `r_i^{(j)}`. Substituting `x_j` and expanding to sources,
both `r_i^{(j)}` and `x_j` are linear combinations of the independent `e`. Track the coefficient on
`e_j`: in `x_j` it is `1`; in `r_i^{(j)}` it is `−cov(x_i,x_j)/var(x_j)`, nonzero by the choice of `x_i`.
So two linear combinations of independent variables share the source `e_j` with nonzero weight in both,
and `e_j` is non-Gaussian. Darmois–Skitovitch: if two such combinations were independent, every shared
source with nonzero coefficient in both would have to be Gaussian — contrapositive, a shared
non-Gaussian source forces them *dependent*. So `r_i^{(j)}` and `x_j` are dependent. Therefore a
non-exogenous `x_j` fails the independence test against at least one residual. Combining both directions:
`x_j` is exogenous **iff** it is independent of all its least-squares residuals. The non-Gaussianity of
`e_j` is the *entire* reason the converse works — the same fuel as ICA, now spent on a direct test
instead of a non-convex optimization.

I must make the recursion legitimate: peeling a source leaves a LiNGAM, and the recovered order on
residuals is the order of the originals. Take `x_j` exogenous; permute so `B` is strictly lower
triangular and `x_j = x_1`, then `A` is lower triangular with unit diagonal and `a_{i1}` equals the
regression coefficient of `x_i` on `x_1`. Remove `x_1`'s effect from every `x_i` by least squares: the
first column of `A` becomes zero, and the submatrix deleting the first row and column is still lower
triangular with unit diagonal — so the residual vector `r^{(1)} = A^{(1)} e^{(1)}` is again a LiNGAM, one
dimension smaller, with independent non-Gaussian sources. And deleting the first row and column leaves
the relative order untouched, so the residuals' causal order equals the originals'. The algorithm is
forced: find the exogenous variable, append it, regress it out of the rest, recurse on the residuals;
after `d − 1` peeling rounds one variable remains and goes last. This is the structure I want — finite,
deterministic, no initialization, no step size, no global optimization.

Now the landmine. My exogeneity test is "`x_j` independent of every residual," but least squares
*guarantees* the residual is uncorrelated with the regressor — for *every* `j`, source or not. So
uncorrelatedness is useless here; I need a measure of genuine *independence* that sees the higher-order
dependence uncorrelatedness misses. Mutual information `I(y_1,y_2) = H(y_1) + H(y_2) − H(y_1,y_2)` is
zero exactly when independent. A kernel estimate would work but estimates a two-dimensional dependency
per pair, needs a bandwidth and a regularizer, costs `O(n d^3 M^2 + d^4 M^3)`, and is noisy on small
samples — and small samples is exactly the SF100 regime (1000 samples, 100 nodes) where the ICA route
just failed. The structure of my problem is friendlier than full 2-D MI, though. I am always comparing
the same pair of variables, regressed two ways: candidate-as-cause versus candidate-as-effect. Standardize
two variables `x` and `y` to zero mean and unit variance, and consider the two two-variable LiNGAMs:
`x → y` (`y = ρx + d`, `d` independent of `x`) and `y → x` (`x = ρy + e`, `e` independent of `y`), with
the same `ρ` (the correlation) in both. Which model do the data prefer? The principled comparator is the
likelihood ratio; its log, normalized by `T`, is `R = (1/T) Σ_t [G_x(x_t) + G_d((y_t − ρx_t)/√(1−ρ²)) −
G_y(y_t) − G_e((x_t − ρy_t)/√(1−ρ²))]`, with `G` a standardized log-pdf. `R > 0` favors `x → y`. In the
asymptotic limit the sample averages converge to negative differential entropies, so `R → −H(x) −
H(d̂/σ_d) + H(y) + H(ê/σ_e)` — the likelihood ratio is comparing the total non-Gaussianity of the
regressor-plus-residual pair in each direction. I never have to estimate a 2-D entropy.

Make that explicit, because it is what licenses replacing kernel MI with one-dimensional entropies. The
linear map `(x,y) → (x,d)` with `d = y − ax` has determinant 1, and differential entropy transforms as
`H(Tu) = H(u) + log|det T|`, so `H(x,d) = H(x,y)` and likewise `H(y,e) = H(x,y)`. Therefore
`I(x,d) − I(y,e) = [H(x) + H(d) − H(x,d)] − [H(y) + H(e) − H(y,e)] = H(x) + H(d) − H(y) − H(e)` — the
joint entropies cancel. Folding in standardized residuals (both residual variances are `1 − ρ²`) the
extra log terms cancel too, so `I(x,d) − I(y,e) = H(x) + H(d̂/σ_d) − H(y) − H(ê/σ_e) = −R`, i.e.
`R = I(y,e) − I(x,d)`. The criterion is the same comparison — choose the direction in which the regressor
is more independent of its residual — now needing only 1-D entropies, never a 2-D density. This is the
exogeneity test from the lemma, made cheap.

I need a good, cheap 1-D differential entropy of a standardized variable — exactly the negentropy-
approximation problem. Differential entropy of a unit-variance variable is maximized by the Gaussian, so
write entropy as the Gaussian entropy minus a non-Gaussianity penalty, approximated by expectations of a
couple of well-chosen contrast functions: `Ĥ(u) = H(ν) − k1 [E{log cosh u} − γ]² − k2 [E{u
exp(−u²/2)}]²`, with `H(ν) = (1 + log 2π)/2`, `k1 ≈ 79.047`, `k2 ≈ 7.4129`, `γ ≈ 0.37457`. Each bracket
is a non-Gaussianity measure: the first uses `log cosh` (a robust even contrast for super-Gaussian
heavy-tailed sources) and `γ` is `E{log cosh}` under a standard Gaussian, so it vanishes for a Gaussian;
the second uses the odd `u exp(−u²/2)` to capture asymmetry/skew. Both are subtracted because a more
non-Gaussian variable has lower entropy than the Gaussian of the same variance, and the constants are the
fixed maximum-entropy weights making the approximation second-order accurate around the Gaussian — given
numbers, not things to tune. The approximation is valid only for *standardized* `u`, so I standardize
every variable and divide every residual by its own standard deviation before feeding it in. Why fix the
contrast functions rather than estimate each variable's log-pdf? Because the recovered direction is
insensitive to the exact log-pdf as long as the shape is roughly right, and per-variable density fitting
is many parameters and unreliable at small sample sizes — exactly the regime that broke the global ICA on
SF100. A fixed log-cosh-plus-skew contrast covers the super-Gaussian and asymmetric cases with zero
tuning. And — importantly for SF100 — this criterion is scale-invariant by construction, because every
variable and residual is standardized before scoring, so rescaling an input cannot flip the order: the
exact defect that made the ICA permutation fragile is gone.

Now assemble the per-pair difference the algorithm computes. For a pair `(i,j)`, regress each on the
other, standardize the residuals, and form `diff_MI(i,j) = [H(x_j_std) + H(r_i^{(j)}/σ)] − [H(x_i_std) +
H(r_j^{(i)}/σ)]`, the first bracket treating `j` as cause, the second treating `i` as cause; the common
joint entropy has cancelled, so the difference is `I(x_j,r_i^{(j)}) − I(x_i,r_j^{(i)})`. Read the sign:
if `i` is truly the cause, regressing `j` on `i` leaves an independent residual so the second term is
near zero, while regressing `i` on `j` the wrong way leaves a dependent residual so the first term is
large — hence `diff_MI(i,j) > 0` exactly when `i`-as-cause is more plausible. To turn pairwise
comparisons into one exogeneity score for candidate `i`, count only *evidence against* `i` being the
source: if `i` is exogenous, every pair should favor `i`-as-cause (`diff_MI(i,j) ≥ 0`), so any negative
term is a vote that `j`, not `i`, is upstream. Accumulate `M(i) = Σ_{j≠i} min(0, diff_MI(i,j))²` — the
squared magnitude of just the unfavorable-sign votes — and select the variable with the *least*
accumulated counter-evidence (store `−M(i)` and argmax). A genuinely exogenous variable collects ~0 and
wins. This is the pairwise specialization of the residual-independence statistic, computed from cheap 1-D
entropies, no kernel, no bandwidth, no 2-D density.

The cost and convergence story is the whole reason I left ICA behind. There is no iterative search in
parameter space anywhere: when `q` variables remain, a round scores all ordered pairs among them, then
after the source is chosen it does the `q − 1` residual updates that peel that source out; after `d − 1`
nontrivial rounds the last variable remains. With a true independence score the exogenous variable is
identified each round, the residual system stays in the model class, and the order is recovered; the
fixed 1-D entropy approximation keeps that finite-step, no-init, no-step-size structure while replacing
the expensive kernel score with a cheaper likelihood-ratio proxy. Once the order is in hand, the
connection strengths are a triangular regression of each variable on its predecessors — and the edit
hands the order to the same `causal-learn` `_BaseLiNGAM._estimate_adjacency_matrix` the previous rung
used: adaptive lasso per node (OLS pilot, `|coef|^γ` weights with `γ = 1`, BIC-selected lasso, unweight),
consistent selection driving absent edges to exactly zero, returned with `B[i,j] != 0` meaning `j -> i`.
So the *only* thing that changed from the previous rung is the engine that produces the causal order:
the global non-convex ICA fit is replaced by a deterministic, scale-invariant, pairwise-entropy peeling.
(The full scaffold fill — `_residual`, `_entropy`, `_diff_mutual_info`, the `M(i)` source search, the
peeling loop, and the `_BaseLiNGAM` adaptive-lasso re-estimation — is in the answer.)

Now the falsifiable expectations against the previous rung's numbers. On ER30 and ER50 the ICA route was
already near-perfect (F1 0.989/0.994, SHD 2.3/3.0, recall 1.0), so the direct method should match —
*not* dramatically beat — those, since both use the same non-Gaussian identifiability and the same
adaptive-lasso edge step; I expect F1 around 0.99 and SHD in the low single digits on both, perhaps a
hair worse on one and a hair better on another, essentially a tie on the easy scenarios where ICA did
not struggle. The decisive test is SF100, where the ICA machinery failed: F1 0.804, SHD 120, precision
0.702, and the seed-123 SHD-159 blowup. If my diagnosis is right — that the SF100 failure was the global
non-convex ICA optimization under sub-Gaussian uniform noise, not the non-Gaussian principle itself —
then replacing that optimization with a deterministic pairwise peeling should *collapse* the SHD and
restore precision. I expect SF100 SHD to drop from ~120 into the single digits, F1 to jump from 0.804 to
above 0.98, precision to recover from 0.702 to near 0.97, and the seed-to-seed variance to shrink hard
(no more seed-123-style 159 outlier), because there is no longer any non-convex optimization to land in a
bad basin. If instead SF100 stays poor, my diagnosis is wrong and the difficulty is intrinsic to the
sub-Gaussian noise — but the scale-invariance and the absence of a global optimization make me expect the
former. That gap on SF100 — three rungs failing it for three different reasons (missed hub edges, then
ICA-scrambled false edges) finally closing — is the result this rung must deliver to earn the top of the
ladder.
