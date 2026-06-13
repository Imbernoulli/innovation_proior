The function is the whole task, and before I reach for anything nonlinear I want the floor — the
simplest principled fill of `run_causal_discovery` that already returns *directed* edges, not a
correlation blur. The floor matters because it tells me how much of the work is the *nonlinearity*
versus the *direction*: if a method that ignores nonlinearity entirely already does well, the
nonlinear machinery the later rungs add is wasted; if it falls over, I have measured exactly the gap
the rest of the ladder has to close. So I deliberately start with a method that breaks
forward/backward symmetry by the *other* identifiability route — non-Gaussian noise, not nonlinear
mechanisms — and is linear throughout. That is DirectLiNGAM, and it is the floor by construction
because the data here is generated nonlinearly while this method assumes linearity.

Let me write down what actually blocks me, because the floor still has to be principled. I have `X`
of shape `(n_samples, n_variables)`, no interventions, no time stamps, nothing telling me which
variable is upstream. The one object I can compute most cheaply from observational data is the
covariance, so I should ask what it can give me — and the answer is the reason the whole field is
hard. Take two correlated variables. Model one: `x_2 = b x_1 + e_2`, so `x_1` causes `x_2`. Model
two: `x_1 = b' x_2 + e_1`, the reverse. With Gaussian noise I can pick `b, b', var(e)` in each model
to reproduce the *exact* same 2×2 covariance. The covariance literally cannot tell these apart, and
this is not a defect of one estimator: PC tests conditional independences, GES scores DAGs, but under
Gaussianity both are functions of the covariance too, so they bottom out at the Markov equivalence
class — and on two variables that class contains both directions. Covariance is direction-blind. If I
only ever look at second-order statistics I am stuck at the equivalence class, full stop. Whatever
breaks the tie must be something the covariance does not see.

What does the covariance throw away? Everything above second order. A Gaussian is fully described by
its mean and covariance, so for Gaussian data there *is* nothing above second order — exactly why the
Gaussian-linear case is hopeless. But suppose the disturbances `e_i` are *non-Gaussian*. Now the data
carries higher-order structure, and the asymmetry between cause and effect can leave a fingerprint
there. Solve the linear SEM for `x`: `x = (I - B)^{-1} e = A e`, with `A = (I-B)^{-1}` permutable to
lower-triangular with unit diagonal because `B` is permutable to strictly lower-triangular. That is a
linear mixture of mutually independent non-Gaussian sources — the ICA model exactly — and Comon's
theorem says the mixing is recoverable up to permutation, scaling, and sign when at most one source is
Gaussian. So non-Gaussianity turns the unidentifiable covariance problem into an identifiable one.
This is the *other* escape from the equivalence class, parallel to the nonlinear one the rest of my
ladder will use, and it is why a linear method can still produce arrows at all.

The first LiNGAM estimator ran ICA, undid its permutation/scaling indeterminacies, and read off `B`.
But ICA's contrast is non-convex — local minima, an init guess that can give an outright wrong answer,
a step size and a stopping rule with no principled setting — and worse, its permutation scoring is
scale-dependent, so normalizing the variables (a routine preprocessing step) can *reverse* the
recovered order at finite sample size. An estimator whose answer flips when I rescale an input is
keying on the wrong thing. I want to keep the non-Gaussianity, which is what makes the problem
identifiable, but drop the ICA machinery. DirectLiNGAM is precisely that: get the causal *order*
directly, with no iterative parameter search.

The structural fact that makes it possible: in a DAG with no latent confounders, acyclicity forces at
least one variable to have no parents — a source, an exogenous variable. For an exogenous `x_j` the
model says `x_j = e_j`: it simply *equals* its own independent non-Gaussian disturbance. If I could
find that variable, I would know it sits first in the order; peel its effect off all the others, and
the remaining system is again a LiNGAM with one fewer variable, so I recurse. The whole problem
reduces to one question: how do I detect, from data, which variable is exogenous? Suppose I tentatively
treat `x_j` as the source and regress every other `x_i` on it by least squares, forming the residual
`r_i^{(j)} = x_i - (cov(x_i,x_j)/var(x_j)) x_j`. The clean characterization — provable in both
directions — is: `x_j` is exogenous if and only if `x_j` is independent of *every* residual `r_i^{(j)}`.
The forward direction is direct from `x = Ae`: when `x_j = e_j`, the least-squares coefficient is
exactly `a_{ij}`, the residual is the bundle of *other* sources, and that bundle is independent of
`e_j`. The converse is where the non-Gaussianity earns its keep: if `x_j` has a parent, then by
Darmois–Skitovitch two linear combinations sharing a non-Gaussian source with nonzero weight in both
cannot be independent, so some residual stays dependent on `x_j`. With *Gaussian* `e_j` the converse
fails — independent combinations may share a Gaussian source — and the test gives false positives.
That is the precise sense in which this method needs non-Gaussian noise, and the precise sense in which
it will struggle on the ER20-Gauss scenario.

There is one landmine I must step around in code, and the task's edit handles it correctly. Least-
squares regression *guarantees* the residual is uncorrelated with the regressor — by construction, for
*every* candidate, source or not — so uncorrelatedness is useless as the exogeneity test. I need a
measure of genuine independence that sees the higher-order dependence uncorrelatedness misses. The
canonical such measure is mutual information, and DirectLiNGAM's `pwling` variant computes it cheaply
from *one-dimensional* differential entropies via a likelihood-ratio between the two pairwise
directions: the joint-entropy terms cancel under a unit-determinant change of variables, leaving a
difference of regressor-plus-residual entropies. Each entropy is the maximum-entropy approximation of a
standardized variable — the Gaussian entropy minus two non-Gaussianity penalties, a log-cosh
(heavy-tail) term and an odd `u·exp(-u²/2)` (skew) term, with fixed constants `k1 ≈ 79.047`,
`k2 ≈ 7.4129`, `gamma ≈ 0.37457`. This is exactly the code in the editable function: `_entropy`,
`_diff_mutual_info`, and `_search_causal_order` aggregate, for each candidate `i`, the squared
*unfavorable-sign* pairwise differences `M(i) = sum_{j} min(0, diff_MI(i,j))^2`, and pick the variable
with the least accumulated counter-evidence — the one most consistent with being the source. Note what
the task's fill leans on and what it omits: it standardizes every variable and residual before scoring
(so the order is scale-invariant — the exact defect that sank ICA-LiNGAM), and it reuses the library's
`_BaseLiNGAM._estimate_adjacency_matrix` to turn the recovered order `K` into the connection strengths.
This last detail is the one place the task's edit diverges from the most polished version of the method:
the standalone derivation I keep as reference prunes edges with a sparse adaptive lasso (BIC-selected),
whereas the harness fill estimates the full triangular adjacency from the order with no extra sparsity
step — so I should expect a denser graph and lower precision than the lasso-pruned variant would give.

So the floor fill is settled, and it is a faithful transcription of causal-learn's DirectLiNGAM core
into `run_causal_discovery`: build `U = arange(n)` (variables not yet placed), repeatedly call
`_search_causal_order` on the working data to find the exogenous index `m`, append it to the order `K`,
regress `m` out of every other working variable (the residual peeling that keeps the system a LiNGAM
one dimension smaller), and after the order is complete hand `K` to a local `_BaseLiNGAM` to estimate
the adjacency. The output already obeys the harness convention `B[i,j] != 0` means `j -> i`. The full
scaffold module is in the answer.

Now the part that actually matters for the ladder: reasoning about what this floor must do on *this*
data, because that is the entire point of running it first. The data is generated by *nonlinear*
mechanisms. DirectLiNGAM assumes *linear* ones. The mismatch bites in two compounding ways. First, the
residual `r_i^{(j)} = x_i - β x_j` is a *linear* residual; if the true mechanism is `x_i = f(x_j) +
e_i` with `f` curved, the linear fit leaves a structured, `x_j`-dependent remainder *even in the
correct direction*, so the "independent residual at the source" signal is muddied — the test's clean
forward direction no longer holds exactly. Second, the entropy criterion reads the non-Gaussianity of
those residuals, but a nonlinear function of a non-Gaussian input produces residuals whose higher-order
structure no longer matches the additive-source picture the lemma assumed. Both effects push the
recovered order toward noise. So I expect DirectLiNGAM to land arrows that are *frequently reversed or
spurious* — its F1 should be low across the board, and because it estimates a fairly full triangular
graph from a possibly-wrong order, its SHD should be large and its precision poor.

The scenarios should split predictably. On SF20-GP the noise is exponential — strongly non-Gaussian,
which helps the source detection — but the GP mechanisms are smooth and very nonlinear, which hurts the
linear residual; I expect a middling-to-low F1 and a large SHD on the 20-node graph. On ER20-Gauss the
noise is *Gaussian*, which is the worst case for this method on two counts at once: the converse of the
exogeneity lemma fails for Gaussian sources (false positives in source detection), and the mechanisms
are nonlinear; I expect the lowest or near-lowest F1 here, with a very large SHD on a 20-node ER graph
that is denser than the scale-free one. On ER12-LowSample the graph is smaller (12 nodes) but `n` is
only 150, so every entropy and covariance estimate is noisy; I expect low F1 driven mostly by variance
across seeds, though the smaller graph caps the absolute SHD.

The falsifiable expectation I will hold the next rung to: DirectLiNGAM is a *linear* floor, so its F1
across all three scenarios should sit well below what any genuinely nonlinear method can reach, and its
failure mode is *wrong/dense direction* (low precision, high SHD), not missing edges. If the next rung
— the first one that actually models nonlinearity — does not clearly beat this on F1 *and* shrink SHD,
then the nonlinearity is not where the leverage is and I have misdiagnosed the task. I am starting here
precisely so that gap is measured, not assumed.
