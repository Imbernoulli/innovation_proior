Let me start from what actually blocks me. I have a sample of `x = (x_1, ..., x_p)`, no interventions, no time stamps, nothing that tells me which variable is upstream. I believe the world is a linear acyclic SEM, `x_i = sum_{k(j)<k(i)} b_{ij} x_j + e_i`, in matrix form `x = Bx + e` with `B` permutable to strictly lower triangular under the unknown order `k`, and the `e_i` are mutually independent. I want `B` — the structure and the strengths — from `x` alone. The thing I can compute most easily from observational data is the covariance matrix, so let me ask what it can give me. Take the simplest case, two correlated variables. Model one: `x_2 = b x_1 + e_2`, so `x_1` causes `x_2`. Model two: `x_1 = b' x_2 + e_1`, the reverse. With Gaussian noise I can choose `b, b', var(e)` in each model to produce the exact same `2x2` covariance matrix. The covariance literally cannot tell these apart. And it is not a weakness of one estimator: PC tests conditional independences, GES scores DAGs, but under Gaussianity both of those are functions of the covariance too, so they bottom out at the Markov equivalence class — a set of DAGs that share all the same (conditional) correlations — and on two variables that class contains both directions. Covariance is direction-blind. If I only ever look at second-order statistics I am stuck, full stop. So whatever breaks the tie has to be something the covariance does not see.

What does the covariance throw away? Everything above second order. A Gaussian is fully described by its mean and covariance, so for Gaussian data there *is* nothing above second order — that is exactly why the Gaussian case is hopeless. But suppose the disturbances `e_i` are *non-Gaussian*. Now the data carries higher-order structure, and maybe the asymmetry between cause and effect leaves a fingerprint there. Let me chase this. Solve the model for `x`: `x = (I - B)^{-1} e = A e` with `A = (I-B)^{-1}`. Because `B` is permutable to strictly lower triangular, `A` is permutable to lower triangular with ones on the diagonal. Now stare at `x = A e`: a linear mixture of mutually independent, non-Gaussian sources `e`. That is the independent component analysis model exactly. And ICA has a theorem I can lean on — Comon's identifiability result: if you observe a linear invertible mixture of independent sources and at most one of them is Gaussian, the mixing matrix `A` is recoverable up to permutation, scaling, and sign of its columns. So non-Gaussianity has quietly turned my unidentifiable covariance problem into an *identifiable* ICA problem. The full structure — every edge direction — is in principle recoverable from observational data. That is the whole reason this is solvable, and it is worth pinning down before I write a single line of algorithm.

So the obvious first plan is: just run ICA, get `A`, invert to `W = A^{-1} = I - B`, read off `B`. Let me try it and see where it cracks. ICA gives me `W` only up to a permutation and a diagonal rescaling: what it returns is `W_ICA = P D W` for some unknown permutation `P` and diagonal `D`. I need to undo both. The permutation first. The true `W = I - B` has ones on its diagonal — every diagonal entry is exactly 1, nonzero. The wrong row-permutations of `W_ICA` will scatter those nonzero entries off the diagonal and leave (in the infinite-data limit) zeros on the diagonal instead. So the correct row permutation is the unique one giving an all-nonzero diagonal. To find it with noisy estimates, penalize small diagonal magnitudes: minimize `sum_i 1/|W̃_{ii}|` over permutations, which is a linear assignment problem solvable in `O(p^3)`. Then fix the scaling: divide each row by its own diagonal entry so the diagonal becomes all ones, and set `B̂ = I - W̃'`. Finally I still need a causal *order* — permute `B̂` by a simultaneous row-and-column permutation toward strict lower triangularity, scoring how close I am by `sum_{i<=j} B̃_{ij}^2` and, for many variables, iteratively zeroing the smallest entries and testing whether what remains can be permuted to strict lower triangular.

This works, and it was the first LiNGAM estimator. But let me be honest about where it hurts, because the cracks are what tell me what to build next. The first crack is the ICA step itself. ICA maximizes non-Gaussianity (equivalently minimizes the components' mutual information), and that objective is *non-convex*. Iterative search on a non-convex contrast can settle into a local optimum; there is no guarantee it reaches the right unmixing in any finite number of steps. And it has knobs with no principled setting: an initial guess (a bad one can give an outright wrong answer), a step size for gradient variants, a convergence criterion for all of them. None of these has a systematic recipe. So the very first stage of my pipeline is an unbounded, parameter-laden, possibly-wrong optimization, and everything downstream inherits that. The second crack is subtler and I almost missed it: both permutation steps are *not scale-invariant*. `1/|W̃_{ii}|` and `sum_{i<=j} B̃_{ij}^2` both depend on the variances of the variables. But the causal *order* has nothing to do with scale — multiplying `x_3` by 1000 cannot change whether `x_3` is upstream of `x_5`. Yet because the scoring is scale-dependent, if I normalize the variables to unit variance (a completely innocuous, standard preprocessing step) I can get a different — even reversed — ordering at finite sample size. An estimator whose answer flips when I rescale the inputs is telling me it is keying on the wrong thing.

I want to keep the non-Gaussianity, since that is what makes the problem identifiable at all, but drop the ICA machinery and its pathologies. Can I get the causal order *directly*, without ever solving a non-convex optimization in parameter space? Let me think about what the order actually is, structurally. In a DAG with no latent confounders, acyclicity forces at least one variable to have no parents — a source of the graph. Call such a variable exogenous. For an exogenous `x_j`, the model says `x_j = e_j`: there is no `b_{jh} x_h` term, so the variable simply *equals* its own independent non-Gaussian disturbance. That is a very special, very clean object — an observed copy of an independent source. If I could *find* the exogenous variable, I would know it sits first in the order. Then, if I peel its effect off all the other variables and the remaining system is again a LiNGAM with one fewer variable, I find *its* source, and so on, building the order one variable at a time, `p-1` regressions and an exogeneity test each round, no iterative parameter search anywhere. So the entire problem reduces to one question: how do I detect, from data, which variable is exogenous?

Here is where the non-Gaussianity has to earn its keep. Suppose I tentatively treat `x_j` as the source and regress every other `x_i` on it by least squares, forming the residual `r_i^{(j)} = x_i - (cov(x_i,x_j)/var(x_j)) x_j`. What does independence between `x_j` and these residuals tell me? Let me work both directions of the claim carefully, because if it holds it is the whole algorithm.

First suppose `x_j` really is exogenous, `x_j = e_j`. From `x = Ae`, write `x_i = a_{ij} x_j + ē_i^{(j)}` where `ē_i^{(j)} = sum_{h != j} a_{ih} e_h` collects every source other than `e_j`. Since `x_j = e_j` and the sources are mutually independent, `x_j` is independent of `ē_i^{(j)}`. Now what is the least-squares coefficient of `x_i` on `x_j`? It is `cov(x_i, x_j)/var(x_j)`, and because `x_j = e_j` is independent of `ē_i^{(j)}`, `cov(x_i, x_j) = a_{ij} var(x_j)`, so the coefficient is exactly `a_{ij}`. Then the residual `r_i^{(j)} = x_i - a_{ij} x_j = ē_i^{(j)}` — it is precisely the bundle of other sources, which is independent of `x_j`. So when `x_j` is exogenous, `x_j` is independent of *every* residual `r_i^{(j)}`. Good, one direction.

Now the converse — and this is where I need the non-Gaussianity, via Darmois–Skitovitch. Suppose `x_j` is *not* exogenous, so it has at least one parent; let `P_j` be its nonempty parent set, `x_j = sum_{h in P_j} b_{jh} x_h + e_j`. I want to show some residual `r_i^{(j)}` is *dependent* on `x_j`. First, is there even a parent `x_i` with `cov(x_i, x_j) != 0`? Collect the parents into a vector `x_{P_j}` with weights `b_{P_j}`. Then `E(x_{P_j} x_j) = E(x_{P_j}(b_{P_j}^T x_{P_j} + e_j)) = E(x_{P_j} x_{P_j}^T) b_{P_j}`, since each parent is built from sources other than `e_j` and so is uncorrelated with `e_j`. The parent covariance matrix `E(x_{P_j} x_{P_j}^T)` is positive definite — the parents' own disturbances are independent with positive variance, so they cannot be collinear — and `b_{P_j} != 0`, so the product cannot be the zero vector. Hence at least one parent `x_i` has `cov(x_i, x_j) != 0`. Take that `x_i` and form `r_i^{(j)}`. Substituting `x_j = sum_h b_{jh} x_h + e_j` and expanding everything back to sources, both `r_i^{(j)}` and `x_j` come out as linear combinations of the independent sources `e`. Crucially, track the coefficient on `e_j`: in `x_j` the coefficient on `e_j` is `1` (nonzero); in `r_i^{(j)}`, after substitution, the coefficient on `e_j` is `-cov(x_i,x_j)/var(x_j)`, which is nonzero because I chose `x_i` with nonzero covariance. So I have two linear combinations of independent variables, `r_i^{(j)}` and `x_j`, that *share* the source `e_j` with nonzero weight in both, and `e_j` is non-Gaussian. Darmois–Skitovitch says: if two such combinations were independent, every shared source with nonzero coefficient in both would have to be Gaussian. Contrapositive — a shared non-Gaussian source with nonzero coefficients in both forces them to be *dependent*. So `r_i^{(j)}` and `x_j` are dependent. Therefore if `x_j` is not exogenous, it fails the independence test against at least one residual.

Putting the two directions together: `x_j` is exogenous if and only if `x_j` is independent of its residuals `r_i^{(j)}` for all `i != j`. That is a population-exact characterization of the source, and notice what it leans on — the non-Gaussianity of `e_j` is the *entire* reason the converse works. With Gaussian `e_j`, Darmois–Skitovitch would let independent combinations share a (Gaussian) source, the test would give false positives, and I would be back to the covariance ceiling. This is the same fuel as ICA, but spent on a direct test instead of a non-convex optimization.

Now I have to make the recursion legitimate, which means proving two more things: that peeling the source leaves a LiNGAM, and that the order I recover on the residuals is the order of the originals. For the first, take `x_j` exogenous; without loss of generality permute so `B` is already strictly lower triangular and `x_j = x_1`. Then `A` is lower triangular with unit diagonal, and since `x_1` is exogenous, `a_{i1}` equals the regression coefficient of `x_i` on `x_1`. Remove `x_1`'s effect from every `x_i` by least squares. The first column of `A` becomes zero — `x_1` no longer enters the residuals — and the submatrix of `A` obtained by deleting the first row and column is still lower triangular with unit diagonal. So the residual vector has the same linear-mixing form, `r^{(1)} = A^{(1)} e^{(1)}`, with a lower-triangular unit-diagonal mixing matrix and independent non-Gaussian sources: it is again a LiNGAM, one dimension smaller. For the second, because deleting the first row and column of a lower-triangular matrix leaves the relative order of the rest untouched, the causal order of the residuals equals the causal order of the corresponding originals: `k_{r}(l) < k_{r}(m)` iff `k(l) < k(m)`. So I may find the source of the residual system, and it is the second variable of the original order, and so on. The algorithm is forced: find the exogenous variable, append it to the order, regress it out of the rest, recurse on the residuals; after `p-1` peeling rounds one variable remains and goes last. Then, with the order in hand, the connection strengths are a triangular regression of each variable on its predecessors, with a sparse adaptive-lasso version useful in code for pruning redundant edges. The hard part was always the order, and covariance alone could never give it.

There is one practical landmine I have to step around. My exogeneity test is "`x_j` independent of every residual," but least-squares regression *guarantees* the residual is uncorrelated with the regressor — that is what least squares does, by construction, for *every* `j`, source or not. So uncorrelatedness is useless here; it is true for the wrong variables too. I need a measure of genuine *independence*, one that sees the higher-order dependence that uncorrelatedness misses. The canonical such measure is mutual information `I(y_1, y_2) = H(y_1) + H(y_2) - H(y_1, y_2)`, which is zero exactly when the variables are independent. So score each candidate `x_j` by the total dependence between it and its residuals,

```
T(x_j) = sum_{i != j} I(x_j, r_i^{(j)}),
```

and pick the variable that minimizes it — the most independent of its own residuals, i.e. the one most consistent with being the source. For a concrete `I`, a nonparametric kernel estimate (Bach & Jordan) from Gaussian-kernel Gram-matrix determinants works and is consistent, with the regularizer `kappa` and bandwidth `sigma` set by sample size. That gives a complete, convergent, parameter-light algorithm.

But let me look hard at the cost and small-sample behavior of that kernel choice, because it bothers me. The kernel MI estimates a genuinely two-dimensional dependency for each pair, needs a bandwidth and a regularizer, and the ordering step has the quoted cost `O(n p^3 M^2 + p^4 M^3)` with `M` the low-rank decomposition rank — and on small samples a 2-D kernel estimate is noisy. The structure of my problem is friendlier than full 2-D MI, though, and I think I can exploit it. I am always comparing the *same pair* of variables, just regressed two ways: candidate as cause versus candidate as effect. Let me set up the pairwise question cleanly. Standardize two variables `x` and `y` to zero mean and unit variance, and consider the two two-variable LiNGAMs: `x -> y`, i.e. `y = rho x + d` with disturbance `d` independent of `x`; and `y -> x`, i.e. `x = rho y + e` with `e` independent of `y`. The same `rho` appears in both because, for standardized variables, it is just the correlation coefficient. I want to decide which model the data prefer.

The principled comparator is the likelihood ratio. The log-likelihood of the `x -> y` model is `sum_t [G_x(x_t) + G_d((y_t - rho x_t)/sqrt(1-rho^2))] - T log sqrt(1-rho^2)`, where `G` denotes a standardized log-pdf and the last term normalizes for the variance scaling of the residual. Form the ratio against the reverse model and normalize by `T`:

```
R = (1/T) sum_t [ G_x(x_t) + G_d((y_t - rho x_t)/sqrt(1-rho^2))
                 - G_y(y_t) - G_e((x_t - rho y_t)/sqrt(1-rho^2)) ].
```

`R > 0` favors `x -> y`, `R < 0` favors `y -> x`. Now let me take the asymptotic limit and see what `R` becomes in information terms. The sample averages of the standardized log-pdfs converge to negative differential entropies, so `R -> -H(x) - H(d̂/sigma_d) + H(y) + H(ê/sigma_e)`, with `d̂ = y - rho x`, `ê = x - rho y`, and `sigma_d, sigma_e` the residual standard deviations. So the likelihood ratio is comparing the total negative entropy, or total non-Gaussianity, of the regressor-plus-residual pair in each direction. I never have to estimate a 2-D entropy: the joint terms cancel when I rewrite the same comparison as a mutual-information difference.

Let me make that cancellation explicit, because it is what justifies replacing kernel MI with one-dimensional entropies, and I want to be sure the algebra holds. Mutual information is `I(x, y) = H(x) + H(y) - H(x, y)`. Consider the linear map from `(x, y)` to `(x, d)` where `d = y - a x`: its matrix is `[[1, 0], [-a, 1]]`, with determinant 1. Differential entropy transforms under a linear map `T` by `H(Tu) = H(u) + log|det T|`, so with unit determinant `H(x, d) = H(x, y)`, and likewise `H(y, e) = H(x, y)`. Therefore the difference of the regressor-residual mutual informations is

```
I(x, d) - I(y, e) = [H(x) + H(d) - H(x, d)] - [H(y) + H(e) - H(y, e)]
                  = H(x) + H(d) - H(y) - H(e),
```

since the joint entropies are equal and cancel. Folding the residual variances into standardized residuals adds `log sigma_d` and `log sigma_e` terms, and those cancel too because both residual variances are `1 - rho^2`. Thus `I(x, d) - I(y, e) = H(x) + H(d̂/sigma_d) - H(y) - H(ê/sigma_e)`. With the likelihood-ratio sign above, this is `-R`: if `x -> y` is right, `I(x, d)` is the smaller mutual information and `I(x, d) - I(y, e)` is negative, while `R` is positive. Equivalently, `R = I(y, e) - I(x, d)`. So the criterion is still the same comparison: choose the direction in which the regressor is more independent of its residual, and use the sign convention that makes a positive pairwise score favor the candidate-as-cause direction. That is precisely the exogeneity test from the lemma, now in a form that needs only 1-D entropies. The kernel MI and this are after the same thing; this one is cheaper and never touches a 2-D density.

So I need a good, cheap estimate of a one-dimensional differential entropy of a standardized variable. This is exactly the negentropy-approximation problem from ICA. Differential entropy of a unit-variance variable is maximized by the Gaussian, so I can write entropy as the Gaussian entropy minus a non-Gaussianity penalty, and approximate the penalty by expectations of a couple of well-chosen nonquadratic contrast functions. The maximum-entropy approximation gives, for a standardized `u`,

```
Ĥ(u) = H(nu) - k1 [E{log cosh u} - gamma]^2 - k2 [E{u exp(-u^2/2)}]^2,
```

where `H(nu) = (1 + log 2pi)/2` is the entropy of the standard Gaussian, and the constants are `k1 ~ 79.047`, `k2 ~ 7.4129`, `gamma ~ 0.37457`. Let me make sure I understand every piece, because the constants are not arbitrary and the signs have to be right. The two bracketed terms are each a *non-Gaussianity* measure: a statistic of `u` minus the value that statistic takes for a Gaussian, squared. The first uses `log cosh` — a slowly-growing, robust even function, the same log-cosh contrast that ICA uses for super-Gaussian (sparse, heavy-tailed) sources — and `gamma` is precisely `E{log cosh}` under a standard Gaussian, so this whole bracket vanishes for a Gaussian, as a non-Gaussianity measure must. The second uses the odd function `u exp(-u^2/2)`, whose expectation under a symmetric Gaussian is zero, so it captures *asymmetry* / skewness. Both terms are subtracted: a more non-Gaussian variable has lower entropy than the Gaussian of the same variance, so we subtract the (squared, hence nonnegative) non-Gaussianity from the Gaussian ceiling. The constants `k1, k2` are the fixed weights from the maximum-entropy derivation that make this a second-order-accurate approximation around the Gaussian; I take them as given numbers, not things to tune. And the approximation is only valid for *standardized* `u`, which is why I must standardize each variable and divide each residual by its own standard deviation before feeding it in — the residual `r_i^{(j)}` is uncorrelated with the regressor but its variance is not 1, so `r_i^{(j)}/std(r_i^{(j)})` is the right argument.

Why fix the contrast functions instead of estimating each variable's log-pdf? Because a result well known in ICA is that the recovered direction is insensitive to the exact log-pdf used, as long as its shape is roughly right — and estimating a flexible log-pdf per variable means many parameters and is unreliable at small sample sizes, which is exactly the regime I care about. A fixed sparse-friendly contrast (`log cosh`) plus a skew term covers the super-Gaussian and asymmetric cases that dominate real data, with zero tuning. This is the same reason FastICA fixes `tanh` rather than learning the nonlinearity.

I can even see the simplest version of the criterion by first-order-expanding the likelihood ratio, which makes concrete *what* asymmetry it is reading. Approximate `G((y - rho x)/sqrt(1-rho^2)) ~ G(y) - rho x g(y) + O(rho^2)` with `g = G'`. Substituting into `R` and cancelling, the leading term is `R̃ = (rho/T) sum_t [-x_t g(y_t) + g(x_t) y_t]`. With the logistic-density contrast `G(u) = -2 log cosh((pi/(2 sqrt 3)) u)`, whose derivative is proportional to `tanh`, this becomes the nonlinear correlation `R̃_sparse = rho E{x tanh(y) - tanh(x) y}`. And a cumulant analysis pins down what it measures exactly: replacing `tanh` by its Taylor tail `tanh(u) = u - u^3/3 + ...`, the linear pieces cancel and the surviving cubic gives, under `x -> y`, `R̃ proportional to kurt(x)(rho^2 - rho^4)`. Since `|rho| < 1`, `rho^2 - rho^4 > 0`, so for positive-kurtosis (super-Gaussian) sources the sign of the criterion is the sign of the kurtosis and the direction is read off cleanly; the exact shape of the nonlinearity does not matter, only that it captures the higher-order asymmetry. The entropy version is the more robust, full form of the same idea — but seeing the cubic fall out tells me the criterion is genuinely keying on the third- and fourth-order structure that the covariance discarded, which is the whole point.

Now let me assemble the per-pair difference the algorithm will actually compute. For a pair `(i, j)`, regress each on the other, standardize the residuals, and form

```
diff_MI(i, j) = [ H(x_j_std) + H(r_i^{(j)}/std(r_i^{(j)})) ]
              - [ H(x_i_std) + H(r_j^{(i)}/std(r_j^{(i)})) ],
```

where the first bracket is the one-dimensional entropy sum for treating `j` as the cause, and the second is the corresponding sum for treating `i` as the cause. The common joint entropy is the same in the two directions and has already cancelled, so the difference is the leftover-dependence difference `I(x_j, r_i^{(j)}) - I(x_i, r_j^{(i)})`, which is exactly the likelihood-ratio sign for `i -> j`. Read the sign: if `i` is really the cause, regressing `j` on `i` leaves an independent residual so `I(x_i, r_j^{(i)})` is near zero, while regressing `i` on `j` the wrong way leaves a dependent residual so `I(x_j, r_i^{(j)})` is large — hence `diff_MI(i, j) > 0` exactly when `i`-as-cause is the more plausible direction. To turn these pairwise comparisons into a single exogeneity score for candidate `i`, I only want to count *evidence against* `i` being the source. If `i` is truly exogenous, then for every other `j` the pair should favor `i`-as-cause, i.e. `diff_MI(i, j)` should be positive or zero; any term where it comes out *negative* is a pair voting that `j`, not `i`, is upstream — evidence against `i`. So accumulate

```
M(i) = sum_{j != i} min(0, diff_MI(i, j))^2,
```

which sums the squared magnitude of just the votes against `i` (the `min(0, .)` keeps only the unfavorable sign, and squaring makes it a smooth penalty), and select the variable with the *least* accumulated counter-evidence. Storing `-M(i)` and taking the argmax is the same as taking the argmin of `M`. A genuinely exogenous variable collects essentially no counter-evidence, so its `M` is near zero and it wins. This is the pairwise specialization of the `T(x_j)` independence statistic — same "most independent of its residuals" decision, computed from cheap one-dimensional entropies with no kernel, no bandwidth, no 2-D density.

Let me also be clear about the convergence and cost story, since this is the entire reason I left ICA behind. There is no iterative search in parameter space anywhere: when `q` variables remain, a round scores all ordered pairs among those variables, then after the source is chosen it performs the `q-1` residual updates that peel that source out; after `p-1` nontrivial peeling rounds only the last variable remains. The exact population argument belongs to the independence characterization: with a true independence score, Lemma 1 identifies the exogenous variable, Lemma 2 keeps the residual system inside the same model class, and the order-preservation corollary lets the recursion recover the order. The `pwling` score I am about to code is the fixed one-dimensional entropy approximation to that independence comparison, so it keeps the finite-step, no-init, no-step-size structure while replacing the expensive kernel score by a cheaper likelihood-ratio proxy. And it is scale-invariant by construction, because I standardize every variable and every residual before scoring, so rescaling an input cannot change the order — the exact defect that made ICA-LiNGAM's permutations flip under normalization.

Let me write the whole thing as the code I would actually run, filling the one empty slot — the rule that picks the next variable in the causal order — with the entropy-difference criterion, and keeping the generic outer loop, residual peeling, and sparse triangular regression.

```python
import numpy as np
from sklearn.linear_model import LinearRegression, LassoLarsIC
from sklearn.utils import check_array


def _residual(xi, xj):
    """Residual of xi after least-squares projection on xj.
    cov(xi, xj)/var(xj) is the LS coefficient; the residual is uncorrelated
    with xj by construction -- which is exactly why we cannot use correlation
    as the exogeneity test and must measure genuine independence below."""
    return xi - (np.cov(xi, xj, bias=True)[0, 1] / np.var(xj)) * xj


def _entropy(u):
    """Maximum-entropy approximation of the differential entropy of a
    STANDARDIZED variable u (zero mean, unit variance). Gaussian entropy
    minus two non-Gaussianity penalties: a log-cosh (heavy-tail / sparsity)
    term and an odd u*exp(-u^2/2) (asymmetry / skew) term. gamma is E{log cosh}
    under a standard Gaussian, so the first bracket vanishes for a Gaussian."""
    k1 = 79.047
    k2 = 7.4129
    gamma = 0.37457
    return (1 + np.log(2 * np.pi)) / 2 \
        - k1 * (np.mean(np.log(np.cosh(u))) - gamma) ** 2 \
        - k2 * (np.mean(u * np.exp((-u ** 2) / 2))) ** 2


def _diff_mutual_info(xi_std, xj_std, ri_j, rj_i):
    """Entropy-difference between the two directions for the pair (i, j).
    The two brackets are one-dimensional entropy sums. Their difference equals
    I(x_j, r_i^{(j)}) - I(x_i, r_j^{(i)}) after the common joint entropy cancels,
    so diff > 0 means i-as-cause is more plausible. Residuals are standardized
    because the entropy approximation is only valid for unit-variance arguments."""
    return (_entropy(xj_std) + _entropy(ri_j / np.std(ri_j))) \
        - (_entropy(xi_std) + _entropy(rj_i / np.std(rj_i)))


def _predict_adaptive_lasso(X, predictors, target, gamma=1.0):
    """Adjacency-stage regression used after the order is known.
    The initial least-squares fit supplies adaptive weights; BIC selects the
    lasso penalty on the weighted predictors, then coefficients are unweighted."""
    lr = LinearRegression()
    lr.fit(X[:, predictors], X[:, target])
    weight = np.power(np.abs(lr.coef_), gamma)
    reg = LassoLarsIC(criterion="bic")
    reg.fit(X[:, predictors] * weight, X[:, target])
    return reg.coef_ * weight


def _search_causal_order(X, U):
    """Return the index in U most consistent with being exogenous: the one
    that accumulates the least evidence *against* being the source. For each
    candidate i, sum the squared unfavorable-sign pairwise differences; a true
    source collects ~0, so we take the argmax of -M (= argmin of M)."""
    if len(U) == 1:
        return int(U[0])
    M_list = []
    for i in U:
        M = 0.0
        for j in U:
            if i == j:
                continue
            xi_std = (X[:, i] - np.mean(X[:, i])) / np.std(X[:, i])
            xj_std = (X[:, j] - np.mean(X[:, j])) / np.std(X[:, j])
            ri_j = _residual(xi_std, xj_std)   # i regressed on j
            rj_i = _residual(xj_std, xi_std)   # j regressed on i
            M += np.min([0.0, _diff_mutual_info(xi_std, xj_std, ri_j, rj_i)]) ** 2
        M_list.append(-1.0 * M)
    return int(U[np.argmax(M_list)])


def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    """X: (n_samples, n_variables) -> adjacency B with B[i, j] != 0 meaning j -> i."""
    X = check_array(X)
    n = X.shape[1]
    U = np.arange(n)         # indices not yet placed in the order
    K = []                   # the recovered causal order
    Xw = np.copy(X)
    for _ in range(n):
        m = _search_causal_order(Xw, U)        # find the exogenous variable
        for i in U:                            # peel its effect off the rest:
            if i != m:                         # residuals are again a LiNGAM
                Xw[:, i] = _residual(Xw[:, i], Xw[:, m])
        K.append(m)
        U = U[U != m]

    # Order known -> sparse triangular regression on the original data.
    B = np.zeros((n, n), dtype=float)
    for rank in range(1, len(K)):
        target = K[rank]
        predictors = K[:rank]                  # everything earlier in the order
        B[target, predictors] = _predict_adaptive_lasso(X, predictors, target)
    return B
```

Let me retrace the causal chain so I am sure nothing is hand-waved. I started boxed in: covariance is direction-blind, so on observational data the best a second-order method can do is a Markov equivalence class, and on two variables it cannot direct the edge at all. Non-Gaussian disturbances put structure above second order, and rewriting `x = (I-B)^{-1} e` revealed the model is exactly ICA, whose mixing matrix is identifiable up to permutation/scale/sign — so the directions are recoverable in principle. Running ICA and undoing its indeterminacies gives a working estimator, but it inherits ICA's non-convex iterative search (local minima, no finite-step guarantee, init/step/stop knobs) and uses scale-dependent permutations that flip under normalization. To avoid all of that I went after the causal order *directly*: acyclicity guarantees an exogenous variable equal to its own source, and I proved — using Darmois–Skitovitch, which needs the non-Gaussianity — that a variable is exogenous if and only if it is independent of all its least-squares residuals. Peeling that variable out leaves a smaller LiNGAM with the same relative order, so the procedure recurses for `p-1` peeling rounds, appends the last remaining variable, and reads off the strengths by a sparse triangular regression. Because least-squares residuals are always *uncorrelated* with the regressor, I needed a true independence measure, not correlation; mutual information is the right one, and the likelihood-ratio between the two pairwise directions can be written with only *one-dimensional* differential entropies because the joint entropies cancel under a unit-determinant map. The code uses the sign convention `diff_MI(i, j) = I(j-as-cause leftover) - I(i-as-cause leftover)`, so positive values favor `i` as the cause. I estimate those one-dimensional entropies with the fixed maximum-entropy approximation built from a log-cosh non-Gaussianity term and a skew term — no kernel, no bandwidth, no per-variable density fitting, standardizing every variable and residual so the approximation is valid and the whole procedure is scale-invariant. Aggregating only the evidence *against* each candidate being the source and picking the least-penalized one gives the exogenous variable each round. The result is a direct, fixed-step estimator of the full directed DAG from observational non-Gaussian data, with exactness coming from the residual-independence characterization and the code using the `pwling` entropy approximation to make that comparison practical.
