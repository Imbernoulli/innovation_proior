ICA-LiNGAM confirmed the thesis on the easy scenarios and exposed exactly where the ICA *machinery* breaks on the hard one. ER30 went from NOTEARS' unstable F1 0.919 (SHD 16, seed 456 collapsing) to a near-perfect F1 0.989, SHD 2.3, recall a clean 1.0 on every seed; ER50 went from F1 0.872, SHD 58 to F1 0.994, SHD 3.0, again recall 1.0 throughout — the heavy-tailed Laplace and skewed exponential noise are ideal ICA fuel and the missing-edge problem vanished. But SF100 tells the real story, and it flipped the failure mode: F1 rose only to 0.804 (from 0.716), SHD stayed catastrophic at 120, and the asymmetry inverted — NOTEARS was high-precision/low-recall (it missed edges), whereas ICA-LiNGAM is now high-recall (0.944) and low-*precision* (0.702), finding the true edges but spewing false ones, with one ugly outlier (seed 123 SHD **159**, F1 0.756). That fingerprint points at one culprit: the single *global, non-convex* FastICA fit on a 100-dimensional unmixing under *uniform* (sub-Gaussian, light-tailed) noise — precisely where ICA's super-Gaussian-tuned contrast is weakest and a non-convex optimization over a 100-dim matrix lands in a bad local optimum, scrambling the separating matrix, corrupting the assignment-based permutation, and yielding a wrong causal order into which the adaptive lasso then dutifully fits dense, false edges. There are two further pathologies I now have evidence for: the ICA step has knobs (initial guess, step size, convergence tolerance) with no principled setting, and both its permutation cost $1/\lvert W_{ii}\rvert$ and its lower-triangularization scoring are *not scale-invariant*, even though the causal order cannot depend on a variable's variance. So the move is forced: keep the non-Gaussianity — it is what makes the problem identifiable and what won ER30 and ER50 — but discard the ICA apparatus wholesale and recover the order *directly*, one variable at a time, with no iterative search in parameter space anywhere.

I propose **DirectLiNGAM**. The structural anchor is that a DAG with no latent confounders always has at least one variable with no parents — an exogenous source — and for such an $x_j$ the model reads $x_j = e_j$: the variable simply *equals* its own independent non-Gaussian disturbance. If I can find that variable I know it sits first in the order; peeling its effect off the others leaves a LiNGAM with one fewer variable whose source I find next, building the whole order in $d-1$ rounds of regression with no non-convex optimization. The entire method reduces to one question: how do I detect, from data, which variable is exogenous? The test is to tentatively treat $x_j$ as the source, regress every other $x_i$ on it by least squares to form the residual $r_i^{(j)} = x_i - (\mathrm{cov}(x_i,x_j)/\mathrm{var}(x_j))\,x_j$, and ask whether $x_j$ is *independent* of these residuals. Both directions of the equivalence hold. If $x_j=e_j$ truly is exogenous, then writing $x_i = a_{ij}x_j + \bar e_i^{(j)}$ with $\bar e_i^{(j)}$ collecting every source other than $e_j$, independence of the sources makes $\mathrm{cov}(x_i,x_j)=a_{ij}\mathrm{var}(x_j)$, so the regression coefficient is exactly $a_{ij}$ and the residual $r_i^{(j)} = \bar e_i^{(j)}$ is precisely the bundle of *other* sources, independent of $x_j$ — an exogenous variable is independent of every residual. The converse is where non-Gaussianity is indispensable: if $x_j$ is *not* exogenous, with a nonempty parent set, some parent $x_i$ has nonzero covariance with it (the parent covariance is positive definite and the weights are nonzero), and substituting through to the sources, both $x_j$ and $r_i^{(j)}$ are linear combinations of the independent $e$ that share the non-Gaussian source $e_j$ with nonzero coefficient in both (coefficient $1$ in $x_j$, $-\mathrm{cov}(x_i,x_j)/\mathrm{var}(x_j)\neq 0$ in $r_i^{(j)}$). By Darmois–Skitovitch, two linear combinations of independent variables sharing a *non-Gaussian* source with nonzero weight in both must be *dependent* — so $r_i^{(j)}$ and $x_j$ are dependent and $x_j$ fails the test. Combining both directions: $x_j$ is exogenous **iff** it is independent of all its least-squares residuals, with the non-Gaussianity of $e_j$ the entire reason the converse holds. The recursion is legitimate because removing the source's effect by least squares zeros its column in $A$ and leaves a lower-triangular submatrix with unit diagonal — again a LiNGAM, one dimension smaller, with the same relative order — so peeling sources one at a time recovers the true order.

The landmine is that least squares *guarantees* the residual is uncorrelated with the regressor for *every* $j$, source or not, so uncorrelatedness is useless and I need a measure of genuine independence that sees the higher-order dependence. A kernel mutual-information estimate would work but needs a bandwidth and a regularizer, costs $O(nd^3M^2 + d^4M^3)$, and is noisy on small samples — exactly the SF100 regime that just broke ICA. The structure of my problem is friendlier than full 2-D MI: I always compare the *same pair* regressed two ways, candidate-as-cause versus candidate-as-effect, so the principled comparator is the likelihood ratio between $x\to y$ and $y\to x$, and its key simplification is that the joint entropy cancels. The linear map $(x,y)\mapsto(x,d)$ with $d=y-ax$ has determinant 1, and differential entropy transforms as $H(Tu)=H(u)+\log\lvert\det T\rvert$, so $H(x,d)=H(x,y)=H(y,e)$, and therefore
$$I(x,d) - I(y,e) = H(x)+H(\hat d/\sigma_d) - H(y) - H(\hat e/\sigma_e),$$
a difference of **one-dimensional** entropies only — the comparison "choose the direction in which the regressor is more independent of its residual," never a 2-D density. The remaining ingredient is a good, cheap 1-D differential entropy of a standardized variable, which is the negentropy-approximation problem: entropy is the Gaussian entropy minus a non-Gaussianity penalty,
$$\hat H(u) = \frac{1+\log 2\pi}{2} - k_1\big[\mathbb{E}\{\log\cosh u\} - \gamma\big]^2 - k_2\big[\mathbb{E}\{u\,e^{-u^2/2}\}\big]^2,$$
with $k_1\approx 79.047$, $k_2\approx 7.4129$, $\gamma\approx 0.37457$. The first bracket uses the even $\log\cosh$ contrast (robust for super-Gaussian heavy-tailed sources, with $\gamma$ the Gaussian value of $\mathbb{E}\{\log\cosh\}$ so it vanishes for a Gaussian), the second uses the odd $u\,e^{-u^2/2}$ to capture skew; both are subtracted because a more non-Gaussian variable has *lower* entropy than the Gaussian of the same variance, and the constants are the fixed maximum-entropy weights making the approximation second-order accurate around the Gaussian — given numbers, not knobs. Fixing the contrasts rather than fitting each variable's log-pdf is deliberate: the recovered direction is insensitive to the exact log-pdf as long as the shape is roughly right, and per-variable density fitting is many unreliable parameters at small sample sizes — exactly what broke the global ICA on SF100. Crucially, because the approximation is valid only for *standardized* $u$, every variable and every residual is divided by its own standard deviation before scoring, which makes the whole criterion **scale-invariant** by construction — the exact defect that made the ICA permutation fragile is gone.

Assembling the per-pair quantity: for $(i,j)$, regress each on the other, standardize the residuals, and form $\mathrm{diff\_MI}(i,j) = [H(x_j^{\text{std}}) + H(r_i^{(j)}/\sigma)] - [H(x_i^{\text{std}}) + H(r_j^{(i)}/\sigma)]$, which equals $I(x_j, r_i^{(j)}) - I(x_i, r_j^{(i)})$ after the joint entropy cancels. If $i$ is truly the cause, regressing $j$ on $i$ leaves an independent residual (second bracket near zero) while regressing $i$ on $j$ the wrong way leaves a dependent residual (first bracket large), so $\mathrm{diff\_MI}(i,j)>0$ exactly when $i$-as-cause is more plausible. To turn pairwise comparisons into one exogeneity score for candidate $i$ I count only *evidence against* it: if $i$ is exogenous every pair should favor $i$-as-cause ($\mathrm{diff\_MI}(i,j)\ge 0$), so I accumulate $M(i) = \sum_{j\neq i}\min(0,\mathrm{diff\_MI}(i,j))^2$ — the squared magnitude of just the unfavorable-sign votes — and select the variable with the least counter-evidence (storing $-M(i)$ and taking the argmax). A genuinely exogenous variable collects $\approx 0$ and wins. There is no iterative search in parameter space anywhere: each round scores all ordered pairs among the remaining variables, picks the source, peels it out with $q-1$ residual updates, and after $d-1$ rounds the last variable goes last. Once the order is in hand the connection strengths come from the *same* `causal-learn` `_BaseLiNGAM._estimate_adjacency_matrix` the previous rung used — adaptive lasso per node (OLS pilot, $\lvert\text{coef}\rvert^{\gamma}$ weights with $\gamma=1$, BIC-selected lasso, unweight), consistent selection driving absent edges to exactly zero, returned with $B[i,j]\neq 0$ meaning $j\to i$. The only thing that changed from the ICA rung is the engine producing the causal order: the global non-convex ICA fit is replaced by a deterministic, scale-invariant, pairwise-entropy peeling with no initialization, no step size, and no random optimization in the ordering step.

```python
def run_causal_discovery(X: np.ndarray) -> np.ndarray:
    """
    Input:  X of shape (n_samples, n_variables)
    Output: adjacency matrix B of shape (n_variables, n_variables)
            B[i, j] != 0  means j -> i  (follows causal-learn convention)
    """
    import os
    import numpy as np
    from sklearn.utils import check_array
    from causallearn.search.FCMBased.lingam.base import _BaseLiNGAM

    X = check_array(X)
    seed = int(os.environ.get("SEED", "42"))
    n_features = X.shape[1]

    # Core DirectLiNGAM steps (from causallearn.search.FCMBased.lingam.direct_lingam)
    def _residual(xi: np.ndarray, xj: np.ndarray) -> np.ndarray:
        return xi - (np.cov(xi, xj)[0, 1] / np.var(xj)) * xj

    def _entropy(u: np.ndarray) -> float:
        k1 = 79.047
        k2 = 7.4129
        gamma = 0.37457
        return (1 + np.log(2 * np.pi)) / 2 -                k1 * (np.mean(np.log(np.cosh(u))) - gamma) ** 2 -                k2 * (np.mean(u * np.exp((-u ** 2) / 2))) ** 2

    def _diff_mutual_info(
        xi_std: np.ndarray,
        xj_std: np.ndarray,
        ri_j: np.ndarray,
        rj_i: np.ndarray,
    ) -> float:
        return (_entropy(xj_std) + _entropy(ri_j / np.std(ri_j))) -                (_entropy(xi_std) + _entropy(rj_i / np.std(rj_i)))

    def _search_causal_order(X_work: np.ndarray, U: np.ndarray) -> int:
        if len(U) == 1:
            return int(U[0])
        M_list = []
        for i in U:
            M = 0.0
            for j in U:
                if i == j:
                    continue
                xi_std = (X_work[:, i] - np.mean(X_work[:, i])) / np.std(X_work[:, i])
                xj_std = (X_work[:, j] - np.mean(X_work[:, j])) / np.std(X_work[:, j])
                ri_j = _residual(xi_std, xj_std)
                rj_i = _residual(xj_std, xi_std)
                M += np.min([0.0, _diff_mutual_info(xi_std, xj_std, ri_j, rj_i)]) ** 2
            M_list.append(-1.0 * M)
        return int(U[np.argmax(M_list)])

    U = np.arange(n_features)
    K = []
    X_work = np.copy(X)
    for _ in range(n_features):
        m = _search_causal_order(X_work, U)
        for i in U:
            if i != m:
                X_work[:, i] = _residual(X_work[:, i], X_work[:, m])
        K.append(m)
        U = U[U != m]

    class _LocalDirectLiNGAM(_BaseLiNGAM):
        def fit(self, X):
            return self

    model = _LocalDirectLiNGAM(random_state=seed)
    model._causal_order = K
    model._estimate_adjacency_matrix(X)
    return model.adjacency_matrix_
```
