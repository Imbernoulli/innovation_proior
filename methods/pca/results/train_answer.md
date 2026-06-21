We observe $n$ individuals, each carrying $q$ correlated measurements — stature and leg-length, a battery of test scores, several sensors recording the same dynamics — and we want a compact, low-dimensional stand-in for where the cloud of points actually lives. The natural first tool is the regression line: name one variable $y$ as the thing to be predicted and minimize the squared residuals in $y$, $S(y'-y)^2$. But this answer is the wrong shape. Swap the roles and regress $x$ on $y$, minimizing $S(x'-x)^2$, and a *different* line runs through the same points. There are two regression lines, not one, and the reason is built in: the residual is measured along a single coordinate axis, so the fit secretly presumes that axis is exact and the other carries all the error. That is false when every coordinate is observed with error — a man at a given instant has one true position, yet both his recorded time and his recorded position wobble from trial to trial. We need a criterion symmetric in all $q$ variables that returns one unique flat. The factor-analytic alternative (a few latent traits explaining the correlations) fails differently: it fixes the number of latent factors in advance by hypothesis, its consistency conditions are essentially never met exactly in a finite sample, and when fewer factors than variables are posited the latent axes can be freely rotated, so it never delivers one determinate, ordered decomposition of the variance.

I propose Principal Component Analysis. The symmetric notion of distance from a point to a line is the *perpendicular* distance, which treats every direction alike and does not care which axis is called dependent, so I make the objective the sum of squared perpendicular distances $U = S(p^2)$ and minimize it. This is exactly the second moment of the point system about the flat, and a fact borrowed from dynamics — that the second moment about a family of parallel flats is least for the one through the centroid — tells me the best fit passes through the mean. It falls out of the algebra too: writing a plane as $\sum_i l_i x_i = p$ with $\sum_i l_i^2 = 1$, differentiating $U=S\big((\sum_i l_i x_i)-p\big)^2$ first in the unconstrained $p$ gives $p=\sum_i l_i\,\bar x_i = l\cdot\bar x$, so the plane goes through the centroid. After shifting the origin there I am only hunting a *direction* $l$. Adjoining the constraint with a Lagrange multiplier $Q$ and varying $l_u$ yields, for each $u$, $\sum_v(\sigma_u\sigma_v r_{uv})\,l_v + (Q/n)\,l_u = 0$, that is $C\,l = \Sigma^2 l$ with $C_{uv}=\sigma_u\sigma_v r_{uv}$ the covariance matrix. Multiplying through by $l_u$ and summing, the quadratic form is $U_m/n$ and the constraint collapses the other term, giving $Q=-U_m$ and $Q/n=-\Sigma^2$: the multiplier I introduced as bookkeeping *is* the residual variance, up to sign. So the stationary directions are eigenvectors of the second-moment matrix, and the eigenvalue carries the meaning of the fit. For a plane the residual is the variance along the discarded normal, so I take the *least* root of $\det(C-\Sigma^2 I)=0$.

The one-dimensional summary — the dominant direction of the cloud — needs the best-fitting *line*. The squared perpendicular distance of a centered point from the line through the origin with unit direction $l$ is, by Pythagoras, $|x-\bar x|^2-(l\cdot(x-\bar x))^2$, so the mean perpendicular residual is
$$\Sigma'^2 = \big(\sigma_1^2+\cdots+\sigma_q^2\big) - \operatorname{var}\!\big(l\cdot(x-\bar x)\big).$$
The first bracket, the total of the per-coordinate variances, does not depend on $l$ at all; the second is the variance of the data projected onto $l$. Total scatter splits Pythagoreanly into "along" plus "perpendicular," and the total is fixed, so minimizing the perpendicular residual is *identical* to maximizing the projected variance. The best-fitting line is therefore the eigenvector of $C$ belonging to the *greatest* eigenvalue — the mirror image of the plane case. In two dimensions this is the major axis of the correlation ellipse, $\tan 2\theta = 2 r_{xy}\sigma_x\sigma_y/(\sigma_x^2-\sigma_y^2)$: a single symmetric direction at last, instead of two regression lines. And when the ellipsoid degenerates to a sphere — all correlations zero, all $\sigma$ equal — every direction is equally good and the method correctly refuses to pick one.

Coming at the reductive problem from the accounting side gives the same object and orders the components. Seek mutually uncorrelated unit-variance new variables $\gamma_j$ with $z_i=\sum_j a_{ij}\gamma_j$, and choose $\gamma_1$ to explain as much of the total variance of the $z$'s as possible. Writing the first component as a score $\gamma=w^\top z$ with $\operatorname{var}(\gamma)=w^\top R w=1$, its loading vector is $a=Rw$ and the variance it accounts for is $S=a^\top a=w^\top R^2 w$. Maximizing $w^\top R^2 w$ subject to $w^\top R w=1$ gives $R^2 w = k\,Rw$, i.e. $R\,a = k\,a$ on the correlation matrix, with $S=k$. So once more an eigenproblem: the first component's loadings are the top eigenvector of $R$, and to explain the most variance I take the greatest root of the characteristic equation $\det(R-kI)=0$. This is Cauchy's equation for the principal axes of a quadric; he showed every root is real, and here each root is non-negative because it is a variance — a negative root would make the constant-density locus a hyperboloid of infinite total probability, which is absurd. Real, non-negative, ordered: exactly what an importance ordering needs. That the geometric least-squares route and the variance-accounting route land on the same eigenproblem of the second-moment matrix is the confirmation that this is the right object.

The remaining components write themselves. After $\gamma_1$ is fixed, the next should explain as much of the *residual* variance as possible. Its rank-one contribution to the cross-structure is $a_1 a_1^\top$ (with the loading scaled so $a_1^\top a_1=k_1$), and the residual structure is $R' = R - a_1 a_1^\top$ — precisely the correlation matrix of the residuals $z_i-a_{i1}\gamma_1$. Then $\gamma_2$ is the top eigenvector of $R'$, and so on, peeling one rank-one layer at a time; because the eigenvectors of a symmetric matrix are mutually orthogonal, this deflation is automatic and the full eigendecomposition hands me every component at once, ordered by eigenvalue. The fraction of total variance kept by $m$ components is $(k_1+\cdots+k_m)/\sum_i k_i$ — for the correlation matrix the total is the trace $q$ — and that fraction is the stopping rule: drop the small-eigenvalue directions where the cloud is thin, the noise and redundancy. Whether to diagonalize the covariance or the correlation matrix is a choice of metric: with comparable, meaningful units, center only and use the covariance; with incommensurable units, standardize $z_i=(x_i-\bar x_i)/\sigma_i$ first and use the correlation matrix, so that an arbitrary rescaling cannot squeeze the ellipsoid and move the axes. I keep centering as the default.

For actual computation, expanding $\det(R-kI)=0$ by hand is brutal, so the hand-engine is power iteration: hit a trial direction with the matrix, $a\to Ra\to R^2a\to\cdots$. Expanding $a=\sum_j c_j v_j$ in the eigenbasis gives $R^t a=\sum_j c_j k_j^t v_j$, and since $k_1$ is largest, $k_1^t$ dominates geometrically, so the iterate aligns with the top eigenvector and the per-step scale factor converges to $k_1$; normalize so $\sum a_i^2=k_1$, deflate, and repeat. The industrial form of the identical object is the singular value decomposition of the centered data. With $X_c$ of shape $(n,q)$, forming $C=X_c^\top X_c$ and calling a symmetric eigensolver squares the condition number, so small singular values get squared and drown in roundoff exactly on the small-variance directions. Instead factor $X_c = U S V^\top$; then $X_c^\top X_c = V S^2 V^\top$, so the right singular vectors $V$ *are* the eigenvectors of the covariance — the principal directions — the variance of component $i$ is $\mathrm{explained\_variance}_i = S_i^2/(n-1)$, and the embedding is $X_c V[:,:k] = (US)[:,:k]$, so the coordinates can be read straight off the SVD. The singular-vector signs are arbitrary ($USV^\top=(-U)S(-V)^\top$), so I fix each right singular vector's sign deterministically — forcing the largest-magnitude entry of each row of $V^\top$ positive — to make the embedding reproducible across runs and LAPACK builds.

```python
import numpy as np
from numpy.typing import NDArray
from scipy import linalg


def svd_flip(U: NDArray[np.float64], Vt: NDArray[np.float64]):
    """Deterministic signs from rows of Vt, as in full SVD with u_based_decision=False."""
    max_abs = np.argmax(np.abs(Vt), axis=1)
    signs = np.sign(Vt[np.arange(Vt.shape[0]), max_abs])
    return U * signs[np.newaxis, :], Vt * signs[:, np.newaxis]


class PCA:
    """Project data onto its top-k directions of greatest variance,
    via the SVD of the centered data matrix."""

    def __init__(self, n_components: int = 2, random_state: int | None = None):
        self.n_components = n_components
        self.random_state = random_state

    def _fit_full_svd(self, X: NDArray[np.float64]):
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        self.mean_ = X.mean(axis=0)                      # centroid
        Xc = X - self.mean_                              # center
        U, S, Vt = linalg.svd(Xc, full_matrices=False)  # SVD, not eig(Xc^T Xc)
        U, Vt = svd_flip(U, Vt)                          # reproducible signs
        k = self.n_components
        self.components_ = Vt[:k]                        # top-k principal axes (rows)
        self.explained_variance_ = (S[:k] ** 2) / (n_samples - 1)      # variance per component
        self.singular_values_ = S[:k]
        total = (S ** 2).sum() / (n_samples - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / total
        self.n_samples_ = n_samples
        return U, S, Vt, Xc

    def fit(self, X: NDArray[np.float64]) -> "PCA":
        self._fit_full_svd(X)
        return self

    def transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        Xc = np.asarray(X, dtype=np.float64) - self.mean_
        return Xc @ self.components_.T                   # coordinates in principal frame

    def fit_transform(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        U, S, Vt, Xc = self._fit_full_svd(X)
        # Xc @ components_.T == (U S)[:, :k]
        return U[:, : self.n_components] * S[: self.n_components]
```
