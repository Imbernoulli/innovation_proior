We are given a high-dimensional state $\mathbf{x}\in\mathbb{R}^n$ — a vorticity field, a pressure distribution, an image — and we may place only $p\ll n$ point sensors, each reading a single component $x_{\gamma_i}$. The task is to choose the locations $\gamma_1,\dots,\gamma_p$ so that the full field can be reconstructed from those $p$ readings. The obstacle is combinatorial: there are $\binom{n}{p}$ possible subsets, far too many to enumerate for realistic grid sizes. Even if the field is intrinsically low-rank, a bad choice of sensor rows can leave some modal coordinates nearly invisible to the measurements, so reconstruction error explodes in the presence of noise. The original gappy POD framework reconstructs missing entries by least-squares fitting of POD modes, but it leaves the mask of observed entries arbitrary or random and gives no guarantee on the conditioning of the resulting linear system. Empirical interpolation methods such as DEIM produce a square $p=r$ selection in a mode-by-mode fashion that depends on the ordering of the basis and offers no natural way to use more sensors than modes. Convex relaxations for D-optimal design are principled, but each Newton iteration costs $O(n^3)$, the optimization must be rerun for every new $p$, and rounding fractional weights back to a hard subset discards the optimality certificate. What is needed is a single, scalable factorization that returns a ranked list of sensor locations and works for both the square and oversampled cases.

The saving structure is that physical states of this kind live on a low-dimensional attractor. By collecting training snapshots, stacking them as columns of $\mathbf{X}\in\mathbb{R}^{n\times m}$, and taking the truncated singular value decomposition, we obtain the leading $r$ POD modes $\boldsymbol{\Psi}_r\in\mathbb{R}^{n\times r}$. The Eckart-Young theorem guarantees that this is the optimal rank-$r$ least-squares approximation of the data, so any state is well represented as $\mathbf{x}\approx\boldsymbol{\Psi}_r\mathbf{a}$ with $\mathbf{a}\in\mathbb{R}^r$ and $r\ll n$. Point sensors then pick out $p$ rows of $\boldsymbol{\Psi}_r$, giving the small linear system $\mathbf{y}=\mathbf{C}\mathbf{x}\approx(\mathbf{C}\boldsymbol{\Psi}_r)\mathbf{a}=\boldsymbol{\Theta}\mathbf{a}$. Recovering $\mathbf{a}$ from $\mathbf{y}$ is a least-squares problem whose sensitivity to noise is governed by the condition number of $\boldsymbol{\Theta}$ and whose error covariance is $\eta^2(\boldsymbol{\Theta}^\top\boldsymbol{\Theta})^{-1}$. Therefore the sensor placement problem is exactly the problem of choosing rows of $\boldsymbol{\Psi}_r$ so that $\boldsymbol{\Theta}$ is as well-conditioned as possible, which corresponds to maximizing $\det(\boldsymbol{\Theta}^\top\boldsymbol{\Theta})$, the D-optimal experimental design criterion.

I propose **Gappy POD with QR-pivot sensor placement**. The method uses column-pivoted QR to greedily maximize the volume of the selected row set. When $p=r$, I factor $\boldsymbol{\Psi}_r^\top$ and take the first $r$ pivots; this maximizes $|\det(\mathbf{C}\boldsymbol{\Psi}_r)|$ and is equivalent to the Q-DEIM selection. When $p>r$, I factor the $n\times n$ Gram matrix $\boldsymbol{\Psi}_r\boldsymbol{\Psi}_r^\top$ and take the first $p$ pivots; because $\det(\boldsymbol{\Theta}^\top\boldsymbol{\Theta})$ equals the product of the leading singular values of the corresponding principal submatrix, this step greedily maximizes the D-optimal objective for the oversampled case. In both branches the pivots are returned as a hierarchical ranking, so a single factorization yields the best $r$ sensors, then the best $r+1$, and so on, with no re-solve. The reconstruction step is standard gappy POD least squares: fit the modal coordinates to the measured entries and lift back to the full state.

```python
from scipy.linalg import lstsq, qr, solve
from sklearn.decomposition import TruncatedSVD


def pod_basis(X, r):
    """Leading r POD/SVD modes of centered snapshots X with shape (n_samples, n_features)."""
    svd = TruncatedSVD(n_components=r).fit(X)
    return svd.components_.T                          # Psi_r, shape (n_features, r)


def select_sensors(Psi_r, p):
    """Rank p >= r sensor rows by greedy QR volume maximization."""
    _, r = Psi_r.shape
    if p < r:
        raise ValueError("QR gappy-POD reconstruction requires p >= r sensors")
    if p == r:
        _, _, pivots = qr(Psi_r.conj().T, pivoting=True)
    else:
        _, _, pivots = qr(Psi_r @ Psi_r.conj().T, pivoting=True)
    return pivots[:p]


def reconstruct(Psi_r, sensors, y):
    """Fit modal coordinates to measured entries and lift back to the full state."""
    Theta = Psi_r[sensors, :]                         # C Psi_r
    if Theta.shape[0] == Theta.shape[1]:
        a_hat = solve(Theta, y)                       # p == r
    else:
        a_hat, *_ = lstsq(Theta, y)                   # p > r
    return Psi_r @ a_hat


# Usage:
# Psi_r = pod_basis(X_train, r)
# gamma = select_sensors(Psi_r, p)
# x_hat = reconstruct(Psi_r, gamma, x_test[gamma])
```
