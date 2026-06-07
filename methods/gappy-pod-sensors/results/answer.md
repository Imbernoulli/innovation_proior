# QR-pivot sensor placement with gappy-POD reconstruction

## Problem

Place $p\ll n$ point sensors, each reading one component $x_{\gamma_i}$ of a high-dimensional state $\mathbf{x}\in\mathbb{R}^n$, so the full field can be reconstructed from the $p$ readings. Exact optimal placement is a $\binom{n}{p}$ combinatorial search; the goal is a scalable, noise-robust surrogate built from standard linear algebra.

## Key idea

1. **Tailored low-rank basis.** From training snapshots $\mathbf{X}\in\mathbb{R}^{n\times m}$ take the SVD and keep the leading $r$ left singular vectors (POD modes) $\boldsymbol{\Psi}_r\in\mathbb{R}^{n\times r}$. By Eckart-Young this is the optimal rank-$r$ least-squares basis, so $\mathbf{x}\approx\boldsymbol{\Psi}_r\mathbf{a}$ with $\mathbf{a}\in\mathbb{R}^r$, $r\ll n$.
2. **Reconstruct by gappy POD.** Point sensors $\mathbf{C}=[\mathbf{e}_{\gamma_1}\cdots\mathbf{e}_{\gamma_p}]^\top$ give $\mathbf{y}=\mathbf{C}\mathbf{x}\approx\boldsymbol{\Theta}\mathbf{a}$ with $\boldsymbol{\Theta}=\mathbf{C}\boldsymbol{\Psi}_r=\boldsymbol{\Psi}_r(\gamma,:)$. Estimate $\hat{\mathbf{a}}=\boldsymbol{\Theta}^\dagger\mathbf{y}$ (= $\boldsymbol{\Theta}^{-1}\mathbf{y}$ when $p=r$) and lift $\hat{\mathbf{x}}=\boldsymbol{\Psi}_r\hat{\mathbf{a}}$.
3. **Place sensors to condition $\boldsymbol{\Theta}$.** Under noise $\mathbf{y}=\boldsymbol{\Theta}\mathbf{a}+\boldsymbol{\xi}$, $\boldsymbol{\xi}\sim\mathcal{N}(0,\eta^2\mathbf{I})$, the error covariance is $\eta^2(\boldsymbol{\Theta}^\top\boldsymbol{\Theta})^{-1}$ and worst-case error amplification is $\kappa(\boldsymbol{\Theta})$. Minimizing the error-ellipsoid volume is the D-optimal criterion $\max_\gamma\det(\boldsymbol{\Theta}^\top\boldsymbol{\Theta})$.
4. **Greedy volume selection.** Pivoted QR picks the next column of largest residual norm and deflates the rest, greedily maximizing the diagonal product $\prod_i|r_{ii}|=\prod_i\sigma_i=|\det|$, with diagonal dominance $|r_{ii}|^2\ge\sum_{j=i}^k|r_{jk}|^2$.

## Algorithm

Given $\boldsymbol{\Psi}_r$ and a sensor count $p\ge r$:
$$\gamma=\begin{cases}\mathrm{qrPivot}(\boldsymbol{\Psi}_r^\top,\ r), & p=r\quad(\text{cost }O(nr^2)),\\[2pt]\mathrm{qrPivot}(\boldsymbol{\Psi}_r\boldsymbol{\Psi}_r^\top,\ p), & p>r\quad(\text{cost }O(n^3)),\end{cases}\qquad \mathbf{C}=[\mathbf{e}_{\gamma_1}\cdots\mathbf{e}_{\gamma_p}]^\top.$$

- $p=r$: pivots of $\boldsymbol{\Psi}_r^\top$ maximize $|\det(\mathbf{C}\boldsymbol{\Psi}_r)|$.
- $p>r$: since $\det(\boldsymbol{\Theta}^\top\boldsymbol{\Theta})=\prod_{i=1}^r\sigma_i(\boldsymbol{\Theta}\boldsymbol{\Theta}^\top)$ and $\boldsymbol{\Theta}\boldsymbol{\Theta}^\top$ is the selected principal block of $\boldsymbol{\Psi}_r\boldsymbol{\Psi}_r^\top$, pivoting that Gram matrix ranks points for the oversampled D-optimal objective.

QR pivoting needs one factorization total, is invariant to unitary changes of an orthonormal basis, and returns a hierarchical pivot list that can be truncated at any $p\ge r$. Choose $r$ at the informative part of the singular-value spectrum, for example with the Gavish-Donoho hard threshold; beyond that point sensors are spent on noise modes.

## Code

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


# usage
# Psi_r = pod_basis(X_train, r)
# gamma = select_sensors(Psi_r, p)
# x_hat = reconstruct(Psi_r, gamma, x_test[gamma])
```
