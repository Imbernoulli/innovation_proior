# Data-driven sparse sensor placement via QR pivoting on a POD basis (gappy-POD reconstruction)

## Problem

Place $p\ll n$ point sensors (each reading one component $x_{\gamma_i}$ of a high-dimensional state $\mathbf{x}\in\mathbb{R}^n$) so the full field can be reconstructed from the $p$ readings. Exact optimal placement is a $\binom{n}{p}$ combinatorial search; we want a scalable, noise-robust, near-optimal surrogate built from standard linear algebra.

## Key idea

1. **Tailored low-rank basis.** From training snapshots $\mathbf{X}\in\mathbb{R}^{n\times m}$ take the SVD and keep the leading $r$ left singular vectors (POD modes) $\boldsymbol{\Psi}_r\in\mathbb{R}^{n\times r}$. By Eckart–Young this is the optimal rank-$r$ least-squares basis, so $\mathbf{x}\approx\boldsymbol{\Psi}_r\mathbf{a}$ with $\mathbf{a}\in\mathbb{R}^r$, $r\ll n$.
2. **Reconstruct by gappy POD.** Point sensors $\mathbf{C}=[\mathbf{e}_{\gamma_1}\cdots\mathbf{e}_{\gamma_p}]^\top$ give $\mathbf{y}=\mathbf{C}\mathbf{x}\approx\boldsymbol{\Theta}\mathbf{a}$ with $\boldsymbol{\Theta}=\mathbf{C}\boldsymbol{\Psi}_r=\boldsymbol{\Psi}_r(\gamma,:)$. Estimate $\hat{\mathbf{a}}=\boldsymbol{\Theta}^\dagger\mathbf{y}$ (= $\boldsymbol{\Theta}^{-1}\mathbf{y}$ when $p=r$) and lift $\hat{\mathbf{x}}=\boldsymbol{\Psi}_r\hat{\mathbf{a}}$.
3. **Place sensors to condition $\boldsymbol{\Theta}$.** Under noise $\mathbf{y}=\boldsymbol{\Theta}\mathbf{a}+\boldsymbol{\xi}$, $\boldsymbol{\xi}\sim\mathcal{N}(0,\eta^2\mathbf{I})$, the error covariance is $\eta^2(\boldsymbol{\Theta}^\top\boldsymbol{\Theta})^{-1}$ and worst-case error amplification is $\kappa(\boldsymbol{\Theta})$. Minimizing the error-ellipsoid volume is the **D-optimal** criterion $\max_\gamma\det(\boldsymbol{\Theta}^\top\boldsymbol{\Theta})$.
4. **Greedy = column-pivoted QR.** Pivoted QR picks the next column of largest residual norm and deflates the rest, greedily maximizing the diagonal product $\prod_i|r_{ii}|=\prod_i\sigma_i=|\det|$ — exactly greedy volume (D-optimal) maximization, with diagonal dominance $|r_{ii}|^2\ge\sum_{j=i}^k|r_{jk}|^2$.

## Algorithm

Given $\boldsymbol{\Psi}_r$ and a sensor count $p$:
$$\gamma=\begin{cases}\mathrm{qrPivot}(\boldsymbol{\Psi}_r^\top,\ r), & p=r\quad(\text{cost }O(nr^2)),\\[2pt]\mathrm{qrPivot}(\boldsymbol{\Psi}_r\boldsymbol{\Psi}_r^\top,\ p), & p>r\quad(\text{cost }O(n^3)),\end{cases}\qquad \mathbf{C}=[\mathbf{e}_{\gamma_1}\cdots\mathbf{e}_{\gamma_p}]^\top.$$

- $p=r$: pivots of $\boldsymbol{\Psi}_r^\top$ maximize $|\det(\mathbf{C}\boldsymbol{\Psi}_r)|$ (this is Q-DEIM).
- $p>r$ (**oversampling**): since $\det(\boldsymbol{\Theta}^\top\boldsymbol{\Theta})=\prod_{i=1}^r\sigma_i(\boldsymbol{\Theta}\boldsymbol{\Theta}^\top)$ and $\boldsymbol{\Theta}\boldsymbol{\Theta}^\top$ is the row-block of the Gram matrix $\boldsymbol{\Psi}_r\boldsymbol{\Psi}_r^\top$, pivoting that Gram matrix selects the $p$ points whose row-submatrix has maximal volume, increasing $\det(\boldsymbol{\Theta}^\top\boldsymbol{\Theta})$.

QR pivoting needs **one** factorization total (vs one per iteration for convex/SDP design), is invariant to unitary basis changes, and returns a **hierarchically ranked** list of all pivots, so the first $p$ are optimized for any $p\ge r$. Choose $r$ at the knee of the singular-value spectrum (e.g. Gavish–Donoho threshold); beyond it, sensors chase noise modes and reconstruction overfits.

## Code

Faithful to the `pysensors` implementation (`dynamicslab/pysensors`): an `SVD` basis, a pivoted-`QR` selector, and an `SSPOR` reconstructor.

```python
import numpy as np
from scipy.linalg import qr, lstsq, solve
from sklearn.decomposition import TruncatedSVD


class SVDBasis:
    """POD/SVD basis: columns of basis_matrix_ are the leading n_basis_modes POD modes."""
    def __init__(self, n_basis_modes=10):
        self.n_basis_modes = n_basis_modes

    def fit(self, X):                                    # X: (n_samples, n_features), mean-centered
        svd = TruncatedSVD(n_components=self.n_basis_modes).fit(X)
        self.basis_matrix_ = svd.components_.T           # Psi_r: (n_features, n_basis_modes)
        return self


class QR:
    """Greedy QR optimizer: ranks sensor locations by pivoted QR of the basis."""
    def __init__(self):
        self.pivots_ = None

    def fit(self, basis_matrix, **kw):
        # pivots of Psi_r^T (p = r case). scipy's column pivoting = Businger-Golub greedy volume.
        _, _, self.pivots_ = qr(basis_matrix.conj().T, pivoting=True, **kw)
        return self

    def get_sensors(self):
        return self.pivots_


class CCQR(QR):
    """Cost-constrained QR (Householder, explicit pivots) -- biases pivot by -cost[i],
       and is the template for oversampled QR (pivot Psi_r @ Psi_r^T when p > r)."""
    def __init__(self, sensor_costs=None):
        super().__init__()
        self.sensor_costs = sensor_costs

    def fit(self, basis_matrix):
        n, m = basis_matrix.shape
        costs = np.zeros(n) if self.sensor_costs is None else self.sensor_costs
        R = basis_matrix.conj().T.copy()
        p = np.arange(n)
        for j in range(min(m, n)):
            u, i_piv = _qr_reflector(R[j:, j:], costs[p[j:]])
            i_piv += j
            p[[j, i_piv]] = p[[i_piv, j]]
            R[:, [j, i_piv]] = R[:, [i_piv, j]]
            R[j:, j:] -= np.outer(u, u @ R[j:, j:])      # Householder deflation
            R[j + 1:, j] = 0
        self.pivots_ = p
        return self


def _qr_reflector(r, costs):
    dlens = np.sqrt(np.sum(np.abs(r) ** 2, axis=0))      # residual column norms
    i_piv = np.argmax(dlens - costs)                     # largest-norm (cost-biased) pivot
    dlen = dlens[i_piv]
    if dlen > 0:
        u = r[:, i_piv] / dlen
        u[0] += np.sign(u[0]) + (u[0] == 0)
        u /= np.sqrt(abs(u[0]))
    else:
        u = r[:, i_piv]; u[0] = np.sqrt(2)
    return u, i_piv


class SSPOR:
    """Sparse Sensor Placement Optimization for Reconstruction."""
    def __init__(self, basis=None, optimizer=None, n_sensors=None):
        self.basis = basis or SVDBasis()
        self.optimizer = optimizer or QR()
        self.n_sensors = n_sensors

    def fit(self, X):
        self.basis.fit(X)
        self.basis_matrix_ = self.basis.basis_matrix_                 # Psi_r
        if self.n_sensors is None:
            self.n_sensors = self.basis_matrix_.shape[0]
        self.ranked_sensors_ = self.optimizer.fit(self.basis_matrix_).get_sensors()
        return self

    @property
    def selected_sensors(self):
        return self.ranked_sensors_[: self.n_sensors]

    def predict(self, y):
        """Reconstruct full state from measurements y taken at the selected sensors.
           Gappy-POD least squares: a_hat = Theta^dagger y, x_hat = Psi_r a_hat."""
        sensors = self.selected_sensors
        Theta = self.basis_matrix_[sensors, :]                       # C Psi_r
        if len(sensors) == self.basis_matrix_.shape[1]:              # square (p == r)
            a_hat = solve(Theta, y)
        else:                                                        # rectangular (p > r)
            a_hat = lstsq(Theta, y)[0]
        return self.basis_matrix_ @ a_hat                            # Psi_r a_hat


# usage
# model = SSPOR(basis=SVDBasis(n_basis_modes=r), n_sensors=p).fit(X_train)
# gamma = model.selected_sensors                # ranked sensor locations
# x_hat = model.predict(x_test[gamma])          # reconstruct the field from p readings
```

For oversampling $p>r$ the selector pivots the Gram matrix $\boldsymbol{\Psi}_r\boldsymbol{\Psi}_r^\top$ instead of $\boldsymbol{\Psi}_r^\top$ (in MATLAB: `[Q,R,pivot] = qr(Psi_r*Psi_r','vector')`), yielding the first $p$ ranked pivots optimized for $\det(\boldsymbol{\Theta}^\top\boldsymbol{\Theta})$.
