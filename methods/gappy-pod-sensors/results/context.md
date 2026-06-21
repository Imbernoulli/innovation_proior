# Context: optimized point-sensor placement for reconstructing a high-dimensional state

## Research question

A great many physical systems are described by a very high-dimensional state $\mathbf{x}\in\mathbb{R}^n$ — a velocity or vorticity field on a fine CFD grid, a pressure field over a surface, an image of $n$ pixels — yet their dynamics live on a low-dimensional attractor, so the state is highly compressible. We rarely want, or can afford, to measure all $n$ components. The question is: **given that we may place only $p\ll n$ point sensors, each reading a single component $x_{\gamma_i}$ of the state, which $p$ locations let us reconstruct the entire field most accurately?**

This setting arises whenever sensors are expensive or slow: ocean buoys, surface pressure taps on a vehicle, disease-surveillance stations, or low-latency feedback control where a state estimate must be produced from a handful of readings in real time. The choice of locations is combinatorial: there are $\binom{n}{p}=\tfrac{n!}{(n-p)!\,p!}$ ways to choose $p$ locations out of $n$.

## Background

**Low-rank structure and POD.** High-dimensional data from fluids, imaging, climate, and neuroscience typically exhibit a few dominant coherent structures. These are extracted by the proper orthogonal decomposition (POD) — equivalently the Karhunen–Loève expansion, principal component analysis (Pearson 1901), or empirical orthogonal functions. Given a snapshot matrix $\mathbf{X}=[\mathbf{x}_1\,\cdots\,\mathbf{x}_m]\in\mathbb{R}^{n\times m}$, the singular value decomposition $\mathbf{X}=\boldsymbol{\Psi}\boldsymbol{\Sigma}\mathbf{V}^\top$ yields orthonormal left singular vectors $\boldsymbol{\Psi}$ (the POD modes). Keeping the leading $r$ of them gives $\boldsymbol{\Psi}_r\in\mathbb{R}^{n\times r}$, and by the Eckart–Young theorem $\boldsymbol{\Psi}_r\boldsymbol{\Sigma}_r\mathbf{V}_r^\top$ is the best rank-$r$ least-squares approximation of $\mathbf{X}$. Any state then has the compact representation $\mathbf{x}\approx\boldsymbol{\Psi}_r\mathbf{a}$ with coordinates $\mathbf{a}=\boldsymbol{\Psi}_r^\top\mathbf{x}\in\mathbb{R}^r$, $r\ll n$. The truncation rank $r$ is set by thresholding the singular-value spectrum; an optimal hard threshold for additive Gaussian noise was given by Gavish & Donoho (2014).

**Conditioning controls reconstruction error.** If we observe only $p$ entries of $\mathbf{x}$, collected by a measurement matrix $\mathbf{C}\in\mathbb{R}^{p\times n}$ whose rows are canonical basis vectors $\mathbf{e}_{\gamma_i}^\top$, then $\mathbf{y}=\mathbf{C}\mathbf{x}\approx(\mathbf{C}\boldsymbol{\Psi}_r)\mathbf{a}=\boldsymbol{\Theta}\mathbf{a}$. Recovering $\mathbf{a}$ from $\mathbf{y}$ requires (pseudo)inverting $\boldsymbol{\Theta}=\mathbf{C}\boldsymbol{\Psi}_r$. The sensitivity of this inversion to noise is governed by the condition number $\kappa(\boldsymbol{\Theta})=\sigma_{\max}(\boldsymbol{\Theta})/\sigma_{\min}(\boldsymbol{\Theta})$: in the worst case the signal-to-noise ratio is degraded by exactly the factor $\kappa$. A small $\sigma_{\min}(\boldsymbol{\Theta})$ means some direction of $\mathbf{a}$ is nearly invisible to the sensors, and its estimate explodes. The entries of $\boldsymbol{\Theta}=\mathbf{C}\boldsymbol{\Psi}_r$ are determined entirely by *which rows of $\boldsymbol{\Psi}_r$* the sensors pick out.

**Optimal experimental design.** This is the classical statistical problem of selecting experiments (here, sensor rows) to estimate $r$ parameters $\mathbf{a}$ from noisy outputs $\mathbf{y}=\boldsymbol{\Theta}\mathbf{a}+\boldsymbol{\xi}$, $\boldsymbol{\xi}\sim\mathcal{N}(0,\eta^2\mathbf{I})$. The estimator's error covariance is $\mathrm{Var}(\mathbf{a}-\hat{\mathbf{a}})=\eta^2(\boldsymbol{\Theta}^\top\boldsymbol{\Theta})^{-1}$, so D-optimal design minimizes the determinant/volume of this covariance (equivalently maximizes $\det(\boldsymbol{\Theta}^\top\boldsymbol{\Theta})$), A-optimal design minimizes its trace/mean-squared error, and E-optimal design minimizes its largest eigenvalue/worst-case variance. All three are combinatorial in exact form.

**Greedy pivoting and the QR factorization.** Column-pivoted QR (Businger & Golub 1965) factors $\mathbf{A}\mathbf{C}^\top=\mathbf{Q}\mathbf{R}$ by repeatedly choosing the not-yet-selected column of largest residual Euclidean norm and then subtracting its contribution (via a Householder reflector) from all remaining columns. It imposes a diagonal-dominance structure $|r_{ii}|^2\ge\sum_{j=i}^k|r_{jk}|^2$ on the triangular factor. Pivoted QR is implemented and tuned in every standard package (LAPACK, NumPy, MATLAB), at $O(nr^2)$ for selecting $r$ pivots of an $n\times r$ matrix.

## Baselines

**Gappy POD (Everson & Sirovich 1995).** A method for reconstructing incomplete data in a POD/KL basis. For a vector $\mathbf{g}$ with missing entries one defines a *mask* $\mathbf{n}$ ($n_i=1$ where data is present, $0$ where missing) and a *gappy inner product* $(\mathbf{u},\mathbf{v})_\mathbf{n}=\sum_i n_i u_i v_i$ with induced norm $\|\cdot\|_\mathbf{n}$. The repaired vector is expanded as $\tilde{\mathbf{g}}=\sum_{k=1}^r b_k\boldsymbol{\phi}_k$, and the coefficients $\mathbf{b}$ are found by minimizing $E=\|\mathbf{g}-\tilde{\mathbf{g}}\|_\mathbf{n}^2$, which reduces to a small linear system $\mathbf{M}\mathbf{b}=\mathbf{f}$ with $M_{ij}=(\boldsymbol{\phi}_i,\boldsymbol{\phi}_j)_\mathbf{n}$ over the observed entries only. With point sensors the mask simply selects the measured components, so this is least-squares fitting of POD coordinates to a few measurements, $\hat{\mathbf{a}}=\boldsymbol{\Theta}^\dagger\mathbf{y}$. The method was demonstrated reconstructing full airfoil pressure fields from a handful of surface taps, with the kept entries chosen by random or ad-hoc sub-sampling.

**Empirical interpolation methods — EIM (Barrault et al. 2004) and DEIM (Chaturantabut & Sorensen 2010).** Greedy schemes from reduced-order modeling that pick interpolation points to cheaply evaluate nonlinear terms. DEIM, given orthonormal modes $\mathbf{u}_1,\dots,\mathbf{u}_m$, selects $p_1=\arg\max_i|u_1(i)|$, then for each subsequent mode solves $\mathbf{S}_{j-1}^\top\mathbf{U}_{j-1}\mathbf{z}=\mathbf{S}_{j-1}^\top\mathbf{u}_j$, forms the residual $\mathbf{r}_j=\mathbf{u}_j-\mathbf{U}_{j-1}\mathbf{z}$, and takes $p_j=\arg\max_i|r_j(i)|$. The resulting interpolant $\hat{\mathbf{f}}=\mathbf{U}(\mathbf{S}^\top\mathbf{U})^{-1}\mathbf{S}^\top\mathbf{f}$ obeys $\|\mathbf{f}-\hat{\mathbf{f}}\|_2\le\|(\mathbf{S}^\top\mathbf{U})^{-1}\|_2\,\|(\mathbf{I}-\mathbf{U}\mathbf{U}^\top)\mathbf{f}\|_2$, with the constant bounded by $\|(\mathbf{S}^\top\mathbf{U})^{-1}\|_2\le(1+\sqrt{2n})^{m-1}/\|\mathbf{u}_1\|_\infty$. It selects exactly $p=m$ points (square $\mathbf{S}^\top\mathbf{U}$), processing one basis mode per iteration with a residual recomputed each step.

**Q-DEIM (Drmač & Gugercin 2016).** Replaces DEIM's residual greedy with a single column-pivoted QR of $\mathbf{U}^\top$, $\mathbf{U}^\top\boldsymbol{\Pi}=\mathbf{Q}\mathbf{R}$, taking the first $m$ pivots as points. This yields the diagonal-dominance bound $|T_{ii}|^2\ge\sum_{j=i}^k|T_{jk}|^2$, and from $\mathbf{R}\mathbf{R}^\top=\mathbf{I}_m$ the guarantee $\min_i|T_{ii}|=|T_{mm}|\ge 1/\sqrt{n-m+1}$, giving the sharper bound $\|(\mathbf{S}^\top\mathbf{U})^{-1}\|_2\le\sqrt{n-m+1}\,\big(\sqrt{4^m+6m-1}/3\big)$; the selection is invariant under unitary changes of the orthonormal basis. It is framed for $p=m$ (square selection) and for interpolating nonlinear ROM terms.

**Convex relaxation (Joshi & Boyd 2009).** Relaxes the Boolean choice $\beta_i\in\{0,1\}$ to $\beta_i\in[0,1]$ and solves $\max_{\boldsymbol{\beta}}\log\det\sum_{i=1}^n\beta_i\boldsymbol{\theta}_i^\top\boldsymbol{\theta}_i$ subject to $\sum_i\beta_i=p$ — a convex surrogate for D-optimal selection that also certifies a bound on the achievable performance. Each Newton iteration factors an $n\times n$ matrix ($O(n^3)$ per iteration, $O(n^2)$ storage), the optimization is run for a given $p$, and the relaxed weights are rounded back to a hard subset.

**Random sensing / compressed sensing.** As an alternative paradigm, one uses random measurements in a *universal* basis (Fourier, wavelet) and recovers the signal by $\ell_1$ minimization, requiring $\mathcal{O}(K\log(n/K))$ samples for a $K$-sparse signal and incoherent (random) measurements. The measurements are drawn at random in a fixed universal basis rather than fit to training data, and recovery is by convex $\ell_1$ minimization.

## Evaluation settings

The natural yardsticks are high-dimensional fields with known low-rank structure, reconstructing held-out snapshots from a few sensors and reporting reconstruction error versus the number of sensors $p$ (and versus sensor-noise variance $\eta$), against random sensor placement and against full-state POD projection.

- **Flow past a cylinder.** Vorticity snapshots from a linearized Navier–Stokes simulation (immersed boundary projection method) at Reynolds number $100$, laminar periodic vortex shedding; $151$ timesteps, $\delta t=0.02$; state dimension $n\approx 90000$ gridpoints (downsampled to $n=3600$ candidate locations to bound storage); first $100$ snapshots for training, remaining for validation. The singular values decay rapidly and occur in pairs (shedding harmonics).
- **Extended Yale B faces.** $64$ aligned images each of $38$ individuals under varied lighting; each image a $32\times 32=1024$-pixel column; eigenfaces trained on $32$ randomly chosen images per subject, missing pixels reconstructed on a held-out image.
- **Sea surface temperature.** NOAA OISST V2 weekly global maps, $1990$–$2016$; features trained on the first $16$ years ($832$ snapshots), a held-out snapshot reconstructed.

Metric: reconstruction error (e.g. root-mean-square / percent error) of the recovered field, as a function of $p$ and of $\eta$, with POD modes and sensors trained on training snapshots and applied to validation snapshots. The number of modes $r$ is set by singular-value thresholding (Gavish–Donoho).

## Code framework

The pieces that already exist are an SVD/POD routine, a pivoted-QR routine, and least-squares solvers. The placement procedure and the reconstruction map are the empty slots.

```python
from scipy.linalg import qr, lstsq, solve
from sklearn.decomposition import TruncatedSVD

# --- already available: build a tailored low-rank basis from data ---
def pod_basis(X, r):
    """Leading r left singular vectors (POD modes) of snapshot matrix X (n_samples, n_features)."""
    svd = TruncatedSVD(n_components=r).fit(X)
    return svd.components_.T          # Psi_r, shape (n_features, r)

# --- already available: pivoted QR (Businger-Golub) returns column pivots ---
# scipy.linalg.qr(A, pivoting=True) -> Q, R, pivots

# --- SLOT 1: choose p sensor locations from the basis ----------------
def select_sensors(Psi_r, p):
    """Return p >= r ranked sensor indices (rows of Psi_r) for stable reconstruction."""
    # TODO
    pass

# --- SLOT 2: reconstruct the full state from sensor measurements ------
def reconstruct(Psi_r, sensors, y):
    """Estimate full state x_hat from measurements y taken at `sensors`."""
    # TODO: fit basis coordinates a to the measured entries, then lift x_hat = Psi_r @ a
    pass
```
