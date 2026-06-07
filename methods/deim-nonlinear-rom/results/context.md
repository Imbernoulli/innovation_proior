# Context: making the nonlinear term of a reduced-order model actually cheap

## Research question

A great many engineering simulations are large nonlinear systems of ordinary differential equations that come from discretizing a PDE in space:
$$\frac{d}{dt}\mathbf{y}(t)=\mathbf{A}\,\mathbf{y}(t)+\mathbf{F}(\mathbf{y}(t)),\qquad \mathbf{y}(t)\in\mathbb{R}^n,$$
where $\mathbf{A}\in\mathbb{R}^{n\times n}$ is the discrete linear operator and $\mathbf{F}$ is a nonlinear term — for a scalar reaction–diffusion problem $\mathbf{F}$ is evaluated componentwise, $\mathbf{F}(\mathbf{y})=[F(y_1),\dots,F(y_n)]^\top$. The state dimension $n$ is the number of grid points and must be very large (thousands to tens of thousands) for the numerical solution to be accurate, so each simulation is expensive, and parametric studies, control, and optimization — which need many simulations — become infeasible.

The trajectories of such systems are typically attracted to a low-dimensional manifold, so one hopes to replace the $n$-dimensional system by a surrogate of dimension $k\ll n$ that reproduces nearly the same input/output behavior. The precise question is: **can a projection-based reduced model of order $k\ll n$ be made genuinely cheap to evaluate — cheap *per time step*, with a cost independent of the full dimension $n$ — even when $\mathbf{F}$ is a general nonlinearity?** A usable answer must (i) keep the well-understood optimality of the existing reduced-basis machinery, (ii) drive the per-step cost down to depend only on $k$ (and a second small dimension), not on $n$, and (iii) come with a guarantee that the cheapened evaluation is almost as accurate as the expensive one it replaces.

## Background

**Low-rank structure and proper orthogonal decomposition (POD).** High-dimensional solution data from a fixed set of boundary conditions and inputs usually lives near a low-dimensional subspace. POD extracts that subspace. Given a set of solution snapshots $\{\mathbf{y}_1,\dots,\mathbf{y}_{n_s}\}\subset\mathbb{R}^n$ collected along trajectories, stack them as $\mathbf{Y}=[\mathbf{y}_1\,\cdots\,\mathbf{y}_{n_s}]\in\mathbb{R}^{n\times n_s}$. A POD basis of dimension $k$ is the orthonormal set $\{\boldsymbol{\phi}_i\}_{i=1}^k$ whose span best fits the snapshots,
$$\min_{\{\boldsymbol{\phi}_i\}}\ \sum_{j=1}^{n_s}\Big\|\mathbf{y}_j-\sum_{i=1}^k(\mathbf{y}_j^\top\boldsymbol{\phi}_i)\boldsymbol{\phi}_i\Big\|_2^2,\qquad \boldsymbol{\phi}_i^\top\boldsymbol{\phi}_j=\delta_{ij}.$$
Its solution is the leading $k$ left singular vectors of $\mathbf{Y}=\mathbf{V}\boldsymbol{\Sigma}\mathbf{W}^\top$, and the residual of the fit is exactly the tail of the squared singular values $\sum_{i=k+1}^r\sigma_i^2$. POD is thus the optimal (Eckart–Young) rank-$k$ basis for the data — equivalently the Karhunen–Loève expansion / principal component analysis. The rapid decay of $\sigma_i$ for a system on a low-dimensional manifold is what makes $k\ll n$ possible. POD has supplied reduced models in compressible flow, fluid dynamics, aerodynamics, and optimal control.

**Galerkin projection.** Given an orthonormal basis $\mathbf{V}_k\in\mathbb{R}^{n\times k}$, one writes $\mathbf{y}\approx\mathbf{V}_k\tilde{\mathbf{y}}$ with $\tilde{\mathbf{y}}\in\mathbb{R}^k$ and projects the dynamics onto the basis (require the residual orthogonal to $\mathrm{Range}(\mathbf{V}_k)$, i.e. left-multiply by $\mathbf{V}_k^\top$):
$$\frac{d}{dt}\tilde{\mathbf{y}}=\underbrace{\mathbf{V}_k^\top\mathbf{A}\mathbf{V}_k}_{\tilde{\mathbf{A}}\in\mathbb{R}^{k\times k}}\tilde{\mathbf{y}}+\mathbf{V}_k^\top\mathbf{F}(\mathbf{V}_k\tilde{\mathbf{y}}).$$
The constant matrix $\tilde{\mathbf{A}}=\mathbf{V}_k^\top\mathbf{A}\mathbf{V}_k$ is formed once, offline, and is genuinely $k\times k$. For steady parametrized problems $\mathbf{A}\mathbf{y}(\mu)+\mathbf{F}(\mathbf{y}(\mu))=0$ the same projection gives a $k$-dimensional residual solved by Newton's method, whose Jacobian carries the term $\mathbf{V}_k^\top\mathbf{J}_F(\mathbf{V}_k\tilde{\mathbf{y}})\mathbf{V}_k$ with $\mathbf{J}_F=\mathrm{diag}\{F'(y_1),\dots,F'(y_n)\}$.

**The diagnosed complexity bottleneck.** While $\tilde{\mathbf{A}}$ is small, the reduced nonlinear term
$$\tilde{\mathbf{N}}(\tilde{\mathbf{y}})=\underbrace{\mathbf{V}_k^\top}_{k\times n}\,\underbrace{\mathbf{F}(\mathbf{V}_k\tilde{\mathbf{y}})}_{n\times 1}$$
is not. Evaluating it each step requires lifting $\mathbf{V}_k\tilde{\mathbf{y}}$ back to $\mathbb{R}^n$, evaluating $\mathbf{F}$ at all $n$ components, and projecting the result down — a cost of order $\mathcal{O}(\alpha(n)+4nk)$, where $\alpha(q)$ is the cost of evaluating $\mathbf{F}$ at $q$ components. This still scales with $n$. The linear part of the model reduced once and for all, but the nonlinear part must be re-touched at every one of the $n$ original grid points on every step, so the reduced simulation can take essentially as long as the full one. For the steady Newton case the reduced Jacobian $\mathbf{V}_k^\top\mathbf{J}_F(\mathbf{V}_k\tilde{\mathbf{y}})\mathbf{V}_k$ is order $\mathcal{O}(\alpha(n)+4nk+2nk^2)$ — also $n$-dependent. Indeed, with a sparse $\mathbf{A}$ the full-order step is only $\mathcal{O}(n)$, so the reduced model's $\mathcal{O}(k^2+nk)$ can *exceed* the full model once $k$ is moderate: dimension was reduced but cost was not. Removing this $n$-dependence in the nonlinear term is the whole obstacle.

**Replacing projection by interpolation for non-affine terms.** The same obstacle appears for reduced-basis methods on parametrized PDEs when the operator depends *non-affinely* on the parameter: the reduced system loses its parameter-separable (offline/online) structure because the nonlinear/non-affine function must be assembled over the full mesh. The empirical interpolation method (Barrault, Maday, Nguyen, Patera 2004) addresses this by approximating a non-affine parametrized function $g(\cdot;\mu)$ not by orthogonal projection onto an empirical basis but by *interpolation* of that basis at a small set of greedily selected spatial points (the "magic points"). It builds, one step at a time, a nested set of basis functions $q_1,\dots,q_M$ and points $t_1,\dots,t_M$ such that the interpolant matches $g$ at the $t_i$; the interpolation matrix is made lower triangular by the nested construction, and the next point is chosen where the current interpolant's residual is largest. The payoff is that only $M$ evaluations of $g$ — at the magic points — are needed to fit the $M$ coefficients, instead of an evaluation over the whole domain. This was posed in a continuous, function-space setting on a bounded domain $\Omega$.

**Reconstruction from incomplete data.** Everson and Sirovich (1995) ("gappy POD") reconstruct a vector $\mathbf{g}$ with missing entries in a POD/KL basis: define a mask $\mathbf{n}$ ($n_i=1$ where data is present, $0$ where missing), a gappy inner product $(\mathbf{u},\mathbf{v})_\mathbf{n}=\sum_i n_i u_iv_i$, expand $\tilde{\mathbf{g}}=\sum_{k}b_k\boldsymbol{\phi}_k$, and choose the coefficients by least squares over the *observed* entries, $\mathbf{M}\mathbf{b}=\mathbf{f}$, $M_{ij}=(\boldsymbol{\phi}_i,\boldsymbol{\phi}_j)_\mathbf{n}$. It is the canonical way to recover a full field from few samples, and the structural cousin of fitting basis coefficients from a handful of components.

## Baselines

**POD-Galerkin reduced-order model.** Core idea: $\mathbf{y}\approx\mathbf{V}_k\tilde{\mathbf{y}}$, project to get $\dot{\tilde{\mathbf{y}}}=\tilde{\mathbf{A}}\tilde{\mathbf{y}}+\mathbf{V}_k^\top\mathbf{F}(\mathbf{V}_k\tilde{\mathbf{y}})$, with $\mathbf{V}_k$ the leading POD modes of solution snapshots. It reduces the *number of variables* optimally and handles linear/bilinear terms perfectly (their reduced operators are precomputed $k\times k$/$k\times k\times k$ tensors). **Gap:** for a general nonlinearity the term $\mathbf{V}_k^\top\mathbf{F}(\mathbf{V}_k\tilde{\mathbf{y}})$ costs $\mathcal{O}(\alpha(n)+4nk)$ per step and the Newton Jacobian $\mathcal{O}(\alpha(n)+4nk+2nk^2)$ — both still depend on $n$, so per-step cost is not reduced and can exceed the (sparse) full model.

**Empirical interpolation method (EIM, Barrault–Maday–Nguyen–Patera 2004).** Core idea: approximate a non-affine parametrized function by greedily building basis functions $q_1,\dots,q_M$ and interpolation points $t_1,\dots,t_M$, matching the function at the $t_i$ via a (lower-triangular) interpolation system, the next point chosen at the largest current residual. Only $M$ point-evaluations of the function are needed, restoring offline/online separation. **Gap:** it is posed in a continuous function space with its own greedy basis construction; it is not expressed as a clean discrete matrix factorization on $\mathbb{R}^n$, and it is not connected to the optimal POD basis of the discrete nonlinear snapshots nor accompanied by a sharp finite-dimensional error bound in terms of the orthogonal-projection error.

**Gappy POD (Everson & Sirovich 1995).** Core idea: fit POD coefficients to the observed entries of a partially known vector by least squares over a mask, $\hat{\mathbf{b}}=(\boldsymbol{\Theta}^\top\boldsymbol{\Theta})^{-1}\boldsymbol{\Theta}^\top\mathbf{y}$ with $\boldsymbol{\Theta}$ the observed rows of the basis. It shows a full field can be recovered from few samples in a tailored basis. **Gap:** the choice of which entries to keep (the mask) is left to random or ad-hoc sub-sampling, there is no principled selection of the sample locations and no guarantee on the conditioning of $\mathbf{M}=\boldsymbol{\Theta}^\top\boldsymbol{\Theta}$; and it targets reconstructing a sampled field, not cheapening a nonlinear term inside reduced dynamics.

**Orthogonal projection of the nonlinear term onto a POD basis.** Core idea: build a second POD basis $\mathbf{U}\in\mathbb{R}^{n\times m}$ for the *nonlinear* snapshots $\{\mathbf{F}(\mathbf{y}_j)\}$ and replace $\mathbf{F}$ by its best $m$-term approximation $\mathbf{U}\mathbf{U}^\top\mathbf{F}$ in that subspace. This is the most accurate $m$-dimensional approximation of $\mathbf{F}$ from $\mathrm{Range}(\mathbf{U})$. **Gap:** computing the coefficients $\mathbf{U}^\top\mathbf{F}(\mathbf{V}_k\tilde{\mathbf{y}})$ still requires *all $n$* entries of $\mathbf{F}$, so the $\mathcal{O}(\alpha(n))$ evaluation cost is not removed at all — it merely moves the bottleneck.

## Evaluation settings

The natural yardsticks are discretized nonlinear PDEs whose reduced models would be benchmarked on accuracy of the reduced trajectory and on per-step (or per-Newton-iteration) cost versus the full model and versus plain POD-Galerkin, as the reduced dimensions vary.

- **Unsteady 1-D nonlinear reaction–diffusion (neuron modeling, FitzHugh–Nagumo).** Coupled PDEs $\varepsilon v_t=\varepsilon^2 v_{xx}+f(v)-w+c$, $w_t=bv-\gamma w+c$, cubic nonlinearity $f(v)=v(v-0.1)(1-v)$, on $x\in[0,L]$ with $L=1$, $\varepsilon=0.015$, $b=0.5$, $\gamma=2$, $c=0.05$, stimulus $i_0(t)=50000\,t^3e^{-15t}$ entering through the Neumann boundary $v_x(0,t)=-i_0(t),\,v_x(L,t)=0$, zero initial data. Finite-difference discretization gives a system of the form $\dot{\mathbf{y}}=\mathbf{A}\mathbf{y}+\mathbf{F}(\mathbf{y})$ of full dimension $n=1024$; snapshots are taken at equally spaced times over $t\in[0,8]$; the solution exhibits a limit cycle in the $v$–$w$ phase plane. Metrics: relative error of the reduced trajectory and scaled CPU time per forward-Euler step, as functions of the POD dimension $k$ and the nonlinear-approximation dimension $m$.
- **Steady parametrized 2-D nonlinear problem.** A highly nonlinear steady-state equation on a 2-D grid (full dimension $n$ in the thousands) solved by Newton's method, with the reduced Jacobian assembled each iteration. Metrics: relative error and scaled CPU time per Newton iteration versus the reduced dimensions.
- **Nonlinear-function approximation tests.** Parametrized scalar functions sampled on 1-D and 2-D grids (e.g. $s(x;\mu)=(1-x)\cos(3\pi\mu(x+1))e^{-(1+x)\mu}$ on $n=100$ points; a 2-D function with a near-singularity on $n=400$ points), with snapshots at a training set of parameters and accuracy measured at a *different, larger* test set. Metrics: average approximation error of the interpolated function versus the orthogonal-projection error and versus the analytic error bound, as the number of points $m$ grows.

Throughout, the singular-value spectrum of the snapshot matrices sets the achievable reduced dimensions, and the number $m$ of interpolation points is paired with the nonlinear-basis dimension.

## Code framework

The pieces that already exist are a truncated SVD for POD, a small dense linear solver, Galerkin projection for the linear operator, and a time-stepping loop. The open slots are the row-selection rule for interpolation, the small constant operators assembled from those rows, and the reduced right-hand side that evaluates the nonlinear function only through the selected entries.

```python
import numpy as np

def pod_basis(S, r):
    """Leading r left singular vectors of snapshot matrix S (n x n_s)."""
    U, sv, _ = np.linalg.svd(S, full_matrices=False)
    return U[:, :r], sv


def galerkin(A, Vk):
    return Vk.T @ A @ Vk                          # k x k, precomputed once


def select_interpolation_dofs(U):
    """Given a collateral basis U (n x m), return m row indices for interpolation."""
    # TODO: choose rows so the interpolation matrix is well-conditioned
    pass


def assemble_reduced_operators(A, Vk, U, idx):
    """Build the constant operators for a reduced nonlinear model."""
    At = galerkin(A, Vk)
    # TODO: build the coupling from U and the selected rows, plus sampled rows of Vk
    pass


def reduced_rhs(y_red, t, At, coupling, sample_basis, F_sample):
    """Evaluate the reduced right-hand side using sampled nonlinear values."""
    # TODO: sample the reduced state, evaluate the nonlinear function there, and combine
    pass


def integrate(rhs, y0_red, dt, nt):
    y = y0_red.copy()
    for step in range(nt):
        y = y + dt * rhs(y, step * dt)
    return y
```
