# Spectral clustering and the discrete Cheeger inequality

## Problem

Partition the vertices of a graph $G=(V,E,w)$ into two well-separated clusters by minimizing the
**conductance** of a cut $S\subseteq V$,

$$\phi(S)=\frac{|\partial(S)|}{\min(d(S),\,d(V\setminus S))},\qquad
\phi_G=\min_{\varnothing\ne S\subsetneq V}\phi(S),$$

where $\partial(S)$ are the edges leaving $S$ and $d(S)=\sum_{u\in S}d(u)$ is its volume. Finding the
minimum-conductance / sparsest cut is NP-hard. We want a polynomial-time proxy with a *provable*
relationship to $\phi_G$.

## Key idea

The cut size of an indicator is a Laplacian quadratic form, $\chi_S^{T}L\chi_S=|\partial(S)|$ with
$L=D-A$ and $x^{T}Lx=\sum_{(u,v)\in E}w_{u,v}(x(u)-x(v))^2$. Relaxing the $0/1$ indicator to a real
vector orthogonal to the kernel $\mathbf 1$ (resp. to $d$, for the volume-normalized version) turns
conductance into a Rayleigh quotient whose minimum is the second-smallest eigenvalue of the
(normalized) Laplacian. That eigenvalue therefore brackets the sparsest cut from both sides, and
thresholding its eigenvector (the **Fiedler vector**) yields a cut achieving the upper side.

## The Cheeger inequality (discrete)

Let $N=D^{-1/2}LD^{-1/2}$ be the normalized Laplacian, $0=\nu_1\le\nu_2\le\cdots\le\nu_n$ its
eigenvalues. Then

$$\boxed{\ \frac{\nu_2}{2}\ \le\ \phi_G\ \le\ \sqrt{2\,\nu_2}\ }$$

where $\nu_2=\min_{y\perp d,\ y\ne0}\dfrac{y^{T}Ly}{y^{T}Dy}$, and a cut meeting the upper bound is
found by sweeping thresholds of the Fiedler vector.

**Easy direction $\nu_2/2\le\phi_G$.** For the optimal $S$, set $y=\chi_S-\sigma\mathbf 1$,
$\sigma=d(S)/d(V)$. Then $y\perp d$, $y^{T}Ly=|\partial(S)|$, and
$y^{T}Dy=d(S)d(V\setminus S)/d(V)$, so $\nu_2\le |\partial(S)|d(V)/(d(S)d(V\setminus S))\le 2\phi(S)$,
since $\max(d(S),d(V\setminus S))\ge d(V)/2$.

**Hard direction $\phi_G\le\sqrt{2\nu_2}$ (sweep rounding; Trevisan-style proof).** Let $y\perp d$
have Rayleigh quotient $\rho=y^{T}Ly/y^{T}Dy$ (take $y$ the eigenvector, $\rho=\nu_2$). Sort
$y(1)\le\cdots\le y(n)$, let $j$ be the half-volume median ($\sum_{u\le j}d(u)\ge d(V)/2$ least),
and center $z=y-y(j)\mathbf 1$. The numerator is unchanged, and since $y\perp d$ minimizes
$(y+t\mathbf 1)^TD(y+t\mathbf 1)$ over shifts, $z^{T}Dz\ge y^{T}Dy$, hence
$z^{T}Lz/z^{T}Dz\le\rho$. Scale $z(1)^2+z(n)^2=1$.
Consider sweep cuts $S_t=\{u:z(u)\le t\}$ with $t$ drawn at density $2|t|$ on $[z(1),z(n)]$
($\int 2|t|=z(1)^2+z(n)^2=1$). Then:

- $\displaystyle \mathbb E[|\partial(S_t)|]=\sum_{(u,v)\in E}w_{u,v}\Pr[(u,v)\in\partial(S_t)]
  \le\sum_{(u,v)\in E}w_{u,v}|z(u)-z(v)|(|z(u)|+|z(v)|)$, because in both the same-sign case
  $|z(u)^2-z(v)^2|$ and the opposite-sign case $z(u)^2+z(v)^2$ are $\le|z(u)-z(v)|(|z(u)|+|z(v)|)$.
- $\displaystyle \mathbb E[\min(d(S_t),d(V\setminus S_t))]=\sum_u z(u)^2 d(u)=z^{T}Dz$: the median
  choice gives $d(\{z<0\})<d(V)/2$ and $d(\{z\le0\})\ge d(V)/2$, so negative thresholds use $S_t$
  as the smaller side and nonnegative thresholds use $V\setminus S_t$.
- Cauchy–Schwarz: $\sum_{(u,v)}w_{u,v}|z(u)-z(v)|(|z(u)|+|z(v)|)\le\sqrt{z^{T}Lz}\,\sqrt{\sum_{(u,v)}w_{u,v}(|z(u)|+|z(v)|)^2}
  \le\sqrt{\rho\,z^{T}Dz}\,\sqrt{2\,z^{T}Dz}=\sqrt{2\rho}\,z^{T}Dz$,
  using $z^{T}Lz\le\rho z^{T}Dz$ and
  $\sum_{(u,v)}w_{u,v}(|z(u)|+|z(v)|)^2\le 2\sum_u z(u)^2d(u)=2z^{T}Dz$.

So $\mathbb E[|\partial(S_t)|]\le\sqrt{2\rho}\,\mathbb E[\min(d(S_t),d(V\setminus S_t))]$, whence some
threshold $t$ gives $\phi(S_t)\le\sqrt{2\rho}=\sqrt{2\nu_2}$. $\qquad\blacksquare$

The rounding used only $y\perp d$, not that $y$ is an eigenvector. The $\sqrt{\cdot}$ gap is tight:
a cycle $C_n$ has $\nu_2=1-\cos(2\pi/n)=\Theta(1/n^2)$ but $\phi_G=\Theta(1/n)$, so
$\phi_G\asymp\sqrt{\nu_2}$.

## Algorithm

```python
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

def build_laplacian(adj):
    d = np.asarray(adj.sum(axis=1)).ravel()
    return (sp.diags(d) - adj).tocsr(), d

def conductance(adj, d, S_mask):
    volS = d[S_mask].sum(); volC = d.sum() - volS
    if min(volS, volC) == 0:
        return np.inf
    return adj[S_mask][:, ~S_mask].sum() / min(volS, volC)

def fiedler_vector(adj):
    # nu_2 of N = D^{-1/2} L D^{-1/2}; map eigenvector back by D^{-1/2} to get y orthogonal to d.
    L, d = build_laplacian(adj)
    if np.any(d <= 0):
        raise ValueError("normalized Laplacian requires positive degrees")
    dinv_half = sp.diags(1.0 / np.sqrt(d))
    N = (dinv_half @ L @ dinv_half).tocsr()
    vals, vecs = eigsh(N.astype(float), k=2, which="SM")
    order = np.argsort(vals)
    y = np.asarray(dinv_half @ vecs[:, order[1]]).ravel()
    return y, max(float(vals[order[1]]), 0.0)

def relax_and_round(adj):
    # Sweep all n-1 threshold cuts of the sorted Fiedler vector; keep the best.
    _, d = build_laplacian(adj)
    y, nu2 = fiedler_vector(adj)
    order = np.argsort(y); n = len(y)
    in_S = np.zeros(n, dtype=bool)
    best_S, best_phi = None, np.inf
    for k in range(n - 1):
        in_S[order[k]] = True
        phi = conductance(adj, d, in_S)
        if phi < best_phi:
            best_phi, best_S = phi, in_S.copy()
    # Cheeger bracket in exact arithmetic: nu2/2 <= phi_G <= best_phi <= sqrt(2*nu2).
    return best_S, best_phi, nu2
```

This is **spectral clustering / spectral bisection** by the Fiedler vector: a continuous
eigenvector relaxation of an NP-hard combinatorial cut, rounded by a sweep, provably within a
square-root factor of optimal via the discrete Cheeger inequality.
