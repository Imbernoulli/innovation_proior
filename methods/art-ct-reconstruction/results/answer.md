# Algebraic Reconstruction Technique (ART) and SART for CT

## Problem

A CT detector measures only *line integrals* of an object's density: for ray `i`, the log-attenuation `p_i = ∫ f dℓ`. Reconstructing the cross-section `f(x,y)` from many such rays is inverting the Radon transform. The analytic inverse (filtered backprojection) is fast but assumes dense, uniformly sampled, straight-line projections over a full angular range with low noise, and cannot incorporate priors — failing in the few-view, limited-tilt, noisy regimes of electron microscopy and low-dose X-ray, and for diffracting sources whose rays bend. Discretizing onto an `N`-cell grid gives a sparse linear system `A f = p`, with `a_ij` = intersection length of ray `i` with cell `j` and `p_i` the measured ray-sum. `A` is huge (`N` up to ~10⁵), sparse, and typically inconsistent/under- or over-determined, so direct inversion or least squares is infeasible. ART solves it by iterative row-action projection.

## Key idea

Each equation `a_i · f = p_i` is a **hyperplane** in `R^N`. The orthogonal projection of a point `f` onto it is the nearest point on it, obtained by moving along the normal `a_i`:

  `f ← f + (p_i − a_i·f)/‖a_i‖² · a_i`   (minimize `‖f−z‖²` s.t. `a_i·z = p_i`; Lagrange gives `t = (p_i − a_i·f)/‖a_i‖²`).

Cycling this projection over all rays is **Kaczmarz's method / ART**. The `‖a_i‖²` normalization is exactly what makes each step *land on* the plane (after it, `a_i·f = p_i`). For a consistent system the cyclic projections converge to the intersection; for an underdetermined consistent system started at `f₀` they converge to the solution closest to `f₀` (minimum-norm when `f₀ = 0`); for a noisy overdetermined system they settle into a limit cycle near the near-intersection.

Because the discrete system is mildly inconsistent (discretization + noise), full projections (`λ = 1`) make rays fight — each lays a stripe, the next overwrites it — producing **salt-and-pepper noise**. A **relaxation** factor `λ < 1`,

  `f ← f + λ (p_i − a_i·f)/‖a_i‖² · a_i`,

softens each step so contradictions average instead of fight (smoother, more sweeps). Convergence rate depends on the angle between successive hyperplanes (orthogonal → one sweep; near-parallel → slow), so rays are visited in a **wide-angle interleaved order**. Nonnegativity and finite support are folded in by clipping after each projection (each is itself a projection onto a convex set).

**SART** removes the noise at its source — the within-view ambiguity from sequential updates — by applying all of a projection's corrections *simultaneously* against the same frozen image, with two normalizations:

  `g_j ← g_j + λ · [ Σ_{i∈V} a_ij (p_i − q_i)/(Σ_k a_ik) ] / ( Σ_{i∈V} a_ij ),   q_i = Σ_k a_ik g_k`

— residual normalized by **row sum** (ray length `L_i = Σ_k a_ik`), back-projected with weight `a_ij`, then normalized by **column sum** (per-pixel coverage `Σ_{i∈V} a_ij`) so a pixel's correction is independent of how many rays graze it; if that column sum is zero, the view did not touch that pixel and the update leaves it unchanged. SART also uses bilinear (pyramid) basis elements for a continuous, more accurate forward model — with first/last ray weights adjusted so `Σ_j a_ij = L_i` — and a longitudinal Hamming window emphasizing mid-ray over entry/exit corrections. The compact implementation below keeps the same ART projection and SART row/column weighting with a sparse length matrix; the bilinear/windowed version changes how the weights are built and back-distributed.

## Algorithm

ART (Kaczmarz), one sweep:
1. `f ← 0`; choose ray visiting order (wide-angle interleaved).
2. For each ray `i`: residual `r = p_i − a_i·f`; update `f ← f + λ r/‖a_i‖² · a_i`; clip `f ≥ 0` (and to support). Repeat sweeps.

SART, one iteration:
1. For each view `V` (against the frozen image `g`): for rays `i∈V`, `r_i = p_i − Σ_k a_ik g_k`; for covered pixels, correction `c_j = Σ_{i∈V} a_ij (r_i/Σ_k a_ik) / Σ_{i∈V} a_ij`, while uncovered pixels get `c_j = 0`; update `g ← g + λ c`; clip. Repeat over views, then iterate.

## Code

```python
import numpy as np
from scipy.sparse import csr_matrix


def build_system_matrix(n, angles, n_rays=None, det_spacing=1.0):
    """Parallel-beam forward model: A[i, j] = intersection length of ray i with
    pixel j; ray_view[i] = which projection ray i belongs to. Each ray is traced
    by oversampling points along it and accumulating the step length ds into the
    pixel each sample lands in."""
    if n_rays is None:
        n_rays = n
    rows, cols, vals = [], [], []
    s = (np.arange(n_rays) - (n_rays - 1) / 2.0) * det_spacing      # ray offsets
    half = n / 2.0
    n_samp = 4 * n
    t = (np.arange(n_samp) - (n_samp - 1) / 2.0) * (np.sqrt(2.0) * n / n_samp)
    ds = np.sqrt(2.0) * n / n_samp
    i = 0
    for theta in angles:
        c, sn = np.cos(theta), np.sin(theta)
        for off in s:
            x = off * (-sn) + t * c
            y = off * (c) + t * sn
            px = np.floor(x + half).astype(int)
            py = np.floor(y + half).astype(int)
            inside = (px >= 0) & (px < n) & (py >= 0) & (py < n)
            flat = py[inside] * n + px[inside]
            if flat.size:
                uniq, cnt = np.unique(flat, return_counts=True)
                rows += [i] * uniq.size
                cols += uniq.tolist()
                vals += (cnt * ds).tolist()                          # length per pixel
            i += 1
    A = csr_matrix((vals, (rows, cols)), shape=(len(angles) * n_rays, n * n))
    ray_view = np.repeat(np.arange(len(angles)), n_rays)
    return A, ray_view


def forward_project(A, x):
    return A.dot(x)


def apply_constraints(x, nonneg=True):
    if nonneg:
        np.clip(x, 0.0, None, out=x)
    return x


def art(A, b, n_sweeps=10, lam=1.0, x0=None, nonneg=True, order=None):
    """Cyclic Kaczmarz / ART: project the estimate onto each ray's hyperplane.
        x <- x + lam * (b_i - a_i.x) / ||a_i||^2 * a_i"""
    m, ncol = A.shape
    x = np.zeros(ncol) if x0 is None else x0.astype(float).copy()
    A = A.tocsr()
    row_norm2 = np.asarray(A.multiply(A).sum(axis=1)).ravel()        # ||a_i||^2
    idx = np.arange(m) if order is None else np.asarray(order)
    for _ in range(n_sweeps):
        for i in idx:
            if row_norm2[i] == 0.0:
                continue
            ai = A.getrow(i)
            resid = b[i] - ai.dot(x)[0]                              # b_i - a_i.x
            x = x + lam * (resid / row_norm2[i]) * ai.toarray().ravel()
            apply_constraints(x, nonneg=nonneg)                      # projection onto x>=0
    return x


def sart(A, b, ray_view, n_iters=5, lam=1.0, x0=None, nonneg=True):
    """SART: per-view simultaneous update with row-sum and column-sum weighting.
        x_j <- x_j + lam * [sum_{i in V} a_ij (b_i - a_i.x)/(sum_k a_ik)]
                          / (sum_{i in V} a_ij)"""
    m, ncol = A.shape
    x = np.zeros(ncol) if x0 is None else x0.astype(float).copy()
    A = A.tocsr()
    row_sum = np.asarray(A.sum(axis=1)).ravel()                     # L_i = sum_k a_ik
    row_sum = np.where(row_sum == 0.0, 1.0, row_sum)
    for _ in range(n_iters):
        for v in np.unique(ray_view):
            sel = np.where(ray_view == v)[0]
            Av = A[sel]
            resid = b[sel] - Av.dot(x)                              # b_i - a_i.x
            weighted = resid / row_sum[sel]                         # / ray length L_i
            num = Av.T.dot(weighted)                                # back-project sum_i a_ij(.)
            col = np.asarray(Av.sum(axis=0)).ravel()                # sum_i a_ij over view
            col = np.where(col == 0.0, 1.0, col)
            x = x + lam * num / col                                 # / per-pixel coverage
            apply_constraints(x, nonneg=nonneg)
    return x


def reconstruct(A, b, ray_view=None, mode="view", **params):
    if mode == "row":
        return art(A, b, **params)
    if mode == "view":
        if ray_view is None:
            raise ValueError("ray_view is required for view-wise reconstruction")
        return sart(A, b, ray_view, **params)
    raise ValueError("mode must be 'row' or 'view'")


if __name__ == "__main__":
    n = 32
    yy, xx = np.mgrid[0:n, 0:n]
    cx = cy = (n - 1) / 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    img = np.where(r < n * 0.42, 0.3, 0.0)                          # small phantom
    img += np.where((xx - n*0.40)**2 + (yy - n*0.55)**2 < (n*0.12)**2, 0.6, 0.0)
    img += np.where((xx - n*0.62)**2 + (yy - n*0.42)**2 < (n*0.08)**2, 0.5, 0.0)

    angles = np.linspace(0, np.pi, 30, endpoint=False)
    A, ray_view = build_system_matrix(n, angles, n_rays=n)
    b = forward_project(A, img.ravel())                             # simulated ray-sums

    nv = len(angles)                                               # wide-angle ray order
    perm = np.argsort([(k * (nv // 2 + 1)) % nv for k in range(nv)])
    order = np.concatenate([np.where(ray_view == v)[0] for v in perm])

    x_art = reconstruct(A, b, ray_view, mode="row", n_sweeps=8, lam=0.2, order=order).reshape(n, n)
    x_sart = reconstruct(A, b, ray_view, mode="view", n_iters=8, lam=1.0).reshape(n, n)
    rmse = lambda u: np.sqrt(np.mean((u - img) ** 2))
    print("ART RMSE:", round(rmse(x_art), 4), " SART RMSE:", round(rmse(x_sart), 4))
```
