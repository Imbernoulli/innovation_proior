The reconstruction problem is to recover a two-dimensional density f(x, y) from its line integrals measured by a detector. After taking logarithms, each ray gives an equation p_i = integral of f along that ray. Analytically this is inversion of the Radon transform, and filtered backprojection is the classical solution, but it assumes dense, uniformly sampled, straight-line projections over a full angular range and low noise. In electron microscopy or low-dose X-ray imaging the tilt range is limited, the views are few, and the data are noisy, so filtered backprojection amplifies noise and leaves streak artifacts. It also cannot incorporate simple physical priors such as nonnegativity or a known support region. A practical method must therefore work directly with the sparse discretized system A f = p, where a_ij is the intersection length of ray i with pixel j, without ever forming or inverting the full matrix.

The Algebraic Reconstruction Technique, ART, is the right tool for this setting. It views each equation a_i · f = p_i as a hyperplane in the high-dimensional space of pixel values and reconstructs by projecting the current estimate onto one hyperplane at a time, cycling through the rays. The orthogonal projection of a point x onto the hyperplane a_i · z = p_i is x + ((p_i - a_i·x) / ||a_i||^2) a_i. Here a_i·x is the ray-sum predicted by the current image, p_i - a_i·x is the residual, and the squared row norm ||a_i||^2 is what makes the step land exactly on the hyperplane. Because the update only touches the pixels that ray i actually crosses, it needs no dense matrix storage and costs O(number of nonzero entries) per sweep.

In the real, discretized problem the hyperplanes do not all intersect cleanly. Full projections with relaxation factor lambda = 1 satisfy each ray exactly, but successive rays from nearby angles overwrite each other and produce salt-and-pepper noise. Using underrelaxation, lambda < 1, softens each correction so that the contradictions average rather than fight, trading a few extra sweeps for a much cleaner image. Convergence speed also depends on the angle between successive hyperplanes: near-parallel rays converge slowly, so the rays should be visited in an order that jumps widely in angle rather than proceeding sequentially through neighboring views. After each step it is cheap to enforce physical constraints by clipping negative values to zero and zeroing pixels outside the known support.

A smoother refinement is SART, the Simultaneous Algebraic Reconstruction Technique. Instead of correcting pixels ray by ray, SART processes one projection view at a time against a frozen image, computes all residuals for that view, and applies them together. Each residual is divided by the ray's total weight, its physical length through the region, and the back-projected correction is divided by the total weight each pixel receives from that view. This removes the within-view ambiguity that causes salt-and-pepper noise while retaining the fast, view-by-view structure of ART. The code below implements both variants: ART as cyclic Kaczmarz with a wide-angle ordering, and SART as the per-view simultaneous update with row-sum and column-sum normalization.

```python
import numpy as np
from scipy.sparse import csr_matrix


def build_system_matrix(n, angles, n_rays=None, det_spacing=1.0):
    """Parallel-beam forward model: A[i, j] = intersection length of ray i
    with pixel j.  ray_view[i] identifies the projection angle ray i belongs to."""
    if n_rays is None:
        n_rays = n
    rows, cols, vals = [], [], []
    s = (np.arange(n_rays) - (n_rays - 1) / 2.0) * det_spacing
    half = n / 2.0
    n_samp = 4 * n
    t = (np.arange(n_samp) - (n_samp - 1) / 2.0) * (np.sqrt(2.0) * n / n_samp)
    ds = np.sqrt(2.0) * n / n_samp
    i = 0
    for theta in angles:
        c, sn = np.cos(theta), np.sin(theta)
        for off in s:
            x = off * (-sn) + t * c
            y = off * c + t * sn
            px = np.floor(x + half).astype(int)
            py = np.floor(y + half).astype(int)
            inside = (px >= 0) & (px < n) & (py >= 0) & (py < n)
            flat = py[inside] * n + px[inside]
            if flat.size:
                uniq, cnt = np.unique(flat, return_counts=True)
                rows += [i] * uniq.size
                cols += uniq.tolist()
                vals += (cnt * ds).tolist()
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
    """Cyclic Kaczmarz / ART."""
    m, ncol = A.shape
    x = np.zeros(ncol) if x0 is None else x0.astype(float).copy()
    A = A.tocsr()
    row_norm2 = np.asarray(A.multiply(A).sum(axis=1)).ravel()
    idx = np.arange(m) if order is None else np.asarray(order)
    for _ in range(n_sweeps):
        for i in idx:
            if row_norm2[i] == 0.0:
                continue
            ai = A.getrow(i)
            resid = b[i] - ai.dot(x)[0]
            x = x + lam * (resid / row_norm2[i]) * ai.toarray().ravel()
            apply_constraints(x, nonneg=nonneg)
    return x


def sart(A, b, ray_view, n_iters=5, lam=1.0, x0=None, nonneg=True):
    """Simultaneous Algebraic Reconstruction Technique, per-view update."""
    m, ncol = A.shape
    x = np.zeros(ncol) if x0 is None else x0.astype(float).copy()
    A = A.tocsr()
    row_sum = np.asarray(A.sum(axis=1)).ravel()
    row_sum = np.where(row_sum == 0.0, 1.0, row_sum)
    for _ in range(n_iters):
        for v in np.unique(ray_view):
            sel = np.where(ray_view == v)[0]
            Av = A[sel]
            resid = b[sel] - Av.dot(x)
            weighted = resid / row_sum[sel]
            num = Av.T.dot(weighted)
            col = np.asarray(Av.sum(axis=0)).ravel()
            col = np.where(col == 0.0, 1.0, col)
            x = x + lam * num / col
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
    img = np.where(r < n * 0.42, 0.3, 0.0)
    img += np.where((xx - n*0.40)**2 + (yy - n*0.55)**2 < (n*0.12)**2, 0.6, 0.0)
    img += np.where((xx - n*0.62)**2 + (yy - n*0.42)**2 < (n*0.08)**2, 0.5, 0.0)

    angles = np.linspace(0, np.pi, 30, endpoint=False)
    A, ray_view = build_system_matrix(n, angles, n_rays=n)
    b = forward_project(A, img.ravel())

    nv = len(angles)
    perm = np.argsort([(k * (nv // 2 + 1)) % nv for k in range(nv)])
    order = np.concatenate([np.where(ray_view == v)[0] for v in perm])

    x_art = reconstruct(A, b, ray_view, mode="row", n_sweeps=8, lam=0.2, order=order).reshape(n, n)
    x_sart = reconstruct(A, b, ray_view, mode="view", n_iters=8, lam=1.0).reshape(n, n)
    rmse = lambda u: np.sqrt(np.mean((u - img) ** 2))
    print("ART RMSE:", round(rmse(x_art), 4), " SART RMSE:", round(rmse(x_sart), 4))
```
