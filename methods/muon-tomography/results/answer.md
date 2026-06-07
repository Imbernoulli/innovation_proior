# Muon scattering tomography: PoCA and the maximum-likelihood upgrade

## Problem

Image the interior of a large, dense, opaque object (a cargo container or shielded cask) and
detect concealed high-`Z` material (uranium, plutonium, lead, tungsten) — passively, with no
artificial radiation source, in 3D, in a usable exposure time. X-rays and absorption radiography
fail: they are stopped by heavy shielding, are only weakly `Z`-sensitive, and give a 2D
line-integral, not a 3D map.

## Key idea

Use the natural cosmic-ray muon flux (`~1 cm^-2 min^-1`, mean energy 3–4 GeV, highly penetrating,
low enough rate to track one muon at a time). As a muon crosses material it undergoes **multiple
Coulomb scattering**; the projected scattering angle is approximately zero-mean Gaussian with the
Highland / Lynch–Dahl width

```
theta_0 = (13.6 MeV / (beta c p)) * z * sqrt(L / X0) * [1 + 0.038 * ln(L z^2 / (X0 beta^2))]
        ~= (15 MeV / p) * sqrt(L / X0)   for muons (beta ~ 1).
```

The radiation length `X0` drops sharply with atomic number (water 36 cm, iron 1.8 cm, lead 0.56
cm, uranium 0.32 cm), so high-`Z` material scatters far more per centimetre. Define the
**scattering density**

```
lambda := (15 / p0)^2 * (1 / X0) = theta_0^2 / L     (mean-square projected scatter per unit length),
```

a momentum- and thickness-normalized, strongly `Z`-sensitive material fingerprint. Measure each
muon's incoming and outgoing straight-line tracks with tracking stations above and below the volume,
and reconstruct a 3D map of `lambda`.

## Algorithm

**PoCA (Point of Closest Approach) — the cheap first cut.** Assume *all* of a muon's scattering
happened at a single point. Estimate that vertex as the closest-approach point of the extended
incoming and outgoing tracks (in 2D they cross; in 3D they are skew, so take the midpoint of the
common perpendicular). In one projected view deposit `s = (theta_out - theta_in)^2`; in 3D use
`s = 1/2[(Delta_theta_x)^2 + (Delta_theta_y)^2]`, the average of the two independent projected
variance samples. Add `s` to the voxel containing the closest-approach point and count every voxel
crossed by the estimated straight path. Each voxel's scattering density is
`lambda_hat(j) = (sum assigned s) / (path_count_j * L)`.
O(M), non-iterative; sharp for small isolated high-`Z` objects, but it smears where scattering is
distributed (multiple/extended objects) because the single-scatter assumption breaks, and it uses
only the angle.

**MLS (Maximum Likelihood, Scattering) — the tomographic upgrade.** The angle is one noisy sample
of a zero-mean projected Gaussian whose *variance* is the path-length raysum
`v_i = sum_j L_ij lambda_j`. Two projected views can be represented as two independent samples with
the same path-length row. Maximize the data likelihood:

```
lambda_hat = argmin_lambda  sum_i [ ln(v_i) + theta_i^2 / v_i ],   v_i = (L lambda)_i,
             subject to  lambda_j >= lambda_air.
```

Each voxel on a path carries its share of the signal, and many overlapping rays from many angles
jointly resolve distributed scattering — what PoCA cannot.

**MLSD (+ Displacement) — lift the along-ray degeneracy.** Angle alone cannot tell where on a path
the scattering sat. Add the lateral displacement `Delta_x`. Per ray, with `T_j` = path length
downstream of cell `j`, three linear weight vectors give the 2×2 covariance:

```
W_theta(j)  = L_j
W_thetax(j) = L_j^2/2 + L_j T_j
W_x(j)      = L_j^3/3 + T_j L_j^2 + T_j^2 L_j
Sigma_i = [[W_theta·lambda, W_thetax·lambda], [W_thetax·lambda, W_x·lambda]]
d_i = [theta_out - theta_in ;  (x_out - x_proj) cos(theta_avg)]
lambda_hat = argmin_lambda  sum_i [ ln|Sigma_i| + d_i^T Sigma_i^{-1} d_i ],   lambda_j >= lambda_air.
```

The `L^3/3` and `L^2/2` moments and the downstream lever arms encode position-along-path (the
single-slab angle/displacement correlation is `sqrt(3)/2 ≈ 0.866`), resolving the top-vs-bottom
ambiguity MLS leaves.

## Code

```python
import numpy as np
from scipy.optimize import minimize

P0_MEV = 3000.0  # nominal muon momentum, 3 GeV/c

def scattering_density(X0_cm, p0_mev=P0_MEV):
    """lambda = (15/p0)^2 / X0  (Highland width squared per unit length), rad^2/cm."""
    return (15.0 / p0_mev) ** 2 / X0_cm

LAMBDA_AIR = scattering_density(X0_cm=3.04e4)   # X0(air) ~ 304 m

class VoxelGrid:
    def __init__(self, lo, hi, n):
        self.lo, self.hi, self.n = np.asarray(lo, float), np.asarray(hi, float), np.asarray(n, int)
        self.size = (self.hi - self.lo) / self.n
        self.L = float(np.mean(self.size))
    def index(self, pt):
        idx = ((np.asarray(pt) - self.lo) / self.size).astype(int)
        if np.any(idx < 0) or np.any(idx >= self.n):
            return None
        return tuple(idx)

def line_voxels(p0, p1, grid):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    n = max(2, int(np.ceil(np.linalg.norm(p1 - p0) / np.min(grid.size))) * 3)
    hits, seen = [], set()
    for alpha in np.linspace(0.0, 1.0, n):
        j = grid.index(p0 + alpha * (p1 - p0))
        if j is not None and j not in seen:
            seen.add(j)
            hits.append(j)
    return hits

# ----- 3D geometry: closest-approach vertex of the in/out tracks -----
def closest_approach_point(p_in, v_in, p_out, v_out):
    v_in = v_in / np.linalg.norm(v_in)
    v_out = v_out / np.linalg.norm(v_out)
    w0 = p_in - p_out
    a, b, c = v_in @ v_in, v_in @ v_out, v_out @ v_out
    d, e = v_in @ w0, v_out @ w0
    denom = a * c - b * b
    if abs(denom) < 1e-9:                  # parallel tracks: no kink to localize
        return None
    t = (b * e - c * d) / denom
    s = (a * e - b * d) / denom
    return 0.5 * ((p_in + t * v_in) + (p_out + s * v_out))

def projected_angles(v):
    v = v / np.linalg.norm(v)
    vz = max(abs(v[2]), 1e-12)
    return np.array([np.arctan2(v[0], vz), np.arctan2(v[1], vz)])

def projected_scattering_signal(v_in, v_out):
    dtheta = projected_angles(v_out) - projected_angles(v_in)
    return 0.5 * float(dtheta @ dtheta)

# ----- PoCA reconstruction -----
def reconstruct_poca(muons, grid):
    S = np.zeros(tuple(grid.n)); I = np.zeros(tuple(grid.n))
    for p_in, v_in, p_out, v_out in muons:
        p_in, v_in = np.asarray(p_in, float), np.asarray(v_in, float)
        p_out, v_out = np.asarray(p_out, float), np.asarray(v_out, float)
        for j in line_voxels(p_in, p_out, grid):
            I[j] += 1
        pt = closest_approach_point(p_in, v_in, p_out, v_out)
        if pt is None:
            continue
        j = grid.index(pt)
        if j is None:
            continue
        if I[j] == 0:
            I[j] += 1
        S[j] += projected_scattering_signal(v_in, v_out)
    lam = np.zeros_like(S); nz = I > 0
    lam[nz] = S[nz] / (I[nz] * grid.L)   # mean projected theta^2 per unit length
    return lam

# ----- Maximum-likelihood (MLS) reconstruction -----
def reconstruct_mls(signals, Lmat, lambda_air=LAMBDA_AIR, lam0=None):
    Lmat = np.asarray(Lmat, float)
    s2 = np.asarray(signals, float) ** 2
    n = Lmat.shape[1]
    def nll(lam):
        v = np.maximum(Lmat @ lam, 1e-12)
        return np.sum(np.log(v) + s2 / v)        # -2 logL of projected Gaussians
    x0 = np.full(n, lambda_air) if lam0 is None else np.asarray(lam0, float)
    return minimize(nll, x0, method="L-BFGS-B", bounds=[(lambda_air, None)] * n).x
```

PoCA is the standard fast first-pass reconstruction; MLS/MLSD are the statistical refinements that
trade compute for resolution of distributed scattering and along-path position.
