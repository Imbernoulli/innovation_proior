# Experimental-Torsion Knowledge Distance Geometry (ETKDG)

## Problem

A molecule arrives as a 2D graph — atoms, bonds, bond orders, formal charges, stereo flags — with no
coordinates. Downstream work (docking, pharmacophore matching, 3D shape comparison, descriptors) needs
Cartesian coordinates, and a flexible molecule is a *cloud* of shapes, so the task is to emit an
*ensemble* of 3D conformers from the graph alone. The ensemble must be **realistic** (sane bonds and
angles, closed rings, flat aromatics, correct chirality, torsions in the basins real molecules
occupy), **diverse** (covering accessible space, since the bioactive conformer can sit a few kcal/mol
off the minimum), and **cheap** (millions of molecules, no starting geometry, no human in the loop).

## Key idea

A conformation is fixed, up to rigid motion and reflection, by its matrix of pairwise distances plus
the signs of selected tetrahedral volumes (chirality). Distances are rotation/translation invariant —
exactly the right object — so reason about the `N×N` distance matrix instead of the `3N` coordinates.
The graph hands over distances directly: 1‑2 (bonded) and 1‑3 (shared neighbor, by the law of cosines)
are tight; 1‑4 (across a rotatable bond) is bounded by the syn↔anti torsion sweep; everything farther
is bounded below by van der Waals radii. So carry an `N×N` **bounds matrix** `[L_ij, U_ij]`.

Four moves turn that into conformers:

1. **Triangle smoothing.** Local bounds ignore global consistency, but `d_ij ≤ d_ik + d_kj` and
   `d_ij ≥ |d_ik − d_kj|` for any third atom `k`. Tighten `U_ij ← min(U_ij, U_ik + U_kj)` and
   `L_ij ← max(L_ij, L_ik − U_kj, L_kj − U_ik)`, iterating over all `k` — this is all-pairs shortest
   paths (Floyd–Warshall, `O(N³)`). It also flags contradictions: if ever `L_ij > U_ij`, reject.

2. **Sample + metric-matrix embed.** Draw each `d_ij` uniformly in `[L_ij, U_ij]`; a different seed
   gives a different conformer, so diversity is free. Recover coordinates in *closed form* (no local
   minima) via classical MDS. With the centroid at the origin and `D_ij = d_ij²`,

       D0_i = (1/N) Σ_j D_ij − (1/(2N²)) Σ_{j,k} D_jk          (squared distance to centroid)
       G_ij = ½ (D0_i + D0_j − D_ij)                            (Gram matrix ⟨x_i, x_j⟩)

   `G` is symmetric PSD of rank 3 when the distances are realizable; eigendecompose
   `G = Σ_a λ_a v_a v_aᵀ`, and the coordinate of atom `i` on axis `a` is `√λ_a (v_a)_i`. Keep the top
   eigenpairs (the power method suffices, `O(N²)`/pass). A "needed" eigenvalue going negative means the
   draw was not 3D‑realizable — jitter that axis, or resample if the structure collapses.

3. **Refine against an error function.** Sampled distances only approximately satisfy all bounds at
   once, and the distance matrix is blind to reflection, so chirality can invert. Minimize a penalty
   for each pair outside `[L_ij, U_ij]` plus chiral‑volume penalties. Embedding in **4D** and
   penalizing the fourth coordinate gives wrong‑handed/tangled structures room to relax continuously
   before the fourth dimension is squeezed to zero.

4. **Inject experimental knowledge (the ET + K step).** Generic bounds leave torsions roughly
   *uniform*, whereas crystallographic (CSD) histograms show each torsion motif is sharply
   *multimodal*; and pairwise vdW lower bounds never force aromatic rings flat or sp2 centers planar.
   Leaning on a full force field to fix this is expensive and collapses diversity. Instead, after
   embedding run a *short* minimization of a small field: per‑motif torsion potentials fitted to CSD
   histograms (**ET**), plus stiff planarity/linearity rules (**K**), alongside the distance‑violation
   terms. A multimodal torsion preference is not a single interval, so this acts on the 3D coordinates,
   not on the bounds. Each torsion uses a sixfold cosine series whose minima sit on the histogram peaks:

       V(φ) = Σ_{n=1..6} V_n · (1 + s_n cos(nφ)),    s_n ∈ {−1, +1},  V_n ≥ 0.

Because the wells are the experimental basins, a few hundred steps slide each conformer into a
realistic basin without homogenizing the ensemble — so the heavy MMFF/UFF cleanup can usually be
dropped.

## Algorithm (per conformer)

1. `AddHs`; build the bounds matrix (1‑2/1‑3 from experimental bond/angle data, 1‑4 from torsion
   range, vdW lower bounds elsewhere).
2. Triangle‑smooth the bounds; reject on inconsistency.
3. Sample one distance matrix from the smoothed box (optionally metrize for better sampling).
4. Metric‑matrix embed in 4D (top eigenvectors of the Gram matrix).
5. First refinement: distance + chiral + fourth‑dimension penalties; check/repair stereocenters;
   squeeze the fourth dimension.
6. ET+K minimization: distance‑violation + experimental‑torsion + improper/planarity terms (~300 steps).
7. Reseed for the next conformer. Optional final MMFF/UFF minimization.

## Code

Faithful self-contained implementation of the distance-geometry engine; the ET+K force-field terms
are sketched at the level the reasoning derived them.

```python
import numpy as np

def triangle_smooth(L, U):
    """Tighten bounds via the triangle inequality (Floyd-Warshall, O(N^3)).
    L, U are N x N lower/upper bound matrices, modified in place.
    Returns False if the bounds are inconsistent."""
    N = L.shape[0]
    for k in range(N):
        for i in range(N):
            for j in range(N):
                if i == j:
                    continue
                if U[i, j] > U[i, k] + U[k, j]:          # d_ij <= d_ik + d_kj
                    U[i, j] = U[i, k] + U[k, j]
                lb = max(L[i, k] - U[k, j], L[k, j] - U[i, k])   # d_ij >= |d_ik - d_kj|
                if lb > L[i, j]:
                    L[i, j] = lb
                if L[i, j] - U[i, j] > 1e-9:             # contradiction
                    return False
    return True


def sample_distance_matrix(L, U, rng):
    """One distance matrix from the smoothed box: each d_ij uniform in [L_ij, U_ij]."""
    N = L.shape[0]
    d = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d[i, j] = d[j, i] = rng.uniform(L[i, j], U[i, j])
    return d


def metric_matrix_embed(dmat, dim=4):
    """Classical MDS / EMBED: distances -> coordinates in closed form via the Gram matrix.
    Returns an N x dim coordinate array (centroid at origin)."""
    N = dmat.shape[0]
    D = dmat ** 2                                        # squared distances
    sumSqD2 = D.sum() / (N * N)                          # (1/N^2) sum_{j,k} D_jk
    D0 = D.sum(axis=1) / N - 0.5 * sumSqD2               # D0_i = squared dist to centroid
    G = 0.5 * (D0[:, None] + D0[None, :] - D)            # Gram matrix G_ij = <x_i, x_j>

    eigval, eigvec = np.linalg.eigh(G)                   # ascending; take the top `dim`
    idx = np.argsort(eigval)[::-1][:dim]
    coords = np.zeros((N, dim))
    for a, e in enumerate(idx):
        lam = eigval[e]
        if lam > 0:
            coords[:, a] = np.sqrt(lam) * eigvec[:, e]   # x_i on axis a = sqrt(lambda_a) * (v_a)_i
        else:
            coords[:, a] = 1e-3 * np.random.randn(N)     # non-realizable axis -> jitter
    return coords


def distance_error(coords, L, U):
    """Penalty for every pair whose distance leaves [L_ij, U_ij], plus a 4th-dim squeeze."""
    N = coords.shape[0]
    err = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(coords[i] - coords[j])
            if d < L[i, j]:
                err += (L[i, j] ** 2 / d ** 2 - 1.0) ** 2       # lower-bound violation
            elif d > U[i, j]:
                err += (d ** 2 / U[i, j] ** 2 - 1.0) ** 2       # upper-bound violation
    if coords.shape[1] > 3:
        err += (coords[:, 3] ** 2).sum()                        # squeeze the 4th dimension
    return err


def chiral_volume(coords, c, n1, n2, n3):
    """Signed tetrahedral volume at a stereocenter; sign encodes handedness (R vs S)."""
    return np.dot(coords[n1, :3] - coords[c, :3],
                  np.cross(coords[n2, :3] - coords[c, :3],
                           coords[n3, :3] - coords[c, :3]))


def exp_torsion_energy(phi, terms):
    """ET potential fitted to a CSD histogram: V(phi) = sum_n V_n (1 + s_n cos(n phi))."""
    return sum(Vn * (1.0 + sn * np.cos((n + 1) * phi))
               for n, (sn, Vn) in enumerate(terms))             # n = 0..5  -> 1..6 fold


def embed_conformer(mol, seed, use_exp_torsions=True):
    """Full ETKDG per-conformer pipeline. `mol` carries bounds, stereo, and torsion motifs."""
    from scipy.optimize import minimize
    rng = np.random.default_rng(seed)
    L, U = mol.lower_bounds.copy(), mol.upper_bounds.copy()      # 1-2,1-3 tight; 1-4 range; vdW else

    if not triangle_smooth(L, U):                               # tighten + consistency check
        return None
    dmat = sample_distance_matrix(L, U, rng)
    coords = metric_matrix_embed(dmat, dim=4)                   # 4D for chirality untangling

    # First refinement: distance + chiral + 4th-dimension penalties.
    def obj1(x):
        c = x.reshape(coords.shape)
        e = distance_error(c, L, U)
        for (a, n1, n2, n3, target_sign) in mol.chiral_centers:
            v = chiral_volume(c, a, n1, n2, n3)
            if np.sign(v) != target_sign:                      # inverted -> penalize
                e += v ** 2
        return e
    coords = minimize(obj1, coords.ravel(), method="L-BFGS-B").x.reshape(coords.shape)
    coords = coords[:, :3]                                      # 4th dim squeezed -> back to 3D

    # ET + K minimization: distance-violation + experimental-torsion + planarity terms.
    if use_exp_torsions:
        def obj2(x):
            c = x.reshape(coords.shape)
            e = distance_error(c, np.maximum(L, 0)[:, :], U)   # keep bounds satisfied
            for (i, j, k, l, terms) in mol.torsion_motifs:     # ET: per-SMARTS CSD-fitted series
                e += exp_torsion_energy(dihedral(c, i, j, k, l), terms)
            for (a, n1, n2, n3) in mol.sp2_centers:            # K: improper -> planarity (stiff)
                e += 100.0 * chiral_volume(np.c_[c, np.zeros(len(c))], a, n1, n2, n3) ** 2
            return e
        coords = minimize(obj2, coords.ravel(), method="L-BFGS-B",
                          options={"maxiter": 300}).x.reshape(coords.shape)
    return coords                                              # optional MMFF/UFF cleanup afterwards


def dihedral(c, i, j, k, l):
    b1, b2, b3 = c[j] - c[i], c[k] - c[j], c[l] - c[k]
    n1, n2 = np.cross(b1, b2), np.cross(b2, b3)
    m = np.cross(n1, b2 / np.linalg.norm(b2))
    return np.arctan2(np.dot(m, n2), np.dot(n1, n2))


# Ensemble = many seeds. In RDKit this is the ETKDGv3 parameter set:
#   from rdkit.Chem import AllChem
#   mol = AllChem.AddHs(mol); params = AllChem.ETKDGv3(); params.randomSeed = 0xf00d
#   cids = AllChem.EmbedMultipleConfs(mol, numConfs=50, params=params)
def embed_ensemble(mol, n_confs=50, base_seed=0):
    return [c for s in range(base_seed, base_seed + n_confs)
            if (c := embed_conformer(mol, s)) is not None]
```

The engine is exactly the four moves: `triangle_smooth` is the bound-tightening all-pairs loop;
`metric_matrix_embed` is the centroid-centered Gram-matrix eigendecomposition (`D0` → `G` →
`√λ · v`); the first `minimize` enforces bounds + chirality with a fourth dimension to untangle
handedness; and the ET+K `minimize` slides each conformer into an experimentally-preferred basin via
CSD-fitted torsion series `V(φ) = Σ V_n(1 + s_n cos nφ)` plus stiff sp2-planarity impropers — cheaply,
and without homogenizing the ensemble, so the heavy classical minimization can usually be dropped.
