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

1. `AddHs`; build the bounds matrix (1‑2/1‑3 from bond and angle rules, 1‑4 from torsion range, vdW
   lower bounds elsewhere).
2. Triangle‑smooth the bounds; reject on inconsistency.
3. Sample one distance matrix from the smoothed box (optionally metrize for better sampling).
4. Metric‑matrix embed in 4D (top eigenvectors of the Gram matrix).
5. First refinement: distance + chiral + fourth‑dimension penalties; check/repair stereocenters;
   squeeze the fourth dimension.
6. ET+K minimization: distance‑violation + experimental‑torsion + improper/planarity terms (~300 steps).
7. Reseed for the next conformer. Optional final MMFF/UFF minimization.

## Code

User-facing RDKit path:

```python
from rdkit import Chem
from rdkit.Chem import AllChem

def make_embedding_parameters(seed=0xF00D):
    params = AllChem.ETKDG()
    params.randomSeed = seed
    params.enforceChirality = True
    params.randNegEig = True
    params.numZeroFail = 1
    params.optimizerForceTol = 1e-3
    add_geometric_knowledge(params)
    return params

def add_geometric_knowledge(params):
    params.useExpTorsionAnglePrefs = True
    params.useBasicKnowledge = True

def embed_ensemble_from_graph(smiles, n_confs=50, seed=0xF00D, do_forcefield_cleanup=False):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = make_embedding_parameters(seed)
    cids = list(AllChem.EmbedMultipleConfs(mol, numConfs=n_confs, params=params))
    if do_forcefield_cleanup:
        AllChem.MMFFOptimizeMoleculeConfs(mol)
    return mol, cids

mol, conf_ids = embed_ensemble_from_graph("CN1CCC(O)(CC1)c1ccccc1Cl", n_confs=50)
```

Internal pieces filled by that parameter set:

```cpp
// TriangleSmooth.cpp: upper triangle stores U, lower triangle stores L.
U_ij = min(U_ij, U_ik + U_kj);
L_ij = max(L_ij, L_ik - U_kj, L_kj - U_ik);
if (L_ij - U_ij > 0.) return false;

// DistGeomUtils.cpp: metric-matrix EMBED.
// RDKit's symmetric storage sums each distinct off-diagonal squared distance once:
// sumSqD2 = (1/N^2) sum_{i<j} D_ij = (1/(2N^2)) sum_{i,j} D_ij for a full matrix.
sqD0i[i] = (1.0 / N) * sum_j(D_ij) - sumSqD2;
T.setVal(i, j, 0.5 * (sqD0i[i] + sqD0i[j] - D_ij));
powerEigenSolver(nEigs, T, eigVals, eigVecs, ...);
coord[i][a] = sqrt(eigVals[a]) * eigVecs[a][i];       // negative axis -> random jitter

// TorsionPreferences.cpp: per-SMARTS experimental torsion potential.
V(phi) = V1*(1 + s1*cos(phi)) + ... + V6*(1 + s6*cos(6*phi));

// DistGeomUtils.cpp / Embedder.cpp: ET+K force field, then a short minimization.
addExperimentalTorsionTerms(field, etkdgDetails, atomPairs, N);
addImproperTorsionTerms(field, 10.0, etkdgDetails.improperAtoms, isImproperConstrained);
add12Terms(field, etkdgDetails, atomPairs, positions, KNOWN_DIST_FORCE_CONSTANT, N);
add13Terms(field, etkdgDetails, atomPairs, positions, KNOWN_DIST_FORCE_CONSTANT,
           isImproperConstrained, true, mmat, N);
addLongRangeDistanceConstraints(field, etkdgDetails, atomPairs, positions,
                                KNOWN_DIST_FORCE_CONSTANT, mmat, N);
field->minimize(300, optimizerForceTol);
```

`add_geometric_knowledge` is where ET+K enters: experimental torsion terms and basic-knowledge
planarity/linearity terms are enabled before embedding. `EmbedMultipleConfs` then runs the bounds,
smoothing, distance sampling, EMBED, distance/chirality cleanup, fourth-dimension squeeze, and ET+K
minimization loop once per conformer seed.
