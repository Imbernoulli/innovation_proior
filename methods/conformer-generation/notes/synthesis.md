# Synthesis — ETKDG conformer generation

## Three sources (retrieved this run)
1. **Primary**: ETKDG method (Riniker & Landrum 2015, J. Chem. Inf. Model. 55:2562, "Better Informed
   Distance Geometry"). The ACS full text is paywalled; the method's *exact* equations are captured
   from its canonical implementation in RDKit (CrystalFF TorsionPreferences.cpp + Embedder.cpp,
   DistGeomUtils.cpp), plus the BLOPIG explainer and RDKit docs. The classic DG conformer method
   (the EMBED algorithm; Crippen/Havel; Blaney & Dixon review) is the background ancestor.
2. **Background**: Havel, "Distance Geometry: Theory, Algorithms, and Chemical Applications"
   (refs/havel-dg-review.pdf, 10pp, read in full) — the bounds matrix, triangle-inequality bound
   smoothing (all-pairs shortest paths), metrization, and the metric-matrix EMBED algorithm
   (eigendecomposition). Blaney & Dixon "Distance Geometry in Molecular Modeling" (Rev. Comp. Chem.)
   is the same lineage.
3. **Third-party explainer**: BLOPIG "Advances in Conformer Generation: ETKDG and ETDG" + RDKit
   "Getting Started in Python" conformer section + RDKit blog (greglandrum) on ETKDG + distance
   constraints / random-coordinate embedding.

## The problem (research question)
Given a 2D molecular graph (atoms + bonds, no coordinates), produce an ensemble of diverse,
physically-realistic, low-energy 3D conformers. Needed for docking, pharmacophore search,
shape screening, 3D-QSAR, property prediction. Must be: fast (thousands of molecules), diverse
(cover the accessible conformation space, not collapse to one basin), and realistic (bond lengths,
angles, ring planarity, torsion preferences right).

## Background math (Havel, fully grounded)
- A conformation can be *described* purely by interatomic distances + chirality (signed tetrahedron
  volumes). The graph fixes some distances exactly (1-2 bonds, 1-3 from bond angles) and bounds the
  rest. So: build an N×N **bounds matrix** with lower bound L_ij and upper bound U_ij for every pair.
  RDKit convention: upper bounds in the upper triangle, lower in the lower triangle (BoundsMatrix.h).
- **Triangle-inequality bound smoothing.** The naive bounds are loose. For any third atom k:
  U_ij ≤ U_ik + U_kj (upper bound can't exceed going via k), and
  L_ij ≥ max( L_ik − U_kj , L_kj − U_ik ) (lower bound: can't be closer than the difference).
  Iterating this to convergence over all k = computing all-pairs shortest paths (Floyd-Warshall),
  O(N³). This is `triangleSmoothBounds` (TriangleSmooth.cpp) — matches Havel eq. for upper limits
  (shortest path with arc lengths = upper bounds) and lower limits ℓ_ij = ℓ_km − u_ik − u_jm.
  Tighter bounds (tetrangle, Cayley-Menger 4-point) exist but are O(N⁴) and rarely used.
- **Sample a distance matrix.** Pick each d_ij uniformly (or biased) in [L_ij, U_ij]
  (`pickRandomDistMat`). Different RNG seed → different conformer → ensemble diversity. (Metrization:
  after fixing one distance, re-smooth, so the chosen distances stay mutually consistent — improves
  sampling; RDKit's default samples independently then minimizes.)
- **EMBED = metric-matrix eigendecomposition (classical MDS).** Distances aren't coordinates. To get
  coordinates:
  1. squared distances D_ij = d_ij².
  2. squared distance of each atom i to the centroid (Havel eq 48; for unit/equal masses):
     D0_i = (1/N) Σ_j D_ij − (1/N²) Σ_{j<k... } D_jk  ≡  (1/N) Σ_j D_ij − (1/(2N²)) Σ_{j,k} D_jk.
     In RDKit code: sqD0i = (1/N)Σ_j D_ij − sumSqD2, where sumSqD2 = (1/N²)Σ_{all i,j} D_ij.
     (Σ over full matrix already double-counts, so sumSqD2 = (1/N²)Σ_{i,j}D_ij = the (1/(2N²))Σ_{j,k}
     of distinct-pair counting × 2. The code's full-matrix sum is the consistent form.)
  3. metric (Gram) matrix centered at centroid: G_ij = ½(D0_i + D0_j − D_ij)  [= ⟨x_i, x_j⟩].
     RDKit: T.setVal(i,j, 0.5*(sqD0i[i] + sqD0i[j] − D_ij)).
  4. eigendecompose G = Σ λ_a v_a v_aᵀ. Coordinates: x_i along axis a = √λ_a · (v_a)_i. Take the top
     3 (largest) eigenvalues → 3D coordinates that best reproduce the distances (classical MDS /
     Young-Householder). RDKit uses `powerEigenSolver` for the top nEigs; coord = sqrt(eig)·eigvec.
     If a needed eigenvalue is negative (distances not exactly Euclidean) → that axis gets random
     jitter; too many zero eigenvalues → reject and resample.
  This is closed-form, no local minima — the key reason DG embedding is robust.
- The raw embedded coords are rough (distances only approximately satisfied, chirality maybe wrong).
  **Cleanup = minimize an error/force field**: distance-violation penalty (push every pair back into
  [L,U]), chiral-volume penalty, and a 4th-dimension penalty (RDKit embeds in 4D so chirality/knots
  can untangle, then squeezes the 4th dim to zero). Classic DG then does MMFF/UFF FF minimization.

## The pain point that motivates ETKDG (grounded in BLOPIG + the method itself)
Plain DG with only triangle-smoothed generic bounds produces conformers that are geometrically valid
but *unrealistic*: torsion angles land in a smeared/uniform distribution instead of the sharp basins
real molecules prefer; aromatic rings/sp2 centers buckle out of plane; structures look "expanded."
Fixing this with a full classical-force-field minimization is expensive and pulls every structure
toward the FF minimum (collapsing diversity / changing the answer). Known fact (from crystallography):
torsion angles in real small-molecule crystal structures (CSD) are NOT uniform — each torsion-motif
SMARTS pattern has a characteristic multimodal preferred-angle distribution.

## The ETKDG idea
Replace generic/uniform geometry with *experimental knowledge*:
- **ET (experimental torsion):** for each rotatable bond matched by a SMARTS torsion pattern, add a
  torsion potential fitted to the CSD histogram of that motif:
    V(φ) = Σ_{n=1..6} V_n · (1 + s_n cos(nφ)),   s_n ∈ {−1,+1}, V_n ≥ 0 fitted per pattern.
  (TorsionPreferences.cpp lines 42-45, exact form.) This biases conformers into experimentally
  preferred torsion basins.
- **K (basic knowledge):** hard chemical facts as extra terms — improper/out-of-plane (inversion)
  terms force sp2 N/O/C with 3 neighbors to be planar; flat-ring proper torsions (4-6-membered all-
  sp2 rings) with a large force constant (100) keep aromatic rings flat; triple bonds kept linear.
  (TorsionPreferences.cpp lines 269-360.)
- These terms enter as a **3D force field applied after the metric-matrix embedding**, alongside the
  distance-violation terms from the bounds matrix (`construct3DForceField`), and the structure is
  minimized (300 steps). For ETKDG the experimental-torsion + basic-knowledge FF *replaces* the need
  for a generic MMFF/UFF cleanup in many cases.
- Result: conformers land in realistic basins directly out of the embedding+ETK-minimize, so you need
  fewer of them and skip (or shorten) classical FF minimization.

## Pipeline (Embedder.cpp), exact order
For each conformer (loop with fresh RNG seed):
1. Build initial bounds matrix from topology (`setTopolBounds`): 1-2, 1-3 exact-ish; 1-4 from torsion
   ranges; VdW lower bounds for the rest. (ETKDG passes the experimental bond/angle info in.)
2. `triangleSmoothBounds` → tighten. If it fails, relax (drop 1-5 bounds, VdW scaling) and retry.
3. `pickRandomDistMat` → sample d_ij ∈ [L,U].
4. `computeInitialCoords` → metric-matrix eigen-embed into (3 or 4) D.
5. `firstMinimization` → distance + chiral + 4th-dim error FF.
6. checkTetrahedralCenters, checkChiralCenters; if chiral or random-coords, `minimizeFourthDimension`.
7. if ET/K: `minimizeWithExpTorsions` → construct3DForceField(distance-violation + experimental-torsion
   M6 + improper/planarity) and minimize 300 steps; planarity check via improper-only FF.
8. accept; optionally prune by RMSD (`pruneRmsThresh`) for diversity.
Then optional MMFF/UFF optimize (`AllChem.MMFFOptimizeMoleculeConfs`).

Default params (rdDistGeom): useExpTorsionAnglePrefs=True, useBasicKnowledge=True, ETversion=2 (v2),
boxSizeMult=2.0, randNegEig=True, numZeroFail=1, enforceChirality=True, optimizerForceTol=0.001,
pruneRmsThresh=-1. ETKDG default since 2018.09; ETKDGv3 default since 2024.03.

## Python API
- `Chem.AddHs(mol)` (Hs needed for real geometry) → `AllChem.EmbedMolecule(mol, params)` /
  `AllChem.EmbedMultipleConfs(mol, numConfs=N, params)`. params = `AllChem.ETKDGv3()` /
  `ETKDGv2()` / `ETKDG()`; set `params.randomSeed`. Optional `AllChem.MMFFOptimizeMoleculeConfs(mol)`.

## Empirical discipline
- Established pre-method facts → context: CSD torsions are non-uniform/multimodal; plain DG gives
  buckled sp2/expanded structures; triangle bounds loose, especially lower; embedding is closed-form
  & local-minimum-free; metrization improves sampling.
- The proposed method's own results (e.g. "% reproduced within 1.0 Å RMSD", runtime ratios from
  BLOPIG) are ETKDG's OWN evaluation outcomes → EXCLUDED from context and reasoning.
- Derive on the page: triangle-inequality bounds; the metric-matrix/eigendecomposition (classical
  MDS) identity; why intermediate generic bounds give smeared torsions.

## Design-decision → why
- Distances not coordinates as the variable: graph gives distances/bounds directly; coordinates would
  need a search. Distance description is rotation/translation invariant. → bounds matrix.
- Triangle smoothing: must, else sampled distance matrix is wildly non-embeddable → garbage / many
  negative eigenvalues. Tighter (tetrangle) too slow (O(N⁴)), marginal gains.
- Sample-then-embed with eigen method: closed-form, no local minima (Havel). Random seed = diversity.
- Embed in 4D not 3D: extra dimension lets chirality/entanglements relax, then collapse 4th dim.
- Metric matrix uses distances-to-centroid (eq 48): error-tolerant averaging, robust to inexact d_ij.
- ET torsion as 6-term cosine: matches the multimodal CSD histograms (need up to 6-fold periodicity);
  fitted per SMARTS motif. K terms as impropers/flat-ring torsions: enforce hard planarity/linearity
  that generic distance bounds don't capture. Applied as post-embed FF so cheap and diversity-safe.
