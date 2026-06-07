# Context: from a 2D molecular graph to 3D conformers (the distance-geometry landscape)

## Research question

A molecule arrives as a 2D graph: a list of atoms and the bonds between them, with bond orders,
formal charges, stereochemistry, but no coordinates. Almost everything one wants to *do* with a
molecule downstream — dock it into a protein pocket, match it against a pharmacophore, compare its
3D shape to another molecule, compute a 3D descriptor, feed it to a property model — needs actual
Cartesian coordinates, and not just one set: a flexible molecule populates many conformations, and
the relevant one for binding is rarely the single lowest-energy gas-phase structure. So the problem
is to take the graph and emit an *ensemble* of 3D conformers.

The ensemble has to satisfy three things at once, and they pull against each other. It must be
**realistic** — bond lengths and angles right, rings closed and (where appropriate) flat, chirality
correct, and torsions sitting in the basins that real molecules actually occupy. It must be
**diverse** — genuinely covering the accessible conformational space, not collapsing to one family of
near-identical structures, because the conformer you need might be a few kcal/mol above the minimum.
And it must be **cheap** — a virtual screen touches millions of molecules, so per-molecule cost of
seconds, not minutes, with no human babysitting and no dependence on a good starting geometry (there
*is* no starting geometry; all you have is the graph). A method that produces realistic-but-collapsed
ensembles, or diverse-but-distorted ones, or correct-but-slow ones, fails. The open problem is a
generator that hits all three.

## Background

**A conformation can be described entirely by interatomic distances and chirality.** This is the
foundational observation of distance geometry (Crippen, Havel; reviewed in Havel, *Distance Geometry:
Theory, Algorithms, and Chemical Applications*, and Blaney & Dixon, *Distance Geometry in Molecular
Modeling*, Rev. Comp. Chem.). A set of 3D points is determined, up to overall rotation/translation
and reflection, by the matrix of pairwise distances together with the signs of selected tetrahedral
volumes (chirality). Distances are invariant under rigid motion, which is exactly the right
invariance for a conformation. So instead of searching directly in 3N Cartesian coordinates, one can
reason about the N×N distance matrix.

**The graph fixes some distances and bounds the rest.** Standard bond lengths fix every 1-2 distance
(bonded pair) to a narrow range. Standard bond angles, combined with the two bond lengths, fix every
1-3 distance (two atoms sharing a common neighbor) by the law of cosines. 1-4 distances (across a
rotatable bond) are not fixed but are bounded: as the torsion swings from cis to trans the 1-4
distance ranges between a computable minimum and maximum. For all more-distant pairs, the only firm
statement is that two atoms cannot interpenetrate — a van der Waals lower bound — and an upper bound
from the molecule's overall extent. The natural object is therefore an N×N **bounds matrix** holding,
for each atom pair (i,j), a lower bound L_ij and an upper bound U_ij on their distance.

**The bounds are loose and must be tightened — the triangle inequality.** Bounds read off locally
from the graph ignore global consistency. But distances in any metric space obey the triangle
inequality: for any third atom k, d_ij ≤ d_ik + d_kj and d_ij ≥ |d_ik − d_kj|. This means an upper
bound can be lowered, U_ij ← min(U_ij, U_ik + U_kj), and a lower bound raised,
L_ij ← max(L_ij, L_ik − U_kj, L_kj − U_ik), using any intermediary k. Iterating to convergence over
all triples is equivalent to an all-pairs-shortest-paths computation on a graph whose edge lengths are
the bounds (Floyd–Warshall, O(N³)); Havel works this out as the "triangle inequality limits." It is
the cheap, reliable, and most important smoothing step; it also detects contradictions (a lower bound
forced above an upper bound). Tighter limits exist — the tetrangle (four-point Cayley–Menger)
inequalities — but cost O(N⁴) and yield large improvements only on near-coplanar atom subsets, so in
practice they are seldom used.

**From a distance matrix to coordinates: the metric-matrix (EMBED) algorithm.** Given an actual
distance matrix, coordinates that best reproduce it can be recovered in closed form, with no local
minima — this is classical multidimensional scaling (Young–Householder), and in this field it is *the
EMBED algorithm* (Havel §3.2). The steps: from squared distances D_ij = d_ij², compute the squared
distance of each atom to the centroid,

  D0_i = (1/N) Σ_j D_ij − (1/N²) Σ_{j,k} D_jk,

an averaging that is notably tolerant of errors in the input distances; form the centered
inner-product (Gram / "metric") matrix

  G_ij = ½ (D0_i + D0_j − D_ij)   (= ⟨x_i, x_j⟩ for centroid-centered coordinates);

and diagonalize G = Σ_a λ_a v_a v_aᵀ. Coordinates along principal axis a are then √λ_a · (v_a)_i, and
keeping the three largest eigenvalues gives the 3D coordinates that best fit the distances. Only the
top few eigenpairs are needed, so the power method suffices (O(N²) per pass). If the sampled distances
are not exactly realizable in 3D, some "needed" eigenvalue can come out negative; the standard
remedies are to jitter that axis randomly or reject and resample.

**Diversity comes from sampling distances; multiple seeds give multiple conformers.** Picking a
specific distance matrix from within the smoothed bounds — each d_ij drawn in [L_ij, U_ij] — and then
embedding gives one conformer; a different random draw gives another. It was observed early that
drawing each distance independently produces structures that are over-expanded and more similar to one
another than they should be; *metrization* (fix one distance, re-smooth the bounds so the rest stay
mutually consistent, repeat) markedly improves the sampling, at O(N³) cost.

**The embedded structure is rough and needs cleanup.** Sampled distances only approximately satisfy
all bounds simultaneously, and chirality can come out inverted (the distance matrix is blind to
reflection). So the embedded coordinates are refined against an error function: a penalty for every
pair whose distance leaves [L_ij, U_ij], plus chiral-volume penalties to fix stereocenters. A useful
trick is to embed in *four* dimensions and penalize the fourth coordinate: the extra room lets
tangled or wrong-handed structures relax continuously before the fourth dimension is squeezed out.
After this, a classical molecular-mechanics minimization (MMFF94 or UFF) can be run to give a clean
local-minimum geometry.

**Known limitations of generic-bounds distance geometry — the pre-method facts that matter.** Distance
geometry with only generic, triangle-smoothed bounds is geometrically valid but physically smeared.
Two well-documented symptoms: (1) torsion angles come out roughly uniformly distributed, whereas the
torsions of real molecules are sharply *non-uniform* — small-molecule crystal structures in the
Cambridge Structural Database (CSD) show, for each local torsion motif, a characteristic multimodal
preferred-angle distribution, and generic bounds reproduce none of that; (2) planar fragments buckle —
nothing in a pairwise van der Waals lower bound forces an aromatic ring or an sp2 center to stay flat,
or a triple bond to stay linear, so the raw embedding lets them pucker. The usual fix, a full
classical-force-field minimization, is expensive and drags every structure toward the force-field
minimum, eroding the very diversity the ensemble was meant to provide.

## Baselines

**Classical distance geometry (the EMBED pipeline; Crippen, Havel; Blaney & Dixon).** Build the bounds
matrix from generic bond/angle/torsion ranges and van der Waals radii; triangle-smooth; sample a
distance matrix; metric-matrix embed; refine against the distance/chirality error function; optionally
MMFF/UFF minimize. Core math as above. *Gap:* generic bounds encode no information about which
torsions or ring geometries are actually preferred, so conformers are realistic only after an
expensive force-field cleanup, and even then the torsion distribution is whatever the force field
happens to give, not what crystallography says. It also tends to over-expand structures and
under-sample.

**Systematic / rule-based torsion enumeration (e.g. knowledge-based template builders).** Fragment the
molecule, place each fragment from a library of preferred geometries, and enumerate combinations of
preferred torsion values. *Gap:* relies on a curated fragment/template library and combinatorial
enumeration; coverage is bounded by the library, and stitching fragments while respecting ring
closure and long-range clashes is awkward. It does, however, make the point that *experimental torsion
preferences are the missing ingredient* — a point a distance-geometry method could absorb directly.

**Force-field / stochastic search (e.g. low-mode or random-start MMFF minimization).** Start from
random or perturbed coordinates and minimize/sample on a molecular-mechanics surface. *Gap:* needs
starting coordinates, is prone to local minima, and is slow per conformer; the energy surface's
torsion preferences are those of the force field, which may not match experiment.

## Evaluation settings

The natural yardstick is reproduction of *experimentally determined* 3D structures. Two standard
sources of ground-truth small-molecule geometries existed: small-molecule crystal structures from the
Cambridge Structural Database (CSD), and ligand geometries extracted from protein–ligand complexes in
the Protein Data Bank (PDB). For a molecule whose graph is taken from such a structure, one generates
an ensemble and measures how well some generated conformer matches the experimental geometry, by
heavy-atom RMSD after optimal alignment (e.g. the minimum RMSD over the ensemble, as a function of
ensemble size), alongside the wall-clock cost per molecule. The CSD torsion-angle histograms — the
empirical distributions of each torsion motif across many crystal structures — are themselves a
pre-existing resource that any knowledge-based method would draw on.

## Code framework

The primitives that already exist: an RDKit molecule from a graph, explicit-hydrogen handling, a
bounds-matrix class, a triangle-smoothing routine, the metric-matrix embedder, an error-function
minimizer over distance/chirality terms, and the MMFF/UFF force fields. The slot to be filled is *how
the per-pair bounds and the post-embedding refinement are informed* — i.e. what knowledge, beyond
generic geometry, shapes the conformers. Sketch:

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DistanceGeometry

def embed_conformer(mol, seed):
    mol = Chem.AddHs(mol)                       # Hs needed for realistic geometry
    N = mol.GetNumAtoms()

    bounds = make_initial_bounds_matrix(mol)    # 1-2, 1-3 tight; 1-4 from torsion range; VdW else
    # TODO: what additional knowledge shapes the bounds / refinement?
    DistanceGeometry.DoTriangleSmoothing(bounds)

    dmat = sample_distance_matrix(bounds, seed) # each d_ij uniform in [L_ij, U_ij]
    coords = metric_matrix_embed(dmat, dim=4)   # classical MDS: top eigenvectors of Gram matrix

    refine_against_error_function(coords, bounds)   # distance + chiral + 4th-dim penalties
    # TODO: a post-embedding refinement that injects the knowledge above
    return coords

def make_initial_bounds_matrix(mol):
    # bonds -> 1-2 bounds; angles -> 1-3 bounds; rotatable bonds -> 1-4 bounds; VdW lower bounds
    # TODO
    pass

def metric_matrix_embed(dmat, dim):
    D = dmat ** 2                               # squared distances
    # D0_i = (1/N) sum_j D_ij - (1/N^2) sum_{j,k} D_jk      (squared dist to centroid)
    # G_ij = 0.5 * (D0_i + D0_j - D_ij)          (Gram / metric matrix)
    # eigendecompose G; coords along axis a = sqrt(lambda_a) * v_a ; keep top `dim`
    pass

def refine_against_error_function(coords, bounds):
    # penalty for d_ij outside [L_ij, U_ij]; chiral-volume penalties; squeeze 4th dimension
    pass

# multiple seeds -> ensemble; optional final MMFF/UFF minimization for clean local minima
```

The two `# TODO`s are the same slot seen twice: the knowledge that turns a geometrically valid
embedding into a *realistic* one. The final code fills exactly these in.
