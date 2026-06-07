OK. I have a molecule as a 2D graph — atoms, bonds, bond orders, stereo flags — and I want 3D
conformers out of it. Plural. Downstream everything (docking, pharmacophore matching, shape
comparison, 3D descriptors) needs real coordinates, and a flexible molecule isn't one shape, it's a
cloud of them, so I want an *ensemble*. Let me think about what "good" even means here before I reach
for any machinery, because the requirements fight each other. The conformers have to be realistic —
sane bond lengths and angles, rings closed, aromatic rings flat, stereocenters correct, torsions
sitting where real molecules put them. They have to be diverse — spanning the accessible space, not
twenty copies of the same folded shape, because the bioactive conformer is often a few kcal/mol off
the global minimum. And they have to be cheap, because I might run this on millions of molecules with
no human in the loop and, crucially, no starting geometry — all I have is the graph.

So the very first obstacle: I have no coordinates at all. If I had a rough 3D structure I could just
minimize a force field and perturb it. I don't. I have connectivity. What do I actually *know* in 3D
from connectivity alone? I know that two bonded atoms sit at a standard bond length. I know that two
atoms bonded to a common third atom sit at a distance fixed by the bond angle — law of cosines on the
two bond lengths and the angle between them. I don't know where these atoms *are*, but I know how far
apart certain pairs are. That's interesting. Distances, not positions.

And distances have a property positions don't: they're invariant to where I put the molecule and how
I rotate it. A conformation really only matters up to rigid motion (and the reflection that flips
chirality, which I'll have to watch). The matrix of all pairwise distances captures the shape and
throws away exactly the rotation/translation degrees of freedom I don't care about. So maybe the
right variable isn't the 3N coordinates — it's the N×N matrix of interatomic distances. Let me commit
to that and see how far it goes.

How much of that matrix does the graph hand me? The 1-2 distances (bonded) — essentially fixed, a
narrow range. The 1-3 distances (two atoms sharing a neighbor) — fixed by the angle, again a narrow
range. The 1-4 distances, across a rotatable bond — *not* fixed, but bounded: as the torsion rotates
from syn to anti the 1-4 distance sweeps between a minimum and a maximum I can compute from the
geometry. And everything farther than that? The only hard fact is that two atoms can't occupy the
same space — a van der Waals lower bound — plus some generous upper bound from the molecule's size.
So I don't get a distance matrix. I get, for each pair (i,j), a *lower bound* L_ij and an *upper
bound* U_ij. An N×N bounds matrix. Fine — that's the object I'll carry.

Now, are those bounds any good? Locally yes, globally no. I read each one off from a little
neighborhood of the graph, ignoring the rest. But the molecule is one rigid-ish object and distances
in any real geometry are tied together. The cleanest tie is the triangle inequality. For any three
atoms i, j, k: d_ij ≤ d_ik + d_kj, and d_ij ≥ |d_ik − d_kj|. The first says I can never be farther
from j than the trip through k, so my upper bound on (i,j) can be lowered to U_ik + U_kj if that's
smaller. The second says I can never be closer to j than the difference of the two legs, so my lower
bound on (i,j) can be raised to L_ik − U_kj (and symmetrically L_kj − U_ik) if that's larger. Wait —
let me get that lower-bound form right, it's the one that's easy to flip. I want the *largest* d_ik
that's still forced, and the *closest* k can be to j; the tightest valid statement is d_ij ≥ d_ik −
d_kj ≥ L_ik − U_kj. Yes: lower bound of one leg minus upper bound of the other. Good.

So I can sweep over all intermediary atoms k and tighten every pair. Does one sweep finish it? No — a
bound I tighten using k might let me tighten another pair using a different intermediary, so I iterate
to convergence. And that pattern — repeatedly relaxing d_ij against d_ik + d_kj over all k — is just
all-pairs shortest paths. The upper bounds are shortest-path lengths in a graph whose edges are the
upper bounds; that's Floyd–Warshall, O(N³). Cheap enough. It also catches contradictions for free: if
smoothing ever forces some L_ij above U_ij, the bounds were inconsistent and I bail. This step — call
it triangle smoothing — is the workhorse. There are tighter inequalities (the four-point /
tetrangle / Cayley–Menger ones), and they do squeeze the bounds further, but they cost O(N⁴) and only
help much on near-coplanar clusters, so I'll leave them on the shelf and live with triangle limits.

I'll sketch the smoothing the way it actually has to run, lower triangle holding lower bounds, upper
triangle holding uppers:

    for k in atoms:
      for i, j:
        U_ij = min(U_ij, U_ik + U_kj)                 # triangle inequality, upper
        L_ij = max(L_ij, L_ik - U_kj, L_kj - U_ik)    # triangle inequality, lower
        if L_ij > U_ij: fail                          # inconsistent bounds

Now I have a tightened box of distance matrices: every entry pinned to [L_ij, U_ij]. A *conformer*
should be one concrete distance matrix from inside that box. So pick one — draw each d_ij uniformly in
its interval. And here's where the ensemble comes from almost for free: a different random draw gives
a different distance matrix gives a different conformer. Diversity is just reseeding the sampler.

But a distance matrix isn't coordinates. I need to turn N(N−1)/2 distances into N points in 3D. This
is the crux, and naively it smells like a hard nonlinear fit: minimize Σ (‖x_i − x_j‖ − d_ij)² over
all coordinates. That objective is full of local minima; if I have to anneal my way out of them for
every single conformer, the whole thing is dead on cost. Let me stare at it. The trouble is the
square root inside ‖·‖. What if I work with squared distances and inner products instead? Inner
products are linear in a Gram matrix, and a Gram matrix has a clean spectral structure. Let me try to
recover *inner products* from distances.

Put the centroid at the origin. Then ⟨x_i, x_j⟩ is what I want (the Gram matrix), and I have
D_ij = ‖x_i − x_j‖² = ‖x_i‖² + ‖x_j‖² − 2⟨x_i, x_j⟩. So ⟨x_i, x_j⟩ = ½(‖x_i‖² + ‖x_j‖² − D_ij). I know
D_ij. I just need ‖x_i‖², the squared distance of each atom to the centroid — and I can get *that*
from the distances too. Average D_ij over j: (1/N)Σ_j ‖x_i − x_j‖² = ‖x_i‖² + (1/N)Σ_j‖x_j‖² −
2⟨x_i, (1/N)Σ_j x_j⟩, and the last term vanishes because the centroid is the origin. So
(1/N)Σ_j D_ij = ‖x_i‖² + (1/N)Σ_j‖x_j‖². The leftover (1/N)Σ_j‖x_j‖² is a constant I can pin down by
averaging once more: it equals (1/(2N²))Σ_{j,k} D_jk (each squared norm shows up symmetrically). So

    ‖x_i‖²  =  (1/N) Σ_j D_ij  −  (1/(2N²)) Σ_{j,k} D_jk  ≡  D0_i,

the squared distance to the centroid, written purely in the D's. That double-averaging is also nicely
forgiving — small errors in individual D_ij wash out. Now the Gram matrix:

    G_ij  =  ½ ( D0_i + D0_j − D_ij ).

And here's the payoff: G is symmetric positive-semidefinite (if the distances were truly 3D) of rank
3, so eigendecompose G = Σ_a λ_a v_a v_aᵀ, and the coordinates are x_i along axis a equal to
√λ_a (v_a)_i. Keep the three largest eigenvalues and I have the 3D coordinates that best reproduce the
distances — and this is *closed form*, no local minima, just an eigendecomposition. That's the thing I
was hoping the square root would let me dodge, and it did. (This is classical multidimensional
scaling; in this corner of chemistry it's *the EMBED algorithm*.) I only need the top few eigenpairs,
so the power method gets them in O(N²) per pass — no need to diagonalize the whole matrix.

What if a sampled distance matrix isn't *exactly* realizable in 3D? Then G isn't rank-3 PSD and one of
the top-3 eigenvalues can come out negative. I don't want to throw the conformer away every time, so
if an eigenvalue is negative I jitter that axis with random values; if too many eigenvalues collapse
to zero (the structure is degenerate, fewer than 3 real dimensions) I give up on that draw and
resample. Reasonable.

Let me also reconsider the sampling, because there's a subtlety. If I draw every d_ij independently
and uniformly, the distances won't be mutually consistent — they'll usually violate the triangle
inequality among themselves even though each lies in its smoothed interval — and the embedding has to
paper over that. It's been observed that independent uniform draws give over-expanded structures that
are oddly similar to each other. The fix (metrization) is to fix one distance, re-smooth the bounds so
the rest stay consistent with that choice, then fix the next, and so on — but that re-smooth is O(N³)
per distance, expensive. A practical compromise: sample, embed, and let the subsequent refinement
absorb the inconsistency, accepting that I'll lean on cleanup.

Because — right — the embedded coordinates are rough. The distances only approximately satisfy all the
bounds at once, and worse, the distance matrix is blind to reflection, so chirality can come out
inverted: a distance description can't tell an R center from an S center. So after embedding I refine
the coordinates against an error function: a penalty for every pair whose distance leaves [L_ij, U_ij]
(push it back in), plus chiral-volume penalties that check the signed tetrahedron volume at each
stereocenter and fix inverted ones. One neat trick for the chirality/entanglement problem: embed in
*four* dimensions and add a penalty on the fourth coordinate. The extra dimension gives a wrong-handed
or knotted structure room to pass through itself and relax continuously; then I squeeze the fourth
coordinate to zero and I'm back in 3D, untangled. After that I can run a real molecular-mechanics
minimization — MMFF94 or UFF — to settle into a clean local minimum.

Let me step back and ask whether this is actually good enough, because so far I've only used *generic*
geometry: standard bonds, standard angles, torsion *ranges*, van der Waals radii. Picture what comes
out. Bond lengths fine, angles fine, no atomic clashes. But the torsions — I bounded each 1-4 distance
by the full syn-to-anti sweep and then sampled uniformly in the box. So the torsion angles in my
ensemble come out roughly *uniformly distributed*. Now compare that to reality. Real molecules don't
put their torsions uniformly — they sit in sharp preferred basins. A C–C single bond between sp3
centers wants gauche/anti; an amide stays planar near 0 or 180; an aryl–aryl bond has its own
preferred twist. Crystallographers have measured exactly this: pull every instance of a given local
torsion motif out of the Cambridge Structural Database and histogram its angle, and you get a
characteristic, sharply multimodal distribution — not a uniform smear. My generic-bounds ensemble
reproduces none of that structure. The conformers are geometrically legal and physically smeared.

And there's a second leak. Nothing in a pairwise van der Waals *lower* bound says an aromatic ring has
to be flat, or that an sp2 carbon's three substituents are coplanar with it, or that a triple bond is
linear. Those are facts about local hybridization, not about any single interatomic distance, so my
bounds matrix simply doesn't encode them. The raw embedding is free to pucker an aromatic ring or bend
an sp2 center out of plane, and it will, because I gave it that slack.

I could try to fix both by leaning harder on the final force field — minimize MMFF and let it pull the
torsions into its wells and flatten the rings. But that's exactly the move I want to avoid. It's
expensive on every conformer, and a thorough minimization drags every structure down toward the force
field's nearest minimum, which collapses the diversity I worked to create and, worse, replaces "what
crystallography says about torsions" with "what this particular force field says." I'd be paying a lot
to make my ensemble *less* diverse and only as realistic as MMFF's torsion parameters. Wrong trade.

So the real question is: how do I inject the knowledge — preferred torsions, planarity — *cheaply* and
without flattening diversity? The knowledge I'm missing is empirical and pattern-specific: it's the
CSD torsion histograms and a handful of hard chemical rules. Let me bring those in directly rather
than hope a generic force field reconstructs them.

For torsions: classify each rotatable bond by its local environment — a SMARTS pattern over the four
torsion atoms and their neighbors — and attach to each pattern a potential fitted to that motif's CSD
angle histogram. A torsion potential has to be periodic in φ and, to match a multimodal histogram with
up to several preferred values, it needs harmonics up to sixfold. The natural form is a cosine series:

    V(φ)  =  Σ_{n=1..6}  V_n · ( 1 + s_n cos(n φ) ),     s_n ∈ {−1, +1},  V_n ≥ 0,

where, for each SMARTS pattern, the amplitudes V_n and signs s_n are fit so the minima of V land on
the peaks of the experimental histogram. Sixfold periodicity is enough to carve out the
two-, three-, and higher-fold preferences seen in the data; the s_n flip whether each harmonic favors
the staggered or eclipsed positions. This is a genuinely experimental term — its wells are where
*crystals* put that torsion, not where a transferable force field guesses.

For the hard rules — call it "basic knowledge": add out-of-plane (improper / inversion) terms that
penalize an sp2 N, O, or C with three neighbors for leaving its plane, forcing the planarity that no
pairwise distance bound captures; add proper-torsion terms on the bonds of small all-sp2 rings
(4-, 5-, 6-membered) with a large force constant so aromatic and conjugated rings come out flat; keep
triple bonds linear. These are facts, so they get stiff terms.

Now *where* do these terms act? Two choices. I could try to fold torsion preferences back into the
distance bounds — and the experimental bond/angle information does sharpen the initial bounds matrix —
but a multimodal torsion preference can't be expressed as a single [L,U] interval on a 1-4 distance; a
sharp double-welled preference isn't an interval at all. So the torsion and planarity knowledge has to
act *after* the embedding, as a force field on the 3D coordinates. That's cheap: it's a short
minimization of a handful of torsion and improper terms (plus the distance-violation terms that keep
the bounds satisfied), not a full classical force field, and because its wells are the experimental
torsion basins, a few hundred steps slide each conformer into a realistic basin without homogenizing
the ensemble — different draws started in different basins and stay there.

So the pipeline becomes: build the bounds matrix, now *informed* by experimental bond/angle data;
triangle-smooth; sample a distance matrix; metric-matrix embed (in 4D, for the chirality untangling);
do the first error-function refinement (distance + chiral + fourth-dimension); then a second
minimization with the *experimental-torsion* terms and the *basic-knowledge* planarity/linearity
terms, alongside the distance-violation terms. Reseed for the next conformer. And after that I can
often *skip* the heavy MMFF/UFF cleanup entirely, because the conformer already lands in an
experimentally-preferred basin — which is the whole point: fewer conformers needed, no expensive
generic minimization, and diversity preserved. This is experimental-torsion knowledge distance
geometry — ET for the torsion histograms, K for the basic-knowledge rules.

Let me write it the way it actually runs in RDKit. First the engine — bounds, smoothing, sampling,
embedding, refinement — and then the experimental-knowledge terms layered on top.

```python
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DistanceGeometry

# ---- the part I can do from generic + experimental geometry: bounds -> embed ----
def show_distance_geometry_core(mol):
    mol = Chem.AddHs(mol)                      # Hs matter for real geometry
    N = mol.GetNumAtoms()

    bounds = AllChem.GetMoleculeBoundsMatrix(mol)   # 1-2,1-3 tight; 1-4 from torsion range; VdW else
    DistanceGeometry.DoTriangleSmoothing(bounds)    # triangle inequality: O(N^3) all-pairs

    # sample one distance matrix from the smoothed box: each d_ij uniform in [L_ij, U_ij]
    # embed via the metric (Gram) matrix:
    #   D_ij  = d_ij**2
    #   D0_i  = (1/N) sum_j D_ij - (1/(2 N^2)) sum_{j,k} D_jk        # squared dist to centroid
    #   G_ij  = 0.5 * (D0_i + D0_j - D_ij)                          # inner products <x_i, x_j>
    #   eigendecompose G; coords along axis a = sqrt(lambda_a) * v_a; keep top 3 (or 4)
    # negative needed-eigenvalue -> jitter that axis; too many zeros -> resample
    return bounds
```

The metric-matrix step is exactly the eigendecomposition I derived; in the C++ engine it reads as
this (squared distances → squared distances to centroid → Gram matrix T → top eigenvalues/vectors →
coordinates scaled by √λ):

```cpp
// for each i: sqD0i[i] = (1/N) sum_j D_ij - sumSqD2     // sumSqD2 = (1/N^2) sum_{i,j} D_ij
for (i = 0; i < N; i++)
  for (j = 0; j <= i; j++)
    T.setVal(i, j, 0.5 * (sqD0i[i] + sqD0i[j] - sqMat.getVal(i, j)));   // Gram matrix G_ij
powerEigenSolver(nEigs, T, eigVals, eigVecs, ...);                      // top-k eigenpairs
for (j = 0; j < dim; j++)
  if (eigVal[j] > 0) eigVal[j] = sqrt(eigVal[j]);                       // scale = sqrt(lambda)
// coord:  (*pt)[j] = eigVal[j] * eigVecs.getVal(j, i);   (negative eig -> random jitter)
```

And the triangle smoothing is the bound-tightening loop, upper triangle = upper bounds, lower = lower:

```cpp
// for each intermediary k, over pairs (i,j):
U_ij = min(U_ij, U_ik + U_kj);                       // d_ij <= d_ik + d_kj
L_ij = max(L_ij, L_ik - U_kj, L_kj - U_ik);          // d_ij >= |d_ik - d_kj|
if (L_ij - U_ij > 0.) return false;                  // inconsistent -> reject
```

Now the experimental knowledge. Each torsion-motif SMARTS carries the fitted cosine series

```
// V(phi) = sum_{n=1..6} V_n * (1 + s_n * cos(n*phi)),  per-pattern [s1,V1,...,s6,V6] from CSD
```

and the basic-knowledge rules add improper/out-of-plane terms on 3-coordinate sp2 N/O/C (planarity)
and stiff proper torsions on 4–6-membered all-sp2 rings (flatness; force constant 100). These become a
3D force field built on the embedded coordinates — distance-violation terms (keep the bounds) +
experimental-torsion terms + improper terms — and minimized:

```cpp
// minimizeWithExpTorsions: built from the bounds matrix + the CSD torsion + planarity terms
field.reset(construct3DForceField(mmat, positions3D, etkdgDetails));   // ET + K terms
field->minimize(300, optimizerForceTol);                               // slide into experimental basins
// (a separate impropers-only field checks planarity passed)
```

Putting the whole per-conformer loop together, the way it's driven:

```cpp
gotCoords = generateInitialCoords(positions, eargs, params, distMat, rng); // sample + metric embed
gotCoords = firstMinimization(positions, eargs, params);                   // dist + chiral + 4th-dim
gotCoords = checkTetrahedralCenters(...); checkChiralCenters(...);         // stereo sane?
gotCoords = minimizeFourthDimension(...);                                  // squeeze 4th dim
if (useExpTorsionAnglePrefs || useBasicKnowledge)
  gotCoords = minimizeWithExpTorsions(*positions, eargs, params);          // ET + K basins
```

And the user-facing call — add hydrogens, pick the experimental-torsion-knowledge parameter set,
embed many conformers with distinct seeds, optionally (no longer always) finish with MMFF:

```python
mol = Chem.AddHs(Chem.MolFromSmiles("CN1CCC(O)(CC1)c1ccccc1Cl"))
params = AllChem.ETKDGv3()                 # exp-torsion prefs + basic knowledge on
params.randomSeed = 0xf00d
cids = AllChem.EmbedMultipleConfs(mol, numConfs=50, params=params)   # one embed per seed
# AllChem.MMFFOptimizeMoleculeConfs(mol)   # optional now, not required
```

The causal chain, start to end: I can't search 3N coordinates from nothing, but the graph hands me
interatomic distances and bounds, which are rotation-invariant — so I carry an N×N bounds matrix;
the bounds are locally loose, so the triangle inequality (all-pairs shortest paths) tightens them and
flags contradictions; a conformer is one distance matrix sampled from the smoothed box, and reseeding
the sampler gives the ensemble; a distance matrix becomes coordinates not by a local-minima-ridden fit
but in closed form, by reading the top three eigenvectors of the centroid-centered Gram matrix
(classical MDS); the rough embedding is refined against distance and chirality penalties, with a
fourth dimension to untangle handedness; but generic bounds leave the torsions smeared and the sp2
fragments buckled, and leaning on a full force field to fix that is expensive and collapses diversity;
so I inject the missing knowledge directly — per-motif torsion potentials V(φ)=Σ V_n(1+s_n cos nφ)
fitted to CSD crystal histograms, plus improper/flat-ring terms for planarity and linearity — as a
short post-embedding minimization, which slides each conformer into an experimentally-preferred basin
cheaply and without homogenizing the ensemble, so the heavy classical minimization can usually be
dropped.
