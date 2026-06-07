# Explainer captures

## BLOPIG "Advances in Conformer Generation: ETKDG and ETDG" (2016)
- ETKDG/ETDG build on classical Distance Geometry, converting molecular topology to 3D coords.
- ETDG = DG + knowledge of preferred torsional angles derived from experimental crystal structures.
- ETKDG = ETDG + chemical-knowledge constraints ("aromatic rings are flat", "bonds connected to triple
  bonds are colinear"), i.e. extra geometric restrictions beyond torsion preferences.
- [OUT OF SCOPE — proposed-method results, NOT for context/reasoning] reported coverage/RMSD numbers
  and runtime ratios vs DG/CONFECT; tested on CSD small-molecule + PDB protein-ligand sets.

## RDKit Getting Started (Python) — DG pipeline RDKit lists
1. distance bounds matrix from connection table + rules.
2. smooth bounds via triangle-bounds smoothing.
3. random distance matrix within bounds.
4. embed distance matrix into 3 (or 4) D → coordinates per atom.
5. clean up coords with a crude force field + the bounds matrix.
ETKDG uses CSD torsion-angle preferences (SMARTS + potential terms) to correct conformers after DG.
Hs should be added before embedding for realistic geometry.
API: AllChem.EmbedMolecule(m), AllChem.EmbedMultipleConfs(m, numConfs=10),
params = AllChem.ETKDGv3(); params.randomSeed=...; EmbedMultipleConfs(m, numConfs=10, params=params).

## RDKit blog (greglandrum) — ETKDG and distance constraints
- Distance constraints implemented by editing bounds matrix entries then DoTriangleSmoothing(bounds).
- ETKDG adds torsion + basic-knowledge terms to the distance-geometry force field that optimize the
  embedded structure; these terms can override bounds, so constraints "aren't strict" under ETKDG.

## Torsion potential (CrystalFF TorsionPreferences.cpp, exact form)
V = V1(1+s1 cos1φ) + V2(1+s2 cos2φ) + V3(1+s3 cos3φ) + V4(1+s4 cos4φ) + V5(1+s5 cos5φ) + V6(1+s6 cos6φ)
per-SMARTS [pattern, s1,V1,...,s6,V6], fitted to CSD histograms. Flat-ring torsion: signs=1 except
s2=-1, V2=100. Basic knowledge: improper/out-of-plane (inversion) terms on 3-coordinate sp2 N/O/C.
