# Force-field torsion (dihedral) parameter fitting

## Problem

A classical force field writes a molecule's energy as bonds + angles + torsions + nonbonded
(Lennard-Jones + Coulomb, with 1–4 interactions scaled). The bond, angle, and nonbonded terms do
not, on their own, reproduce the barrier to internal rotation about a bond — conjugation,
hyperconjugation, and the imperfect 1–4 nonbonded description leave a residual barrier. The torsion
term exists to carry exactly that residual. Given a high-level quantum-mechanical (QM) rotational
profile of a molecule, the task is to fix the coefficients of the torsion term so that, added to
the rest of the force field, the model reproduces the QM barrier.

## Key idea

1. **Target = residual, not the raw QM barrier.** Run a *relaxed* QM torsion scan: constrain the
   dihedral phi to a grid over -180..180 degrees, optimize all other coordinates, record E_QM(phi).
   Evaluate those same scanned conformations in the MM engine with **all torsion parameters set to
   zero**, giving E_MM_without_torsion(phi). The torsion term must reproduce only what the rest of
   the force field misses:

       DeltaE(phi) = E_QM(phi) - E_MM_without_torsion(phi).

2. **The torsion term is a cosine Fourier series, linear in its amplitudes.** Rotation is periodic,
   so expand in harmonics whose periodicities are chemically meaningful (n=3 for sp3–sp3 staggering,
   n=2 for conjugation/planarity, n=1 for gauche/anti asymmetry):

       E_tors(phi) = sum_n (V_n/2) [1 + cos(n phi)],   write K_n = V_n/2.

   Pinning the phase to its natural values {0, 180} is the same as letting K_n take either sign
   (cos(n phi - 180) = -cos(n phi)), which removes the only nonlinear parameter. The model is then
   **linear** in the K_n.

3. **Fitting is linear least squares -> a single linear solve.** Minimize, over the N scanned
   conformations, the squared mismatch F(K) = sum_k ( sum_n K_n (1+cos(n phi_k)) - DeltaE_k )^2.
   Setting dF/dK_m = 0 gives the normal equations

       M K = B,   M_{mn} = sum_k a_{mk} a_{nk},   B_m = sum_k DeltaE_k a_{mk},   a_{nk} = 1 + cos(n phi_k),

   solved in closed form as K = M^{-1} B. No iterative optimizer; M is the (symmetric, PSD) Gram
   matrix of the cosine basis over the scan.

4. **Transferable types share parameters.** All dihedrals with the same four-atom type quadruplet
   (and its reverse) share one K set; the design entry for a type sums the cosine basis over that
   type's instances, A_{ikn} = sum_{l in type i} (1 + cos(n phi_{ikl})). The normal equations keep
   the same form with the index (type, harmonic).

5. **Control overfitting.** Restrict the allowed periodicities, or admit n=1..4 and regularize
   (L1/Lasso drives redundant amplitudes to zero), or bound the amplitudes and solve a bounded
   residual problem for the assembled linear system. Optionally weight scan points (Boltzmann
   exp(-DeltaE/kT), or flat-then-attenuating) to emphasize populated conformers.

6. **Export.** Convert the fitted Fourier amplitudes to Ryckaert–Bellemans coefficients (psi=phi-180,
   E = sum_{m=0}^5 C_m cos(psi)^m) exactly:

       C0 = K1+K3+2K4,  C1 = -K1+3K3,  C2 = 2K2-8K4,  C3 = -4K3,  C4 = 8K4,  C5 = 0.

## Algorithm

```
input: matched scan of N conformations
       E_QM[k]       : relaxed QM energies (relative)
       E_MM_without_torsion[k] : MM no-torsion energies for the same conformations
       phi[k, l]     : angle of dihedral l at conformation k, tagged with its atom-type quad
1. DeltaE[k] = E_QM[k] - E_MM_without_torsion[k]            # residual = fit target
2. group dihedrals into types (quad == reverse-quad)
3. for each conformation k: build A_k[(type i, harmonic n)] = sum_{l in type i}(1+cos(n*phi[k,l]))
4. accumulate M += outer(A_k, A_k);  B += DeltaE[k] * A_k
5. K = inv(M) @ B                                           # all types/harmonics at once
6. (optional) bounded solve lsq_linear(M, B, bounds);  (optional) Fourier -> RB
```

## Code

```python
import os, json
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.optimize import lsq_linear

def fourier_to_RB(k1, k2, k3, k4=0.0):
    """Exact Fourier (sum_n k_n (1+cos n phi)) -> Ryckaert-Bellemans (psi = phi - 180)."""
    return (k1 + k3 + 2*k4, -k1 + 3*k3, 2*k2 - 8*k4, -4*k3, 8*k4, 0.0)

def _convert_result(fourier, coefficient_type):
    if coefficient_type == "RB":
        return {key: list(fourier_to_RB(*values)) for key, values in fourier.items()}
    return fourier

def estimate_rotational_coefficients(qm_file, mm_file, dih_dir, coeff=4,
                                      coefficient_type="Fourier", bounds=None):
    """Fit rotational Fourier coefficients from matched scan files.

    qm_file: first column is relaxed QM relative energy.
    mm_file: second column is no-torsion MM relative energy for the same conformations.
    dih_dir: *.dih files; first line is the atom-type quadruplet, followed by angle rows.
    """
    qm = pd.read_csv(qm_file, sep=r"\s+", header=None)[0]
    zero_term = pd.read_csv(mm_file, sep=r"\s+", header=None)[1]
    target = (qm - zero_term).to_numpy()

    dih_files = [f for f in os.listdir(dih_dir) if f.endswith(".dih")]
    dih_types = {f: open(os.path.join(dih_dir, f)).readline().strip() for f in dih_files}
    groups = {}
    for f, t in dih_types.items():
        rev = " ".join(t.split()[::-1])
        key = t if (t in groups or rev not in groups) else rev
        groups.setdefault(key, []).append(f)

    sizes = [len(v) for v in groups.values()]
    file_order = [f for v in groups.values() for f in v]
    type_order = list(groups.keys())
    angles = pd.concat([pd.read_csv(os.path.join(dih_dir, f)) for f in file_order], axis=1)

    total = coeff * len(sizes)
    M = np.zeros((total, total))
    B = np.zeros(total)
    for k, (_, row) in enumerate(angles.iterrows()):
        basis = np.array([[1 + np.cos(n * angle * np.pi / 180.0)
                           for n in range(1, coeff + 1)]
                          for angle in row])
        A_k, offset = [], 0
        for size in sizes:
            A_k.append(basis[offset:offset + size].sum(axis=0))
            offset += size
        A_k = np.concatenate(A_k)
        M += np.outer(A_k, A_k)
        B += target[k] * A_k

    K = inv(M) @ B
    fourier = {type_order[i]: list(K[coeff*i:coeff*i + coeff]) for i in range(len(type_order))}
    result = {"best": _convert_result(fourier, coefficient_type)}

    if bounds is not None:
        lo, hi = bounds
        bounded = lsq_linear(M, B, bounds=([lo]*total, [hi]*total)).x
        fourier_bounded = {
            type_order[i]: list(bounded[coeff*i:coeff*i + coeff])
            for i in range(len(type_order))
        }
        result["bounded"] = _convert_result(fourier_bounded, coefficient_type)

    return result
```

The fit returns one amplitude set per dihedral type; adding `sum_n K_n (1 + cos(n phi))` back into
the force field reproduces the residual barrier the bond/angle/nonbonded terms could not. The
solution is the exact global least-squares optimum, obtained without any iterative optimizer.
