A classical molecular-mechanics force field writes a molecule's potential energy as a sum of cheap, interpretable pieces: harmonic springs for bonds and angles, a Lennard-Jones-plus-Coulomb sum over nonbonded pairs, and an explicit term for rotation about single bonds. What bothers me is that I trust nearly all of these terms and distrust exactly one. A C–C stretch constant is a C–C stretch constant; bond and angle springs are local and stiff and transfer between molecules with negligible error, and the nonbonded parameters, fit to bulk and dimer data, transfer too. But the energy of turning one part of a molecule relative to another about a rotatable bond — the torsion profile as a function of a dihedral angle $\phi$ — is the part the model keeps getting wrong, and it is wrong for a structural reason. As $\phi$ sweeps from $-180$ to $180$ degrees, the barrier is shaped not only by sterics (atoms three bonds apart on either end of the bond clashing and relaxing) but by genuinely electronic effects: conjugation penalizes twisting out of planarity when there is $\pi$ character across the bond, and hyperconjugation favors staggered over eclipsed in a plain alkane. A harmonic angle spring knows nothing of these, and the nonbonded sum captures the steric part only approximately — the "1–4" pairs that straddle the rotatable bond are precisely the ones a point-charge-plus-LJ model describes worst, which is why those interactions are scaled down by a fixed factor in the first place. So there is real leftover barrier the rest of the force field structurally cannot produce, and a dedicated torsion term must carry it.

The truth to match is quantum mechanics: constrain $\phi$ to each value on a grid (steps of ten to fifteen degrees over the full circle), let every other degree of freedom relax under QM, and read off the energy, giving the *relaxed* profile $E_{\mathrm{QM}}(\phi)$. The scan must be relaxed rather than rigid — freezing all other coordinates while twisting would measure a mixture of the torsion barrier and artificial bond/angle strain that does not transfer to how the molecule actually moves. The tempting next step is to make the torsion term reproduce $E_{\mathrm{QM}}(\phi)$ directly, and that is wrong because it double-counts: the force field already computes bond, angle, and nonbonded energy along the same scan, and the torsion term is an *addition* on top. The prior baselines all understood the residual idea but stumbled on the fit. The original OPLS-AA parameterization adjusted $V_1, V_2, V_3$ by hand against RHF profiles of many small molecules — per-molecule, partly manual, with no guarantee of a global least-squares optimum. AMBER keeps an explicit phase $\gamma_n$ in each term, which makes the model nonlinear in its parameters and forces an iterative optimizer with local minima. Modern ForceBalance-style pipelines pose the fit as minimizing a weighted sum of squared QM-vs-MM deviations and descend it with steepest descent or Levenberg–Marquardt — slower than necessary, dependent on initialization and step size, and able to stop short of the true minimum even when only amplitudes are free.

What I propose is torsion fitting reduced to a single linear solve, which I will call torsion fitting by linear least squares over the residual barrier. The first move is to construct the honest target. I take the same scanned conformations, in the same order, run them through the MM engine with all torsion amplitudes set to zero, and call the result $E_{\mathrm{MM\text{-}no\text{-}torsion}}(\phi)$. The cleanest way to obtain this is to take the production topology and literally zero out every torsion parameter, then re-evaluate the scan, so the 1–4 scaling and every other term are treated exactly as they will be in production — if the torsion term is going to absorb the slop in the 1–4 interactions, it must absorb the slop that is actually there. The quantity the torsion term must reproduce is then the residual

$$\Delta E(\phi) = E_{\mathrm{QM}}(\phi) - E_{\mathrm{MM\text{-}no\text{-}torsion}}(\phi).$$

The second move is the functional form. Rotation is periodic, $E_{\mathrm{tors}}(\phi + 2\pi) = E_{\mathrm{tors}}(\phi)$, so I expand in a cosine Fourier series whose harmonics carry chemical meaning known before any fit: $n = 3$ for the three staggered minima of an sp3–sp3 bond, $n = 2$ for the two-minima cis/trans shape of a conjugated or planarity-preferring bond, $n = 1$ for any asymmetry between gauche and anti, with little gained above $n = 4$ for a single bond. I write

$$E_{\mathrm{tors}}(\phi) = \sum_n \frac{V_n}{2}\,[\,1 + \cos(n\phi)\,],$$

where the $1+$ is a per-term constant offset that gives each piece a clean zero and shifts the baseline without changing the shape. The crucial design decision is the phase. AMBER's general $\cos(n\phi - \gamma_n)$ is more flexible, but $\cos(n\phi - \gamma)$ is trigonometric, not linear, in $\gamma$; the instant $\gamma$ is free the fit becomes nonlinear, with all the iterative machinery and local-minimum worries that follow. Before committing to that, I check whether the free phase is needed. For most torsions the minima sit at the symmetric places — $0, 60, 120, 180$ — so the natural phase is either $0$ or $180$ degrees, and $\cos(n\phi - 180) = -\cos(n\phi)$. Flipping the phase between its two natural values is therefore, up to the baseline I am free to re-zero anyway, identical to flipping the sign of $V_n$. So if I let each amplitude range over all reals, the $\{0, 180\}$ phase choice is already baked in — a negative amplitude *is* the $180$-degree-phase term — and I need no $\gamma$ at all. The instant the free phase is gone, $\cos(n\phi)$ at a known scan angle is a fixed number, and the model is purely linear in the amplitudes. That changes the whole character of the problem.

The third move is to exploit that linearity. Folding the half into the coefficient, $K_n = V_n/2$, and writing the basis value $a_{nk} = 1 + \cos(n\phi_k)$ for conformation $k$, I minimize the sum of squared mismatches over the $N$ scanned conformations,

$$F(K) = \sum_{k} \Big( \sum_n K_n\,a_{nk} - \Delta E_k \Big)^2.$$

This is a convex quadratic bowl in the $K_n$ — one global minimum, no initialization, no step size, no local minima — and I get it by setting the gradient to zero. Differentiating with respect to one coefficient $K_m$,

$$\frac{\partial F}{\partial K_m} = \sum_k 2\Big(\sum_n K_n a_{nk} - \Delta E_k\Big) a_{mk} = 0,$$

dropping the factor of two, expanding, and swapping the order of summation to pull $K_n$ out (it does not depend on $k$) gives $\sum_n \big(\sum_k a_{mk} a_{nk}\big) K_n = \sum_k a_{mk}\,\Delta E_k$. Defining the matrix $M_{mn} = \sum_k a_{mk} a_{nk}$ and the vector $B_m = \sum_k \Delta E_k\, a_{mk}$, the stationarity conditions across all $m$ are exactly the linear system

$$M K = B, \qquad K = M^{-1} B.$$

That is the entire fit. $M$ is the Gram matrix of the cosine basis over the scan — entry $(m,n)$ is the dot product of basis functions $m$ and $n$ across all conformations — and $B$ is the projection of the residual target onto each basis function. Being a Gram matrix of real basis vectors, $M$ is symmetric positive semidefinite, and as long as the basis functions are not degenerate over the scan it is invertible, handing back the global least-squares optimum in one shot. The thing the nonlinear optimizers were grinding toward, once the phase is pinned to its natural values, is simply the solution of a linear system: they were descending a paraboloid whose minimum I can write down.

Making this work for a real molecule rather than a toy adds transferability. A molecule has many bonds sharing the same four-atom type signature, and they must share one set of amplitudes or the parameters are not transferable. So the unknowns are the coefficients of a dihedral *type* $i$, and every individual dihedral of that type contributes its own geometry to the same coefficients. The right basis quantity is then not a single cosine but the sum of the cosine basis over all dihedrals of that type at that conformation,

$$A_{ikn} = \sum_{l \in \text{type } i} \big(1 + \cos(n\,\phi_{ikl})\big),$$

and the model energy at conformation $k$ is $\sum_i \sum_n K_{in} A_{ikn}$ — still linear in the unknowns, just with a sum-over-instances basis. One subtlety in binning: a dihedral read forward, A–B–C–D, and read backward, D–C–B–A, are the same physical torsion, so a quadruplet and its reverse must be treated as identical. Re-deriving the normal equations with the grouped basis, $F(K) = \sum_k \big(\sum_{i,n} K_{in} A_{ikn} - \Delta E_k\big)^2$, and differentiating with respect to $K_{i'n'}$, gives $M_{(i'n'),(in)} = \sum_k A_{i'kn'} A_{ikn}$ and $B_{i'n'} = \sum_k \Delta E_k A_{i'kn'}$ — identical structure, just double-indexed, so the grouping never breaks the linear solve; $M$ simply grows to (coefficients per type) times (number of types).

Two further pressures slot into the same machinery. Overfitting: too many cosines let the series thread every scattered scan point while ringing between them, and redundant harmonics trade off into large, cancelling, unphysical amplitudes. Two complementary controls handle this — restrict the basis up front to the chemically expected periodicities, or admit $n = 1..4$ and regularize. An L1/Lasso penalty is especially apt because it drives genuinely redundant amplitudes exactly to zero, selecting which harmonics survive; a blunter, cheaper guardrail bounds the amplitudes to a physical window and solves a bounded residual problem against the already-built $M$ and $B$. The unconstrained $MK = B$ is the fast exact default; the bounded solve is the guardrail when the data is thin or the basis rich. Weighting: plain least squares weights every angle equally, including high barriers the molecule never visits, but for simulation the low-energy, thermally populated wells matter most. Putting a weight $w_k$ on conformation $k$ — a Boltzmann factor $\exp(-E_{\mathrm{rel}}/kT)$ in the conformer's relative energy above the minimum, or a flat-then-attenuating profile — simply makes $M_{mn} = \sum_k w_k a_{mk} a_{nk}$ and $B_m = \sum_k w_k \Delta E_k a_{mk}$, still one linear solve. Finally, some engines take not a cosine series but the Ryckaert–Bellemans polynomial $E = \sum_{m=0}^{5} C_m \cos(\psi)^m$ with $\psi = \phi - 180$. Rather than refit, I convert exactly: $\cos(\psi) = -\cos(\phi)$, each $\cos(n\phi)$ is the Chebyshev polynomial $T_n$ in $\cos(\phi)$, so expanding $\sum_n K_n(1 + \cos n\phi)$ as a polynomial in $\cos(\phi) = -\cos(\psi)$ and matching coefficients gives the closed form $C_0 = K_1 + K_3 + 2K_4$, $C_1 = -K_1 + 3K_3$, $C_2 = 2K_2 - 8K_4$, $C_3 = -4K_3$, $C_4 = 8K_4$, $C_5 = 0$ (the vanishing fifth-power term reflecting that I stopped at the fourth harmonic).

The pipeline as it runs needs three matched-scan inputs: the relative QM energies, the relative no-torsion MM energies on the same conformations from the same engine and 1–4 treatment, and, per dihedral, the measured angle at every conformation tagged with its atom-type quad. The residual is their elementwise difference; the grouped summed-cosine basis builds $M$ and $B$ over all conformations; $K = M^{-1} B$ solves every type's amplitudes at once, with the optional bounded solve and the exact Fourier-to-RB relabeling as add-ons.

```python
import os
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
    if coeff not in (3, 4):
        raise ValueError("coeff must be 3 or 4 (the RB export covers up to the 4th harmonic)")
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
