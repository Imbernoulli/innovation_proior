Let me start from the thing that actually bothers me about my force field, which is that I trust most of its energy terms and distrust exactly one of them. I write a molecule's energy as a sum: harmonic springs for bonds and angles, a Lennard-Jones plus Coulomb sum over nonbonded pairs, and a term for rotation about single bonds. The bond and angle springs are local and stiff and they transfer between molecules with barely any error — a C–C stretch constant is a C–C stretch constant. The nonbonded parameters are fit to bulk and dimer data and they transfer too. But the energy of turning one part of a molecule relative to another about a rotatable bond — the torsion profile — is the part I keep getting wrong, and I want to understand why before I try to fix it.

So picture rotating about one bond, sweeping the dihedral angle phi from -180 to 180, and ask what physically shapes the energy as I turn. There's the obvious steric stuff: atoms three bonds apart, on either end of the bond I'm twisting, swing past each other and clash or relax. But there's also conjugation — if there's pi character across the bond, the molecule pays to twist out of planarity, and that's an *electronic* effect, not a pairwise distance thing. There's hyperconjugation favoring staggered over eclipsed in a plain alkane. None of these is something a harmonic angle spring knows about, and only the first is even partly captured by my nonbonded sum — and that sum is itself approximate across a bond, because the 1–4 pairs straddling the rotatable bond are the ones whose real interaction is most poorly described by a point-charge-plus-LJ model, which is exactly why people scale those 1–4 interactions down by a fixed factor. So there is genuinely leftover barrier that the rest of my force field structurally cannot produce. That's the gap. I need a dedicated torsion term to carry it.

Now, what is the *truth* I'm trying to match? Quantum mechanics. I can take the molecule, fix the dihedral to a value, let everything else relax, compute the QM energy, and step the dihedral across a grid. That gives me E_QM(phi), the real rotational profile — the energy of the least-strained conformer at each twist angle. This has to be a *relaxed* scan, not a rigid one: if I froze every other coordinate while turning, I'd be measuring a mix of the torsion barrier and a bunch of artificial bond/angle strain I forced on the molecule, and that's not transferable to how the molecule actually moves. So: constrain phi to each grid value, optimize all other degrees of freedom under QM, read off the energy. Steps of ten or fifteen degrees over the full -180..180 circle.

The naive thing is to make my torsion term reproduce E_QM(phi) directly — fit the term so that "torsion energy" equals "QM energy." That's wrong, and it's wrong in a way that would double-count. My force field is *already* going to compute bond, angle, and nonbonded energy along that same scan. Whatever those terms produce, they produce; the torsion term is an *addition* on top. So if I made the torsion term equal the whole QM barrier, then the total MM energy would be the QM barrier *plus* whatever the bonds/angles/nonbonded already contribute, and I'd be way over. The torsion term must only carry the part that's *left over* after the rest of the force field has had its say.

So let me actually construct that leftover. Take the same scanned conformations — the geometries from the relaxed QM scan, in the same order — and run them through my MM engine with the torsion term switched off, all torsion amplitudes set to zero. Call that E_MM_without_torsion(phi): everything my force field knows how to compute on those conformations, minus the torsion term, along the rotation. Now the quantity I want my torsion term to be is the residual

    DeltaE(phi) = E_QM(phi) - E_MM_without_torsion(phi).

This is the honest target. It's the barrier the rest of the model fails to explain. And I want to be careful to compute E_MM_without_torsion with the *exact* nonbonded treatment I'll use in production — same 1–4 scaling, same charges — because if the torsion term is going to absorb the slop in the 1–4 interactions, it has to absorb the slop that's actually there. In practice the cleanest way to get this is to take my real topology and literally zero out every torsion parameter and re-evaluate the scan conformations, rather than try to reconstruct the no-torsion energy by hand; let the engine do it so nothing is inconsistent.

OK. I have DeltaE(phi) sampled at a few dozen angles. What functional form do I make the torsion term? It's rotation, so it's periodic in phi with period 2pi — E_tors(phi + 2pi) = E_tors(phi). Anything periodic I can write as a Fourier series. And the physical structure tells me which harmonics to expect: a plain sp3–sp3 bond has three equivalent staggered minima as I go around, which is a cos(3 phi) shape; a conjugated or planar-preferring bond has two minima, cis and trans, which is cos(2 phi); and any asymmetry between gauche and anti — different energies for the two non-eclipsed wells — is a cos(1 phi) tilt. Higher than n = 4 rarely buys anything for a single bond. So a few cosine harmonics, n = 1 through 3 or 4, should span the shapes I see. Let me write it the way people pin the phase to the natural symmetry points:

    E_tors(phi) = sum_{n} (V_n / 2) [1 + cos(n phi)].

The "1 +" is just a constant offset per term so each piece has a clean zero; it shifts the baseline and doesn't change the shape. Now, what about a phase — cos(n phi - gamma)? If I keep gamma as a genuine free angle, I get more flexibility, but watch what it does to the math: cos(n phi - gamma) is not linear in gamma, it's trigonometric in it. The moment gamma is a free parameter, fitting becomes a *nonlinear* optimization — I'd need an iterative optimizer, I'd worry about local minima, initialization, step sizes. That's the world AMBER lives in with its explicit gamma_n, and it's the world a general force-field optimizer lives in when it descends a weighted sum of squared QM-vs-MM deviations with a steepest-descent or Levenberg–Marquardt loop. Before I commit to that machinery, let me check whether I actually need the free phase.

For most torsions the minima sit at the symmetric places — 0, 60, 120, 180 — so the phase is either 0 or 180 degrees. And cos(n phi - 180) = -cos(n phi). So flipping the phase between its two natural values is the *same thing as flipping the sign of V_n*. If I let V_n range over all reals, positive and negative, the {0, 180} phase choice is already baked in — a negative amplitude is the 180-degree-phase term. So for the standard case I don't need gamma at all; I just need V_n that can be either sign. And the instant I drop the free phase, look at what happens to the model: cos(n phi) at the known scan angles is a *fixed number*. The unknowns are only the amplitudes V_n, and they enter

    E_tors(phi) = sum_n (V_n / 2) [1 + cos(n phi)]

purely *linearly*. The torsion energy is a linear combination of known basis functions with the V_n as coefficients. That changes everything about how hard this problem is.

Let me lean into that. I want to choose the amplitudes so that, summed over my scan of N conformations, the torsion term matches the residual as closely as possible. Least squares is the natural objective — minimize the sum of squared mismatches:

    F(V) = sum_{k=1}^{N} ( E_tors(phi_k) - DeltaE_k )^2
         = sum_k ( sum_n (V_n/2)(1 + cos(n phi_k)) - DeltaE_k )^2.

Let me clean up notation: fold the 1/2 into the coefficient and call it K_n = V_n/2, and define the basis value a_{nk} = 1 + cos(n phi_k), which is just a number I can compute the moment I know the scan angle phi_k. Then the model at conformation k is sum_n K_n a_{nk}, and

    F(K) = sum_k ( sum_n K_n a_{nk} - DeltaE_k )^2.

This is a quadratic bowl in the K_n. There's no initialization, no step size, no local minima — a convex quadratic has a single global minimum, and I can get it by setting the gradient to zero. Let me actually take the derivative with respect to one coefficient K_m and set it to zero.

    dF/dK_m = sum_k 2 ( sum_n K_n a_{nk} - DeltaE_k ) a_{mk} = 0.

Drop the 2, expand:

    sum_k a_{mk} sum_n K_n a_{nk} = sum_k a_{mk} DeltaE_k.

Swap the order of the sums on the left, pulling K_n out since it doesn't depend on k:

    sum_n ( sum_k a_{mk} a_{nk} ) K_n = sum_k a_{mk} DeltaE_k.

There it is. Define a matrix and a vector:

    M_{mn} = sum_k a_{mk} a_{nk}        (one equation per coefficient m)
    B_m    = sum_k DeltaE_k a_{mk}.

Then the stationarity conditions across all m are exactly the linear system

    M K = B,    so    K = M^{-1} B.

That's the entire fit. M is the Gram matrix of my cosine basis evaluated over the scan — entry (m, n) is the dot product of basis function m and basis function n across all conformations. B is the projection of the residual target onto each basis function. Because M is a Gram matrix of real basis vectors it's symmetric positive semidefinite, and as long as my basis functions aren't degenerate over the scan it's invertible, and the inverse hands me the global least-squares optimum in one shot. No iteration. The thing the nonlinear optimizers were grinding toward, when the phase is fixed to its natural values, is just the solution of a linear system — they were descending a paraboloid whose minimum I can write down. That's the realization: torsion fitting, posed this way, isn't an optimization problem at all in the iterative sense, it's a linear solve.

Now I have to make this work for a real molecule, not a toy with one dihedral. Transferability is the first pressure. A molecule doesn't have one dihedral of a given kind — it has many bonds whose four-atom type signature is identical, and they must *share* one set of amplitudes, or the parameters aren't transferable and the whole point of a force field collapses. So the unknowns aren't "the coefficients of dihedral l," they're "the coefficients of dihedral *type* i," and every individual dihedral of that type contributes its own geometry to the same coefficients. At a given conformation k, the total torsion energy from type i is the sum over all dihedrals l of that type:

    sum_{l in type i} sum_n K_{in} (1 + cos(n phi_{ikl})).

So the right basis quantity isn't a single cosine — it's the *sum* of the cosine basis over all the dihedrals of that type at that conformation:

    A_{ikn} = sum_{l in type i} (1 + cos(n phi_{ikl})).

The model energy at conformation k is then sum_i sum_n K_{in} A_{ikn} — still linear in the unknowns K_{in}, just with the basis being a sum-over-instances. One subtlety in identifying "the same type": a dihedral read forward, A–B–C–D, and the same atoms read backward, D–C–B–A, are the same physical torsion, so when I bin dihedrals into types I have to treat a quadruplet and its reverse as identical. With that, the normal equations are exactly as before, only now the index that was "n" becomes the composite "(type i, harmonic n)," and M is a bigger Gram matrix — its size is (number of coefficients per type) times (number of types). Same derivation, same M K = B, same one-shot solve.

Let me re-derive the entries with the grouped basis to be sure nothing shifted. Objective:

    F(K) = sum_k ( sum_i sum_n K_{in} A_{ikn} - DeltaE_k )^2.

Differentiate with respect to K_{i'n'}:

    sum_k ( sum_i sum_n K_{in} A_{ikn} - DeltaE_k ) A_{i'kn'} = 0,

which rearranges to

    sum_{i,n} ( sum_k A_{i'kn'} A_{ikn} ) K_{in} = sum_k DeltaE_k A_{i'kn'},

i.e. M_{(i'n'),(in)} = sum_k A_{i'kn'} A_{ikn} and B_{i'n'} = sum_k DeltaE_k A_{i'kn'}. Identical structure, just double-indexed. Good — the grouping doesn't break the linearity, which is the whole reason this stays a linear solve.

Overfitting is the next pressure. If I throw in too many cosines, the series has enough freedom to thread through every scattered scan point while oscillating between them — ringing — and worse, redundant harmonics start trading off against each other into large, unphysical, cancelling amplitudes that happen to fit the sample but mean nothing. Two ways to control this, and they're complementary. One is to restrict the basis up front to the chemically expected periodicities — if I know the bond is sp3–sp3 I might only allow n = 3, or n = 1..3 — and just not give the fit the harmonics it could abuse. The other is to admit all harmonics up to a cutoff, say n = 1..4, and *regularize*: penalize large or redundant amplitudes. An L1 (Lasso) penalty is especially apt here because it drives genuinely redundant amplitudes exactly to zero, effectively selecting which harmonics survive rather than just shrinking them — you end up keeping the periodicities the data actually supports. A cheaper, blunter version is to bound the amplitudes to a physically reasonable window and solve a bounded residual problem for the already-built linear system; that keeps |K_n| from blowing up without committing to a full penalty. The unconstrained M K = B is the fast, exact default; the bounded or penalized solve is the guardrail when the data is thin or the basis is rich.

There's also the question of *which scan points matter*. Plain least squares weights every angle equally, including high barriers the molecule almost never visits. But for simulation what matters is getting the low-energy, thermally populated part of the profile right. So I can weight the residuals — a Boltzmann factor exp(-DeltaE / kT) leans the fit toward the populated wells, or a flat-then-attenuating weight keeps full weight up to about 1 kcal/mol of relative energy and tapers it off by ~10 kcal/mol so absurdly high barriers don't drag the fit around. Weighting slots straight into the same linear machinery: put a weight w_k on conformation k and the normal equations become M_{mn} = sum_k w_k a_{mk} a_{nk}, B_m = sum_k w_k DeltaE_k a_{mk}. Still linear, still one solve.

One more thing worth nailing down, because some MM engines don't take a cosine Fourier series — they take the Ryckaert–Bellemans polynomial in cos(psi), with psi = phi - 180 (so the trans state is at psi = 0), E = sum_{m=0}^{5} C_m cos(psi)^m. I don't want to refit; I want to convert my fitted Fourier amplitudes into RB coefficients exactly. Note cos(psi) = cos(phi - 180) = -cos(phi). And each cos(n phi) is a polynomial in cos(phi) — the Chebyshev polynomial T_n. So I expand my series E = sum_n K_n (1 + cos(n phi)) using cos(2phi) = 2cos^2 - 1, cos(3phi) = 4cos^3 - 3cos, cos(4phi) = 8cos^4 - 8cos^2 + 1, write everything as a polynomial in cos(phi) = -cos(psi), and match coefficients of cos(psi)^m. Grinding the algebra for K_1..K_4:

    C_0 = K_1 + K_3 + 2 K_4
    C_1 = -K_1 + 3 K_3
    C_2 = 2 K_2 - 8 K_4
    C_3 = -4 K_3
    C_4 = 8 K_4
    C_5 = 0.

(The C_5 = 0 makes sense — I only went up to the fourth harmonic, so no fifth-power term appears.) This is an exact, closed-form relabeling, no second fit.

Let me put the whole pipeline together as it would actually run. I need three inputs along a matched scan: the QM energies (relative to the lowest conformer), the MM energies with all torsion parameters set to zero on the same conformations (also relative, same engine, same 1–4 treatment), and, for each dihedral I care about, the measured angle at every conformation, tagged with its four-atom type so I can group. The residual is the elementwise difference of the first two. I build, for each dihedral type and each harmonic, the summed cosine basis across that type's dihedrals; I accumulate the Gram matrix M and the projection vector B over all conformations; I solve K = M^{-1} B for every type's amplitudes at once. If I want the implementation's bounded guardrail, I also solve the bounded residual problem for M K against B and keep it as an additional output; if the engine wants Ryckaert–Bellemans coefficients, I convert each type's Fourier amplitudes exactly. That's it.

The implementation follows that data path.

```python
import os, json
import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy.optimize import lsq_linear

def fourier_to_RB(k1, k2, k3, k4=0.0):
    return (k1 + k3 + 2*k4, -k1 + 3*k3, 2*k2 - 8*k4, -4*k3, 8*k4, 0.0)

def _convert_result(fourier, coefficient_type):
    if coefficient_type == "RB":
        return {key: list(fourier_to_RB(*values)) for key, values in fourier.items()}
    return fourier

def estimate_rotational_coefficients(qm_file, mm_file, dih_dir, coeff=4,
                                      coefficient_type="Fourier", bounds=None):
    # (1) Residual target: QM energy minus no-torsion MM energy for the same scan rows.
    qm = pd.read_csv(qm_file, sep=r"\s+", header=None)[0]
    zero_term = pd.read_csv(mm_file, sep=r"\s+", header=None)[1]
    target = (qm - zero_term).to_numpy()

    # (2) Read one *.dih file per dihedral; first line = atom-type quadruplet.
    #     A quadruplet and its reverse are the same physical torsion type.
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

    # (3) Accumulate M K = B using the grouped basis
    #     A[i,k,n] = sum_{l in type i} (1 + cos(n * phi_{ikl})).
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

    # (4) The exact unconstrained normal-equation solution.
    K = inv(M) @ B
    fourier = {type_order[i]: list(K[coeff*i:coeff*i + coeff]) for i in range(len(type_order))}
    result = {"best": _convert_result(fourier, coefficient_type)}

    # (5) Optional bounded output, matching the script's bounded solve on M K ~= B.
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

The chain start to finish: the bond/angle/nonbonded terms structurally miss the rotational barrier, so a QM relaxed scan gives the truth E_QM(phi); subtracting the torsion-free MM energy isolates the residual DeltaE(phi) that the torsion term alone must reproduce; that term is a Fourier cosine series, and pinning its phases to the natural {0, 180} via the sign of the amplitudes makes it *linear* in the amplitudes; least squares over the scan then has a quadratic objective whose gradient-zero condition is the linear system M K = B, solved in one shot for all dihedral types sharing parameters; regularization or bounds tame overfitting, optional weights focus the fit on populated conformers, and a closed-form relabeling exports the result to the Ryckaert–Bellemans convention.
