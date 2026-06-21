# Context

## Research question

A ferromagnet below its Curie temperature carries a spontaneous magnetization inside each domain — the *intrinsic domain magnetization* — that is not quite saturated: thermal agitation tilts individual atomic spins away from perfect alignment, and an applied magnetic field pulls them back. The practical question is quantitative: how does the intrinsic magnetization of a domain depend on temperature and, in particular, on the strength of an applied field? Equivalently, what is the intrinsic susceptibility — the slope of magnetization versus field — and how does it behave at large fields?

Answering this requires the *excitation spectrum* of an ordered spin system. The fully aligned state is easy; the relevant states are the low-lying excited states that get thermally populated and depopulated as temperature and field change. The microscopic Hamiltonian is a lattice of quantum spins coupled by the exchange interaction, with two further terms that matter for the real material: the Zeeman coupling of each spin to the applied field, and the magnetic dipole–dipole interaction between the atomic moments (which makes the magnetization depend on the *shape* of the sample and on the field in a non-trivial way). A solution must (i) give the energies of the elementary excitations of this Hamiltonian, including the external field and the dipolar term, and (ii) let those energies be populated by ordinary statistical mechanics so that the field- and temperature-dependence of the magnetization follows.

## Background

**The exchange interaction (Heisenberg, 1928).** Ferromagnetic alignment is not caused by the weak magnetic dipole forces between atomic moments — those are far too small to explain ordering at hundreds of kelvin. Heisenberg traced it to a purely quantum effect: the Coulomb repulsion between electrons, combined with the Pauli principle, makes the energy of two neighboring atoms depend on whether their spins are parallel or antiparallel. The Heitler–London calculation of the hydrogen molecule (two electrons, two protons) makes this concrete: the symmetric and antisymmetric spatial wavefunctions have energies that differ by twice an *exchange integral* U, and the spin state is locked to the spatial symmetry by antisymmetry of the total wavefunction. Coarse-graining this to a lattice, the spin-dependent part of the energy is captured by

    H_ex = − Σ_⟨ij⟩ J_ij  S_i · S_j ,

with J_ij the exchange constant between sites i and j (often kept to nearest neighbors). For J>0 the energy is minimized by parallel spins — ferromagnetism. The operators S_i are quantum spin-S angular momenta obeying [S_i^α, S_j^β] = i δ_ij ε^{αβγ} S_i^γ (ħ=1), with S^± = S^x ± i S^y, so that

    [S^z, S^±] = ± S^± ,   [S^+, S^-] = 2 S^z ,   S^+|S,m⟩ = √(S(S+1) − m(m+1)) |S,m+1⟩ .

**The ground state and its first excitations.** Take J>0 and a single domain aligned along +z. The state with every spin at its maximum projection, |all m=+S⟩, is an exact eigenstate: S_i·S_j = S_i^z S_j^z + ½(S_i^+ S_j^- + S_i^- S_j^+), and the flip terms S^+S^- annihilate a fully raised state, so only the diagonal piece survives and the energy is −Σ_⟨ij⟩ J S². That is the classical ground state. One step up, flip a single spin at site n by one unit, |…m=S, m=S−1, m=S…⟩. This is *not* an eigenstate: the term S_n^- S_{n+1}^+ moves the flipped spin to a neighbor, so a localized flip is not stationary — it hops.

**Spin waves (Bloch, 1930).** Because a localized flip hops with the same amplitude to each neighbor, the stationary single-excitation states are the plane-wave superpositions

    |k⟩ = N^{−1/2} Σ_n e^{i k·r_n} |flip at n⟩ ,

a *spin wave*: a long-wavelength, collective tilt of the magnetization that precesses and propagates through the lattice. Diagonalizing the one-flip sector gives the single-excitation energy

    ε_k = 2 J S (1 − cos k d)   (one dimension, spacing d),   ε_k ≈ J S (k d)²  as k → 0,

quadratic at long wavelength. Treating these excitations as independent and populating them by statistics yields Bloch's result that the spontaneous magnetization falls off from saturation as the temperature power law T^{3/2} at low T. Each excitation lowers the total z-spin by one unit, i.e. carries spin −1; the number of excitations at a site is a non-negative integer bounded above by 2S (a spin-S cannot be flipped more than 2S times).

**The structure of the spin Hamiltonian.** Bloch's construction is a *counting* of exact single-excitation eigenstates, exact for one excitation. The exchange term is a product of two spin operators whose components do not commute, S_i·S_j = S_i^z S_j^z + ½(S_i^+S_j^- + S_i^-S_j^+). The two-excitation sector involves bound states and the constraint that the on-site flip count is bounded by 2S. The full microscopic Hamiltonian for the field problem carries, in addition to exchange, an external Zeeman term and a dipole–dipole term.

**The Zeeman and dipolar terms.** A static field H along z adds the Zeeman energy g μ_B H Σ_i S_i^z (each spin's moment is −g μ_B S, so lowering S^z costs energy in a field). The classical motion of a single moment in a field is Larmor precession at frequency γ H; quantum-mechanically the field splits the (2S+1) levels by g μ_B H. The dipole–dipole interaction couples the moments through the long-range magnetic field they produce; it is weak compared to exchange and is what ties the magnetization to sample shape (demagnetizing fields) and gives the field dependence its measurable structure. The full microscopic Hamiltonian for the problem is exchange + Zeeman + dipolar.

## Baselines

**Heisenberg exchange model (1928).** Ferromagnetism is electrostatic-plus-Pauli, encoded as H = −Σ J_ij S_i·S_j. It fixes the ground state and the form of the energy, and it is the starting Hamiltonian. It is written in non-commuting spin operators, S_i·S_j = S_i^z S_i^z + ½(S_i^+S_j^- + S_i^-S_j^+). Beyond the fully aligned eigenstate, evaluating thermodynamics means working in the (2S+1)^N-dimensional Hilbert space.

**Bloch spin-wave theory (1930).** Diagonalize the single-flip sector by translational invariance to get the plane-wave excitations with ε_k = 2JS(1−cos kd) and ε_k∝k² at small k, then populate them statistically to obtain the T^{3/2} fall-off of the spontaneous magnetization. The math is the diagonalization of the hopping matrix of a single reversed spin. It furnishes one-excitation eigenstates by an explicit basis construction. A second excitation involves the on-site occupancy bound (≤2S) and two-flip bound states; the field and dipolar terms enter the diagonalization directly. It gives the spectrum for the single-excitation sector and its k→0 shape.

**Classical / phenomenological small-oscillation pictures.** One can treat each spin as a classical vector and linearize its precession about the aligned axis (the Landau–Lifshitz route, or normal-mode analysis of coupled precessing tops). This recovers the spin-wave dispersion as the normal-mode frequencies. It is a classical treatment: the modes are not quantized, so it gives the wave but not the integer mode occupation, the zero-point structure, or the statistical-mechanics population.

## Evaluation settings

The natural yardstick is the low-temperature, ordered phase of a Heisenberg ferromagnet on a regular lattice (e.g. simple cubic, coordination number q; or a one-dimensional chain for the cleanest closed form), with spin S per site, nearest-neighbor exchange J>0, an applied static field H along the magnetization axis, and the magnetic dipole–dipole coupling included. The regime of interest is k_B T ≪ J (few excitations) and large or moderate S, where a small-deviation treatment should hold. The quantities a theory must deliver are: the elementary-excitation energy spectrum ε_k(H); the temperature- and field-dependence of the intrinsic domain magnetization ⟨S^z⟩(T,H); and the intrinsic susceptibility ∂M/∂H, including its behavior at large H. Spin-wave dispersions in such systems are, in principle, accessible experimentally through inelastic neutron and light (Brillouin, Raman) scattering and through ferromagnetic resonance, which probe ε_k directly. The correctness checks internal to the theory are that any proposed treatment must (i) reproduce the spin matrices and the su(2) commutation relations, and (ii) reduce to Bloch's ε_k=2JS(1−cos kd) and the T^{3/2} law in the appropriate limit.

## Code framework

A scaffold for *checking* any proposed treatment of the spin operators against the exact spin algebra on a finite spin-S space. The available primitives are: building the standard (2S+1)-dimensional spin matrices in the |S,m⟩ basis, a generic matrix square-root for a positive-semidefinite operator, and forming commutators. The empty slot is the proposed treatment.

```python
import numpy as np

def spin_matrices(S):
    """Standard (2S+1)-dim spin-S matrices S^z, S^+, S^- in the |S,m> basis."""
    dim = int(round(2 * S + 1))
    m = np.array([S - k for k in range(dim)])      # m = S, S-1, ..., -S
    Sz = np.diag(m).astype(complex)
    Sp = np.zeros((dim, dim), dtype=complex)
    for k in range(1, dim):                          # S^+|m> = sqrt(S(S+1)-m(m+1))|m+1>
        mk = m[k]
        Sp[k - 1, k] = np.sqrt(S * (S + 1) - mk * (mk + 1))
    Sm = Sp.conj().T
    return Sz, Sp, Sm

def msqrt(M):
    """Matrix square root of a positive-semidefinite operator (generic numerics)."""
    w, V = np.linalg.eigh(M)
    return (V * np.sqrt(np.clip(w.real, 0, None))) @ V.conj().T

def commutator(A, B):
    return A @ B - B @ A

def proposed_representation(S):
    """Return (Sz, Sp, Sm) re-expressed in whatever the method introduces."""
    # TODO
    raise NotImplementedError

def check(S):
    Sz, Sp, Sm = spin_matrices(S)
    pz, pp, pm = proposed_representation(S)
    repro = max(np.abs(Sz - pz).max(), np.abs(Sp - pp).max(), np.abs(Sm - pm).max())
    alg = np.abs(commutator(pp, pm) - 2 * pz).max()  # must satisfy [S^+,S^-] = 2 S^z
    return repro, alg
```
