# Context: the geometry of phase in adiabatic quantum transport

## Research question

A quantum system sits in an unchanging environment described by a Hamiltonian, and it occupies one stationary state. Now let the environment change — encode the change as a set of external parameters **R** = (X, Y, …) on which the Hamiltonian H(**R**) depends — and vary **R** slowly around a closed loop C, returning H to exactly its starting form. Two facts are taken as settled. By the adiabatic theorem, the system, if started in the n-th eigenstate, stays in the instantaneous n-th eigenstate throughout and so returns to its original state — up to a phase. And that phase, everyone agrees, contains the *dynamical* part exp(−(i/ℏ)∫E_n dt) accumulated by any stationary state as its internal clock ticks.

The precise question is whether there is anything *else* in that returning phase: a component that depends on the *circuit* C traced in parameter space rather than on the elapsed time, and that does not wash out under a change of the (otherwise arbitrary) phase convention for the eigenstates. If such a component exists it would be a new, observable feature of adiabatic transport — measurable by splitting a beam, carrying one part around C while the other is held fixed, and recombining them so the leftover phase shows up as interference fringes. The pain point is that the standard treatment of the adiabatic theorem appears to make any such extra phase *vanish*: it can seemingly be removed by a clever choice of how the instantaneous eigenstates are phased along the way. Establishing whether that removal is legitimate for a *closed* circuit — and if not, computing what survives — is the goal.

## Background

**The adiabatic theorem (Born & Fock 1928; Messiah 1962).** For a Hamiltonian H(t) varied slowly, with instantaneous eigenstates H(t)|n(t)⟩ = E_n(t)|n(t)⟩ separated by a finite gap, a system started in |n(0)⟩ remains in |n(t)⟩. The instantaneous coupling rate to another level is ⟨m|ṅ⟩ = ⟨m|Ḣ|n⟩/(E_n − E_m), and the adiabatic condition is equivalently ℏ|⟨m|Ḣ|n⟩|/|E_m − E_n|² ≪ 1, so slow variation and an open gap suppress transitions. Substituting the single-state ansatz into the time-dependent Schrödinger equation and projecting onto ⟨n| produces, beyond the familiar dynamical phase θ_n(t) = −(1/ℏ)∫E_n dt′, a second term governed by i⟨n(t)|ṅ(t)⟩. Because ⟨n|n⟩ = 1 forces ⟨n|ṅ⟩ + ⟨ṅ|n⟩ = 0, the quantity ⟨n|ṅ⟩ is purely imaginary, so this extra term contributes a *real* phase. The prevailing wisdom is that it is inert: redefine the eigenstate phase, |n(t)⟩ → e^{iμ(t)}|n(t)⟩, choosing μ to make ⟨n|ṅ⟩ vanish along the path, and the extra phase is gone. On an open path this works; the extra phase is treated as a gauge artifact and dropped from textbook accounts.

**The Aharonov–Bohm effect (Aharonov & Bohm 1959; observed by Chambers 1960; anticipated by Ehrenberg & Siday 1949).** A charged particle confined to a region where the magnetic field is exactly zero, but the vector potential A is not, nevertheless picks up an observable phase (q/ℏ)∮A·dr = qΦ/ℏ when its support encircles a line of flux Φ. This is a phase produced by transport around a loop, with no local field acting anywhere the particle goes — a holonomy. It established that a quantity which is "pure gauge" locally can still produce real interference when integrated around a closed path that is topologically nontrivial.

**Degeneracies and the no-crossing theorem (Von Neumann & Wigner 1929).** A generic Hermitian Hamiltonian needs *three* real parameters tuned simultaneously to make two levels coincide accidentally — degeneracies have codimension three. If the Hamiltonian is also *real* (real symmetric matrices, e.g. systems with time-reversal symmetry), two parameters suffice — codimension two. These isolated degeneracy points are the organizing singularities of the parameter space.

**The sign change of real eigenstates (Herzberg & Longuet-Higgins 1963; Longuet-Higgins 1975).** When a Hamiltonian is real and symmetric and two of its eigenstates are continued smoothly around a circuit in parameter space that encloses a degeneracy, each of the two real eigenvectors comes back with its sign reversed. This is used in practice as a *detector* of a true level coincidence (versus a near-miss avoided crossing): carry the wavefunction around the suspected point and check whether it flips sign. Berry & Wilkinson's triangle-billiard "diabolical points" rest on exactly this test. The sign reversal is the same algebraic fact as the −1 a spinor acquires under a 2π rotation (Aharonov & Susskind 1967; reviewed by Silverman 1980).

**The molecular Aharonov–Bohm effect (Mead 1979, 1980; Mead & Truhlar 1979).** In the Born–Oppenheimer treatment of a molecule, the electronic eigenstates depend parametrically on the nuclear coordinates. Mead and Truhlar found, by perturbation theory and for an infinitesimal circuit, essentially the same loop/surface phase expression, and showed that enforcing single-valuedness of the electronic state forces a vector-potential-like term into the nuclear Schrödinger equation. Their phase was attached to a particular continuation rule for eigenstates in coordinate space, not to slow evolution under the time-dependent Schrödinger equation, but the formula coincides.

**Gauge structure of the eigenstate phase (Wu & Yang 1975).** The freedom |n(**R**)⟩ → e^{iμ(**R**)}|n(**R**)⟩ shifts ⟨n|∇_R n⟩ → ⟨n|∇_R n⟩ + i∇μ. This is exactly a gauge transformation of an abstract "vector potential" living in parameter space: the connection ⟨n|∇_R n⟩ is itself gauge-dependent.

## Baselines

These are the prior treatments a new account of the loop phase would be measured against.

- **Standard adiabatic theorem (Messiah 1962).** Core: the single-state ansatz with the dynamical phase only; the i⟨n|ṅ⟩ term is acknowledged but absorbed into a redefinition of the eigenstate phase. **Gap:** it never asks whether that absorption is consistent *around a closed loop* — whether a single, globally single-valued phase choice can kill the term everywhere on C at once. It therefore declares the extra phase unobservable without proof for the cyclic case.

- **Aharonov–Bohm analysis (Aharonov & Bohm 1959).** Core: exact solution of the Schrödinger equation in the vector potential of a flux line, yielding the observable qΦ/ℏ phase. **Gap:** it is a single, specific physical mechanism (electromagnetic potentials), derived case-by-case; it is not presented as an instance of a general phase that *any* adiabatically cycled eigenstate carries, and the elementary "two paths around the flux" presentations rely on a multivalued wavefunction.

- **Herzberg–Longuet-Higgins sign rule (1963).** Core: real eigenstates flip sign around a degeneracy; a practical degeneracy detector. **Gap:** it is restricted to real Hamiltonians and yields only the discrete value −1. It is silent about what happens to the continued wavefunction when the Hamiltonian is made complex (e.g. a magnetic field breaks time-reversal symmetry, lifting the codimension-two degeneracy to codimension three) — the very case where the real-symmetric restriction that pinned the eigenvector to ±1 no longer applies.

- **Mead–Truhlar molecular phase (1979).** Core: a perturbative, infinitesimal-circuit derivation of the loop/surface phase and the induced vector potential in nuclear dynamics. **Gap:** tied to a specific continuation rule and to the molecular Born–Oppenheimer setting, for an infinitesimal loop; not formulated as a general property of slow evolution of any quantum system around a finite circuit, and not connected to the adiabatic theorem's dynamical-versus-geometric split or to the spin/Aharonov–Bohm cases.

## Evaluation settings

The natural yardsticks are physical situations where a loop phase could in principle be exhibited or measured, all available before any general formula exists:

- **Spin in a slowly rotated magnetic field.** A particle of spin s with H = κℏ **B**·**S**; the parameters are the components of **B**; a circuit is a slow sweep of the field *direction* (magnitude fixed) around a closed cone or loop subtending a solid angle Ω at **B** = 0. The energies E_n = κℏBn are insensitive to field direction, so the dynamical phase is identical for two beams differing only in the swept direction — an ideal arena for isolating a direction-loop phase. A concrete protocol: split a polarized monoenergetic beam, hold **B** fixed on one arm, sweep its direction round a cone of semiangle θ on the other, recombine, and read interference fringes versus the solid angle.

- **Real-Hamiltonian circuits around a degeneracy.** Triangle billiards and other real-symmetric spectra (Berry & Wilkinson's diabolical points), where a circuit in the two-parameter shape space encloses a level coincidence; the observable is the sign of the continued real eigenfunction.

- **Aharonov–Bohm geometry.** A localized system (a box holding a charged particle) carried around a line of magnetic flux it never touches; the metric is the interference phase between transported and untransported copies.

- **Metric / instrument.** In every case the quantity of interest is the extra interference phase accumulated around the closed circuit, read out as fringe shifts when a cycled copy is recombined with an uncycled reference, expressed against the *geometry* of the circuit (its enclosed solid angle, its winding number) rather than against elapsed time.

## Code framework

The analytic work can be checked numerically once a candidate closed-form is in hand. The generic primitives are instantaneous diagonalization of a parametrized Hamiltonian, smooth transport of an eigenvector along a discretized loop, accumulation of the consecutive-overlap phase around the loop, and one empty slot for whatever closed-form prediction the analysis yields.

```python
import numpy as np

def eig_state(H):
    """Instantaneous eigenstates/energies of a Hermitian H(R)."""
    w, V = np.linalg.eigh(H)
    return w, V  # columns V[:,k] are |k(R)>

def hamiltonian(R):
    """Parametrized Hamiltonian H(R). Filled per physical system."""
    raise NotImplementedError

def transport_phase(loop, n):
    """
    Carry eigenstate n around a discretized closed loop in R-space and read off
    the loop phase from the consecutive-site overlaps <n_k|n_{k+1}>. The per-site
    eigenvector phases are arbitrary, so the phase is taken from the product of
    overlaps around the whole loop, which does not depend on those choices.
    """
    states = []
    for R in loop:
        w, V = eig_state(hamiltonian(R))
        states.append(V[:, n].copy())
    prod = 1.0 + 0j
    M = len(states)
    for k in range(M):
        prod *= np.vdot(states[k], states[(k + 1) % M])   # %M closes the loop
    return -np.angle(prod)

def geometric_prediction(loop):
    """
    Closed-loop phase predicted from the geometry enclosed in parameter space.
    """
    # TODO: closed-form geometry expression
    pass
```
