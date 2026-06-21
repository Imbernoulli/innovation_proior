When a quantum Hamiltonian is carried slowly around a closed circuit in its parameter space, the adiabatic theorem says the system stays locked to the same instantaneous eigenstate. The returning wavefunction picks up the familiar dynamical phase, exp(-(i/ℏ)∫E_n dt), which is just the eigenstate's internal clock ticking. But the standard textbook treatment quietly assumes that any additional phase accumulated along the way can be erased by rephasing the eigenstates at each parameter value. That maneuver works on an open path, where the phase choice at the start and end can differ, but it collapses on a closed loop because a single-valued phase function must satisfy ∮∇μ·dR = 0. The result is that a circuit genuinely encodes a geometric piece of the returning phase that cannot be gauged away.

Existing ideas stop short in different ways. The adiabatic theorem discards the i⟨n|∇n⟩ term as a convention. The Aharonov-Bohm effect shows that loop phases can be physical, but it is usually presented as a special electromagnetic curiosity rather than a general feature of adiabatic transport. The Herzberg-Longuet-Higgins sign rule gives a discrete -1 around a real-Hamiltonian degeneracy, yet it freezes the phenomenon to time-reversal-symmetric systems and hides the continuous phase that survives when the Hamiltonian is made complex. What is needed is a gauge-invariant, geometric phase attached to the circuit itself, computable from the Hamiltonian without choosing a global phase convention.

The new method is the Berry phase. Consider a Hamiltonian H(R(t)) transported slowly along a closed path C with R(T) = R(0), and focus on the n-th isolated eigenstate H(R)|n(R)⟩ = E_n(R)|n(R)⟩. Write the adiabatic solution as |ψ(t)⟩ = exp(-(i/ℏ)∫E_n dt') exp(iγ_n(t)) |n(R(t))⟩. Substituting into the Schrödinger equation and canceling the dynamical phase leaves γ̇_n = i⟨n(R)|∇_R n(R)⟩·Ṙ. Integrating around the circuit gives the Berry phase

γ_n(C) = i ∮_C ⟨n(R)|∇_R n(R)⟩·dR.

This expression is real because ⟨n|∇n⟩ is purely imaginary, and it is independent of how fast the loop is traversed: time drops out when dR replaces Ṙ dt. Under a local rephasing |n⟩ → e^{iμ(R)}|n⟩ the connection shifts by -∇μ, but the closed-loop integral of a gradient of a single-valued function is zero, so γ_n(C) is gauge invariant up to an integer multiple of 2π, which leaves exp(iγ_n) unchanged.

For computation it is better to avoid differentiating eigenvectors and fixing phases. In three-dimensional parameter space, apply Stokes' theorem to write the phase as a surface integral over any surface S spanning C. Inserting a complete set of states and using ⟨m|∇n⟩ = ⟨m|∇H|n⟩/(E_n - E_m) for m ≠ n gives

γ_n(C) = -∬_S V_n(R)·dS,

with the Berry curvature

V_n(R) = Im Σ_{m≠n} ⟨n|∇H|m⟩ × ⟨m|∇H|n⟩ / (E_n - E_m)².

This field depends on no phase convention, is divergence-free away from degeneracies, and has monopole-like singularities at level crossings. For the generic two-level Hamiltonian H(R) = ½ σ·R near a degeneracy, the curvature is V_±(R) = ±R/(2R³), so the upper state accumulates γ_+(C) = -½ Ω(C), where Ω(C) is the solid angle subtended by the circuit at the degeneracy. Restricting to real symmetric Hamiltonians collapses the loop to a plane, the solid angle becomes ±2π, and exp(iγ) = -1 recovers the Herzberg-Longuet-Higgins sign change. For a spin-s particle in a slowly rotated magnetic field, H = κℏ B·S, the energies depend only on |B|, so the dynamical phase cancels in an interferometric comparison, and the geometric phase is γ_n(C) = -n Ω(C), where n is the spin projection along B. Spin-½ with n = +½ again gives -½Ω, while integer-spin states can also produce phase factors of -1 on suitably chosen cones.

The Aharonov-Bohm phase is the same structure in disguise: when the parameter is the position of a charged particle's confining box and a flux line threads the circuit, the Berry connection becomes the literal electromagnetic vector potential A_n = qA/ℏ, giving γ_n(C) = qΦ/ℏ. In every case the Berry phase is an observable holonomy tied to the geometry of parameter-space transport, not to elapsed time.

A direct numerical check for spin-½ confirms the -½Ω prediction. The discrete holonomy accumulated from consecutive overlaps is gauge invariant because the arbitrary phase returned by the diagonalizer at each site cancels in the loop product:

```python
import numpy as np

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

def H(B):
    return 0.5 * (B[0] * sx + B[1] * sy + B[2] * sz)

def upper_eigvec(B):
    w, V = np.linalg.eigh(H(B))
    return V[:, 1]

def loop_cone(theta, N=400):
    phi = np.linspace(0, 2 * np.pi, N, endpoint=False)
    return [np.array([np.sin(theta) * np.cos(p),
                      np.sin(theta) * np.sin(p),
                      np.cos(theta)]) for p in phi]

def berry_phase(loop):
    states = [upper_eigvec(B) for B in loop]
    prod = 1.0 + 0j
    M = len(states)
    for k in range(M):
        prod *= np.vdot(states[k], states[(k + 1) % M])
    return -np.angle(prod)

theta = np.pi / 3
Omega = 2 * np.pi * (1 - np.cos(theta))
print(berry_phase(loop_cone(theta)), -0.5 * Omega)
```

The output agrees with -½Ω modulo 2π, showing that the extra phase left over after the dynamical phase is exactly the Berry phase set by the solid angle of the circuit.
