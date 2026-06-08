# The Berry phase

## Problem it solves

A quantum system in the n-th eigenstate of a Hamiltonian H(**R**) is transported adiabatically (slowly) around a closed circuit C in the space of parameters **R**, with H returned to its starting form. The adiabatic theorem guarantees it comes back to the same eigenstate, up to a phase. The question is whether, beyond the familiar **dynamical phase** exp[−(i/ℏ)∫E_n dt], the returning state carries an *extra* phase — one fixed by the geometry of the circuit rather than by elapsed time — and whether that phase is physical or merely a removable phase convention.

## Key idea

It does carry an extra **geometric phase** γ_n(C), and it is gauge-invariant and observable. The standard treatment of the adiabatic theorem removes the term i⟨n|∇n⟩ by rephasing the eigenstates, |n⟩ → e^{iμ(R)}|n⟩, which shifts the connection by −∇μ. That removal is legitimate on an *open* path but impossible around a *closed* loop, because ∮∇μ·dR = 0 for any single-valued μ. So the loop integral of the connection survives every legal rephasing — it is gauge-invariant, the holonomy of the **Berry connection**, exactly analogous to the Aharonov–Bohm phase. The previously-ignored adiabatic phase is real physics whenever the circuit closes.

## The result, stated cleanly

Starting from H(R(t))|ψ⟩ = iℏ∂_t|ψ⟩ with instantaneous eigenstates H(R)|n(R)⟩ = E_n(R)|n(R)⟩, the adiabatic state is

  |ψ(t)⟩ = exp[−(i/ℏ)∫₀ᵗ E_n dt′] · exp[iγ_n(t)] · |n(R(t))⟩.

Substituting into the Schrödinger equation and projecting onto ⟨n| gives γ̇_n = i⟨n|∇_R n⟩·Ṙ, hence the **Berry phase** as a circuit integral of the **Berry connection** A_n = i⟨n|∇_R n⟩:

  γ_n(C) = i ∮_C ⟨n(R)|∇_R n(R)⟩·dR = ∮_C A_n·dR.

It is independent of how fast C is traversed (time-parametrization cancelled), and real because ⟨n|∇n⟩ is purely imaginary (from ∇⟨n|n⟩ = 0).

**Gauge invariance (closed loop).** Under |n⟩ → e^{iμ(R)}|n⟩, A_n → A_n − ∇μ, so γ_n(C) → γ_n(C) − ∮∇μ·dR = γ_n(C) for single-valued μ; allowing e^{iμ} to wind shifts γ only by 2πk, leaving exp(iγ) unchanged. Removable on an open arc, invariant as a phase factor on a circuit.

**Berry curvature via Stokes.** In three-parameter space, with the m = n term vanishing and the off-diagonals from differentiating the eigenvalue equation, ⟨m|∇n⟩ = ⟨m|∇H|n⟩/(E_n − E_m) (m ≠ n),

  γ_n(C) = −∬_S **V**_n(R)·d**S**,  **V**_n(R) = Im ∑_{m≠n} ⟨n|∇H|m⟩ × ⟨m|∇H|n⟩ / (E_n − E_m)².

With the convention A_n = i⟨n|∇n⟩, ∇×A_n = −**V**_n, so γ_n is the negative flux of Berry's curvature field **V**_n. **V**_n involves no phase convention, has ∇·**V**_n = 0 away from degeneracies (so the surface S is arbitrary), and its (E_n − E_m)² denominators make degeneracies its singular sources — "monopoles."

**Generic two-level degeneracy.** For H(R) = ½ **σ**·**R** (the standard form of any traceless 2×2 Hermitian Hamiltonian; degeneracy at R = 0 with the usual codimension-three count), E_± = ±½R and

  **V**_±(R) = ± R/(2R³)  ⇒  γ_±(C) = ∓ ½ Ω(C),  exp{iγ_±} = exp{∓ ½ iΩ(C)},

with Ω(C) the solid angle C subtends at the degeneracy. Restricted to real Hamiltonians (a plane through the degeneracy), Ω = ±2π or 0, so exp{iγ} = −1 around the degeneracy and +1 otherwise — the real-eigenstate sign change recovered as a frozen special case.

**Spin in a rotated field.** For H = κℏ **B**·**S** with spin projection n along **B** (E_n = κℏBn, independent of field direction),

  **V**_n(B) = n**B**/B³  ⇒  γ_n(C) = −nΩ(C),  exp{iγ_n} = exp{−inΩ(C)}.

For spin-½ with n = +½ this is **γ = −½Ω — half the solid angle**. A full turn of the field (Ω = 2π) gives exp(iγ) = −1 for half-integer n (the spinor sign change under 2π rotation) and +1 for integer n; bosons still show the phase on other circuits (n = 1 on a 60° cone gives Ω = π, exp(iγ) = −1). γ_n depends only on the projection n, not on the spin s.

**Aharonov–Bohm as a special case.** A charged box carried around a flux line Φ has eigenstates dressed by a Dirac phase, giving ⟨n|∇_R n⟩ = −iqA(R)/ℏ, so A_n = qA/ℏ is the literal vector potential and γ_n(C) = (q/ℏ)∮A·dR = qΦ/ℏ — independent of n and not reliant on an adiabatic approximation, derived with only single-valued wavefunctions.

## A small worked check (spin-½, the −½Ω prediction)

Code is not the field-natural artifact here — the formulas above are. But the −½Ω factor can be checked by transporting the spin-½ n = +½ eigenstate around a cap of fixed |**B**| and accumulating the connection phase, then comparing to minus half the enclosed solid angle.

```python
import numpy as np

sx = np.array([[0, 1], [1, 0]], dtype=complex)
sy = np.array([[0, -1j], [1j, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)

def H(B):                                   # H = (1/2) sigma . B
    return 0.5 * (B[0]*sx + B[1]*sy + B[2]*sz)

def upper_eigvec(B):                        # n = +1/2 state, E = |B|/2
    w, V = np.linalg.eigh(H(B))
    return V[:, 1]

def loop_cone(theta, N=400):                # circle at polar angle theta
    phi = np.linspace(0, 2*np.pi, N, endpoint=False)
    return [np.array([np.sin(theta)*np.cos(p),
                      np.sin(theta)*np.sin(p),
                      np.cos(theta)]) for p in phi]

def berry_phase(loop):                      # gauge-invariant discrete holonomy:
    states = [upper_eigvec(B) for B in loop] # gamma = -arg( prod_k <n_k|n_{k+1}> )
    prod = 1.0 + 0j
    M = len(states)
    for k in range(M):
        prod *= np.vdot(states[k], states[(k + 1) % M])   # close the loop with %M
    return -np.angle(prod)

theta = np.pi / 3
Omega = 2 * np.pi * (1 - np.cos(theta))     # solid angle of the cap
# n = +1/2  =>  gamma = -n * Omega = -Omega/2  (agreement is mod 2*pi)
print(berry_phase(loop_cone(theta)), -0.5 * Omega)
```
