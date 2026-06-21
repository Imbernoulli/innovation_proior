# Bogoliubov theory of the weakly interacting Bose gas

## Problem

Explain superfluidity — frictionless flow of a degenerate Bose liquid below the λ-point —
*microscopically*, starting from a many-boson Hamiltonian, so that the property of dissipationless flow
emerges from the equations rather than being postulated. A degenerate *ideal* Bose gas cannot do this:
its free-particle spectrum ε(f)=f²/2m gives Landau critical velocity min_f ε(f)/|f| = 0, so flow at any
speed can shed momentum into arbitrarily-low-energy single-particle excitations. The fix must come from
the interaction reshaping the excitation spectrum.

## Key idea

Take a weakly repulsive Bose gas. Because the condensate is macroscopically occupied (N₀ ~ N), replace
the zero-momentum operators by c-numbers, a₀, a₀⁺ → √N₀ (their non-commutativity is an O(1/N₀)
correction). This collapses the quartic interaction to a Hamiltonian *quadratic* in the excited-mode
operators, whose only nontrivial feature is **anomalous pair terms** b_f b_{−f} + h.c. — created by the
interaction scattering pairs in and out of the condensate, coupling momenta +f and −f. A **canonical
(Bogoliubov) transformation** that mixes a creation operator with an annihilation operator,
   ξ_f = (b_f − L_f b_{−f}⁺)/√(1 − L_f²),
diagonalizes it into a free gas of quasiparticles. The quasiparticle spectrum is **linear (phonon-like)
at small momentum**, which by Landau's criterion yields a finite critical velocity — superfluidity.

## Final result (derivation summary)

Second-quantized Hamiltonian in the plane-wave basis, with v(f) = ∫Φ(|q|)e^{−i(f·q)/ℏ}dq the Fourier
transform of the pair potential and T(f) = f²/2m:

After c-numbering the condensate and dropping terms of third and higher order in the excited operators,
the equations of motion for the excited mode f close on the pair (b_f, b_{−f}⁺):

   iℏ ∂b_f/∂t = {T(f) + (N₀/V)v(f)} b_f + (N₀/V)v(f) b_{−f}⁺
   −iℏ ∂b_{−f}⁺/∂t = (N₀/V)v(f) b_f + {T(f) + (N₀/V)v(f)} b_{−f}⁺ .

The eigenfrequency of this 2×2 system is the **Bogoliubov dispersion relation**:

   E(f) = √[ T(f)² + 2 T(f) (N₀/V) v(f) ] = √[ 2 T(f) (N₀/V) v(f) + T²(f) ].

Equivalently, with v = V/N (volume per molecule):
   E(f) = √[ |f|² v(f)/(m v) + |f|⁴/(4m²) ].

**Limits.**
- Small f (T(f) = f²/2m → 0 dominates the cross term): **E(f) = c|f|·(1+…)**, a phonon, with
     c = √[ (N₀/V) v(0)/m ] = √[ v(0)/(m v) ] = √(∂P/∂ρ) = velocity of sound at T = 0.
- Large f (v(f) → 0): **E(f) = f²/2m + v(f)/v + … → T(f)**, the bare molecule.
- One continuous curve; in this dilute model there is no separate roton branch.

**Stability.** The radicand is non-negative for all f iff
   v(0) = ∫Φ(|q|)dq > 0  (net repulsion),
which is identical to the thermodynamic-stability condition ∂P/∂ρ > 0 at T = 0 (using
P = (N²/2V²)v(0)). For v(0) < 0 the small-f energy becomes imaginary and the condensate is unstable.

**Diagonalizing transform.** With
   L_f = (V/(N₀v(f))) · { E(f) − T(f) − (N₀/V)v(f) },
   |L_f|² = [ (N₀/V)v(f) / (E(f)+T(f)+(N₀/V)v(f)) ]² ,
   1 − |L_f|² = 2E(f)/(E(f)+T(f)+(N₀/V)v(f)) ,
the new operators ξ_f = (b_f − L_f b_{−f}⁺)/√(1−|L_f|²) are canonical bosons
([ξ_f, ξ_{f'}⁺] = δ_{f,f'}), obey iℏ ∂ξ_f/∂t = E(f) ξ_f, and give

   H = H₀ + Σ_{f≠0} E(f) n_f,   n_f = ξ_f⁺ξ_f,
   H₀ = (1/2)(N²/V)Φ₀ + (1/2)Σ_{f≠0}[ E(f) − T(f) − (N₀/V)v(f) ],   Φ₀ = v(0).

The weakly excited gas is therefore a **perfect Bose gas of non-interacting elementary excitations
("quasiparticles")** with energy E(f); H₀ contains the classical condensate energy plus the
quasiparticle zero-point (quantum) ground-state energy.

**Ground-state depletion** (self-consistency of the expansion): even at T = 0 a fraction of molecules
sits at nonzero momentum,
   (N − N₀)/N = (1/N)·V/(2πℏ)³ ∫ { [E(f)+T(f)+(N₀/V)v(f)]/(2E(f)) − 1 } df > 0,
which in the dilute contact limit equals (8/3√π)√(n a³), small when n a³ ≪ 1. This must be ≪ 1 for the
approximation to hold.

**Superfluidity (Landau criterion, now with a derived spectrum).** A quasiparticle gas drifting at
velocity u has equilibrium occupations n̄_f = [exp((E(f) − f·u)/Θ) − 1]^{-1}; positivity for all f
requires E(f) > f·u, hence

   |u| < V_c = min_{f≠0} E(f)/|f| .

Because E(f)/|f| → c > 0 as f → 0 (the linear phonon branch) and grows like |f|/2m at large f, the
minimum is **strictly positive**: a finite critical velocity exists and the flow is frictionless below
it. With the free-particle spectrum the minimum would be 0 — no superfluidity. The linear spectrum,
itself a consequence of the repulsive interaction (v(0) > 0), *is* the superfluidity.

## Worked numerical illustration

Closed-form evaluation in dimensionless units (ℏ = m = 1; contact limit v(f) → g = v(0) = 4πa,
condensate density n₀). It confirms V_c = c, the phonon-to-free-particle crossover at the healing
length, and u² − v² = 1.

```python
import numpy as np

def kinetic(k):
    "T(f) = f^2 / 2m, the bare molecule kinetic energy."
    return 0.5 * k * k

def excitation_energy(k, g, n0):
    "E(f) = sqrt( 2 T(f) (N0/V) v(f) + T(f)^2 ): the 2x2 eigenvalue."
    T = kinetic(k)
    return np.sqrt(T * (T + 2.0 * g * n0))          # = sqrt(T^2 + 2 T g n0)

def transform_weights(k, g, n0):
    "u^2 - v^2 = 1 (canonical). u->1,v->0 at large k; both ~1/k (phonon) at small k."
    T = kinetic(k); E = excitation_energy(k, g, n0)
    u2 = (T + g * n0 + E) / (2.0 * E)
    v2 = (T + g * n0 - E) / (2.0 * E)
    return u2, v2

def sound_speed(g, n0):
    "c = sqrt(g n0) = sqrt(dP/drho): the linear small-k slope of E(f)."
    return np.sqrt(g * n0)

def healing_length(g, n0):
    "xi = 1/sqrt(2 m g n0): phonon <-> free-particle crossover scale."
    return 1.0 / np.sqrt(2.0 * g * n0)

def critical_velocity(g, n0, kmax=50.0, npts=200000):
    "Landau: V_c = min_f E(f)/|f| > 0 because the small-f branch is linear."
    k = np.linspace(kmax / npts, kmax, npts)
    return np.min(excitation_energy(k, g, n0) / k)

def quantum_depletion(g, n0):
    "(N - N0)/N = (8/3 sqrt(pi)) sqrt(n a^3), a = g/4pi: ground-state depletion."
    a = g / (4.0 * np.pi)
    return (8.0 / (3.0 * np.sqrt(np.pi))) * np.sqrt(n0 * a ** 3)

if __name__ == "__main__":
    g, n0 = 1.0, 0.01                      # weak coupling, dilute (n0 a^3 << 1)
    print("c  =", sound_speed(g, n0))
    print("xi =", healing_length(g, n0))
    print("Vc =", critical_velocity(g, n0), " (equals c)")
    print("depletion =", quantum_depletion(g, n0))
    for k in (0.01, 0.1, 1.0, 10.0):
        E = excitation_energy(k, g, n0); u2, v2 = transform_weights(k, g, n0)
        print(f"k={k:6.2f} E={E:9.4f} ck={sound_speed(g,n0)*k:9.4f} "
              f"k^2/2={0.5*k*k:9.4f} u^2={u2:7.3f} v^2={v2:7.3f}")
```

Running it gives c = 0.1, V_c = 0.1 (= c), depletion ≈ 3.4×10⁻³, with E ≈ c k for k ≪ 1/ξ and
E ≈ k²/2 for k ≫ 1/ξ — the phonon-to-free-particle crossover, and the strictly positive critical
velocity, that constitute the microscopic mechanism of superfluidity.
