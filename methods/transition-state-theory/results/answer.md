# Transition state theory (activated-complex theory): absolute reaction rates

## Problem

Compute the absolute rate constant k of an elementary reaction — including the pre-exponential factor that the empirical law k = A exp(−E_a/RT) leaves unexplained — from molecular data alone (masses, geometries, vibrational frequencies, and the height of the barrier on the potential energy surface).

## Key idea

A reaction is the flux of the system's representative point across a dividing surface placed at the **col (saddle point)** of the potential energy surface — the bottleneck separating reactants from products. The configurations on that surface form the **activated complex**. Two assumptions turn an intractable dynamical flux into an equilibrium statistical-mechanics calculation:

1. **Quasi-equilibrium**: the activated complexes are Boltzmann-distributed, in equilibrium with the reactants.
2. **No recrossing**: a complex that crosses the surface in the forward direction becomes product.

The activated complex is treated as an ordinary molecule in its 3N − 1 *bound* degrees of freedom, plus **one special degree of freedom** — motion along the reaction coordinate, which at the saddle is the crossing motion, not a vibration. When this one degree of freedom is factored out — whether modeled as a loose vibration (ν → 0), as free translation through a thin slab, or as a forward-momentum flux integral — its contribution to the population of complexes and its contribution to the crossing rate are reciprocal, and a single **universal frequency k_B T/h** survives.

## Final result

For A + B → products,

    k = κ · (k_B T / h) · (Q‡ / (Q_A Q_B)) · exp(−E₀ / k_B T),

where
- k_B = Boltzmann constant, h = Planck constant, T = temperature; k_B T/h ≈ 6.25 × 10¹² s⁻¹ at 300 K;
- Q_A, Q_B = complete molecular partition functions of the reactants (per unit volume);
- Q‡ = partition function of the activated complex with the **reaction-coordinate mode removed** (3N − 1 bound degrees of freedom);
- E₀ = barrier height at the col, referenced to the reactant zero-point energy;
- κ ≤ 1 = transmission coefficient absorbing recrossing, tunnelling, and nonadiabatic effects (κ = 1 default).

Equivalent thermodynamic form, with K‡ = exp(−ΔG‡/RT) and ΔG‡ = ΔH‡ − TΔS‡:

    k = κ · (k_B T / h) · exp(−ΔG‡ / RT) = κ · (k_B T / h) · exp(ΔS‡ / R) · exp(−ΔH‡ / RT),

times a standard-state factor (c⊖)^{1−m} for molecularity m. Matching to Arrhenius, the activation energy is E_a = ΔH‡ + RT (unimolecular/condensed) or ΔH‡ + 2RT (bimolecular gas), and the pre-exponential factor is A = e·(k_B T/h)·exp(ΔS‡/R)·(c⊖)^{1−m} (unimolecular/condensed) or e²·(k_B T/h)·exp(ΔS‡/R)·(c⊖)^{1−m} (bimolecular gas). The empirical "steric factor" is exactly exp(ΔS‡/R).

## Derivation (compact)

**Population on the surface.** Treating the activated complex as a species in equilibrium with reactants,
C‡/(C_A C_B) = K‡ = (F‡/(F_A F_B)) exp(−E₀/k_B T), from μ = −k_B T ln(F/N) with a common energy zero.

**Crossing the reaction-coordinate degree of freedom.** Split it off, three equivalent ways:

- *Loose vibration*: q* = lim_{ν→0} 1/(1−e^{−hν/k_B T}) = k_B T/hν; decomposition rate = ν; the ν cancels: ν·(k_B T/hν) = k_B T/h.
- *Free translation in a slab of width δ*: q_tr = (2π m* k_B T)^{1/2} δ/h; mean forward speed ⟨v⟩ = (k_B T/2π m*)^{1/2}; crossing rate ⟨v⟩/δ; product (⟨v⟩/δ)·q_tr = k_B T/h, with δ and m* cancelling.
- *Flux integral*: rate per complex-density = ∫₀^∞ (p/m*) e^{−p²/2m* k_B T} dp / [h (2π m* k_B T)^{1/2}] = k_B T/[h (2π m* k_B T)^{1/2}]; the (2π m* k_B T)^{1/2} cancels the one-dimensional translational density of states the same coordinate contributed to F‡, leaving k_B T/h.

All three give k = κ (k_B T/h) (Q‡/(Q_A Q_B)) exp(−E₀/k_B T) with Q‡ = F‡ minus the reaction-coordinate mode.

**Collision theory as a special case.** For structureless atoms A, B and a structureless complex, Q_A, Q_B are pure translation and Q‡ = translation × rotation; substituting the standard partition functions recovers k = π d² ⟨u_rel⟩ exp(−E₀/k_B T) with ⟨u_rel⟩ = (8 k_B T/π μ)^{1/2}. For real molecules, free modes of the reactants get tied up in the complex, so Q‡/(Q_A Q_B) < its structureless value — this deficit, exp(ΔS‡/R) < 1, is the steric factor.

## Worked example and reference code

```python
import math

kB = 1.380649e-23     # J/K
h  = 6.62607015e-34   # J s
NA = 6.02214076e23    # 1/mol
c  = 2.99792458e10    # cm/s  (wavenumber -> frequency)

def q_trans(m, T):                       # translation, per unit volume
    return (2.0 * math.pi * m * kB * T / h**2) ** 1.5

def q_rot_linear(I, T, sigma=1):         # linear rotor
    return 8.0 * math.pi**2 * I * kB * T / (sigma * h**2)

def q_vib_mode(nu, T):                    # one harmonic mode
    return 1.0 / (1.0 - math.exp(-h * nu / (kB * T)))

def crossing_frequency(T):               # universal attempt frequency kB*T/h
    return kB * T / h

def eyring_k(Q_dagger, Q_A, Q_B, E0, T, kappa=1.0):
    return kappa * crossing_frequency(T) * (Q_dagger / (Q_A * Q_B)) \
        * math.exp(-E0 / (kB * T))

def eyring_k_thermo(dG, T, kappa=1.0, c_std=None, molecularity=1):
    R = kB * NA
    k = kappa * crossing_frequency(T) * math.exp(-dG / (R * T))
    if c_std is not None:
        k *= c_std ** (1 - molecularity)
    return k

def collision_rate(mu, sigma_cs, E0, T):
    mean_u_rel = math.sqrt(8 * kB * T / (math.pi * mu))
    return sigma_cs * mean_u_rel * math.exp(-E0 / (kB * T))

if __name__ == "__main__":
    T = 300.0
    amu = 1.66053907e-27
    mH, mH2, mC = 1.008*amu, 2.016*amu, 3.024*amu   # H, H2, linear H..H..H complex
    I_H2, I_C = 4.6e-48, 2.7e-47                     # moments of inertia (kg m^2)
    nu_H2 = 4400.0 * c                               # H2 stretch
    nus_C = [w*c for w in (2000.0, 900.0, 900.0)]    # bound modes at the pass
    E0 = 9.7 * 4184.0 / NA                           # ~9.7 kcal/mol, J/molecule

    Q_A = q_trans(mH, T)
    Q_B = q_trans(mH2, T) * q_rot_linear(I_H2, T, sigma=2) * q_vib_mode(nu_H2, T)
    Q_d = q_trans(mC, T) * q_rot_linear(I_C, T, sigma=2)
    for nu in nus_C:                                 # reaction coordinate omitted
        Q_d *= q_vib_mode(nu, T)

    print("kB*T/h        =", crossing_frequency(T), "s^-1")  # ~6.25e12
    print("k (TST)       =", eyring_k(Q_d, Q_A, Q_B, E0, T))
    mu = mH*mH2/(mH+mH2)
    print("k (collision) =", collision_rate(mu, math.pi*(2.5e-10)**2, E0, T))
```

Running it gives k_B T/h ≈ 6.25 × 10¹² s⁻¹, and the activated-complex rate and the collision-theory rate share the same exp(−E₀/k_B T) and differ only by the structure-dependent ratio Q‡/(Q_A Q_B) versus the bare collision frequency — i.e. by the computed steric factor. The activated-complex formula thus contains collision theory as its structureless limit and supplies, from molecular data, the pre-exponential factor that the empirical rate law leaves open.
