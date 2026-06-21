# Marcus theory of electron transfer (Part I): the activation free energy from solvent reorganization

## Problem

Compute, with no adjustable parameters, the rate of an elementary electron-transfer (redox) reaction
between two ions in solution — a reaction in which **no chemical bonds break or form**. Classical
absolute-rate theory has no reaction coordinate for such a reaction (nothing stretches), and
equilibrium continuum electrostatics (Born, Coulomb) assumes the solvent polarization is always in
equilibrium with the instantaneous charges — exactly the assumption that fails at the instant of
transfer, because the heavy solvent cannot follow the electron jump.

## Key idea

The electron jumps with the nuclei frozen (Franck–Condon) and the reaction proceeds **in the dark**,
so the jump must conserve energy: it can only occur at a nuclear configuration where the energy of the
system with the electron on the donor equals the energy with it on the acceptor. The reactants reach
this energy-matched configuration by a **thermal fluctuation of the slow nuclear (orientation +
atomic) polarization of the solvent** away from its equilibrium value — that fluctuation, not any bond
motion, *is* the activation step. The reaction coordinate is the slow solvent polarization; the
transition state is the most probable nonequilibrium-polarization state consistent with the
equal-energy condition.

## Method (derivation outline)

1. **Weak overlap.** With slight donor–acceptor orbital overlap the activated complex's wavefunction
   is φ_X + c φ_X\*; it is a legitimate eigenstate **only if** the reactant-configuration energy
   equals the product-configuration energy (E_X = E_X\*). The equal-energy (Franck–Condon + energy
   conservation) condition is forced, not assumed.

2. **Free energy of a nonequilibrium-polarization state.** Split the polarization P = P_e + P_u into a
   fast E-type part (electronic, P_e = α_e E, always equilibrated to the instantaneous field) and a
   slow U-type part (orientation + atomic, P_u(r), an independent degree of freedom). With E_c the
   field the charges would exert in vacuum, a reversible two-stage charging (first build the target
   P_u with fictitious charges; then charge the real ions at fixed P_u) gives

   F = ½ ∫ { E_c·E_c/(4π) − P·E_c + P_u·( P_u/α_u − E ) } dV.

   Letting P_u relax to α_u E recovers ordinary equilibrium electrostatics (Born/Coulomb).

3. **Constrained minimization.** Minimize the free energy of forming X\* over P_u(r), subject to the
   equal-energy restraint, with Lagrange multiplier m:

   δF\* = ∫ ( P_u/α_u − E\* )·δP_u dV,   restraint δF\* − δF = 0
   ⇒  P_u = α_u { E\* + m (E\* − E) }.

4. **Close in terms of vacuum fields** (using E − E\* = (E_c − E_c\*)/D_op and the analogous reduction
   for E\*, with 4π(α_e+α_u)=Dₛ−1, 4πα_e=D_op−1), integrate the two-sphere vacuum fields, and subtract
   the isolated-ion work. The free energy of activation beyond the bare Coulomb term is m²λ, with the
   **reorganization energy**

   λ = (Δe)² ( 1/(2a₁) + 1/(2a₂) − 1/R )( 1/D_op − 1/Dₛ ).

5. **The multiplier is the driving force.** The restraint gives −(2m+1)λ = ΔG° (so m = −½ for a
   symmetric self-exchange, ΔG° = 0), i.e. m = −½ − ΔG°/(2λ). Substituting m² λ = (λ + ΔG°)²/(4λ).

## Final result

Free energy of activation (work terms aside):

ΔG‡ = ( λ + ΔG° )² / ( 4 λ )

Outer-sphere reorganization energy (two-sphere dielectric continuum):

λ_o = (Δe)² ( 1/(2a₁) + 1/(2a₂) − 1/R )( 1/D_op − 1/Dₛ )

Inner-shell contribution (harmonic, reduced force constants k_j = 2 f_j^r f_j^p/(f_j^r+f_j^p)):

λ_i = ½ Σ_j k_j (Δq_j)² ,   and   λ = λ_i + λ_o.

Rate constant (small-overlap kinetics reduce the elementary-step sequence to the rate of forming the
energy-matched intermediate):

k = Z · exp( −ΔG‡ / kT ),   Z = collision number in solution.

Consequences, parameter-free:
- **Self-exchange** (ΔG° = 0): ΔG‡ = λ/4. λ shrinks with ion size, so large complex ions are fast,
  small cations slow; the ordered, polarization-frozen intermediate explains the large negative
  activation entropy between like-signed ions.
- **Normal region** (−ΔG° < λ): more exergonic ⇒ smaller barrier (slope ½ in −ΔG° near ΔG° = 0).
- **Activationless** at −ΔG° = λ: ΔG‡ = 0.
- **Inverted region** (−ΔG° > λ): more exergonic ⇒ *larger* barrier, slower reaction.
- **Cross-relation:** additivity λ₁₂ ≈ (λ₁₁ + λ₂₂)/2 gives k₁₂ ≈ (k₁₁ k₂₂ K₁₂ f)^{1/2},
  ln f = (ln K₁₂)² / [ 4 ln(k₁₁ k₂₂ / Z²) ].
- **Optical counterpart:** a vertical (light-driven) transition at the reactant minimum has
  hν_max = λ — the same λ governs the thermal barrier and the charge-transfer absorption band.

## Worked numerical example

```python
import numpy as np

# constants and water at 25 C
kT   = 0.593      # kcal/mol
D_s  = 78.5       # static dielectric constant
D_op = 1.8        # optical dielectric constant
E2   = 332.06     # e^2 in kcal*Angstrom/mol

def lambda_o(dq, a1, a2, R):
    """Outer-sphere reorganization energy (the parabola offset)."""
    g = (1.0/(2*a1) + 1.0/(2*a2) - 1.0/R)
    return E2 * dq**2 * g * (1.0/D_op - 1.0/D_s)

def lambda_inner(force_const, dq_bond):
    """Inner-shell reorganization: 1/2 sum_j k_j (dq_j)^2."""
    return 0.5 * np.sum(np.asarray(force_const) * np.asarray(dq_bond)**2)

def dF_star(lam, dG0, coulomb=0.0):
    """Activation free energy: Coulomb work + (lambda + dG0)^2 / (4 lambda)."""
    return coulomb + (lam + dG0)**2 / (4.0 * lam)

def rate(lam, dG0, Z=1e11, coulomb=0.0):
    return Z * np.exp(-dF_star(lam, dG0, coulomb) / kT)

# small aqueous self-exchange, unit charge transferred, ions at contact
a1 = a2 = 3.0
R   = a1 + a2
lam = lambda_o(dq=1.0, a1=a1, a2=a2, R=R)
print(f"lambda_o = {lam:.1f} kcal/mol;  self-exchange barrier lambda/4 = {lam/4:.1f} kcal/mol")

for dG0 in [0.0, -lam/2, -lam, -1.5*lam, -2.0*lam]:
    print(f"dG0 = {dG0:7.1f}  ->  dF* = {dF_star(lam, dG0):6.2f} kcal/mol")
# barrier vanishes at dG0 = -lambda (activationless); rises again for dG0 < -lambda (inverted region).
```

Output:

```
lambda_o = 30.0 kcal/mol;  self-exchange barrier lambda/4 = 7.5 kcal/mol
dG0 =     0.0  ->  dF* =   7.51 kcal/mol
dG0 =   -15.0  ->  dF* =   1.88 kcal/mol
dG0 =   -30.0  ->  dF* =   0.00 kcal/mol
dG0 =   -45.1  ->  dF* =   1.88 kcal/mol
dG0 =   -60.1  ->  dF* =   7.51 kcal/mol
```

The barrier falls through the normal region to zero at −ΔG° = λ and rises symmetrically in the
inverted region — the qualitative signature of ΔG‡ = (λ + ΔG°)²/4λ.
