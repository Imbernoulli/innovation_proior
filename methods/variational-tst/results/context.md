# Context: computing thermal rate constants from a potential energy surface

## Research question

Given a Born-Oppenheimer potential energy surface V for an elementary gas-phase reaction A + B → C + D (or a high-pressure unimolecular A → C + D, A → C), compute the thermal rate constant k(T) from molecular data alone — masses, geometry, vibrational frequencies, and the shape of V — without running full classical trajectories or solving the quantum scattering problem.

The exact route (Boltzmann-average the energy-dependent reaction cross section, obtained from classical trajectories or coupled-channel quantum scattering) is, for a given surface, numerically converged only for the very simplest systems (H + H₂). For anything with more than three atoms it is impractical. A usable theory must reduce the dynamics to an equilibrium statistical-mechanics calculation on the surface — but it must do so without systematically misestimating the rate, and it must remain valid across a wide temperature range where the balance of energetic and entropic effects shifts. The pain point is precisely that the cheap, popular statistical approximation in hand (below) has a known, sometimes large, systematic error, and patching it after the fact is unsatisfactory.

## Background

**Reaction rate as flux through a dividing surface.** From the standpoint of classical dynamics, the net forward rate of an equilibrated reacting system is the flux of phase-space trajectories across a surface that separates reactants from products, counted in the product direction. For a surface fixed at coordinate value x = x₀ in a mass-scaled coordinate system (so that the same mass μ is associated with motion in every direction), the one-way equilibrium flux at total energy E is

    F_E = ∫ dΓ δ(E − H) δ(x − x₀) Θ(p_x) (p_x/μ),

i.e. the Boltzmann-weighted forward momentum density on the surface (Wigner; Horiuti, *Bull. Chem. Soc. Jpn.* 1938; Keck, *J. Chem. Phys.* 1960). The thermal rate is the Boltzmann average of the per-energy quantity over E, divided by the reactant partition function.

**Mass-scaled coordinates and the minimum energy path.** With X_iγ = (m_i/μ)^{1/2} R_iγ, the nuclear kinetic energy becomes isotropic — the N-atom system moves like a single point of mass μ on V. In such coordinates the path of steepest descent from the saddle point down to reactants and products is the intrinsic reaction coordinate; call it the minimum energy path (MEP), parametrized by signed arc length s (s = 0 at the saddle, s < 0 toward reactants, s > 0 toward products). Along the MEP one has the classical potential V_MEP(s), the bound-mode frequencies {ω_m(s)} of the 3N − 1 degrees of freedom orthogonal to the path (from the projected force-constant matrix of Miller, Handy & Adams, *J. Chem. Phys.* 1980), and the moments of inertia I(s).

**Quasi-equilibrium and quantization.** Eyring's activated-complex picture treats the species on the dividing surface as in quasi-equilibrium with reactants, with 3N − 1 ordinary bound degrees of freedom plus one special crossing mode. Quantum effects are folded in the standard way: the bound modes are assigned quantized energy levels (so the surface carries zero-point energy), and motion along the reaction coordinate is corrected by a multiplicative transmission coefficient κ ≤ 1 that captures tunnelling and recrossing. The vibrational partition function of a bound mode of frequency ω is q_vib = exp(−βħω/2)/[1 − exp(−βħω)].

**The motivating diagnostic.** When real trajectories are integrated on a surface and watched, they are seen to cross the saddle-point dividing surface more than once before committing to reactant or product (documented for collinear Cl + HD → ClH + D on the Stern-Persky-Klein surface: trajectories cross the saddle line repeatedly, then either return to reactants or exit to products). This recrossing is the empirical fact that any statistical flux theory has to contend with. A second diagnostic: for Cl + HD the conventional theory predicts the *wrong direction* of the HD/DH intramolecular kinetic isotope effect (≈1.6 versus accurate ≈0.7), a sign error, not just a magnitude error.

## Baselines

**Conventional (Eyring) transition state theory.** Place the dividing surface rigidly at the saddle point of V — the highest point on the MEP — and equate the rate to the one-way equilibrium flux through it:

    k‡(T) = (σ/βh) · [Q‡(T)/Φ^R(T)] · exp(−βV‡),

where β = 1/k_BT, σ is the reaction-path multiplicity, Φ^R is the reactant partition function per unit volume, Q‡ is the partition function of the saddle-point species with the reaction-coordinate mode removed, and V‡ is the barrier height. The universal frequency factor (βh)^{-1} = k_BT/h ≈ 6.2 × 10¹² s⁻¹ at 300 K survives from the crossing mode. This is cheap and widely used. Its **observed limitation**: because it counts every forward crossing of the saddle-point surface as a completed reaction, it over-counts — trajectories that cross forward and then come back are still tallied as reactive, so the computed flux exceeds the true rate. The discrepancy is real and sometimes large (for collinear Cl + HD the computed classical rate is a factor of ~10 too high at 4000 K, ~3 too high at 1500 K). The criterion that fixes the surface uses *only the potential energy* (the surface sits at the energetic maximum of V along the MEP), yet the rate computed on that surface depends on entropy and zero-point energy too; the surface placement and the rate calculation are built on different information. The transmission coefficient κ is the usual patch, but it is estimated separately and the saddle-point surface itself may be a poor place to count flux, especially for loose or strongly asymmetric transition states.

**Exact collision theory (the yardstick the approximation must not betray).** The rigorous rate is k(T) = (σ / hβΦ^R) ∫ Σ_a P_a(E) exp(−βE) dE, with P_a(E) the exact state-resolved reaction probability (Eliason & Hirschfelder, *J. Chem. Phys.* 1959). The transition-state approximation amounts to replacing the smoothly rising P_a(E) ≤ 1 by a step function at a threshold. This is exact only when no trajectory recrosses; otherwise it is an inequality. Accurate P_a(E) are available only for the smallest systems, so this cannot be the working method — but it is the standard against which the cheap theory's systematic error is understood.

**Keck's flux variational theory.** Keck (*J. Chem. Phys.* 1960; *Adv. Chem. Phys.* 1967) framed the reaction rate as the one-way flux through a trial dividing surface and noted that this flux is sensitive to the surface chosen. His three-body recombination calculations made the surface a free object rather than a fixed one. **Limitation as left at the time:** the practical machinery was tied to particular systems and Monte-Carlo flux evaluation; it had not been turned into a routine, partition-function-based recipe parametrized along the reaction path of a general polyatomic reaction, nor connected to the quantized, free-energy and zero-point structure that real rate calculations on a PES require.

**Miller's unified statistical theory.** Miller (*J. Chem. Phys.* 1974), with rederivations by Pollak & Pechukas, addressed reactions with more than one dynamical bottleneck (complex-forming reactions) by a statistical branching analysis (Hirschfelder-Wigner) interpolating between a single-bottleneck flux theory and phase-space theory. **Limitation:** it presumes one already has the per-bottleneck flux quantities; it is a correction layered on top of whatever single-surface theory supplies the bottleneck count, not itself a prescription for placing the surface(s).

## Evaluation settings

The natural test systems are collinear and three-dimensional atom-diatom reactions of the type A + BC → AB + C for which a potential energy surface and accurate reference rates exist: H + H₂ and isotopic analogs (D + H₂, H + D₂, Mu + H₂), Cl + HD / Cl + DH (Stern-Persky-Klein surface), Cl + HCl, F + H₂, and OH + H₂ → H₂O + H. Reference data are accurate quantal close-coupling rate constants and exact classical trajectory rate constants on the same surface, plus experimental thermal rate constants and kinetic isotope effects over a temperature range (≈200-4000 K). Metrics: ratio of the approximate to the exact rate constant on a fixed surface, and the predicted kinetic isotope effect (including its sign). Partition functions use standard forms: q_trans = (2πμ/βh²)^{3/2} per unit volume, classical rotational q_rot, harmonic q_vib per mode.

## Code framework

The pre-method scaffold below assumes a surface V and the reaction-path machinery already exist (saddle-point search, MEP following, projected normal-mode frequencies). It computes a rate constant from partition functions. The single open slot is how the dividing surface is selected and how the rate constant is formed from the per-surface quantities.

```python
import numpy as np

KB = 1.380649e-23      # J/K
H  = 6.62607015e-34    # J s
HBAR = H / (2*np.pi)

def beta(T):
    return 1.0 / (KB * T)

# --- Reaction-path data assumed available from the PES (pre-method machinery) ---
# s_grid[i]      : arc length along the minimum energy path (s=0 at saddle)
# V_mep[i]       : classical potential energy on the MEP at s_grid[i] (J), zero at reactants
# freqs[i]       : array of bound-mode angular frequencies omega_m(s_i) (rad/s), the 3N-1 modes
# I_rot[i]       : (product of) principal moments of inertia at s_i (for q_rot)
# Phi_R          : reactant partition function per unit volume (incl. relative translation)
# sigma          : reaction-path multiplicity

def q_vib_mode(omega, T):
    """Harmonic vibrational partition function for one bound mode (zero of energy at well bottom)."""
    x = HBAR * omega * beta(T)
    return np.exp(-x/2) / (1.0 - np.exp(-x))

def q_gts(i, T, freqs, I_rot):
    """Partition function of the generalized transition state located at s_grid[i]
    (bound modes only; reaction coordinate removed)."""
    qv = np.prod([q_vib_mode(w, T) for w in freqs[i]])
    qr = classical_qrot(I_rot[i], T)          # standard classical rotational partition function
    return qv * qr

def k_generalized(i, T, V_mep, freqs, I_rot, Phi_R, sigma):
    """One-way equilibrium flux rate constant for the dividing surface at s_grid[i]."""
    Q = q_gts(i, T, freqs, I_rot)
    return (sigma / (beta(T) * H)) * (Q / Phi_R) * np.exp(-beta(T) * V_mep[i])

def rate_constant(T, s_grid, V_mep, freqs, I_rot, Phi_R, sigma):
    """Thermal rate constant from the per-surface flux quantities.

    Conventional TST is the special case of fixing the surface at the saddle point
    (s = 0). The general recipe for choosing the surface and forming the rate is
    the open slot.
    """
    # TODO: decide which dividing surface(s) to use and how to combine the
    #       per-surface quantities into the reported rate constant.
    raise NotImplementedError
```
