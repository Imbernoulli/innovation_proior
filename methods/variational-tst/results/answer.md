# Generalized (variational) transition state theory: CVT, μVT, ICVT, and the unified statistical model

## Problem

Compute the thermal rate constant k(T) of an elementary reaction from a potential energy surface, correcting the systematic overestimate of conventional (saddle-point) transition state theory without resorting to full trajectory or quantum-scattering calculations.

## Key idea

Conventional TST equates the rate to the one-way equilibrium flux through a dividing surface fixed at the saddle point. Because Wigner's no-recrossing assumption counts every forward crossing as a completed reaction, this flux **over-counts** recrossing trajectories: for the classical rate on a given surface, k_TST ≥ k_true. The error is one-sided. Since the dividing surface need only separate reactants from products — the saddle point is not mandatory — every choice of surface gives an upper bound on the *same* true rate. So **vary the dividing surface and take the one that minimizes the computed flux**: the tightest upper bound, located at the genuine dynamical bottleneck, which generally is not the saddle and moves with temperature.

## Construction

Work in mass-scaled coordinates X_iγ = (m_i/μ)^{1/2} R_iγ (μ = reduced mass of relative translation), so the kinetic energy is isotropic and the N-atom system moves like one point of mass μ on V. The steepest-descent path from the saddle is the minimum energy path (MEP), parametrized by signed arc length s (s = 0 at the saddle). The family of dividing surfaces is the hypersurfaces orthogonal to the MEP, one per s — the generalized transition states (GTS).

**Generalized TST rate** through the surface at s (extends Eyring's expression; s = 0 recovers conventional TST):

    k^GT(T,s) = (σ/βh) · [Q^GT(T,s) / Φ^R(T)] · exp(−β V_MEP(s)),

β = 1/k_BT, σ = reaction-path multiplicity, Φ^R = reactant partition function per unit volume, Q^GT = GTS partition function (3N−1 bound modes, reaction coordinate removed; energy zero at V_MEP(s)).

**Canonical variational theory (CVT)** — one optimized surface per temperature:

    k^CVT(T) = min_s k^GT(T,s) = k^GT(T, s^CVT(T)).

**Equivalent free-energy form.** With the generalized free energy of activation
ΔG^GT,0(T,s) = RT [ β V_MEP(s) − ln(Q^GT(T,s)/Φ^R(T)) ] (up to the standard-state constant),

    k^GT(T,s) = (σ/βh) K° exp(−ΔG^GT,0(T,s)/RT),

so **minimizing k ⇔ maximizing ΔG^GT,0**: CVT is the maximum-free-energy-of-activation criterion. The bottleneck is the free-energy maximum (energetic term β V_MEP competing with the entropic/zero-point term −ln Q^GT), not the potential-energy maximum; it shifts with T. This is more internally consistent than conventional TST, which selects the surface from V alone but computes the rate from V and the partition functions.

**Quantization.** Bound modes are quantized (harmonic vibrational partition function per mode q_m = exp(−βħω_m(s)/2)/[1−exp(−βħω_m(s))], carrying zero-point energy); rotation kept classical; reaction-coordinate quantum effects (tunnelling) folded into a multiplicative transmission coefficient κ ≤ 1. The strict upper-bound theorem is classical; once quantized, the variational minimization is a (highly accurate) heuristic. The vibrationally adiabatic ground-state curve

    V_a^G(s) = V_MEP(s) + ½ ħ Σ_m ω_m(s)

is the effective ground-state barrier whose maximum sets the quantal threshold and through which tunnelling is computed.

**Microcanonical variational theory (μVT)** — respect energy conservation by optimizing per energy via the cumulative state count N^GT(E,s):

    N^μVT(E) = min_s N^GT(E,s),    k^μVT(T) = (σ / βh Φ^R(T)) ∫ exp(−βE) N^μVT(E) dE.

More accurate than CVT (the bottleneck moves with E); CVT is its single-surface canonical projection.

**Improved canonical variational theory (ICVT)** — microcanonical below the adiabatic ground-state threshold V^AG, canonical (truncated from below at V^AG) above: μVT's threshold accuracy at near-CVT cost.

**Unified statistical model (US/CUS)** — when N^GT(E,s) has two bottlenecks (complex-forming reactions, double-humped adiabatic barriers), the Hirschfelder-Wigner branching factor R^US (built from the two minima N_a, N_b and the maximum N_c between them) multiplies the variational result: N^US(E) = N^μVT(E) R^US(E). With one bottleneck R = 1 and it reduces to μVT. Canonically, k^CUS(T) = k^CVT(T) R^CUS(T).

**Why the bound is one-sided (grounding).** The exact rate is k(T) = (σ/hβΦ^R) ∫ Σ_a P_a(E) exp(−βE) dE with 0 ≤ P_a(E) ≤ 1 rising smoothly. The TST assumption replaces P_a(E) by a step Θ(E − threshold), which can only over-count; hence k^GT ≥ k_true for every dividing surface, and minimizing over s gives the tightest bound.

## Method (final algorithm)

1. Find the saddle point; follow steepest descent in mass-scaled coordinates to tabulate the MEP: geometries, V_MEP(s), projected bound-mode frequencies {ω_m(s)}, moments of inertia I(s).
2. Build Q^GT(T,s) = Q^GT_el · Q^GT_vib · Q^GT_rot (harmonic, separable) and Φ^R(T).
3. Form k^GT(T,s) (or ΔG^GT,0(T,s)) and minimize over s (maximize ΔG^GT,0): k^CVT(T), at s^CVT(T).
4. (Optional, more accurate) μVT/ICVT via N^GT(E,s); US/CUS for multi-bottleneck recrossing.
5. Multiply by the transmission coefficient κ (tunnelling through V_a^G).

## Reference implementation (canonical variational scan)

```python
import numpy as np

KB   = 1.380649e-23       # J/K
H    = 6.62607015e-34     # J s
HBAR = H / (2.0*np.pi)

def beta(T):
    return 1.0 / (KB * T)

def q_vib_mode(omega, T):
    """Harmonic bound mode; zero-point factor exp(-x/2) folds V_a^G into the product."""
    x = HBAR * omega * beta(T)
    return np.exp(-x/2.0) / (1.0 - np.exp(-x))

def q_gts(i, T, freqs, I_rot, qrot):
    """Q^GT(T, s_i): bound vibrational modes (reaction coordinate removed) x classical rotation."""
    qv = np.prod([q_vib_mode(w, T) for w in freqs[i]])
    return qv * qrot(I_rot[i], T)

def k_generalized(i, T, V_mep, freqs, I_rot, Phi_R, sigma, qrot):
    """k^GT(T, s_i) = (sigma / beta h)(Q^GT/Phi_R) exp(-beta V_mep)."""
    Q = q_gts(i, T, freqs, I_rot, qrot)
    return (sigma / (beta(T) * H)) * (Q / Phi_R) * np.exp(-beta(T) * V_mep[i])

def k_cvt(T, s_grid, V_mep, freqs, I_rot, Phi_R, sigma, qrot, kappa=1.0):
    """Canonical variational theory: minimize the one-way flux over the dividing
    surface location -> tightest upper bound on the rate. Returns (k, s_bottleneck)."""
    k_of_s = np.array([
        k_generalized(i, T, V_mep, freqs, I_rot, Phi_R, sigma, qrot)
        for i in range(len(s_grid))
    ])
    i_star = int(np.argmin(k_of_s))            # s^CVT(T): the variational bottleneck
    return kappa * k_of_s[i_star], s_grid[i_star]

def delta_G_activation(T, s_grid, V_mep, freqs, I_rot, Phi_R, qrot, R=8.314, Kstd=1.0):
    """Generalized free-energy-of-activation curve; its MAXIMUM is s^CVT(T).
    min_s k^GT  <=>  max_s ΔG^GT,0."""
    dG = np.empty(len(s_grid))
    for i in range(len(s_grid)):
        K = (q_gts(i, T, freqs, I_rot, qrot) / Phi_R) * np.exp(-beta(T) * V_mep[i])
        dG[i] = -R * T * np.log(K / Kstd)
    return dG

def k_conventional(T, s_grid, V_mep, freqs, I_rot, Phi_R, sigma, qrot, kappa=1.0):
    """Conventional TST: the same flux with the surface fixed at the saddle (s = 0)."""
    i0 = int(np.argmin(np.abs(s_grid)))
    return kappa * k_generalized(i0, T, V_mep, freqs, I_rot, Phi_R, sigma, qrot)
```

## Worked illustration

For a model collinear atom-diatom reaction, tabulate V_MEP(s) and {ω_m(s)} along the MEP. At high T the free-energy-of-activation curve ΔG^GT,0(T,s) peaks *past* the saddle (s^CVT > 0), because the loosening of the bound modes off the saddle raises the entropic penalty faster than V_MEP falls; placing the counting surface there lowers the computed rate. The CVT rate is the value at that maximum; the conventional rate is the value at s = 0. For collinear Cl + HD on a standard surface, moving to the variational bottleneck cuts the overestimate of the exact classical rate from ~9.6× to ~4.9× at 4000 K and ~2.8× to ~1.9× at 1500 K — the residual being recrossing that a single coordinate-space surface cannot remove (the target of μVT and the unified statistical correction).
