# Context: computing absolute rates of chemical reactions from molecular structure

## Research question

Given two reactants A and B and the forces between their atoms, can we *compute*, from first principles, the rate constant k of the elementary reaction A + B → products — not just its temperature dependence, but its absolute magnitude, including the pre-exponential factor?

For half a century the temperature dependence has been captured by an empirical law of the form

    k = A · exp(-E_a / RT),

with an activation energy E_a and a pre-exponential factor A. The exponential is well understood: only the small Boltzmann fraction of encounters with energy above a threshold can react. A varies over many orders of magnitude from reaction to reaction, it carries the units that make k dimensionally correct, and there is no theory that predicts it from the molecules involved. The question is whether molecular data alone — masses, geometries, vibrational frequencies, the shape of the potential between the atoms — can supply a first-principles value for k.

## Background

**The empirical rate law.** Arrhenius (1889), building on van't Hoff (1884), wrote d ln k / dT = E_a / RT², whose integral is k = A exp(-E_a/RT). It organizes enormous amounts of kinetic data but is descriptive: E_a and A are fitted, not derived. Marcelin (1910–1915) reframed a reaction as the motion of a representative point through phase space, crossing a critical surface, and introduced an activation free energy k ∝ exp(-Δ‡G/RT); this is the right *picture* — a dividing surface separating reactants from products — but it came with no machinery to evaluate the constant in front.

**Statistical mechanics of molecules.** By the early 1930s the partition function is a mature tool. For a molecule with energy levels ε_j the partition function is q = Σ_j g_j exp(-ε_j/kT), and for nearly independent degrees of freedom it factorizes, q = q_trans · q_rot · q_vib · q_el. The standard per-unit-volume expressions are known:

- translation: q_trans = (2π m k_B T / h²)^{3/2},
- rotation (linear rotor, symmetry number σ): q_rot = 8π² I k_B T / (σ h²),
- vibration (one harmonic mode of frequency ν): q_vib = 1 / (1 − e^{−hν/k_B T}).

Equilibrium constants follow directly: equating chemical potentials μ = −k_B T ln(q/N) for an equilibrium aA + bB ⇌ pP + qQ gives K in terms of a ratio of partition functions, K = (q_P^p q_Q^q)/(q_A^a q_B^b), provided all q are referenced to a common zero of energy (one carries an extra factor e^{−ΔD₀/k_B T} for the zero-point offset ΔD₀). So *equilibrium* properties of any molecular system are computable. Rates are not — equilibrium says nothing about how fast.

**The potential energy surface and its pass.** Eyring and Polanyi (1931) constructed, for H + H₂ → H₂ + H, the first semi-empirical potential energy surface: the electronic energy as a function of the internuclear distances, built from London's valence-bond expression calibrated to spectroscopic data. On this surface the reactant valley and the product valley are separated by a *col* — a mountain pass, a saddle point: a maximum along the path joining the valleys, a minimum across it. Pelzer and Wigner (1932) studied the progress of a system over this col and were the first to make the saddle point the central object, computing a rate by following the system through the pass. The col is now an explicit, computable feature of the surface.

**A suggestive constant.** Herzfeld (1919), treating the thermal dissociation equilibrium of a diatomic molecule, obtained a rate of the form k₁ = (k_B T/h)(1 − e^{−hν/k_B T}) exp(−E/RT). The group k_B T/h — with dimensions of frequency, ≈ 6.2 × 10¹² s⁻¹ at room temperature — appeared here for the first time in a rate, attached to a vibrational degree of freedom. It was tied to that one special case and not given general meaning.

**The barrier picture, quantitatively.** A reaction proceeds because the system climbs from the reactant valley to the col and descends into the product valley. Only the Boltzmann-weighted population with enough energy to reach the col can react, which supplies the exp(−E₀/kT). Light atoms at the col have closely spaced quantized levels; the zero-point energy of the configuration at the pass differs from that of the reactants, and that difference is what makes reactions of H differ from reactions of D (the kinetic isotope effect), since the vibrational zero-point energy ½hν scales as μ^{−1/2} with reduced mass μ. Quantum tunnelling through, rather than over, a barrier of width a and height V₀ goes as P ≈ exp[−2(2m(V₀−E)/ħ²)^{1/2} a], independent of temperature — a small, mass-sensitive correction at ordinary temperatures.

## Baselines

**Collision theory (Trautz 1916; W. C. M. Lewis 1918).** Model molecules as hard spheres; a reaction happens when two of them collide with relative kinetic energy along the line of centers exceeding a threshold E*. Averaging the collision cross section σ over the Maxwell–Boltzmann distribution f(u) of relative speed u,

    k_C = ∫₀^∞ σ u f(u) du ≈ σ ⟨u_rel⟩ exp(−E*/k_B T),  ⟨u_rel⟩ = (8 k_B T / π μ)^{1/2},

gives an absolute rate: the pre-exponential factor is a collision frequency, computable from molecular diameters and masses. For reactions where the observed rate differs from k_C, a steric factor p = k_obs/k_C is recorded.

**Marcelin's phase-space activation (1910–1915).** A reaction is the passage of the system's representative point across a dividing surface in phase space; the rate is governed by an activation free energy, k ∝ exp(−Δ‡G/RT). This correctly locates the bottleneck as a surface in configuration/phase space and includes an entropic contribution to the rate, but it stops at the proportionality: there is no prescription for the constant of proportionality in terms of the molecular partition functions.

**Trajectory crossing of the col (Pelzer–Wigner 1932).** With the H + H₂ surface in hand, follow the system as it moves over the col and count the flux passing the saddle. This is the right dynamical object — flux through the pass — and it firmly identifies the saddle as where the rate is decided.

## Evaluation settings

The natural proving grounds are elementary gas-phase reactions for which a potential energy surface can be built and a rate measured. The benchmark system of the era is the hydrogen-exchange reaction H + H₂ → H₂ + H and its isotopic variants (H/D substitution), whose surface is the one available. The molecular inputs that would feed any candidate formula are: atomic masses and reduced masses; equilibrium geometries and hence moments of inertia of reactants and of the saddle configuration; vibrational frequencies of the reactants and of the bound modes at the saddle; the barrier height E₀ referenced to the reactants' zero-point level; and symmetry numbers. The metrics are the absolute rate constant k(T) and its temperature dependence (yielding an apparent activation energy and pre-exponential factor), the steric factor relative to collision theory, and the H/D kinetic isotope ratio. The standard external yardstick is the measured rate constant and the collision-theory estimate for the same reaction.

## Code framework

The pieces that already exist are the statistical-mechanical partition-function primitives and a way to read molecular data off a surface. The scaffold below makes the existing primitives concrete and leaves the rate model as an empty slot.

```python
import math

kB = 1.380649e-23     # J/K
h  = 6.62607015e-34   # J s
NA = 6.02214076e23    # 1/mol

# --- existing statistical-mechanical primitives (per unit volume) ---
def q_trans(m, T):
    return (2.0 * math.pi * m * kB * T / h**2) ** 1.5

def q_rot_linear(I, T, sigma=1):
    return 8.0 * math.pi**2 * I * kB * T / (sigma * h**2)

def q_vib_mode(nu, T):
    return 1.0 / (1.0 - math.exp(-h * nu / (kB * T)))

# --- existing collision-theory baseline ---
def collision_rate(mu, sigma_cs, E0, T):
    mean_u_rel = math.sqrt(8 * kB * T / (math.pi * mu))
    return sigma_cs * mean_u_rel * math.exp(-E0 / (kB * T))

# --- molecular data read off the potential energy surface ---
class Species:
    """Masses, moments of inertia, vibrational frequencies for a configuration
    on the surface (a reactant, or the configuration at the col)."""
    def __init__(self, mass, inertia=None, vib_freqs=()):
        self.mass, self.inertia, self.vib_freqs = mass, inertia, vib_freqs

def rate_constant(reactants, barrier_config, E0, T):
    # TODO: the rate model we will build from these molecular inputs
    pass
```
