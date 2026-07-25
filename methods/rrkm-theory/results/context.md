# Context: Energy-resolved rates of gas-phase unimolecular reactions

## Research question

A gas-phase molecule that decomposes or isomerizes on its own — a dissociation A → B + C, an isomerization, or the reverse, the recombination of two free radicals — does not obey a single rate law across all conditions. At high pressure its rate is first order in A; as the pressure of an inert bath gas M is lowered, the measured first-order rate "constant" falls off and the reaction becomes effectively second order. Any honest theory must explain this falloff and predict, from molecular properties alone, the absolute rate at every pressure and temperature.

The sharper challenge is quantitative. The qualitative falloff picture has been available for thirty years, yet when it is pushed to numbers it fails badly for real polyatomic molecules: the predicted pressure at which the rate has dropped to half its high-pressure value can be off by many orders of magnitude (for ring openings like cyclopropane the discrepancy reaches ~10⁸), and the high-pressure pre-exponential factor — the Arrhenius A — comes out wrong, sometimes far below the measured value, which can exceed 10¹⁷ s⁻¹. The mirror-image process, free-radical recombination (e.g. 2 CH₃ → C₂H₆), shows anomalous "steric factors" and its own pressure dependence; absolute recombination rates have now been measured directly (pulsed-photolysis methyl-radical experiments). A theory that could compute the dissociation rate from first principles should, by microscopic reversibility, also deliver these recombination steric and pressure effects — they are two faces of one problem.

The goal: a rate theory for unimolecular reactions with **no adjustable parameters** — built only from the molecule's vibrational frequencies and moments of inertia, the corresponding properties of the critical configuration on the reaction path, the barrier height, and the conserved constants of the motion.

## Background

The field state rests on a chain of partial theories, each fixing a defect of the last and exposing a new one.

**Pressure falloff and the Lindemann scheme.** The accepted mechanism (Lindemann 1922; Christiansen 1921) is collisional: A + M ⇌ A* + M (rate constants k₁ forward, k₋₁ back), then the energized molecule reacts, A* → products (k₂). A steady state on A* gives k_uni = k₁k₂[M]/(k₋₁[M] + k₂), which is first order at high [M] (k_∞ = k₁k₂/k₋₁) and second order at low [M] (k₀ = k₁[M]), equivalently 1/k_uni = 1/(k₁[M]) + 1/k_∞. This reproduces falloff in outline.

**Two diagnostic failures of the simple scheme.** Treating k₁, k₂ as energy-independent constants makes the model quantitatively wrong in two measurable ways. (i) The predicted falloff curve sits far above experiment, and the gap grows with molecular size — because the rate of collisional activation is badly underestimated when the molecule's many internal modes are ignored. (ii) The plot of 1/k_uni against 1/[M], predicted to be a straight line, is found experimentally to be curved (e.g. cis–trans isomerization of dideuteroethylene) — because the energized molecules do not all react at one rate.

**Activation over many modes (Hinshelwood, 1926).** Modelling the molecule as s equivalent classical harmonic oscillators, the chance of accumulating energy ≥ E₀ across s modes is far larger than for one mode, raising k₁ to k₁ = (Z/(s−1)!)(E₀/k_BT)^{s−1} exp(−E₀/k_BT), with Z the collision frequency. This repairs failure (i). It leaves k₂ energy-independent, so failure (ii) survives; and s must be fitted (typically about half the normal modes), with no a-priori prescription, so the fit holds only near the middle of the falloff curve.

**Energy localization in a reactive mode (RRK: Rice & Ramsperger 1927, classical; Kassel 1928, quantum).** Reaction needs a critical quantity of energy E₀ collected in one particular mode, and energy flows freely among the modes. The degeneracy of v quanta among s oscillators is g_v = (v+s−1)!/[v!(s−1)!] (the number of ways to place v indistinguishable quanta in s modes). The probability of finding at least m quanta (E₀ = mhν) in the reactive mode is the ratio P = v!(v−m+s−1)! / [(v−m)!(v+s−1)!], which in the classical limit (v, m ≫ s) collapses to (1 − E₀/E)^{s−1}. Hence the reaction step acquires an energy dependence, k₂(E) = A(1 − E₀/E)^{s−1}, with A of order a vibrational frequency. This repairs failure (ii): higher-energy molecules react faster, and larger molecules (bigger s) react more slowly at given E. The remaining limitations are structural: all oscillators are assigned one common frequency; the treatment is essentially classical; A and s are not computed from the molecule; the "reactive mode" is a stand-in with no explicit critical molecular configuration of its own; rotation and the conservation of angular momentum are absent; and the pre-exponential is capped near a single vibrational frequency (~10¹³–10¹⁴ s⁻¹), below what some reactions show.

**The activated complex for thermal rates (Eyring 1935; Evans–Polanyi 1935).** Independently of the falloff problem, the absolute thermal rate of a reaction was expressed through the configuration of highest energy along the reaction path — the activated complex at the saddle point. With the reactants and that complex in quasi-equilibrium, and the one motion that carries the system over the barrier (the reaction coordinate) treated as a free translation through a thin region, the thermal rate constant is k = κ (k_BT/h)(Q‡/Q) exp(−E₀/k_BT). Here Q is the reactant partition function, Q‡ the partition function of the activated complex with the reaction-coordinate mode removed, h Planck's constant, κ a transmission coefficient, and the factor k_BT/h is the universal frequency at which the complex crosses the barrier. Q‡ is built from the **real** vibrational frequencies and moments of inertia of the saddle configuration. The limitation for the present problem: this is a thermal (constant-temperature) result. It assumes a full Boltzmann population at every energy, true only at high pressure; it yields k_∞ but is silent about the energy-resolved reaction probability that controls the entire falloff regime.

So two strong but separate ideas are on the table — RRK's energy-resolved counting of how energy is distributed among modes, and Eyring's use of the actual structure of the critical configuration — and a measured phenomenon (falloff, anomalous recombination steric factors) that neither, alone, captures quantitatively.

## Baselines

- **Lindemann–Christiansen scheme.** Two-step collisional mechanism with energy-independent k₁, k₂; SSA gives k_uni = k₁k₂[M]/(k₋₁[M] + k₂). Reproduces qualitative falloff. Gap: predicted half-falloff pressure wrong by orders of magnitude for polyatomics, and the 1/k_uni vs 1/[M] line is curved in experiment — the model cannot represent that activation spreads over many modes nor that different energies react at different rates.

- **Hinshelwood theory.** Activation over s classical oscillators raises k₁ to (Z/(s−1)!)(E₀/k_BT)^{s−1} exp(−E₀/k_BT). Gap: the reaction step k₂ is still energy-independent, so the falloff shape stays wrong away from the half-point; s is a fitted number with no first-principles value.

- **RRK / Kassel theory.** Energy-dependent reaction step k₂(E) = A(1 − E₀/E)^{s−1} from counting quanta over s equal-frequency oscillators (classical RR; quantum Kassel). Gap: one common frequency for all modes; A and s not derivable from the molecule; no explicit critical configuration with its own frequencies and geometry; no rotational/angular-momentum treatment; pre-exponential cannot exceed roughly a vibrational frequency.

- **Eyring / Evans–Polanyi transition-state theory.** Thermal absolute rate k = κ(k_BT/h)(Q‡/Q) exp(−E₀/k_BT) using the real partition function of the activated complex. Gap: canonical (fixed T) only — assumes a complete Boltzmann distribution at all energies, valid at high pressure, and therefore cannot by itself describe the pressure-dependent falloff, which is governed by how the reaction rate varies with the molecule's internal energy.

## Evaluation settings

The natural testing ground is the set of gas-phase unimolecular reactions whose falloff has been mapped: small-ring openings and isomerizations (cyclopropane → propene, cyclobutane, methylcyclobutane, methyl isocyanide CH₃NC → CH₃CN, the cis–trans isomerization of CHD=CHD), simple dissociations into radicals, and the radical recombinations measured by pulsed photolysis (methyl-radical combination 2 CH₃ → C₂H₆, ethyl cracking). The observables are: the first-order rate constant k_uni as a function of bath-gas pressure [M] and temperature (the falloff curve and its limits k_∞, k₀); the half-falloff pressure [M]₁/₂; the high-pressure Arrhenius parameters (A and activation energy); and, for the reverse direction, the absolute recombination rate and its apparent steric factor. Molecular inputs assumed available: vibrational frequencies and moments of inertia of the stable molecule and of the critical configuration, and the barrier height E₀.

## Code framework

The computational substrate is direct enumeration of quantum states on an energy grid for a set of harmonic oscillators, then assembling those counts into a rate. The state-counting primitive (a Beyer–Swinehart-style direct count) and the physical constants are known; what is missing is the object that turns molecular state counts into the energy-resolved rate.

```python
import numpy as np

# Physical constants (known)
H_PLANCK = 6.62607015e-34   # J s
C_LIGHT  = 2.99792458e10    # cm/s   (for cm^-1 -> s^-1)
K_B      = 1.380649e-23     # J/K

def count_states_up_to(freqs_cm, e_grid_cm, kind):
    """Direct count of harmonic-oscillator states on an energy grid.

    freqs_cm : list of oscillator frequencies (cm^-1)
    e_grid_cm: uniform energy grid (cm^-1), spacing = grain
    kind     : 'sum'  -> cumulative number of states up to each energy
               'density' -> number of states per grain at each energy
    Returns an array over e_grid_cm.
    """
    grain = e_grid_cm[1] - e_grid_cm[0]
    n = len(e_grid_cm)
    if kind == 'sum':
        T = np.ones(n)              # one (the ground configuration) at every energy
    else:
        T = np.zeros(n); T[0] = 1.0 # density seed: a single state at zero
    for nu in freqs_cm:
        step = int(round(nu / grain))
        if step == 0 or step >= n:
            continue
        for e in range(step, n):
            T[e] += T[e - step]     # add configurations with one more quantum in this mode
    return T

def microcanonical_rate(E_cm, molecule_freqs, ts_freqs, E0_cm, e_grid_cm):
    """Energy-resolved specific rate k(E) for a unimolecular reaction.

    molecule_freqs : vibrational frequencies of the energized molecule (cm^-1)
    ts_freqs       : frequencies of the critical configuration on the reaction
                     path, reaction-coordinate mode removed (cm^-1)
    E0_cm          : barrier height (cm^-1)
    Returns k(E) in s^-1.
    """
    # TODO: the object we will define here -- how to combine state counts of the
    #       energized molecule and of the critical configuration into a rate.
    pass

def thermal_rate(T_kelvin, *args):
    """Boltzmann average of the energy-resolved rate -> k(T)."""
    # TODO: average the energy-resolved rate over the thermal population.
    pass
```
