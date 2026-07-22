Let me start where the trouble actually is. A free radical recombination — two methyl radicals snapping together to make ethane, 2 CH₃ → C₂H₆ — has just been measured directly by pulsed photolysis, and the number that comes out has an awkward shape to it: there is an apparent "steric factor," the rate is lower than a naive collision estimate, and it depends on pressure. People want to read that as geometry, two radicals having to find the right orientation. But I don't believe the steric factor is really about orientation. Recombination is the exact reverse of a dissociation — C₂H₆* falling apart into two methyls — and dissociation rates I think I can compute. If I can get the dissociation right, microscopic reversibility hands me the recombination, steric factor and all, for free. So let me forget the recombination for now and stare at the unimolecular dissociation, because that is the thing I can actually build a theory of.

The dissociation has the pressure problem everyone knows. At high bath-gas pressure the rate is first order in the reactant; drop the pressure and the first-order "constant" falls off and the kinetics slide toward second order. The Lindemann picture explains the shape: a molecule gets energized by collision, A + M ⇌ A* + M with forward and back constants k₁ and k₋₁, and the energized A* then reacts, A* → products with constant k₂. Steady state on A* gives

  k_uni = k₁ k₂ [M] / (k₋₁ [M] + k₂).

At large [M] this is k_∞ = k₁ k₂ / k₋₁, first order; at small [M] it is k₁[M], second order. Fine as a cartoon. But when I try to make it a number it is hopeless. The half-falloff pressure — where the rate has dropped to half of k_∞ — comes out wrong by enormous factors for any real polyatomic, growing worse the bigger the molecule, up toward eight orders of magnitude for something like cyclopropane. And the diagnostic plot, 1/k_uni against 1/[M], which the formula says must be a straight line, is curved in the data. Two failures, and they point at the two energy-independent constants k₁ and k₂.

Take k₁ first. Hinshelwood already saw the fix: a polyatomic has many internal modes, and the probability of piling up energy ≥ E₀ across s oscillators is hugely larger than across one. Counting s classical oscillators gives

  k₁ = (Z / (s−1)!) (E₀/k_BT)^{s−1} exp(−E₀/k_BT),

with Z the collision frequency. The factor (E₀/k_BT)^{s−1} is gigantic because E₀ ≫ k_BT, and it is exactly what was needed to raise the activation rate for big molecules. So the first failure is understood. But this leaves k₂ a constant, and the second failure is untouched — the 1/k_uni vs 1/[M] line is still curved, because in reality the energized molecules don't all react at one rate.

So the real action is in k₂: it must depend on the energy of the molecule. RRK said why. Reaction needs a critical amount of energy E₀ localized in one particular mode — the bond that breaks — and energy sloshes freely among the modes, so the question is purely combinatorial: given total energy E spread over the modes, what is the chance that at least E₀ of it sits in the reactive one? Count quanta. The number of ways to put v indistinguishable quanta into s oscillators is the number of ways to arrange v balls and s−1 movable walls,

  g_v = (v + s − 1)! / [ v! (s − 1)! ].

To have at least m quanta (E₀ = m h ν) in the reactive mode, fix m of them there and distribute the rest; the count of favorable arrangements is (v − m + s − 1)!/[(v − m)!(s − 1)!], and the probability is the ratio

  P = (v − m + s − 1)! v! / [ (v − m)! (v + s − 1)! ].

In the classical limit v, m ≫ s the factorials telescope — (v−m+s−1)!/(v−m)! → (v−m)^{s−1}, v!/(v+s−1)! → v^{−(s−1)} — so

  P → (v − m)^{s−1} / v^{s−1} = (1 − m/v)^{s−1} = (1 − E₀/E)^{s−1},

and the reaction step becomes k₂(E) = A (1 − E₀/E)^{s−1}. Higher energy reacts faster; bigger s reacts slower at fixed E. This is the right qualitative cure for the second failure, and folded back into the Lindemann steady state it bends the falloff and the 1/k vs 1/[M] plot the right way.

But now I have to be honest about what RRK is actually assuming, because this is where it stops being predictive. It gives all s oscillators a single common frequency ν. It is essentially a classical count (Kassel's quantum version still keeps the equal frequencies). The amplitude A and the number s are not computed from anything — s gets fitted, usually to about half the real modes, and A is taken to be "of order a vibrational frequency." And there is no actual molecular configuration at the moment of reaction: the "reactive mode" is a bookkeeping device, not a structure with its own geometry and its own frequencies. That last one bothers me most. When a bond is breaking, the molecule is not the molecule anymore — some of its modes have gone soft, the geometry has opened up, and the vibrational frequencies at that critical configuration are completely different from the frequencies of the stable molecule. RRK can't see any of that. And the pre-exponential it predicts can never get much above a single vibrational frequency, ~10¹³–10¹⁴ s⁻¹, yet some reactions show A as high as 10¹⁷. A number that large is screaming that the critical configuration is very loose — lots of low frequencies — and that there are far more ways to be "about to react" than RRK's single-frequency count allows.

There is a separate idea that knows exactly how to talk about the critical configuration: the activated-complex theory. For the thermal rate, you put the reactants and the configuration of highest energy along the path — the activated complex at the saddle — in quasi-equilibrium, and you let the one motion that carries the system across the barrier, the reaction coordinate, be a free translation through a thin slab. The thermal rate constant comes out

  k = κ (k_BT/h) (Q‡ / Q) exp(−E₀/k_BT),

where Q is the reactant partition function, Q‡ the partition function of the activated complex with the reaction-coordinate mode stripped out, and the universal factor k_BT/h is the frequency of crossing. The beauty is that Q‡ is built from the real frequencies and moments of the saddle configuration — exactly the structural information RRK throws away. A loose complex with soft modes gives a large Q‡ and a large A, naturally, with no cap. So this theory has the structure RRK lacks.

But it is a thermal theory. It assumes a full Boltzmann distribution over energies — true only at high pressure — and gives me k_∞. It says nothing about the energy-resolved reaction probability that runs the whole falloff. So I have two halves: RRK knows how to resolve by energy but not how to describe the critical configuration; activated-complex theory knows the configuration but only at fixed temperature.

The move is to fuse them: do the activated-complex construction, but at fixed energy instead of fixed temperature, and count the states quantum-mechanically using the real frequencies of both the molecule and the critical configuration. Let me try to actually derive what comes out, because I want it to be parameter-free.

Set up the Lindemann scheme energy-resolved. The molecule is energized into a narrow band [E, E+dE] with some energization rate, deactivated by collisions at k₋₁, and reacts at an energy-dependent k₂(E):

  A + M ⇌ A*(E)  (forward dk₁(E→E+dE), back k₋₁),   A*(E) → products at k₂(E).

I want two things: the equilibrium population in the band, and k₂(E) itself.

The population first. In equilibrium the ratio dk₁(E→E+dE)/k₋₁ is just an equilibrium constant — the fraction of molecules sitting in the band [E, E+dE]. Statistical mechanics gives that directly: it is the partition-function weight of that band over the whole partition function,

  dk₁(E→E+dE) / k₋₁ = Q(A*_{[E,E+dE]}) / Q(A).

Now the partition function of a thin energy band is (number of states in the band) times the Boltzmann factor at that energy. The number of states in [E, E+dE] is ρ(E) dE where ρ(E) is the density of states of the molecule. So

  dk₁(E→E+dE) / k₋₁ = [ ρ(E) dE / Q(A) ] exp(−E/k_BT).

Good — that's the quantum Boltzmann distribution over energy, and it already contains the real density of states ρ(E), not a single-frequency surrogate. (I am taking strong collisions here: k₋₁ is the collision frequency, every collision fully deactivates. I'll come back to whether that's safe.)

Now the reaction step k₂(E). This is where I have to build the activated-complex picture at fixed energy. Write the reaction as the energized molecule passing into the critical configuration A‡ and then over the top: A*(E) → A‡ → products. Steady state on A‡ gives the rate of crossing as

  k₂(E) = (rate the complex crosses the barrier forward) × [A‡]/[A*].

Let me get each factor.

The crossing rate. At the dividing surface the reaction coordinate is a translation. Picture it as a particle of mass m moving in a thin box of length δ, with kinetic energy x along the coordinate. Its velocity is v = (2x/m)^{1/2}, and the rate at which it traverses the slab and emerges as product is v/δ — call this k‡ = (2x/(mδ²))^{1/2}. But only the molecules moving toward products cross; in equilibrium half are moving each way. So I keep a factor ½:

  k₂(E) = ½ k‡ [A‡]/[A*].

The ratio [A‡]/[A*]. The energized molecule and the complex are both at the same total energy in the same band [E, E+dE], with the same zero of energy. So their relative populations are the ratio of how many states each has there — at chemical energies, the ratio of densities of states. For the complex I must split off the reaction-coordinate mode, which is the translation carrying energy x; the rest of the complex's energy, E‡, is in its internal (vibrational–rotational) modes. So

  [A‡]/[A*] = P(E‡) N‡(x) / ρ*(E*),

where P(E‡) is the number of internal quantum states of the complex with non-fixed energy E‡, N‡(x) is the number of translational states of the reaction coordinate at energy x, and ρ*(E*) is the density of internal states of the energized molecule at its active energy E*. The available energy splits as E* = E‡ + x, and E* itself is what's left after the barrier: E* = E − E₀.

Now combine. I have

  k₂(E) = ½ k‡ P(E‡) N‡(x) / ρ*(E*).

I need ½ k‡ N‡(x). The translational states in a 1-D box of length δ: the number of states with momentum up to p is 2 (δ/h) p, the factor 2 for ±p. Per unit translational energy x, with x = p²/2m so dp = (m/2x)^{1/2} dx, the number of translational states per dx is (d/dx)[2δp/h] = 2δ/h · (m/2x)^{1/2}. The ½ in front cancels the factor 2 (forward-movers only), leaving (δ/h)(m/2x)^{1/2} states per unit x. Multiply by the crossing rate k‡ = (2x/(mδ²))^{1/2}:

  ½ k‡ × (translational states per unit x) = (2x/(mδ²))^{1/2} × (δ/h)(m/2x)^{1/2}
                                          = (1/h) × [ (2x/(mδ²)) (mδ²/2x) ]^{1/2}
                                          = (1/h).

Everything cancels — the box length δ, the mass m, the energy x — and what's left is exactly 1/h. That is the whole point: the crossing rate times the spacing of the translational states is a pure constant, Planck's constant, the same constant that shows up as k_BT/h in the thermal theory, only now it has appeared at fixed energy. So summing the contributions over all ways to partition E* between the reaction-coordinate energy x and the internal energy of the complex E‡ turns into (1/h) times a count over E‡:

  k₂(E) = (1/(h ρ*(E*))) Σ_{E‡=0}^{E*} P(E‡).

And Σ_{E‡=0}^{E*} P(E‡) is just the cumulative number of internal states of the critical configuration with energy up to the available energy E* = E − E₀. Call that sum N‡(E*) = N‡(E − E₀). So

  k(E) = N‡(E − E₀) / ( h ρ(E) ).

There it is. The specific rate at energy E is the total number of open channels at the critical configuration — every internal state of the complex with energy ≤ E − E₀ leaves enough left over for the reaction coordinate, so every one of them is a doorway through which the system can leave — divided by Planck's constant times the density of states of the molecule at E. The numerator is a sum of states (a cumulative count); the denominator is a density of states. I should make sure I never confuse those: the sum belongs upstairs because the rate is the count of how many doorways are open; the density belongs downstairs because I prepared the molecule in a thin band and the rate per molecule is flux divided by how many molecules are in that band.

Let me check it isn't nonsense at the edges. For E < E₀, the available energy E − E₀ is negative, N‡ counts nothing, k(E) = 0 — no reaction below the barrier, correct. Just above E₀ only the lowest internal state of the complex is accessible, N‡ = 1, so k(E) = 1/(h ρ(E)) — a single open channel, a slow rate set by how dense the molecule's states are. As E climbs, N‡ grows fast (many internal configurations of the complex become accessible) while ρ(E) grows too, and the rate rises. Good shape.

Now the real test: does this collapse to the two theories I already trust? Take the thermal average and see if Eyring's k_∞ falls out. The thermal rate at high pressure is the energy-resolved rate weighted by the equilibrium population:

  k_∞ = Σ_E k(E) ρ(E) exp(−E/k_BT) / Σ_E ρ(E) exp(−E/k_BT).

The numerator: k(E) ρ(E) = N‡(E − E₀)/h, so the numerator is (1/h) Σ_E N‡(E − E₀) exp(−E/k_BT). Pull out the barrier, E = E₀ + E*, exp(−E/k_BT) = exp(−E₀/k_BT) exp(−E*/k_BT):

  numerator = (1/h) exp(−E₀/k_BT) Σ_{E*} N‡(E*) exp(−E*/k_BT).

Now Σ_{E*} N‡(E*) exp(−E*/k_BT), with N‡ a cumulative sum of states — its Laplace/Boltzmann transform is, up to the k_BT from summing the cumulative count, the partition function of the complex: Σ_{E*} N‡(E*) exp(−E*/k_BT) = k_BT · Q‡ (the cumulative sum transforms to the density transform divided by the Boltzmann variable, i.e. multiply Q‡ by k_BT). The denominator Σ_E ρ(E) exp(−E/k_BT) = Q, the reactant partition function. So

  k_∞ = (1/h) exp(−E₀/k_BT) k_BT Q‡ / Q = (k_BT/h)(Q‡/Q) exp(−E₀/k_BT).

That is precisely the activated-complex thermal rate. The microcanonical theory contains the canonical one as its high-pressure thermal average, and the universal k_BT/h emerges from the very same 1/h that I found in the crossing-times-spacing cancellation. And if I were to flatten all the frequencies to a single common ν and count classically, N‡/ρ would reduce to (1 − E₀/E)^{s−1} — RRK reappears as the equal-frequency classical special case. So I haven't replaced the old theories; I've found the parent both descend from.

Fold k(E) into the full pressure dependence to confirm the falloff. Steady state on A*(E) with the energy-resolved energization, deactivation, and reaction gives, integrating over the band populations,

  k_uni = ∫_{E₀}^{∞} [ (ρ(E)/Q) exp(−E/k_BT) k(E) / (1 + k(E)/ω) ] dE,

where ω = k₋₁[M] is the collision frequency. At high [M], ω ≫ k(E), the 1 + k/ω → 1 in a way that returns the Boltzmann-averaged k(E), i.e. k_∞ above; at low [M], ω ≪ k(E), the integrand becomes (ρ(E)/Q) exp(−E/k_BT) ω, proportional to [M], so the rate is second order and equals the rate of energization above the barrier. Both limits right, and in between the curve is set by where k(E) crosses ω, which is exactly the energy-resolved competition the simple Lindemann model couldn't represent. No fitted s, no fitted A — the inputs are the molecule's frequencies, the complex's frequencies, and E₀.

Two refinements I shouldn't gloss over, because the molecule has rotations and sometimes several ways to react.

Rotations and angular momentum. I lumped "internal states" together, but not every internal mode behaves the same during reaction. The overall rotation of the molecule carries angular momentum, which is conserved on the timescale of the reaction — it cannot dump its energy into the vibrations and back the way the vibrations exchange among themselves. So I should split the modes: the **active** ones, which freely share energy and which I count in N‡ and ρ, and the **adiabatic** ones — overall rotation, tied to angular-momentum conservation — whose quantum number rides through the reaction unchanged. If I wrongly counted the rotation as active I'd inflate the state count. The clean way: the adiabatic rotational energy is fixed by the conserved angular momentum and shifts the effective barrier — it is the centrifugal energy at the configuration — so the energy available to the active modes is reduced by the rotational energy of the complex. The properly resolved rate is

  k(E, J) = N‡( E − E₀ − E_rot‡(J) ) / ( h ρ( E − E_rot(J) ) ),

with N‡ and ρ now counting only active modes, and E_rot the adiabatic rotational energies of complex and molecule at angular momentum J. To turn this into k(E) I average over J with the rotational degeneracy (2J+1) and the rotational Boltzmann factor exp(−J(J+1)ℏ²/(8π²I k_BT)); equivalently the adiabatic rotations enter as a ratio of rotational partition functions Q‡_rot/Q_rot multiplying the active-mode result. This is what makes the high-pressure limit land exactly on the activated-complex k_∞ including its rotational partition functions.

Reaction-path degeneracy. If the molecule can react by several equivalent routes — equivalent bonds, equivalent hydrogens — there are that many equivalent critical configurations, and I should multiply by the number of them, a statistical factor L‡ that is really a ratio of symmetry numbers (molecule to complex). So the general form is

  k(E) = L‡ (Q‡_rot/Q_rot) (1/(h ρ*(E*))) Σ_{E‡} P(E‡).

One thing I must get right numerically: at energies just above the barrier, the density of states of the complex is NOT a smooth continuum. For something like cyclopropane the energized molecule has ~10⁹ states per kcal at its excitation energy, but the complex, with most of its energy locked into clearing the barrier, has only ~10² — that is a discrete handful of states, and pretending it's a continuous function would be badly wrong. So I cannot use the classical continuous-partition-function approximation for N‡; I have to count the quantum states one by one. That's the whole reason this theory is quantitative where RRK is not: real frequencies, counted exactly.

Let me convince myself by hand on a small case so I trust the counting picture. Take HCO• → CO + H, a three-mode molecule. The barrier is about 68.2 kJ/mol, which in vibrational quanta is roughly E₀ ≈ 5700 cm⁻¹. The three modes are an HCO bend at ~1145, a CO stretch at ~1900, and a CH stretch at ~2750 cm⁻¹. The bond that breaks is the C–H, so it's the CH-stretch quanta that have to pile up. I enumerate states by their quantum numbers (n₁, n₂, n₃) and their energies n₁·1145 + n₂·1900 + n₃·2750. The lowest several have energy below 5700 — those can't react at all, k = 0 there. Above 5700 the molecule has enough total energy in principle. But to actually break the C–H bond I need enough quanta in mode 3: two quanta is 2×2750 = 5500, still under the barrier, not enough; three quanta is 3×2750 = 8250, over the barrier, enough. So the states that are genuinely "ready to go" are the specific ones with sufficient C–H excitation — and there are only a few of them low down. N‡(E − E₀) is the running count of accessible complex configurations as I raise E; ρ(E) is how thickly the molecule's own states are packed at E; and k(E) = N‡/(hρ) is their ratio times 1/h. Doing it by hand for three modes makes it concrete that the numerator is a cumulative count of open doorways and the denominator a local density — exactly the formula.

So how do I count states for a real molecule with many modes, where I can't list them by hand? I need the cumulative sum N‡(E) and the density ρ(E) for a set of harmonic oscillators on an energy grid. The direct way: lay down an energy grid in grains, and convolve one oscillator in at a time. For each oscillator of frequency ν, occupying it with one more quantum shifts a configuration up by hν; so on the grid I sweep upward and add, at each energy, the count from one quantum lower in that mode. Start the sum-of-states array as all ones (the bare cumulative count before any oscillator is folded in is "one configuration at or below every energy") and the density array as a single one at zero; fold each frequency in by the same add-from-below recursion. That builds the exact harmonic state counts with no continuous approximation, which is exactly what I argued I need.

Putting it together as code:

```python
import numpy as np

# Physical constants
H_PLANCK = 6.62607015e-34   # J s
C_LIGHT  = 2.99792458e10    # cm/s   (cm^-1 -> s^-1 via h*c)
K_B      = 1.380649e-23     # J/K

def count_states(freqs_cm, e_grid_cm, kind):
    """Direct (Beyer-Swinehart) count of harmonic-oscillator states on a grid.

    kind='sum'     -> N(E), cumulative number of states up to each energy
    kind='density' -> rho(E), states per grain at each energy
    The recursion folds one oscillator in at a time: occupying a mode with one
    more quantum raises a configuration by h*nu, so we add the count from one
    quantum lower in that mode (this is the 'energy flows among the modes' /
    quanta-distribution count, done exactly rather than classically).
    """
    grain = e_grid_cm[1] - e_grid_cm[0]
    n = len(e_grid_cm)
    if kind == 'sum':
        T = np.ones(n)               # cumulative count seed
    else:
        T = np.zeros(n); T[0] = 1.0  # density seed: one state at zero energy
    for nu in freqs_cm:
        step = int(round(nu / grain))
        if step <= 0 or step >= n:
            continue
        for e in range(step, n):
            T[e] += T[e - step]
    return T

def microcanonical_rate(molecule_freqs, ts_freqs, E0_cm, e_grid_cm,
                        reaction_path_degeneracy=1.0):
    """Energy-resolved specific rate k(E) = L * N_dagger(E-E0) / (h * rho(E)).

    molecule_freqs : frequencies of the energized molecule (cm^-1)
    ts_freqs       : frequencies of the critical configuration, reaction
                     coordinate removed (cm^-1)
    E0_cm          : barrier height (cm^-1)
    Returns k(E) in s^-1 on e_grid_cm.
    """
    grain = e_grid_cm[1] - e_grid_cm[0]
    # density of states of the molecule (per cm^-1), denominator
    rho = count_states(molecule_freqs, e_grid_cm, 'density') / grain   # states / cm^-1
    # cumulative sum of states of the critical configuration, numerator,
    # evaluated at the energy available above the barrier, E* = E - E0
    Nts = count_states(ts_freqs, e_grid_cm, 'sum')
    shift = int(round(E0_cm / grain))

    k = np.zeros(len(e_grid_cm))
    # h in (cm^-1 * s) units so that N_dagger/(h*rho) comes out in s^-1:
    # h*c converts cm^-1 to J; here rho is per cm^-1, so use h*c [J*s*cm? ]:
    hc = H_PLANCK * C_LIGHT          # J*s*cm/s ... = J*cm ; times 1/cm^-1 gives J*s -> use below
    for i in range(len(e_grid_cm)):
        if i - shift < 0:
            continue                 # below the barrier: no open channel, k = 0
        Ndag = Nts[i - shift]        # number of open channels at the complex
        if rho[i] > 0:
            # k(E) = N_dagger / (h * rho):  rho [1/cm^-1], convert h to (cm^-1 . s)
            h_in_cm = H_PLANCK / hc  # = 1/c, gives Planck's constant in (cm^-1 * s)
            k[i] = reaction_path_degeneracy * Ndag / (h_in_cm * rho[i])
    return k

def thermal_rate(T_kelvin, molecule_freqs, ts_freqs, E0_cm, e_grid_cm,
                 reaction_path_degeneracy=1.0):
    """High-pressure thermal rate: Boltzmann average of k(E).

    k_inf = sum_E k(E) rho(E) exp(-E/kT) / sum_E rho(E) exp(-E/kT)
          -> (kT/h)(Q_dagger/Q) exp(-E0/kT) in the high-pressure limit.
    """
    grain = e_grid_cm[1] - e_grid_cm[0]
    rho = count_states(molecule_freqs, e_grid_cm, 'density')
    k_E = microcanonical_rate(molecule_freqs, ts_freqs, E0_cm, e_grid_cm,
                              reaction_path_degeneracy)
    # E in J for the Boltzmann factor
    E_J = e_grid_cm * H_PLANCK * C_LIGHT
    w = rho * np.exp(-E_J / (K_B * T_kelvin))
    return np.sum(k_E * w) / np.sum(w)
```
