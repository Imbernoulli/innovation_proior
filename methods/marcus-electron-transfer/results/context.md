# Context: rates of electron-transfer reactions between ions in solution

## Research question

Why do the rates of oxidation–reduction reactions that consist of nothing more than the transfer of
a single electron between two ions in solution span such an enormous range, and can that range be
predicted *a priori* from quantities one already knows — the ionic charges, the ionic radii, the
solvent's dielectric properties, and the thermodynamics of the reaction — with **no adjustable
parameters**?

The phenomenon that demands an explanation is sharp. Isotopic-exchange reactions that differ only in
the valence of the exchanging ion (for example, an electron hopping between a small +2 and a small +3
aqueous cation) are *slow*, while the analogous exchange between large complex ions
(Fe(CN)₆³⁻/Fe(CN)₆⁴⁻, MnO₄⁻/MnO₄²⁻) is *fast*. A simple electron transfer of this kind breaks and
forms no chemical bonds and, in the self-exchange case, produces products identical to the reactants —
so the usual rate-controlling factor, the relative stability of products versus reactants, is removed
entirely. These are arguably the simplest reactions in chemistry, and yet the standard rate theory of
the day cannot say why one is slow and the other fast, nor even get the *sign* of their measured
entropies of activation right (they are large and negative, between like-charged ions, where naive
arguments predicted a positive contribution). A correct theory has to identify what actually
constitutes the activation barrier when no bonds move, and turn it into a computable number.

## Background

**Transition-state / absolute-rate theory (Eyring; Glasstone, Laidler & Eyring 1941).** A chemical
reaction is pictured as the system moving on a potential-energy surface — the electronic energy of
all the atoms plotted against their positions. The system climbs from the reactant valley over a
saddle ("the pass") into the product valley; the dividing surface at the pass is the *transition
state* (activated complex). Assuming a quasi-equilibrium between transition state and reactants, the
rate constant is k = (kT/h)·exp(−ΔF\*/kT), with ΔF\* the free energy of forming the activated complex.
For an ordinary reaction the reaction coordinate is a bond length being stretched or a group being
transferred, and ΔF\* follows from the stretching/compression energetics. Wigner (1938) gave the
dynamical justification: the quasi-equilibrium holds if trajectories do not recross the dividing
surface.

**Equilibrium continuum electrostatics of ions in solution (Born 1920; the q₁q₂/DR interaction).**
The free energy of solvation of an ion of charge e and radius a in a medium of static dielectric
constant Dₛ is the Born expression, −(e²/2a)(1 − 1/Dₛ); two ions of charge q₁, q₂ a distance R apart
interact with free energy q₁q₂/(Dₛ R). Both rest on one assumption: at every point the dielectric
**polarization of the solvent is in equilibrium with the local field produced by the instantaneous
ionic charges**, so the polarization can be read straight off the charge distribution and the
dielectric constant. This machinery is what one would normally reach for to estimate the solvation
free energy of an activated complex, treating the complex as a sphere bearing the summed charge.

**The Franck–Condon principle and its first application to these rates (Libby 1952).** In
spectroscopy the Franck–Condon principle says an electronic transition is so fast that the nuclei do
not move during it. Libby imported this idea to electron-transfer kinetics: when the electron jumps
from one ion to the other, it jumps far faster than the heavy solvent molecules can reorient, so the
two product ions are created *in the solvent configuration that was appropriate to the reactants*.
For a small cation the surrounding solvent is tightly and very differently oriented in the +2 and +3
states, so the newly made ion finds itself in a strongly "wrong," high-energy environment — a large
barrier, hence a slow reaction. For large complex ions the field at the solvent changes much less on
transfer, the environment is less foreign, and the reaction is fast. The same picture, raised in the
ensuing symposium discussion, explained why a complex ion with a large change in metal–ligand bond
length on reduction (Co(NH₃)₆³⁺/²⁺) is nonetheless slow: now it is the *inner-shell* nuclear
coordinates that are left in a foreign configuration. This qualitative account captured the size
trend strikingly well. Two things were left unsettled, however: the actual back-of-the-envelope
estimate of the resulting solvation barrier did not hold up quantitatively; and, taken literally, a
frozen-nuclei electron jump straight up from the reactant's equilibrium configuration is a *vertical*
transition — the kind a photon induces — so as a description of a reaction proceeding **in the dark**
it leaves the energy books unbalanced.

**Nonequilibrium polarization as a physical state (Platzman & Franck 1952; polaron work of Pekar and
Fröhlich, early 1950s).** When a dissolved ion absorbs light, the electron is promoted essentially
instantaneously while the solvent dipoles, needing far longer to reorient, are momentarily left in
their old arrangement. For that brief interval the medium's electrical polarization is **not** the
value the new charge distribution would dictate — a genuine nonequilibrium-polarization state. This
concept was used to interpret the optical absorption spectra of halide ions in water. It established
that the polarization of a medium can usefully be regarded as having more than one part, relaxing on
very different timescales, and that the fast and slow parts can be temporarily out of step.

**Timescales of the solvent's polarization.** The electrical polarization of a solvent is the sum of
electronic, atomic, and orientation contributions, which respond to a sudden change in the charges in
roughly 10⁻¹⁵, 10⁻¹³, and 10⁻¹¹ seconds respectively. The electronic part tracks essentially any
change instantly; the atomic and orientation parts are sluggish.

**The earlier tunnelling treatment (Weiss 1954; Marcus, Zwolinski & Eyring 1954).** A prior attempt
treated the slow step as the electron tunnelling through a barrier between the two ions, with a
probability falling off exponentially in the tunnelling distance, ≈ exp(−β r_ab). It correctly sensed
that a reorganization of the solvation atmospheres precedes the jump, but it mishandled which
elementary step is rate-limiting (tacitly letting the reorganized state only go forward to products,
never relax back to reactants), and it omitted the frequency with which the electron actually strikes
the barrier, which threw the estimated tunnelling rate off by orders of magnitude.

**The empirical facts to be explained (pre-theory observations).** (i) Self-exchange between small
aqueous cations is slow; between large complex ions, fast. (ii) Measured entropies of activation for
clean-mechanism exchanges are large and negative, and occur between ions of *like* sign. (iii) A
complex ion with a large inner-shell bond-length change on electron transfer is slow despite its
size. These are facts about existing systems, established by isotopic-exchange kinetics with the
post-war flood of radioactive tracers and by the new fast-reaction instrumentation (e.g. stopped-flow
for inorganic redox), and any theory must reproduce them.

## Baselines

The methods a quantitative electron-transfer theory would be measured against, and where each stalls:

- **Absolute-rate theory applied directly to an electron-transfer "activated complex."** Take the
  reacting pair as a single activated complex, compute its free energy of formation by ordinary means,
  insert into k=(kT/h)exp(−ΔF\*/kT). *Where it stalls:* the theory presupposes a reaction coordinate
  along which bonds stretch, and a recipe for ΔF\* from that motion. In a pure electron transfer no
  bonds break or form, so there is no such coordinate and no prescription for the barrier; the
  framework gives the *form* of a rate law but cannot furnish ΔF\* for this class of reaction.

- **Equilibrium continuum electrostatics for the activated complex (Born / q₁q₂DR).** Estimate the
  solvation free energy of the activated complex by treating it as a charged sphere whose polarization
  field is in equilibrium with its charge. *Where it stalls:* its defining assumption — polarization
  always equilibrated to the instantaneous charges — is precisely what cannot hold at the moment of an
  electron transfer, where the heavy solvent has not had time to follow the charge. Used as written it
  describes a state the reaction never passes through, and it contains no quantity that could register
  how far the solvent's slow polarization sits from equilibrium.

- **Libby's Franck–Condon argument.** Qualitatively correct and the source of the key physical
  picture (frozen nuclei → foreign solvent environment → barrier; barrier larger for small ions).
  *Where it stalls:* the numerical estimate of the barrier did not survive scrutiny, and the literal
  vertical-jump-from-the-reactant-minimum picture corresponds to an energy-non-conserving (optical)
  event, leaving open how the same physics operates thermally, in the dark, with the books balanced.
  It offers a mechanism but not a computable, parameter-free free energy of activation.

- **The electron-tunnelling-barrier treatment.** Locates the slowness in a tunnelling probability
  exp(−β r_ab) and recognizes that solvent reorganization precedes the jump. *Where it stalls:* it
  gets the kinetic bookkeeping of the elementary steps wrong (no reverse relaxation of the reorganized
  state) and leaves out the strike frequency, so its rate estimates are unreliable; and it locates the
  difficulty in the electronic step rather than in the cost of preparing the nuclear configuration.

## Evaluation settings

The natural yardstick is the kinetics of well-characterized electron-transfer reactions in aqueous
solution at 25 °C, where the mechanism is simple and the concentrations of the true reacting species
are known. The canonical test set is the isotopic self-exchange reactions measured with radioactive
tracers — small aqueous cations (e.g. Fe²⁺/Fe³⁺) versus large complex ions (Fe(CN)₆³⁻/⁴⁻,
MnO₄⁻/²⁻) — together with cross reactions between two different redox couples measured by fast-reaction
methods (stopped-flow). Reported observables are second-order rate constants (M⁻¹ s⁻¹), activation
free energies and entropies extracted from their temperature dependence, and the dependence of rate
on solvent dielectric constant and on ionic strength. Required inputs that exist independently of any
rate theory: ionic crystallographic radii, the charges, the solvent's static and optical dielectric
constants (for water, Dₛ ≈ 78.5 and D_op ≈ 1.8 at 25 °C), and the standard free energy of the
reaction. A successful theory is judged by whether calculated and observed rates agree across this set
without arbitrary parameters.

## Code framework

A scaffold for turning electrostatic inputs into a predicted rate. The pieces that already
exist are the dielectric/geometric inputs, the Born and Coulomb electrostatics, and the
transition-state rate law k = Z·exp(−ΔF\*/kT). What is missing is the one quantity that the whole
problem turns on — the free energy of activation ΔF\* for a reaction in which no bonds move — and the
electrostatic state from which it must be computed.

```python
import numpy as np

# --- known physical constants and solvent properties ---
kT_kcal = 0.593            # kT at 25 C, kcal/mol
D_s     = 78.5             # static dielectric constant of water, 25 C
D_op    = 1.8              # optical (high-frequency) dielectric constant of water
# e^2 in kcal*Angstrom/mol so that e^2/(a) is a solvation energy in kcal/mol:
E2      = 332.06           # (elementary charge)^2 in kcal*Angstrom/mol

class ReactingPair:
    """Two ions in a dielectric solvent: charges, saturated-sphere radii, separation."""
    def __init__(self, q1, q2, a1, a2, R):
        self.q1, self.q2 = q1, q2     # ionic charges (units of e)
        self.a1, self.a2 = a1, a2     # radii of the saturated spheres (Angstrom)
        self.R = R                    # center-to-center separation (Angstrom)

    def coulomb(self):
        """Equilibrium Coulomb interaction free energy of the two ions, q1 q2 / (Ds R)."""
        return E2 * self.q1 * self.q2 / (D_s * self.R)

    def born_solvation(self, q, a):
        """Born solvation free energy of an isolated ion, -(q^2/2a)(1 - 1/Ds)."""
        return -E2 * q**2 / (2 * a) * (1 - 1 / D_s)

def free_energy_of_activation(pair, dG0):
    """
    Free energy of activation for transferring an electron in this pair, given the
    reaction's standard free energy dG0.
    TODO: the object we will define and the quantity we will extract from it.
    """
    pass

def rate_constant(pair, dG0, Z=1e11):
    """Transition-state rate law: k = Z * exp(-dF_star / kT)."""
    dF_star = free_energy_of_activation(pair, dG0)
    return Z * np.exp(-dF_star / kT_kcal)
```
