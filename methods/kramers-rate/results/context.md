# Context: computing the rate of a thermally activated barrier crossing

## Research question

A great many processes — a chemical reaction, the racemization of an optically
active molecule, the dissociation of a diatomic, the fission of an excited heavy
nucleus modeled as a charged liquid drop — share one structure: a system sits in
a metastable configuration, and only rarely, when the surrounding medium happens
to push it hard enough, does it surmount an energy barrier and end up somewhere
else. The temperature dependence of how often this happens is, empirically, the
Arrhenius/Van't Hoff law

    k = ν · exp(−E_b / k_B T),

where E_b is the height of the barrier and ν is a prefactor. The exponential is
understood. **The prefactor ν is not.** What sets its size? Is it a property of
the system alone (the shape of the well and the saddle), or does it depend on how
strongly the system is coupled to its surroundings — the pressure of a gas, the
viscosity of a solvent, the internal friction of nuclear matter?

The concrete goal: build a model in which the strength of the coupling to the
medium is a *tunable knob*, compute the escape rate as a function of that knob and
of temperature, and use the result to decide **when the equilibrium-flux recipe
for the prefactor (below) is trustworthy and what corrects it when it is not.**
A one-dimensional model suffices to expose the mechanism.

## Background

**Arrhenius (1889) / Van't Hoff (1884).** Reaction-rate data, plotted against
inverse temperature on a logarithmic scale, fall on straight lines: the rate is
governed by a threshold energy E_b (an "active state" the reactants must reach),
with a temperature-insensitive prefactor. This fixes the exponential but says
nothing about the prefactor's magnitude or its dependence on the environment.

**Transition-state theory — Eyring (1935); Polanyi–Wigner (1928), Pelzer–Wigner
(1932), Evans–Polanyi (1935).** The decisive idea: on the potential-energy
surface the reactant basin A and the product basin B are separated by a saddle
("col") C, the activated complex. *Assume the activated complex is in thermal
equilibrium with the reactants*, then count the equilibrium one-way flux of phase
points crossing C from A toward B. Statistical mechanics turns this into an
absolute rate,

    k = κ · (k_B T / h) · (Z‡ / Z_A) · exp(−E_b / k_B T),

with Z‡, Z_A the partition functions of the activated complex and the reactant
state. This is parameter-free and remarkably successful. It carries a
"transmission coefficient" κ ≤ 1 written in front to absorb the rate's failure to
reach the equilibrium value — but the theory provides no way to compute κ, and no
statement of the conditions under which κ = 1.

**The two unproven assumptions inside the equilibrium-flux recipe.** (i) Every
trajectory that crosses C in the forward direction is counted as a completed
reaction — none turns back. (ii) The population at C is held at its equilibrium
value, i.e. the supply of activated systems from A ("Nachlieferung", subsequent
delivery) keeps the saddle topped up despite the constant leak toward B. Both are
statements about *dynamics in contact with a medium*, yet the recipe uses only
*equilibrium* statistical mechanics; nothing in it can certify either assumption.

**The theory of Brownian motion — Einstein (1905), Smoluchowski (1906),
Langevin, Fokker, Planck, Ornstein; Uhlenbeck–Ornstein (1930).** A particle in a
medium at temperature T obeys a Langevin equation: deterministic force, a
systematic frictional drag −η·(velocity), and a rapidly fluctuating random force.
Equipartition is not disturbed by the Brownian motion, which ties the strength of
the random force to the friction and the temperature (the fluctuation–dissipation
relation). Equivalently, the probability density of the particle's state evolves
by a Fokker–Planck equation. Uhlenbeck and Ornstein's 1930 article collects the
theory and treats such diffusion equations (though for the velocity alone, not in
the full position–velocity phase space). The friction coefficient η is precisely
the single number that measures "how intensely the system exchanges energy and
momentum with its surroundings."

**Smoluchowski (1916/1917).** For *strongly damped* motion the velocity
relaxes almost instantly and one is left with diffusion of the position alone, the
Smoluchowski equation, with diffusion constant D = k_B T / η. Smoluchowski used
this to compute diffusion-controlled coagulation rates. This is the natural tool
for the high-friction / dense-solvent extreme — but only that extreme.

**Christiansen — chemical reaction as a one-dimensional diffusion problem.**
Christiansen analyzed reactions by treating the progress variable as a coordinate
along which the system diffuses from reactants to products, with the rate read off
from a stationary diffusion current between two reservoirs. This furnishes the
quasi-stationary, flux-between-reservoirs viewpoint.

**Farkas (1927) — flux over population.** In the analogous nucleation problem
Farkas isolated the cornerstone of any rate calculation: maintain a steady current
through the bottleneck by holding equilibrium far inside the metastable state and
absorbing systems past the bottleneck; the rate is then the **stationary flux
divided by the metastable population**, k = j / n_a. This converts "rate" into a
boundary-value problem for a steady current.

**Bohr–Wheeler (1939).** The fission of an excited heavy nucleus, modeled as a
hot charged liquid drop, was computed by exactly the equilibrium-flux recipe along
a fission coordinate. Whether that recipe is justified depends on the "internal
friction" of nuclear matter — an instance where the coupling-to-the-medium
question is physically pressing and entirely open.

**Diagnostic facts that frame the problem.** (a) In a dilute gas at low pressure
the rate of a unimolecular reaction falls below the equilibrium-flux value: too
few collisions arrive to keep the activated state populated — assumption (ii) is
visibly failing. (b) In a dense solvent the reaction coordinate near the barrier
top no longer flies straight across but is buffeted, so a forward crossing is not a
reliable predictor of a completed reaction — assumption (i) is under threat. So
the equilibrium-flux prefactor is empirically known to be an *upper* estimate that
both extremes of coupling can spoil, for opposite reasons.

## Baselines

**Equilibrium-flux (transition-state) prefactor.** k = (k_B T/h)(Z‡/Z_A)
exp(−E_b/k_B T), or in a classical 1-D well of angular frequency ω_0,
k = (ω_0/2π) exp(−E_b/k_B T). *Core idea:* count the equilibrium one-way flux
across the saddle. *Gap it leaves:* it is built entirely from equilibrium
statistical mechanics and so is blind to whether forward crossings actually
complete and whether the saddle stays populated under a running current; it
supplies a symbol κ for the discrepancy but no machinery to evaluate it or to say
when κ = 1. It cannot represent any dependence of the rate on the medium's
coupling strength, because that strength does not appear in it.

**Smoluchowski overdamped diffusion.** ∂σ/∂t = ∂/∂q[(η⁻¹U'(q))σ +
(k_B T/η)∂σ/∂q]; with a stationary current between two points, the rate follows
from ∫exp(U/k_B T)dq across the barrier. *Core idea:* in the strong-damping
limit the position alone diffuses. *Gap it leaves:* it discards the velocity
entirely, so it applies only when the friction is large; it has nothing to say
about weak coupling, where the bottleneck is energy supply rather than spatial
transport, and it cannot interpolate to the equilibrium-flux regime.

**Brownian-motion / Fokker–Planck description of the velocity.** Einstein–
Smoluchowski–Uhlenbeck–Ornstein give the evolution of the velocity distribution
under friction and noise. *Core idea:* random force tied to friction by
fluctuation–dissipation; densities obey Fokker–Planck equations. *Gap it leaves:*
as developed, it lives in velocity space (or position alone), not in the joint
position–velocity phase space that an escape-over-a-barrier problem demands, and it
has not been carried over to compute a reaction rate.

## Evaluation settings

The natural yardstick is one-dimensional and fully specified by a few numbers, so
the regimes can be dialed: a smooth double-well (or single metastable well plus
barrier) potential U(q) with a reactant minimum at A, a barrier of height E_b at a
saddle C, and a product side B; the well curvature gives an angular frequency ω_0,
the inverted barrier curvature an angular frequency ω_b. The medium is a heat bath
at temperature T (so k_B T is an energy), coupled through a single friction
coefficient η that is to be swept from very small to very large. The regime of
interest is E_b ≫ k_B T (large barrier), so that escape is a rare event and a rate
is well defined. The quantity to report is the escape rate k (equivalently the
mean escape time 1/k) as a function of η and of E_b/k_B T, and its ratio to the
equilibrium-flux prefactor. A representative illustrative case is ω_b = ω_0 and
E_b/k_B T = 10. The comparison standard is the equilibrium-flux (transition-state)
value; the questions are *for which η does the model reproduce it, and what is the
correcting factor outside that range.* Physical reference points for η: the
viscosity of water for a solute reaction; the collision frequency in a gas; the
(unknown) internal friction of nuclear matter for fission.

## Code framework

Pre-existing numerical primitives only: array math, exponentials, error functions,
quadrature, root-finding. The task is to fill in the rate as a function of the
medium-coupling knob, given a well/barrier shape. The slots below are empty; the
method supplies their bodies and decides what closed forms (if any) replace the
quadratures.

```python
import numpy as np

# --- given: the well/barrier geometry and the bath ---
# omega0 : angular frequency at the reactant well bottom
# omegab : angular frequency of the unstable mode at the barrier top
# Eb     : barrier height
# T      : temperature in energy units (k_B = 1); mass = 1
# eta    : medium friction coefficient (the knob to be swept)

def equilibrium_flux_rate(omega0, Eb, T):
    """The baseline prefactor: equilibrium one-way flux over the saddle.
    k = (omega0 / 2pi) * exp(-Eb / T).  Independent of the medium."""
    return (omega0 / (2.0 * np.pi)) * np.exp(-Eb / T)

def medium_correction(omega0, omegab, eta, Eb, T):
    """How the coupling to the medium modifies the baseline prefactor.
    TODO: the object we will derive — what multiplies (or replaces) the
    equilibrium-flux rate once the running current and the energy supply are
    accounted for, as a function of the friction eta."""
    pass  # TODO

def escape_rate(omega0, omegab, eta, Eb, T):
    """Assemble the escape rate from the geometry and the medium coupling.
    TODO: fill in once `medium_correction` is known."""
    pass  # TODO

def sweep_friction(omega0, omegab, Eb, T, etas):
    """Escape rate vs. the medium-coupling knob, to be compared against the
    equilibrium-flux baseline across the full range of eta."""
    pass  # TODO
```
