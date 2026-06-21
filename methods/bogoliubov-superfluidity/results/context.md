# Context: a molecular theory of frictionless flow in a degenerate Bose liquid

## Research question

Liquid helium below the λ-point (≈2.17 K) flows through narrow capillaries and thin films with no
measurable viscosity — it is *superfluid*. The challenge is to explain this from the molecules
themselves: to write down a Hamiltonian for a system of identical Bose particles, find its low-lying
excited states, and show *from those equations* that the liquid can move relative to its surroundings
without dissipation, up to some finite velocity. The bar is not a quantitative fit to real helium (a
strongly interacting liquid is hopeless for a pure molecular theory at this stage) but a *qualitative*
mechanism: a microscopic system of interacting bosons that genuinely supports frictionless flow, with
the property of superfluidity falling out of the basic equations rather than being put in by hand.

There is a sharp obstacle that any such theory has to clear. A degenerate *ideal* Bose gas in its
ground state — every particle in the zero-momentum state — looks like the obvious candidate for a
frictionless fluid, and indeed that was the first hope. But it cannot be a superfluid: nothing
prevents a condensate particle from exchanging momentum with an excited particle that collides with
it, so an object dragged through the gas can always shed momentum into single-particle excitations.
There is no energetic protection against friction. The free-particle excitation spectrum is precisely
the problem. So the question becomes: what has to be added to make a *non-ideal* Bose gas protect its
flow, and can that protection be derived rather than postulated?

## Background

**Bose–Einstein condensation as the substrate (London 1938).** Fritz London, reacting to the fountain
effect, proposed that liquid ⁴He at the λ-point undergoes a form of Bose–Einstein condensation: ⁴He
atoms are bosons, the liquid is a "quantum liquid" (large zero-point energy keeps it from
crystallizing), and a macroscopic fraction of atoms accumulates in the single lowest-energy
one-particle state below T_λ. This identified the condensate — a macroscopically occupied
zero-momentum state — as the object responsible for the anomalous low-temperature phase. The picture
gave a transition and an order parameter but, by itself, no frictionless-flow mechanism.

**The two-fluid description (Tisza 1938).** László Tisza turned London's condensate into a
hydrodynamic two-fluid model: helium II behaves as an interpenetrating mixture of a "superfluid"
component (the condensate, carrying no entropy, flowing without dissipation) and a "normal" component
(the excited atoms, viscous, carrying the entropy), with densities ρ_s, ρ_n that vary with
temperature (ρ_s→ρ as T→0, ρ_n→ρ as T→T_λ). This organized the thermomechanical phenomena —
fountain effect, heat as a wave — into one framework. Its normal component, however, was modeled as
the *non-condensed atoms treated as an ideal gas*, which leaves the friction objection untouched: an
ideal-gas normal component has a free-particle spectrum and no protection.

**Quantized elementary excitations and the criterion (Landau 1941).** Lev Landau quantized the
hydrodynamics of a quantum liquid and argued that every weakly excited state is an aggregate of
*elementary excitations* — quasiparticles — rather than excited individual atoms. He divided them into
"phonons" with a linear dispersion ε = c p (c the sound velocity) and "rotons" with a gapped
dispersion ε = Δ + p²/2µ, and showed how to get the specific heat and a two-fluid description with a
normal component made of these excitations. Crucially he derived a *criterion*: a body moving through
the liquid at velocity V can create an excitation of momentum p and energy ε(p) only if energy and
momentum can both be conserved, which requires V ≥ ε(p)/p; hence dissipation is impossible below
   V_c = min_p ε(p)/p .
For a phonon spectrum this minimum is c > 0, and for the roton branch it is √(2Δ/µ); either way a
*finite critical velocity* exists and the flow is protected below it. But this rests on *postulating*
the shape of the spectrum (phonons + rotons) from quantized hydrodynamics. The microscopic question —
*why* a system of interacting bosons should have that spectrum, in particular a linear branch at small
momenta rather than the free-particle p²/2m — is left open. With a free-particle spectrum the same
criterion gives V_c = min (p/2m) = 0: no superfluidity, which is exactly the ideal-gas failure
restated.

**The second-quantization machinery.** The tools to attack the many-boson problem microscopically are
in hand. In occupation-number (second-quantized) form one writes the field operator
Ψ(q) = Σ_f a_f φ_f(q) over a complete set; for a uniform system the natural basis is the momentum
plane waves φ_f(q) = V^{-1/2} e^{i(f·q)/ℏ}, so a_f, a_f^+ annihilate/create a particle of momentum f
and obey the boson commutators [a_f, a_{f'}^+] = δ_{f,f'}, [a_f,a_{f'}] = 0. The number of particles
with momentum f is N_f = a_f^+ a_f, and N_0 = a_0^+ a_0 counts the condensate. A pair potential
Φ(|q−q'|) becomes, in this language, a quartic operator coupling four momenta with the constraint of
momentum conservation, weighted by the Fourier transform
   v(f) = ∫ Φ(|q|) e^{−i(f·q)/ℏ} dq ,    v(0) = ∫ Φ(|q|) dq .
Dirac, in his treatment of the interaction of an assembly of bosons with radiation (the "waves and
Bose–Einstein particles" section of his *Principles of Quantum Mechanics*), worked out how to handle a
boson mode whose occupation is large within this operator formalism. The thermodynamic limit is taken
with N→∞, V→∞, v = V/N fixed, replacing momentum sums by integrals Σ_f → V/(2πℏ)³ ∫ df.

**The dilute / weak-interaction premise.** For a dilute, cold gas the range r₀ of the inter-particle
force is far smaller than the mean spacing d = (V/N)^{1/3}, so only two-body encounters matter and the
detailed shape of Φ(r) is irrelevant — only its low-momentum Fourier amplitude (equivalently the
s-wave scattering length a, with the contact value v(0) = 4πℏ²a/m) survives. This is the regime in
which an interacting Bose gas is tractable: the interaction can be carried as a small parameter, so a
perturbative treatment around the degenerate ground state is in reach.

## Baselines

- **Degenerate ideal Bose gas.** Core idea: N non-interacting bosons; below T_c a macroscopic fraction
  condenses into the f=0 state. Excitation spectrum is the bare ε(f)=f²/2m. Gap it leaves: with this
  spectrum min ε(f)/f = 0, so a body of any nonzero velocity can shed momentum into single-particle
  excitations — the ground state offers no protection against friction. It gives condensation but not
  frictionless flow.

- **London's BEC picture of He II (1938).** Core idea: identify the λ-transition with Bose–Einstein
  condensation into a macroscopically occupied lowest state; the condensate is the order parameter of
  the superfluid phase. Gap it leaves: it names the condensate and the transition but provides no
  dynamical account of *why* the condensate's flow is dissipationless — it does not produce an
  excitation spectrum.

- **Tisza's two-fluid hydrodynamics (1938).** Core idea: helium II = superfluid component (condensate,
  zero entropy, inviscid) + normal component (excited atoms, viscous), with temperature-dependent
  densities; organizes the thermomechanical effects. Gap it leaves: the normal component is modeled as
  an *ideal gas of the non-condensed atoms*, i.e. with a free-particle spectrum; it predicts no
  critical velocity and inherits the friction objection at the microscopic level.

- **Landau's quasiparticle theory (1941).** Core idea: quantize the hydrodynamics; the low-energy
  states are phonon (ε=cp) and roton (ε=Δ+p²/2µ) quasiparticles, not excited atoms; from
  energy–momentum conservation, dissipation requires V ≥ ε(p)/p, giving a finite critical velocity
  V_c = min_p ε(p)/p > 0. Gap it leaves: the phonon–roton spectrum is *assumed*. The theory does not
  start from the interacting molecules, so it cannot say which microscopic systems have such a spectrum
  and which (like the ideal gas) do not; the linear small-momentum branch — the thing that makes V_c
  positive — is an input, not an output.

## Evaluation settings

The natural yardstick for a molecular theory at this stage is qualitative-mechanistic, not a numerical
benchmark. A candidate theory is judged by whether, starting from a many-boson Hamiltonian, it
produces: (i) a well-defined low-energy excitation spectrum E(f); (ii) a finite critical velocity
V_c = min_f E(f)/|f| > 0 via Landau's energy–momentum argument, hence frictionless flow; and (iii)
internal consistency — a stable condensate (the approximation must not predict its own ground state to
be unstable) and a self-consistent smallness parameter controlling the expansion. The relevant
"datasets" are the limiting regimes that must be reproduced: small-momentum behavior (should match
sound, with c tied to the compressibility ∂P/∂ρ at T=0), large-momentum behavior (should revert to the
free-molecule kinetic energy f²/2m when the interaction is negligible at that scale), and the dilute
limit |a| n^{1/3} ≪ 1 where the answer should depend on the potential only through v(0) (equivalently
the scattering length). Concrete material systems in view: liquid ⁴He below T_λ (qualitatively) and
the idealized dilute hard-sphere (or weakly repulsive) Bose gas (quantitatively).

## Code framework

A pre-method scaffold for the dimensionless, closed-form evaluation that such a theory would land on.
The known primitives are: a momentum grid, the bare kinetic energy, the Fourier amplitude of the
interaction, and Landau's `min E/|f|` reduction. The excitation energy `E(f)` itself, and the weights
of whatever transformation produces the quasiparticles, are the empty slots.

```python
import numpy as np

def kinetic(k):
    """Free-molecule kinetic energy T(k) = k^2 / (2 m), m = 1."""
    return 0.5 * k * k

def interaction_fourier(k, g):
    """v(k): Fourier transform of the two-body potential.
    Contact (dilute) limit -> constant g = v(0) = 4*pi*a (hbar=m=1)."""
    return g  # constant in the dilute limit

def excitation_energy(k, g, n0):
    """The low-energy excitation spectrum E(k) of the interacting,
    condensed Bose gas, expressed through T(k), v(k) and the condensate
    density n0."""
    # TODO: the spectrum we will derive
    pass

def transform_weights(k, g, n0):
    """The weights of the transformation that maps the bare modes to the
    independent low-energy excitations."""
    # TODO: the transformation we will define
    pass

# --- known reduction (Landau): critical velocity from a spectrum ---
def critical_velocity(g, n0, kmax=50.0, npts=200000):
    k = np.linspace(kmax / npts, kmax, npts)
    return np.min(excitation_energy(k, g, n0) / k)
```
