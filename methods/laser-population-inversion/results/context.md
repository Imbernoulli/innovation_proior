## Research question

The goal is to turn microscopic light-matter interaction into a macroscopic source of coherent optical radiation. Ordinary lamps and discharge tubes emit many independent wave trains: the phases are not locked, the spectrum is broad, and a resonator does not by itself make the emission grow. A successful mechanism must explain how an electromagnetic mode can gain energy from atoms in phase with the field already present, how that gain can exceed all losses, and how a cavity can select and reinforce the growing mode.

## Background

Einstein's radiation theory separates three elementary processes for a two-level transition. An atom in the upper state can emit spontaneously with coefficient `A21`. Radiation at the transition frequency can also induce upward absorption with coefficient `B12` and induced downward emission with coefficient `B21`. The induced terms are proportional to the radiation energy density, so they scale with the field already present.

The key symmetry is that stimulated absorption and stimulated emission are parallel competitors. For an optical mode, the net stimulated change is proportional to the difference between the number of atoms able to emit and the number able to absorb, weighted by the relevant transition probabilities. In ordinary thermal equilibrium the lower state is more populated than the upper state.

Microwave masers had already shown that coherent radiation could be produced when molecules were prepared so that an active transition supplied more stimulated emission than absorption. That success made the optical question concrete: optical wavelengths demand much smaller resonators, stronger diffraction constraints, and a cavity geometry that can feed a selected transverse mode back through the excited material repeatedly.

Optical resonators supply mode selection and feedback. Two mirrors define longitudinal modes whose round-trip phase fits the cavity, and the electromagnetic field samples the active medium on each pass. Losses come from imperfect mirrors, scattering, diffraction, and residual absorption. Gain must therefore be compared against a round-trip loss budget.

Pumping is any external process that maintains a nonthermal atomic distribution against decay and absorption. It can be optical, electrical, collisional, or chemical, but the common role is to put enough atoms in an excited state associated with the radiating transition.

## Baselines

**Spontaneous emission sources.** A hot gas, discharge lamp, or fluorescent material emits because excited atoms decay. The rate has an `A21` part that does not require a pre-existing field and does not force the emitted photon into the phase, direction, polarization, and frequency of a selected optical mode. These sources can be bright, but the wave trains are mostly independent.

**Thermal equilibrium radiation.** Einstein's balance argument recovers Planck radiation by requiring absorption, stimulated emission, and spontaneous emission to balance for Boltzmann populations. The field drives both directions, with detailed balance holding throughout.

**Single-pass amplifying media.** An excited medium can increase a passing beam if it has more emitters than absorbers at the beam frequency. One pass gives finite gain, but noise, linewidth, and output direction remain poorly controlled.

**Microwave maser cavities.** Molecular beam and ammonia maser work established that state selection plus a resonant cavity can generate coherent microwave radiation. Optical frequencies need optical-quality reflectors and open resonator geometry.

## Evaluation settings

The natural theoretical check is the sign of the small-signal gain on a chosen atomic transition. With lower and upper populations `N1` and `N2`, the stimulated terms predict attenuation for ordinary populations and amplification for a prepared nonthermal distribution. Spontaneous emission may seed the field, but it is not counted as coherent gain by itself.

The cavity check is round-trip self-consistency. A wave at an allowed cavity frequency passes through an active length, reflects from both mirrors, suffers distributed and mirror losses, and returns with the same phase. Oscillation begins only when the round-trip gain equals or exceeds the round-trip loss.

The experimental observables are narrowband output near the atomic transition, strong directionality along the resonator axis, a threshold in pump strength, and output whose frequency and spatial mode are constrained by the cavity.

## Code framework

The field-natural scaffold is analytic. The available primitives are Einstein transition coefficients, two-level rate balances, a pump that changes level populations, propagation through an active medium, and a two-mirror resonator with loss.

```text
given:
  transition frequency nu
  coefficients A21, B12, B21
  populations N1, N2 maintained by an external pump
  active length L
  resonator mirror reflectivities R1, R2
  distributed loss alpha_loss

find:
  net small-signal gain sign
  round-trip condition for growth
  steady selected optical mode
```
