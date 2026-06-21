## Research question

Classical vacuum between two uncharged metal plates looks inert: no free charge, no applied electromagnetic field, and no ordinary pressure difference. Quantum field theory gives a sharper problem. If the electromagnetic field is quantized, every allowed normal mode has a ground-state contribution `hbar omega / 2`, so the word "empty" cannot mean "no field energy at all." A pair of conducting boundaries changes which field modes are allowed. The question is whether the change in allowed vacuum modes, after the meaningless infinite background is removed, leaves a finite mechanical force between the plates.

A successful answer has to do four things at once: count the allowed modes between parallel plates, handle the divergent zero-point sum without assigning direct meaning to the absolute infinity, compare the plate geometry with the corresponding empty or widely separated geometry, and return a measurable force per unit area with the correct attraction sign.

## Background

Quantized fields inherit the normal-mode structure of the corresponding classical fields. In a cavity, each standing electromagnetic wave is a harmonic oscillator; its ground state still contributes `hbar omega / 2`. The zero-point sum over all modes is divergent in free space, so the absolute value is not directly observable. What can be observable is a change in the zero-point energy when matter imposes different boundary conditions.

Perfect conductors give an idealized but clean boundary problem. At a conducting surface, the tangential electric field and the normal magnetic field vanish. For two large parallel plates separated by `a`, the transverse components of the wave vector remain continuous in the large-area limit, while the normal component is discretized in units of `pi/a`. This makes the plate separation appear directly in the allowed frequencies.

The physical intuition is not that photons are bouncing in a classical empty box. The useful statement is spectral: boundaries alter the spectrum of quantum electromagnetic modes. Modes with wavelengths comparable to the separation are affected; very short wavelengths see a real metal as increasingly transparent, which is why a regulator or a real optical response has a physical role rather than being only algebra.

The force is naturally a pressure. If the finite part of the vacuum energy per area depends on separation, virtual work gives `P = F/A = -d(E/A)/da`. Attraction means the finite energy is lower at smaller separation, so increasing `a` requires work against the force.

## Baselines

**Pairwise London dispersion forces.** London's nonretarded van der Waals picture explains attraction between neutral atoms through correlated dipole fluctuations. It gives the right qualitative origin for neutral-body attraction, but it is microscopic, material-specific, and short-distance in character. It does not immediately produce a universal force law between ideal conducting plates.

**Retarded atom-atom and atom-wall interactions.** Retardation adds the finite propagation speed of the electromagnetic field to dispersion forces. The large-distance behavior changes because the field cannot correlate dipoles instantaneously. This gives a bridge from molecular attraction to field fluctuations, but a direct summation over all microscopic pairs in macroscopic plates is not the simplest route to a clean plate-pressure formula.

**Classical cavity electrodynamics.** Conducting walls already determine standing-wave frequencies. Classical theory therefore supplies the boundary-value problem and mode spectrum, but a classical vacuum has no ground-state oscillator energy to sum. It has the geometry but not the quantum pressure.

**Naive zero-point summation.** Writing `E0 = (1/2) sum hbar omega` for the plate modes captures the right object, but by itself it diverges. Without a cutoff, subtraction, or equivalent regularization, the expression has no physical value. The useful comparison must discard separation-independent bulk terms and keep only the finite dependence on `a`.

## Evaluation settings

The clean theoretical yardstick is two perfectly conducting, plane, parallel plates in vacuum, with plate area much larger than the square of the separation. The temperature is taken as zero, edge effects are neglected, and the surfaces are treated as perfect reflectors for the modes relevant to the separation.

The natural measured quantity is the normal force per unit area, or pressure, as a function of plate separation. The sign distinguishes attraction from repulsion, and the dependence on separation tests whether the force comes from a geometric change in the mode spectrum.

Real experiments replace the ideal limit with finite conductivity, roughness, finite temperature, alignment constraints, and often a sphere-plane geometry because maintaining parallelism is difficult. Those corrections are part of experimental comparison, while the ideal parallel-plate calculation gives the reference law.

## Code framework

The field-natural scaffold is analytic. The inputs are a quantum electromagnetic field, two large conducting boundaries, a separation parameter, and a rule for virtual work. The available primitives are classical cavity boundary conditions, mode frequencies, zero-point oscillator energy, a regulator or equivalent subtraction, and differentiation of energy with respect to geometry.

The empty analytic slot is the finite comparison:

```text
given boundary geometry:
    enumerate allowed electromagnetic modes
    form the regulated zero-point mode sum
    subtract the corresponding geometry-independent vacuum contribution
    take the regulator away after the difference is finite
    differentiate the finite energy per area with respect to separation
```

Completing the slot for parallel plates produces the energy per area, pressure, sign, and physical meaning.
