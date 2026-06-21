## Research question

The problem is to explain why a two-dimensional electron gas in a strong perpendicular magnetic field develops Hall conductance plateaus that are exact integer multiples of `e^2/h` across finite ranges of magnetic field or carrier density. A material-specific explanation is not enough: the striking fact is that the measured value is set by fundamental constants and survives impurities, interface details, and ordinary sample imperfections.

The question is therefore not merely why electron orbits are quantized. Landau quantization explains why special fillings exist, but the observed plateaus require more: the conductance must stay pinned while the microscopic Hamiltonian changes continuously. A satisfactory mechanism must identify a quantity that is integer-valued, measurable as Hall conductance, and unable to drift under small perturbations.

## Background

In the classical Hall effect, a transverse voltage appears because charged particles moving through a magnetic field are deflected. The Drude description gives a conductivity tensor with off-diagonal entries, but its Hall response depends smoothly on density, magnetic field, scattering time, and material parameters.

Confining electrons to a narrow semiconductor interface produces a two-dimensional electron gas. In a strong perpendicular magnetic field, the single-particle spectrum collapses into Landau levels with energies separated by the cyclotron scale. Each Landau level contains one state per flux quantum through the sample, so changing magnetic field changes both the level spacing and the number of states in each level.

If an integer number of Landau levels is filled and the Fermi energy lies in a gap, a clean noninteracting calculation gives `sigma_xy = nu e^2/h`. This explains isolated integer values but not plateaus over intervals. When the field or density is varied continuously, a clean system would normally move through partially filled levels rather than remain pinned.

Disorder changes the problem in a useful way. It broadens Landau levels and creates many localized bulk states. Localized states can absorb changes in electron number without carrying longitudinal current across the sample, while extended states remain necessary at transitions between plateaus. A boundary adds another unavoidable ingredient: the confining potential bends the Landau levels near the edge, producing chiral current-carrying edge modes.

The missing conceptual bridge is that Hall conductance can be written by linear response as a geometric quantity built from occupied quantum states. When those states form an isolated band or a mobility-gapped occupied subspace, the relevant geometric integral is integer-valued.

## Baselines

- **Classical Hall transport.** The Hall response follows from Lorentz force balance and a scattering time. Gap: it predicts a smooth response controlled by density, magnetic field, and disorder, not exact plateaus fixed by `h` and `e`.

- **Clean filled Landau levels.** A filled Landau level gives the right conductance unit because the degeneracy per area is set by magnetic flux. Gap: this accounts for exact integer fillings only; it does not explain why finite intervals of filling remain on the same conductance value.

- **Material and impurity explanations.** Semiconductor details, interface roughness, and impurities are physically present and affect the density of states. Gap: treating them as accidental details points in the wrong direction, because the plateau value is insensitive to such details while the perturbation does not close the relevant gap or delocalize the occupied subspace.

- **Gauge-invariance flux insertion.** Laughlin's annulus argument relates inserting one flux quantum to quantized charge transport between edges. Gap: it gives a powerful robustness argument, but by itself it does not fully express the Hall coefficient of a general band or lattice problem as a calculable bulk invariant.

- **Ordinary band theory.** Filled bands can be described by Bloch wave functions over a Brillouin zone. Gap: the usual energy-band picture does not by itself say which filled bands carry Hall conductance; the missing information is geometric, not just spectral.

## Evaluation settings

The physical setting is a two-dimensional electron gas at low temperature in a strong perpendicular magnetic field, with Hall and longitudinal transport measured while magnetic field or carrier density is varied. The signatures to explain are flat Hall conductance intervals, suppressed longitudinal conduction on plateaus, and transitions where extended states appear.

The theoretical setting is a noninteracting or effectively single-particle occupied subspace separated from current-carrying unoccupied states by an energy gap or mobility gap. Natural geometries include a torus for bulk linear-response calculations, an annulus for flux insertion, and a finite strip or disk for edge-state reasoning.

The mathematical yardstick is whether the same integer controls bulk response, plateau robustness, and the net chirality of boundary channels. The mechanism should not rely on the chemical composition of the semiconductor except insofar as the sample realizes the two-dimensional magnetic problem.

## Mechanism artifact

The final artifact should state the integer quantum Hall mechanism as a bulk topological theorem with an edge interpretation: for filled isolated bands or filled Landau levels,

`sigma_xy = (e^2/h) sum C_n`,

where each `C_n` is a first Chern number of the occupied quantum-state bundle. For a filled Landau level the Chern number is `1`, so `nu` filled levels give `sigma_xy = nu e^2/h`.

The proof should start from linear response, reduce the Hall conductance to a Berry-curvature integral over the Brillouin zone or flux-parameter torus, and identify the integral divided by `2*pi` as an integer Chern number. The robustness statement should follow from the discreteness of that integer: continuous perturbations cannot change it unless the gap or mobility gap closes.

The physical mechanism should then explain plateaus: localized disorder-broadened bulk states let the Fermi energy move without changing the current-carrying topological sector, while plateau transitions require extended states. The boundary version should state that the same integer counts net chiral edge modes, so the current can be carried robustly along the edge even when the bulk is localized.
