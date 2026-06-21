## Research question

The problem is to explain why a two-dimensional electron gas in a strong perpendicular magnetic field develops Hall conductance plateaus that are exact integer multiples of `e^2/h` across finite ranges of magnetic field or carrier density. The measured value is set by fundamental constants and persists across impurities, interface details, and ordinary sample imperfections.

The question is therefore why the conductance stays pinned at an integer while the microscopic Hamiltonian changes continuously. Landau quantization explains why special fillings exist; the plateaus extend over intervals. The task is to identify a quantity that is integer-valued, measurable as Hall conductance, and stable under small perturbations.

## Background

In the classical Hall effect, a transverse voltage appears because charged particles moving through a magnetic field are deflected. The Drude description gives a conductivity tensor with off-diagonal entries whose Hall response depends smoothly on density, magnetic field, scattering time, and material parameters.

Confining electrons to a narrow semiconductor interface produces a two-dimensional electron gas. In a strong perpendicular magnetic field, the single-particle spectrum collapses into Landau levels with energies separated by the cyclotron scale. Each Landau level contains one state per flux quantum through the sample, so changing magnetic field changes both the level spacing and the number of states in each level.

If an integer number of Landau levels is filled and the Fermi energy lies in a gap, a clean noninteracting calculation gives `sigma_xy = nu e^2/h`. This gives isolated integer values at exact fillings. When the field or density is varied continuously, a clean system moves through partially filled levels.

Disorder broadens Landau levels and creates many localized bulk states. Localized states absorb changes in electron number without carrying longitudinal current across the sample, while extended states are present at transitions between plateaus. A boundary adds another ingredient: the confining potential bends the Landau levels near the edge, producing chiral current-carrying edge modes.

Linear response writes Hall conductance as a quantity built from occupied quantum states. When those states form an isolated band or a mobility-gapped occupied subspace, this opens the question of what determines the integer.

## Baselines

- **Classical Hall transport.** The Hall response follows from Lorentz force balance and a scattering time. It predicts a smooth response controlled by density, magnetic field, and disorder.

- **Clean filled Landau levels.** A filled Landau level gives the conductance unit because the degeneracy per area is set by magnetic flux, yielding exact integer fillings.

- **Material and impurity explanations.** Semiconductor details, interface roughness, and impurities are physically present and affect the density of states.

- **Gauge-invariance flux insertion.** Laughlin's annulus argument relates inserting one flux quantum to quantized charge transport between edges, giving a robustness argument tied to gauge invariance.

- **Ordinary band theory.** Filled bands can be described by Bloch wave functions over a Brillouin zone, characterized by their energy spectrum.

## Evaluation settings

The physical setting is a two-dimensional electron gas at low temperature in a strong perpendicular magnetic field, with Hall and longitudinal transport measured while magnetic field or carrier density is varied. The signatures to explain are flat Hall conductance intervals, suppressed longitudinal conduction on plateaus, and transitions where extended states appear.

The theoretical setting is a noninteracting or effectively single-particle occupied subspace separated from current-carrying unoccupied states by an energy gap or mobility gap. Natural geometries include a torus for bulk linear-response calculations, an annulus for flux insertion, and a finite strip or disk for edge-state reasoning.

The mathematical yardstick is whether the same integer controls bulk response, plateau robustness, and the net chirality of boundary channels. The mechanism should not rely on the chemical composition of the semiconductor except insofar as the sample realizes the two-dimensional magnetic problem.

## Mechanism artifact

The final artifact should explain the integer quantum Hall plateaus: why the Hall conductance for filled isolated bands or filled Landau levels is

`sigma_xy = nu e^2/h`,

an exact integer multiple of `e^2/h`, and why that value is stable under continuous changes of the Hamiltonian.

The account should start from linear response, reduce the Hall conductance to a calculable expression over the Brillouin zone or flux-parameter torus, and explain why the result is integer-valued. The robustness statement should explain why continuous perturbations cannot change the conductance unless the gap or mobility gap closes.

The physical mechanism should explain plateaus: localized disorder-broadened bulk states let the Fermi energy move without changing the current-carrying sector, while plateau transitions require extended states. The boundary version should connect the same integer to the net chirality of boundary channels, so the current can be carried robustly along the edge even when the bulk is localized.
