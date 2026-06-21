## Research question

Electromagnetic potentials enter the Hamiltonian of a charged quantum particle, but classical mechanics teaches a different lesson: fields give forces, and potentials are a redundant way to calculate those fields. Consider a region where the electric and magnetic fields vanish along every possible path of the particle. Classically, with no field there is no force and no effect. Quantum mechanically, the wavefunction is sensitive to phase, and the canonical momentum in the Schrodinger equation contains the scalar and vector potentials themselves.

The question is whether a charged particle can show an observable interference change caused by electromagnetic potentials in a region where the particle never encounters nonzero fields, while preserving gauge invariance and without invoking a Lorentz force.

## Background

**Classical fields and gauge redundancy.** In classical electrodynamics the measurable force on a particle of charge `q` is the Lorentz force, determined by `E` and `B`. The potentials satisfy `B = curl A` and `E = -grad phi - (1/c) partial_t A` in Gaussian units, but the replacement `A -> A + grad Lambda`, `phi -> phi - (1/c) partial_t Lambda` leaves the fields unchanged. The potentials serve as bookkeeping useful for canonical formalisms and calculations.

**Quantum canonical coupling.** The nonrelativistic Hamiltonian for a charged particle in prescribed electromagnetic potentials is

`H = (1/2m)(-i hbar grad - q A/c)^2 + q phi`

in Gaussian units. The field strengths do not appear in this operator directly. The theory remains gauge invariant: a gauge transformation can be compensated by a local phase change of the wavefunction.

**Interference measures relative phase.** A single overall phase is unobservable, but a coherent beam split into two alternatives and recombined converts a relative phase into a shift of interference fringes. A potential that changes only the phase of one branch relative to another, while leaving the particle in field-free space, would have no classical force counterpart.

**Shielded electric and magnetic geometries.** A conducting tube can keep an electron wave packet in a region of zero electric field while the scalar potential inside the tube is changed after the packet enters and before it exits. A long thin solenoid or confined flux tube can keep `B` localized inside an excluded core while the electron beams pass outside. In the magnetic case, the accessible exterior is not simply connected: loops encircling the excluded core cannot be continuously shrunk without crossing the field region.

**Electron interferometry.** Electron biprism and split-beam experiments had already made coherent electron interference a practical tool. The natural experimental readout is the displacement, blurring, or reshaping of the interference pattern when the enclosed flux or branch potential history is changed.

## Baselines

**Local Lorentz-force reasoning.** A charged particle responds to `E` and `B` only through `q(E + v x B/c)`. If both fields vanish along the particle paths, this account predicts no local force, no acceleration, and treats the absence of local force as the relevant information.

**Gauge-removal argument.** If `B = curl A = 0` in a region, then locally `A = grad chi`, and the vector potential can be removed from the Hamiltonian by rephasing the wavefunction. The argument applies on a simply connected patch or a single open branch.

**Semiclassical electron-optics reasoning.** Ehrenberg and Siday treated electron propagation through the analogy with optical path length and an effective refractive index containing the canonical momentum. By this route, a phase difference between two electron rays can contain a magnetic-flux term even when the rays run outside the field region, presented as a fact about electron lenses and refractive indices.

**Ordinary two-slit interference.** Two coherent alternatives accumulate phases and recombine, making relative phase observable as a fringe shift. By itself it specifies how to read a phase.

## Evaluation settings

The clean magnetic test is a coherent electron beam split into two branches that pass on opposite sides of a narrow, shielded magnetic flux tube and then recombine. The electron wavefunction must have negligible support in the magnetic-field region, and the flux must be varied or compared against a zero-flux setting while all ordinary electrostatic and magnetic leakage effects are controlled. The measured quantity is the displacement or contrast change of the interference pattern as a function of enclosed flux.

The clean electric test is a split coherent beam sent through two conducting tubes. Each packet is fully inside its tube while the tube potential is raised and lowered, so the electron is not exposed to the electric field near the tube ends. The measured quantity is the relative interference phase set by the time integral of the branch potentials.

The theoretical checks are gauge invariance, locality, and topology. The predicted phase must be unchanged by single-valued gauge transformations, depend only on the closed spacetime or spatial loop integral of the potential, reduce to no observable effect for a single unsplit beam, and become trivial when the closed-loop phase is an integer multiple of `2 pi`.

## Code framework

The field-natural scaffold is analytic. The needed objects are a charged-particle Schrodinger Hamiltonian with minimal coupling, a coherent split-and-recombine interferometer, a field-free accessible region, an excluded source region carrying flux or time-dependent potential history, and a rule for converting branch phases into a fringe shift.

The empty slot is the phase rule. It takes as input two spacetime branches with the same endpoints, uses the potentials along the accessible branches, and returns the relative phase for the closed contour formed by the two branches.
