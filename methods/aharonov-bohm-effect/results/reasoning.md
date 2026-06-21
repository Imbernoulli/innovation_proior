I start with the thing that keeps being swept under the rug. In classical electrodynamics I can say the fields are real and the potentials are a convenient way to write them. If `E` and `B` vanish where the particle is, the Lorentz force vanishes there. That sounds decisive until I write the quantum Hamiltonian for a charged particle:

`H = (1/2m)(-i hbar grad - q A/c)^2 + q phi`.

The fields are derivatives of the potentials, but the wave equation itself contains the potentials. I cannot simply say that is meaningless, because the wavefunction carries phase, and phase differences are exactly what an interferometer measures. Still, gauge invariance is a real obstruction. If a change of potential is only `A -> A + grad Lambda` and `phi -> phi - (1/c) partial_t Lambda`, the wavefunction can be rephased by `exp(i q Lambda / hbar c)`, and local probabilities do not change. So the question is not whether the numerical value of `A` at a point is observable. It is whether there is a gauge-invariant phase left after I close an interferometer loop.

Let me test the scalar potential first because it is almost embarrassingly simple. Put a charged particle inside a conducting cage. Inside the cage the electric field is zero, and if the potential is spatially uniform but time dependent, the Hamiltonian is just `H_0 + q phi(t)` while the packet is inside. If `psi_0` solves the field-free equation, then

`psi(t) = exp[-(i q / hbar) int phi(t') dt'] psi_0(t)`

solves the new equation, because the extra time derivative of the exponential cancels `q phi(t)`. For one packet that is only an overall phase, so nothing happens. That is the easy escape, but it fails as soon as I split one coherent beam into two packets and give the two shielded tubes different potential histories. The recombined state carries

`Delta theta_electric = -(q / hbar) [int phi_1(t) dt - int phi_2(t) dt]`.

No electric field has acted on the electron packet if the potential is raised only after it is inside the conductor and lowered before it leaves. The observable is not a force; it is a relative phase. This already tells me that the classical slogan "fields alone act" is too coarse for quantum interference.

Now I want the magnetic version, because it is harder to dismiss as a time-dependent energy offset. In a field-free simply connected region, `B = curl A = 0` lets me write `A = grad chi`. Then the Hamiltonian with `A` is locally equivalent to the free Hamiltonian by

`psi = exp[(i q / hbar c) chi] psi_0`.

On a single path this looks like the end of the story. The particle only picks up an endpoint phase `(q / hbar c)[chi(b)-chi(a)]`, and if the two beams share endpoints the difference should vanish. But I have smuggled in a global assumption: I assumed the same single-valued `chi` exists over the whole accessible region.

Put the magnetic field inside a long thin solenoid and keep the electron beams outside. Outside the solenoid `B=0`, but the exterior has a hole. A standard outside potential is azimuthal, `A_theta = Phi/(2 pi r)` in Gaussian notation for a flux `Phi`, so locally it is a gradient:

`A = grad[(Phi / 2 pi) theta]`.

The catch is that `theta` is not single-valued around the solenoid. Increase `theta` by `2 pi` and `chi` changes by `Phi`. Locally I can gauge the potential away; globally I pay a phase

`exp[i q Phi / (hbar c)]`.

That is the piece no local field-only account can see. Each arm of the interferometer lies in a simply connected strip, so I can write one branch as `psi_1 = exp[(i q / hbar c) int_1 A . dr] psi_1^0` and the other as `psi_2 = exp[(i q / hbar c) int_2 A . dr] psi_2^0`. When the arms recombine, the relative phase is not an endpoint difference of one global function. It is

`Delta theta_magnetic = (q / hbar c) [int_1 A . dr - int_2 A . dr] = (q / hbar c)oint A . dr`.

By Stokes' theorem that closed line integral is the magnetic flux through a spanning surface:

`oint A . dr = int_S B . dS = Phi`.

The electron paths can avoid the field completely; the surface used in Stokes' theorem has to cut through the excluded solenoid core. That is not action by a local field on the electron. It is a statement about a flat potential in the accessible region with nontrivial holonomy around a hole.

Gauge invariance survives, but it changes form. If `A -> A + grad Lambda`, then the open integral on one branch changes by `Lambda(b)-Lambda(a)`, and the open integral on the other branch changes by the same amount, provided `Lambda` is a legal single-valued gauge function on the accessible region. The difference cancels. Around the closed loop,

`oint grad Lambda . dr = 0`.

So the observable is not `A` at a point and not an open-path phase by itself. The observable is the closed-loop phase modulo `2 pi`,

`exp[i (q / hbar c)oint A . dr] = exp[i q Phi / (hbar c)]`.

This also shows when the effect disappears. If `q Phi/(hbar c) = 2 pi n`, the holonomy is one. Then the multivalued `chi = (Phi/2 pi)theta` produces a single-valued wavefunction rephasing, and the outside potential becomes globally invisible to the interferometer. The period in flux is therefore

`Phi_0 = 2 pi hbar c / |q| = h c / |q|`

in Gaussian units, or `h/|q|` in SI.

I should be careful about what exactly is being claimed. This does not mean the gauge-dependent vector potential has a unique measurable value. It means that quantum mechanics needs more than the local fields in the accessible region. A flat connection can still have a nontrivial closed-loop phase. Saying "but the flux is a field inside the solenoid" does not restore a local force explanation, because the electron wavefunction is excluded from that region. It only says that the gauge-invariant holonomy can be computed either as a line integral of `A` along the electron paths or as a flux through a surface that passes through inaccessible field. The line-integral description is local to where the electron can go; the field-only description reaches into where it cannot.

There is one more check I want, because the split-beam derivation treats the two arms almost geometrically. If I solve the Schrodinger equation outside an ideal flux line, the outside equation in polar coordinates contains the angular shift produced by `A_theta = Phi/(2 pi r)`. The partial waves are no longer labeled by ordinary integer angular momentum alone; the flux shifts the angular part by the dimensionless number `alpha = q Phi/(2 pi hbar c)`. The solution is periodic under `alpha -> alpha + n`, because that changes the closed-loop phase by `2 pi n`, and it has no flux-dependent scattering when `alpha` is an integer. That is exactly what the interferometer argument says should happen: only the holonomy modulo `2 pi` matters. The exact wave solution is not a new mechanism; it is the same phase condition enforced by single-valuedness of the full wavefunction.

So the finished experiment is now forced. Split a coherent electron beam. Send the two parts around opposite sides of a shielded flux tube so that the magnetic field is confined to a region where the electron wavefunction has negligible support. Recombine the beams. Changing the enclosed flux by `Phi` shifts the interference phase by

`Delta theta = q Phi / (hbar c)`

in Gaussian units, with the SI replacement `q Phi / hbar`. The fringe pattern is periodic in `Phi_0`, and it changes even though the local fields vanish along the electron paths. The potential has not become a directly measurable vector at each point; the physically observable object is its gauge-invariant holonomy. That is the real lesson: in quantum mechanics a potential that is locally removable can still leave a global phase around a multiply connected field-free region, and an interference experiment reads that phase.
