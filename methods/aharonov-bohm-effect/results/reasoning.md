I start with the thing that keeps being swept under the rug. In classical electrodynamics I can say the fields are real and the potentials are a convenient way to write them. If `E` and `B` vanish where the particle is, the Lorentz force vanishes there. That sounds decisive until I write the quantum Hamiltonian for a charged particle:

`H = (1/2m)(-i hbar grad - q A/c)^2 + q phi`.

The fields are derivatives of the potentials, but the wave equation itself contains the potentials. I cannot simply say that is meaningless, because the wavefunction carries phase, and phase differences are exactly what an interferometer measures. Still, gauge invariance is a real obstruction. If a change of potential is only `A -> A + grad Lambda` and `phi -> phi - (1/c) partial_t Lambda`, the wavefunction can be rephased by `exp(i q Lambda / hbar c)`, and local probabilities do not change. So the question I have to keep in front of me is not whether the numerical value of `A` at a point is observable. It is whether anything gauge-invariant survives once I close an interferometer loop.

Let me test the scalar potential first because it is almost embarrassingly simple. Put a charged particle inside a conducting cage. Inside the cage the electric field is zero, and if the potential is spatially uniform but time dependent, the Hamiltonian is just `H_0 + q phi(t)` while the packet is inside. I want to claim that if `psi_0` solves the field-free equation, then

`psi(t) = exp[-(i q / hbar) int phi(t') dt'] psi_0(t)`

solves the new equation. Before I trust that, I should actually substitute it in rather than wave at it. Write `S(t) = int phi dt'`, so `dS/dt = phi`, and `g = exp(-i q S/hbar)`. Then

`i hbar partial_t (g psi_0) = i hbar [(partial_t g) psi_0 + g partial_t psi_0]`.

The first term gives `i hbar (-i q phi/hbar) g psi_0 = q phi g psi_0`. The second term, using `i hbar partial_t psi_0 = H_0 psi_0`, gives `g H_0 psi_0`. Adding them, `i hbar partial_t psi = g H_0 psi_0 + q phi g psi_0 = (H_0 + q phi) psi`, since `H_0` is a spatial operator and commutes with the scalar `g`. The residual is exactly zero, so the ansatz really does solve the modified equation, and the only effect of `phi` on one packet is the overall phase `exp(-i q S/hbar)`. For a single beam that phase is unobservable. That is the easy escape.

It stops being an escape the moment I split one coherent beam into two packets and send them through two shielded tubes with different potential histories `phi_1(t)`, `phi_2(t)`. Each branch carries its own overall phase, but the recombined state is sensitive to the difference:

`Delta theta_electric = -(q / hbar) [int phi_1(t) dt - int phi_2(t) dt]`.

No electric field has acted on either packet, because the potentials are raised only after the packets are inside the conductors and lowered before they leave; the field at the electron is zero throughout. The observable is not a force; it is a relative phase. So already in the simplest case the slogan "only fields act" is too coarse for quantum interference. What I do not yet know is whether the magnetic side has the same structure or whether it is some artifact of a time-dependent energy offset, which the electric case could be accused of being.

Now the magnetic version, which I cannot dismiss as an energy offset because nothing is time dependent. In a field-free simply connected region, `B = curl A = 0` lets me write `A = grad chi`, and then the Hamiltonian with `A` is locally equivalent to the free Hamiltonian by

`psi = exp[(i q / hbar c) chi] psi_0`.

On a single path this looks like the end of the story. The particle picks up an endpoint phase `(q / hbar c)[chi(b)-chi(a)]`, and if two beams share endpoints the difference cancels. But that argument quietly assumed one single-valued `chi` exists over the whole accessible region. I should check whether that assumption can fail.

Put the magnetic field inside a long thin solenoid and keep the electron beams outside. Outside, `B = 0`, but the exterior has a hole. A standard outside potential is azimuthal, `A_theta = Phi/(2 pi r)` for a flux `Phi`. I want three things verified before I lean on it: that it is genuinely field-free outside, that it really is a gradient locally, and what its loop integral is. Take `A = (A_r, A_theta) = (0, Phi/(2 pi r))` and compute the only nonzero curl component in 2D, `(1/r)[partial_r(r A_theta) - partial_theta A_r]`. Here `r A_theta = Phi/(2 pi)` is constant in `r`, so its `r`-derivative is zero, and `A_r = 0`: the curl outside the core is zero, as it must be. Next, is it a gradient? Take `f = (Phi/2 pi) theta`. The polar gradient is `(partial_r f, (1/r) partial_theta f) = (0, Phi/(2 pi r))`, which is exactly `A`. So locally `A = grad f`. But `f(theta + 2 pi) - f(theta) = (Phi/2 pi)(2 pi) = Phi`: `f` is not single-valued, it jumps by `Phi` each time I go around. Finally the loop integral, directly: on a circle of radius `R`, `oint A . dr = int_0^{2pi} (Phi/(2 pi R)) R d theta = Phi`. So the catch is concrete. Locally I can gauge `A` away with `f`; globally I cannot, because the global `f` is multivalued, and going once around multiplies the wavefunction by

`exp[i q Phi / (hbar c)]`.

That phase is the piece no local field-only account can see. Each arm of the interferometer lies in a simply connected strip, so I write each branch as `psi_j = exp[(i q / hbar c) int_j A . dr] psi_j^0`. When the arms recombine, the relative phase is not an endpoint difference of one global function:

`Delta theta_magnetic = (q / hbar c) [int_1 A . dr - int_2 A . dr] = (q / hbar c)oint A . dr`,

where the two open paths, traversed in opposite senses, close into a loop around the solenoid. The loop integral I just computed is `Phi`, and Stokes' theorem says the same thing geometrically:

`oint A . dr = int_S B . dS = Phi`,

with the spanning surface `S` necessarily cutting through the excluded core. The electron paths avoid the field completely; the surface used by Stokes' theorem cannot. So the magnetic effect is not a local field acting on the electron. It is a flat potential in the accessible region with nontrivial holonomy around a hole.

I should make sure gauge invariance still holds in this new form, since that was the original obstruction. Under `A -> A + grad Lambda` with `Lambda` a legal single-valued gauge function on the accessible region, each open branch integral changes by `Lambda(b) - Lambda(a)`, and both branches share the same endpoints, so the change cancels in the difference. Around the closed loop, `oint grad Lambda . dr = 0` precisely because single-valued `Lambda` returns to its starting value. The closed-loop phase is therefore gauge invariant, while the open-path phase and the pointwise value of `A` are not. The observable that survives the loop is the holonomy modulo `2 pi`:

`exp[i (q / hbar c)oint A . dr] = exp[i q Phi / (hbar c)]`.

This holonomy also tells me when the effect must vanish, and I can check that quantitatively rather than assert it. The phase is `q Phi/(hbar c)`. Work in units where `q`, `hbar`, `c` are scaled to 1, so the natural flux scale is `Phi_0 = 2 pi hbar c/|q| -> 2 pi`. At `Phi = n Phi_0`, the phase is `2 pi n` and `exp(2 pi i n) = 1` for `n = 0, 1, 2, 3` exactly: the holonomy is trivial. At fractional flux the phase is genuinely nontrivial: `Phi = Phi_0/4` gives `exp(i pi/2) = i`, `Phi = Phi_0/2` gives `exp(i pi) = -1`, and `Phi = 3 Phi_0/2` again gives `-1`, the same value as `Phi_0/2`. So the holonomy depends on flux only modulo `Phi_0`, and adding a whole flux quantum changes nothing. The period in flux is

`Phi_0 = 2 pi hbar c / |q| = h c / |q|`

in Gaussian units, or `h/|q|` in SI. The reason is exactly the single-valuedness I traced above: at integer `n`, the multivalued `chi = (Phi/2 pi) theta` produces a rephasing whose jump per loop, `q Phi/(hbar c) = 2 pi n`, is invisible to `exp(i ...)`, so the outside potential becomes globally removable.

I want to be careful about what is and is not being claimed. This does not mean the gauge-dependent vector potential has a unique measurable value at a point. It means quantum mechanics needs more than the local fields in the accessible region: a flat connection can still carry a nontrivial closed-loop phase. Replying "but the flux is a real field inside the solenoid" does not restore a local-force explanation, because the electron wavefunction is excluded from that region. It only says the same gauge-invariant holonomy can be evaluated two ways — as a line integral of `A` along the electron paths, or as a flux through a surface passing through inaccessible field. The line-integral description lives entirely where the electron can go.

There is one more check I want, because the split-beam argument treats the two arms almost geometrically and I would like the same conclusion to fall out of the actual differential equation. Solve the Schrodinger equation outside an ideal flux line. The kinetic angular piece of `(-i hbar grad - q A/c)` along `theta` is `(-i hbar/r) partial_theta - (q/c) A_theta`. Act on a single-valued angular mode `psi = e^{i m theta}` with integer `m` and `A_theta = Phi/(2 pi r)`: the operator returns `(hbar m - q Phi/(2 pi c))/r` times `psi`. So the effective angular label is not `m` but `m - alpha` with

`alpha = q Phi / (2 pi hbar c)`,

the same dimensionless number, since `alpha = (q Phi/(hbar c))/(2 pi)` is the loop phase divided by `2 pi`. Now I can test periodicity directly on the spectrum. The allowed labels form the set `{m - alpha : m in Z}`. Take a window of integers `m` and list `m - alpha`. For `alpha = 0.3` the set near zero is `..., -1.3, -0.3, 0.7, ...`; for `alpha = 1.3` it is `..., -2.3, -1.3, -0.3, ...`; for `alpha = 5.3` it is the same set shifted by relabeling `m`. Numerically the sets for `alpha` and `alpha + 1` coincide, because shifting `alpha` by an integer is absorbed by sending `m -> m + 1`, and `m + 1` is still an integer. So the exact wave problem is invariant under `alpha -> alpha + 1`, i.e. under `Phi -> Phi + Phi_0`, and shows flux-dependent scattering only through the fractional part of `alpha`. That matches the interferometer result rather than supplying a new mechanism: both say only the holonomy modulo `2 pi` is physical, one through single-valuedness of the full wavefunction and the other through the closed-loop integral.

Putting the pieces together, the experiment that would expose all of this is determined. Split a coherent electron beam. Send the two parts around opposite sides of a shielded flux tube so the magnetic field is confined to a region where the electron wavefunction has negligible support. Recombine the beams. Changing the enclosed flux by `Phi` shifts the interference phase by

`Delta theta = q Phi / (hbar c)`

in Gaussian units, with the SI replacement `q Phi / hbar`. The fringe pattern is periodic in `Phi_0`, and it shifts even though the local fields vanish along the electron paths. The potential has not become a directly measurable vector at each point; the physically observable object is its gauge-invariant holonomy. That is the lesson the computations keep pointing back to: in quantum mechanics a potential that is locally removable can still leave a global phase around a multiply connected field-free region, and an interference experiment reads that phase.
