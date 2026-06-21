# The Aharonov-Bohm effect

## Problem it solves

A charged quantum particle can move only through regions where `E = 0` and `B = 0`, yet still show an interference shift caused by electromagnetic potentials. Classical field-only reasoning predicts no effect because the Lorentz force vanishes along the particle paths. The quantum question is whether the phase of the wavefunction preserves information about the potential around a loop.

## Core idea

The vector potential is not observed as a gauge-dependent local value. The observable is the gauge-invariant closed-loop phase, or holonomy,

`exp[i(q / hbar c)oint A . dr]`

in Gaussian units. A field-free region can still be multiply connected, so `curl A = 0` locally does not imply that `A` can be gauged away globally by a single-valued phase convention. Around an excluded flux tube, the loop integral of `A` is the enclosed flux.

## Magnetic effect

For a coherent charged beam split into two arms that pass around opposite sides of a confined magnetic flux `Phi`, with the particle excluded from the magnetic-field region,

`Delta theta = (q / hbar c) [int_1 A . dr - int_2 A . dr]`

and the two paths form a closed contour, so

`Delta theta = (q / hbar c)oint A . dr = (q / hbar c)Phi`.

In SI units the phase is

`Delta theta = q Phi / hbar`.

The interference pattern is periodic in the flux quantum

`Phi_0 = 2 pi hbar c / |q| = h c / |q|`

in Gaussian units, or `h/|q|` in SI. For an electron, the sign is set by `q = -e`; the observed fringe displacement depends on the relative phase modulo `2 pi`.

## Gauge invariance and topology

In any simply connected field-free patch, `B = curl A = 0` permits `A = grad chi`, and the wavefunction can be rephased:

`psi = exp[(i q / hbar c) chi] psi_0`.

This removes `A` locally. Around a solenoid exterior, however, a representative potential is

`A_theta = Phi/(2 pi r)`,

which equals `grad[(Phi/2 pi)theta]` only with a multivalued angle `theta`. After one circuit,

`theta -> theta + 2 pi`,

so the wavefunction acquires

`exp[i q Phi/(hbar c)]`.

A legal single-valued gauge transformation changes open-path phases but leaves the closed-loop integral unchanged:

`oint grad Lambda . dr = 0`.

Thus the physical object is not the local value of `A`, but the equivalence class captured by the holonomy. Local field strengths in the accessible region underdescribe the quantum state because they do not encode the loop phase around the excluded flux.

## Electric analogue

If two electron wave packets pass through separate conducting tubes and each packet is fully inside its tube while the tube potential is changed, the electron can remain in a region of zero electric field. A spatially uniform scalar potential adds `q phi(t)` to the Hamiltonian and contributes only a phase on each branch:

`theta_j = -(q / hbar) int phi_j(t) dt`.

The recombined interference phase is

`Delta theta_electric = -(q / hbar)[int phi_1(t) dt - int phi_2(t) dt]`.

Again, a single branch sees only an unobservable overall phase, while the split-and-recombine experiment turns the branch phase difference into a fringe shift.

## Protocol artifact

1. Prepare a coherent charged-particle beam and split it into two spatially separated arms.
2. Place a narrow, shielded flux tube between the arms so the wavefunction has negligible support where `B` is nonzero.
3. Recombine the arms and record the interference pattern.
4. Vary the enclosed magnetic flux `Phi` while keeping ordinary field leakage away from the electron paths.
5. Predict the fringe phase shift by `Delta theta = q Phi/(hbar c)` in Gaussian units, or `q Phi/hbar` in SI, modulo `2 pi`.

The result is a direct test that quantum phase can make electromagnetic potentials physically effective in a field-free accessible region. Field-only local reasoning misses the effect because it discards the global information contained in the potential around a noncontractible loop.
