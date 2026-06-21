# The Casimir effect

## Problem it solves

Two neutral, perfectly conducting, parallel plates in vacuum can attract even with no applied classical electromagnetic field. The effect is explained by the quantum electromagnetic vacuum: conducting boundaries change the allowed normal modes of the field, and the finite change in zero-point energy depends on the plate separation.

## Mode spectrum

Let the plates be at `z = 0` and `z = a`, with area `A` large enough that edge effects are neglected. The transverse wave vector is continuous, while the normal wave number is quantized:

`k_z = n pi / a`.

For a scalar polarization,

`omega_n(k) = c sqrt(k^2 + (n pi / a)^2)`,

and the formal zero-point energy per area is

`E_s/A = (hbar c / 2) sum_{n=1}^infty int d^2k/(2 pi)^2 sqrt(k^2 + (n pi / a)^2)`.

## Regularized finite energy

The absolute zero-point sum diverges. The physical object is the finite difference between the plate geometry and the corresponding empty or far-separated geometry. Using analytic regularization,

`sqrt(Q) = -1/(2 sqrt(pi)) int_0^infty dt t^(-3/2) exp(-tQ)`,

so

`E_s/A = -hbar c/(16 pi^(3/2)) sum_n int_0^infty dt t^(-5/2) exp[-t(n pi/a)^2]`.

With

`int_0^infty dt t^(-5/2) exp(-beta t) = beta^(3/2) Gamma(-3/2)`,

`Gamma(-3/2) = 4 sqrt(pi)/3`, and `zeta(-3) = 1/120`,

`E_s/A = -pi^2 hbar c/(1440 a^3)`.

The electromagnetic field supplies two polarizations in the ideal parallel-plate result, giving

`E/A = -pi^2 hbar c/(720 a^3)`.

## Force per area

The pressure follows from virtual work:

`P = F/A = -d(E/A)/da`.

Therefore

`F/A = -pi^2 hbar c/(240 a^4)`.

The negative sign means attraction: the finite vacuum energy decreases as the plates move closer together.

## Physical meaning

The result is not a force from a classical substance filling the gap. It is a spectral effect. The vacuum of a quantum field is the ground state of its allowed modes, and ideal conducting plates alter those modes by imposing boundary conditions. The divergent absolute vacuum energy is discarded through subtraction or regularization; the separation-dependent residue is finite and measurable.
