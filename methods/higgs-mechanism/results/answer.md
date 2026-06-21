# Higgs Mechanism

## Problem

A gauge vector mass term,

`(1/2)m^2 A_mu A^mu`,

would make a vector-mediated force short range, but it is not invariant under local gauge transformations. A broken global continuous symmetry has the opposite problem: it keeps the Lagrangian symmetric but produces a physical massless Goldstone boson. The mechanism solves both at once: it generates massive vector bosons in a locally gauge-invariant theory, with no physical massless Goldstone scalar left over.

## Abelian Prototype

Use a complex scalar field coupled to a local U(1) gauge field:

`D_mu = partial_mu - i e A_mu`,

`phi -> exp(i alpha(x)) phi`,

`A_mu -> A_mu + (1/e) partial_mu alpha`.

The gauge-invariant Lagrangian is

`L = (D_mu phi)^*D^mu phi - lambda(phi^*phi - v^2/2)^2 - (1/4)F_{mu nu}F^{mu nu}`,

with `lambda > 0`.

Write the scalar in polar variables:

`phi = (rho/sqrt(2)) exp(i theta)`.

Then

`(D_mu phi)^*D^mu phi = (1/2)(partial_mu rho)^2 + (1/2)rho^2(partial_mu theta - e A_mu)^2`.

The combination

`B_mu = A_mu - (1/e)partial_mu theta`

is gauge invariant, and `F_{mu nu}(A) = F_{mu nu}(B)`. Thus

`L = (1/2)(partial rho)^2 + (1/2)e^2 rho^2 B_mu B^mu - lambda(rho^2 - v^2)^2/4 - (1/4)B_{mu nu}B^{mu nu}`.

Expanding `rho = v + h` gives the quadratic Lagrangian

`L_quad = (1/2)(partial h)^2 - (1/2)(2 lambda v^2)h^2 - (1/4)B_{mu nu}B^{mu nu} + (1/2)e^2 v^2 B_mu B^mu`.

So the physical masses are

`m_h^2 = 2 lambda v^2`,

`m_B^2 = e^2 v^2`.

The field `theta` is not a physical massless scalar. It is the would-be Goldstone mode, and it supplies the longitudinal polarization of the massive vector field `B_mu`.

## Degree Count

Before the mechanism:

- massless gauge vector: 2 transverse polarizations
- complex scalar: 2 real scalar degrees of freedom
- total: 4 physical degrees of freedom

After the mechanism:

- massive vector: 3 polarizations
- massive real scalar `h`: 1 degree of freedom
- total: 4 physical degrees of freedom

Nothing is lost. The Goldstone degree of freedom is reclassified as the vector's longitudinal polarization.

## General Gauge Group

For a scalar multiplet with vacuum value `v` and covariant derivative

`D_mu phi = partial_mu phi - i g A_mu^a T^a phi`,

the scalar kinetic term gives the gauge-boson mass matrix

`(M_V^2)_{ab} = g^2 (T^a v)^\dagger (T^b v)`.

Generators satisfying `T^a v = 0` remain unbroken in the physical spectrum and their gauge bosons stay massless. Generators with `T^a v != 0` produce massive gauge bosons, with the corresponding would-be Goldstone directions becoming longitudinal vector polarizations.

## Mechanism

The mass is not put into the gauge field explicitly. It emerges from the gauge-invariant term `|D_mu phi|^2` once the scalar field has a nonzero vacuum modulus. Local gauge redundancy makes the scalar phase nonphysical, and the gauge field absorbs that phase as its longitudinal mode. Gauge invariance remains exact; the physical spectrum contains massive vector bosons and a massive scalar radial excitation, not explicit gauge breaking and not unwanted physical Goldstone bosons.

