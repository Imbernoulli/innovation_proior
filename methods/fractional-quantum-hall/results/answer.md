# Fractional Quantum Hall Correlated Fluid

## Problem

At fractional filling of the lowest Landau level, independent electrons cannot explain an incompressible Hall plateau. The kinetic energy is quenched, the Landau level is only partially occupied, and no one-electron Landau gap remains. The missing object is a many-body state selected by Coulomb repulsion inside the lowest-Landau-level degeneracy.

## Wavefunction

Use complex coordinates `z_j = x_j + i y_j` and magnetic length `ell = sqrt(hbar/(eB))`. For odd integer `m`, define

`Psi_m(z_1,...,z_N) = prod_{i<j}(z_i-z_j)^m exp[-sum_j |z_j|^2/(4 ell^2)]`.

For `m = 3`, this is the `nu = 1/3` state.

Why the formula works:

- The Gaussian is the lowest-Landau-level orbital factor.
- The analytic polynomial keeps the state inside the lowest Landau level.
- Odd `m` makes the wavefunction antisymmetric for electrons.
- The factor `(z_i-z_j)^m` creates an order-`m` zero whenever two electrons meet, strongly suppressing Coulomb-costly close approaches.
- The highest coordinate power is approximately `mN`, so `N_phi ~= mN` and `nu = N/N_phi -> 1/m`.

## Mechanism

The state is a correlated incompressible quantum fluid. It is not independent electrons occupying one third of ordinary orbitals. In the lowest Landau level, Coulomb repulsion chooses a collective polynomial whose zeros bind correlation holes to electron positions. The squared wavefunction maps to a screened two-dimensional plasma, which fixes a uniform density of one electron per `m` flux quanta and supports a finite cost for changing that density.

## Quasiparticle Artifact

A quasihole at `eta` is

`Psi_h(eta; z_1,...,z_N) = prod_j (z_j - eta) Psi_m(z_1,...,z_N)`.

The extra zero repels electronic density from `eta`. Flux insertion gives the charge directly: one inserted flux quantum draws charge `e/m` through a large loop because the fluid density is `1/m` electron per flux quantum. The remaining localized defect therefore carries fractional charge `+e/m` for a quasihole, and the corresponding quasielectron carries `-e/m`.

For the primary `nu = 1/m` fluids, the quasiparticle exchange phase is `theta = pi/m`. Thus the elementary charged excitations are fractional anyons of the fluid, not screened electrons.

## Distinctive Insight

The right object is the whole incompressible fluid. Lowest-Landau-level degeneracy creates the arena, Coulomb repulsion selects the correlated zeros, the zeros keep electrons apart while preserving a uniform liquid, and the fixed density-flux relation turns local defects into fractionally charged quasiparticles.
