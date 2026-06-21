# Anderson Localization

## Problem

A random lattice is usually treated as a metal with more scattering: impurities shorten the mean free path, and long-time motion remains diffusive. Anderson localization identifies the stronger quantum possibility. In a closed random tight-binding system, disorder can prevent diffusion altogether by making the exact eigenstates spatially localized.

## Model

Use localized site states `|j>` and amplitudes `a_j(t)`:

`i d a_j/dt = E_j a_j + sum_k V_jk a_k`.

The onsite energies `E_j` are independent random variables with width `W`. The hopping matrix elements `V_jk` are short-ranged, or decay sufficiently fast with distance. The competition is between energy mismatch, set by `W`, and hopping plus connectivity, set by `V_jk`.

Equivalently,

`H = sum_j E_j |j><j| + sum_{j != k} V_jk |j><k|`.

## Mechanism Theorem

Consider the local resolvent

`G_00(z) = <0|(z-H)^(-1)|0>`.

Expanding around the localized-site basis gives a locator series:

`G_00(z) = 1/(z - E_0 - Sigma_0(z))`,

where

`Sigma_0(z) = sum_k |V_0k|^2/(z-E_k) + sum_{k,l} V_0k V_kl V_l0/[(z-E_k)(z-E_l)] + ...`.

If the distribution of the random path terms is such that the locator series converges with probability one as `z` approaches the real axis, then `G_00` has isolated poles with nonzero residues rather than a continuous imaginary part. The initial local state has finite overlap with exact eigenstates concentrated near the starting site. Transport is absent: the particle may acquire virtual tails on nearby sites, but its probability does not relax into ordinary diffusion.

For a nearest-neighbor estimate with effective connectivity `K` and hopping scale `V`, the path-counting condition has the form

`F(K,W/V) < 1`.

In the original locator estimate, the critical disorder is of order the hopping times a connectivity-enhanced logarithm; representative bounds are obtained from equations like

`e K ln(W/2V) = W/2V`

and

`2 K ln(W/2V)/(1 - 4V^2/W^2) = W/2V`.

The constants are model-dependent approximations. The mechanism is not: path proliferation must be beaten by the probability cost of resonant energy denominators.

## Physical Content

The localized state is not a classical particle trapped in a single potential well. It is a coherent quantum eigenfunction built from many off-resonant virtual hops. Nearby sites with mismatched `E_j` admix weakly; rare resonant clusters can share appreciable amplitude; but when those resonances do not percolate, the eigenfunction decays away from its center with a localization length `xi`.

The decisive step is to study probability distributions of locator terms, not their ensemble averages. Averages are dominated by rare resonances and can falsely suggest a finite scattering rate. A specific sample contains specific denominators, and typical denominators can block real transport even when their average is singular.

## Dimensional And Scaling Perspective

Dimensionality enters through the growth of available paths and through interference from returns. In one-dimensional disordered systems, localization is especially robust. In two dimensions, later scaling theory treats the system as marginal: weak disorder can give very large localization lengths, but the infinite system lacks a stable ordinary metallic phase for the standard noninteracting, time-reversal-symmetric case. In three dimensions, the dimensionless conductance can flow either toward a metal or toward an insulator, allowing a mobility edge between extended and localized eigenstates.

The scaling version uses `g`, the dimensionless conductance, and

`beta(g) = d ln g / d ln L`.

In an ohmic conductor, large-`g` behavior is `beta(g) ~ d-2`. In a localized regime, `g(L) ~ exp(-L/xi)`, so `beta(g)<0`. A three-dimensional mobility edge corresponds to an unstable fixed point separating the two flows.

## Artifact

The final mechanism is a localization criterion for a random tight-binding Hamiltonian:

1. Write the problem in the site basis with random onsite energies and hopping.
2. Expand the local resolvent in hopping, producing a locator series over paths.
3. Count resonant paths by their probability distribution, not by an ensemble-averaged scattering rate.
4. If the path series converges, exact eigenfunctions are localized and a local initial condition does not diffuse.
5. If the path series breaks down, extended states can appear, with localized and extended spectral regions separated by a mobility edge.

Disorder is therefore not merely scattering that averages to diffusion. In a random quantum potential, coherent interference and energy-denominator statistics can stop transport.
