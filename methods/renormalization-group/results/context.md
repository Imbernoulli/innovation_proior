## Problem

Critical phenomena expose a failure mode of ordinary physical modeling. Near a Curie point or
liquid-vapor critical point, the correlation length xi grows without bound, so fluctuations are not
confined to one microscopic or macroscopic scale. They occur at every intermediate scale. Mean-field
and Landau-Ginzburg descriptions can write a smooth order-parameter theory, but below four dimensions
they give the wrong critical exponents because they average away the long-wavelength fluctuations
that actually control the transition.

The central puzzle is therefore not just how to solve a complicated microscopic model. It is how an
analytic-looking partition function can produce non-analytic macroscopic behavior, and why different
microscopic systems can share the same singular laws.

## What Changed

The renormalization group reframes a physical theory as a scale-dependent effective description.
Instead of asking for one final model of all degrees of freedom at once, it asks how the description
changes when the observer changes scale. Integrate out the shortest-wavelength modes, rewrite the
remaining theory in dimensionless variables, and repeat. The result is a flow in the space of
couplings, with the logarithm of length scale playing the role of time.

This is the distinctive Wilsonian move: "changing scale" becomes a dynamical system. A theory is no
longer a fixed microscopic object to be rearranged; it is a point moving through a space of effective
Hamiltonians.

## Earlier Baselines

Landau theory reduced the problem to an analytic free energy in an order parameter and predicted
mean-field exponents such as beta = 1/2 and nu = 1/2. This worked above the upper critical dimension
but failed for many real three-dimensional systems and for the exactly solved two-dimensional Ising
model.

Gell-Mann and Low had already introduced scale-dependent couplings in quantum electrodynamics, but
their flow involved a fixed, small set of parameters. Kadanoff's block-spin picture gave the right
physical intuition for coarse-graining and scaling, but it assumed the blocked system could be
described by the same few couplings and did not supply a general calculational transformation.

## Wilson's Reframing

Wilson's method does not preserve the microscopic form of the model. Coarse-graining generally
generates new local interactions: next-neighbor terms, multi-spin terms, higher powers in a field
theory, and so on. The effective Hamiltonian is allowed to become more complicated. Locality and
dimensional analysis then order the generated couplings by importance, making controlled truncation
possible.

Once this enlarged coupling space is accepted, the renormalization-group step is conceptually simple:
eliminate one scale, rescale, and read off the new couplings. Iterating this map produces fixed
points. A critical fixed point is scale invariant and has xi = infinity. Linearizing the flow around
it separates directions into relevant directions that grow under coarse-graining, irrelevant
directions that decay, and marginal directions that require further analysis.

## Evaluation Grounding

The method must explain both numbers and structure. Fixed points explain why critical systems become
scale invariant. Relevant directions explain why only a few experimental knobs, such as temperature
and magnetic field, must be tuned to reach criticality. Irrelevant directions explain universality:
microscopic details flow away and do not affect the critical exponents.

Simple decimation of the one-dimensional Ising chain gives an exact miniature example:
K' = (1/2) log cosh(2K), equivalently tanh K' = tanh^2 K. Its only finite fixed point is K = 0, so
every finite temperature flows to the disordered fixed point and there is no finite-temperature
critical transition. In higher-dimensional phi^4 theory, the epsilon = 4 - d expansion makes the
nontrivial Wilson-Fisher fixed point perturbatively accessible just below four dimensions.
