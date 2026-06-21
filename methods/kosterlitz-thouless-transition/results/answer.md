# Kosterlitz-Thouless Transition

## Problem

A two-dimensional system with a continuous symmetry cannot have ordinary spontaneous
long-range order at finite temperature. In the XY model, superfluid film, or two-dimensional
crystal, long-wavelength spin waves, phase waves, or phonons destroy the conventional order
parameter. The transition, if it exists, must therefore be something other than local
order-parameter symmetry breaking.

## Mechanism

Use the phase field, but do not keep only its smooth part. For the XY model,

$$
H = -J\sum_{\langle ij\rangle}\cos(\theta_i-\theta_j)
  \simeq {J\over 2}\int d^2r\,|\nabla\theta|^2 .
$$

Smooth spin waves give zero magnetization and algebraic correlations. The additional
degrees of freedom are vortices:

$$
q = {1\over 2\pi}\oint \nabla\theta\cdot d\ell \in \mathbb{Z}.
$$

A unit isolated vortex has

$$
E_v = \pi J\ln(L/a)+E_c,\qquad
S_v = 2k_B\ln(L/a),
$$

so

$$
\Delta F_v = E_c + (\pi J - 2k_B T)\ln(L/a).
$$

At low temperature isolated vortices are suppressed; at high temperature entropy favors
them. Neutrality makes the elementary excitation a vortex-antivortex pair, whose energy
grows logarithmically with separation:

$$
E_{\rm pair}(r) = 2E_c + 2\pi J\ln(r/a).
$$

Thus the vortex sector is a neutral two-dimensional Coulomb gas.

## RG Flow Artifact

Let

$$
K = {J\over k_B T}
$$

or, for a superfluid, the dimensionless phase stiffness, and let

$$
y = e^{-E_c/k_B T}
$$

be the vortex fugacity. Integrating out vortex-antivortex pairs with separations between
`a` and `a e^{dl}` gives the leading dilute-gas flow

$$
{dK^{-1}\over dl} = 4\pi^3 y^2 + O(y^4),
$$

$$
{dy\over dl} = (2-\pi K)y + O(y^3).
$$

Consequences:

- `y = 0` is a fixed line: the long-distance theory below the transition is a spin-wave
  theory with finite renormalized stiffness and algebraic correlations.
- For `pi K > 2`, fugacity is irrelevant: vortices remain bound in neutral pairs.
- For `pi K < 2`, fugacity is relevant: vortex-antivortex pairs unbind, free vortices
  screen the logarithmic interaction, and correlations become exponential.
- The transition is the separatrix ending at

$$
K_R(T_c^-) = {2\over \pi}.
$$

For the XY stiffness this gives the universal stiffness jump

$$
J_R(T_c^-) = {2k_B T_c\over \pi}.
$$

For a neutral superfluid film, the same condition is the Nelson-Kosterlitz jump, with the
phase-stiffness conversion to areal superfluid mass density:

$$
{\rho_s^R(T_c^-)\over T_c} = {2m^2 k_B\over \pi\hbar^2}.
$$

## Critical Behavior

Below the transition:

$$
\langle s(0)\cdot s(r)\rangle \sim r^{-\eta(T)},\qquad
\eta(T) = {1\over 2\pi K_R(T)},
$$

with no spontaneous magnetization and infinite susceptibility. At the transition,

$$
\eta(T_c)= {1\over 4}.
$$

Above the transition, free vortices create a finite screening length. The correlation length
has an essential singularity rather than a power-law divergence:

$$
\xi(T>T_c) \sim a\exp\left({b\over \sqrt{(T-T_c)/T_c}}\right).
$$

The singular free-energy density scales as `xi^{-2}`, so thermodynamic singularities are
extremely weak. The defining physical event is vortex-antivortex unbinding, not the onset
of a nonzero local order parameter.
