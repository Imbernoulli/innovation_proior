## Research question

Massless gauge fields are not a nuisance one can simply patch by hand. Local gauge invariance is the organizing principle that makes electrodynamics and Yang-Mills theory coherent, but it also seems to forbid the direct vector mass term `m^2 A_mu A^mu/2`: under a local transformation it is not invariant. That is a severe problem for short-range fundamental forces, because a force mediated by a massless vector field is long range, while a massive vector field would give the finite range suggested by weak-interaction phenomenology.

The question is whether a gauge theory can produce massive vector bosons while the Lagrangian keeps its local symmetry exact. A solution has to satisfy three pressures at once: it must not insert an explicit gauge-breaking Proca mass, it must not leave a physical massless scalar behind merely because a continuous symmetry is hidden in the vacuum, and it must account for the number of physical degrees of freedom before and after the mass appears.

## Background

The global symmetry-breaking story is already sharp. A complex scalar field with a Mexican-hat potential has a circle of degenerate minima. Choosing one vacuum gives a massive radial fluctuation and a massless angular fluctuation. Goldstone, Salam, and Weinberg turn this into a theorem: in a manifestly Lorentz-invariant relativistic theory with a broken continuous global symmetry, either the vacuum is invariant or a massless spinless particle appears. For particle physics this is a danger, not just a detail, because unwanted massless scalars would mediate long-range effects not seen in the spectrum.

Superconductivity points in a different direction. In a neutral superfluid or neutral broken global U(1) system, the phase mode is the expected massless mode. In a charged superconductor, long-range electromagnetic interactions alter the spectrum: Anderson describes the would-be Goldstone mode as becoming part of a massive plasmon/electromagnetic excitation. This suggests that coupling a phase mode to a gauge field can change the conclusion, but by itself it does not yet give a relativistic field-theory model that respects gauge redundancy.

Gauge theory also carries a conceptual warning. A local gauge transformation relates different descriptions of the same physical configuration, not different observable states. Before any mass mechanism is trusted, the phase of a charged scalar field must be separated into what is physical and what is merely gauge. Gauge fixing can hide Lorentz covariance in intermediate variables, as Schwinger's Coulomb-gauge treatment of electrodynamics illustrates, without changing Lorentz-invariant physics.

## Baselines

The direct Proca route writes `-1/4 F_{mu nu}F^{mu nu} + m^2 A_mu A^mu/2`. It has exactly the right massive-vector degree count in isolation, but it explicitly breaks the gauge invariance that made the vector theory controlled. For non-Abelian gauge fields this is especially costly: the mass term is not compatible with the local symmetry structure one wants to preserve.

The global Goldstone route keeps the scalar Lagrangian symmetric and lets the vacuum choose a point on the valley. It explains how a symmetric theory yields an asymmetric expansion point, but it predicts a physical massless scalar for every broken continuous generator. That is the wrong outcome if the goal is a short-range vector interaction with no extra long-range scalar.

The superconductivity route gives the most suggestive physical precedent. The electromagnetic field in a superconductor behaves as if it has acquired a mass and the phase oscillation is not a separate massless particle. The limitation is that this argument lives in condensed-matter conditions with a medium, Coulomb forces, and nonrelativistic dynamics. The relativistic gauge-theory question remains open.

Gauge fixing provides a formal baseline rather than a mass mechanism. It can remove redundant variables and expose physical polarizations, but an arbitrary gauge choice cannot by itself create a legitimate mass. The mass must emerge from a gauge-invariant Lagrangian and survive in gauge-independent variables.

## Evaluation settings

The first test is algebraic consistency in the simplest local U(1) scalar gauge theory: keep the gauge-invariant kinetic term for a charged scalar, use a symmetry-respecting potential with a nonzero vacuum modulus, expand around the low-energy configuration, and read off the quadratic spectrum.

The second test is degree-of-freedom counting. Before the effect, the fields contain two transverse polarizations of a massless vector and two real scalar fluctuations. Afterward, the physical spectrum should still contain four propagating degrees of freedom, arranged without a physical massless Goldstone scalar.

The third test is gauge independence. A derivation may use a convenient gauge, but the same mass spectrum must be expressible in gauge-invariant variables. The phase variable of the scalar must not be mistaken for an observable particle if it is only labeling gauge-equivalent representatives.

The fourth test is compatibility with the global limit. When the gauge coupling is removed, the model should reduce to the ordinary broken global U(1) scalar theory with a massive radial mode and a massless angular mode. The change in the spectrum must come from making the symmetry local and coupling it to the gauge field.

## Code framework

The natural scaffold is not executable code but a field-theory template. Start with a complex scalar field `phi`, a vector field `A_mu`, a covariant derivative `D_mu`, a gauge-invariant field strength `F_{mu nu}`, and a scalar potential whose minimum has nonzero modulus:

`L = (D_mu phi)^* D^mu phi - V(phi^* phi) - (1/4)F_{mu nu}F^{mu nu}`.

The open slots are the choice of gauge-invariant field variables, the expansion around the minimum, the quadratic spectrum, and the physical interpretation of the scalar phase. The final artifact should fill those slots by showing which combination of `A_mu` and the scalar phase is gauge invariant, which fields are massive, which variables are redundant, and how the physical degrees of freedom are counted.

