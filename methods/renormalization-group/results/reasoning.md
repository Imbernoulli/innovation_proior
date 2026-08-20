The starting point is the failure of one-scale thinking. At a critical point the correlation length
diverges, so fluctuations exist at all sizes at once. A microscopic lattice Hamiltonian is too detailed
to solve directly, while a smooth mean-field order parameter has already thrown away the fluctuations
that dominate the singular behavior. The method has to respect the fact that the physics is organized
by scale.

Kadanoff's block-spin idea gives the right image: replace small blocks by effective degrees of
freedom and compare the blocked system with the original one. But if the blocked model is forced to
have the same microscopic form with only a temperature and field, that closure is an assumption, and
an assumption is worth testing rather than trusting. Work the block-spin step through by hand on the
two-dimensional square lattice: eliminate the center spin of each five-spin block by summing over its
two states, which turns the local piece of the Boltzmann weight into 2cosh[J(sum of the four
neighboring spins)], then expand that factor over the sixteen configurations the four outer spins can
take. The sum does not fold back onto one rescaled coupling. It splits into a coupling between
adjacent outer spins, a second and independent coupling between the two outer spins that now sit
diagonally across the decimated block, and a four-spin coupling tying all four corners together --
more coupling types than the temperature and field the closure assumption allows, after a single
decimation step, with still more appearing at every further step. That matches what years of trying
to derive renormalization-group transformations for a fixed handful of couplings -- Gell-Mann and
Low's single electric charge, Kadanoff's temperature and field -- had already run into: no such
transformation could be made to work, because the fixed-coupling requirement itself does not survive
contact with an explicit calculation. Wilson's decisive step is to stop treating the extra terms the
decimation produces as a defect to truncate away. They are the natural coordinates of the effective
theory at the new scale, and once the fixed-coupling requirement is dropped, defining a
renormalization-group transformation at all stops being the hard problem -- finding a computable
approximation to it becomes the hard problem instead.

This changes the object being studied. The object is not a single Hamiltonian that is repeatedly
massaged into a familiar microscopic form. It is a point in a usually infinite-dimensional coupling
space. A coarse-graining step integrates out short-distance degrees of freedom, then rescales length
and fields so the result can be compared with the starting description. Repeating that step defines a
map on coupling space. In this sense, Wilson turns scale change into a dynamical system.

The logarithm of length scale acts like time. A microscopic system starts at some initial point, and
the renormalization-group transformation gives its trajectory as shorter-distance information is
discarded. The trajectory is a flow of effective descriptions: what interactions are visible, which
ones grow, and which ones become negligible as the observer moves to longer distances.

The central organizing objects of this flow are fixed points. At a fixed point, rescaling leaves the
dimensionless theory unchanged. That is exactly the mathematical expression of scale invariance. A
critical fixed point is the fixed point with infinite correlation length, so the non-analytic behavior
of the thermodynamic limit is no longer mysterious: it comes from approaching a scale-invariant
attractor or saddle in coupling space.

Universality follows from the geometry near the fixed point. Linearize the renormalization-group map.
Eigenvectors with eigenvalues larger than one are relevant directions; deviations along them grow
under coarse-graining, so they must be tuned away to reach criticality. Temperature and external
field are the standard examples. Eigenvectors with eigenvalues smaller than one are irrelevant
directions; deviations along them shrink, so microscopic details disappear. Marginal directions sit
at the boundary and need higher-order analysis.

This classification explains why systems with different atomic details share the same exponents. If
their flows enter the same basin of attraction and differ mainly in irrelevant coordinates, the
long-distance critical behavior is controlled by the same fixed point. The exponents are not fitted
features of each material. They are eigenvalues of the linearized scale-change map.

The one-dimensional Ising decimation illustrates the logic without approximation, and it is the
exception rather than the rule for the coupling count: a chain has no closed loops for a decimation
step to generate new couplings around, so summing over every other spin folds back onto a single
coupling exactly, unlike the two-dimensional block above. That single decimation step gives
K' = (1/2) log cosh(2K), or tanh K' = tanh^2 K. Finite K flows toward K = 0, while
K = infinity is the zero-temperature unstable fixed point. The absence of a finite-K critical fixed
point is the renormalization-group reason there is no finite-temperature phase transition in one
dimension.

For real critical phenomena the coupling space is larger and the transformation is only approximate,
but the structure is the same. In phi^4 theory, four dimensions mark the upper critical dimension:
above four, the Gaussian fixed point gives mean-field behavior; below four, a nontrivial
Wilson-Fisher fixed point appears. Because it is close to the Gaussian fixed point when epsilon =
4 - d is small, the fixed point and its linearized eigenvalues can be computed perturbatively.

The innovation is therefore not a clever reorganization of the microscopic model. It is a change in
what counts as the solution. The solution is the flow of effective theories under changes of scale,
with critical phenomena explained by fixed points and by the relevant and irrelevant directions
around them.
