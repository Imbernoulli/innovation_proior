# Gromov Compactness

Gromov compactness is the principle that a controlled family of geometric objects has a
subsequential limit after convergence is defined intrinsically, rather than through a fixed
coordinate chart, parametrization, or embedding.

Its distinctive insight is this:

> Compactness should be forced by the quantities the geometry itself controls: metric scale bounds
> for spaces, and energy bounds for curves.

For compact metric spaces, the relevant control is uniform total boundedness. If a family has, for
every epsilon > 0, a uniform bound on the number of epsilon-balls needed to cover each space, then
one can extract a Gromov-Hausdorff convergent subsequence. Riemannian hypotheses such as curvature,
dimension, and diameter bounds are useful because they often imply these covering estimates. The
limit may be singular or collapsed, but it remains an intrinsic metric object.

For J-holomorphic curves, the relevant control is energy. A sequence with uniformly bounded energy
may fail to converge as parametrized maps because energy can concentrate and bubbles can form. The
compactness theorem says that, after reparametrization and after allowing stable limits, a
subsequence converges: smoothly away from concentration points, and with bubbles or nodes recording
the energy that would otherwise disappear.

The mechanism is the same in both settings. Instead of tracking the same point through a coordinate
system, the proof extracts finite intrinsic data:

- in metric compactness, finite epsilon-nets and their distance matrices;
- in curve compactness, regions of bounded energy density plus finitely many concentration points;
- in both cases, a diagonal subsequence across scales.

This is why Gromov compactness changed the way geometric families are studied. It turns compactness
from a pointwise parametrized question into an intrinsic one:

1. Identify the natural geometric measurements.
2. Prove uniform bounds on those measurements.
3. Enlarge the notion of limit only as much as the bounds require.
4. Extract a subsequence in that intrinsic compactification.

The payoff is that degeneration becomes analyzable rather than fatal. Collapsing spaces still have
metric limits. Curves with concentrating energy still have stable-map limits. The method does not
pretend that geometric sequences behave like uniformly convergent coordinate functions; it replaces
that fragile expectation with a compactness principle built from distance and energy.

