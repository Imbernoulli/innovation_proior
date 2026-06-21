I want to isolate what is genuinely new in Gromov compactness. Ordinary compactness arguments are
comfortable when all objects already sit in one fixed space. If I have functions on one compact
domain, Arzela-Ascoli tells me to control oscillation and values. If I have maps into one target, I
can ask for uniform convergence, weak convergence, or smooth convergence in coordinates. But a
sequence of geometric spaces does not naturally give me "the same point" in every term of the
sequence. The first problem is not a missing estimate; it is the wrong notion of convergence.

So the first move is to stop treating coordinates or embeddings as part of the data. A compact
metric space is already a complete object: its distances say what it means for points to be near.
To compare two such spaces, I do not need a privileged coordinate system. I can embed both
isometrically into a third metric space and ask whether their images are Hausdorff-close, or I can
use a correspondence whose distortion of pairwise distances is small. This makes convergence a
relation among intrinsic distance structures. The limit is allowed to be a metric space, not
necessarily a smooth manifold with the same charts as the approximating sequence.

Once convergence is defined intrinsically, the compactness hypothesis also changes shape. Instead
of bounding derivatives in one coordinate chart, I need bounds that prevent the metric spaces from
having infinitely much unresolved structure at small scales. Uniform covering numbers do exactly
that: for every epsilon, each space can be covered by at most N(epsilon) balls of radius epsilon.
This is total boundedness made uniform across the family. With a diameter bound and such
scale-by-scale covering bounds, I can choose finite epsilon-nets in each space with uniformly
bounded size, record their finite distance matrices, and pass to convergent subsequences of those
matrices. A diagonal extraction over epsilon scales then builds a limiting metric object. The
important point is that the extraction happens through distances between finitely many sample
points, not through coordinates of those points in a common ambient space.

Riemannian geometry supplies these covering bounds from geometric hypotheses. Curvature and volume
comparison estimates turn lower Ricci bounds, dimension bounds, and diameter bounds into scale
control. That is why a theorem about abstract compact metric spaces becomes a theorem about
families of manifolds. The compactness conclusion is weaker than smooth convergence, but it is much
more robust: collapsing, singularities, and loss of differentiable structure can appear in the
limit, while the metric information remains meaningful.

The same thought appears in the compactness theorem for J-holomorphic curves, although the
quantity doing the work is energy rather than covering number. A naive parametrized limit can fail
because domains reparametrize and energy concentrates. But bounded energy says that concentration
cannot happen everywhere. Away from concentration points, elliptic estimates give ordinary smooth
convergence after choosing good parametrizations. At a concentration point, one rescales. If a
fixed positive amount of energy is present, the rescaled sequence yields a nontrivial bubble. Since
the total energy is bounded and each nontrivial bubble costs positive energy, only finitely many
bubbles can appear above any threshold in the compactness argument. The limit is therefore not a
single original map, but a stable map with components, nodes, and energy accounting.

This explains why bubbling is not a defect tacked onto the theorem. It is the exact price of making
compactness intrinsic to the geometric problem. If I insisted on convergence as maps from one fixed
smooth domain, the sequence would simply fail to converge. If I enlarge the category of limits to
stable curves, the lost energy has a place to go, and compactness is restored. The theorem does not
erase degeneration; it classifies the allowable degeneration under an energy bound.

The shared insight is now clear. Gromov compactness is a way of asking for subsequential limits
without first choosing a global coordinate language. For spaces, metric bounds produce finite
approximations at every scale. For curves, energy bounds produce smooth convergence away from
finitely controlled concentration and bubbles at the exceptional scales. In both cases the compact
object is recovered by what the family itself controls. The method turns the question from "do
these parametrized points converge?" into "do the intrinsic measurements of the family force a
limit after passing to the right category?"

That is the mental shift from pointwise parametrization to intrinsic compactness principles. The
limit is not guaranteed because every coordinate function behaves nicely. It is guaranteed because
the family cannot create unlimited new metric complexity or unlimited hidden energy. The theorem
therefore gives a way to study moduli: prove uniform bounds that are natural to the geometry, then
compactify by adding exactly the limit objects those bounds permit.

