## Classification Target

Closed 3-manifolds are not classified by local inspection. A manifold may look ordinary in every small ball while its global topology is governed by connected sums, essential spheres, incompressible tori, finite-volume hyperbolic pieces, Seifert-fibered pieces, or graph-manifold behavior.

The organizing conjectural picture says that the manifold should split into geometric pieces. The first splitting is along essential 2-spheres, producing prime summands. A second splitting, when needed, is along incompressible tori. After those cuts, each remaining piece should carry one of the standard three-dimensional geometries.

## Dynamic Route

One available route is to stop searching directly for the splitting surfaces and instead place an arbitrary smooth Riemannian metric on the manifold. The metric can then be evolved by the intrinsic equation `partial_t g_ij = -2 R_ij`.

This equation behaves like a nonlinear heat equation for geometry. In favorable curvature settings it smooths and rounds the metric, and in some special cases convergence already identifies the underlying topology. The attraction of the route is that it turns a discrete topological classification problem into a continuous geometric process.

## Singular Obstruction

The same evolution that smooths favorable metrics can form finite-time singularities. Curvature may become unbounded in a small region while staying controlled elsewhere. At such a time the differential equation cannot simply be continued as a smooth flow.

The hopeful picture is that the singular region is topologically meaningful. A shrinking cylindrical neck has cross-sections that are 2-spheres, and cutting through such a neck is exactly the kind of operation used in topological decomposition. The difficulty is that the flow must prove the neck picture rather than assume it.

## Missing Control

A direct singularity strategy needs compactness after rescaling around high-curvature points. That requires more than curvature bounds. It also needs a lower volume scale, otherwise the rescaled pictures can collapse and fail to produce usable local models.

The strategy must also decide when a high-curvature region is standard enough to justify a cut, when it is instead a terminal positive-curvature component, and how the estimates survive after the cut. Without these controls, surgery is only a plausible heuristic.

## Required Artifact

The final artifact should be an evolution-and-surgery protocol with a proof pipeline: start from any smooth metric, run the geometric heat equation, prove enough compactness and model control near singular regions, continue past legitimate singular times, and read off the resulting geometric decomposition.

It should also distinguish the case where all components disappear in finite time from the case where a long-time geometric decomposition remains to be analyzed.
