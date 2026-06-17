I have to start somewhere on the ladder, and the scaffold default is not a real start: with the
operator slot empty, every point is mapped from its own seven features and three coordinates
straight to four output channels, with no point ever seeing any other point. A PDE field is the
opposite of pointwise — the pressure at the rear of a car is set by the geometry at its nose, the
velocity in the wake by the whole upstream body. A model with zero cross-point interaction cannot
represent that coupling at all; it can only learn a local regression from "these are my local
boundary conditions" to "this is my local field value," and on steady aerodynamics that is a
ceiling I want to break immediately. So the very first design decision is: what is the *cheapest*
mechanism that lets a point's prediction depend on the rest of the mesh? That choice is what the
first rung answers, and it should answer it with the least machinery possible, so that whatever it
leaves on the table tells me what the next rung must add.

The data type forces the shape of the answer. The input is an unstructured 3D point cloud — a
*set* of points with no canonical ordering, ~5000–10000 of them, varying per mesh, batch size one.
The thing I most want to avoid on the first rung is anything that needs the mesh graph: building
and message-passing over `geo` adds a whole machinery (neighborhoods, normalization, edge bookkeeping)
that I would rather introduce deliberately at a later rung than fold into the floor. So I ask: can I
get global coupling out of the point set *without* edges, with one symmetric reduction? The
cheapest symmetric reduction over a set is a pool — sum, mean, or max — and of those, max has the
cleanest reading. If I lift every point to a feature with a shared per-point MLP and then take the
element-wise maximum across all N points, I get a single global descriptor whose j-th coordinate is
"the strongest response of learned feature j anywhere on the mesh." That descriptor is
permutation-invariant by construction (max does not care about order) and it summarizes the whole
shape in one vector. Then if I hand that global vector back to every point — concatenate it onto the
point's own local feature and run another shared MLP — each point predicts its field value from both
its local boundary conditions *and* a global summary of the entire body. That is exactly the
minimal cross-point coupling I wanted: one global max-pool, broadcast back, no edges.

This is the per-point-MLP-then-symmetric-pool form, and the reason it is the right floor is that the
global descriptor is provably a sufficient statistic of the set for a wide class of set functions: a
shared lift followed by a symmetric max can approximate any continuous set function given enough
pooling width, and the output is pinned down by at most a handful of "critical" points whose
features win the maxima. So the architecture is not a hack — it is the canonical permutation-invariant
set encoder, and it gives me a principled floor: *what can you predict about every point's field if
the only thing you know about the rest of the body is a single max-pooled summary of it?*

Now I have to land this in the task's edit surface, and here is where I must be careful, because the
implementation the harness ships under the `pointnet` baseline is *not* the full PointNet from the
classification/segmentation literature — it is a stripped, design-task-specific variant, and the
trajectory's code must match that, not the paper. Three differences matter. First, there are **no
T-nets**: the paper's signature move is the input and feature alignment networks (the 3×3 and 64×64
learned transforms, the latter kept orthogonal by a regularizer on the loss) that canonicalize pose
before pooling. The harness drops both. That is defensible here — the design meshes already arrive in
a consistent physical frame (a car is a car, oriented the same way; the inlet direction is fixed), so
there is no pose ambiguity to canonicalize away, and the orthogonality regularizer would have to be
threaded through the *frozen* loss, which the edit surface does not let me touch. So I do not reach
for T-nets, and I do not pretend the rung has them. Second, the pooling is **global over the entire
mesh in one shot**, with the result repeated back to every point and concatenated — exactly the
segmentation-style local-plus-global concatenation, but with the single global descriptor only,
because there is one mesh per forward pass (batch is one, so the `global_max_pool` over a single
graph is just a column-wise max, and the `repeat_interleave` broadcasts it back across all N points).
Third, the channel plumbing is the design-task one and it is wide on purpose: the encoder lifts the
ten input dimensions to `n_hidden`; an `in_block` lifts to `2·n_hidden`; the `max_block` lifts to
`32·n_hidden` before the global max; the `out_block` consumes the concatenation of the local
`2·n_hidden` feature with the `32·n_hidden` global descriptor and brings it down to `4·n_hidden`; a
final linear and the decoder produce the four output channels. The shared-per-point MLPs are the
`MLP(... n_layers=0 ...)` blocks from `layers.Basic`, run on the squeezed `(N, C)` tensor — the
batch dimension is squeezed away because the loop guarantees one mesh, and a zero `batch` index
vector is what feeds `global_max_pool`.

The width choice is forced by the budget. The paper-faithful PointNet script uses a small width,
and the harness honors that: `CONFIG_OVERRIDES = {'n_hidden': 16}`. This is not a free knob —
`n_hidden=16` is what keeps PointNet a paper-faithful baseline, and it also means the `32·n_hidden`
global descriptor is 512-wide, modest. Crucially, the model *raises* `ValueError` if `geo` is None —
the harness builds an edge_index for every forward pass and PointNet here is registered as a graph
model even though it does not message-pass over the edges; it only uses the squeezed point set and
the zero batch vector for the global pool. So I keep that guard: it is part of the literal scaffold
edit, an artifact of the harness treating all of {PointNet, GraphSAGE, Graph_UNet} as graph models
that squeeze the batch dimension.

Let me reason about what this floor should and should not be able to do, because that is the
diagnosis the next rung will start from. A single global max-pool is a very coarse channel. Two
points that sit in completely different flow regimes — one on the high-pressure stagnation nose, one
deep in the low-pressure wake — receive the *identical* global descriptor; the only thing
distinguishing their predictions is their own local feature. So the model can express "this is a
nose-type point in a body whose overall summary is X" but it cannot express any *relation* between
the nose and the wake beyond what survives the bottleneck of one max vector. Drag is a surface
integral of pressure (and shear) projected on the inlet direction, so the drag *coefficient* depends
on getting the pressure field's coarse structure right — the front-to-back pressure difference. A
global descriptor can plausibly capture that coarse front/back contrast, so I expect the **rank
correlation of drag (rho_d) to be reasonable** — the model can order designs by their gross pressure
asymmetry even from a coarse summary. But the *field* errors, especially velocity, depend on local
spatial structure that one global vector cannot reconstruct: the velocity field varies sharply and
non-locally near the surface, and a point that only knows "my features + one global max" has no way
to see *where it sits relative to its neighbors*. So I expect the **relative L2 errors on pressure
and velocity to be the worst of any rung that adds real spatial structure**, and I expect the **drag
*error* (c_d) to be loose** even if the *ordering* is decent, because integrating a coarse pressure
field gives a coefficient that is roughly right in rank but off in magnitude.

That is precisely the gap I want to expose. PointNet establishes that one global max-pool already
beats pure pointwise prediction (it must — it is the only rung that adds *any* cross-point signal
over the empty default), but it also establishes that a single global bottleneck is too coarse for
the field reconstruction the metrics reward. The natural next move, when these numbers come in, is to
replace the one global summary with *local* aggregation — let each point pool from its actual mesh
neighbors via the `geo` edges, so a point's context is its physical neighborhood rather than a single
shape-wide vector. That is message passing, and it is where the next rung goes. For now the floor is
deliberately the leanest cross-point mechanism: shared per-point lift, one global max, broadcast back,
concatenate, decode. The full scaffold module is in the answer. I will watch the field errors most of
all — they are the metric that a global-only model has the least ability to drive down, and the size
of that gap is the budget the next rung gets to spend on locality.

To restate the causal chain compactly so the next rung inherits it cleanly: the empty default has no
cross-point coupling and so cannot represent the non-local structure of a PDE field, which forces me
to add the cheapest possible coupling; among edge-free options a shared per-point MLP plus a single
symmetric max-pool is the canonical permutation-invariant set encoder and provably a sufficient
summary for set functions, so I lift each point, max-pool to one global descriptor, broadcast it back
to every point, concatenate with the local feature, and decode — matching the harness's
T-net-free, single-mesh, graph-registered PointNet variant at `n_hidden=16` — and I expect this to
order designs by drag passably (decent rho_d) while leaving the field errors and the drag-magnitude
error loose, because one global bottleneck cannot carry the local spatial structure those metrics
reward, which is exactly the gap the next rung must close with local message passing.
