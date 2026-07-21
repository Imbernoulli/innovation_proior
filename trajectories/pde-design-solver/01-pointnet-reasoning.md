I have to start somewhere, and the scaffold default is not a real start: with the operator slot
empty, every point is mapped from its own seven features and three coordinates straight to four
output channels, with no point ever seeing any other point. A PDE field is the opposite of
pointwise — the pressure at the rear of a car is set by the geometry at its nose, the velocity in
the wake by the whole upstream body. A model with zero cross-point interaction cannot represent that
coupling at all; it can only learn a local regression from local boundary conditions to local field
value, and on steady aerodynamics that is a ceiling I want to break immediately. So the first design
decision is: what is the *cheapest* mechanism that lets a point's prediction depend on the rest of
the mesh? I want to answer it with the least machinery possible, so that whatever it leaves on the
table tells me what to add next.

The data type forces the shape of the answer. The input is an unstructured 3D point cloud — a *set*
of points with no canonical ordering, ~5000–10000 of them, varying per mesh, batch size one. The
thing I most want to avoid first is anything that needs the mesh graph: building and message-passing
over `geo` adds a whole apparatus (neighborhoods, normalization, edge bookkeeping) that I would
rather introduce deliberately later than fold into the floor. So can I get global coupling out of
the point set *without* edges, with one symmetric reduction? The cheapest symmetric reduction over a
set is a pool — sum, mean, or max — and the choice among the three is not cosmetic. Take a mesh with
N≈8000 points and one channel that fires on stagnation-type geometry, large at a handful of nose
points and near zero everywhere else. Under a *sum* pool the descriptor scales with N, and N ranges
~5000–10000 across meshes, so the same shape sampled at two resolutions hands the decoder a global
vector whose magnitude differs by ~2×, with no normalization anywhere in the frozen loop to undo it
— a spurious signal capacity must fight. Under a *mean* pool the scale is stable, but the handful of
large nose responses get divided by N: a stagnation peak that should dominate is attenuated by
~1/8000 and drowned by the ~7999 near-zero points, so exactly the extremal structure that carries
the physics is averaged away. Under a *max* pool the scale is stable *and* the extremal response
survives intact — the j-th coordinate of the descriptor becomes the strongest response of learned
feature j anywhere on the mesh, which for aerodynamic fields (stagnation, separation, the pressure
peak) is precisely the quantity I want preserved. So max is not aesthetic: sum fails the variable-N
stability test and mean fails the extremum-preservation test, and max passes both.

That fixes the reduction; the rest of the floor follows. Lift every point to a feature with a shared
per-point MLP, take the element-wise maximum across all N points to get a single global descriptor
that is permutation-invariant by construction (max over a set does not see order), then hand that
vector back to every point — concatenate it onto the point's own local feature and run another shared
MLP. Each point predicts its field value from both its local boundary conditions *and* a global
summary of the entire body. That is the minimal cross-point coupling I wanted: one global max-pool,
broadcast back, no edges.

The two edge-using alternatives really are more machinery than the floor deserves. One round of
message passing over `geo` would give each point its neighbors' features — genuinely local context —
but it commits me to the whole neighborhood/normalization/edge apparatus I want to introduce on
purpose later, and it gives *local* coupling when the empty default's most glaring failure is that
it has *no* coupling at all; the floor's job is to establish that any cross-point signal helps, and
the widest, cheapest such signal is the global one. A single attention layer over the points would
give global coupling directly, but at O(N²): with N≈8000 that is ~6.4×10⁷ point-pairs per layer, and
using the most expressive operator as the *floor* would leave nothing to escalate to and no clean
diagnosis of what locality buys. So the max-pool set encoder is the right floor: the leanest thing
that couples every point to every other, leaving both local structure and cheap global correlation
as explicit unspent budget.

This per-point-MLP-then-symmetric-pool form is principled rather than a hack: the global descriptor
is a sufficient statistic of the set for a wide class of set functions — a shared lift followed by a
symmetric max can approximate any continuous set function given enough pooling width — and the output
is pinned down by at most a handful of critical points whose features win the maxima. That last fact
has a concrete, sobering consequence. With a pooling width of 32·n_hidden = 512 channels, the entire
global descriptor is determined by at most 512 argmax-winning points — one per channel, fewer with
ties — out of the ~8000 in the mesh. So at most ~512 points write the shape-wide summary and the
other ~7500 contribute *nothing* to it. The floor is a genuinely narrow channel: a 512-dimensional
bottleneck sourced from a few hundred critical points, broadcast identically to all 8000. The sharp
question attached is: *what can you predict about every point's field if the only thing you know
about the rest of the body is a single max-pooled summary written by a few hundred of its points?*

Stated as an information-flow change, the contribution is exactly one thing. In the empty default,
point i's prediction is a function of the ten numbers at point i and nothing else — receptive field
one point. The floor makes it a function of those ten numbers *plus* a 512-vector that is itself a
function of all N points, so the receptive field jumps from 1 to N in a single step. That is the
gain, and its limitation is in the same sentence: the new N-wide reach passes through a rank-≤512 max
bottleneck that discards ordering and position, so it is reach without resolution. Adding resolution
keyed to *where* a point sits is what neighbor structure would buy.

Could I sidestep that by simply *widening* the pooling channel until the bottleneck stops hurting?
Nothing stops me making the descriptor 5120 or 51200 wide instead of 512. But it does not touch the
failure. The descriptor, however wide, is computed once per mesh and broadcast *identically* to all
N points, so two points a centimetre apart on the surface — one just upstream of a separation line,
one just downstream, in genuinely different flow states — receive the *same* global vector, of any
width, and the only thing left to tell them apart is their own local feature, which the pointwise
decoder already had. Widening adds resolution to the *shape summary*, not a *relation between
locations*, and the metrics punish exactly the second thing. So the ceiling is structural, not a
matter of descriptor width, and the fix has to give a point context *keyed to where it is* — its
neighbors. That is why locality, not width, is the lever, and why I keep the descriptor at the
canonical modest 512.

Now to land this in the edit surface. The `pointnet` baseline the task ships is *not* the full
PointNet from the classification/segmentation literature — it is a stripped, design-task variant, and
my code must match that. Three differences matter. First, there are **no T-nets**: the full
version's signature move is the input and feature alignment networks (the 3×3 and 64×64 learned
transforms, the latter kept orthogonal by a loss regularizer) that canonicalize pose before pooling.
This variant drops both — defensible here, since the design meshes arrive in a consistent physical
frame (a car is oriented the same way; the inlet direction is fixed), so there is no pose ambiguity
to canonicalize, and the orthogonality regularizer would have to thread through the *frozen* loss
anyway. Second, the pooling is **global over the entire mesh in one shot**, repeated back to every
point and concatenated — the segmentation-style local-plus-global concatenation, with the single
global descriptor only, because there is one mesh per forward pass (batch is one, so `global_max_pool`
over a single graph is just a column-wise max and `repeat_interleave` broadcasts it back). Third, the
channel plumbing is wide on purpose: the encoder lifts ten input dims to `n_hidden`; an `in_block` to
`2·n_hidden`; the `max_block` to `32·n_hidden` before the global max; the `out_block` consumes the
concatenation of the local `2·n_hidden` feature with the `32·n_hidden` global descriptor and brings
it to `4·n_hidden`; a final linear and the decoder produce the four output channels. The concat width
is (2+32)·n_hidden, which is what the `out_block` input expects — the one place a plumbing slip would
show. The shared MLPs are the `MLP(... n_layers=0 ...)` blocks run on the squeezed `(N, C)` tensor,
with a zero `batch` index vector feeding `global_max_pool`.

That plumbing surfaces a structural bias worth naming, because it sharpens what I expect to fail. The
`out_block` sees (2+32)·n_hidden channels, of which the local feature is 2 parts and the broadcast
global descriptor is 32 — so ~94% of the context feeding every point's decode is the *same vector
shared by all points*, and only ~6% is that point's own signal. The floor leans hard on the shape
summary and gives the point's individuality a thin channel. That is a deliberate consequence of the
widths (the `max_block`'s 32× lift makes the global descriptor sixteen times wider than the local
feature), coherent with what the floor is *for*, but a reason to expect the locally structured fields
— velocity most of all — to come out poorly: the model is wired to prefer the global story.

The width choice needs an honest accounting rather than a reflexive "forced by budget."
`budget_check.py` rejects anything over 1.05× the wide attention anchor — a many-layer 256-wide
network, millions of parameters. PointNet at `n_hidden=16`: each `MLP(a,b,c, n_layers=0)` is two
linears, `a·b+b` then `b·c+c`. Encoder `MLP(10,32,16)` = 880; `in_block MLP(16,32,32)` = 1600;
`max_block MLP(32,128,512)` = 70272; `out_block MLP(544,256,64)` = 155968; `fcfinal Linear(64,16)` =
1040; decoder `MLP(16,32,4)` = 676. Summed, ≈230k parameters — a few percent of a multi-million-
parameter cap, nowhere near binding. So `n_hidden=16` is not budget-forced; it is the *canonical
PointNet width* that keeps this a faithful baseline, and it happens to leave the `32·n_hidden` global
descriptor at a modest 512. I set `CONFIG_OVERRIDES = {'n_hidden': 16}` for fidelity, not budget. One
literal detail I keep verbatim: the model *raises* `ValueError` if `geo` is None — the task registers
PointNet as a graph model and builds an edge_index every forward pass even though it never
message-passes over the edges, only using the squeezed point set and the zero batch vector for the
global pool.

Now what this floor should and should not be able to do — the diagnosis the next step starts from. Two
points in completely different flow regimes — one on the high-pressure stagnation nose, one deep in
the low-pressure wake — receive the *identical* global descriptor; only their local features
distinguish their predictions. So the model can express "this is a nose-type point in a body whose
overall summary is X" but not any *relation* between nose and wake beyond what survives one max
vector. Drag is a surface integral of pressure (and shear) projected on the inlet direction,
c_d ∝ ∮ (p·n − τ)·d̂, dominated by the pressure term projected on d̂ — to first order the integrated
front-to-back pressure difference. Two things follow. rho_d is a Spearman *rank* correlation across
designs, so it survives any monotone distortion of the predicted field that is consistent from
design to design: if the coarse summary compresses or shifts the pressure field but preserves the
*ordering* of the front-to-back contrast across bodies, the ranking of the integral survives even
though its value is wrong. c_d is a *relative error of the value*, so the same monotone distortion
rho_d forgives shows up directly as a loose coefficient. A global descriptor can plausibly capture
the coarse front/back contrast well enough to order designs, so I expect a **reasonable rho_d** with
a **loose c_d**. The *field* errors, though, depend on local spatial structure one global vector
cannot reconstruct, so I expect the **relative L2 errors on pressure and velocity to be the worst of
any rung that adds real spatial structure** — and since the velocity field varies sharply and
non-locally near the surface, I expect **velocity error at least as bad as pressure error**. The
self-falsifying signature to look for is precisely that split: a decent rho_d sitting above a loose
c_d and field-error pair, with velocity no better than pressure.

So PointNet establishes that one global max-pool already beats pure pointwise prediction — it must,
being the only cross-point signal over the empty default — while exposing that a single global
bottleneck is too coarse for the field reconstruction the metrics reward. The natural next move is to
replace the one global summary with *local* aggregation: let each point pool from its actual mesh
neighbors via the `geo` edges, so a point's context is its physical neighborhood rather than a single
shape-wide vector. That is message passing. For now the floor is the leanest cross-point mechanism:
shared per-point lift, one global max, broadcast back, concatenate, decode — the full module is in the
answer. I will watch the field errors most, since they are what a global-only model has the least
ability to drive down, and the size of that gap is the budget the next step gets to spend on locality.
