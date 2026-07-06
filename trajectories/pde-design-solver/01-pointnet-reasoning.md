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
cheapest symmetric reduction over a set is a pool — sum, mean, or max — and before I commit to one I
want to actually reason about the three, because the choice is not cosmetic here. Take a mesh with
N≈8000 points and imagine one channel that has learned to fire on stagnation-type geometry, so it is
large at a handful of nose points and near zero everywhere else. Under a *sum* pool the descriptor
scales with N, and N ranges from ~5000 to ~10000 across meshes — a factor of two — so the same shape
sampled at two resolutions would hand the decoder a global vector whose magnitude differs by ~2×,
with no normalization anywhere in the frozen loop to undo it; that is a spurious signal the network
must waste capacity fighting. Under a *mean* pool the scale is stable, but the handful of large nose
responses get divided by N: a stagnation peak that should dominate is attenuated by ~1/8000 and
drowned by the ~7999 near-zero points, so exactly the extremal structure that carries the physics is
averaged away. Under a *max* pool the scale is stable *and* the extremal response survives intact —
the j-th coordinate of the descriptor becomes "the strongest response of learned feature j anywhere
on the mesh," which for aerodynamic fields (stagnation, separation, the pressure peak) is precisely
the quantity I want preserved. So max is not an aesthetic pick; sum fails the variable-N stability
test and mean fails the extremum-preservation test, and max is the reduction that passes both.

That fixes the reduction; the rest of the floor follows. If I lift every point to a feature with a
shared per-point MLP and then take the element-wise maximum across all N points, I get a single
global descriptor that is permutation-invariant by construction — reorder the points and the
column-wise max is unchanged, since max over a set does not see order. Let me actually check that on
a two-point, two-channel toy so I am not hand-waving: features `[[3,1],[2,5]]` max to `[3,5]`, and
the reordering `[[2,5],[3,1]]` maxes to `[3,5]` as well — identical, as required, and the winning
index differs per channel (point 0 wins channel 0, point 1 wins channel 1). Then if I hand that
global vector back to every point — concatenate it onto the point's own local feature and run another
shared MLP — each point predicts its field value from both its local boundary conditions *and* a
global summary of the entire body. That is exactly the minimal cross-point coupling I wanted: one
global max-pool, broadcast back, no edges.

Before I settle, I should confirm the two edge-using alternatives really are more machinery than the
floor deserves, rather than dismissing them by reflex. One round of message passing over `geo` would
give each point its neighbors' features — genuinely local context — but it commits me to the whole
neighborhood/normalization/edge apparatus I wanted to introduce on purpose later, and it gives
*local* coupling when the empty default's most glaring failure is that it has *no* coupling at all,
local or global; the floor's job is to establish that any cross-point signal helps, and the widest,
cheapest such signal is the global one. A single attention layer over the points would give global
coupling directly, but at O(N²): with N≈8000 that is ~6.4×10⁷ point-pairs per layer, and even setting
aside cost, using the most expressive operator as the *floor* would leave me nothing to escalate to
and no clean diagnosis of what locality buys. So the max-pool set encoder is the right floor: it is
the leanest thing that couples every point to every other, and it leaves both "local structure" and
"cheap global correlation" as explicit, unspent budget for later rungs.

This per-point-MLP-then-symmetric-pool form is also principled rather than a hack: the global
descriptor is a sufficient statistic of the set for a wide class of set functions — a shared lift
followed by a symmetric max can approximate any continuous set function given enough pooling width,
and the output is pinned down by at most a handful of "critical" points whose features win the
maxima. That last fact has a concrete, and sobering, consequence I want to keep in view. With a
pooling width of 32·n_hidden = 512 channels, the entire global descriptor is determined by at most
512 argmax-winning points — one per channel, and with ties fewer — out of the ~8000 in the mesh. So
roughly 512 points, at most, write the shape-wide summary, and the other ~7500 contribute *nothing*
to it. The floor is therefore a genuinely narrow channel: a 512-dimensional bottleneck sourced from a
few hundred critical points, broadcast identically to all 8000. That is the canonical
permutation-invariant set encoder, and it gives me a principled floor with a sharp question attached:
*what can you predict about every point's field if the only thing you know about the rest of the body
is a single max-pooled summary of it, written by a few hundred of its points?*

Stated as an information-flow change, the rung's contribution is exactly one thing and I can quantify
it. In the empty default, point i's prediction is a function of the ten numbers at point i and nothing
else — its receptive field is a single point. The floor makes it a function of those ten numbers *plus*
a 512-vector that is itself a function of all N points, so the receptive field jumps from 1 to N in a
single step. That is the entire gain, and its entire limitation is written in the same sentence: the
new N-wide reach passes through a rank-≤512 max bottleneck that discards ordering and position, so it
is reach without resolution. The next rung's job is to add resolution to that reach, and the only way
to add resolution keyed to *where* a point sits is neighbor structure.

There is a tempting escape I should close off before I go further, because if it worked it would
change the whole plan: could I simply *widen* the pooling channel until the bottleneck stops hurting?
At `n_hidden=16` the descriptor is 512-wide; nothing stops me making it 5120 or 51200. But this does
not touch the actual failure. The descriptor, however wide, is computed once per mesh and broadcast
*identically* to all N points, so consider two points a centimetre apart on the surface — one just
upstream of a separation line, one just downstream, in genuinely different flow states. They receive
the *same* global vector, of any width, and the only thing left to tell them apart is their own local
feature, which the pointwise decoder already had. Widening adds resolution to the *shape summary*, not
a *relation between locations*, and the metrics that punish this rung reward exactly the second thing.
So the ceiling is structural, not a matter of descriptor width, and the fix has to be a mechanism that
gives a point context *keyed to where it is* — its neighbors — rather than a richer copy of the same
global vector. That is the argument that makes locality, not width, the lever the next rung must pull,
and it is why I keep the descriptor at the canonical modest 512 rather than spending budget widening a
channel whose defect is qualitative.

Now I have to land this in the task's edit surface, and here is where I must be careful, because the
implementation the harness ships under the `pointnet` baseline is *not* the full PointNet from the
classification/segmentation literature — it is a stripped, design-task-specific variant, and the
trajectory's code must match that, not the full version. Three differences matter. First, there are
**no T-nets**: the full version's signature move is the input and feature alignment networks (the 3×3
and 64×64 learned transforms, the latter kept orthogonal by a regularizer on the loss) that
canonicalize pose before pooling. The harness drops both. That is defensible here — the design meshes
already arrive in a consistent physical frame (a car is a car, oriented the same way; the inlet
direction is fixed), so there is no pose ambiguity to canonicalize away, and the orthogonality
regularizer would have to be threaded through the *frozen* loss, which the edit surface does not let
me touch. So I do not reach for T-nets, and I do not pretend the rung has them. Second, the pooling
is **global over the entire mesh in one shot**, with the result repeated back to every point and
concatenated — exactly the segmentation-style local-plus-global concatenation, but with the single
global descriptor only, because there is one mesh per forward pass (batch is one, so the
`global_max_pool` over a single graph is just a column-wise max, and the `repeat_interleave`
broadcasts it back across all N points). Third, the channel plumbing is the design-task one and it is
wide on purpose: the encoder lifts the ten input dimensions to `n_hidden`; an `in_block` lifts to
`2·n_hidden`; the `max_block` lifts to `32·n_hidden` before the global max; the `out_block` consumes
the concatenation of the local `2·n_hidden` feature with the `32·n_hidden` global descriptor and
brings it down to `4·n_hidden`; a final linear and the decoder produce the four output channels. The
shared-per-point MLPs are the `MLP(... n_layers=0 ...)` blocks from `layers.Basic`, run on the
squeezed `(N, C)` tensor — the batch dimension is squeezed away because the loop guarantees one mesh,
and a zero `batch` index vector is what feeds `global_max_pool`.

Let me trace the tensor shapes through the whole forward once, with N=8000 and the Car head
(space_dim=3, out_dim=4), because a dimension check is the cheapest way to catch a plumbing error
before it becomes a training run. The concatenated input is `(1, 8000, 10)`, squeezed to
`(8000, 10)`. The encoder maps it to `(8000, 16)`; `in_block` to `(8000, 32)`; `max_block` to
`(8000, 512)`. `global_max_pool` with the all-zero batch vector collapses the point axis to
`(1, 512)`. I then read `nb_points = 8000` from the batch vector and `repeat_interleave` the
descriptor back to `(8000, 512)`. Concatenating with the carried `(8000, 32)` local feature gives
`(8000, 544)` — and 544 = (2+32)·16, which matches the `out_block` input width exactly, so the concat
is wired right. `out_block` brings it to `(8000, 64)`, `fcfinal` to `(8000, 16)`, the decoder to
`(8000, 4)`, and the final `unsqueeze(0)` restores `(1, 8000, 4)` — the contract's `(1, N, out_dim)`.
For AirfRANS the only change is space_dim=2, so the input concat is `(N, 9)` and the encoder's first
linear absorbs it; for AirCraft out_dim=6, so the decoder's last linear widens to six. The
shapes close, which is the only "verification" available before I have numbers.

That trace also surfaces a structural bias worth naming, because it sharpens what I expect to fail.
The `out_block` sees 544 channels, of which 32 are the point's own carried-forward local feature and
512 are the broadcast global descriptor — so 94% of the context feeding every point's decode is the
*same vector shared by all points*, and only 6% is that point's own signal. Architecturally the floor
leans hard on the shape summary and gives the point's individuality a thin channel. That is a
deliberate consequence of the design-task widths (the `max_block`'s 32× lift is what makes the global
descriptor sixteen times wider than the local feature), and it is coherent with what the floor is
*for* — establishing how far a shape-wide summary alone carries — but it is also a reason to expect the
locally structured fields, velocity most of all, to come out poorly: the model is wired to prefer the
global story over the local one.

The width choice deserves an honest accounting, because it would be easy to say "forced by budget"
when it is really not. `budget_check.py` rejects anything over 1.05× the wide attention model the task
uses as its anchor, and that anchor is a many-layer 256-wide network — millions of parameters. Let me
count PointNet at `n_hidden=16` to see how much of that cap it actually uses. Each `MLP(a,b,c,
n_layers=0)` is two linears, `a·b+b` then `b·c+c`. The encoder `MLP(10,32,16)` is 352+528=880; the
`in_block MLP(16,32,32)` is 544+1056=1600; the `max_block MLP(32,128,512)` is 4224+66048=70272; the
`out_block MLP(544,256,64)` is 139520+16448=155968; `fcfinal Linear(64,16)` is 1040; the decoder
`MLP(16,32,4)` is 544+132=676. Summed, ≈230k parameters. Against a multi-million-parameter budget
anchor that is a few percent of the cap — the budget is nowhere near binding here. So `n_hidden=16` is
not a budget-forced knob; it is the *canonical PointNet width* that keeps this rung a faithful
baseline, and it happens to leave the `32·n_hidden` global descriptor at a modest 512. I set
`CONFIG_OVERRIDES = {'n_hidden': 16}` for baseline fidelity, not because the budget made me. One more
literal-scaffold detail I must keep: the model *raises* `ValueError` if `geo` is None — the harness
builds an edge_index for every forward pass and registers PointNet as a graph model even though it
never message-passes over the edges; it only uses the squeezed point set and the zero batch vector for
the global pool. That guard is an artifact of the harness treating all of {PointNet, GraphSAGE,
Graph_UNet} as graph models that squeeze the batch dimension, and I keep it verbatim.

Let me reason about what this floor should and should not be able to do, because that is the
diagnosis the next rung will start from. A single global max-pool is a very coarse channel. Two
points that sit in completely different flow regimes — one on the high-pressure stagnation nose, one
deep in the low-pressure wake — receive the *identical* global descriptor; the only thing
distinguishing their predictions is their own local feature. So the model can express "this is a
nose-type point in a body whose overall summary is X" but it cannot express any *relation* between
the nose and the wake beyond what survives the bottleneck of one max vector written by a few hundred
critical points. Drag is a surface integral of pressure (and shear) projected on the inlet direction,
so the drag *coefficient* is c_d ∝ ∮ (p·n − τ)·d̂ over the surface, dominated by the pressure term
projected on the inlet direction d̂ — which is to first order the integrated front-to-back pressure
difference. Two things follow from that integral. rho_d is a Spearman *rank* correlation across
designs, so it survives any monotone distortion of the predicted field that is consistent from design
to design: if the coarse summary systematically compresses or shifts the pressure field but preserves
the *ordering* of the front-to-back contrast across bodies, the ranking of the integral is preserved
even though its value is wrong. c_d, by contrast, is a *relative error of the value*, so the same
monotone distortion that rho_d forgives shows up directly as a loose coefficient. A global descriptor
can plausibly capture the coarse front/back contrast well enough to order designs, so I expect the
**rank correlation of drag (rho_d) to be reasonable** — ranking needs only the monotone signal, not
the calibrated magnitude — while the coefficient itself stays loose. But the *field* errors, especially velocity, depend on
local spatial structure that one global vector cannot reconstruct: the velocity field varies sharply
and non-locally near the surface, and a point that only knows "my features + one global max" has no
way to see *where it sits relative to its neighbors*. So I expect the **relative L2 errors on pressure
and velocity to be the worst of any rung that adds real spatial structure**, and I expect the **drag
*error* (c_d) to be loose** even if the *ordering* is decent, because integrating a coarse pressure
field gives a coefficient that is roughly right in rank but off in magnitude — a rank statistic
forgives a monotone distortion that an absolute error does not. If I had to falsify myself, the
signature to look for is precisely that split: a decent rho_d sitting *above* a loose c_d and
field-error pair, with velocity error at least as bad as pressure error, since velocity is the more
locally structured of the two.

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
symmetric max-pool is the canonical permutation-invariant set encoder, and I chose max over sum (which
drifts with the variable N≈5000–10000) and over mean (which dilutes the critical stagnation/separation
peaks by ~1/N) because it is the only reduction that is both scale-stable and extremum-preserving;
its descriptor is a provably sufficient summary for set functions but is pinned by at most 512 of
~8000 points and broadcast identically to all, so I lift each point, max-pool to one global
descriptor, broadcast it back, concatenate with the local feature, and decode — matching the
harness's T-net-free, single-mesh, graph-registered PointNet variant at `n_hidden=16` (≈230k params,
a few percent of the budget anchor, so width is a fidelity choice not a budget one) — and I expect
this to order designs by drag passably (decent rho_d) while leaving the field errors and the
drag-magnitude error loose, because one global bottleneck cannot carry the local spatial structure
those metrics reward, which is exactly the gap the next rung must close with local message passing.
