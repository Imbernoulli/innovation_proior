PointNet's numbers say exactly what I suspected the single global bottleneck would do, and they say
it with a clean split. On Car the drag *ordering* came out strong — rho_d 0.962, the model can rank
designs by their gross pressure asymmetry from one global max-pool — but the drag *magnitude* is loose
(c_d 0.0247) and the field errors are the worst I expect to see from anything with real spatial
structure: Car pressure relative-L2 0.106, velocity 0.042; AirfRANS pressure 0.039, velocity 0.039.
That is the global-bottleneck signature I predicted: a point on the stagnation nose and a point in the
wake receive the *identical* global descriptor, so the model reconstructs the coarse front/back
contrast (enough to rank drag) but cannot resolve the locally varying field that the relative-L2
metrics reward. The velocity error in particular — 0.042 on Car, 0.039 on AirfRANS — is the tell:
velocity varies sharply and non-locally near the surface, and a point that knows only "my features +
one shape-wide max" has no idea *where it sits relative to its neighbors*. So the diagnosis is sharp:
the floor lacks *spatial locality*. Each point needs context from its actual mesh neighbors, not from
a single shape-wide vector. That is what the next rung must add, and the harness hands me the tool —
the `geo` edge_index that PointNet ignored.

The cheapest way to use those edges is one round of message passing: each point pulls a feature from
its mesh neighbors and combines with its own. But before I just stack a flat graph convolution, I
want to ask what flat message passing would and wouldn't fix, because the answer pushes me to a
specific architecture. A flat GCN at a single resolution mixes each node with its one-hop
neighborhood; r layers reach r hops. That fixes PointNet's *local* blindness — a point now sees its
neighbors — but it does not fix the *long-range* problem PointNet had, the very thing that made me
add a global pool in the first place. To carry information from the nose of the car to the wake
through one-hop steps, I would need as many layers as the graph diameter, and repeated neighbor
averaging pushes node features toward one another (over-smoothing), blurring exactly the distinctions
the field reconstruction needs. So flat message passing trades PointNet's "all-global, no-local" for
"all-local, no-global" — I would expect it to drive the field errors down (local structure recovered)
but to plateau on the long-range coupling, and I do not want to give up the global reach I just
established. I want *both*: local message passing *and* a path for information to travel across the
whole body cheaply.

On images the architecture that delivers both is the U-Net: shrink the grid while convolving so the
deep representation sees large regions, then expand back to full resolution, copying high-resolution
features across the bottleneck so the decoder can localize. Convolution has a graph analogue
(message passing). The *shrinking and expanding* do not — and that missing piece is precisely what
gives long-range reach without diameter-many flat layers, because at a coarsened resolution a single
hop spans a large region of the original mesh. So the second rung is a **graph U-Net**: a multi-scale
encoder-decoder over the mesh graph. This is the right next step over PointNet specifically because it
keeps PointNet's global reach (the coarse levels are a learned, *spatial* analogue of the global pool)
while adding the local message passing PointNet lacked (a GCN-type layer at every resolution).

The hard part is graph pooling, because a graph has no canonical node ordering and no fixed
rectangular windows — I cannot "take every 2×2 block." I need a coarsening that is graph-structured,
adaptive, and cheap. Now I have to match what *this task's* `graphunet` baseline actually does, because
it is not the gPool/gUnpool graph U-Net of the literature — it is the design-task multi-scale
geometric variant, and the differences are load-bearing. Three of them. First, the pooling here is
**random node sampling by a fixed ratio**, not a learned top-k-by-projection. The standard learned
variant scores every node by a normalized scalar projection onto a learned direction, keeps the top k, and
gates the survivors by their sigmoid scores so the projection direction gets a gradient. This task's
implementation instead, at each of the scales, samples `pool_ratio=0.5` of the nodes *uniformly at
random* (`id_sampled = random.sample(range(n), k)`) and keeps those rows — there is no learned
selection vector, no sigmoid gate. So the coarsening is stochastic and unlearned; the only learned
parts are the graph convolutions. That is a real capacity reduction versus a learned-selection variant, and I should not
import the "learnable node selection" story.

Second — and this is the geometric heart of the variant — after sampling, the coarse graph is
**rebuilt by a radius graph in physical space**, not by restricting a graph power of the adjacency.
The standard graph-U-Net repairs connectivity lost to dropped nodes by squaring the (self-looped) adjacency so
two-hop neighbors through a removed node stay connected. This task instead calls
`nng.radius_graph(pos_x, r=list_r[n], ...)` on the *positions* of the surviving points, with the
radius growing across scales (`list_r = [0.05, 0.2, 0.5, 1, 10]`). So locality at each level is
*geometric* — who is within radius r of whom in 3D — and the growing radius is what gives long-range
reach: at the coarsest level r=10 connects essentially everything, which is the spatial analogue of
PointNet's global pool, but reached through a learned multi-scale stack rather than a single max. This
is sensible precisely because the mesh carries real coordinates (the `pos_x` is `x[:, :2]` at the top
level, then the sampled positions thereafter); a radius graph is the natural neighborhood on a point
cloud with metric structure.

Third, the unpooling is **nearest-neighbor interpolation in space**, not scatter-by-saved-index. The
standard gUnpool scatters the coarse rows back to their recorded indices and leaves dropped rows zero
until a skip fills them. This task instead, on the way up, assigns each fine point the feature of its
*nearest* coarse point (`cluster = nng.nearest(pos_x_up, pos_x_down); x_up = x[cluster]`) — a
geometric interpolation that fills *every* fine point, not just the survivors. Then it concatenates
the interpolated feature with the saved encoder feature at that level and runs a SAGE convolution.
The skip is **concatenation** (`torch.cat([z, z_list[n-1]], dim=1)`), and the down path *doubles* the
hidden width at each scale (`out_channels = 2 * size_hidden_layers`), so the up convolutions consume
`3 * size_hidden_layers_init` channels — the interpolated coarse feature plus the saved finer feature.
The convolutions are `SAGEConv` with `BatchNorm` and ReLU; the encoder lifts `fun_dim` (note: only
the features, not the coordinates — `x = fx.squeeze(0)` and the encoder is `MLP(args.fun_dim, ...)`,
with the coordinates entering only through the radius graph and the spatial interpolation, *not*
concatenated into the node features), and the decoder maps back to `out_dim`.

Let me make sure I see why this should beat PointNet and where it might still fall short, because that
is the budget for the rung after it. The multi-scale stack gives every point genuine local context
(SAGE at each resolution) *and* long-range reach (the coarse levels with growing radius), so I expect
the **field errors to improve over PointNet** — the model can now resolve local structure that one
global vector could not. The drag *ordering* should stay strong or improve, since the coarse levels
still capture the global pressure asymmetry, possibly better than a single max because the hierarchy
is spatial. But two things in *this* variant worry me. The pooling is *random*, so the coarsening
throws away nodes blindly — at `n_hidden=16` and 0.5 ratios over five scales, the coarsest graph is
tiny and which points survive is luck, which will inject variance and may *hurt* the field
reconstruction on the fine details, especially velocity. And the nearest-neighbor unpooling is a
piecewise-constant interpolation — every fine point in a coarse cell gets the *same* upsampled
feature — which is geometrically crude and will blur sharp field gradients. So my falsifiable
expectation: graph U-Net should **improve the drag rank correlation over PointNet** (the spatial
hierarchy is a better global summary than one max — I'd expect rho_d to climb past 0.962, plausibly
above 0.98), but the **field errors and the drag-magnitude error may not uniformly beat PointNet**,
because random pooling plus piecewise-constant unpooling are coarse, lossy operations. Concretely I
would not be shocked if the Car pressure relative-L2 *rises* relative to PointNet's 0.106 and the
velocity error stays in the same ballpark, with AirfRANS — a 2D case where the radius graph and
interpolation are even blunter — looking worse on the fields. If that is what comes back, the lesson
for the next rung is written: the *coarsening* is the weak link, so a method that does *flat*,
learned, per-node neighbor aggregation at full resolution — no random pooling, no interpolation loss —
should reclaim the field accuracy that the lossy hierarchy gave up. That is message passing without
the U-Net, and it is where the third rung goes.

The causal chain, threaded from PointNet's result: PointNet's strong drag rank but loose drag
magnitude and worst-in-class field errors diagnose a *locality* failure — one global max-pool gives
every point the same shape-wide context, so local field structure cannot be resolved → the fix is to
use the `geo` edges for neighbor aggregation, but flat message passing alone would trade PointNet's
"global-only" for "local-only" and over-smooth → so I want both local and long-range, which is the
U-Net pattern, realized on the mesh as a multi-scale graph encoder-decoder → matching *this task's*
graph U-Net variant: **random** node sampling at ratio 0.5 per scale (not learned top-k), coarse
graphs **rebuilt by a growing-radius graph in physical space** (not adjacency powers), **nearest-
neighbor spatial unpooling** with **concatenation** skips, width-doubling SAGE convolutions, and the
encoder consuming only `fun_dim` features with coordinates entering through the geometry — at
`n_hidden=16` — expecting the spatial hierarchy to raise rho_d over PointNet while the random,
piecewise-constant coarsening leaves the field errors lossy enough that flat learned message passing
will be the better field model at the next rung. The full scaffold module is in the answer.
