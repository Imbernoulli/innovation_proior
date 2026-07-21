PointNet's numbers say most of what I suspected the single global bottleneck would do, with one
honest correction I want to make before building on them. On Car the drag *ordering* came out strong:
rho_d 0.962, so the model can rank designs by their gross pressure asymmetry from one global
max-pool, exactly as the sufficient-statistic argument promised. The drag *magnitude* is loose (c_d
0.0247), the split I predicted between a rank statistic that forgives a monotone distortion and a
relative error that does not. And the field errors are what a global-only model leaves loose: Car
pressure relative-L2 0.106, velocity 0.042; AirfRANS pressure 0.039, velocity 0.039; the AirCraft
probe far off scale at 0.657/0.388, which I read as an unnormalized task-internal reference and will
not over-interpret. The correction is this: at the floor I leaned toward *velocity* being the hardest
field, and the data says the opposite on the 3D cases — Car pressure 0.106 is about 2.5× its velocity
0.042, and AirCraft pressure 0.657 likewise exceeds its velocity 0.388, with AirfRANS the only tie at
0.039/0.039. So it is *pressure* that a one-vector summary reconstructs worst. The mechanism: the
surface pressure field carries the sharp stagnation and separation extrema, and a rank-≤512 max
bottleneck captures the coarse front/back contrast (enough to rank drag) but blurs the peaks, so the
*relative* error on the peaky field is large; the velocity field, over the surface-plus-volume
sample, is relatively smoother and comes in lower. The floor's failure is still exactly what I said
structurally — no context keyed to *where* a point sits — but the metric it hurts most is pressure,
and I carry that forward rather than the velocity hunch.

Either way the diagnosis is sharp and unchanged: the floor lacks *spatial locality*. Each point needs
context from its actual mesh neighbors, not from a single shape-wide vector, and the task hands me the
tool the floor deliberately ignored — the `geo` edge_index. So the design question is which mechanism
converts those edges into per-point context, and I have three live options.

The first is flat message passing: stack graph-convolution layers at the original mesh resolution,
each letting a point aggregate from its one-hop neighbors, r layers reaching r hops. This fixes
PointNet's *local* blindness directly but not the *long-range* problem — the very thing that made me
add a global pool. To carry information from nose to wake through one-hop steps I need on the order of
the graph diameter many layers. If the mesh is roughly a surface manifold of N≈8000 points its
diameter scales like √N ≈ 90 hops; even as a volumetric cloud it is N^{1/3} ≈ 20 hops. Either count is
far more depth than I would stack, and worse, each averaging step pulls a node's feature toward its
neighbors' mean, so a deep flat stack over-smooths — features converge toward the shape-wide average
and blur the distinctions the field reconstruction needs. So flat message passing trades PointNet's
"all-global, no-local" for "all-local, no-global": field errors down, but a plateau on long-range
coupling and bleeding rho_d as depth grows. I want *both* local message passing and a cheap path for
information to cross the whole body.

The second option is to jump straight to attention over the points — global coupling in one layer —
but that is O(N²) ≈ 6.4×10⁷ pairs at N=8000, and it is the most expressive operator on the table;
spending it now leaves the ladder nothing to escalate to and no clean read on what a *structured*
multi-scale summary buys over a flat one. I want to earn attention, not open with it. That leaves the
third option, which gives both local and long-range without diameter-many layers: shrink the graph
while convolving so the deep representation sees large regions, then expand back to full resolution,
copying high-resolution features across the bottleneck so the decoder can relocalize. On images that
is the U-Net; convolution has a graph analogue (message passing), and the *shrinking and expanding*
are the missing piece, because at a coarsened resolution a single hop spans a large region of the
original mesh. So the second rung is a **graph U-Net**: a multi-scale encoder-decoder over the mesh
graph. It keeps PointNet's global reach (the coarse levels are a learned *spatial* analogue of the
global pool) while adding the local message passing PointNet lacked (a graph convolution at every
resolution).

The multi-scale arithmetic is what tells me the U-Net reaches globally cheaply. With a pooling ratio
of 0.5 held across five scales, the node counts descend 8000 → 4000 → 2000 → 1000 → 500: a 16×
reduction. At the coarsest level a single neighbor-aggregation step spans the whole body if the
neighborhood radius is large enough — which is what the radius schedule delivers. The coarse graph is
rebuilt by a radius graph in physical space with a radius that *grows* across scales,
`list_r=[0.05,0.2,0.5,1,10]`. For geometries normalized to roughly unit extent, r=0.05 connects only
a tight local cluster, r=1 already spans most of the body, and r=10 (far larger than the domain's
diameter) connects essentially everything — a nearly complete graph, so the 500-node coarsest level
is one fully-connected blob, the spatial analogue of PointNet's single global pool, reached in five
levels instead of the ~90 flat layers the diameter would otherwise demand. The count of five is set
by the arithmetic: the goal is a coarsest graph small enough that one large-radius hop is genuinely
global yet large enough to carry a few distinct regions — a few hundred nodes. 8000·0.5⁴ = 500 sits in
that band; three scales would stop at 2000 (too many for one hop to be "global"), eight scales would
drive it to ~62 (so aggressively coarsened that the random sampling would throw away all but a scrap
of the body).

Now to match what *this task's* `graphunet` baseline actually does, because it is not the
gPool/gUnpool graph U-Net of the literature — it is the design-task multi-scale geometric variant, and
three differences are load-bearing. First, the pooling is **random node sampling by a fixed ratio**,
not a learned top-k-by-projection. The standard learned variant scores every node by a normalized
projection onto a learned direction, keeps the top k, and gates survivors by their sigmoid scores so
the projection gets a gradient. This variant instead samples `pool_ratio=0.5` of the nodes *uniformly
at random* (`id_sampled = random.sample(range(n), k)`) and keeps those rows — no learned selection, no
sigmoid gate. So the coarsening is stochastic and unlearned; only the graph convolutions are learned.
That is a real capacity reduction with a concrete downside: a learned top-k would deterministically
carry the most informative nodes (the high-projection peaks) to the bottleneck, but random sampling
gives any specific point only a 0.5 chance of surviving each pooling step, and over four steps down to
the coarsest level a particular critical point reaches the bottom with probability 0.5⁴ = 1/16 ≈ 6%.
Put the other way, ~94% of any specific critical structure is gone by the bottleneck on a given pass,
and *which* structures survive differs mesh to mesh — a lot of blind loss concentrated on exactly the
sharp features the field metrics care about. I should expect this to inject variance into the
fine-field predictions, and I should not import the "learnable node selection" story.

Second — the geometric heart of the variant — after sampling, the coarse graph is **rebuilt by a
radius graph in physical space**, not by restricting a graph power of the adjacency. The standard
graph-U-Net repairs connectivity lost to dropped nodes by squaring the (self-looped) adjacency so
two-hop neighbors through a removed node stay connected. This variant calls
`nng.radius_graph(pos_x, r=list_r[n], ...)` on the *positions* of the surviving points, radius growing
across scales as above. So locality at each level is *geometric* — who is within radius r of whom in
3D — which is sensible precisely because the mesh carries real coordinates (`pos_x` is `x[:, :2]` at
the top level, then the sampled positions thereafter), and the growing radius is what turns the coarse
levels into the spatial analogue of PointNet's global pool.

Third, the unpooling is **nearest-neighbor interpolation in space**, not scatter-by-saved-index. The
standard gUnpool scatters coarse rows back to their recorded indices and leaves dropped rows zero
until a skip fills them. This variant instead assigns each fine point the feature of its *nearest*
coarse point (`cluster = nng.nearest(pos_x_up, pos_x_down); x_up = x[cluster]`) — a geometric
interpolation that fills *every* fine point, not just the survivors. That is convenient but crude:
piecewise-constant, so every fine point in a coarse point's Voronoi cell gets the *identical* upsampled
feature. At the deepest step, 500 coarse features spread over 8000 fine points means each coarse
feature is copied to ~16 neighbors on average, so the deepest information is quantized to ~500 distinct
values across the mesh before the skip refines it — a piecewise-constant reconstruction that will blur
the sharp field gradients the pressure relative-L2 punishes. It then concatenates the interpolated
feature with the saved encoder feature at that level (`torch.cat([z, z_list[n-1]], dim=1)`) and runs a
SAGE convolution; the down path *doubles* the hidden width at each scale, so at `n_hidden=16` the down
widths run 16 → 32 → 64 → 128 → 256, and the up convolutions consume `3·size_hidden_layers_init`
channels (interpolated coarse plus saved finer). The convolutions are `SAGEConv` with BatchNorm and
ReLU. One matching detail I must respect: the encoder lifts `fun_dim` *only* (`x = fx.squeeze(0)`,
encoder `MLP(args.fun_dim, ...)`), so the coordinates do *not* enter the node features — they enter
only through the radius graph and the spatial interpolation. That is the opposite of PointNet, which
concatenated the coordinates into the point feature, and it is a subtle liability: a point's own
position is available only through the geometry operations, never as a learnable channel.

The down path, concretely: start at N=8000 with the Car head; the encoder lifts `fun_dim=7` to 16
channels (`MLP(7,32,16)` = 784 params) giving `(8000, 16)`. `down_layers[0]` is `SAGEConv(16,16)` at
full resolution, still `(8000, 16)`. Then four downsample-and-convolve steps, each halving the node
count and doubling the width: to 4000 and `SAGEConv(16,32)`; to 2000 and `SAGEConv(32,64)`; to 1000
and `SAGEConv(64,128)`; to 500 and `SAGEConv(128,256)` — coarsest level 500 nodes at 256 channels. A
`SAGEConv(in,out)` is two linears, ≈`2·in·out` params, so the down convs cost
≈`2·(256+512+2048+8192+32768) ≈ 87k`, the widest step (128→256) dominating at ~65.5k; the up path
roughly mirrors it through the `3·n_hidden_init`-wide concat convolutions, and with BatchNorms the
whole model is on the order of a couple hundred thousand parameters. As with PointNet that is a few
percent of the multi-million-parameter budget anchor, so `CONFIG_OVERRIDES = {'n_hidden': 16}` is
again a baseline-fidelity choice, not budget-forced — and unlike PointNet, where the defect was a
qualitative bottleneck no width could fix, this variant's defects are the *random* pooling and the
*piecewise-constant* unpooling, which widening also cannot repair (a wider feature smeared across ~16
neighbors is still piecewise-constant). So I keep 16 and treat the coarsening operations as the thing
to address next. Like the other graph baselines it raises if `geo` is None, since it genuinely
message-passes over the edges.

Why this should beat PointNet, and where it might still fall short — the budget for the rung after it.
The multi-scale stack gives every point genuine local context (SAGE at each resolution) *and*
long-range reach (coarse levels with growing radius), so the **field errors should improve over
PointNet** in the best case, and the drag *ordering* should stay strong or improve since the coarse
levels still capture the global pressure asymmetry, possibly *better* than a lone extremum because the
hierarchy is spatial. But the two crude operations worry me. The pooling is *random*, so the coarsest
graph is 500 nodes chosen by luck, injecting variance that may *hurt* the fine field details; and the
nearest-neighbor unpooling is piecewise-constant, ~500 values over 8000 points at the deepest step,
which will blur the sharp gradients — and the field the floor reconstructed worst was the peaky
*pressure*, so this is likely to bite pressure hardest. So my falsifiable expectation splits: the
graph U-Net should **improve the drag rank correlation over PointNet's 0.962** (a spatial hierarchy is
a better global summary than one max), but the **field errors may not uniformly beat PointNet** — I
would not be shocked if the Car pressure relative-L2 *rises* above 0.106 and the velocity stays near
0.042, and I expect AirfRANS — a 2D case where the radius graph and interpolation are even blunter, and
where the floor already sat at 0.039/0.039 — to look the worst of the three, with velocity the more
exposed there because a 2D wake has long thin structures a piecewise-constant interpolation cannot
follow. If that comes back, the lesson is written: the *coarsening* is the weak link, so a method that
does *flat*, learned, per-node neighbor aggregation at full resolution — no random pooling, no
interpolation loss — should reclaim the field accuracy the lossy hierarchy gave up. That is message
passing without the U-Net, and it is where the third rung goes. The full module is in the answer.
