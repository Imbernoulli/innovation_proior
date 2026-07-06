PointNet's numbers say most of what I suspected the single global bottleneck would do, and they say
it with a clean split — with one honest correction I want to make before I build on them. On Car the
drag *ordering* came out strong: rho_d 0.962, so the model can rank designs by their gross pressure
asymmetry from one global max-pool, exactly as the sufficient-statistic argument promised. The drag
*magnitude* is loose (c_d 0.0247), the split I predicted between a rank statistic that forgives a
monotone distortion and a relative error that does not. And the field errors are what a global-only
model leaves loose: Car pressure relative-L2 0.106, velocity 0.042; AirfRANS pressure 0.039, velocity
0.039; the AirCraft probe far off scale at 0.657/0.388, which I read as an unnormalized task-internal
reference and will not over-interpret. The correction is this: at the floor I leaned toward *velocity*
being the hardest field, and the data says the opposite on the 3D cases — Car pressure 0.106 is about
2.5× its velocity 0.042, and AirCraft pressure 0.657 likewise exceeds its velocity 0.388, with
AirfRANS the only tie at 0.039/0.039. So it is *pressure* that a one-vector summary reconstructs
worst, not velocity. Reading that mechanistically rather than as a surprise: the surface pressure
field carries the sharp stagnation and separation extrema, and a rank-≤512 max bottleneck captures the
coarse front/back contrast (enough to rank drag) but blurs the peaks, so the *relative* error on the
peaky field is large; the velocity field, over the surface-plus-volume sample, is relatively smoother
and so its relative-L2 comes in lower. The floor's failure is still exactly what I said structurally —
no context keyed to *where* a point sits — but the metric it hurts most is pressure, and I should
carry that forward rather than the velocity hunch.

Either way the diagnosis is sharp and unchanged: the floor lacks *spatial locality*. Each point needs
context from its actual mesh neighbors, not from a single shape-wide vector, and the harness hands me
the tool the floor deliberately ignored — the `geo` edge_index. So the design question for this rung is
which mechanism converts those edges into per-point context, and I have three live options I should
actually walk rather than jump at the first.

The first is flat message passing: stack graph-convolution layers at the original mesh resolution, each
letting a point aggregate from its one-hop neighbors, r layers reaching r hops. This fixes PointNet's
*local* blindness directly — a point finally sees its neighbors — but it does not fix the *long-range*
problem, the very thing that made me add a global pool in the first place, and I can put numbers to why.
To carry information from the nose of the car to the wake through one-hop steps I need on the order of
the graph diameter many layers. If the mesh is roughly a surface manifold of N≈8000 points its diameter
scales like √N ≈ 90 hops; even treating it as a volumetric cloud it is N^{1/3} ≈ 20 hops. Either count
is far more depth than I would stack, and worse, each averaging step pulls a node's feature toward its
neighbors' mean, so a deep flat stack over-smooths — features converge toward the shape-wide average and
blur exactly the distinctions the field reconstruction needs. So flat message passing trades PointNet's
"all-global, no-local" for "all-local, no-global": I would expect it to drive the field errors down but
to plateau on long-range coupling and to bleed rho_d as depth grows. I do not want to give the global
reach back — I want *both* local message passing and a cheap path for information to cross the whole body.

The second option is to jump straight to attention over the points, which would give global coupling in
one layer — but that is O(N²) ≈ 6.4×10⁷ pairs at N=8000, and more to the point it is the most expressive
operator on the table; spending it now would leave the ladder nothing to escalate to and no clean read
on what a *structured* multi-scale summary buys over a flat one. I want to earn attention, not open with
it. That leaves the third option, and it is the one that gives both local and long-range without
diameter-many layers: shrink the graph while convolving so the deep representation sees large regions,
then expand back to full resolution, copying high-resolution features across the bottleneck so the
decoder can relocalize. On images that is the U-Net; convolution has a graph analogue (message passing),
and the *shrinking and expanding* are precisely the missing piece, because at a coarsened resolution a
single hop spans a large region of the original mesh. So the second rung is a **graph U-Net**: a
multi-scale encoder-decoder over the mesh graph. It is the right next step over PointNet specifically
because it keeps PointNet's global reach (the coarse levels are a learned, *spatial* analogue of the
global pool) while adding the local message passing PointNet lacked (a graph convolution at every
resolution).

Let me make the multi-scale arithmetic concrete, because it is what tells me the U-Net reaches globally
cheaply. With a pooling ratio of 0.5 held across five scales, the node counts descend 8000 → 4000 →
2000 → 1000 → 500: a 16× reduction, so the coarsest graph is tiny. At that coarsest level a single
neighbor-aggregation step already spans the whole body if the neighborhood radius is large enough — and
that is exactly what the radius schedule delivers. The coarse graph is rebuilt by a radius graph in
physical space with a radius that *grows* across scales, `list_r=[0.05,0.2,0.5,1,10]`. For geometries
normalized to roughly unit extent, r=0.05 connects only a tight local cluster, r=1 already spans most of
the body, and r=10 (far larger than the domain's diameter) connects essentially everything, so the
coarsest level is a spatial global summary — reached in five levels instead of the ~90 flat layers the
diameter would otherwise demand. That is the whole point: the growing radius plus the 16× coarsening buy
global reach for a constant, small number of levels.

It is worth pinning down why *five* scales at ratio 0.5, because the count is not arbitrary and the
arithmetic sets it. Each 0.5 pooling step halves the node count, so after L−1 steps the coarsest graph
holds N·0.5^{L−1} nodes; the design goal is for that coarsest graph to be small enough that one
large-radius hop is genuinely a global summary yet large enough to still carry a few distinct regions —
a few hundred nodes, not a handful. With N≈8000, five scales give 8000·0.5⁴ = 500, which sits in exactly
that band. Three scales would stop at 2000 nodes, still far too many for one radius hop to be "global"
and leaving the long-range reach half-built; eight scales would drive the coarsest graph to 8000·0.5⁷ ≈
62 nodes, so aggressively coarsened that the random sampling — which already keeps a specific critical
point with only 0.5^{L−1} probability — would throw away all but a scrap of the body and the bottleneck
would be nearly contentless. Five is the count that reaches ~500 (a genuine global summary that still has
regional structure) in as few levels as possible, and I can sanity-check the radius schedule against it:
r grows 0.05 → 0.2 → 0.5 → 1 → 10, each level's radius chosen so that a hop at that resolution reaches
across the *current* coarsened spacing, and the final r=10 on a unit-extent geometry is a limit case —
larger than any pairwise distance in the body, so `radius_graph` returns a nearly complete graph and the
500-node coarsest level is one fully-connected blob, the spatial analogue of PointNet's single global
pool. That limit check confirms the coarsest level really is global rather than merely "large-ish."

Now I have to match what *this task's* `graphunet` baseline actually does, because it is not the
gPool/gUnpool graph U-Net of the literature — it is the design-task multi-scale geometric variant, and
the differences are load-bearing. Three of them. First, the pooling here is **random node sampling by a
fixed ratio**, not a learned top-k-by-projection. The standard learned variant scores every node by a
normalized scalar projection onto a learned direction, keeps the top k, and gates the survivors by their
sigmoid scores so the projection direction gets a gradient. This task instead, at each scale, samples
`pool_ratio=0.5` of the nodes *uniformly at random* (`id_sampled = random.sample(range(n), k)`) and
keeps those rows — no learned selection vector, no sigmoid gate. So the coarsening is stochastic and
unlearned; the only learned parts are the graph convolutions. That is a real capacity reduction versus a
learned-selection variant, and it has a concrete downside I can put a probability on. A learned top-k
would deterministically keep the most informative nodes — the high-projection peaks — level after level,
so a critical stagnation or separation point would be carried all the way to the bottleneck by design.
Random sampling gives any specific point only a 0.5 chance of surviving each pooling step, and there are
four pooling steps down to the coarsest level, so a particular critical point reaches the bottom with
probability 0.5⁴ = 1/16 ≈ 6%. Put the other way, ~94% of any specific critical structure is gone by the
bottleneck on a given pass, and *which* structures survive differs mesh to mesh and pass to pass. That is
a lot of blind loss concentrated on exactly the sharp features the field metrics care about, and I should
expect this randomness to inject variance into the fine-field predictions. I should not import the
"learnable node selection" story.

Second — and this is the geometric heart of the variant — after sampling, the coarse graph is **rebuilt
by a radius graph in physical space**, not by restricting a graph power of the adjacency. The standard
graph-U-Net repairs connectivity lost to dropped nodes by squaring the (self-looped) adjacency so
two-hop neighbors through a removed node stay connected. This task instead calls `nng.radius_graph(pos_x,
r=list_r[n], ...)` on the *positions* of the surviving points, with the radius growing across scales as
above. So locality at each level is *geometric* — who is within radius r of whom in 3D — which is
sensible precisely because the mesh carries real coordinates (`pos_x` is `x[:, :2]` at the top level,
then the sampled positions thereafter); a radius graph is the natural neighborhood on a point cloud with
metric structure, and the growing radius is what turns the coarse levels into the spatial analogue of
PointNet's global pool.

Third, the unpooling is **nearest-neighbor interpolation in space**, not scatter-by-saved-index. The
standard gUnpool scatters the coarse rows back to their recorded indices and leaves dropped rows zero
until a skip fills them. This task instead, on the way up, assigns each fine point the feature of its
*nearest* coarse point (`cluster = nng.nearest(pos_x_up, pos_x_down); x_up = x[cluster]`) — a geometric
interpolation that fills *every* fine point, not just the survivors. That is convenient but crude: it is
piecewise-constant, so every fine point falling in a coarse point's Voronoi cell receives the *identical*
upsampled feature. Put a number on it at the deepest step: 500 coarse features spread over 8000 fine
points means each coarse feature is copied to ~16 neighbors on average, so the deepest information is
quantized to ~500 distinct values across the mesh before the skip refines it. That is a piecewise-constant
reconstruction that will blur sharp field gradients — the same peaks the pressure relative-L2 punishes.
Then it concatenates the interpolated feature with the saved encoder feature at that level and runs a
SAGE convolution. The skip is **concatenation** (`torch.cat([z, z_list[n-1]], dim=1)`), and the down path
*doubles* the hidden width at each scale (`out_channels = 2 * size_hidden_layers`), so at `n_hidden=16`
the down widths run 16 → 32 → 64 → 128 → 256, and the up convolutions consume `3 *
size_hidden_layers_init` channels — the interpolated coarse feature plus the saved finer feature. The
convolutions are `SAGEConv` with `BatchNorm` and ReLU. One more matching detail I must respect: the
encoder lifts `fun_dim` *only* (`x = fx.squeeze(0)`, encoder `MLP(args.fun_dim, ...)`), so the
coordinates do *not* enter the node features — they enter only through the radius graph and the spatial
interpolation. That is the opposite of PointNet, which concatenated the coordinates into the point
feature, and it is a subtle liability: a point's own position is available to this model only through the
geometry operations, never as a learnable channel it can transform.

Let me trace the down path once with concrete numbers, because the node-count-and-width schedule is what
makes the coarsening arithmetic real. Start at N=8000 with the Car head; the encoder lifts `fun_dim=7` to
16 channels, `MLP(7,32,16)` costing `(7·32+32)+(32·16+16) = 256 + 528 = 784` parameters, giving `(8000,
16)`. `down_layers[0]` is a `SAGEConv(16,16)` at full resolution, still `(8000, 16)`. Then four
downsample-and-convolve steps, each halving the node count and doubling the width: random-sample to 4000
nodes and `SAGEConv(16,32)` → `(4000, 32)`; to 2000 and `SAGEConv(32,64)` → `(2000, 64)`; to 1000 and
`SAGEConv(64,128)` → `(1000, 128)`; to 500 and `SAGEConv(128,256)` → `(500, 256)`. So the coarsest level
is 500 nodes at 256 channels — the 16× coarsening and the width-doubling I described, made concrete. A
`SAGEConv(in,out)` is two linears (root and neighbor), ≈`2·in·out` parameters, so the down convs cost
≈`2·(16·16 + 16·32 + 32·64 + 64·128 + 128·256) = 2·(256+512+2048+8192+32768) ≈ 87k`, the widest single
step (128→256) dominating at ~65.5k. The up path roughly mirrors it through the `3·n_hidden_init`-wide
concatenation convolutions, and the decoder `MLP(16,32,4)` is `(16·32+32)+(32·4+4)=544+132=676`. Summed
with BatchNorms the whole model is on the order of a couple hundred thousand parameters.

That total is the honest accounting for the width choice. `CONFIG_OVERRIDES = {'n_hidden': 16}` is the
canonical Graph_UNet setting, and as with PointNet this is a baseline-fidelity choice rather than a
budget-forced one: a couple-hundred-thousand-parameter model against a multi-million-parameter budget
anchor (1.05× the wide attention model) uses a few percent of the cap, so the budget is again slack and
16 is simply what keeps this a faithful baseline — not a knob the budget forced down. It is worth noting
the contrast with what widening would buy here: unlike PointNet, where the defect was a qualitative
bottleneck that no width could fix, this variant's defects are the *random* pooling and the
*piecewise-constant* unpooling, which widening the channels also cannot repair — a wider feature that is
still smeared across ~16 neighbors is still piecewise-constant. So I keep 16 and treat the coarsening
operations, not the width, as the thing the next rung must address. Like the other graph baselines it
raises if `geo` is None, since it genuinely message-passes over the edges.

Let me make sure I see why this should beat PointNet and where it might still fall short, because that is
the budget for the rung after it. The multi-scale stack gives every point genuine local context (SAGE at
each resolution) *and* long-range reach (the coarse levels with growing radius), so I expect the **field
errors to improve over PointNet** in the best case — the model can now resolve local structure that one
global vector could not. The drag *ordering* should stay strong or improve, since the coarse levels still
capture the global pressure asymmetry, possibly *better* than a single max because the hierarchy is
spatial rather than a lone extremum. But two things in *this* variant worry me, and they are exactly the
two crude operations I quantified. The pooling is *random*, so the coarsening throws nodes away blindly —
at `n_hidden=16` and 0.5 ratios over five scales the coarsest graph is 500 nodes and which survive is
luck, which injects variance and may *hurt* the fine field details. And the nearest-neighbor unpooling is
piecewise-constant, ~500 distinct values smeared over 8000 points at the deepest step, which will blur
the sharp gradients — and the field the floor already reconstructed worst was the peaky *pressure*, so
this is likely to bite pressure hardest. So my falsifiable expectation splits: graph U-Net should
**improve the drag rank correlation over PointNet's 0.962** (the spatial hierarchy is a better global
summary than one max — I would not be surprised to see rho_d climb above 0.98), but the **field errors
may not uniformly beat PointNet**, because random pooling plus piecewise-constant unpooling are lossy.
Concretely I would not be shocked if the Car pressure relative-L2 *rises* above 0.106 and the velocity
stays in the same ballpark as 0.042, and I expect AirfRANS — a 2D case where the radius graph and the
interpolation are even blunter, and where the floor already sat at 0.039/0.039 — to look the worst of the
three, with velocity the more exposed there because a 2D wake has long thin structures a piecewise-constant
interpolation cannot follow. If that is what comes back, the lesson for the next rung is written: the
*coarsening* is the weak link, so a method that does *flat*, learned, per-node neighbor aggregation at
full resolution — no random pooling, no interpolation loss — should reclaim the field accuracy the lossy
hierarchy gave up. That is message passing without the U-Net, and it is where the third rung goes.

The causal chain, threaded from PointNet's result: PointNet's strong drag rank (0.962) but loose drag
magnitude (0.0247) and its worst-field-on-pressure signature (Car 0.106 press vs 0.042 velo) diagnose a
*locality* failure — one global max-pool gives every point the same shape-wide context, so local field
structure cannot be resolved → the fix is to use the `geo` edges for neighbor aggregation, but flat
message passing alone would need ~diameter-many layers (√N ≈ 90 hops) and over-smooth, trading
"global-only" for "local-only," and jumping to attention would burn the most expressive operator too
early → so I want both local and long-range, which is the U-Net pattern realized on the mesh as a
multi-scale graph encoder-decoder, where 0.5-ratio pooling over five scales (8000 → 500) plus a
growing radius (r up to 10, spanning the whole body) buys global reach in five levels instead of ninety
→ matching *this task's* graph U-Net variant: **random** node sampling at ratio 0.5 per scale (not
learned top-k), coarse graphs **rebuilt by a growing-radius graph in physical space** (not adjacency
powers), **nearest-neighbor spatial unpooling** (piecewise-constant, ~500 values over 8000 points at the
bottom) with **concatenation** skips, width-doubling SAGE convolutions (16 → 256 down), and the encoder
consuming only `fun_dim` with coordinates entering through the geometry — at `n_hidden=16` — expecting
the spatial hierarchy to raise rho_d over PointNet while the random, piecewise-constant coarsening leaves
the field errors lossy (pressure and 2D AirfRANS most exposed) enough that flat learned message passing
will be the better field model at the next rung. The full scaffold module is in the answer.
