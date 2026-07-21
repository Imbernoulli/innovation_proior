The graph U-Net result confirmed the prediction I was most worried about, cleanly enough that I can
read the mechanism straight off the numbers. The spatial hierarchy bought a better drag *ordering* —
rho_d rose to 0.981 from PointNet's 0.962, a gain of 0.019, so the coarse levels with growing radius
really are a better global summary than one max-pool. But it *paid* for that on every field metric. On
Car, pressure relative-L2 went 0.106 → 0.113 (up 6.6%) and velocity 0.042 → 0.050 (up 19%). On
AirfRANS the damage is far larger: pressure 0.039 → 0.064 (up 64%) and velocity 0.039 → 0.099 (up
154%, a 2.5× blow-up). The drag *magnitude* barely moved: c_d 0.0247 → 0.0231, essentially flat within
what one seed can tell me. So the U-Net traded field fidelity for a little ordering, and the trade was
worst exactly where I predicted the coarsening would be bluntest — the 2D AirfRANS case, where the
radius graph and the nearest-neighbor interpolation have the least geometric room to work and the
velocity error nearly tripled.

That AirfRANS velocity number is the tell, and the next rung has to be its cure. Two operations in that
variant were lossy, both quantified at the last rung: random node sampling at ratio 0.5 over five
scales keeps any specific critical point with probability only 0.5⁴ ≈ 6% at the bottleneck, so ~94% of
any given sharp feature is gone on a pass and *which* survive is luck; and nearest-neighbor unpooling
is piecewise-constant, so at the deepest step ~500 coarse features are smeared over ~8000 fine points,
quantizing the deep representation to ~500 distinct values. A 2D wake is a long, thin, sharp-gradient
structure; piecewise-constant interpolation cannot follow a thin filament, and random pooling throws
away the very points that define it. So the velocity field over the volume — the sharpest thin
structure — is precisely the one that collapses, hardest in 2D. The diagnosis is therefore that the
*coarsening* is the weak link, not the message passing: the per-scale graph convolution was the thing
the U-Net got *right*; the random down-sampling and piecewise-constant up-sampling wrapped around it
destroyed the fields.

So the design question is narrow: keep the learned neighbor aggregation, strip away everything lossy
around it. Do the message passing at **full resolution**, with **no pooling and no interpolation
anywhere**, so every point keeps its own identity through the network and the field is never
down-then-up-sampled. That is flat message passing: stack graph convolution layers at the original
mesh resolution, each layer letting a point aggregate from its mesh neighbors, with depth alone giving
reach. I have to be honest about the cost, because I argued against exactly this at the U-Net rung:
flat layers reach only k hops in k layers, and the mesh diameter (√N ≈ 90 hops on a surface manifold,
N^{1/3} ≈ 20 even as a volume cloud) dwarfs a handful of layers, so a flat stack cannot carry
nose-to-wake correlation the way the U-Net's coarse levels could. But the U-Net just *proved* its
long-range reach was not worth what it cost: it bought 0.019 of rho_d and paid a 154% AirfRANS velocity
blow-up. On these tasks the field errors and the drag-magnitude error are the fidelity signals I most
need to drive down, and they live in *local* structure a lossless flat stack should reconstruct far
better than any coarsened hierarchy. So the bet is explicit: trade the U-Net's aggressive multi-scale
reach for lossless full-resolution message passing, wagering that field accuracy is where the points
are and accepting a possible small give-back on the one metric the U-Net actually won.

*Which* flat graph-convolution layer, then — three live candidates. The plain graph-convolutional
layer (GCN) folds a node's own feature into the *same* averaging pot as its neighbors:
`h'_v = σ(W·mean(h_u : u ∈ N(v)∪{v}))`. That symmetry is exactly what I must avoid, and a two-node
trace shows it kills me. Take two connected nodes a and b and run one linear GCN step with a self-loop,
so each node's new feature is `W·(h_a + h_b)/2` — *identical* for both, because they share the same
neighborhood-plus-self set. The difference `h_a − h_b` is annihilated in a single layer; stack K of
them and every connected component collapses toward one shared vector, the over-smoothing I feared. A
point on a sharp pressure ridge would be averaged into its smoother neighbors and the ridge would
vanish — the gradient the U-Net's coarsening already destroyed, now destroyed a second way. So GCN is
out on a computed basis.

The second candidate is graph attention (GAT): learn a per-edge attention weight so a node aggregates
a *weighted* neighbor mean. That fixes the uniform averaging — a node could down-weight the neighbors
that would smooth its ridge away — but it adds per-edge attention parameters and a softmax over every
neighborhood each layer, and, decisively, it is *still local*: GAT reaches one hop per layer, so it
does nothing about the global-ordering gap that is the real residual, while spending parameters on
machinery whose gain (edge weighting) is not what this rung is trying to buy. The third candidate fixes
GCN's collapse with the least machinery: keep the node's own representation on a *separate* channel and
combine it with the aggregated neighborhood — `h'_v = σ(W·[h_v ; AGG(h_u : u ∈ N(v))])`, a
concat-then-transform that is a skip connection across depth. That is GraphSAGE. On the same two-node
trace, dropping the nonlinearity to see the algebra: SAGE with mean aggregation and no self-loop gives
node a the neighbor-mean `h_b` and node b the neighbor-mean `h_a`, so `out_a = W_r h_a + W_n h_b` and
`out_b = W_r h_b + W_n h_a`, hence `out_a − out_b = (W_r − W_n)(h_a − h_b)`. The difference is
*preserved*, scaled by `W_r − W_n`, not annihilated — as long as the self-weight and neighbor-weight
matrices differ (and at random init they do), node identities survive every layer. That is why the
separate self-channel matters on a PDE mesh: a point on a sharp ridge keeps its distinctive feature
even as it aggregates from smoother neighbors, so the ridge is not averaged away.

Now to match what *this task's* `graphsage` baseline implements, because it is the design-task SAGE
variant, not the full sample-and-aggregate algorithm, and the differences are load-bearing. First, the
aggregator. The canonical expressive choice is a per-neighbor-MLP-then-elementwise-max pool — a soft
existential over the neighbor set that can isolate a single distinctive neighbor. This uses
`torch_geometric.nn.SAGEConv`, whose default aggregation is the **mean**, so I do not get the pool
aggregator's existential semantics; the neighbor summary is a degree-mean. That is a real capacity
reduction, and I should not claim the max-pool's ability to spotlight one neighbor. Second, and more
important, SAGE's signature move is **fixed-size uniform neighbor sampling**: draw S_k neighbors per
node per layer so per-batch cost is bounded regardless of a hub node's degree. This baseline does **no
sampling** — `SAGEConv(z, edge_index)` aggregates over the *full* neighbor set at every layer. That is
affordable here precisely because the meshes are modest (~5000–10000 points, batch one) and the radius
graph has bounded degree, so the giant-minibatch scaling problem that *motivated* sampling does not
bite; and it is a *virtue*, since full-neighborhood aggregation is strictly more faithful to the local
field than a subsampled estimate — I keep every neighbor's contribution to a point's local physics. So
the inductive-minibatching-over-a-giant-graph story is absent, and dropping it costs nothing here.

The rest of the plumbing I respect exactly. The encoder lifts the *concatenated coordinates and
features*, `fun_dim + space_dim → n_hidden` — a deliberate reversal from the graph U-Net, which fed only
`fun_dim` and let coordinates enter solely through the radius graph and interpolation. Here the point's
own position is part of its node feature, a channel the model can transform, which is right for a flat
model that has *no other* geometric operation to inject position: no radius-graph rebuild, no spatial
unpool, so if the coordinates did not ride in the node feature the model would be spatially blind. Then
a stack of `SAGEConv` layers all at width `n_hidden`: an `in_layer` `n_hidden → n_hidden`, then
`n_layers − 1` hidden layers, then an `out_layer`, each followed by **BatchNorm1d** with
`track_running_stats=False` and ReLU. The disabled running stats are correct: batch is one mesh, so a
running mean/variance accumulated over single graphs of wildly varying point count would be
meaningless; normalizing each mesh by its own batch statistics is the only sensible choice. The decoder
maps `n_hidden → out_dim`, the whole thing runs on the squeezed `(N, C)` tensor and re-adds the batch
axis at the end, and it raises if `geo` is None like the other graph baselines.

Parameters and width: with the Car head the encoder `MLP(10, 256, 128)` costs ≈35.7k; each
`SAGEConv(128,128)` is two linears at ≈32.9k and preserves `(N, 128)` at every layer — *no*
down/up-sampling in the tensor pipeline at all, which is the entire point; the decoder `MLP(128, 256,
4)` costs ≈34k. A five-or-so-layer stack is ≈`35.7k + 6·32.9k + 34k ≈ 267k` parameters — a few percent
of the multi-million-parameter budget anchor, nowhere near binding. So the width is a *fidelity* choice
again, and a notably *large* one: `CONFIG_OVERRIDES = {'n_hidden': 128}`, eight times PointNet's and
the graph U-Net's 16. That 8× is deliberate and part of why I expect SAGE to win on fields — a flat
stack at full resolution with eight times the width has far more capacity to represent local field
structure, and it spends *none* of that capacity on lossy pooling, unlike the U-Net which had to widen
to 256 channels just to feed its coarsening. AirfRANS changes only space_dim=2, AirCraft only out_dim=6.
One consequence to state for the reach budget: with L message-passing layers a point's receptive field
is its L-hop neighborhood and nothing beyond — against a ~90-hop diameter, a small fraction of the body,
which is the limitation I am knowingly accepting in exchange for zero coarsening loss.

Could I recover the U-Net's reach for free by simply *stacking more layers* until L reaches the
diameter? Two computed reasons kill it. First, budget: to reach 90 hops I would need ~90 SAGE layers at
≈32.9k each, roughly 3 million parameters in the message-passing body alone — now genuinely pressing
against the anchor the flat 5-layer stack sat at a few percent of, so "width is a fidelity choice,
budget is slack" would stop holding. Second, and worse, the self-channel *slows* over-smoothing but
does not *eliminate* it: the two-node trace preserved the difference as `(W_r − W_n)(h_a − h_b)`, but
the neighbor branch still injects the neighbor mean at every layer, so over 90 averaging steps a node's
feature drifts steadily toward its 90-hop-neighborhood mean — most of the body on a connected mesh — and
the sharp local field washes out again. So depth cannot cheaply buy global reach; I keep the stack
shallow and accept the L-hop ceiling as a *known residual* rather than spending depth against it.

Why the drag-magnitude error should move even though the U-Net's did not: c_d ∝ ∮ (p·n)·d̂, so its
*value* (not just its rank) is set by how accurately the pressure field is reconstructed near the
surface where the integrand is largest — the stagnation and separation regions with the sharpest
gradients. A lossless full-resolution stack that keeps those sharp surface features intact integrates
to a coefficient closer to the truth, whereas the U-Net's random pooling threw away surface peaks and
its piecewise-constant unpooling flattened the gradient, so its integral stayed loose. This is why I
expect the pressure relative-L2 and c_d to improve *together*: both are functionals of the same sharp
surface pressure field.

The falsifiable bet against the prior rungs' numbers. The whole reason to go flat-and-lossless is field
accuracy, so I expect the **field errors to drop below both prior rungs** — the U-Net's 0.113/0.050 on
Car and especially its blown-out AirfRANS 0.064/0.099 should come down substantially, with AirfRANS
recovering the *most* since that is where the U-Net's coarsening was bluntest. I expect the
**drag-magnitude error c_d to improve below both 0.0247 and 0.0231**, since a sharper lossless pressure
field integrates to a more accurate coefficient. The one place I am *less* sure SAGE wins is the drag
*rank correlation*: flat message passing reaches only L hops, so its summary of front-to-back pressure
asymmetry is weaker than the U-Net's explicit coarse levels, the U-Net's one genuine advantage — so I
would not be surprised if **rho_d comes in slightly below the U-Net's 0.981** even as everything else
improves. On AirCraft, where both prior rungs hovered near 0.66/0.38 with no published reference, I
watch whether it tracks the Car field gains rather than reading it as a verdict; those magnitudes sit an
order above the Car and AirfRANS errors, which I continue to read as an unnormalized task-internal probe.
If the pattern comes back as "SAGE better on c_d and every field error, U-Net marginally better on rho_d
alone," the verdict is clear: lossless local message passing dominates the lossy hierarchy on fidelity,
and the *only* residual is the global rank correlation that even a deep flat stack cannot cheaply carry
across a 90-hop body. That residual — whole-domain correlation with no diameter limit and no coarsening
loss — is what would motivate leaving graph message passing behind for an operator that relates every
point to every other in a single layer. The full module is in the answer.
