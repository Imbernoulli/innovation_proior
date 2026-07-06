The graph U-Net result confirmed the prediction I was most worried about, and it does so cleanly
enough that I can read the mechanism straight off the numbers. The spatial hierarchy bought a better
drag *ordering* — rho_d rose to 0.981 from PointNet's 0.962, a gain of 0.019, so the coarse levels
with growing radius really are a better global summary than one max-pool. But it *paid* for that with
field accuracy on every single field metric. Let me put the deltas on paper rather than eyeballing
them, because their sizes are the diagnosis. On Car, pressure relative-L2 went 0.106 → 0.113, up 6.6%,
and velocity 0.042 → 0.050, up 19%. On AirfRANS the damage is far larger: pressure 0.039 → 0.064, up
64%, and velocity 0.039 → 0.099, up 154% — a 2.5× blow-up. The drag *magnitude* barely moved: c_d
0.0247 → 0.0231, an improvement of only 6.5%, essentially flat within what one seed can tell me. So the
pattern is unmistakable: the U-Net traded field fidelity for a little ordering, and the trade was
worst exactly where I predicted the coarsening would be bluntest — the 2D AirfRANS case, where the
radius graph and the nearest-neighbor interpolation have the least geometric room to work and where
the velocity error nearly tripled.

That AirfRANS velocity number is the tell, and I want to name the mechanism precisely because the next
rung has to be its cure. Two operations in that variant were lossy, and I quantified both at the last
rung: random node sampling at ratio 0.5 over five scales keeps any specific critical point with
probability only 0.5⁴ ≈ 6% at the bottleneck, so ~94% of any given sharp feature is gone on a pass and
*which* features survive is luck that changes mesh to mesh; and nearest-neighbor unpooling is
piecewise-constant, so at the deepest step ~500 coarse features are smeared over ~8000 fine points,
quantizing the deep representation to ~500 distinct values across the body. A 2D wake is a long, thin,
sharp-gradient structure; piecewise-constant interpolation cannot follow a thin filament, and random
pooling throws away the very points that define it. So the velocity field over the volume — the field
with the sharpest thin structure — is precisely the one that collapses, and it collapses hardest in
2D. The diagnosis is therefore sharp: the *coarsening* is the weak link, not the message passing. The
per-scale graph convolution — a point aggregating from its neighbors — was the thing the U-Net got
*right*; it was the random down-sampling and the piecewise-constant up-sampling wrapped around it that
destroyed the fields.

So the design question for this rung is narrow and well-posed: keep the learned neighbor aggregation,
but strip away everything lossy around it. Concretely, do the message passing at **full resolution**,
with **no pooling and no interpolation anywhere**, so every point keeps its own identity through the
entire network and the field is never down-then-up-sampled. That is flat message passing: stack graph
convolution layers at the original mesh resolution, each layer letting a point aggregate from its mesh
neighbors, with depth alone giving reach. I have to be honest about the cost of going flat, because I
argued against exactly this at the U-Net rung: flat layers reach only k hops in k layers, so with a
handful of layers I recover only a handful of hops, and the mesh diameter (√N ≈ 90 hops on a surface
manifold of N≈8000, N^{1/3} ≈ 20 even as a volume cloud) dwarfs that, so a flat stack cannot carry
nose-to-wake correlation the way the U-Net's coarse levels could. But the U-Net just *proved* to me
that its long-range reach was not worth what it cost: it bought 0.019 of rho_d and paid a 154% AirfRANS
velocity blow-up. On these design tasks the field errors and the drag-magnitude error are the fidelity
signals I most need to drive down, and they live in *local* structure that a lossless flat stack should
reconstruct far better than any coarsened hierarchy. So the bet of this rung is explicit: trade the
U-Net's aggressive multi-scale reach for lossless full-resolution message passing, wagering that field
accuracy is where the points are and accepting a possible small give-back on the one metric the U-Net
actually won.

Now, *which* flat graph-convolution layer, because that choice is not cosmetic and I have three live
candidates that I should walk rather than grab the first. The plain graph-convolutional layer (GCN)
folds a node's own feature into the *same* averaging pot as its neighbors: `h'_v =
σ(W·mean(h_u : u ∈ N(v)∪{v}))`. That symmetry is exactly what I must avoid, and I can prove it kills me
with a two-node trace. Take two nodes a and b, connected, and run one linear GCN step with a self-loop,
so each node's new feature is `W·(h_a + h_b)/2` — *identical* for both nodes after one step, because
they share the same neighborhood-plus-self set. The difference `h_a − h_b` is annihilated in a single
layer; stack K of them and every connected component collapses toward one shared vector, the
over-smoothing I feared. A point on a sharp pressure ridge would be averaged into its smoother neighbors
and the ridge would vanish — precisely the gradient the U-Net's coarsening already destroyed, now
destroyed a second way. So GCN is out on a computed, not a felt, basis.

The second candidate is graph attention (GAT): learn a per-edge attention weight so a node aggregates a
*weighted* neighbor mean, the weights from a small attention score on each edge. That fixes the uniform
averaging — a node could down-weight the neighbors that would smooth its ridge away — but it adds
per-edge attention parameters and an extra softmax over every node's neighborhood each layer, and,
decisively, it is *still local*: GAT reaches one hop per layer just like the others, so it does nothing
about the global-ordering gap that is the real residual, while spending parameters and compute on
machinery whose gain (edge weighting) is not the thing this rung is trying to buy. I would be adding
cost against the wrong axis. The third candidate is the one that fixes GCN's collapse with the least
machinery: keep the node's own representation on a *separate* channel and combine it with the
aggregated neighborhood — `h'_v = σ(W·[h_v ; AGG(h_u : u ∈ N(v))])`, a concat-then-transform that is a
skip connection across depth. That is GraphSAGE, and it is the right flat layer here because it directly
answers the over-smoothing objection while adding nothing GAT-like. Let me verify it actually preserves
node identity where GCN did not, on the same two-node trace, dropping the nonlinearity so I can see the
algebra. SAGE with mean aggregation and no self-loop gives node a the neighbor-mean `h_b` and node b
the neighbor-mean `h_a`, so `out_a = W_r h_a + W_n h_b` and `out_b = W_r h_b + W_n h_a`, and therefore
`out_a − out_b = (W_r − W_n)(h_a − h_b)`. The difference is *preserved*, scaled by `W_r − W_n`, not
annihilated — as long as the self-weight and neighbor-weight matrices differ (and at random init they
do), node identities survive every layer. That is the whole reason the separate self-channel matters on
a PDE mesh: a point on a sharp ridge keeps its distinctive feature even as it aggregates from smoother
neighbors, so the ridge is not averaged away. The trace confirms the mechanism rather than my asserting
it.

Now I have to match what *this task's* `graphsage` baseline actually implements, because it is the
design-task SAGE variant and not the full sample-and-aggregate algorithm, and the differences are
load-bearing. First, the aggregator. The canonical expressive choice is a per-neighbor-MLP-then-
elementwise-max pool — a soft existential over the neighbor set that can isolate a single distinctive
neighbor — with mean as the cheap special case and an order-randomized LSTM as the high-capacity option.
This task uses `torch_geometric.nn.SAGEConv`, whose default aggregation is the **mean**, so I do not get
the pool aggregator's existential semantics; the neighbor summary is a degree-mean. That is a real
capacity reduction versus the favored max-pool variant, and I should not claim the max-pool's ability to
spotlight one neighbor. Second, and more important, SAGE's signature move is **fixed-size uniform
neighbor sampling**: draw S_k neighbors per node per layer, fresh each layer, so per-batch cost is
bounded regardless of a hub node's degree. This harness does **no sampling** — `SAGEConv(z, edge_index)`
aggregates over the *full* neighbor set at every layer. That is affordable here precisely because the
meshes are modest (~5000–10000 points, batch size one) and the radius graph the harness builds has
bounded degree, so the giant-minibatch scaling problem that *motivated* sampling simply does not bite at
this scale. And it is not merely tolerable, it is a *virtue* for this task: full-neighborhood
aggregation is strictly more faithful to the local field than a subsampled estimate would be, because it
averages over the true neighborhood rather than a noisy draw of it — I keep every neighbor's
contribution to a point's local physics. So the inductive-minibatching-over-a-giant-graph story is
absent, and dropping it costs me nothing here.

The rest of the plumbing I must respect exactly. The encoder lifts the *concatenated coordinates and
features*, `fun_dim + space_dim → n_hidden` — and this is a deliberate reversal from the graph U-Net,
which fed only `fun_dim` and let coordinates enter solely through the radius graph and the spatial
interpolation. Here the point's own position is part of its node feature, a channel the model can
transform, which is right for a flat model that has *no other* geometric operation to inject position:
there is no radius-graph rebuild, no spatial unpool, so if the coordinates did not ride in the node
feature the model would be spatially blind. Then a stack of `SAGEConv` layers all at width `n_hidden`:
an `in_layer` from `n_hidden → n_hidden`, then `n_layers − 1` hidden SAGE layers at `n_hidden →
n_hidden`, then an `out_layer` at `n_hidden → n_hidden`, each followed by **BatchNorm1d** with
`track_running_stats=False` and a ReLU. The disabled running stats matter and are correct: batch is one
mesh, so a running mean/variance accumulated over single graphs of wildly varying point count would be
meaningless; normalizing each mesh by its own batch statistics is the only sensible choice when the
"batch" is a single variable-size point cloud. The decoder maps `n_hidden → out_dim`. The whole thing
runs on the squeezed `(N, C)` tensor and re-adds the batch axis at the end with `unsqueeze(0)`, and it
raises if `geo` is None, like the other graph baselines, because it genuinely message-passes over the
edges.

Let me trace shapes and parameters once, because a dimension and budget check is the cheapest way to
catch a plumbing error before a 200-epoch run. With N=8000 and the Car head (space_dim=3, out_dim=4),
the concatenated input is `(1, 8000, 10)`, squeezed to `(8000, 10)`. The encoder `MLP(10, 256, 128)`
maps it to `(8000, 128)` and costs `(10·256+256)+(256·128+128) = 2816 + 32896 = 35712` parameters. Each
`SAGEConv(128,128)` carries two linears — a root map and a neighbor map, `Linear(128,128)` each — so
`2·128·128 + 128 ≈ 32896` parameters apiece, and it preserves the `(8000, 128)` shape at every layer, so
there is *no* down/up-sampling in the tensor pipeline at all, which is the entire point. BatchNorm1d on
`(8000, 128)` normalizes across the 8000-point axis per channel — 256 parameters each. The decoder
`MLP(128, 256, 4)` costs `(128·256+256)+(256·4+4) = 33024 + 1028 = 34052`. So a stack of a handful of
SAGE layers plus encoder and decoder is on the order of a few hundred thousand parameters — for a
five-or-so-layer stack, roughly `35712 + 6·32896 + 34052 ≈ 267k` — which against the multi-million-
parameter budget anchor (1.05× the wide attention model) is a few percent, nowhere near binding. So the
width here is a *fidelity* choice, not a budget-forced one, and it is a notably *large* fidelity choice:
`CONFIG_OVERRIDES = {'n_hidden': 128}`, eight times PointNet's and the graph U-Net's 16. That 8× is
deliberate and it is part of why I expect SAGE to win on fields — a flat stack at full resolution with
eight times the width has far more capacity to represent local field structure, and it spends *none* of
that capacity on lossy pooling, unlike the U-Net which had to widen down to 256 channels just to feed
its coarsening.

For AirfRANS the only change is space_dim=2, so the input concat is `(N, 9)` and the encoder's first
linear absorbs it; for AirCraft out_dim=6, so the decoder's last linear widens to six. The shapes close.
One consequence of the flat design worth stating for the reach budget: with L message-passing layers a
point's receptive field is exactly its L-hop neighborhood, so the model sees local structure out to L
hops and nothing beyond. Against a mesh diameter of ~90 hops that is a small fraction of the body, which
is exactly the limitation I am knowingly accepting in exchange for zero coarsening loss.

There is an obvious escape I should close off before I commit, because if it worked it would let flat
SAGE recover the U-Net's global reach for free: why not simply *stack more layers* until L reaches the
diameter, so a flat lossless stack sees the whole body after all? Two computed reasons kill it. First,
the budget arithmetic: to reach 90 hops I would need ~90 SAGE layers at `2·128·128 ≈ 32.9k` parameters
each, roughly 3 million parameters in the message-passing body alone — now genuinely pressing against
the multi-million anchor that the flat 5-layer stack sat at a few percent of, so the "width is a
fidelity choice, budget is slack" claim would stop holding. Second, and worse, the self-channel *slows*
over-smoothing but does not *eliminate* it: my two-node trace showed the difference is preserved as
`(W_r − W_n)(h_a − h_b)`, but the neighbor branch still injects the neighbor mean at every layer, so
over 90 averaging steps a node's feature drifts steadily toward its 90-hop-neighborhood mean — which on
a connected mesh is most of the body — and the sharp local field I went flat to protect washes out
again. So depth cannot cheaply buy global reach here: past a modest L the marginal hop costs
over-smoothing that erodes exactly the field fidelity that is this rung's whole reason for existing.
That is why I keep the stack shallow and accept the L-hop ceiling as a *known residual* rather than
trying to spend depth against it — the global-correlation gap is real and it is not a graph model's to
close.

It is worth being explicit about why the drag-magnitude error should move even though the U-Net's did
not, because the mechanism is the same integral that ranked designs at the floor. The drag coefficient
is a surface integral of pressure projected on the inlet direction, `c_d ∝ ∮ (p·n)·d̂`, so its *value*
(not just its rank) is set by how accurately the pressure field is reconstructed near the surface where
the integrand is largest — the stagnation and separation regions with the sharpest gradients. A lossless
full-resolution stack that keeps those sharp surface features intact integrates to a coefficient closer
to the truth, whereas the U-Net's random pooling threw away surface peaks and its piecewise-constant
unpooling flattened the gradient, so its integral stayed loose (c_d essentially flat at 0.0231). This is
the same reason I expect the pressure relative-L2 and c_d to improve *together* rather than
independently: both are functionals of the same sharp surface pressure field, so the sharpness SAGE buys
should show up in both at once.

Let me make the falsifiable bet precise against the two prior rungs' measured numbers, because that is
what the next rung inherits. The whole reason to go flat-and-lossless is field accuracy, so I expect the
**field errors to drop below both prior rungs** — the U-Net's 0.113/0.050 on Car and especially its
blown-out AirfRANS 0.064/0.099 should come down substantially, because nothing is random-pooled or
piecewise-interpolated and the width is 8× larger. AirfRANS in particular should recover the *most*,
since that is where the U-Net's geometric coarsening was bluntest and where the velocity error nearly
tripled; a lossless full-resolution stack has no reason to blow up a thin 2D wake. I also expect the
**drag-magnitude error c_d to improve below both 0.0247 and 0.0231** — a sharper, lossless pressure
field integrates to a more accurate coefficient, and c_d is a relative error of that value, so it
rewards exactly the local sharpness I am buying. The one place I am *less* sure SAGE wins is the drag
*rank correlation*: flat message passing reaches only L hops, so its summary of front-to-back pressure
asymmetry is weaker than the U-Net's explicit coarse levels, which is the U-Net's one genuine advantage.
So I would not be surprised if **rho_d comes in slightly below the U-Net's 0.981** even as everything
else improves — flat SAGE gives a little of that global ordering back. On AirCraft, where both prior
rungs hovered near 0.66/0.38 with no published reference, I would watch whether it tracks the Car field
gains rather than reading it as a verdict; those magnitudes sit an order above the Car and AirfRANS
errors, which I continue to read as an unnormalized task-internal probe rather than a field the model is
failing on outright, so a small move there tells me little either way and I weight the two published
benchmarks accordingly. If the pattern comes back as "SAGE better on c_d and every
field error, U-Net marginally better on rho_d alone," then the trajectory's verdict is clear: lossless
local message passing dominates the lossy hierarchy on fidelity, and the *only* residual is the global
rank correlation that even a deep flat stack cannot cheaply carry across a 90-hop body. That residual —
whole-domain correlation with no diameter limit and no coarsening loss — is exactly what would motivate
leaving graph message passing behind entirely for an operator that relates every point to every other
in a single layer at the next rung.

The causal chain, threaded from the graph U-Net's result: the U-Net's measured trade — rho_d up 0.019
to 0.981 but every field error worse than PointNet (Car +6.6%/+19%, AirfRANS +64%/+154%) and c_d
essentially flat at 0.0231 — pins the *coarsening* (random pooling keeping a critical point with 6%
probability, plus ~500-value piecewise-constant unpooling) as the destroyer of local fidelity, worst on
the thin 2D wake → so keep learned neighbor aggregation but do it losslessly at full resolution with no
pooling and no interpolation, i.e. flat message passing → among flat layers reject GCN (a two-node trace
shows it annihilates `h_a − h_b` in one step, over-smoothing) and GAT (still one-hop, extra edge
machinery against the wrong axis), and choose SAGE's concat-self-then-transform, verified on the same
trace to preserve `(W_r − W_n)(h_a − h_b)` so each point keeps an undiluted own-channel → match this
task's variant: `SAGEConv` with **mean** aggregation (not the canonical max-pool), **full-neighborhood**
aggregation with **no sampling** (the scaling motivation is absent and full neighborhoods are more
faithful here), coordinates concatenated into the node feature (unlike the U-Net's fun_dim-only encoder,
because a flat model has no other geometric channel), BatchNorm with running stats off + ReLU per layer,
at the canonical **n_hidden=128** (8× the prior graph models, ~267k params, a few percent of the budget)
→ expecting field errors and c_d to drop below both prior rungs (AirfRANS recovering most) while rho_d
may dip just under 0.981, leaving long-range global correlation as the gap a whole-domain operator must
close next. The full scaffold module is in the answer.
