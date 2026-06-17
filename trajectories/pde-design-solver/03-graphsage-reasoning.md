The graph U-Net result confirmed the prediction I was most worried about, and it does so cleanly: the
spatial hierarchy bought a better drag *ordering* but it *cost* field accuracy. rho_d rose to 0.981
(up from PointNet's 0.962 — the coarse levels with growing radius are indeed a better global summary
than one max-pool), but every field error went the wrong way: Car pressure relative-L2 climbed to
0.113 (was 0.106 under PointNet), Car velocity to 0.050 (was 0.042), and AirfRANS got markedly worse —
pressure 0.064 (was 0.039) and velocity a striking 0.099 (was 0.039). The drag *magnitude* also
loosened slightly (c_d 0.0231 vs 0.0247 — roughly flat). So the lossy-coarsening diagnosis was right:
**random** node sampling at ratio 0.5 over five scales throws away nodes blindly, and **nearest-
neighbor unpooling** is piecewise-constant — every fine point in a coarse cell gets the *same*
upsampled feature — which blurs exactly the sharp field gradients the relative-L2 metric punishes. The
AirfRANS collapse is the tell: in the 2D case the radius graph and the spatial interpolation are even
blunter, and the velocity error nearly tripled. The hierarchy is reaching far, but it is destroying
local fidelity on the way down and not fully rebuilding it on the way up.

So the lesson is precise: the *coarsening* is the weak link, not the message passing. I want to keep
learned neighbor aggregation — that is the thing the U-Net got right at each level — but I want to do
it at **full resolution**, with **no random pooling and no interpolation loss**, so that every point
keeps its identity through the whole network and the field is never down-then-up-sampled. That is flat
message passing: stack graph-convolution layers at the original mesh resolution, each layer letting a
point aggregate from its neighbors, with depth giving reach (k layers ≈ k hops) and no lossy
coarsening anywhere. The trade is that flat layers reach only k hops, so very long-range coupling is
weaker than the U-Net's coarse levels — but the U-Net just showed me that the long-range reach was
*not worth* the field-accuracy it cost. On these design tasks the field errors and the drag-magnitude
error are the fidelity signals I most need to drive down, and they live in *local* structure that a
lossless flat stack should reconstruct far better. So the third rung trades the U-Net's aggressive
multi-scale reach for lossless full-resolution message passing, betting that field accuracy is where
the points are.

Now the question is *which* flat message-passing layer, and here I have to think about what makes a
graph convolution work well as a per-node feature generator on a variable-size, hub-heavy mesh — and
match what this task's `graphsage` baseline actually implements, because it is the design-task SAGE
variant and not the full sample-and-aggregate algorithm from the inductive-embedding literature. The
core idea I want is the one that distinguishes SAGE from a plain GCN: instead of folding a node's own
feature into the same averaging pot as its neighbors (which over K layers dilutes the node's own
identity into a growing crowd of neighbors-of-neighbors — the over-smoothing I worried about with flat
GCN), keep the node's own representation on a *separate channel* and combine it with the aggregated
neighborhood. That is the concat-then-transform structure: `h_v = σ(W · [h_v ; AGG(h_u : u ∈ N(v))])`,
a skip connection across depth that gives each node a clean, undiluted channel for its own signal at
every layer. On a PDE mesh this matters a lot: a point on a sharp pressure ridge must keep its own
distinctive feature even as it aggregates from smoother neighbors, or the ridge gets averaged away —
which is exactly the kind of gradient the U-Net's coarsening destroyed.

But I must be honest about what *this* harness exposes, because the differences from the paper SAGE
are real and load-bearing. First, the aggregator: the paper's expressive choice is the per-neighbor-
MLP-then-elementwise-max pool (a soft existential over the neighbor set), with mean as the cheap
special case and an order-randomized LSTM as the high-capacity option. This task uses
`torch_geometric.nn.SAGEConv`, whose default aggregation is the **mean** — so I do not get the pool
aggregator's existential semantics; the neighbor summary is a degree-mean. That is a capacity
reduction versus the paper's favored variant, and I should not claim the max-pool's ability to isolate
a single distinctive neighbor. Second, and more important: the paper's signature move is **fixed-size
uniform neighbor sampling** (draw S_k neighbors per node per layer, fresh each layer) so per-batch
cost is bounded regardless of hub degree. This harness does **no sampling** — `SAGEConv(z, edge_index)`
aggregates over the *full* neighbor set every layer. That is affordable here precisely because the
meshes are modest (~5000–10000 points, batch size one) and the radius graph the loop builds has
bounded degree; the scaling problem that *motivated* sampling in the paper does not bite at this scale.
So the "inductive minibatching over a 200k-node graph" story is simply absent — I drop it, and I note
that full-neighborhood aggregation is actually *more* faithful to the local field here than a
subsampled one would be, which is a virtue for field accuracy, not a compromise.

The concrete plumbing of the variant: the encoder lifts the concatenated coordinates and features
(`fun_dim + space_dim → n_hidden`) — note this *does* concatenate the coordinates, unlike the graph
U-Net which fed only `fun_dim`; here the point's position is part of its node feature, which is right
for a flat model that has no other geometric channel. Then a stack of `SAGEConv` layers all at width
`n_hidden`: an `in_layer` from `n_hidden → n_hidden`, then `n_layers − 1` hidden SAGE layers, then an
`out_layer`, each followed by **BatchNorm** (`track_running_stats=False`, since batch is one mesh and
running stats over single graphs would be meaningless) and a ReLU. The decoder maps `n_hidden →
out_dim`. The whole thing operates on the squeezed `(N, C)` tensor and re-adds the batch dimension at
the end. It raises if `geo` is None, like the other graph baselines. The width is the GraphSAGE
paper-faithful setting and notably *larger* than the graph models before it: `CONFIG_OVERRIDES =
{'n_hidden': 128}` — eight times PointNet's and Graph_UNet's 16. This is deliberate and it is part of
why I expect SAGE to win on fields: a flat stack at full resolution with eight times the width has far
more capacity to represent the local field, and it spends none of that capacity on lossy pooling.

Let me reason about what this should fix relative to the graph U-Net, because that is the falsifiable
bet. The whole point of going flat-and-lossless is field accuracy, so I expect the **field errors to
drop below both prior rungs** — the U-Net's 0.113/0.050 on Car and especially its blown-out AirfRANS
0.064/0.099 should come down substantially, because nothing is being random-pooled or piecewise-
interpolated and the width is 8× larger. AirfRANS in particular should recover the most, since that is
where the U-Net's geometric coarsening was bluntest. I also expect the **drag-magnitude error (c_d) to
improve** — a sharper, lossless pressure field integrates to a more accurate coefficient — so c_d
should fall below both 0.0247 and 0.0231. The one place I am *less* sure SAGE wins is the drag *rank
correlation*: flat message passing reaches only `n_layers` hops, so its global summary of front-to-back
pressure asymmetry is weaker than the U-Net's explicit coarse levels. So I would not be surprised if
**rho_d comes in slightly *below* the graph U-Net's 0.981** even as everything else improves — the
U-Net's one genuine advantage was long-range reach for ordering, and flat SAGE gives a little of that
back. If that is the pattern — SAGE better on c_d and all field errors, U-Net marginally better on
rho_d alone — then the trajectory's verdict is clear: lossless local message passing dominates the
lossy hierarchy on fidelity, and the remaining gap is the *global* correlation that even a deep flat
SAGE cannot cheaply carry across the whole body. That is exactly the gap that motivates leaving graph
message passing behind entirely for an attention-style operator at the next rung — one that models
correlations among physical states across the *whole* domain in a single layer, with no diameter limit
and no coarsening loss.

The causal chain, threaded from the graph U-Net's result: the U-Net's measured trade — rho_d up to
0.981 but every field error worse than PointNet (Car 0.113/0.050, AirfRANS 0.064/0.099) and c_d flat —
diagnoses the *coarsening* (random pooling + piecewise-constant unpooling) as the destroyer of local
fidelity → so keep learned neighbor aggregation but do it at full resolution with no pooling and no
interpolation, i.e. flat message passing → choose SAGE's concat-self-then-transform so each point
keeps an undiluted own-channel against over-smoothing, matching this task's variant: `SAGEConv` with
**mean** aggregation (not the paper's max-pool), **full-neighborhood** aggregation with **no sampling**
(the paper's scaling motivation is absent at this mesh size), coordinates concatenated into the node
features, BatchNorm + ReLU per layer, at the paper-faithful **n_hidden=128** (8× the prior graph
models) → expecting field errors and c_d to drop below both prior rungs (AirfRANS recovering most)
while rho_d may dip just under the U-Net's 0.981, leaving long-range global correlation as the gap a
whole-domain attention operator must close next. The full scaffold module is in the answer.
