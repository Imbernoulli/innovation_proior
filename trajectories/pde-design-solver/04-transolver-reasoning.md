GraphSAGE landed almost exactly where I bet it would, and the pattern across the three graph rungs now
tells me what is left to fix. The lossless flat stack reclaimed field accuracy on every front, and I
want the deltas on paper because their signs and sizes are the whole diagnosis. On Car, pressure
relative-L2 fell 0.113 → 0.089 (down 21% from the U-Net) and 0.106 → 0.089 (down 16% from PointNet), and
velocity fell 0.050 → 0.033 (down 34% from the U-Net, and below PointNet's 0.042 too) — the best fields
of any rung so far. AirfRANS recovered exactly the way I predicted the bluntest case would: pressure
0.064 → 0.046 (down 28%) and velocity 0.099 → 0.037, a 2.7× improvement that essentially undoes the
U-Net's blow-up. The drag-magnitude error improved as well, c_d 0.0193, down 17% from the U-Net's 0.0231
and 22% from PointNet's 0.0247 — the lowest yet, exactly as the "a sharper lossless pressure field
integrates to a truer coefficient" argument predicted, and pressure-L2 and c_d moved together as I said
two functionals of the same surface field should. And the one place I said SAGE might *lose* is exactly
where it did: rho_d came in at 0.978, a hair below the U-Net's 0.981 — a give-back of 0.003, tiny but in
the predicted direction, because a flat stack reaches only L hops and cannot carry front-to-back
correlation across a ~90-hop body.

So the graph trilogy resolves into a single clean statement. Lossless local message passing dominates
the lossy hierarchy on fidelity — every field error and c_d improved — but it gives a little of the
*global ordering* back, because every message-passing stack, flat or hierarchical, is local by
construction and reaches only as far as its depth (or, for the U-Net, buys reach only by paying
coarsening loss). That residual — global correlation with no diameter limit *and* no coarsening loss —
is the gap all three graph methods share in different guises, and it is what this rung has to close.
The question is direct: how do I model correlations across the *entire* domain in a single layer, on an
unstructured mesh of ~5000–10000 points, without a cost that makes it infeasible?

The honest answer is that I have to step out of the message-passing family entirely, because the limit
is structural, not a matter of the particular convolution. The operator-learning frame says each layer
should be a non-local integral operator — `(Ku)(p) = ∫ κ(p,q) u(q) dq` — and message passing is the
crude approximation that restricts the kernel `κ` to a local neighborhood and to a fixed (mean, or
weighted-mean) shape. The most *expressive* parameterization of that integral is attention: softmax
attention is exactly a Monte-Carlo discretization of the integral operator with a *learned* kernel and
the mesh points as the quadrature nodes, `κ(p,q) = softmax_q(⟨Wq_p, Wk_q⟩)`. That is the global,
learnable, geometry-agnostic operator I want — one layer relates every point to every other, no diameter
limit. But with the N mesh points as quadrature nodes the cost is O(N²): at N=8000 that is 6.4×10⁷
point-pairs per layer per head, which is the wall that made attention-over-points infeasible at this
scale in the first place.

I have two live ways around that wall, and I should walk the tempting one before I reject it. The first
is to keep the quadrature nodes as the N points but make the attention *linear* — the Galerkin
reassociation that computes `Q(KᵀV)` instead of `(QKᵀ)V`, dropping the softmax so the O(N²) score
matrix never forms and the cost falls to O(N·C²). That would make point-attention affordable. But it
does not fix the *deeper* problem, and the graph rungs already showed me what that problem is. Attention
over the N points — linear or not — spreads its modeling capacity across N² point-to-point relations,
the overwhelming majority of which are low-level and uninformative: two points a centimetre apart on the
same smooth panel relate trivially, and the *informative* physical correlations (the front region
governing the wake) are a tiny, non-local fraction diluted in that sea. This is the same observation
that made the graph methods leave performance on the table — a radius graph or a mesh edge groups points
by *location*, but the correlations I need group by *physical state*, which spans the domain non-locally.
So "cheaper attention over points" treats the cost symptom and leaves the dilution disease; the
quadrature *nodes* are wrong, and making them cheaper does not make them right.

The second way is the reframing that breaks the deadlock. Mesh points are an artifact of discretization
— a finite, arbitrary sampling of an underlying continuous physics. The physics does not live at the
points; it lives in the *states*. On a car the windshield, the license plate, and the headlights are all
in the same front-stagnation regime that governs drag, though they are scattered across the surface; and
two spatially adjacent points can sit in completely different states across a separation line. So the
move is: group points by *what physical state they are in* — learned from data, the groups free to be
any shape and to span the whole body — encode each group into a single token, run attention among the
few tokens, and broadcast back. If there are M such groups (slices), the encode and broadcast are O(NM)
and the attention is O(M²), so with M a small constant the whole operator is linear in N *and* the
attention — the part doing the global correlation — runs over M *meaningful* nodes instead of N noisy
ones. Both problems, the cost and the dilution, fall to the same change of quadrature nodes. Let me put
the arithmetic beside the point-attention wall to see the size of the win: at N=8000 and M=32, the
encode/broadcast is N·M = 2.56×10⁵ and the token attention is M² = 1024, against the 6.4×10⁷ of
point-attention — a ~250× reduction on the encode side and roughly 6×10⁴× on the attention itself. The
expensive quadratic now runs over 32 nodes, not 8000.

Making "assign each point to a slice" learnable means making it soft and differentiable. For each
point's per-head feature I project to M slice logits and take a **softmax over the slice axis**, so each
point gets a distribution over the M slices that sums to one. Let me verify that this is genuinely a
partition of unity and behaves at the extremes, because the whole construction rests on it. Softmax over
the slice axis gives `Σ_g w_ng = 1` for every point n, so the assignment is a convex partition — each
point's unit of mass is distributed across slices, none created or destroyed. A slice's token is the
**mass-normalized weighted mean** of its members, `t_g = Σ_n w_ng x_n / Σ_n w_ng`, so a slice that
happens to own many points does not get an artificially large token — it gets the *average* state of its
members. Two limit checks confirm the construction. If every point has the identical feature x, then
every token equals x regardless of the weights (the weighted mean of identical values is that value), so
a featureless field collapses to featureless tokens — correct. And if M=1, every point's single slice
weight is forced to 1, the one token is the plain mean of all point features, attention over a single
token is the identity, and the broadcast hands every point that same mean — i.e. M=1 collapses exactly to
global mean-pooling, the PointNet-floor behaviour. That is the lower rail: too few slices and I am back
at one global summary. The upper rail is M→N, where each slice owns essentially one point, the tokens
*are* the points, and I have drifted back to attention-over-points with all its dilution. So M must sit
in the middle, and the softmax *sharpens* between the rails: the exponential pushes each point's
assignment to be low-entropy, so a point commits mostly to one slice and the slices are pressured to
specialize into distinct states rather than all collapsing to the domain-wide average.

Among the M tokens I keep *full softmax* attention — M is only 32, so M² = 1024 is trivial, and there is
no reason to cripple the most expressive operator with a linear approximation when the node count is
already tiny; the Galerkin trick was only ever needed to survive N, and N is gone. Then I broadcast each
transited token back to the points using the **same** slice weights. Tying the encode and decode weights
is not a convenience, it is what makes slice→attend→deslice a single change of variables: move into the
slice domain through the weight map, do the work there, come back through the same map. That tying is
forced by the integral-operator derivation — pushing the kernel through the (determinant-one) slice-domain
map reproduces the slice→token-attention→deslice sandwich term for term — so this is the *same* learnable
integral operator the graph methods approximated locally, now evaluated globally over learned states. The
graph rungs computed a local restriction of `∫κ(p,q)u(q)dq`; this computes a low-rank global one whose
rank is M.

Now I land this on *this task's* edit surface, and the harness exposes Transolver faithfully — this is
the rung where the baseline matches my derivation closely, so the differences are small but I still name
them. The task ships `layers.Physics_Attention.Physics_Attention_Irregular_Mesh` and I use it directly
rather than reimplementing the sublayer; my edit is the `Transolver_block` and the `Model` wrapper. The
block is the canonical pre-norm residual, `fx = Attn(LN(fx)) + fx; fx = mlp(LN(fx)) + fx`, with the last
block carrying a `LayerNorm`+`Linear` read-out head to `out_dim`. The `Model` preprocesses the
concatenated coordinates and features (`fun_dim + space_dim → n_hidden`), adds a small learned
`placeholder` bias to every point, runs `n_layers` blocks, and returns — and the forward signature takes
`geo` but **ignores it entirely**, because Physics-Attention needs no mesh graph at all. This is the one
rung that does *not* raise on `geo=None`; it simply never reads the edges, which is coherent because its
whole thesis is that location-based grouping (the graph) is the wrong grouping. The shipped
`Physics_Attention_Irregular_Mesh` carries the refinements from the derivation: a **two-stream** point
projection (one stream `x_mid` decides the slice, a separate stream `fx_mid` supplies the content
averaged into the token — the assignment criterion and the carried content need not be the same
feature), a **learnable per-head temperature** on the slice softmax initialized on the sharp side
(around 0.5, biasing toward decisive assignment from the start), an **orthogonally-initialized**
slice-projection so the M slice directions begin decorrelated and specialize faster, and the
mass-normalized tokens with the standard `dim_head^{-1/2}` scale on the token attention. There is a
structured-mesh path with kernel-3 conv projections, but on the design tasks `geotype='unstructured'`
selects the irregular-mesh class with plain linear projections, the geometry-general default these
benchmarks need.

Each of those three refinements earns its place, and I want to reason about why rather than inherit them
blindly. The **two-stream** projection decouples the *criterion* by which a point picks its slice from
the *content* it deposits there: `x_mid` answers "which physical state am I in?" while `fx_mid` answers
"what field value do I contribute?", and these need not be the same linear function of the input —
assignment cares about geometry-and-boundary regime, content cares about the solution value, and forcing
one projection to serve both would couple two jobs that a wide model can do better separately. The
**learnable per-head temperature** initialized around 0.5 (i.e. dividing the logits by ~0.5, doubling
their spread) biases the softmax toward decisive assignment from the first step, which matters because at
random init the slice logits are near-uniform and a plain softmax would hand every point a nearly flat
distribution over all 32 slices — the featureless, un-specialized start the M=1 collapse warned about;
starting sharp gives the slices a gradient toward specialization instead of a symmetric plateau to escape.
The **orthogonal initialization** of the slice-projection is the same anti-symmetry argument one level up:
if the 32 slice directions began correlated, many slices would score every point almost identically and
collapse into duplicates, wasting the token budget; orthogonal rows make the 32 directions maximally
decorrelated at init so each slice starts responsive to a distinct axis of the feature space and
specializes faster. The small learned `placeholder` bias added to every point before the blocks is a
minor thing — a shared learnable offset that gives the network a nonzero baseline to modulate — and I keep
it verbatim rather than reasoning it into significance.

Let me trace the shapes through the sublayer once, because a dimension check is the cheapest verification
before a run. With N=8000, n_hidden=256, n_heads=8 (so dim_head=32), and slice_num M=32: the input
`(1, 8000, 256)` projects per head to `x_mid` of shape `(1, 8, 8000, 32)`; the slice projection maps the
32-dim per-head feature to M=32 logits, `(1, 8, 8000, 32)`, and the slice-axis softmax leaves it a
partition of unity per point; the weighted sum over the 8000 points contracts to tokens `(1, 8, 32, 32)`
— the point axis is *gone*, replaced by the 32-slice axis, which is the entire cost win made concrete;
token attention is `(1,8,32,32)×(1,8,32,32)` → `(1,8,32,32)`, the trivial M² work; deslice contracts the
32 tokens back through the same `(1,8,8000,32)` weights to `(1,8,8000,32)`, and the heads recombine to
`(1,8000,256)`. Every contraction axis matches, and the point count N appears only in the O(NM) encode
and broadcast, never inside the attention — the shape trace *is* the linear-in-N claim.

The width is the one place this rung spends its budget aggressively, and it is forced by the canonical
setting and by the task's own budget definition. `CONFIG_OVERRIDES = {'n_hidden': 256, 'slice_num': 32}`
— 256 is double GraphSAGE's 128 and sixteen times the 16 of PointNet and the graph U-Net, and it is
literally the model against which the whole task's parameter budget is anchored (`budget_check.py`
rejects anything over 1.05× Transolver at n_hidden=256, slice_num=32). So unlike the earlier rungs, where
I noted width was a slack fidelity choice, here the width *is* the budget ceiling — I am running the very
configuration the cap is defined by, which sits at 1.00× of it by construction. The slice count M=32 is
the genuinely new hyperparameter, and the extremes I checked above argue for the middle: M=1 collapses to
global pooling, M→N fragments back into diluted point-attention, so a modest M with a wide (256) model
gives each of the 32 tokens ample capacity to be a clean state. Heads stay at 8, layers at 8, the
feed-forward at the `mlp_ratio` width — none of those is where the contribution lives.

It is worth seeing where the parameters actually go at 256, because it explains why this rung is the
budget anchor and the graph rungs were slack. Per block, the dominant cost is not the slicing but the
dense maps at full width: the feed-forward `MLP(256, 256·4, 256)` alone is `(256·1024+1024)+(1024·256+256)
≈ 526k` parameters, and the attention's q/k/v and output projections are each `Linear(256,256)` order
`65k`, so a single block is on the order of ~0.7M parameters and eight of them plus the preprocess and
read-out run into several million — the multi-million regime the earlier rungs sat at a few percent of.
By contrast the *new* machinery, the slice projection, is tiny: `Linear(dim_head=32, slice_num=32)` per
head is ~1k parameters, negligible beside the dense maps. So the slicing that does the conceptual work
costs almost nothing; the budget is spent on width, and width is what jumps from GraphSAGE's 128 to 256.
Doubling the width roughly quadruples each dense layer (both its input and output axes scale), which is
why 256 lands at the budget ceiling while 128 sat comfortably below — and it is the honest reason the cap
is defined against *this* configuration rather than a graph one.

What do I expect against GraphSAGE's measured numbers, falsifiably? The whole reason to switch to
whole-domain physics attention is the *global correlation* the graph methods could not carry, so the
metric I most expect to move is the **drag rank correlation** — Transolver should finally beat *both* the
U-Net's 0.981 and GraphSAGE's 0.978 on rho_d, because a single Physics-Attention layer relates the
front-region state and the wake state directly, with no diameter limit, which is exactly what ordering
designs by front-to-back pressure asymmetry needs. I also expect the **drag-magnitude error to drop below
GraphSAGE's 0.0193** and the **field errors to beat GraphSAGE's 0.089/0.033 on Car and 0.046/0.037 on
AirfRANS**, because the same global operator that fixes ordering also reconstructs the long-range field
structure that local message passing smeared, and at 256 width with 8 layers it has far more capacity.
The place I am least certain is the custom **AirCraft** probe, where the graph models hovered around
0.64/0.38 with no published reference; a wider attention model could overfit a small custom set as easily
as it could win, so I would watch whether AirCraft tracks the Car gains or diverges upward, and I would
not read it as a verdict either way. If Transolver beats GraphSAGE on rho_d, c_d, and all field errors on
the two published benchmarks, the trajectory's verdict is that the residual gap after the graph rungs was
indeed *global physical correlation*, and that grouping points by learned state rather than by location
is the operator that closes it — leaving, as the only remaining lever, whether the *slicing itself* can
be made sharper and its states more distinguishable than a single fixed-temperature softmax allows.

The causal chain, threaded from GraphSAGE's result: the graph trilogy resolved into "lossless local
message passing wins on fidelity (Car 0.089/0.033 down 21%/34% from the U-Net, AirfRANS 0.046/0.037 with
velocity a 2.7× recovery, c_d 0.0193 the lowest yet) but gives back 0.003 of global ordering (rho_d 0.978
< U-Net 0.981)," diagnosing the shared residual as *global correlation without diameter limit or
coarsening loss* → attention is the non-local integral operator that supplies exactly that, but over the
N points it is O(N²)=6.4×10⁷ and, even made linear by the Galerkin reassociation, still dilutes the
physics across N² location-pairs → so change the quadrature nodes from mesh points to **learned physical
states**: soft-assign each point across M slices by a per-point slice-softmax (a verified partition of
unity that collapses to global pooling at M=1 and to point-attention at M→N, sharpening in between),
encode each slice into a mass-normalized token, run full softmax attention among the M=32 tokens (M²=1024,
trivial), broadcast back through the *same* weights (a change of variables reproducing the integral
operator) → land it on this task's faithful Transolver via `Physics_Attention_Irregular_Mesh` (two-stream
projection, learnable per-head temperature ~0.5, orthogonal slice init), `geo` ignored, at the
budget-anchoring **n_hidden=256, slice_num=32** → expecting it to beat GraphSAGE on rho_d, c_d, and the
field errors on both published benchmarks, with AirCraft the one to watch, and the sharpness of the
slicing itself the only lever left. The full scaffold module is in the answer.
