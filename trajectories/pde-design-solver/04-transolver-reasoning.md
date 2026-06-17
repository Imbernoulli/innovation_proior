GraphSAGE landed almost exactly where I bet it would, and the pattern across the three graph rungs now
tells me what is left to fix. The lossless flat stack reclaimed field accuracy on every front: Car
pressure relative-L2 fell to 0.089 and velocity to 0.033 — the best of any rung so far, well below
PointNet's 0.106/0.042 and the graph U-Net's 0.113/0.050 — and AirfRANS recovered exactly the way I
predicted the bluntest case would, pressure dropping to 0.046 and velocity to 0.037 (from the U-Net's
blown-out 0.064/0.099). The drag-magnitude error improved too, c_d 0.0193, the lowest yet — a sharper
lossless pressure field integrates to a more accurate coefficient. And the one place I said SAGE might
*lose* is exactly where it did: rho_d came in at 0.978, a hair *below* the graph U-Net's 0.981. So the
graph trilogy resolved cleanly into a single statement: lossless local message passing dominates the
lossy hierarchy on fidelity, but it gives a little of the *global ordering* back, because a flat stack
reaches only `n_layers` hops and cannot cheaply carry front-to-back correlation across the whole body.
That residual — global correlation without diameter limit and without coarsening loss — is the gap
every graph method shares, and it is what the next rung has to close.

So the question is direct: how do I model correlations across the *entire* domain in a single layer,
on an unstructured mesh, without the O(N²) cost that would make it infeasible at ~5000–10000 points? I
want to step out of the message-passing family entirely. Message passing is local by construction; even
GraphSAGE's full-neighborhood aggregation only reaches one hop per layer, and stacking hops both
over-smooths and still caps reach at the depth. The operator-learning frame says each layer should be a
non-local integral operator, and the most expressive parameterization of that integral is attention —
softmax attention is exactly a Monte-Carlo discretization of the integral operator with a *learned*
kernel and the mesh points as quadrature nodes. That is precisely the global, learnable, geometry-
agnostic operator I want. But with the N mesh points as the quadrature nodes the cost is O(N²), and
even made linear (the Galerkin reassociation Q(KᵀV)) the attention is still computed *over the points
themselves* — and that is the deeper problem the graph rungs also have in a different guise: the
informative physical correlations get diluted across a sea of low-level point-to-point relations. So
"cheaper attention over points" is not the answer; the *nodes* are wrong.

Here is the reframing that breaks the deadlock. Mesh points are an artifact of discretization — a
finite, arbitrary sampling of an underlying continuous physics. The physics does not live at the
points; it lives in the *states*. On a car the windshield, the license plate, and the headlights are
all in the same front regime that governs drag, even though they are scattered across the surface; and
two spatially adjacent points can be in completely different states. This is the same observation that
made the graph methods leave performance on the table: a radius graph or a mesh edge groups points by
*location*, but the correlations I need to model group by *physical state*, which spans the domain
non-locally. So the move is: group points by what physical state they are in — learned from data, the
groups free to be any shape and to span the whole body — encode each group into a single token, run
attention among the few tokens, and broadcast back. If there are M such groups (slices), the attention
is O(M²), the encode and broadcast are O(NM), and with M a small constant the whole operator is linear
in N while the attention — the part doing the global correlation — runs over M *meaningful* nodes
instead of N noisy ones. Both problems, the cost and the dilution, fall to the same change of
quadrature nodes.

Making "assign each point to a slice" learnable means making it soft and differentiable. For each
point's per-head feature I project to M slice logits and take a **softmax over the slice axis**, so
each point gets a distribution over the M slices that sums to one — a partition of unity, and
crucially the softmax *sharpens*: the exponential pushes the assignment to be low-entropy so a point
commits mostly to one slice and the slices are pressured to specialize into distinct states rather than
all collapsing to the domain-wide average. A slice's token is then the **mass-normalized weighted
mean** of its members' features (divide the weighted sum by the slice's total weight, so a slice that
happens to own many points does not get an artificially large token). Among the M tokens I keep *full
softmax* attention — M is only 32, so M² is trivial, and there is no reason to cripple the most
expressive operator with a linear approximation when the node count is already tiny. Then I broadcast
each transited token back to the points using the **same** slice weights — and tying the encode and
decode weights is not a convenience, it is what makes slice-attention-deslice a single change of
variables: move into the slice domain, do the work there, come back through the same map. That tying is
forced by the integral-operator derivation: pushing G through the determinant-one slice-domain map
reproduces the slice→token-attention→deslice sandwich term for term, so this is the same learnable
integral operator the graph methods approximated locally, now evaluated globally over learned states.

Now I have to land this in *this task's* edit surface, and the harness exposes Transolver faithfully —
this is the case where the baseline matches the paper closely, so the differences are small but I
should still name them. The task ships `layers.Physics_Attention.Physics_Attention_Irregular_Mesh` and
I use it directly rather than reimplementing the sublayer; my edit is the `Transolver_block` and the
`Model` wrapper. The block is the canonical pre-norm residual: `fx = Attn(LN(fx)) + fx; fx =
mlp(LN(fx)) + fx`, with the last block carrying a `LayerNorm`+`Linear` read-out head to `out_dim`. The
`Model` preprocesses the concatenated coordinates and features (`fun_dim + space_dim → n_hidden`), adds
a small learned `placeholder` bias to every point, runs `n_layers` blocks, and returns — the forward
signature takes `geo` but **ignores it entirely**, because Physics-Attention needs no mesh graph at
all (this is the one rung that does not raise on `geo=None`; it simply never reads the edges). The
harness's `Physics_Attention_Irregular_Mesh` carries the refinements from the derivation: the two-
stream point projection (one stream `x_mid` decides the slice, a separate stream `fx_mid` supplies the
content that gets averaged into the token — the assignment criterion and the carried content need not
be the same feature), a **learnable per-head temperature** on the slice softmax initialized on the
sharp side (around 0.5), an **orthogonally-initialized** slice-projection layer so the M slice
directions start decorrelated and specialize faster, and the mass-normalized tokens with the standard
`dim_head^{-1/2}` scale on the token attention. There is a structured-mesh path with kernel-3 conv
projections, but on the unstructured design tasks `geotype='unstructured'` selects the irregular-mesh
class with plain linear projections, which is the geometry-general default these benchmarks need.

The width is the one place this rung spends its budget aggressively and it is forced by the paper-
faithful setting: `CONFIG_OVERRIDES = {'n_hidden': 256, 'slice_num': 32}` — 256 is double GraphSAGE's
128 and sixteen times the 16 of PointNet/Graph_UNet, and it is the model against which the whole task's
parameter budget is defined (1.05× Transolver-256). The slice count M=32 is the genuinely new
hyperparameter; I keep it at the paper setting because the extremes argue for the middle: M=1 collapses
to global pooling (attention over a single token is the identity, all correlations lost), while pushing
M toward N fragments the physics into noisy slivers and drifts back to attention-over-points. With a
wide model (256) M=32 already gives each token enough capacity. Heads stay at the canonical 8, layers
at 8, the feed-forward at the `mlp_ratio` width — none of those is where the contribution lives.

What do I expect against GraphSAGE's measured numbers, falsifiably? The whole reason to switch to
whole-domain physics attention is the *global correlation* the graph methods could not carry, so the
metric I most expect to move is the **drag rank correlation** — Transolver should finally beat *both*
the graph U-Net's 0.981 and GraphSAGE's 0.978 on rho_d, because a single Physics-Attention layer
relates the front-region state and the wake state directly, with no diameter limit, which is exactly
what ordering designs by front-to-back pressure asymmetry needs. I also expect the **drag-magnitude
error to drop below GraphSAGE's 0.0193** and the **field errors to beat GraphSAGE's 0.089/0.033 on Car
and 0.046/0.037 on AirfRANS**, because the same global operator that fixes ordering also reconstructs
the field's long-range structure that local message passing smeared — and at 256 width with 8 layers it
has far more capacity. The place I am least certain is the custom **AirCraft** probe, where the graph
models hovered around 0.64/0.38 and there is no published reference; a wider attention model could
overfit a small custom set as easily as it could win, so I would watch whether AirCraft tracks the Car
gains or diverges. If Transolver beats GraphSAGE on rho_d, c_d, and all field errors on the two
published benchmarks, the trajectory's verdict is that the residual gap after the graph rungs was
indeed *global physical correlation*, and that grouping points by learned state rather than by location
is the operator that closes it — leaving, as the only remaining lever, whether the *slicing itself* can
be made sharper and more distinguishable than a fixed-temperature softmax allows.

The causal chain, threaded from GraphSAGE's result: the graph trilogy resolved into "lossless local
message passing wins on fidelity (Car 0.089/0.033, AirfRANS 0.046/0.037, c_d 0.0193) but gives back a
little global ordering (rho_d 0.978 < U-Net 0.981)," diagnosing the shared residual as *global
correlation without diameter limit or coarsening loss* → attention is the non-local integral operator
that supplies exactly that, but over the N points it is O(N²) and dilutes the physics → so change the
quadrature nodes from mesh points to **learned physical states**: soft-assign each point across M
slices by a per-point slice-softmax (partition of unity, sharpening), encode each slice into a
mass-normalized token, run full softmax attention among the M tokens, broadcast back through the *same*
weights (a change of variables that reproduces the integral operator) → land it on this task's
faithful Transolver via `Physics_Attention_Irregular_Mesh` (two-stream projection, learnable per-head
temperature, orthogonal slice init), `geo` ignored, at **n_hidden=256, slice_num=32** → expecting it to
beat GraphSAGE on rho_d, c_d, and the field errors on both published benchmarks, with AirCraft the
one to watch, and the sharpness of the slicing itself the only lever left. The full scaffold module is
in the answer.
