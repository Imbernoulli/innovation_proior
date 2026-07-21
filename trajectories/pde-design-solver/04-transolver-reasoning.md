GraphSAGE landed almost exactly where I bet it would, and the pattern across the three graph rungs now
tells me what is left to fix. The lossless flat stack reclaimed field accuracy on every front. On Car,
pressure relative-L2 fell 0.113 → 0.089 (down 21% from the U-Net, and 16% from PointNet) and velocity
0.050 → 0.033 (down 34% from the U-Net, below PointNet's 0.042 too) — the best fields of any rung so
far. AirfRANS recovered the way I predicted the bluntest case would: pressure 0.064 → 0.046 (down 28%)
and velocity 0.099 → 0.037, a 2.7× improvement that essentially undoes the U-Net's blow-up. The
drag-magnitude error improved too, c_d 0.0193, down 17% from the U-Net and 22% from PointNet — the
lowest yet, exactly as the "a sharper lossless pressure field integrates to a truer coefficient"
argument predicted, with pressure-L2 and c_d moving together as two functionals of the same surface
field should. And the one place I said SAGE might *lose* is where it did: rho_d came in at 0.978, a
hair below the U-Net's 0.981 — a give-back of 0.003, tiny but in the predicted direction, because a
flat stack reaches only L hops and cannot carry front-to-back correlation across a ~90-hop body.

So the graph trilogy resolves into a single statement. Lossless local message passing dominates the
lossy hierarchy on fidelity — every field error and c_d improved — but it gives a little of the *global
ordering* back, because every message-passing stack, flat or hierarchical, is local by construction and
reaches only as far as its depth (or, for the U-Net, buys reach only by paying coarsening loss). That
residual — global correlation with no diameter limit *and* no coarsening loss — is the gap all three
graph methods share in different guises, and it is what this rung has to close. The question is direct:
how do I model correlations across the *entire* domain in a single layer, on an unstructured mesh of
~5000–10000 points, without a cost that makes it infeasible?

I have to step out of the message-passing family entirely, because the limit is structural. The
operator-learning frame says each layer should be a non-local integral operator,
`(Ku)(p) = ∫ κ(p,q) u(q) dq`, and message passing is the crude approximation that restricts the kernel
`κ` to a local neighborhood and a fixed (mean, or weighted-mean) shape. The most *expressive*
parameterization of that integral is attention: softmax attention is exactly a Monte-Carlo
discretization of the integral operator with a *learned* kernel and the mesh points as quadrature
nodes, `κ(p,q) = softmax_q(⟨Wq_p, Wk_q⟩)`. That is the global, learnable, geometry-agnostic operator I
want — one layer relates every point to every other, no diameter limit. But with the N mesh points as
quadrature nodes the cost is O(N²): at N=8000, 6.4×10⁷ point-pairs per layer per head, the wall that
made attention-over-points infeasible at this scale in the first place.

Two live ways around that wall, and I should walk the tempting one before rejecting it. The first is to
keep the quadrature nodes as the N points but make the attention *linear* — the Galerkin reassociation
that computes `Q(KᵀV)` instead of `(QKᵀ)V`, dropping the softmax so the O(N²) score matrix never forms
and the cost falls to O(N·C²). That makes point-attention affordable but does not fix the *deeper*
problem the graph rungs already exposed. Attention over the N points — linear or not — spreads its
modeling capacity across N² point-to-point relations, the overwhelming majority low-level and
uninformative: two points a centimetre apart on the same smooth panel relate trivially, and the
*informative* physical correlations (the front region governing the wake) are a tiny, non-local
fraction diluted in that sea. This is the same observation that made the graph methods leave
performance on the table — a radius graph or a mesh edge groups points by *location*, but the
correlations I need group by *physical state*, which spans the domain non-locally. So "cheaper attention
over points" treats the cost symptom and leaves the dilution disease: the quadrature *nodes* are wrong,
and making them cheaper does not make them right.

The second way breaks the deadlock. Mesh points are an artifact of discretization — a finite, arbitrary
sampling of an underlying continuous physics. The physics does not live at the points; it lives in the
*states*. On a car the windshield, the license plate, and the headlights are all in the same
front-stagnation regime that governs drag, though scattered across the surface; and two spatially
adjacent points can sit in completely different states across a separation line. So group points by
*what physical state they are in* — learned from data, the groups free to be any shape and to span the
whole body — encode each group into a single token, run attention among the few tokens, and broadcast
back. If there are M such groups (slices), the encode and broadcast are O(NM) and the attention is
O(M²), so with M a small constant the operator is linear in N *and* the attention runs over M
*meaningful* nodes instead of N noisy ones. Both problems, cost and dilution, fall to the same change of
quadrature nodes. Beside the point-attention wall: at N=8000 and M=32, the encode/broadcast is N·M =
2.56×10⁵ and the token attention is M² = 1024, against 6.4×10⁷ — a ~250× reduction on the encode side and
the expensive quadratic now running over 32 nodes, not 8000.

Making "assign each point to a slice" learnable means making it soft and differentiable. For each
point's per-head feature I project to M slice logits and take a **softmax over the slice axis**, so each
point gets a distribution over the M slices that sums to one — a convex partition, `Σ_g w_ng = 1`, none
of the point's mass created or destroyed. A slice's token is the **mass-normalized weighted mean** of
its members, `t_g = Σ_n w_ng x_n / Σ_n w_ng`, so a slice that owns many points does not get an
artificially large token — it gets the *average* state of its members. Two limit checks pin the
construction between its rails. If M=1, every point's single slice weight is forced to 1, the one token
is the plain mean of all point features, attention over a single token is the identity, and the
broadcast hands every point that same mean — i.e. M=1 collapses exactly to global mean-pooling, the
PointNet-floor behaviour. The other rail is M→N, where each slice owns essentially one point, the tokens
*are* the points, and I have drifted back to attention-over-points with all its dilution. So M must sit
in the middle, and the softmax *sharpens* between the rails: the exponential pushes each point's
assignment low-entropy, so a point commits mostly to one slice and the slices are pressured to
specialize into distinct states rather than collapsing to the domain-wide average.

Among the M tokens I keep *full softmax* attention — M is only 32, so M² = 1024 is trivial, and there is
no reason to cripple the most expressive operator with a linear approximation when the node count is
tiny; the Galerkin trick was only ever needed to survive N, and N is gone. Then I broadcast each
transited token back to the points using the **same** slice weights. Tying the encode and decode
weights is not a convenience: it is what makes slice→attend→deslice a single change of variables — move
into the slice domain through the weight map, do the work there, come back through the same map. That
tying is forced by the integral-operator derivation (pushing the kernel through the determinant-one
slice-domain map reproduces the slice→token-attention→deslice sandwich term for term), so this is the
*same* learnable integral operator the graph methods approximated locally, now evaluated globally over
learned states — a low-rank global one whose rank is M.

Now to land this on the edit surface, where the task exposes Transolver faithfully — the one rung where
the baseline matches my derivation closely. The task ships
`layers.Physics_Attention.Physics_Attention_Irregular_Mesh` and I use it directly; my edit is the
`Transolver_block` and the `Model` wrapper. The block is the canonical pre-norm residual,
`fx = Attn(LN(fx)) + fx; fx = mlp(LN(fx)) + fx`, with the last block carrying a `LayerNorm`+`Linear`
read-out to `out_dim`. The `Model` preprocesses the concatenated coordinates and features
(`fun_dim + space_dim → n_hidden`), adds a small learned `placeholder` bias to every point, runs
`n_layers` blocks, and returns — and the forward signature takes `geo` but **ignores it entirely**,
because Physics-Attention needs no mesh graph. This is the one rung that does *not* raise on `geo=None`;
it never reads the edges, which is coherent because its whole thesis is that location-based grouping is
the wrong grouping. The shipped `Physics_Attention_Irregular_Mesh` carries three refinements, each of
which earns its place. The **two-stream** point projection decouples the *criterion* by which a point
picks its slice (`x_mid`, "which physical state am I in?") from the *content* it deposits there
(`fx_mid`, "what field value do I contribute?"); assignment cares about geometry-and-boundary regime,
content cares about the solution value, and forcing one projection to serve both would couple two jobs a
wide model does better separately. The **learnable per-head temperature** initialized around 0.5 (i.e.
dividing the logits by ~0.5, doubling their spread) biases the softmax toward decisive assignment from
the first step, which matters because at random init the slice logits are near-uniform and a plain
softmax would hand every point a nearly flat distribution over all 32 slices — the featureless start the
M=1 collapse warned about; starting sharp gives the slices a gradient toward specialization instead of a
symmetric plateau to escape. The **orthogonal initialization** of the slice-projection is the same
anti-symmetry argument one level up: correlated slice directions would score every point almost
identically and collapse into duplicates, wasting the token budget; orthogonal rows start the 32
directions maximally decorrelated so each specializes faster. The small learned `placeholder` bias is a
shared learnable offset giving the network a nonzero baseline to modulate, which I keep verbatim. There
is a structured-mesh path with kernel-3 conv projections, but `geotype='unstructured'` selects the
irregular-mesh class with plain linear projections, the geometry-general default these benchmarks need.

The sublayer shapes, once, at N=8000, n_hidden=256, n_heads=8 (dim_head=32), slice_num M=32: the input
`(1,8000,256)` projects per head to `x_mid` `(1,8,8000,32)`; the slice projection maps to M=32 logits
and the slice-axis softmax leaves a partition of unity per point; the weighted sum over the 8000 points
contracts to tokens `(1,8,32,32)` — the point axis *gone*, replaced by the 32-slice axis, which is the
cost win made concrete; token attention is `(1,8,32,32)` throughout, the trivial M² work; deslice
contracts the 32 tokens back through the same `(1,8,8000,32)` weights, and heads recombine to
`(1,8000,256)`. N appears only in the O(NM) encode and broadcast, never inside the attention.

The width is where this rung spends its budget aggressively, forced by the canonical setting and the
task's own budget definition. `CONFIG_OVERRIDES = {'n_hidden': 256, 'slice_num': 32}` — 256 is double
GraphSAGE's 128 and sixteen times PointNet's and the graph U-Net's 16, and it is literally the model the
whole task's parameter budget is anchored to (`budget_check.py` rejects anything over 1.05× Transolver at
n_hidden=256, slice_num=32). So unlike the earlier rungs, where width was slack, here the width *is* the
budget ceiling — I run the very configuration the cap is defined by, sitting at 1.00× of it by
construction. The slice count M=32 is the genuinely new hyperparameter, and the limits above argue for
the middle: M=1 collapses to global pooling, M→N fragments into diluted point-attention, so a modest M
with a wide model gives each of the 32 tokens ample capacity to be a clean state. Where the parameters
actually go, at 256, explains why this rung is the anchor and the graph rungs were slack: per block the
feed-forward `MLP(256, 256·4, 256)` alone is ≈526k parameters and the q/k/v/output projections are each
`Linear(256,256)` ≈65k, so a single block is ~0.7M and eight of them plus preprocess and read-out run
into several million — the multi-million regime the earlier rungs sat at a few percent of. The *new*
machinery, the slice projection `Linear(32, 32)` per head, is ~1k, negligible; the slicing that does the
conceptual work costs almost nothing, and the budget is spent on width, which is what jumps from
GraphSAGE's 128 to 256. Doubling the width roughly quadruples each dense layer (both axes scale), which
is the honest reason the cap is defined against *this* configuration rather than a graph one.

Falsifiably, against GraphSAGE's numbers: the whole reason to switch to whole-domain physics attention
is the *global correlation* the graph methods could not carry, so the metric I most expect to move is
the **drag rank correlation** — Transolver should finally beat *both* the U-Net's 0.981 and GraphSAGE's
0.978 on rho_d, because a single layer relates the front-region state and the wake state directly with no
diameter limit. I also expect the **drag-magnitude error to drop below GraphSAGE's 0.0193** and the
**field errors to beat GraphSAGE's 0.089/0.033 on Car and 0.046/0.037 on AirfRANS**, because the same
global operator that fixes ordering also reconstructs the long-range field structure local message
passing smeared, and at 256 width with 8 layers it has far more capacity. The place I am least certain is
the custom **AirCraft** probe, where the graph models hovered around 0.64/0.38 with no published
reference; a wider attention model could overfit a small custom set as easily as it could win, so I watch
whether AirCraft tracks the Car gains or diverges upward and do not read it as a verdict either way. If
Transolver beats GraphSAGE on rho_d, c_d, and all field errors on the two published benchmarks, the
verdict is that the residual gap after the graph rungs was indeed *global physical correlation*, and that
grouping points by learned state rather than by location is the operator that closes it — leaving, as the
only remaining lever, whether the *slicing itself* can be made sharper and its states more
distinguishable than a single fixed-temperature softmax allows. The full module is in the answer.
