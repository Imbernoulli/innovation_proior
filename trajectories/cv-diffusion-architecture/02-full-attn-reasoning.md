The convolution-only floor told me where the quality is leaking, and it told me in FID. No-attn
came in at 21.39 small, 11.25 medium, 9.55 large. Read the shape of that: it is competent — the
samples are recognizable and locally clean, which is exactly what I expected from a pure
convolutional denoiser, because convolution gets the local statistics right and the skip
connections preserve the fine detail. But it is the floor, and the gap between the scales is the
tell. At the small scale the FID is far worse (21.39) than at large (9.55); as I pour in capacity
the network closes a lot of that gap, but even at the large scale, with ~140M parameters, a
purely-convolutional denoiser whose only attention is the single free layer at the 4×4 bottleneck
is leaving something on the table. What it is structurally worst at is the thing it has no direct
mechanism for: coordinating two regions of the image that are far apart in space. That has to be
carried indirectly through the long chain of conv-and-downsample layers, and on this budget that
indirect path is where coherence leaks. So the diagnosis is not "the model is too small" — at large
scale it has plenty of capacity — it is "the model has no direct way to couple distant positions."
That is an architecture problem, and it points straight at the operation no-attn deliberately
omitted: per-resolution self-attention.

Let me be precise about *why* convolution fails at the long range, because the precision is what
tells me what to add. A 3×3 kernel mixes a 3×3 neighborhood and nothing else. One conv layer can
only relate a position to its immediate neighbors; for two positions thirty pixels apart to
influence each other, their receptive fields have to overlap, which only happens after the signal
has passed through enough conv and downsampling layers for the fields to grow that wide. So a
long-range dependency in no-attn is never expressed in one place — it is smeared across a long chain
of local operations, each tuned in concert with the others for the global relationship to come out
right. That is three problems stacked: representationally, a shallow part of the net cannot reach
across the image at all; optimization-wise, getting many layers to cooperate to carry one
coordinate-far-from-coordinate signal is hard credit assignment; and statistically, a dependency
that exists only as a fragile conspiracy of many layers tends to break. Local statistics do not have
this problem because they live inside a single kernel — which is exactly why no-attn's *local* output
is fine and its *global* coherence is what trails.

The obvious brute-force patch is bigger kernels — a 7×7 or 11×11 reaches further per layer. But that
is the wrong trade twice over. It buys reach at the price of the very thing that makes convolution
efficient: a k×k kernel costs k² per position, most of it spent on a fixed grid of nearby positions
I did not need to relate. And it still imposes a *fixed-shape* neighborhood when the thing I actually
want to relate — "this part of the object" to "that part" — is two regions whose relative position is
content-dependent, not a fixed offset. Bigger kernels give a bigger fixed window; I want a
content-addressed one. So enlarging the receptive field by force is not the move.

What I actually want as an operation: the response at output location i should be allowed to depend
on the features at *every* location j, in a single step, with the amount it depends on j decided by
how *relevant* j is to i, not by how close. That is a weighted sum over all positions, weighted by a
learned, content-dependent relevance — and that is self-attention. Make it concrete on a feature map
`x ∈ ℝ^{C×N}` with N = H·W spatial locations. Three per-position linear maps — query, key, value —
each a 1×1 convolution, which applies the same linear transform to the channel vector at every
location with no spatial mixing (perfect, because I want the spatial mixing done by the attention,
not the projections). The affinity logit from query i to key j is `qᵢᵀkⱼ/√d`, the attention weight
is the softmax over all keys j, and the attended feature at i is `Σⱼ aᵢⱼ vⱼ` — for every output
location, a content-weighted average over all input locations. One layer, every-to-every,
content-addressed. The `√d` scaling is not cosmetic: the logit is a dot product of d roughly-unit
components, so it has variance d and typical magnitude √d; feed magnitude-√d logits to a softmax and
it saturates onto one key with a tiny Jacobian, and the gradients to the projections vanish.
Dividing by √d keeps the logit variance O(1) and the softmax responsive. And the block goes in as a
residual — output = input + attention — initialized so it is identity at the start, so dropping it
into the working convolutional UNet cannot hurt and it grows its influence only as it helps.

That is the operation no-attn omitted at every per-resolution stage. The question this rung answers
is: where do I put it back? And here the binding constraint reasserts itself — attention is O(N²) in
the number of spatial positions. At the 32×32 feature map N = 1024 and N² ≈ 10⁶ per attention layer;
at 16×16, N = 256; at 8×8, N = 64; at 4×4, N = 16. So the placement of attention across the
resolution levels is itself the design space, and it has a cost gradient running from very expensive
at the fine maps to nearly free at the coarse ones. There is a whole ladder of placements between
no-attn's "attention nowhere per-resolution" and the opposite extreme.

Why take the *maximal* extreme — self-attention at *every* resolution, 32, 16, 8, and 4 — as this
rung rather than a careful middle? Because of what no-attn's numbers actually said. The failure was
long-range coherence, and I do not yet know at *which* scale the missing coordination matters most.
On a 32×32 image the long-range structure is everywhere: the gross layout is a coarse-scale relation,
but on an image this small even "the two ends of an object" is a relation that lives at the finer
feature maps too, because 32×32 is barely downsampled. No-attn proved that leaving *all* the
per-resolution coordination to the convolutional stack costs FID; the cleanest, most expressive
response — the one that asks "what if every feature map, coarse and fine, can see globally?" — is to
put an attention-bearing block at every single resolution level. It is the maximal statement of the
cure, and it directly tests whether unrationed global mixing recovers the coherence no-attn lost.
The cost is real: I pay the O(N²) price at the 32×32 and 16×16 maps too, which is exactly where it
is most expensive, so this is the slowest, most memory-hungry architecture on the ladder. But as a
*hypothesis test* it is the right one to run second, because it brackets the design space from above
— if attention everywhere does not help, then attention is not the answer; if it helps a lot but
costs too much, that motivates finding the cheaper subset that captures most of the gain.

In this scaffold the move is a one-tuple edit. Where no-attn used `("DownBlock2D",)×4` and
`("UpBlock2D",)×4`, full-attn uses `("AttnDownBlock2D",)×4` and `("AttnUpBlock2D",)×4` — the
attention-bearing block at every resolution level. The attention-bearing UNet blocks already do the
processor mechanics I derived: after each convolutional residual block, group-norm the feature map,
flatten the H·W spatial positions to tokens, project to query/key/value, compute the scaled
dot-product softmax over key positions, average the values, project back, and add to the residual
stream. The `attention_head_dim` knob sets the per-head width — at the `UNet2DModel` default of 8,
a C-channel map is split into C/8 heads of width 8, and the `1/√d` rule follows that per-head width.
Multi-head matters here for the same reason it matters anywhere: a single head forces one relevance
pattern, while splitting into heads lets different heads specialize to different relations at once
(the symmetric half, the far edge, a repeated region) at nearly the cost of one full-width attention.
Everything else stays the shared DDPM configuration — the residual blocks, group norm, sinusoidal
timestep embedding, channel schedule, and the fixed loss/sampler/optimizer are untouched, so the
only thing that changed from rung one is *where attention lives*. That is what keeps the comparison
honest: any FID change is attributable to attention placement and nothing else (the full scaffold
module is in the answer).

So the delta from no-attn is exactly the addition I argued for: where rung one had pure convolution
at every per-resolution stage and relied on the indirect conv path plus one free bottleneck
attention for all global coordination, this rung gives every feature map — coarse and fine — a
direct, content-addressed, every-to-every coupling. Reading no-attn's numbers, here is what I expect
and where I am unsure. Full-attn should beat no-attn, and the win should be cleanest where no-attn's
deficit was about coherence rather than capacity. The small scale is the most interesting: no-attn
was worst there (21.39), and attention is a parameter-efficient way to add coordinating power without
adding much width, so I expect the biggest *relative* improvement at small — attention buying global
coherence that a thin convolutional model could not afford to grow. At medium and large, no-attn was
already much better (11.25, 9.55) because capacity partly substituted for the missing coordination,
so the headroom for attention is smaller, but I still expect a gain because direct coupling is
strictly more than indirect coupling. The risk I am watching is the cost: attention at 32×32 is the
most expensive thing on the ladder, and if the global structure to coordinate at the finest maps is
thin — if CIFAR's long-range relations actually live mostly at the coarser feature resolutions —
then the 32×32 and 16×16 attention layers are paying a heavy quadratic price for little FID, and a
cheaper, more targeted placement would match this rung's quality at a fraction of the compute. If
that is what the numbers show — full-attn beats no-attn but the gain is concentrated at coarser
scales and the fine-resolution attention is mostly idle — then the next move is already named: keep
attention only where it earns its cost, at the middle resolution, and drop it from the fine maps
where it is expensive and the global structure is thin.

The causal chain in one breath: no-attn's 21.39/11.25/9.55 is a long-range-coherence failure, not a
capacity failure, because even at 140M params the purely-convolutional denoiser has no direct way to
couple distant positions and must smear every long-range dependency across a fragile chain of local
layers → the cure is self-attention, the content-addressed every-to-every operation convolution
cannot do, implemented as 1×1 query/key/value projections, a `√d`-scaled softmax over all positions,
and a value average wrapped in an identity-at-init residual → the binding O(N²) cost makes attention
*placement* across the four resolution levels the design space → the maximal placement, attention at
every resolution (32/16/8/4), is the right second rung because it brackets the cure from above and
tests whether unrationed global mixing recovers no-attn's lost coherence → expecting the biggest
relative win at the small scale where coherence, not capacity, was the bottleneck, and watching
whether the expensive fine-resolution attention is idle, which would name the cheaper placement to
try next.
