Concat-FiLM came in at 19.39 / 11.50 / 10.52 FID, below the floor's 20.01 / 12.31 / 11.77 at every scale,
and that settles the question I posed at step 2. The floor's bottleneck was the *route*, not the affine:
moving the same family of conditioning from a freshly-inserted post-block conditioner onto the timestep's
own tuned block-internal AdaGN — and in the process dropping from the full scale/shift/gate adaptive norm
down to its additive-bias corner — *improved* FID. So the cheaper, simpler operator won, precisely because
it rode machinery already tuned for a global per-channel signal instead of climbing a new sublayer off a
zero gate. The Large gain (11.77 → 10.52) is real but modest; the Small gain (20.01 → 19.39) is smaller in
*relative* terms than I'd hoped, which is the tell I want to read carefully. Both adanorm and concat-FiLM
are, at bottom, the *same operator* — a global, per-channel affine drawn from the class label, applied
identically at every spatial location and blind to what is currently forming in the feature map. One put it
after the block, one put it inside; the better-routed one is now my strongest result. But the family has a
ceiling, and the residual FID — especially that stubborn Small number that barely moved — looks like the
operator's *structural* limit, not a routing artifact. Concat-FiLM exhausted the cheap road. To go further
I have to change the operator itself.

Let me state the affine's limit precisely, because it points directly at what to build. An adaptive affine
gives each channel one scale and one shift, computed from the label, applied at *every* spatial position
identically. So the class can say "globally turn channel 37 up, channel 12 down." It cannot say "near this
patch of the feature map emphasize this; over there emphasize that," because the modulation has no spatial
address. And it never *looks at* the local content of the feature it modulates — the scale and shift come
from the label alone, not from $h$. For the timestep, that is the right bandwidth: the noise level genuinely
is a single global property of the whole image, so a content-blind global knob fits it perfectly. But "which
of ten classes" wants to act *differently on different parts of the picture, depending on what is already
forming there* — a horse's legs and a horse's head are different local computations, and a global per-channel
gain cannot distinguish them. That is the wall concat-FiLM hit, and it is structural: the affine is
low-bandwidth, spatially uniform, content-agnostic. No amount of re-routing fixes it; I need an operator
that conditions each spatial position *as a function of its own content*.

The strongest class-conditional pixel models confirm the diagnosis by how they had to work around it. ADM's
in-network conditioning is adaptive group norm — exactly the affine I have been using — and to get *strong*
class adherence it bolted on classifier guidance: a *separate* classifier on noised images whose gradient
nudges the sampler toward the class at every step. That is a confession that the in-network affine alone was
too low-bandwidth for "which class," and the fix lived outside the denoiser, at sampling time, at the cost
of a whole second model. I don't want that — the task fixes the sampler and forbids changing it, and I want
the denoiser *itself* to carry strong, content-dependent conditioning. So the affine family is out as the
*strong* in-network operator, and guidance-as-a-crutch is out as the cost.

Let me restate the requirement crisply, because it points somewhere specific. I want an operator where each
spatial position of the feature map can be conditioned *differently*, as a function of its own content. "A
position reads from the conditioning, with the read depending on what the position currently holds" is the
description of an attention layer. Pull up the mechanism: $\mathrm{Attention}(Q,K,V) = \mathrm{softmax}(QK^\top/\sqrt{d})\,V$.
Queries, keys, values; each query dots with every key, a softmax over those scores weights the values, the
output is the weighted sum. The crucial freedom is that queries and keys/values can come from *different*
places — cross-attention. Map it onto my problem: the queries should be the *image feature positions* — I
want each spatial location to do the reading — and the keys and values should be the *condition*. Flatten
the $H\times W$ feature map into $N = HW$ query tokens, let the class embedding source the keys and values;
the output has one row per query, i.e. one per spatial position, so it comes back shaped like a feature map
and slots straight into the convolutional stream. This is genuinely different from the affine: in FiLM the
output at position $p$ is $(1+\text{scale}_c)\,h_p + \text{shift}_c$, where the modulation depends only on
the class $c$; in attention the output at $p$ is $\sum_j \mathrm{softmax}_j(q_p\cdot k_j)\,v_j$ with $q_p$ a
projection of $h_p$, so the *weights* depend on the position's own content. That is exactly the spatially
varying, content-dependent conditioning the affine lacked.

Now build it carefully, because attention has details that bite. First the projections: the feature map has
$C$ channels, the class embedding lives at `cond_dim`, so three linear maps — $W_Q$ from the $C$-dim feature
tokens, $W_K, W_V$ from the condition — give learned separate views for the three roles, with output width
$C$ so the result adds back cleanly. Second the softmax scale, $1/\sqrt{d}$, which I want from first
principles, not by copying. If query and key components are roughly independent, zero-mean, unit-variance,
a single logit $q\cdot k = \sum_{i=1}^{d} q_i k_i$ has mean 0 and variance $d$, so the logits spread like
$\sqrt{d}$; push them through softmax and it saturates toward one-hot, and where softmax saturates its
gradient nearly vanishes — the layer would barely learn. Divide each logit by $\sqrt{d_{\text{head}}}$ and
the variance returns to 1 regardless of width, keeping softmax soft and gradients alive. Third, heads: one
head must compress everything a position might read into a single weighting; multiple heads project into
several subspaces, attend in parallel, concatenate, and project back, so different heads read different
aspects of the condition at once instead of averaging them — with per-head width $C/\text{num\_heads}$ so
total cost stays about one full-width head. Four heads is fine at this scale. Fourth, normalization and
placement: the UNet uses GroupNorm everywhere and already has self-attention sublayers at low resolution, so
I GroupNorm the feature map before projecting queries (matching the backbone and keeping query magnitudes
controlled), and insert one such conditioning block after each down/mid/up block so the class is re-read at
every scale.

The fifth detail is the one that decides whether I can staple this onto the tuned backbone at all, and it is
the same lesson the floor taught me in reverse. If I drop a fresh, randomly-initialized attention sublayer
into a residual stream tuned for the unconditional task, at step zero it injects random garbage and training
must first undo the damage — the floor's "climb off zero" cost, now with a much larger random sublayer. So I
make the block residual, $h \leftarrow h + \mathrm{Block}(h, c)$, and **zero-initialize the output
projection** that writes back into the stream. With $W_{\text{out}} = 0$ the block contributes exactly zero
at init, the network is bit-for-bit the original denoiser, and conditioning strength ramps from nothing as
the output projection moves off zero. This is the same zero-init-residual principle the floor's `AdaLNBlock`
gate used — here on the attention sublayer's output projection. The substrate's `CrossAttentionLayer`
provides exactly this assembly: GroupNorm(32), q from features and k,v from the context token, scaled
multi-head dot-product, a `zero_module`-wrapped output projection, and a residual add.

Now the time path. For the affine baselines I debated whether to route the class through the timestep, and
concat-FiLM won by doing so. Here the logic flips: *all* the class information is to flow through the
attention sublayers — that is the whole point of building them — so if I also dump the class into the time
embedding I split the conditioning across two mechanisms and muddy the time signal. Cleaner to leave the
time embedding a pure noise-level signal: `prepare_conditioning` returns `time_emb` unchanged, exactly as the
floor did, and the cross-attention carries "which class" entirely. Time tells the blocks *how noisy*;
attention tells them *what to draw*. So this rung returns to the floor's routing decision (class only in the
post-block path, timestep untouched) but with a far stronger post-block operator than the affine.

I have to be honest about the degenerate case, because the condition here is a *single* class label, not a
caption. The conditioner for a class is one learnable embedding vector — the context is a single token,
$M = 1$. Softmax over a single key is identically 1 regardless of the query, so every spatial position gets
weight 1 on the one value, and the attended branch receives the *same* vector $v = W_V\,\tau(c)$ at every
position. The content-dependence through $q_p$ that I was so pleased about *vanishes* when there is nothing
to compete in the softmax. So I should not oversell it: with $M = 1$ this layer is, in effect, a learned
class vector projected and added residually to the feature map — the residual preserves each position's own
$h_p$, and later convolutions can use the class offset together with local features, but this particular
softmax is not doing spatial routing. For a bare class label, then, cross-attention is a learned,
zero-init-gated *residual injection* of the class — richer than concat-FiLM mainly in that it is a learned
per-position-additive vector with its own projection and GroupNorm, rather than a per-channel affine bias
folded into the AdaGN. The reason to still build it this way is generality: the identical operator becomes
genuinely content-dependent and spatially varying the instant the condition is a set of $M > 1$ tokens (a
caption, a layout), where the softmax has multiple keys to weigh and each query resolves them differently. A
class label is the $M = 1$ corner of an operator whose payoff is that one mechanism covers class, text, and
layout alike. Here I feed it the single-token condition deliberately.

Let me also be clear about what the harness *omits* relative to the full mechanism, so the reasoning lands the
edit that actually runs. The full-strength conditioning block is a transformer block —
`GroupNorm → 1×1 proj_in → [self-attention, cross-attention(context), GEGLU feed-forward] × depth → 1×1
proj_out (zero-init) → residual`, where only the cross-attention sublayer is the queries-from-features /
keys-from-condition operation. The substrate builds *none* of that scaffolding: `ClassConditioner` here is a
single bare `CrossAttentionLayer` — no self-attention sublayer, no GEGLU FFN, no proj_in/proj_out wrapper,
no depth. So what runs is exactly one cross-attention sublayer (GroupNorm, q/k/v, scaled multi-head softmax,
zero-init output projection, residual) per UNet block, on a single class token. I am porting only the
cross-attention core; the transformer-block apparatus the full mechanism is known for is not present, and I
should not describe the edit as if it were.

So the step-3 edit: `prepare_conditioning` returns `time_emb` unchanged (class flows only through attention),
and `ClassConditioner` wraps one `CrossAttentionLayer(channels, cond_dim, num_heads=4)` per block (the full
scaffold module is in the answer). Training objective and sampler unchanged.

Now the falsifiable expectation against concat-FiLM's 19.39 / 11.50 / 10.52. I am claiming the attention
operator beats the affine because it carries the class as a *learned, content-mixed residual injection with
its own capacity* rather than a per-channel bias on the AdaGN, and because it is re-read at every scale
through a zero-init residual that doesn't disturb the backbone. So I expect cross-attention to come in
*below* concat-FiLM at all three scales — the strongest result on the ladder. But I temper the expectation
with the $M = 1$ honesty: because the softmax degenerates for a single class token, I do *not* expect a large
margin — the gain over concat-FiLM should be *small*, smaller than concat-FiLM's gain over the floor was, and
plausibly concentrated at the larger scales where the extra conditioning capacity has room to matter while
the affine has saturated. The clean discriminating outcome is "cross-attn ≲ concat-FiLM by a thin margin at
every scale": that would confirm the operator is genuinely stronger but that a single class token leaves most
of attention's content-dependent bandwidth unused — which would be the diagnosis to carry forward, that the
remaining headroom is in *what the condition is* (a richer token set, or guidance) rather than in *how it is
injected*. If instead cross-attn fails to beat concat-FiLM, the lesson would be that for a bare class label
the affine's tuned AdaGN routing already captures everything attention can, and the extra parameters bought
nothing — in which case the affine family was the right operator all along and the ladder tops out at
concat-FiLM.
