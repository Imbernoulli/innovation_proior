Concat-FiLM came in at 19.39 / 11.50 / 10.52 FID, below the floor's 20.01 / 12.31 / 11.77 at every scale,
and I want to read the deltas carefully before deciding what to try next, because the *pattern* of the
gains is more informative than the fact of them. Scale by scale the improvement is $20.01\to 19.39 = 0.62$
at Small, $12.31\to 11.50 = 0.81$ at Medium, $11.77\to 10.52 = 1.25$ at Large. So the gain *grows* with
scale — $0.62, 0.81, 1.25$ — and in relative terms even more so: $3.1\%$, $6.6\%$, $10.6\%$. That settles
the question I posed at step 2 on the main point: the floor's bottleneck really was the *route*, not the
affine — moving the same family of conditioning from a freshly-inserted post-block conditioner onto the
timestep's own tuned block-internal AdaGN improved FID at every scale, even while dropping from the full
scale/shift/gate adaptive norm down to its additive-bias corner. The cheaper, simpler operator won,
precisely because it rode machinery already tuned for a global per-channel signal instead of climbing a new
sublayer off a zero gate.

But I have to be honest that the *shape* of the win was not what I bet on, and the mismatch is instructive.
At step 2 I predicted the largest absolute drop at Small, on the reasoning that the blunt, slow-to-engage
floor cost the capacity-tight small model the most. The numbers say the opposite: Small moved *least*
($0.62$), Large moved *most* ($1.25$). I also expected the Small-to-Large spread to *tighten*; instead it
*widened*, from the floor's $8.24$ ($20.01-11.77$) to $8.87$ ($19.39-10.52$), because Large improved more
than Small. So one half of my step-2 forecast was wrong and I should update on it rather than paper over it.
The half that was *right* is the one I derived from the floor's capacity/conditioning decomposition: I had
flagged that the near-flat $12.31\to 11.77$ meant Medium and Large were conditioning-limited, and I
predicted a "disproportionate move at Medium and Large" if routing was the true fix. That is exactly what
happened — the routing gain landed hardest where I had identified the conditioning ceiling. Reconciling the
two: Small barely moved because Small is *capacity*-bound, and no improvement to the conditioner can move a
capacity wall; the routing improvement could only cash out where the model had capacity to spend on it,
which is at Large. So the stubborn Small number is not the conditioner's fault at all — it is the small
backbone's — and the real evidence about the *operator* is at Large, where routing bought a full $1.25$ and
where, presumably, more is still on the table.

That reframing tells me the affine's ceiling is not where I first looked. Both adanorm and concat-FiLM are,
at bottom, the *same operator* — a global, per-channel affine drawn from the label, applied identically at
every spatial position and blind to local feature content. Re-routing it onto the tuned AdaGN was the
entire cheap road, and I have now driven its whole length: post-block to in-block placement, full affine to
additive corner, both measured. The one property I have *not* varied is the operator's spatial uniformity
and content-blindness. For the timestep that is exactly right — the noise level genuinely is a single global
property of the whole image. But "which of ten classes" wants to act *differently on different parts of the
picture, depending on what is already forming there* — a horse's legs and a horse's head are different local
computations, and a global per-channel gain cannot distinguish them. That wall is structural: no re-routing
fixes it, because re-routing is the lever I have exhausted. To go further I have to change the operator
itself, along the one axis I have not touched — content-dependent, spatially varying conditioning.

The strongest class-conditional pixel models confirm the diagnosis by how they had to work around it. ADM's
in-network conditioning is adaptive group norm — exactly the affine I have been using — and to get *strong*
class adherence it bolted on classifier guidance: a *separate* classifier on noised images whose gradient
nudges the sampler toward the class at every step. That is a confession that the in-network affine alone was
too low-bandwidth for "which class," and the fix lived outside the denoiser, at sampling time, at the cost
of a whole second model. I don't want that — the task fixes the sampler and forbids changing it, and I want
the denoiser *itself* to carry strong, content-dependent conditioning. So the affine family is out as the
*strong* in-network operator, and guidance-as-a-crutch is out as the cost.

"A position reads from the conditioning, with the read depending on what the position currently holds" is
the description of an attention layer: $\mathrm{Attention}(Q,K,V) = \mathrm{softmax}(QK^\top/\sqrt{d})\,V$,
each query dotting with every key, a softmax weighting the values. The crucial freedom is cross-attention —
queries and keys/values from *different* places. Map it on: the queries are the *image feature positions*,
the keys and values the *condition*. Flatten the $H\times W$ feature map into $N = HW$ query tokens, let the
class embedding source keys and values; the output has one row per query, comes back shaped like a feature
map, and slots straight into the convolutional stream. On paper this is genuinely different from the affine:
in FiLM the output at $p$ is $(1+\text{scale}_c)\,h_p + \text{shift}_c$, depending only on $c$; in attention
it is $\sum_j \mathrm{softmax}_j(q_p\cdot k_j)\,v_j$ with $q_p = W_Q\,h_p$, so the *weights* depend on the
position's own content. That is exactly the spatially varying, content-dependent conditioning the affine
lacked — *when there is more than one key to weigh*. I will come back to that caveat; here it is not a
footnote.

First build it carefully, because attention has details that bite. The projections: the feature map has $C$
channels, the class embedding lives at `cond_dim`, so three linear maps — $W_Q$ from the $C$-dim feature
tokens, $W_K, W_V$ from the condition — give learned separate views for the three roles, with output width
$C$ so the result adds back cleanly. The softmax scale $1/\sqrt{d}$, from first principles: if query and
key components are roughly independent, zero-mean, unit-variance, a single logit
$q\cdot k = \sum_{i=1}^{d} q_i k_i$ has mean $0$ and variance $d$, so the logits spread like $\sqrt{d}$;
push them through softmax and it saturates toward one-hot, and where softmax saturates its gradient nearly
vanishes — the layer would barely learn. Divide each logit by $\sqrt{d_{\text{head}}}$ and the variance
returns to $1$ regardless of width, keeping softmax soft and gradients alive. Heads: one head must compress
everything a position might read into a single weighting; multiple heads project into several subspaces,
attend in parallel, concatenate, and project back, so different heads read different aspects of the
condition at once instead of averaging them — with per-head width $C/\text{num\_heads}$ so total cost stays
about one full-width head. Four heads is fine at this scale. Normalization and placement: the UNet uses
GroupNorm everywhere and already has self-attention sublayers at low resolution, so I GroupNorm the feature
map before projecting queries (matching the backbone and keeping query magnitudes controlled), and insert
one such conditioning block after each down/mid/up block so the class is re-read at every scale.

Whether I can staple this onto the tuned backbone at all comes down to the floor's lesson in reverse: a
fresh, randomly-initialized attention sublayer dropped into the residual stream injects random garbage at
step zero, and training must first undo it — the "climb off zero" cost, now with a much larger random
sublayer. So I make the block residual, $h \leftarrow h + \mathrm{Block}(h, c)$, and **zero-initialize the
output projection**. With $W_{\text{out}} = 0$ the block contributes exactly zero at init — bit-for-bit the
original denoiser — and conditioning ramps from nothing as $W_{\text{out}}$ moves off zero. Same zero-init
residual principle, same two-phase signature the floor's gate had: $W_{\text{out}}$ gets gradient at init
($\partial(h+o)/\partial W_{\text{out}} = \text{attn} \neq 0$) while $W_V, W_K$ stay frozen behind the zero
projection until it opens. The substrate's `CrossAttentionLayer` is exactly this assembly: GroupNorm(32), q
from features and k,v from the context token, scaled multi-head dot-product, a `zero_module`-wrapped output
projection, and a residual add.

Now the caveat I flagged, and it is the whole honesty of this step, because the condition here is a *single*
class label, not a caption. The conditioner for a class is one learnable embedding vector — the context is a
single token, $M = 1$ — and I need to work out what the mechanism degenerates to at $M=1$ rather than
describe the $M>1$ story and quietly hope. Softmax over a single key: the scores tensor is
$[B, N, 1]$, one logit per query, and $\mathrm{softmax}$ of a length-one vector is identically $1$,
regardless of the logit. So every query gets weight exactly $1$ on the one value, and the attended output is
$v = W_V\,\tau(c)$ at *every* spatial position — the same vector everywhere. Trace the consequences all the
way down, because they are stronger than "the softmax is trivial." First, the output does not depend on the
query at all: $\partial\,\mathrm{softmax}(s)/\partial s = 0$ for a length-one softmax, so
$\partial o/\partial q = 0$ and $W_Q$ receives *no gradient, ever*, as long as $M=1$ — the query projection
is permanently dead weight. Second, the logit itself is discarded, so $W_K$ is dead too, and the
carefully-derived $1/\sqrt{d}$ scale and the multi-head split are moot — there is nothing to sharpen or to
read in parallel when the weight is a constant $1$. Third, and decisively, the output is *spatially
uniform* ($v$ is identical at every position) and *content-blind* ($v = W_V\tau(c)$ has no dependence on
$h$ whatsoever). The very content-dependence through $q_p$ that motivated the whole design *vanishes* when
there is nothing to compete in the softmax. What survives is $W_V$ and $W_{\text{out}}$: the operator
reduces to $h \leftarrow h + W_{\text{out}}\big(W_V\,\tau(c)\big)$, a learned per-channel vector, the same
at every position, added residually through a zero-initialized gate. That is a per-channel *additive bias*
drawn from the class, spatially uniform and content-blind, which is the *same structural class* as the
affine I set out to escape.

So I should not oversell this. With $M=1$ cross-attention is not the content-dependent, spatially varying
operator on paper; it is a learned, zero-init-gated residual injection of the class. But the residual is
what keeps the mechanism from being pointless, and it is worth being precise about what it does buy. The
layer outputs $h_p + v$ at every position, with $v$ a class-dependent constant and $h_p$ the position's own
feature. The *next* block's convolution then computes $f(h_p + v)$, which does depend jointly on the local
content $h_p$ and the class offset $v$ through the conv's weights and nonlinearity — so the class *does*
end up acting differently at different positions, but that content-dependence is manufactured downstream by
the following conv, not by this attention layer's softmax. The layer itself contributes only a uniform
class offset; the spatial specialization lives in how the convolutional stack subsequently mixes that offset
with local features. That is a genuine but second-order effect, and it is the same effect concat-FiLM's
shift already enjoys — its per-channel bias also gets mixed with local content by the next conv. So the
downstream-conv argument does *not* separate cross-attention from the affine; it applies to both.

On parameter cost, this is the fill the budget is measured against. Per block cross-attention carries
$W_Q, W_K, W_V, W_{\text{out}}$: roughly $2C^2 + 2\,\text{cond\_dim}\cdot C$. At Medium ($C=256$, `cond_dim`
$=512$) that is $\approx 0.39$M per block, against concat-FiLM's *zero* — so the ladder spans zero-overhead
to most-expensive, and at $M=1$ the $2C^2$ in the inert $W_Q, W_K$ is paid for a generality this
single-token condition does not use. The honest question is whether it can even beat concat-FiLM, since both
reduce to a class-driven per-channel modulation, and I do not know the sign in advance. For concat-FiLM: a
full per-channel *affine* (scale *and* shift) folded into the block's tuned modulation, shaping what the
block computes with a joint $(t,c)$ coupling through the shared SiLU, at zero parameters. For
cross-attention: a *shift only* but with dedicated capacity — private $W_V$, $W_{\text{out}}$, its own
GroupNorm and nonlinearity — re-applied post-block at every scale through a clean zero-init residual that
keeps the class on a path separate from the timestep. Whether a dedicated post-block injection outbids a
tuned in-block affine for a bare label is not obvious; the extra capacity and clean separation are the
reasons to expect a thin win, but I hold that loosely.

The reason to still build it this way is generality: the identical operator becomes genuinely
content-dependent and spatially varying the instant the condition is $M > 1$ tokens — a caption, a layout —
where $W_Q$, $W_K$, the scale, and the heads all come alive. A class label is the $M = 1$ corner of an
operator whose payoff is that one mechanism covers class, text, and layout alike; here I feed it the
single-token condition deliberately, eyes open about which parts are inert.

As for routing, the logic flips from concat-FiLM: *all* the class information is to flow through the
attention sublayers — that is the whole point of building them — so if I also dump the class into the time
embedding I split the conditioning across two mechanisms and muddy the time signal. Cleaner to leave the
time embedding a pure noise-level signal: `prepare_conditioning` returns `time_emb` unchanged, exactly as
the floor did, and the cross-attention carries "which class" entirely. Time tells the blocks *how noisy*;
attention tells them *what to draw*. So this step returns to the floor's routing decision — class only in
the post-block path, timestep untouched — but with a far stronger post-block operator than the affine (and,
at $M=1$, one whose strength over the affine is smaller than its name suggests).

One thing to be clear about is what the substrate *omits*. The full-strength conditioning block is a
transformer block — `GroupNorm → 1×1 proj_in → [self-attention, cross-attention(context), GEGLU FFN] ×
depth → 1×1 proj_out (zero-init) → residual`, where only the cross-attention sublayer is the
queries-from-features / keys-from-condition operation. `ClassConditioner` here is a single bare
`CrossAttentionLayer` — no self-attention sublayer, no GEGLU FFN, no proj_in/proj_out wrapper, no depth. So
what runs is exactly one cross-attention sublayer per UNet block on a single class token; I am porting only
the cross-attention core.

So the step-3 edit: `prepare_conditioning` returns `time_emb` unchanged (class flows only through
attention), and `ClassConditioner` wraps one `CrossAttentionLayer(channels, cond_dim, num_heads=4)` per
block (the full module is in the answer). Training objective and sampler unchanged.

Now the falsifiable expectation against concat-FiLM's 19.39 / 11.50 / 10.52. I claim the attention operator
beats the affine but by a *thin* margin, because it carries the class as a learned, content-mixed residual
injection with its own capacity rather than a per-channel bias on the AdaGN, re-read at every scale through
a zero-init residual that doesn't disturb the backbone — so *below* concat-FiLM at all three scales, the
strongest result on the ladder. The $M=1$ honesty caps how much: with the softmax degenerate for a single
token, the gain should be *smaller* than concat-FiLM's own biggest gain over the floor ($1.25$ at Large) —
well under a point per scale, not a step-change. By the capacity/conditioning logic, what gain there is
should concentrate at the larger scales while Small, capacity-bound, barely moves. If instead cross-attn
fails to beat concat-FiLM at all, then for a bare label the tuned AdaGN routing already captures everything
attention can — the dedicated post-block capacity did not outbid the in-block affine — and the ladder tops
out at concat-FiLM, its remaining headroom in *what the condition is* (a richer token set, guidance) rather
than *how it is injected*, since a single class token leaves most of attention's bandwidth provably unused.
