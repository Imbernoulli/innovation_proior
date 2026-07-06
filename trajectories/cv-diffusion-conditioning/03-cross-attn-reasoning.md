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

That reframing matters for what I do next, because it tells me the affine's ceiling is not where I first
looked. Both adanorm and concat-FiLM are, at bottom, the *same operator* — a global, per-channel affine
drawn from the class label, applied identically at every spatial position and blind to what is forming in
the feature map. Re-routing it onto the tuned AdaGN was the entire cheap road, and I have now driven the
whole length of it: post-block placement to in-block placement, full affine to additive corner, both
measured. The one property I have *not* varied across two rungs is the operator's spatial uniformity and
content-blindness — every version gives each channel one scale and one shift computed from the label alone
and applied at every position identically. The affine can say "globally turn channel 37 up, channel 12
down"; it cannot say "near this patch of the feature map emphasize this; over there emphasize that,"
because the modulation has no spatial address, and it never *looks at* the local content of the feature it
modulates. For the timestep that is exactly right — the noise level genuinely is a single global property
of the whole image, so a content-blind global knob fits it perfectly. But "which of ten classes" wants to
act *differently on different parts of the picture, depending on what is already forming there* — a horse's
legs and a horse's head are different local computations, and a global per-channel gain cannot distinguish
them. That is the wall the affine hits, and it is structural: no amount of re-routing fixes it, because
re-routing is the lever I have already exhausted. To go further I have to change the operator itself, along
the one axis I have not touched — content-dependent, spatially varying conditioning.

The strongest class-conditional pixel models confirm the diagnosis by how they had to work around it. ADM's
in-network conditioning is adaptive group norm — exactly the affine I have been using — and to get *strong*
class adherence it bolted on classifier guidance: a *separate* classifier on noised images whose gradient
nudges the sampler toward the class at every step. That is a confession that the in-network affine alone was
too low-bandwidth for "which class," and the fix lived outside the denoiser, at sampling time, at the cost
of a whole second model. I don't want that — the task fixes the sampler and forbids changing it, and I want
the denoiser *itself* to carry strong, content-dependent conditioning. So the affine family is out as the
*strong* in-network operator, and guidance-as-a-crutch is out as the cost.

Let me restate the requirement crisply, because it points somewhere specific. I want an operator where each
spatial position of the feature map can be conditioned *differently, as a function of its own content*. "A
position reads from the conditioning, with the read depending on what the position currently holds" is the
description of an attention layer. Pull up the mechanism:
$\mathrm{Attention}(Q,K,V) = \mathrm{softmax}(QK^\top/\sqrt{d})\,V$. Queries, keys, values; each query dots
with every key, a softmax over those scores weights the values, the output is the weighted sum. The crucial
freedom is that queries and keys/values can come from *different* places — cross-attention. Map it onto my
problem: the queries should be the *image feature positions* — I want each spatial location to do the
reading — and the keys and values should be the *condition*. Flatten the $H\times W$ feature map into
$N = HW$ query tokens, let the class embedding source the keys and values; the output has one row per query,
i.e. one per spatial position, so it comes back shaped like a feature map and slots straight into the
convolutional stream. On paper this is genuinely different from the affine: in FiLM the output at position
$p$ is $(1+\text{scale}_c)\,h_p + \text{shift}_c$, where the modulation depends only on the class $c$; in
attention the output at $p$ is $\sum_j \mathrm{softmax}_j(q_p\cdot k_j)\,v_j$ with $q_p = W_Q\,h_p$, so the
*weights* depend on the position's own content. That is exactly the spatially varying, content-dependent
conditioning the affine lacked — *when there is more than one key to weigh*. I will come back to that
caveat, because here it is not a footnote.

First build it carefully, because attention has details that bite. The projections: the feature map has $C$
channels, the class embedding lives at `cond_dim`, so three linear maps — $W_Q$ from the $C$-dim feature
tokens, $W_K, W_V$ from the condition — give learned separate views for the three roles, with output width
$C$ so the result adds back cleanly. The softmax scale $1/\sqrt{d}$, which I want from first principles, not
by copying. If query and key components are roughly independent, zero-mean, unit-variance, a single logit
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

The detail that decides whether I can staple this onto the tuned backbone at all is the same lesson the
floor taught me in reverse. If I drop a fresh, randomly-initialized attention sublayer into a residual
stream tuned for the unconditional task, at step zero it injects random garbage and training must first undo
the damage — the floor's "climb off zero" cost, now with a much larger random sublayer. So I make the block
residual, $h \leftarrow h + \mathrm{Block}(h, c)$, and **zero-initialize the output projection** that writes
back into the stream. With $W_{\text{out}} = 0$ the block contributes exactly zero at init, the network is
bit-for-bit the original denoiser, and conditioning strength ramps from nothing as the output projection
moves off zero. This is the same zero-init-residual principle the floor's `AdaLNBlock` gate used — and it
has the same two-phase gradient signature, which I should trace so I know the module is not dead. Write
$o = W_{\text{out}}(\text{attn})$ and $h \leftarrow h + o$. At init $\partial(h+o)/\partial W_{\text{out}} =
\text{attn} \neq 0$, so $W_{\text{out}}$ receives gradient immediately; but $\partial o/\partial W_V =
W_{\text{out}}\cdot(\dots) = 0$ at init, so $W_V$ (and $W_K$) are frozen behind the zero output projection
until it climbs off zero. Just like the floor's gate, conditioning engages in two phases — output
projection first, then the value/key content. The substrate's `CrossAttentionLayer` provides exactly this
assembly: GroupNorm(32), q from features and k,v from the context token, scaled multi-head dot-product, a
`zero_module`-wrapped output projection, and a residual add.

Now the caveat I flagged, and it is the whole honesty of this rung, because the condition here is a *single*
class label, not a caption. The conditioner for a class is one learnable embedding vector — the context is a
single token, $M = 1$ — and I need to work out what the mechanism actually degenerates to at $M=1$ rather
than describe the $M>1$ story and quietly hope. Softmax over a single key: the scores tensor is
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
at every position, added residually through a zero-initialized gate. That is — I have to say it plainly — a
per-channel *additive bias* drawn from the class, spatially uniform and content-blind, which is the *same
structural class* as the affine I set out to escape.

Let me make the degeneracy concrete with actual shapes at one scale so it is not an abstraction. Take a
Small down block at the top resolution: $C=64$, $H=W=32$, so flattening gives $N = HW = 1024$ query tokens,
each $64$-dim. The context is the single class embedding, one token. The scores tensor is therefore
$[B, 1024, 1]$ — a thousand-and-twenty-four queries, each looking at exactly one key — and softmax along the
length-one last axis returns a $[B, 1024, 1]$ tensor of all ones. So all $1024$ positions read the identical
value $v$, and the "attention map" I might have hoped to inspect is a constant. There is nothing to
visualize because there is nothing being routed.

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

I should also put the parameter cost beside the others, since this is the fill the budget is measured
against. Per block cross-attention carries $W_Q, W_K, W_V, W_{\text{out}}$: roughly $2C^2 +
2\,\text{cond\_dim}\cdot C$ weights. At Medium ($C=256$, `cond_dim` $=512$) that is $2(256^2) + 2(512)(256)
= 131072 + 262144 \approx 0.39$M per block — coincidentally equal to the floor's `AdaLNBlock` there
($3\cdot512\cdot256 = 0.39$M, since `cond_dim` $=2C$ at this level), and against concat-FiLM's *zero*. So the
ladder spans zero-overhead (concat-FiLM) to the most expensive fill (cross-attention), and at $M=1$ half of
cross-attention's cost — the $2C^2$ in $W_Q, W_K$ — sits on projections I proved inert. It fits the budget
by definition, being the reference; I am simply noting that the reference is paying full price for a
generality this single-token condition does not use. The honest question, then, is whether it can
even beat concat-FiLM, since both reduce to a class-driven per-channel modulation. The differences are real
but modest, and I genuinely do not know their sign in advance. In concat-FiLM's favour: it gives a full
per-channel *affine* (a scale *and* a shift) folded into the block's own tuned modulation, and — as I argued
last rung — it shapes what the block *computes*, not just its output, plus it gets a measurable joint
$(t,c)$ coupling through the shared SiLU, all at zero added parameters. In cross-attention's favour: it is a
*shift only* (no scale) but with its own dedicated capacity — a private $W_V$, a private $W_{\text{out}}$,
its own GroupNorm and nonlinearity — re-applied post-block at every scale through a clean zero-init residual
that keeps the class on a path entirely separate from the timestep, so the two signals never dilute each
other. Whether a dedicated post-block additive injection outbids a tuned in-block affine for a bare label is
not obvious; the extra capacity and the clean separation are the reasons to expect a thin win, but I hold
that loosely.

The reason to still build it this way, despite the $M=1$ degeneracy, is generality. The identical operator
becomes genuinely content-dependent and spatially varying the *instant* the condition is a set of $M > 1$
tokens — a caption, a layout — where the softmax has multiple keys to weigh and each query resolves them
differently, and where $W_Q$, $W_K$, the scale, and the heads all come alive. A class label is the $M = 1$
corner of an operator whose payoff is that one mechanism covers class, text, and layout alike. Here I feed
it the single-token condition deliberately, eyes open about which of its parts are inert. It is also, not
incidentally, the *most expensive* fill on the ladder — it is the very model the $1.05\times$ parameter
budget is measured against — and at $M=1$ a chunk of that expense ($W_Q$, $W_K$) buys nothing; I am paying
for generality I am not exercising. That is a cost I accept for this rung and would want to revisit if the
condition stayed a bare label.

As for routing, the logic flips from concat-FiLM: *all* the class information is to flow through the
attention sublayers — that is the whole point of building them — so if I also dump the class into the time
embedding I split the conditioning across two mechanisms and muddy the time signal. Cleaner to leave the
time embedding a pure noise-level signal: `prepare_conditioning` returns `time_emb` unchanged, exactly as
the floor did, and the cross-attention carries "which class" entirely. Time tells the blocks *how noisy*;
attention tells them *what to draw*. So this rung returns to the floor's routing decision — class only in
the post-block path, timestep untouched — but with a far stronger post-block operator than the affine (and,
at $M=1$, one whose strength over the affine is smaller than its name suggests).

Let me also be clear about what the harness *omits* relative to the full mechanism, so the reasoning lands
the edit that actually runs. The full-strength conditioning block is a transformer block —
`GroupNorm → 1×1 proj_in → [self-attention, cross-attention(context), GEGLU feed-forward] × depth → 1×1
proj_out (zero-init) → residual`, where only the cross-attention sublayer is the queries-from-features /
keys-from-condition operation. The substrate builds *none* of that scaffolding: `ClassConditioner` here is a
single bare `CrossAttentionLayer` — no self-attention sublayer, no GEGLU FFN, no proj_in/proj_out wrapper,
no depth. So what runs is exactly one cross-attention sublayer (GroupNorm, q/k/v, scaled multi-head softmax,
zero-init output projection, residual) per UNet block, on a single class token. I am porting only the
cross-attention core; the transformer-block apparatus the full mechanism is known for is not present, and I
should not describe the edit as if it were.

So the step-3 edit: `prepare_conditioning` returns `time_emb` unchanged (class flows only through
attention), and `ClassConditioner` wraps one `CrossAttentionLayer(channels, cond_dim, num_heads=4)` per
block (the full scaffold module is in the answer). Training objective and sampler unchanged.

Now the falsifiable expectation against concat-FiLM's 19.39 / 11.50 / 10.52, and I can make it numerically
sharp using the two tables I now have. I claim the attention operator beats the affine — but by a *thin*
margin — because it carries the class as a learned, content-mixed residual injection with its own capacity
rather than a per-channel bias on the AdaGN, and because it is re-read at every scale through a zero-init
residual that doesn't disturb the backbone. So I expect cross-attention *below* concat-FiLM at all three
scales, the strongest result on the ladder. The $M=1$ honesty puts a *ceiling* on how much: because the
softmax degenerates for a single class token, the gain over concat-FiLM should be *smaller* than
concat-FiLM's own biggest gain over the floor, which was $1.25$ at Large — so I am predicting a per-scale
improvement well under a point, not a step-change. And by the same capacity/conditioning logic that finally
held up last rung, I expect what gain there is to concentrate at the larger scales, where the extra
conditioning capacity has room to matter while Small stays capacity-bound and barely moves — Small has
resisted every conditioning change so far, and I do not expect a degenerate-softmax operator to be what
frees it. The clean discriminating outcome is "cross-attn ≲ concat-FiLM by a thin margin, widest at Large,
negligible at Small": that would confirm a genuinely stronger operator whose remaining headroom is in *what
the condition is* — a richer token set, or guidance — rather than in *how it is injected*, since a single
class token leaves most of attention's content-dependent bandwidth provably unused. If instead cross-attn
fails to beat concat-FiLM at all, the lesson would be that for a bare class label the affine's tuned AdaGN
routing already captures everything attention can — the dedicated post-block capacity did not outbid the
in-block affine — and the ladder tops out at concat-FiLM, with the affine family the right operator all
along.
