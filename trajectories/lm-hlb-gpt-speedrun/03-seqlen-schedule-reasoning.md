The dynamic effective-batch controller cut the run nearly in half, to about three and a half minutes. I've made
the *effective batch* adapt; the other big knob I froze at a constant in the baseline is the *sequence length* —
256 tokens, the same at step one and at the last step. Attention is quadratic in length, and I'm carrying the
full 256 from the very first step. Is that context earning its cost early on?

Before I get excited about "quadratic," let me price it, because in the baseline the attention *mixing* was only
~10% of a block's arithmetic at L=256. Per token per layer the length-independent work is the attention
projections (8d² ≈ 1.18M FLOPs) plus the MLP (16d² ≈ 2.36M), while the length-dependent mixing is 4·L·d — 49k
at L=32, 98k at 64, 197k at 128, 393k at 256, 786k at 512, so the mixing *fraction* of a block runs 1.4%, 2.7%,
5.3%, 10%, 18.2%. Even at 512 the quadratic term is under a fifth of a block's compute, and at the short end
it's a rounding error. So the naive story — "attention is quadratic, so short sequences are dramatically
cheaper" — is only weakly true in FLOPs at this width. If shortening the sequence is going to pay, most of the
payoff has to come from somewhere other than raw mixing FLOPs.

Two places it does. First, attention *memory*: the score tensor is B × heads × L × L, often the memory-bound
part even when its FLOPs look small. Here's the lever — I can hold the *token budget per step* constant by
growing the batch as I shrink the length (same tokens, arranged as more/shorter sequences). At fixed token
budget T = B·L, the score-tensor size B·L² = T·L scales *linearly with L*, not quadratically, so running the
early phase at L=32 with the batch grown 8× cuts attention memory traffic ~8× — and at L=512 it would be 2×
the baseline's, which I only pay late. Second, and this is what turns a modest saving into a real one: the
schedule lets me push the *maximum* context to 512, twice the baseline's 256, so the model ends with a longer
window it can actually use — while the *average* length over the run stays well below 256 because it starts at
32. More context capability at less average attention cost than a flat-256 baseline. That's the trade worth
making.

The schedule follows from what the model can learn when. At initialization it knows nothing; the first thing it
picks up is local structure — short bigram/trigram-scale dependencies. Long-range dependencies can only be
learned *after* the local structure is in place, because they're built on top of it. So early on, feeding 256
or 512 tokens is mostly wasted: the model pays attention memory over a long window when all it can extract is
short-range signal a 32-token window would have given it just as well, far cheaper and at 8× the batch. Begin
at length 32, then *double* the length periodically up to 512, rescaling the batch inversely so the token
budget holds and halving the batch when I'm already at peak batch and doubling length, to avoid running out of
memory (`grow_sequence_length` in the answer). The inverse `round(batchsize · max_len / current_len)` makes the
max batch 16× the reference at length 32, 2× at 256, 1× at 512, so B and L move inversely and each length runs
at the largest batch it allows — which keeps the tensor cores fed while the attention cost slides down at the
short end. With `growth_steps = 250` the length doubles every 250 steps, so the four doublings
32→64→128→256→512 take ~1000 steps of ramp; because the batch grows inversely and each phase is the same 250
steps, every phase processes about the same *token* budget (250·T), so the ramp isn't a negligible warm-up —
it's a real, equal-weight chunk of the run spent at cheap short lengths where the memory saving is banked.

The *doubling* cadence rather than a linear ramp is deliberate: each doubling adds one octave of reach, and the
difficulty of learning "dependencies up to distance D" scales with the *order of magnitude* of D, so equal step
budget per octave is the natural allocation — a linear schedule would pour most of its steps into short lengths
already learned and starve the long ones. Starting at 32 specifically is the smallest length that still holds
real trigram-scale structure while being 8× cheaper in attention memory than 256, and it's a power of two so
every length in the schedule stays tensor-core-aligned.

Now the problem this creates, and it's the interesting part. The baseline injects order with a *learned
absolute-position embedding*: one trainable vector per position index. If I train mostly at length 32, only the
first 32 position vectors ever get gradients; positions 33–512 stay near random. Then I jump to length 512 and
the model has to use 480 barely-trained position vectors — a cold, near-random signal injected right where the
run is supposed to be exploiting long context. Worse, absolute embeddings don't *generalize* across lengths at
all: a vector for "position 30 in a 32-window" carries no guarantee about "position 30 in a 512-window,"
because the representation is keyed to an index slot, not to anything the two windows share. So the very
schedule that saves me time breaks the position mechanism it sits on. Growing the length needs a way of
encoding order that is *length-agnostic* — the same at 32 and 512, trained uniformly regardless of the current
window.

There's more than one length-agnostic option, but they mostly fail me for concrete reasons. Interpolating the
absolute embeddings re-imposes the index-keyed representation I just argued breaks. Fixed sinusoidal encodings
solve the cold-slots problem but are still keyed to absolute index and give no tunable knob for *how far* to
look. Rotary is genuinely relative and length-agnostic and strong in practice — the tempting one — but it's a
*multiplicative* phase bolted onto the attention inputs per position and head, a separate mechanism, when my
attention already runs through an additive masked-logit path and I want the smallest change that makes the
schedule work. And there's a subtler reason to want the *minimal* relative scheme specific to this schedule: a
richer relative encoding — a learned per-distance bias table — would have O(L) parameters per layer, most of
them *cold* under my schedule (distances 33–511 barely occur until the run grows past them), reintroducing
exactly the untrained-slot problem I'm fleeing, just moved from absolute indices to large distances. A single
linear slope has the opposite property: *every* query–key pair at *every* length contributes to the one
scalar's gradient, so it's trained by all distances at once and extrapolates to distances it has never literally
seen. For a schedule that spends its early life at short lengths, one parameter trained by everything beats
many parameters each trained by little.

That points at the minimal thing. What order needs to convey to attention is mostly *relative distance* — a
token cares about how far back another token is, not its absolute index. If I encode order as a function of the
gap (i − j) and add it directly to the attention logits, it's automatically length-agnostic: "5 tokens back"
means the same at 32 or 512. The simplest such function is *linear* in the distance, a bias proportional to how
far back a key is, and it drops straight into the masked additive path I already have — one learnable scalar,
the slope, per attention layer, so each layer picks its own effective attention range.

Building it, I precompute a base matrix and check the sign, because a positional bias with the wrong sign would
push attention toward the future. `arange(-L+1,1)` over columns contributes (−L+1+j) at column j and
`arange(L-1,-1,-1)` over rows contributes (L−1−i) at row i, so base[i,j] = j − i. For any causal entry j ≤ i,
so j − i ≤ 0: the bias is 0 on the diagonal and grows increasingly *negative* as the key recedes. Make the
slope a learnable parameter passed through a softplus so it stays non-negative; a non-negative slope times a
≤0 base gives a bias that is 0 for the current token and increasingly negative for distant keys — a smooth
*recency* penalty, which is exactly the prior local structure wants. The positional contribution
`softplus(mult·scaler)·base` rides *inside* the mask (attention block in the answer).

That quietly changes the mask type, and the change is what makes the fusion free. The baseline used a *boolean*
mask (True = forbidden). Now the mask is a *float* built via `torch.where`: at allowed entries it writes the
additive bias, at forbidden entries −∞. A float `attn_mask` is *added* to the logits before the softmax, so the
−∞ entries drive their softmax weights to e^(−∞) = 0 — the causal constraint preserved exactly — while allowed
entries get their logits shifted by the recency bias. The positional encoding therefore rides along inside the
masked-softmax the model was already doing: no separate positional op, no extra matmul, just a different fill.
That's the sense in which the linear bias is the *cheap* length-agnostic choice — it costs an add, not a
mechanism.

The `lr_mult = 50` with the scaler stored as −.05/50 = −.001 is a poor-man's per-parameter learning rate: the
stored scalar is tiny but multiplied by 50 before the softplus, so the gradient the optimizer sees is
effectively amplified 50×, letting this one scalar keep pace under an AdamW LR set for weight matrices. At init
the effective slope is softplus(50·−.001) = softplus(−.05) ≈ 0.668 — a mild, already-active recency bias rather
than a flat one. Concretely, at that slope two equal-content keys 1 and 10 tokens back differ by slope·9 ≈ 6.0
in logit space, so the far key gets e^(−6.0) ≈ 0.0025 of the near key's weight: a firm but not absolute recency
prior at init, and since each layer's slope is its own parameter, layers needing longer reach can flatten
theirs over training while local layers keep theirs sharp — which is the whole reason the scalar is per block.

The length-agnosticism is the point of the switch and it's exact, not approximate: because base[i,j] = j − i
depends only on the gap, I build the base once at the maximum length and crop its top-left corner to the
current length (`[:x.shape[1], :x.shape[1]]`). The top-left 32×32 block of the 512×512 base is entry-for-entry
the base a fresh 32×32 would produce, so the slope I learn at 32 is the identical slope applied at 512 — nothing
length-specific is left to go cold when I grow the window. The learned absolute-position embedding is gone, and
with it the mechanism the schedule would have broken.

One integration point: the grad-norm effective-batch controller from the previous rung is still running
underneath. The seqlen schedule sets the *shape* of a microbatch (B × L, moving inversely) while the controller
sets *how many* to accumulate — orthogonal axes, so they compose. But they do talk: doubling the length (and
halving the batch at peak) changes the per-microbatch gradient noise, which the controller feels and re-adapts
to automatically, accumulating a little more when a doubling makes each microbatch noisier. I don't have to
hand-coordinate the two; the closed loop absorbs the disturbances the open-loop length schedule injects.

So the two changes lock together: the length schedule front-loads the run with cheap, large-batch short-context
steps that cut attention memory traffic ~8× in the early phase and match the local structure learned first,
while topping out at a 512-token window the baseline never used; the learnable linear positional bias is the
order-encoding that *makes that schedule possible* by meaning the same thing at every length. I should
calibrate the expected win: since the attention mixing is only ~10% of a block's FLOPs at 256 and less below,
I do *not* expect another near-halving the way the batch controller gave — the saving is bounded by the
attention fraction plus the ~8× lighter early-phase memory traffic, so a further, more modest speedup is the
honest prediction, and a dramatic drop would make me suspect the schedule had accidentally shortened the run
itself. The honest risk is that a linear distance bias is a weaker order representation than a fully-learned
per-position embedding and could cost a little final quality; the per-layer learnable slope is the hedge, and
the test is the same — as long as the model still lands at ~3.8, the saved time is real.
