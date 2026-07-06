The dynamic effective-batch controller cut the run nearly in half, to about three and a half minutes. I've
made the *effective batch* adapt to the run; the other big knob I froze at a constant in the baseline is the
*sequence length*: 256 tokens, the same at step one and at the last step. Attention is quadratic in length,
so on paper every token of context I carry costs proportionally to the context I'm carrying — and I'm
carrying the full 256 from the very first step. Is that context earning its cost early on?

Before I get excited about "quadratic," let me actually price it, because in the baseline I noticed the
attention *mixing* is only about 10% of a block's arithmetic at L=256 — the projections and the MLP dominate.
Let me redo that fraction across the lengths a schedule would touch. Per token per layer the length-independent
work is the attention projections (8d² ≈ 1.18M FLOPs) plus the MLP (16d² ≈ 2.36M), while the length-dependent
mixing is 4·L·d. At d=384 that mixing is 49k at L=32, 98k at L=64, 197k at L=128, 393k at L=256, and 786k at
L=512, so the mixing *fraction* of a block runs 1.4%, 2.7%, 5.3%, 10%, 18.2% as L grows through those values.
That's a sobering number: even at 512 the quadratic term is under a fifth of a block's compute, and at the
short end it's a rounding error. So the naive story — "attention is quadratic, therefore short sequences are
dramatically cheaper" — is only weakly true in FLOPs at this width. If shortening the sequence is going to
pay, most of the payoff has to come from somewhere other than raw mixing FLOPs.

Two places it does. First, the attention *memory*: the score tensor is B × heads × L × L, and it is often the
memory-bound part of attention even when its FLOPs look small, because it's materialized (or streamed) per
step. Here's the lever that makes the schedule work — I can keep the *token budget per step* constant by
growing the batch as I shrink the length (same tokens, arranged as more/shorter sequences). At fixed token
budget T = B·L, the score-tensor size B·L² = T·L scales *linearly with L*, not quadratically. So running the
early phase at L=32 instead of 256, with the batch grown 8× to hold T constant, cuts the attention memory
traffic by 8× — and at L=512 it would be 2× the baseline's, which I only pay late. Second, and this is the
part that turns a modest saving into a real one: the schedule lets me push the *maximum* context to 512, twice
the baseline's 256, so the model ends with a longer window it can actually use — while the *average* length
over the run stays well below 256 because it starts at 32. I get more context capability at less average
attention cost than a flat-256 baseline. That's the trade worth making.

So the schedule writes itself from what the model can learn when. At initialization the network knows nothing;
the first thing it picks up is local structure — which token tends to follow which, short bigram/trigram-scale
dependencies. Long-range dependencies (this pronoun refers to that subject forty tokens back) can only be
learned *after* the local structure is in place, because they're built on top of it. So early in training,
feeding the model 256 or 512 tokens of context is mostly wasted: it's paying attention memory and mixing over
a long window when all it can extract is short-range signal a 32-token window would have given it just as
well, far cheaper and at 8× the batch. Begin at length 32, race through the local-structure phase, then
*double* the length periodically as the run progresses, up to 512, rescaling the batch inversely so the token
budget holds and halving the batch when I'm already at peak batch and doubling length, to avoid running out of
memory:

```python
def grow_sequence_length(current_sequence_length, current_max_batchsize, current_batchsize):
    current_sequence_length = min(2 * current_sequence_length, hyp['misc']['sequence_length']['max'])
    current_max_batchsize = round(batchsize * hyp['misc']['sequence_length']['max'] / current_sequence_length)
    if current_batchsize >= current_max_batchsize:    # at peak batch: halve it when doubling length, to avoid OOM
        current_batchsize = min(current_batchsize // 2, current_max_batchsize)
    return current_sequence_length, current_max_batchsize, current_batchsize
```

Notice the `round(batchsize * max_len / current_len)`: at current_len = 32 the max batch is 512/32 = 16× the
reference batch, at 256 it's 2×, at 512 it's 1×. Batch and length move inversely, so T = B·L stays roughly
fixed and each length runs at the largest batch that length allows — which is exactly what keeps the tensor
cores fed while the attention cost slides down at the short end. The OOM guard `if current_batchsize >= current_max_batchsize: halve` is what protects the last doublings,
where the batch can't keep growing forever. Trace the 256→512 step: at 256 the max batch is 2× the reference,
so if the run is sitting at that ceiling and I double the length, the new max at 512 is 1× the reference, and
since the current batch (2× ref) now exceeds the new ceiling (1× ref) the guard halves it back to 1× ref.
Token budget check: 256 × 2ref = 512·ref before, 512 × 1ref = 512·ref after — constant, exactly as intended,
with the guard doing nothing but keeping the memory footprint from doubling at the moment the window doubles.
With `growth_steps = 250`, the length doubles
every 250 steps, so the four doublings 32→64→128→256→512 take ~1000 steps of ramp before the run settles at
the full window. Because the batch is grown inversely and each phase is the same 250 steps, every phase
processes about the same *token* budget (250·T), so the ramp isn't a negligible warm-up — it's a real,
equal-weight chunk of the run spent at cheap short lengths, which is where the memory saving is banked.

The *doubling* cadence (rather than a linear 32→64→96→… ramp) is deliberate, and it follows from how
dependency range grows. Each doubling adds one octave of reach — 32 to 64 to 128 — and the difficulty of
learning "dependencies up to distance D" scales with the *order of magnitude* of D, not with D itself, so
equal step budget per octave is the natural allocation; a linear schedule would pour most of its steps into
the short lengths that are already learned and starve the long ones. Starting at 32 specifically is a floor
set by the same logic: a window shorter than a handful of tokens can't even represent the trigram-scale
structure that's worth learning first, so 32 is the smallest length that still holds real local dependencies
while being 8× cheaper in attention memory than the baseline's 256, and it's a power of two so every length in
the schedule stays tensor-core-aligned.

Now the problem this immediately creates, and it's the interesting part. My baseline injects order with a
*learned absolute-position embedding*: one trainable vector per position index, added to the token embedding.
That embedding is shaped by the sequence length. If I train mostly at length 32, only the first 32 position
vectors ever get gradients; positions 33–512 are never seen during the cheap phase and stay near their random
initialization. Then I jump to length 512 and suddenly the model has to use 480 position vectors it has barely
trained — a cold, near-random signal injected right where the run is supposed to be exploiting long context.
Worse, absolute-position embeddings don't *generalize* across lengths at all: a vector learned for "position
30 in a 32-long window" carries no guarantee about "position 30 in a 512-long window," because the
representation is keyed to an index slot, not to anything the two windows share. So the very schedule that
saves me time breaks the position mechanism it sits on top of. Growing the length needs a way of encoding
order that is *length-agnostic* — that means the same thing at 32 and at 512 and trains uniformly regardless
of the current window.

Let me walk the options for length-agnostic order, because there's more than one. I could keep absolute
embeddings and interpolate them to new lengths — but that's a hack that re-imposes the index-keyed
representation I just argued breaks, so no. I could use *fixed sinusoidal* absolute encodings: deterministic,
defined at any length, no untrained slots. That solves the cold-vectors problem, but it's still keyed to
absolute index rather than distance, and it gives the model no tunable knob for *how far* to look — every
layer gets the same fixed positional signal. I could use rotary embeddings, which rotate queries and keys by
a position-dependent phase: genuinely relative, genuinely length-agnostic, and strong in practice — this is
the tempting one. What makes me not reach for it here is surface area and composition. Rotary is a
*multiplicative* phase applied to q and k per position (and per head), a separate mechanism I'd bolt onto the
attention inputs; my attention already runs through an additive masked-logit path, and I want the smallest
change that makes the schedule work, not the most powerful positional scheme in the abstract.

There's also a subtler reason to want the *minimal* relative scheme rather than the most expressive one, and
it's specific to the length schedule. A richer relative encoding — a learned lookup table with one bias per
distance, like the additive relative-position tables some transformers use — would have O(L) parameters per
layer, and under my schedule most of those entries would be *cold*: distances 33 through 511 barely occur
until the run has grown past them, so a per-distance table reintroduces exactly the untrained-slot problem I'm
fleeing from, just moved from absolute indices to large distances. A single linear slope has the opposite
property: *every* query–key pair at *every* length contributes to the one scalar's gradient, so the slope is
trained by all distances at once and extrapolates to distances it has never literally seen, by construction.
For a schedule that spends its early life at short lengths, "one parameter trained by everything" beats "many
parameters each trained by little" — the expressivity I'd gain from a table is exactly the expressivity the
schedule would leave untrained.

That points at the minimal thing. What does order actually need to convey to attention? Mostly *relative
distance*: a token usually cares more about nearby tokens than far ones, and it cares about *how far* another
token is, not its absolute index. If I encode order as a function of the gap (i − j) between query position i
and key position j, and add it directly to the attention logits, then it's automatically length-agnostic — the
gap "5 tokens back" means the same thing whether the window is 32 or 512. The simplest such function is
*linear* in the distance: a bias proportional to how far back a key is. And it drops straight into the masked
additive path I already have — I add it wherever the causal mask permits attention and fill −∞ elsewhere so
the softmax zeros it — with exactly one learnable scalar, the slope, per attention layer, so each layer can
pick its own effective attention range. It's a smaller change than rotary, it composes with my existing mask
for free, and the per-layer slope is a knob rotary wouldn't give me.

Let me build it and check the sign, because a positional bias with the wrong sign would push attention toward
the future rather than the recent past. Precompute a base matrix `linear_encoding_base[i,j]`. Working it out,
`arange(-L+1,1)` broadcast over columns contributes (−L+1+j) at column j, and `arange(L-1,-1,-1)` broadcast
over rows contributes (L−1−i) at row i, so base[i,j] = (−L+1+j) + (L−1−i) = j − i. For any causal entry the
key sits at or before the query, j ≤ i, so j − i ≤ 0: the bias is 0 on the diagonal and grows increasingly
*negative* as the key recedes into the past. Make the slope a learnable parameter `linear_encoding_scaler`
passed through a softplus (with an LR multiplier so the lone scalar moves at a useful rate under a global LR
tuned for matrices), so the effective slope stays non-negative. A non-negative slope times a ≤0 base gives a
bias that is 0 for the current token and increasingly negative for distant keys — a smooth *recency* penalty,
attention decaying with distance, which is exactly the prior local structure wants. The positional
contribution to the logits is `softplus(mult·scaler) · base`, added inside the causal mask:

```python
# in the attention block:
self.linear_encoding_lr_mult = 50.
self.linear_encoding_scaler = nn.Parameter(torch.tensor(-.05 / self.linear_encoding_lr_mult, device='cuda'))
# signed query-key distance for every (i, j) pair, length-agnostic by construction
self.linear_encoding_base = (torch.arange(-L+1, 1).unsqueeze(0) + torch.arange(L-1, -1, -1).unsqueeze(1))
self.linear_encoding_mask = lambda mask, base, scaler: torch.where(
    mask, F.softplus(self.linear_encoding_lr_mult * scaler) * base, torch.full_like(base, -float('inf')))
self.causal_mask = torch.tril(torch.ones((L, L), device='cuda', dtype=torch.bool))

def forward(self, x):
    residual = x
    x = self.norm(x)
    attn_mask = self.linear_encoding_mask(self.causal_mask, self.linear_encoding_base, self.linear_encoding_scaler)
    x, _ = self.attention(x, x, x, attn_mask=attn_mask[:x.shape[1], :x.shape[1]], need_weights=False)
    return x + residual
```

This quietly changes what kind of mask I'm passing to attention, and the change is what makes the fusion
free. The baseline used a *boolean* mask (True = forbidden) and let the kernel apply the causal constraint.
Now `linear_encoding_mask` builds a *float* mask via `torch.where`: at allowed entries (the boolean tril is
True, j ≤ i) it writes the additive bias `softplus(mult·scaler)·base`, and at forbidden entries it writes
−∞. A float `attn_mask` is *added* to the attention logits before the softmax, so the −∞ entries drive their
softmax weights to e^(−∞) = 0 — the causal constraint is preserved exactly — while the allowed entries get
their logits shifted by the recency bias. The positional encoding therefore rides along inside the mask the
attention already applies; there's no separate positional op, no extra matmul, just a different fill in the
same masked-softmax the model was already doing. That's the sense in which the linear bias is the *cheap*
length-agnostic choice: it costs an add, not a mechanism.

The `lr_mult = 50` with the scaler stored as −.05/50 = −.001 is a small trick worth understanding: the stored
parameter is tiny, but it's multiplied by 50 before the softplus, so the gradient the optimizer sees on the
stored scalar is effectively amplified 50×. That's a poor-man's per-parameter learning rate, letting this one
scalar keep pace under an AdamW LR that was set for weight matrices. At init the effective slope is
softplus(50·−.001) = softplus(−.05) = ln(1 + e^(−.05)) ≈ 0.668, so training starts with a mild, already-active
recency bias rather than a flat one, and the slope is free to grow or shrink per layer from there.

Let me trace how strong that init recency actually is, because a bias too sharp would blind the model to
long-range structure from the start and one too weak would do nothing. Take a query with two keys of equal
content score, one 1 token back and one 10 tokens back. Their positional biases differ by slope·(−1) −
slope·(−10) = slope·9 = 0.668·9 ≈ 6.0 in logit space, so after the softmax the far key gets e^(−6.0) ≈ 0.0025
of the near key's weight — about a quarter of a percent. So at init a token ten back with equal content is
almost entirely ignored in favour of the immediate neighbour: a firm but not absolute recency prior, and since
each layer's slope is its own learnable parameter, layers that need to look further can flatten their slope
toward zero over training while layers doing local work can keep it sharp. That per-layer freedom is the whole
reason I put the scalar per block instead of sharing one.

Now the verification that this actually is length-agnostic, which is the whole reason I switched. Look at the
slice `[:x.shape[1], :x.shape[1]]` in the forward: because base[i,j] = j − i depends only on the gap, I can
build the base matrix once at the maximum length and crop its top-left corner to the current length. Take a
concrete pair of lengths: at L=32 the entry for "5 tokens back" (query i, key j=i−5) is slope·(j−i) = slope·
(−5); at L=512 the entry for "5 tokens back" is *also* slope·(−5); and the top-left 32×32 block of the
512×512 base is exactly the base a freshly-built 32×32 would produce, entry for entry, since j−i doesn't know
how big the matrix is. So the slope I learn while training at 32 is the identical slope applied at 512 — the
crop is exact, not approximate — and there is nothing length-specific left to go cold when I grow the window.
The learned absolute-position embedding is gone, and with it the mechanism that the schedule would have
broken.

One integration point I should not gloss over: the grad-norm effective-batch controller from the previous
rung is still running underneath all of this. The seqlen schedule sets the *shape* of a microbatch (B × L,
with B and L moving inversely), while the controller sets *how many* microbatches to accumulate before a step.
They act on orthogonal axes, so they compose rather than collide — but they do talk, because doubling the
length (and halving the batch at peak) changes the per-microbatch gradient noise, which the controller will
feel and re-adapt to. That's fine, even desirable: when a length doubling makes each microbatch noisier, the
grad-norm signal jumps and the controller accumulates a little more to compensate, automatically. I don't have
to hand-coordinate the two schedules; the closed-loop controller absorbs the disturbances the open-loop length
schedule injects.

So the two changes lock together. The sequence-length schedule (start at 32, double up to 512, batch grown
inversely so the token budget holds) front-loads the run with cheap, large-batch short-context steps that cut
the attention memory traffic ~8× in the early phase and match the local structure the model learns first,
while topping out at a longer 512-token window than the baseline ever used. The learnable linear positional
bias is the order-encoding that *makes that schedule possible* by meaning the same thing at every length and
training uniformly regardless of the current window. The bet against the previous ~3.5-minute rung: the early
phase was paying long-window attention to learn local structure a 32-token window learns just as well far
more cheaply, and front-loading the run with short, large-batch steps should shave the wall-clock again while
the ~3.8 bar holds. If the mechanism is right, the A100 seconds should drop further below 3.5 minutes and the
val loss should still land at ~3.8 (perplexity ~44.7) — the schedule changes *when* the model sees long
context, not the objective it converges to. I should calibrate the size of the expected win, though: since I
priced the attention mixing at only ~10% of a block's FLOPs at 256 and less below it, I do *not* expect
another near-halving the way the batch controller gave — the saving here is bounded by the attention fraction
plus the ~8× lighter early-phase memory traffic, so a further, more modest speedup is the honest prediction,
and if I saw a dramatic drop I'd suspect the length schedule had accidentally shortened the run itself rather
than made its steps cheaper. The honest risk is that a linear distance bias is a *weaker* order
representation than a fully-learned per-position embedding and could cost a little final quality; the per-layer
learnable slope, which lets each layer set its own recency range rather than sharing one fixed decay, is the
hedge, and the test is the same as always — as long as the model still lands at ~3.8, the saved time is real.
