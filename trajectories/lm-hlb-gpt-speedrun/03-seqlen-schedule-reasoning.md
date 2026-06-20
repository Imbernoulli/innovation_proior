I've made the *effective batch* adapt to the run. The other big knob I froze at a constant in the baseline
is the *sequence length*: 256 tokens, the same at step one and at the last step. Attention is quadratic in
length, so every token of context I carry costs proportionally to the context I'm carrying — and I'm
carrying the full 256 from the very first step. Is that context earning its cost early on?

Think about what the model can possibly learn early. At initialization the network knows nothing; the first
thing it picks up is local structure — which token tends to follow which, short bigram/trigram-scale
dependencies. Long-range dependencies (this pronoun refers to that subject forty tokens back) can only be
learned *after* the local structure is in place, because they're built on top of it. So early in training,
feeding the model 256 tokens of context is mostly wasted: it's paying quadratic attention cost over a long
window when all it can extract is short-range signal that a much shorter window would have given it just as
well, far cheaper. The expensive long context only becomes worth its cost later, once the model is ready to
use it.

That suggests a schedule on the sequence length itself: start short, grow long. Begin training at a tiny
length — 32 tokens — where attention is cheap and the model races through the local-structure phase, then
*double* the length periodically as the run progresses, up to the full target length (512). Early steps are
cheap and exactly matched to what the model can learn; long-context steps only happen once the model can
actually exploit them. And there's a free win in throughput: when I halve the sequence length I can roughly
double the batch (same token budget per step, much cheaper attention), so the short-sequence phase is fast
on two axes at once. I'll grow length and rescale batch together:

```python
def grow_sequence_length(current_sequence_length, current_max_batchsize, current_batchsize):
    current_sequence_length = min(2 * current_sequence_length, hyp['misc']['sequence_length']['max'])
    current_max_batchsize = round(batchsize * hyp['misc']['sequence_length']['max'] / current_sequence_length)
    if current_batchsize >= current_max_batchsize:    # at peak batch: halve it when doubling length, to avoid OOM
        current_batchsize = min(current_batchsize // 2, current_max_batchsize)
    return current_sequence_length, current_max_batchsize, current_batchsize
```

Now a problem this immediately creates, and it's the interesting part. My baseline injects order with a
*learned absolute-position embedding*: one trainable vector per position index, added to the token
embedding. That embedding is shaped by the sequence length. If I train mostly at length 32, only the first
32 position vectors ever get gradients; positions 33–512 are never seen during the cheap phase and stay
near their initialization. Then I jump to length 512 and suddenly the model has to use position vectors it
has barely trained. Worse, absolute-position embeddings don't *generalize* across lengths at all — a
representation learned for "position 30 in a 32-long window" carries no guarantee about "position 30 in a
512-long window." So the very schedule that saves me time breaks the position mechanism it sits on top of.
Growing the length needs a way of encoding order that is *length-agnostic* — that means the same thing at
32 and at 512 and trains uniformly regardless of the current window.

What does order actually need to convey to attention? Mostly *relative distance*: a token usually cares
more about nearby tokens than far ones, and it cares about *how far* another token is, not its absolute
index. If I encode order as a function of the gap (i − j) between query position i and key position j, and
add it directly to the attention logits, then it's automatically length-agnostic — the gap "5 tokens back"
means the same thing whether the window is 32 or 512. The simplest such function is *linear* in the
distance: a bias that grows linearly with how far back a key is, pushing attention to decay (or strengthen)
smoothly with distance. The slope is the only thing the model needs to learn, and one scalar slope per
attention layer lets each layer pick its own effective attention range. This is a learnable linear
positional bias on the logits — order as an additive, distance-linear, length-independent term, not as a
per-position vector.

Let me build it. Precompute a base matrix of signed distances, `linear_encoding_base[i,j]`, that encodes
the gap between every query position i and key position j. Make the slope a learnable parameter
`linear_encoding_scaler`, passed through a softplus (with an LR multiplier so the scalar can move at a
useful rate) so the effective slope stays non-negative and the magnitude is learned smoothly. Then the
positional contribution to the logits is just `softplus(mult·scaler) · linear_encoding_base`, added inside
the causal mask — wherever the causal mask permits attention, add the linear distance bias; everywhere else
fill −∞ so the softmax zeros it:

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

Notice the slice `[:x.shape[1], :x.shape[1]]` — because the bias is just a function of distance, I can
build it once at the maximum length and crop it to whatever the current sequence length is. That is exactly
the length-agnosticism I needed: the same bias matrix serves length 32 and length 512, and the slope I
learn at 32 is the same slope that's used at 512. The learned absolute-position embedding is gone — there's
nothing length-specific left to break when I grow the window.

So the two changes lock together: the sequence-length schedule (start at 32, double up to 512, growing
batch as length shrinks) gives the cheap-early/expensive-late curve I wanted, and the learnable linear
positional bias is the order-encoding that makes that schedule possible by being identical across lengths.
The bet against the previous ~3.5-minute rung: the early phase was paying full-length quadratic attention
to learn local structure that a 32-token window learns just as well far more cheaply, and front-loading the
run with short, large-batch steps should shave the wall-clock again. The risk is that the linear bias is a
weaker order representation than a fully-learned per-position embedding and costs a little final quality;
the per-layer learnable slope is the hedge, and as long as the model still lands at ~3.8 the saved time is
real.
