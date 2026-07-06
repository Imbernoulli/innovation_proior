I want to train a small GPT to ~3.8 val loss on WikiText-103 on one A100, and I want every later speedup
to be measured against something I actually trust. So before I optimize anything, I need a baseline that
is *known-correct* — a GPT that reaches the bar by the book — so that when I start cutting corners I can
tell whether a change is a real speedup or a silent regression hidden by the noise. The temptation is to
start clever; the discipline is to start boring and fast-to-reason-about, then earn every deviation.

It's worth being explicit about why a short run to a fixed, modest loss is a legitimate object to optimize
at all, rather than a toy that would rank methods differently from a real training run. A language-model
loss curve is steep at the start and flattens: the first nats come almost for free as the model learns
letter- and word-frequency statistics, and each further nat costs disproportionately more compute as it
chases rarer, longer-range structure. Placing the target at ~3.8, where the curve is beginning to bend but
hasn't yet entered the region of severe diminishing returns, means the run has to learn genuine temporal
and topical dependencies — it can't be won by frequency-matching alone — while I avoid paying for the long,
expensive tail. And because most method changes that help at the knee also helped on the way up to it, the
ranking of two training recipes at ~3.8 is a good stand-in for their ranking at convergence, at a small
fraction of the wall-clock. That is the whole bet of the speedrun format, and it's what makes iterating on
this baseline cheap enough to iterate on at all.

Before I write a line of it, let me pin down what "~3.8 val loss" concretely asks the model to do, because
that number is the whole contract. Cross-entropy of 3.8 nats is a perplexity of e^3.8; working that out,
e^3.8 = e^3 · e^0.8 ≈ 20.09 · 2.2255 ≈ 44.70 — which is exactly the "44.7 perplexity" the target is
quoted at, so the two numbers are the same statement and I've got the bar unambiguously pinned. The other
end of the scale is the cold model: at initialization, over a padded vocabulary of 50304 with no
information at all, the best it can do is the uniform distribution, whose cross-entropy is ln(50304) ≈
10.83 nats — perplexity 50304. So the run has to carry the model from perplexity ~50304 down to ~44.7,
about 7 nats of cross-entropy, a factor of ~1130 in perplexity, and then *stop* the instant it crosses 3.8,
because every nat past the bar is wall-clock I'm not being paid for. That reframing — cut perplexity by
three orders of magnitude and quit — is what turns "as fast as possible to a fixed bar" into a crisp target
rather than a vibe.

And I should be honest about how much data there is to learn from, because it bounds what "reach the bar"
even means. WikiText-103 is ~100M tokens of Wikipedia; stored as GPT-2 BPE ids in 16-bit integers (the
vocab fits under 65536) that's ~200MB, which sits comfortably resident on the 40GB A100 alongside a 30M-param
model, so batches can be drawn by random-offset sampling straight from GPU memory with no host↔device
transfer in the training loop — one less thing between me and the tensor cores. The scale ratio is worth a
glance too: ~100M tokens against ~30M parameters is a few tokens per parameter, far under the ~20-ish that a
compute-optimal run would want, so this is firmly a step-limited regime where the model will revisit the
corpus rather than run out of it. That's the setting in which weight decay earns its place — a small model
seeing a small corpus more than once is exactly where a little regularization keeps late training from
memorizing instead of generalizing — and it's why I keep AdamW's decay on rather than off.

What is the most boring correct GPT? A decoder-only transformer. Embed tokens. Add position information,
because attention is permutation-equivariant and a language model has to know token order; the standard
move is a learned absolute-position embedding — one vector per position, added to the token embedding —
so I'll take that. Then a stack of residual blocks. Each block is pre-norm: layer-norm, then the sublayer,
then add back to the residual stream. I'll keep the two classic sublayers. The first is causal
self-attention: project to queries/keys/values, attend with a lower-triangular mask so position t can
only see ≤ t, recombine. The second is a position-wise MLP that widens the channels, applies a
pointwise nonlinearity, and projects back; the conventional width is 4× and the conventional nonlinearity
is GELU. After the stack, a final norm and a linear projection to vocabulary logits. Cross-entropy against
the next token. That's it — that's a GPT. And the premise of this whole exercise is that a short run to a
flattening-point loss carries most of the signal of a long one, so a small, fast configuration of exactly
this is what I want.

Before I decide which of the boring choices are load-bearing, let me count where the parameters and the
FLOPs actually live, because that tells me which "defaults" are doing real work and which are decoration.
The token embedding is vocab × d = 50304 × 384 ≈ 19.3M parameters — one matrix, and by a wide margin the
largest single tensor in the model. Each block carries an attention sublayer with in/out projections of
4d² = 4·384² ≈ 0.59M, and a 4× MLP of 8d² ≈ 1.18M, so ~1.77M per block and ~10.6M across the six blocks.
The learned position table is 256 × 384 ≈ 0.098M, a rounding error. Adding the final norm, the total is
~30.0M parameters, of which the embedding matrix is ~19.3M — about 64%, two-thirds of the whole model in
one tensor. That single fact justifies two of my "defaults" on the spot. Weight-tying the input embedding
and the output projection means I store and train that 19.3M matrix once instead of twice — untied, a
separate output head would add another 19.3M and push the model to ~49.3M, so tying removes ~39% of the
parameters — and it is a well-understood aid to small models besides. And rounding the vocabulary up to
50304, a multiple of 64, keeps the *biggest matmul in the model* tensor-core-aligned rather than padded
awkwardly by the kernel.

It's not only parameters — the vocab head dominates the arithmetic too, which surprised me enough to
double-check. Per token, the final projection to logits is d × vocab = 384 × 50304 ≈ 19.3M multiply-adds,
about 38.6M FLOPs; the entire six-layer stack is roughly 6 × 3.93M ≈ 23.6M FLOPs per token (each layer:
8d² ≈ 1.18M for the attention projections, 4·L·d ≈ 0.39M for the score-and-value mixing at L = 256, and
16d² ≈ 2.36M for the MLP). So *more forward compute goes into the single vocabulary projection than into
all six transformer blocks combined*, and the cross-entropy itself is taken over a 50304-way distribution
whose logit tensor B × L × 50304 is the largest activation the model ever holds. Weight-tying and vocab
alignment aren't cosmetic bookkeeping; they sit directly on the hottest matmul I have. Worth noting for
later, too: of that 3.93M per-layer compute, the quadratic attention mixing at L = 256 is 0.39M — exactly
10% — so at this width the "attention is quadratic" cost is a *minority* of a block's arithmetic, a fact
I'll want to remember before I get excited about attacking sequence length.

A few more details I should fix the way the well-understood reference does, so the baseline is trustworthy
rather than idiosyncratic. Initialize the residual projections — the second linear in the MLP and the
attention output projection — with a standard deviation scaled down by 1/√(2·num_blocks). This one is
worth deriving rather than asserting, because it's the difference between a stream that stays calm and one
that's already several times too hot by the top block. Every sublayer writes x ← x + sublayer(LN(x)), so
the residual stream accumulates the outputs of *all* sublayers. With six blocks and two sublayers each,
that's 2·6 = 12 additions into the stream over the depth of the network. If each sublayer's output carried
variance σ² at init, the twelve independent additions would leave the stream with variance ~12σ² — it grows
linearly with depth, and the late blocks would be reading an input dominated by the pile-up of everything
before them rather than by their own normalized signal. Scaling the residual-projection init std by 1/√12
scales each sublayer's *output* variance by 1/12, so the twelve additions now contribute ~12 · (σ²/12) = σ²
total — the stream variance stays O(1) independent of depth. Concretely, the base init std of 0.02 becomes
0.02/√12 = 0.02/3.464 ≈ 0.00577 on the MLP-project and attention-out weights. That is the same reasoning
that stabilizes deep residual nets at initialization, specialized to twelve residual additions.

Use AdamW, the transformer default, with weight decay. And use a warmup-then-decay learning-rate schedule,
because a cold transformer hit with a large LR diverges and a warmup ramp is the standard cure; I'll use
the one-cycle form since it's the native scheduler and gives a clean single-peak curve. Here I want to be
concrete about the actual numbers, because the schedule's shape is a thing I'll be retuning constantly as
the run gets faster. With max_lr = 2e-3 and div_factor = 1e2, one-cycle starts the LR at max_lr/100 = 2e-5,
ramps it to the 2e-3 peak over the warmup fraction, then cosine-anneals. The final_div_factor = 0.05 sets
the floor at initial_lr / 0.05 = 2e-5 / 0.05 = 4e-4 — so it does *not* decay to zero; it settles at 4e-4,
about a fifth of the peak. I take the native one-cycle form because the total step count and the warmup
percentage are the two knobs I know I'll be re-fitting as the dynamics change, and it's the scheduler wired
into the scaffold; the exact anneal shape is a default I'm choosing not to fight yet.

Now the part that's specific to *this* task: the scale and the precision, chosen so a run finishes in
minutes, not hours. The model is small — a residual width of 384, six heads, six blocks — because I want
the experiment cycle short. Sequence length 256: short, which keeps the quadratic attention cost down at
baseline. Vocabulary rounded up to 50304, a multiple of 64, so every matmul shape is tensor-core-friendly.
And I'll run in bf16 on the A100's tensor cores. This is the one place I'm deliberately trading numerics
for speed, so it's worth being precise about the bits. fp32 is 1 sign / 8 exponent / 23 mantissa; bf16 is
1 / 8 / 7; fp16 is 1 / 5 / 10. The thing that matters is the *exponent* width: bf16 keeps fp32's full 8
exponent bits, so it covers the same dynamic range (up to ~3.4e38), and the only thing it gives up relative
to fp32 is mantissa precision — 7 bits, roughly two decimal digits. fp16, by contrast, has only 5 exponent
bits and tops out near 65504, which is exactly why fp16 training needs loss-scaling to keep small gradients
from underflowing. bf16 doesn't: its range *is* fp32's, so gradients and activations that fit in fp32 fit
in bf16, and I skip loss-scaling entirely, dodging a whole class of tuning that isn't correctness-neutral.
On the A100 the bf16 tensor cores run matmuls at roughly double the fp32 rate, so this is close to a free
2× on the heavy matmuls, with only mantissa precision at risk. Concretely I turn on TF32 for the fp32
fallback paths and wrap the forward in a bf16 autocast, so the heavy matmuls run in bf16 while the
reductions — softmax, layernorm, the cross-entropy — stay in fp32. That mixed-precision policy is the safe
default, and "plain autocast" is another default I'm choosing not to optimize yet.

Two more A100-era levers I'll take for free because they're correctness-neutral. First, call the fused
scaled-dot-product-attention path rather than a hand-rolled softmax(QKᵀ) that materializes the full N×N
matrix. The reason is memory, not FLOPs: a materialized score tensor is B × heads × L × L, and at L = 256
with six heads that's B · 6 · 65536 floats written to and read from HBM, whereas the flash path tiles the
softmax online and never writes that tensor out. Second, `torch.compile` the network so the graph is fused
ahead of time — and I'll compile it *inside* the timed region, so the compilation cost is counted honestly
against the wall-clock rather than hidden. A speedrun that hides its compile time is lying to itself about
the clock it's setting.

For the batching, the only subtlety at small scale, and the one I trust least: a width-384 six-block model
is tiny, so a single microbatch already fits comfortably and I can use a fairly large batch directly; where
I still want a larger *effective* batch (mostly near the end of training, where the gradient noise needs
averaging down), I accumulate a fixed number of microbatches before stepping. I want to flag this loudly,
because it's the assumption I'm least willing to defend. Gradient statistics are not stationary over a run:
early on the gradient is large and points obviously downhill, so even one noisy microbatch already points
the right way; late on the gradient is small and a large *fraction* of it is sampling noise, so averaging
many microbatches is what keeps a step from being dominated by that noise. A single constant accumulation
count has to serve both regimes, and whatever number I pick is wrong somewhere — too much averaging early
(paying forward/backward passes to average down noise that isn't hurting me), too little late (noisy steps),
or a compromise that's mediocre at both ends. I'm setting it to a constant now because it's the obvious
first choice and I want the baseline boring, but "a constant for the whole run" is an assumption I am
explicitly *not* justifying; it's the first place I'd look for time later.

One more shape to check while I'm choosing the head count, because it has to divide evenly or the attention
kernel won't even run. Six heads over a width of 384 gives a head dimension of 384/6 = 64, and 64 is about
as friendly a head dim as I could ask for on this hardware — a multiple of 8, so the flash attention kernel
takes its fast path, and small enough that the per-head query/key/value slabs are cheap. Six heads of 64 is
also a reasonable amount of attention diversity for a model this size: enough distinct subspaces to route
different kinds of dependency, not so many that each head is starved of dimensions. So 6 heads it is, and
the divisibility check (6 · 64 = 384) is the trivial-but-load-bearing kind — get it wrong and nothing runs.

One shape I want to get exactly right before I trust any loss number, because a causal LM is only as honest
as its mask. The mask must forbid position i from attending to any key j > i. Built as
`logical_not(triu(ones(L,L))).T`: `triu(ones)` is upper-triangular including the diagonal, True where j ≥ i;
`logical_not` flips that to True where j < i (strictly lower); the transpose then sends True to where i < j,
i.e. True exactly where j > i. In the MultiheadAttention boolean-mask convention True marks a *forbidden*
attention, so True-where-j>i is precisely causal. Let me verify at the smallest case that can catch a
future-leak: L = 2, positions {0,1}. Position 0 must see only itself, position 1 must see both — so exactly
three of the four query/key entries are allowed, and the single forbidden entry is (query 0, key 1). My
constructed mask marks True at (0,1) and nowhere else, matching. If that one entry were ever *not* masked,
the model would peek one token into the future and the val loss would collapse to something implausibly low;
that discrepancy is my canary if a later refactor ever breaks the mask.

Let me write the two blocks. The attention block, pre-norm with the causal mask, using the framework's
multi-head attention:

```python
class AttentionBlock(nn.Module):
    def __init__(self, num_features, sequence_length, num_heads):
        super().__init__()
        self.norm = LayerNorm(num_features, bias=False)
        self.attention = nn.MultiheadAttention(num_features, num_heads, bias=False, batch_first=True)
        # lower-triangular causal mask: position t may attend only to positions <= t
        self.causal_mask = torch.logical_not(
            torch.triu(torch.ones((sequence_length, sequence_length), device='cuda', dtype=torch.bool))).T
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x, _ = self.attention(x, x, x, attn_mask=self.causal_mask, need_weights=False)
        return x + residual

class MLPBlock(nn.Module):
    def __init__(self, num_channels, expansion_factor=4):
        super().__init__()
        self.norm = LayerNorm(num_channels, bias=False)
        self.expand  = nn.Linear(num_channels, num_channels*expansion_factor, bias=False)
        self.project = nn.Linear(expansion_factor*num_channels, num_channels, bias=False)
        self.activation = nn.GELU()
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.project(self.activation(self.expand(x)))
        return x + residual
```

And the net assembly: token embedding, learned absolute-position embedding, the interleaved attention/MLP
blocks, a final norm, the weight-tied output projection, the 1/√(2·num_blocks) residual-projection init,
AdamW, and a one-cycle schedule:

```python
network_dict = nn.ModuleDict({
    'embedding': nn.Embedding(hyp['misc']['num_tokens'], hyp['net']['residual_depth']),
    'position':  PositionEmbedding(hyp['misc']['sequence_length'], hyp['net']['residual_depth']),
    'norm':      LayerNorm(hyp['net']['residual_depth'], bias=False),
    'attn_layers': nn.ModuleList([AttentionBlock(hyp['net']['residual_depth'], hyp['misc']['sequence_length'], hyp['net']['num_heads']) for _ in range(hyp['net']['num_blocks'])]),
    'mlp_layers':  nn.ModuleList([MLPBlock(hyp['net']['residual_depth']) for _ in range(hyp['net']['num_blocks'])]),
    'outputs':     nn.Linear(hyp['net']['residual_depth'], hyp['misc']['num_tokens'], bias=False),
})
net.net_dict['embedding'].weight = net.net_dict['outputs'].weight   # weight tying
# residual-projection init scaled by 1/sqrt(2*num_blocks) so the residual stream variance stays bounded with depth
opt = torch.optim.AdamW([params_non_decay, params_decay], fused=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hyp['opt']['lr'],
              total_steps=hyp['opt']['total_train_steps'], pct_start=hyp['opt']['warmup_percent'],
              anneal_strategy='cos', cycle_momentum=False, div_factor=1e2, final_div_factor=.05)
net = torch.compile(net)   # compiled inside the timed region, so compile time counts honestly
```

This is the floor: a textbook GPT, bf16 on the A100, fused attention, compiled — ~30M parameters, of which
two-thirds is the tied vocab matrix, carrying the model from perplexity ~50304 down to the 44.7 bar. It
should reach the bar; that's the whole point of starting from a known-correct recipe, and I'm not going to
pretend I know the wall-clock in advance — I expect single-digit minutes on this hardware at this scale, and
whatever it turns out to be is the clock every later rung has to beat. What I'm watching is *where* that time
goes, because every assumption I just made the boring way is a place I deliberately did not optimize, and
therefore a place a later run might find time to give back: the fixed accumulation count I flagged; the fixed
256 sequence length paying full attention from the very first step, even though that quadratic cost is only
~10% of a block's arithmetic here so I shouldn't expect the world from touching it; the GELU MLP that is the
single biggest chunk of per-block compute and whose nonlinearity is a *fixed* pointwise gate — the same
smooth "is this positive" curve applied everywhere, with only the surrounding linear maps learned, so I've
done nothing to ask whether a richer, learned gate would earn its keep; the learned absolute positions,
tied to specific index slots rather than to distances; and the plain autocast policy carrying an fp32 safety
margin bf16's range may not actually need. The baseline's job is to be trustworthy and to set
the clock; beating that clock is the rest of the ladder.
