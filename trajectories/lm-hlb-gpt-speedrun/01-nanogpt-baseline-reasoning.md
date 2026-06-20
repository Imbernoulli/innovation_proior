I want to train a small GPT to ~3.8 val loss on WikiText-103 on one A100, and I want every later speedup
to be measured against something I actually trust. So before I optimize anything, I need a baseline that
is *known-correct* — a GPT that reaches the bar by the book — so that when I start cutting corners I can
tell whether a change is a real speedup or a silent regression hidden by the noise. The temptation is to
start clever; the discipline is to start boring and fast-to-reason-about, then earn every deviation.

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

A few details I should fix the way the well-understood reference does, so the baseline is trustworthy
rather than idiosyncratic. Weight-tie the input embedding and the output projection: they're both maps
between the residual space and the vocabulary, sharing them removes a big parameter block (vocab is 50304,
the largest single tensor in the model) and is known to help small models. Initialize the residual
projections — the second linear in the MLP and the attention output projection — with a standard deviation
scaled down by 1/√(2·num_blocks). This matters: every block *adds* its output into the residual stream, so
if each block's output had unit-ish variance, the residual stream's variance would grow with depth and the
late blocks would be fighting an exploding-magnitude signal. Scaling the residual-projection init by
1/√(2·num_blocks) keeps the variance added per block small enough that the stream stays well-behaved
through all the blocks — it's the same logic that keeps deep residual nets stable at init. Use AdamW, the
default for transformers, with weight decay. And use a warmup-then-decay learning-rate schedule, because a
cold transformer with a large LR diverges and a warmup ramp is the standard cure; I'll use the one-cycle
form (ramp up over a small warmup fraction, then anneal toward near-zero) since it's the native scheduler
and gives a clean single-peak curve.

Now the part that's specific to *this* task: the scale and the precision, chosen so a run finishes in
minutes, not hours. The model is small — a residual width of 384, six heads, six blocks — because I want
the experiment cycle short. Sequence length 256: short, which keeps
the quadratic attention cost down at baseline. Vocabulary rounded up to 50304, a multiple of 64, so every
matmul shape is tensor-core-friendly. And I'll run in bf16 on the A100's tensor cores — bf16 has fp32's
exponent range so I don't need fp16 loss-scaling, and it roughly doubles matmul throughput over fp32.
Concretely that means turning on TF32 for the fp32 fallback paths and wrapping the forward in a bf16
autocast so the heavy matmuls run in bf16 while the reductions stay in fp32. Two more A100-era levers I'll
take for free because they're correctness-neutral: call the fused/flash scaled-dot-product attention path
rather than a hand-rolled softmax(QKᵀ) that materializes the full N×N matrix, and `torch.compile` the
network so the graph is fused ahead of time — and I'll compile it *inside* the timed region, so the
compilation cost is counted honestly against the wall-clock rather than hidden.

For the batching, the only subtlety at small scale: a width-384 six-block model is tiny, so a single
microbatch already fits comfortably and I can use a fairly large batch directly; where I still want a
larger *effective* batch (mostly near the end of training, where the gradient noise needs averaging down),
I accumulate a fixed number of microbatches before stepping. I'll set that accumulation count to a
constant for now — a number that works — and note that "a constant for the whole run" is an assumption I'm
making, not a thing I've justified. It's the obvious first choice; whether it's the *right* policy is a
question for later.

Let me write the two blocks. The attention block, pre-norm with a causal mask, using the framework's
multi-head attention with the triangular mask:

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

This is the floor: a textbook GPT, bf16 on the A100, fused attention, compiled. It should reach the bar —
that's the whole point of starting from a known-correct recipe. What I'm watching is how *long* it takes,
because every assumption I just made the boring way — fixed sequence length, fixed accumulation count, a
GELU MLP, learned absolute positions, a plain autocast precision policy — is a place I deliberately did not
optimize, and therefore a place a later run might find time to give back. The baseline's job is to be
trustworthy and to set the clock; beating that clock is the rest of the ladder.
