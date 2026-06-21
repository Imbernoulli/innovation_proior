Before I optimize anything, I want a baseline I can actually trust: a GPT that reaches the ~3.8 val-loss bar on WikiText-103 on one A100 *by the book*, so that when I later start cutting corners I can tell whether a change is a real speedup or a silent regression hiding in the noise. The discipline is to start boring and fast-to-reason-about, then earn every deviation from it.

What I propose for the floor is a **nanoGPT-style decoder transformer**, made fast on the A100 only in correctness-neutral ways. The architecture is the most boring correct GPT there is. Tokens are embedded; because attention is permutation-equivariant and a language model must know token order, I add a *learned absolute-position embedding* — one trainable vector per position index, added to the token embedding — which is the standard, well-understood default. Then a stack of pre-norm residual blocks, each of the two classic sublayers wrapped as $x \mapsto x + \text{sublayer}(\text{LN}(x))$. The first sublayer is causal multi-head self-attention: project to queries/keys/values, attend with a lower-triangular mask so position $t$ can only see positions $\le t$, recombine. The second is a position-wise MLP that widens the channels by $4\times$, applies GELU pointwise, and projects back. After the stack comes a final LayerNorm and a linear projection to vocabulary logits, trained with next-token cross-entropy. The premise of the whole speedrun is that a short run to a flattening-point loss carries most of the signal of a long one, so I deliberately keep this small — residual width 384, six heads, six blocks, sequence length 256 — and I round the vocabulary up to 50304 (a multiple of 64) so every matmul shape is tensor-core-friendly.

A few details I fix the way the reference recipe does, because they are what make the baseline *trustworthy* rather than idiosyncratic. I weight-tie the input embedding and the output projection: both are maps between the residual space and the vocabulary, and since the vocabulary tensor (50304 rows) is the single largest block of parameters in the model, sharing it removes a large parameter block and is known to help small models. I scale the init of the *residual projections* — the second linear in each MLP and each attention output projection — by $1/\sqrt{2\,\text{num\_blocks}}$. This one matters and is worth being precise about: every block *adds* its output back into the residual stream, so if each block's output carried unit-ish variance, the stream's variance would grow with depth and the late blocks would be fighting an exploding-magnitude signal. Scaling the residual-projection init down by $1/\sqrt{2\,\text{num\_blocks}}$ keeps the variance added per block small enough that the stream stays well-behaved through all six blocks — the same logic that stabilizes deep residual nets at initialization. The optimizer is AdamW with weight decay (the transformer default), and the learning rate follows a one-cycle warmup-then-decay schedule, because a cold transformer hit with a large LR diverges and a warmup ramp is the standard cure; the one-cycle form ramps up over a small warmup fraction and then cosine-anneals toward near-zero, giving a clean single-peak curve.

The part specific to *this* task is the precision and the A100-era levers, chosen so a run finishes in minutes. I run in bf16 on the A100's tensor cores: bf16 has fp32's full exponent range, so I don't need the fp16-style loss-scaling, and it roughly doubles matmul throughput over fp32. Concretely I turn on TF32 for the fp32 fallback paths and wrap the forward in a bf16 autocast, so the heavy matmuls run in bf16 while reductions stay in fp32. Two more levers come for free because they are math-equivalent: I call the fused/flash scaled-dot-product-attention path inside `nn.MultiheadAttention` rather than a hand-rolled $\text{softmax}(QK^\top)$ that materializes the full $N\times N$ matrix, and I `torch.compile` the network so the graph is fused ahead of time — and I compile it *inside* the timed region so the compilation cost is counted honestly against the wall-clock rather than hidden. For batching, a width-384 six-block model is tiny, so a single microbatch fits comfortably and I use a fairly large batch directly, accumulating a *fixed* number of microbatches before each optimizer step where I want a larger effective batch.

This is the point of the baseline being honest about its defaults. Every choice I made the boring way — fixed sequence length, fixed accumulation count, a GELU MLP, learned absolute positions, a plain autocast precision policy — is a place I deliberately did *not* optimize, and therefore a place a later rung can find time to give back. The baseline's only job is to be trustworthy and to set the clock at ~3.8 val loss; beating that clock is the rest of the ladder.

```python
class AttentionBlock(nn.Module):
    """ A standard attention block. """
    def __init__(self, num_features, sequence_length, num_heads):
        super().__init__()
        self.norm = LayerNorm(num_features, bias=False)
        self.attention = nn.MultiheadAttention(num_features, num_heads, bias=False, batch_first=True)
        self.causal_mask = torch.logical_not(
            torch.triu(torch.ones((sequence_length, sequence_length), device='cuda', dtype=torch.bool))).T
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x, _ = self.attention(x, x, x, attn_mask=self.causal_mask, need_weights=False)
        return x + residual

class MLPBlock(nn.Module):
    """ A standard MLP block. """
    def __init__(self, num_channels, expansion_factor=4):
        super().__init__()
        self.norm = LayerNorm(num_channels, bias=False)
        self.expand  = nn.Linear(num_channels, num_channels*expansion_factor, bias=False)
        self.project = nn.Linear(expansion_factor*num_channels, num_channels, bias=False)
        self.activation = nn.GELU()
    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.expand(x)
        x = self.activation(x)
        x = self.project(x)
        return x + residual

# assembly: token + learned-position embeddings, interleaved attn/MLP blocks, final norm, tied output head
network_dict = nn.ModuleDict({
    'embedding':   nn.Embedding(hyp['misc']['num_tokens'], hyp['net']['residual_depth']),
    'position':    PositionEmbedding(hyp['misc']['sequence_length'], hyp['net']['residual_depth']),
    'norm':        LayerNorm(hyp['net']['residual_depth'], eps=1e-5, bias=False),
    'mlp_layers':  nn.ModuleList([MLPBlock(hyp['net']['residual_depth']) for _ in range(hyp['net']['num_blocks'])]),
    'attn_layers': nn.ModuleList([AttentionBlock(hyp['net']['residual_depth'], hyp['misc']['sequence_length'], hyp['net']['num_heads']) for _ in range(hyp['net']['num_blocks'])]),
    'outputs':     nn.Linear(hyp['net']['residual_depth'], hyp['misc']['num_tokens'], bias=False),
})
net.net_dict['embedding'].weight = net.net_dict['outputs'].weight   # weight tying

# residual-projection init scaled by 1/sqrt(2*num_blocks) to bound residual-stream variance with depth
for name, parameter in net.named_parameters():
    if (('project' in name and 'mlp' in name) or 'out_proj' in name or 'c_proj' in name) and 'weight' in name:
        torch.nn.init.normal_(parameter.data, mean=0., std=.02/((2 * hyp['net']['num_blocks'])**.5))

# A100-era levers: TF32 on, bf16 autocast, fused attention (inside MultiheadAttention), compile in the timed region
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
autocast_tensors = torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)
opt = torch.optim.AdamW([params_non_decay, params_decay], fused=True)
scheduler = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=hyp['opt']['lr'],
              total_steps=hyp['opt']['total_train_steps'], pct_start=hyp['opt']['warmup_percent'],
              anneal_strategy='cos', cycle_momentum=False, div_factor=1e2, final_div_factor=.05)
net = torch.compile(net)
```
