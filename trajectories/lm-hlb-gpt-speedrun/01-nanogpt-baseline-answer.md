**Problem.** Train a small GPT to the ~3.8 val-loss bar on WikiText-103 on one A100, starting from a
*known-correct* recipe so later speedups can be measured against something trustworthy.

**Key idea (the baseline).** A textbook nanoGPT-style decoder transformer: token + learned
absolute-position embeddings; a stack of pre-norm residual blocks, each a causal multi-head self-attention
sublayer and a 4×-expansion GELU MLP; final norm; weight-tied output projection; AdamW with a one-cycle
(warmup→cosine-decay) learning-rate schedule. Make it fast on the A100 the correctness-neutral way: bf16
tensor-core math (TF32 on, bf16 autocast on the forward), the fused/flash scaled-dot-product attention
path, and `torch.compile` — compiled *inside* the timed region so compile time is counted honestly.
Residual-projection weights are init-scaled by 1/√(2·num_blocks) to keep the residual-stream variance
bounded with depth. Batching uses a fixed sequence length and a fixed gradient-accumulation count.

**Why it works.** Everything here is the well-understood default, so the model is trustworthy and reaches
the bar by the book; the A100-era levers (bf16, fused attention, compile) are math-equivalent throughput
wins. The deliberately-unoptimized choices — fixed sequence length, fixed accumulation count, GELU MLP,
learned absolute positions, plain autocast — are left in place precisely so the rest of the ladder has
somewhere to claw time back.

**Change / code.** The two block internals and the net assembly.

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
