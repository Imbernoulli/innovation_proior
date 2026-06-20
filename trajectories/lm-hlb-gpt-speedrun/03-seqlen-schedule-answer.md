**Problem (from rung 2).** The sequence length is still frozen at one value for the whole run, so every
step pays full quadratic attention cost. But early in training the model can only learn short-range
structure — a long context window is wasted compute until the model is ready to use it. Cutting the
early-run length, though, breaks the baseline's learned absolute-position embedding: at short lengths only
the first few position vectors get gradients, and absolute embeddings don't generalize across window sizes.

**Key idea.** Two locked-together changes. (1) **Sequence-length scheduling**: start at length 32, double
periodically up to the target 512, and grow the batch as the length shrinks (same token budget, far cheaper
attention) — cheap, large-batch early steps matched to the local structure the model learns first;
expensive long-context steps only once it can exploit them. (2) **Learnable linear positional bias**:
replace the per-position absolute embedding with an additive, distance-linear bias on the attention logits
— `softplus(mult·scaler) · base`, where `base[i,j]` is the signed query–key distance and `scaler` is one
learnable slope per layer. Because it is a function of the gap only, it is length-agnostic: build it once
at max length and crop it to the current window, so the slope learned at 32 is the same slope used at 512.

**Why it works.** The length schedule front-loads the run with cheap short-context steps, removing the
quadratic cost of carrying long context the model can't yet use; the linear positional bias is what makes
that schedule possible by meaning the same thing at every length, so growing the window breaks nothing.
Per-layer learnable slopes let each layer pick its own effective attention range.

**Change / code.** The length-growth function and the attention block with the learnable linear bias.

```python
hyp['misc']['sequence_length'] = {'max': 512, 'initial': 32, 'growth_steps': 250}

def grow_sequence_length(current_sequence_length, current_max_batchsize, current_batchsize):
    current_sequence_length = min(2 * current_sequence_length, hyp['misc']['sequence_length']['max'])
    current_max_batchsize = round(batchsize * hyp['misc']['sequence_length']['max'] / current_sequence_length)
    if current_batchsize >= current_max_batchsize:
        current_batchsize = min(current_batchsize // 2, current_max_batchsize)
    return current_sequence_length, current_max_batchsize, current_batchsize

class AttentionBlock(nn.Module):
    def __init__(self, num_features, sequence_length, num_heads):
        super().__init__()
        self.norm = LayerNorm(num_features, bias=False)
        self.attention = nn.MultiheadAttention(num_features, num_heads, bias=False, batch_first=True)
        # learnable linear positional bias on the logits (length-agnostic, one slope per layer)
        self.linear_encoding_lr_mult = 50.
        self.linear_encoding_scaler = nn.Parameter(torch.tensor(-.05 / self.linear_encoding_lr_mult, device='cuda'))
        self.linear_encoding_base = (torch.arange(-sequence_length+1, 1, dtype=torch.float, device='cuda').unsqueeze(0)
                                     + torch.arange(sequence_length-1, -1, -1, dtype=torch.float, device='cuda').unsqueeze(1))
        self.linear_encoding_mask = lambda mask, base, scaler: torch.where(
            mask, F.softplus(self.linear_encoding_lr_mult * scaler) * base, torch.full_like(base, -float('inf')))
        self.causal_mask = torch.tril(torch.ones((sequence_length, sequence_length), device='cuda', dtype=torch.bool))

    def forward(self, x):
        residual = x
        x = self.norm(x)
        attn_mask = self.linear_encoding_mask(self.causal_mask, self.linear_encoding_base, self.linear_encoding_scaler)
        # crop the once-built bias to the current sequence length — same slope serves every length
        x, _ = self.attention(x, x, x, attn_mask=attn_mask[:x.shape[1], :x.shape[1]], need_weights=False)
        return x + residual
```
