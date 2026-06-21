The dynamic effective-batch controller cut the run nearly in half, to ~3.5 minutes. With the *effective batch* now adapting to the run, the other big knob I froze at a constant in the baseline stands out: the *sequence length*, 256 tokens, the same at step one and at the last step. Attention is quadratic in length, so every token of context costs proportionally to the context carried — and I'm carrying the full 256 from the very first step. The question is whether that context is earning its cost early on. At initialization the network knows nothing, and the first thing it picks up is *local* structure: which token tends to follow which, short bigram/trigram-scale dependencies. Long-range dependencies — this pronoun refers to that subject forty tokens back — can only be learned *after* the local structure is in place, because they are built on top of it. So early in training, feeding 256 tokens of context is mostly wasted: it pays quadratic attention cost over a long window when all the model can extract is short-range signal a much shorter window would have given just as well, far cheaper. The long context only becomes worth its cost later.

So I propose two changes that lock together: **sequence-length scheduling** plus a **learnable linear positional bias**. The schedule is the direct consequence of the observation above — start short, grow long. Training begins at a tiny length of 32 tokens, where attention is cheap and the model races through the local-structure phase, then I *double* the length periodically as the run progresses, up to the target 512. Early steps are cheap and exactly matched to what the model can learn; long-context steps only happen once the model can actually exploit them. There is a second free win: when I halve the sequence length I can roughly double the batch, because the token budget per step stays the same while attention gets much cheaper — so the short-sequence phase is fast on two axes at once. I grow the length and rescale the batch together, halving the batch only when I'm at peak batch and doubling the length, to avoid running out of memory.

The schedule immediately breaks the mechanism it sits on, and fixing that is the interesting part. My baseline injected order with a *learned absolute-position embedding*: one trainable vector per position index, added to the token embedding. That embedding is shaped by the sequence length — if I train mostly at length 32, only the first 32 position vectors ever receive gradients, and positions 33–512 stay near their initialization. Then I jump to length 512 and the model must suddenly use position vectors it has barely trained. Worse, absolute-position embeddings don't *generalize* across lengths at all: a representation learned for "position 30 in a 32-long window" carries no guarantee about "position 30 in a 512-long window." The very schedule that saves time breaks the position mechanism, so I need a way of encoding order that is *length-agnostic* — that means the same thing at 32 and at 512 and trains uniformly regardless of the current window.

What does order actually need to convey to attention? Mostly *relative distance*: a token cares more about nearby tokens than far ones, and it cares about *how far* another token is, not its absolute index. If I encode order as a function of the gap $(i-j)$ between query position $i$ and key position $j$ and add it directly to the attention logits, then it is automatically length-agnostic — "5 tokens back" means the same thing whether the window is 32 or 512. The simplest such function is *linear* in the distance: a bias that grows linearly with how far back a key is, so attention decays (or strengthens) smoothly with distance. The slope is the only thing the model needs to learn, and one scalar slope *per attention layer* lets each layer pick its own effective attention range — similar in spirit to additive-distance biasing of the attention logits, but with a per-layer learnable slope. Concretely I precompute a base matrix of signed distances `linear_encoding_base[i,j]` encoding the gap between every query position $i$ and key position $j$, make the slope a learnable parameter `linear_encoding_scaler`, and pass it through a softplus (with an LR multiplier so the scalar can move at a useful rate) so the effective slope stays non-negative and its magnitude is learned smoothly. The positional contribution to the logits is then $\text{softplus}(\text{mult}\cdot\text{scaler})\cdot\text{base}$, added *inside* the causal mask: wherever the causal mask permits attention, add the linear distance bias; everywhere else fill $-\infty$ so the softmax zeros it.

The load-bearing detail is the slice `[:x.shape[1], :x.shape[1]]` in the forward pass. Because the bias is purely a function of distance, I can build the base matrix *once* at the maximum length and crop it to whatever the current sequence length is — and that is exactly the length-agnosticism the schedule needs: the same bias matrix serves length 32 and length 512, and the slope learned at 32 is the same slope used at 512. The learned absolute-position embedding is gone, so there is nothing length-specific left to break when I grow the window. The two changes therefore compose cleanly: the length schedule front-loads the run with cheap, large-batch short-context steps and removes the quadratic cost of carrying long context the model can't yet use, while the linear positional bias is what makes that schedule possible by meaning the same thing at every length. The risk is that a linear bias is a weaker order representation than a fully-learned per-position embedding and costs a little final quality; the per-layer learnable slope is the hedge, and as long as the model still lands at ~3.8 the saved time is real.

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
