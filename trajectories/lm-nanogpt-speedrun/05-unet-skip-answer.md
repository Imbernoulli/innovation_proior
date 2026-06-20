**Problem (from step 4).** The shortcuts record reached 8.2 min / 3200 steps, and the embed shortcut (x0
fed to every block) showed that giving a deep block a direct path back to an *earlier* representation pays
off. But x0 is only the degenerate, single-source case: deep blocks could also benefit from direct paths to
the richer, partially-processed activations from the middle of the network, not just the raw embedding.

**Key idea (U-net skip connections over depth, by @brendanh0gan).** Generalize the embed shortcut from
"one source → every block" to the full encoder/decoder U-net pattern over the transformer's depth. Treat the
first half of the 12 blocks as an **encoder** and the second half as a **decoder**. Save each encoder
block's output on a stack; in the decoder, `pop()` before each block and add the popped encoder activation
back in, scaled by a **learnable per-decoder-layer skip weight** initialized to one. LIFO popping gives the
classic symmetric U-net pairing for free — encoder layer i pairs with decoder layer (num_decoder_layers − 1
− i), so the freshest encoder layer feeds the last decoder layer and the deepest encoder layer feeds the
first. Because these long skips shorten the gradient paths and better-condition the landscape, **double the
learning rate**.

**Why it works.** The decoder reuses the encoder's cleaner intermediate features directly instead of
reconstructing them from the residual stream alone, and the long skips give the head's gradient a direct
route back to encoder layers — the same property that makes residual nets trainable at depth, now operating
across the whole network. A better-conditioned landscape tolerates a 2× learning rate without diverging, so
the same loss drop is covered in fewer steps. The skip weights start at one (full pass-through) and are tiny
1-D params optimized as scalars by Adam.

**Change / code.** First `n_layer // 2` blocks encode and push outputs; remaining blocks decode, each adding
`skip_weights[i] * skip_connections.pop()` before running; learning rate doubled.

```python
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        # U-net design by @brendanh0gan
        self.num_encoder_layers = config.n_layer // 2   # first half = encoder
        self.num_decoder_layers = config.n_layer - self.num_encoder_layers
        # learnable skip-connection weights for the decoder layers
        self.skip_weights = nn.Parameter(torch.ones(self.num_decoder_layers))
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
        ))
        self.lm_head = CastedLinear(config.n_embd, config.vocab_size)
        self.lm_head.weight.data.zero_()

    def forward(self, idx, target):
        x = norm(self.transformer.wte(idx[None]))
        x0 = x
        # Encoder pass — first half of the blocks
        skip_connections = []
        for i in range(self.num_encoder_layers):
            x = self.transformer.h[i](x, x0)  # block also threads the value-residual signal (rung 4)
            skip_connections.append(x)
        # Decoder pass — remaining blocks with weighted skip connections
        for i in range(self.num_decoder_layers):
            x = x + self.skip_weights[i] * skip_connections.pop()
            x = self.transformer.h[self.num_encoder_layers + i](x, x0)
        x = norm(x)
        logits = 30 * torch.tanh(self.lm_head(x).float() / 30)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))
        return loss

# the skip weights are 1D params optimized as scalar_params by Adam
scalar_params = [p for p in params if p.ndim < 2] + [raw_model.skip_weights]
```
