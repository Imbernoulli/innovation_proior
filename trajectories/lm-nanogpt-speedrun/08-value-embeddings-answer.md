**Problem (from step 7).** With the window warmup landing 3.2759 in 1750 steps (≈4.66 min), the value
residual is one of the cheapest wins on the ladder — but the early-value signal it injects is *derived*.
The first block's value v1 is the embedding seen through one attention layer's c_v, a single shared
residual-stream quantity that all eleven later layers must take as-is, and that overloads the first block's
c_v into being both block 1's value projection and the universal early-value source.

**Key idea (per-layer token value embeddings).** Keep the value-residual lambda mechanism, but replace the
*content* it mixes in. Add a second embedding table `vte` (value-token-embedding) of width `n_embd*12` —
one `n_embd`-wide slice per layer. Look it up from the token ids, chunk it into 12 per-layer slices, and
mix slice i into layer i's attention value via the same learnable lambda: `v = (1 - self.lamb)*v +
self.lamb * vi.view_as(v)`. Each attention layer now gets a direct, learned, per-token value injection that
depends only on token identity, never carried through the residual stream or filtered through the first
block. `vte` is optimized by Adam alongside `wte`.

**Why it works.** v1 is one vector that has to serve every downstream layer, but a shallow layer and a
deep layer want different aspects of the token's value — a single computed v1 can't give them different
things, a per-layer learned table can. Looking the value up fresh from the token ids sidesteps the lossy
residual stream entirely, and frees the first block's c_v to be just block 1's value projection again. The
extra twelve-times-wider lookup raises the per-step time, but a dedicated, direct per-token value signal
converges in enough fewer steps that total wallclock still falls.

**Change / code.** Add `vte = nn.Embedding(vocab_size, n_embd*12)`; `vi = vte(idx[None]).chunk(12)`; mix
`vi[i]` into layer i's value with the existing lambda; put `vte.weight` in the same Adam group as `wte`.

```python
# in GPT.__init__:
self.transformer = nn.ModuleDict(dict(
    wte = nn.Embedding(config.vocab_size, config.n_embd),
    # token value embeddings by @KoszarskyB - inspired by @Grad62304977's value residual learning
    vte = nn.Embedding(config.vocab_size, config.n_embd*12),
    h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
))

# in GPT.forward, look up and chunk into one slice per layer:
vi = self.transformer.vte(idx[None]).chunk(12, dim=-1)
# ... each block i receives vi[i]:
# x = self.transformer.h[i](x, vi[i], x0, block_mask)

# in the attention forward, mix the value embedding into v with the learnable lambda:
def forward(self, x, vi, block_mask):
    B, T = x.size(0), x.size(1)
    q = self.c_q(x).view(B, T, self.n_head, -1)
    k = self.c_k(x).view(B, T, self.n_head, -1)
    v = self.c_v(x).view(B, T, self.n_head, -1)
    v = (1 - self.lamb) * v + self.lamb * vi.view_as(v)   # @Grad62304977
    q, k = norm(q), norm(k)
    q, k = self.rotary(q), self.rotary(k)
    y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask)
    y = y.transpose(1, 2).contiguous().view_as(x)
    return self.c_proj(y)

# vte optimized by Adam alongside wte:
optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight, raw_model.transformer.vte.weight], lr=0.6, betas=(0.8, 0.95), fused=True)
```
