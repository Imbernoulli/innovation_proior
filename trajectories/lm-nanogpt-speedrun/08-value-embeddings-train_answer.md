Growing the window linearly from 64 to ~1792 tokens made early steps cheaper and acted as a short-to-long-context curriculum: the step count dropped to 1750 at a slightly lower per-step time and a steadier val loss of 3.2759, ~4.66 minutes. The body, the optimizers, and the attention schedule are all tuned, so I go back to one of the cheapest wins on the ladder and ask whether I got everything out of it — the value residual. When attention concentrates with depth, late layers lose the clean early value signal, and the fix gave every attention layer a line back to the first block's value $v_1$, mixed in with a learnable per-layer lambda, $v = (1-\lambda)v + \lambda v_1$. That helped, and the lambdas confirmed the layers actually wanted the early signal. But look hard at *what* that signal is. $v_1$ is the first block's *computed* value — the embedding, RMS-normed, passed through one attention layer's value projection $c_v$ — so when block 8 reaches back for "clean early value," what it gets is already one block deep, the embedding seen through the lens of the first layer's learned $c_v$ and mixed with whatever the first block's residual update did. It is cleaner than block 8's own value, but it is a *derived* quantity, and every layer that wants it must take the *same* $v_1$, because there is only one first block. That overloads the first block's $c_v$ into two jobs — block 1's own value projection *and* the universal early-value source for all eleven later layers — the same overloading I broke when I untied the embedding from the head.

I propose **per-layer token value embeddings**: keep the value-residual lambda mechanism exactly, but replace the *content* it mixes in. Give every attention layer a direct line not to a derived value but to a *learned, token-identity-based* value — looked up from a table indexed by token id, never carried through the residual stream or filtered through the first block's projection. The residual stream is the network's working memory, and the deeper a block sits the more processed and lossy its view of the raw token; a table indexed by token id sidesteps that entirely. For token "the" it holds a value vector for layer 5 to mix in and a (possibly different) one for layer 9, and these are parameters the optimizer sets directly, with no constraint that they be reconstructible from the residual stream. It is the value-stream analogue of what `wte` already does for the *input* — a learned, token-identity seed — now seeding the *value* of every attention layer.

Why this beats reaching back to $v_1$ is that $v_1$'s content is bottlenecked: the first block computes one vector per token, and that single vector has to encode whatever all downstream layers collectively need from the early signal — but a shallow layer and a deep layer want different aspects of the token's identity, and one vector cannot give them different things. A table can. I make it wide enough to hold a *separate* value slice per layer, so layer $i$ looks up its own dedicated per-token value and the optimizer shapes each layer's slice independently; the first block's $c_v$ is freed to just be block 1's value projection again. Concretely I add a second embedding table `vte`, the value-token-embedding, alongside `wte`. `wte` has width `n_embd`; `vte` I make width `n_embd*12` — one `n_embd`-wide slice for each of the twelve layers — look it up once from the token ids exactly like `wte`, and chunk the result into twelve per-layer slices. Layer $i$ receives its slice $v_i$, and inside the attention forward I mix it into the freshly-projected value with the *same* learnable-lambda machinery I already built: `v = (1 - self.lamb)*v + self.lamb * vi.view_as(v)`. The mechanism is identical — the per-layer lambda, the convex mix, the starting behavior the model can dial — but the thing being mixed in has changed meaning, from one shared computed $v_1$ to this layer's dedicated slice of a learned, token-indexed table, independent of the residual stream and of the first block.

The cost is real but bounded. The `vte` table is `vocab_size × (n_embd*12)`, twelve times the width of `wte` — a genuinely large table — and being a lookup table indexed by token id it belongs with `wte` under Adam, not Muon (orthogonalizing matrix updates is the wrong tool for an embedding table): `optimizer1 = torch.optim.Adam([wte.weight, vte.weight], lr=0.6, betas=(0.8, 0.95), fused=True)`. There is also a per-step compute cost — every forward now does a second, twelve-times-wider lookup and twelve view/mix operations — which pushes the step time up a little. The bet is that the step *count* falls by more than enough to pay for it: this is not free structure the model can ignore at init the way the embed shortcut was, since the table has to *learn* to be useful, but lambda controls how much leaks in and could suppress a useless table, so the curve should look slightly worse very early and then pull ahead as the per-token value slices sharpen. Given how much the lambdas wanted $v_1$ in the first place, I expect the layers to want a dedicated, richer version of that signal even more, and the net step count at the bar to come down by more than the extra lookup time costs.

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
