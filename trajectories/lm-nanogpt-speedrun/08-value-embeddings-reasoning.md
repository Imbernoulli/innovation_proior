Four point six six minutes now, 1750 steps, val_loss 3.2759 — the window warmup squeezed another chunk
of wallclock out by letting attention start narrow and grow. The body is tuned, the optimizers are tuned,
the attention schedule is tuned. So I go back to one of the cheapest wins on the ladder and ask whether I
got everything out of it: the value residual. When attention concentrates with depth, late layers lose
the clean early-layer value signal, and the fix was to give every attention layer a direct line back to
the first block's value vector v1, mixed in with a learnable per-layer lambda — `v = (1 - self.lamb)*v +
self.lamb * v1`. That helped, and the lambdas confirmed the layers actually wanted the early signal. But
let me look hard at *what* that early signal really is.

v1 is the first decoder block's *computed* value: it's the embedding, RMS-normed, passed through one
attention layer's value projection c_v. So when block 8 reaches back for "clean early value," what it gets
is already one block deep — it's the embedding seen through the lens of the first attention layer's
learned c_v weights, mixed with whatever the first block's residual update did. It's *cleaner* than block
8's own value, yes, but it isn't the rawest possible token signal. It's a derived quantity, and every
layer that wants it has to take the *same* v1, because there's only one first block. The thing I'm
injecting is a single shared vector computed once, and its content is whatever the first attention layer
happened to learn to project. There's a tension hiding here: the first block's c_v is being asked to serve
two masters — it's optimized to produce a good value for block 1's *own* attention, but it's also,
implicitly, the source of the early-value signal that all eleven later layers depend on. One projection,
two jobs. That's the same overloading I broke when I untied the embedding from the head.

So here's the move. What if every attention layer got a direct line not to a *derived* value but to a
*learned, token-identity-based* value signal — a value that depends only on the token id, looked up from a
table, never carried through the residual stream or filtered through the first block's projection at all?
The residual stream is the network's working memory: everything that flows through it is a transformation
of the input, and the deeper a block sits the more processed and the more lossy its view of the raw token
becomes. A learned table indexed by token id sidesteps that entirely. It says: for token "the", here is a
value vector layer 5 should mix in, here is a (possibly different) value vector layer 9 should mix in, and
these are parameters the optimizer sets directly, with no constraint that they be reconstructible from the
residual stream. It's the value-stream analogue of what the input embedding wte already does for the
*input* — wte is a learned, token-identity-based seed for the residual stream — except now I'm seeding the
*value* of every attention layer directly from token identity.

Why would that be better than reaching back to v1? Because v1's content is bottlenecked. The first block
computes one value per token and that single vector has to encode whatever all the downstream layers
collectively need from the early signal — but they don't all need the *same* thing. A shallow layer and a
deep layer want different aspects of the token's identity in their values. v1 can't give them different
things; it's one vector. A table can. If I make the table wide enough to hold a *separate* value slice per
layer, then layer i looks up its own dedicated per-token value, and the optimizer can shape each layer's
slice independently. The first block's c_v is freed to just be block 1's value projection again, no longer
secretly the universal early-value source. And the injection no longer has to survive the trip through the
residual stream, because it never enters the residual stream — it's looked up fresh from the token ids at
every layer.

Concretely: add a second embedding table, call it vte, the value-token-embedding, alongside wte. wte has
width n_embd; vte I make width n_embd*12 — one n_embd-wide slice for each of the twelve layers. I look it
up once from the token ids, exactly like wte, and chunk the result into twelve per-layer slices. Then layer
i receives its slice vi[i], and inside the attention forward I mix it into the freshly-projected value with
the *same* learnable-lambda mechanism I already built for value residual: `v = (1 - self.lamb)*v +
self.lamb * vi.view_as(v)`. The machinery is identical — the per-layer lambda, the convex mix, starting
behavior the model can dial — but the thing being mixed in has changed meaning. Before, vi was the first
block's computed v1, a derived residual-stream quantity shared by all layers. Now vi is this layer's
dedicated slice of a learned, token-indexed table, independent of the residual stream and independent of
the first block's projection. Same lambda, dedicated learned table instead of one shared computed value.

The cost is real but bounded. The vte table is vocab_size × (n_embd*12) — that's twelve times the width of
wte, a genuinely large parameter table, and it has to be optimized. It's a lookup table indexed by token
id, the same kind of object as wte, so it belongs with wte under Adam, not Muon — Muon orthogonalizes
matrix updates and that's the wrong tool for an embedding table. So I hand vte to the same Adam group as
wte: `optimizer1 = torch.optim.Adam([raw_model.transformer.wte.weight,
raw_model.transformer.vte.weight], lr=0.6, betas=(0.8, 0.95), fused=True)`. And there's a per-step compute
cost: every forward now does a second, twelve-times-wider embedding lookup and twelve view/mix operations.
That will push the step time up a little. The bet is that the step *count* falls by more than enough to
pay for it — that giving every layer its own direct, learned, per-token value is a better source of clean
value signal than reaching back to one shared v1, and the model converges in meaningfully fewer steps.

One thing I want to keep honest with myself: this isn't free structure the model can ignore at init the
way the embed shortcut was. The vte table starts from whatever its initialization is, and lambda controls
how much of it leaks in, so if the table were useless the lambdas could learn to suppress it — the safety
valve is there. But the table is big and it has to *learn* to be useful, which costs some of the early
steps. So I expect the curve to look slightly worse very early and then pull ahead as the per-token value
slices sharpen. If the net step count at the 3.28 bar comes down, the extra lookup time is paid for and
this is a win. Given how much the lambdas wanted v1 in the first place, I think the layers will want a
dedicated, richer version of that signal even more.

So: keep the value-residual lambda mechanism exactly, but replace the *content* it mixes in. Instead of
every layer reaching back to the first block's single computed value v1 — a derived, shared,
residual-stream quantity whose c_v is overloaded — give every layer its own dedicated slice of a learned,
token-identity-indexed value table vte (width n_embd*12, chunked twelve ways), looked up fresh from the
token ids and optimized by Adam alongside wte. Each attention layer gets a direct, per-token value
injection that never has to be carried through the residual stream or filtered through the first block.
The step time will tick up from the extra lookups; the bet is the step count falls by more.

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
# ... each block i receives vi[i] (encoder/decoder indexing matches the U-net layout):
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

The chain: the value residual gave every layer a line back to the first block's value v1, but v1 is a
derived, residual-stream quantity, shared across all layers, sourced from one overloaded c_v; replace it
with a dedicated learned table vte of width n_embd*12 looked up from the token ids and chunked into one
per-layer slice, so each attention layer mixes in its own direct, per-token value via the same learnable
lambda — `v = (1 - self.lamb)*v + self.lamb * vi.view_as(v)` — with vte optimized by Adam alongside wte.
The extra lookups raise the step time, but a direct learned value signal that never has to survive the
residual stream should cut the step count by more.
