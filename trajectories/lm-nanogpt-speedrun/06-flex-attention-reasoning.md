Seven minutes and twelve seconds now, 3000 steps, val_loss 3.2753 — the U-net skips plus the doubled LR
got me here. The body is dense and clean, the optimizer is fast, and when I profile a single step the cost
has migrated almost entirely into one place: attention. Everything else — the MLPs, the embeddings, the
head — is cheap and well-scaled. Attention is what each step is now spent on, and so attention is what I
have to think hard about, because it's the thing standing between me and a shorter run.

The reason attention is expensive is that it's dense and causal over my context, and dense causal attention
is O(T²). Right now T is modest, about 1024 tokens, and that ceiling on T is not an accident — it's exactly
because the cost grows quadratically that I've kept the context short. But look at what context length buys
me. Every step processes some number of tokens, and the total work to reach the bar is roughly fixed by how
many tokens the model needs to see. If I could pack far more tokens into each forward — a much longer
context — I'd need proportionally fewer steps to cover the same token budget, and fewer steps with the same
per-token cost is a straight win on wallclock. The obvious lever, then, is to lengthen the context
dramatically. The only thing stopping me is the O(T²) wall: at 1024 tokens the quadratic is tolerable, but
push T to tens of thousands and a dense map is hopeless.

So the question isn't "should I lengthen the context" — it's "can I make attention cheap enough at long T to
afford it." And the structure of the problem says yes, because at long T I do not actually want every query
attending to every key. Two facts make most of the dense map wasteful. First, FineWeb is a packed stream:
many independent documents are concatenated together with an end-of-text delimiter between them, and a query
in document seven has no business attending to a key in document three — that's cross-document leakage that
can only add noise. Second, even within a document, the useful signal for next-token prediction is
overwhelmingly local; a query a long way back from a key rarely needs that key, especially with rotary
already encoding relative position. So the *true* attention pattern I want at long context is sparse: each
query attends only to keys (a) at or before it — causal, as always; (b) in the *same* document as it; and
(c) within some bounded window behind it. That is a tiny fraction of the full T² map, and if I can get a
kernel that only computes the kept entries, the cost of attention stops scaling with T² and starts scaling
with T times the window — linear in context length. That changes everything: a linear-in-T attention means
I can make T enormous and pay for it.

How do I express that without hand-writing a fused CUDA kernel? This is exactly what FlexAttention is for —
the `torch.nn.attention.flex_attention` interface that @KoszarskyB has been wiring in. I write a small
boolean predicate over `(q_idx, kv_idx)` that returns whether that query is allowed to see that key, hand it
to `create_block_mask`, and FlexAttention compiles a *block-sparse* fused kernel that skips whole blocks of
the map where the mask is entirely false. I don't pay for the masked-out region at all — it's not computed
and then zeroed, it's never touched. So the three conditions I want become three predicates AND-ed together.
Causal is the familiar `q_idx >= kv_idx`. The document constraint I get by computing a cumulative-sum
document id: every time I pass the EoS token id 50256 the doc id increments, so `docs = (idx ==
50256).cumsum(0)` tags each position with which document it belongs to, and `docs[q_idx] == docs[kv_idx]`
is true exactly when query and key live in the same document. And the window is `q_idx - kv_idx <
attn_blocksize` — the query only reaches `attn_blocksize` tokens back. The mask is the AND of the three, and
that single predicate is the whole specification; FlexAttention turns it into the kernel.

```python
docs = (idx == 50256).cumsum(0)
def document_causal_mask(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    document_mask = docs[q_idx] == docs[kv_idx]
    window_mask = q_idx - kv_idx < attn_blocksize
    return causal_mask & document_mask & window_mask
```

Now I can be aggressive with context. If attention is windowed and document-masked, its cost no longer
explodes with T, so instead of 1024 tokens I take the sequence length all the way to 64×1024 = 65536 tokens.
The data that used to be eight separate sequences of length ~1024 becomes, in effect, one very long packed
stream per device — and that forces a structural choice I have to be careful about: with FlexAttention here
the batch dimension B must be 1. There is one long stream, not a batch of short ones, so inside the attention
forward I assert `B == 1` and treat the whole thing as a single sequence of length T, letting the document
mask do the work of keeping the packed documents from talking to each other. The block mask itself has to be
rebuilt each forward, because the document boundaries depend on the actual tokens in this step's stream —
`create_block_mask(document_causal_mask, None, None, S, S, device="cuda", _compile=True)` where `S = len(idx)`
— and I let it compile, caching the kernel. The attention call becomes
`flex_attention(q.transpose, k.transpose, v.transpose, block_mask=block_mask)` in place of
`F.scaled_dot_product_attention`, everything else in the block — the QK-norm, the rotary, the value-residual
mix `v = (1-lamb)*v + lamb*vi` — left exactly as it was.

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
flex_attention = torch.compile(flex_attention, dynamic=False)
create_block_mask = torch.compile(create_block_mask, dynamic=False)

def forward(self, x, vi, block_mask):
    B, T = x.size(0), x.size(1)
    assert B == 1, "Must use batch size = 1 for FlexAttention"
    q = self.c_q(x).view(B, T, self.n_head, -1)
    k = self.c_k(x).view(B, T, self.n_head, -1)
    v = self.c_v(x).view(B, T, self.n_head, -1)
    v = (1 - self.lamb) * v + self.lamb * vi.view_as(v)
    q, k = norm(q), norm(k)
    q, k = self.rotary(q), self.rotary(k)
    y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask)
    y = y.transpose(1, 2).contiguous().view_as(x)
    return self.c_proj(y)

S = len(idx)
block_mask = create_block_mask(document_causal_mask, None, None, S, S, device="cuda", _compile=True)
# sequence_length : 64*1024 (was ~1024); batch_size : 8 sequences
```

What I expect: the per-step time barely moves, because each step still does roughly the same amount of
useful attention work per token (the window caps how much any query attends to), but each step now ingests
sixty-odd times the tokens, so the number of steps to cover my token budget collapses. That's a big
wallclock cut, and it comes from buying context cheaply rather than from any change to what the model learns.

There's one wrinkle I want to name honestly before I commit, because it's a real cost and not a free lunch.
Shortening the run this much makes it twitchier. The previous record had already doubled the LR, and a high
LR over a much shorter duration leaves less time for the loss to settle, so run-to-run variance goes up — I
measure something like 0.005 standard deviation across repeats. The *mean* val_loss sits safely under the
bar, around 3.279, but not literally every single run lands below 3.28; the occasional run pokes above it.
I'll accept that for now as the price of the speed, and treat it as something to manage rather than a reason
to back off — the expected loss is comfortably under the bar and the wallclock saving is large. If the
variance becomes the binding constraint later, the natural place to attack it is the window itself, which I'm
holding fixed at a single size for the whole run here; there may be slack in *how* the window is scheduled
rather than in whether it exists. But that's later. For now: replace dense causal attention with a windowed,
document-masked FlexAttention, lengthen the context to 64K, and let the step count fall.

The chain: each step is now dominated by dense O(T²) causal attention, which caps the context at ~1024
tokens; but the attention I actually want at long context is sparse — causal, same-document, within a bounded
window — and FlexAttention compiles exactly that sparse pattern from a three-predicate block mask, so the
cost goes linear in context; that lets me push the sequence length to 64K with B=1 packed streams, which
covers the token budget in far fewer steps and cuts the wallclock hard — at the known price of higher
run-to-run variance, whose mean still clears the bar.
