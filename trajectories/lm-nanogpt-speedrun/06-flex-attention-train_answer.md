The symmetric U-net skips plus the doubled learning rate they make safe cut the step count to 3000 and the wallclock to ~7.2 minutes, at a better val loss of 3.2753. The body is dense and clean and the optimizer is fast, so when I profile a single step the cost has migrated almost entirely into one place: attention. The MLPs, the embeddings, the head are all cheap and well-scaled. Attention is expensive because it is dense and causal over the context, an $O(T^2)$ cost, and that quadratic is exactly why I have kept $T$ short, about 1024 tokens. But context length is a lever I have not pulled: the total work to reach the bar is roughly fixed by how many tokens the model must see, so if I could pack far more tokens into each forward I would need proportionally fewer steps to cover the same budget — fewer steps at the same per-token cost is a straight win. The only thing stopping me is the $O(T^2)$ wall; the question is not *whether* to lengthen the context but whether I can make attention cheap enough at long $T$ to afford it.

The structure of the problem says yes, because at long $T$ I do not actually want every query attending to every key. Two facts make most of the dense map wasteful. First, FineWeb is a *packed* stream — many independent documents concatenated with an end-of-text delimiter — and a query in document seven has no business attending to a key in document three; that is cross-document leakage that can only add noise. Second, even within a document the useful signal for next-token prediction is overwhelmingly local, especially with rotary already encoding relative position; a query far back from a key rarely needs it. So the true attention pattern I want at long context is sparse: each query attends only to keys (a) at or before it — causal; (b) in the *same* document; and (c) within a bounded window behind it. That is a tiny fraction of the $T^2$ map, and a kernel that computes only the kept entries makes attention scale as $T$ times the window — linear in context length.

I propose to express that sparsity with **FlexAttention and a document-plus-sliding-window block mask** (the integration is by @KoszarskyB). FlexAttention takes a small boolean predicate over `(q_idx, kv_idx)` returning whether that query may see that key, and `create_block_mask` compiles a *block-sparse* fused kernel that skips whole blocks of the map where the mask is entirely false — the masked-out region is never touched, not computed-then-zeroed. So my three conditions become three predicates AND-ed together. Causal is the familiar `q_idx >= kv_idx`. The document constraint comes from a cumulative-sum document id: every time the stream passes the EoS token 50256 the id increments, so `docs = (idx == 50256).cumsum(0)` tags each position with its document, and `docs[q_idx] == docs[kv_idx]` is true exactly when query and key share a document. The window is `q_idx - kv_idx < attn_blocksize`. That single AND-ed predicate is the whole specification; FlexAttention turns it into the kernel.

Now I can be aggressive with context. With attention windowed and document-masked, its cost no longer explodes with $T$, so I take the sequence length all the way to $64\times1024 = 65536$ tokens. The data that used to be eight separate ~1024-length sequences becomes one very long packed stream per device, which forces a structural choice: with FlexAttention here the batch dimension $B$ must be $1$. There is one long stream, not a batch of short ones, so inside the attention forward I assert `B == 1` and treat the whole thing as a single sequence of length $T$, letting the document mask keep the packed documents from talking to each other. The block mask must be rebuilt each forward because document boundaries depend on the actual tokens this step — `create_block_mask(document_causal_mask, None, None, S, S, device="cuda", _compile=True)` with `S = len(idx)` — and I let it compile, caching the kernel. The attention call becomes `flex_attention(q, k, v, block_mask=block_mask)` in place of `F.scaled_dot_product_attention`, with everything else in the block — the QK-norm, the rotary, the value-residual mix $v = (1-\lambda)v + \lambda v_i$ — left exactly as it was. I expect the per-step time to barely move, since each step still does roughly the same useful attention work per token, but each step now ingests sixty-odd times the tokens, so the steps needed to cover the budget collapse — a large wallclock cut bought from cheap context rather than any change to learning. One honest cost: shortening the run this much, on top of the prior record's doubled LR, makes it twitchier — run-to-run variance climbs to about $0.005$ std, the *mean* val loss sits safely near $3.279$ but the occasional single run pokes above $3.28$. I accept that as the price of the speed; if the variance ever becomes the binding constraint, the natural place to attack it is *how* the window is scheduled rather than whether it exists, since I am holding it at a single size here.

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

# in GPT.forward, build the block mask each step (attn_blocksize fixed for this rung):
docs = (idx == 50256).cumsum(0)
def document_causal_mask(b, h, q_idx, kv_idx):
    causal_mask = q_idx >= kv_idx
    document_mask = docs[q_idx] == docs[kv_idx]
    window_mask = q_idx - kv_idx < attn_blocksize
    return causal_mask & document_mask & window_mask
S = len(idx)
block_mask = create_block_mask(document_causal_mask, None, None, S, S, device="cuda", _compile=True)

# hyperparameters that change:
# sequence_length : 64*1024   (was ~1024)
# batch_size : 8 sequences
```
