**Problem (from step 5).** With the U-net skips and doubled LR at the bar (≈7.2 min, 3000 steps, val_loss
3.2753), per-step cost has migrated almost entirely into attention, which is dense and causal — O(T²) over a
modest ~1024-token context. That quadratic is exactly what caps the context length, yet a longer context
(more tokens per step → fewer steps for the same token budget) is the obvious next lever. The catch:
lengthening T is only affordable if attention is made cheap at long context, *and* it must respect document
boundaries, since FineWeb packs many independent documents into one stream.

**Key idea (FlexAttention with a document + sliding-window block mask).** Replace dense causal
`F.scaled_dot_product_attention` with **FlexAttention** (`torch.nn.attention.flex_attention`, contributed by
@KoszarskyB). Specify the kept attention pattern as a boolean predicate over `(q_idx, kv_idx)` that is the
AND of three masks: **causal** (`q_idx >= kv_idx`); a **document mask** (`docs[q_idx] == docs[kv_idx]`, where
`docs = (idx == 50256).cumsum(0)` tags each position by document using the EoS id 50256), so a query attends
only within its own document; and a **sliding-window mask** (`q_idx - kv_idx < attn_blocksize`), so a query
reaches only `attn_blocksize` tokens back. `create_block_mask` compiles this into a block-sparse fused kernel
that never touches the masked-out blocks, so attention cost goes linear in T instead of quadratic.

**Why it works.** A windowed, document-masked attention computes only a tiny fraction of the T² map, so the
context can be lengthened enormously: `sequence_length` goes from ~1024 to 64×1024 = 65536, the eight short
sequences becoming one long packed stream per device (B must be 1 here). The same token budget is now covered
in far fewer steps, with per-step time roughly unchanged — a large wallclock cut bought from cheap context
rather than any change to learning. Known cost: shortening the run this much (on top of the prior record's
doubled LR) raises run-to-run variance (~0.005 std); the mean val_loss clears the bar (~3.279) but not every
single run lands below 3.28.

**Change / code.** Swap SDPA for `flex_attention`; build the AND-of-three block mask each forward via
`create_block_mask` (rebuilt per step because document boundaries depend on the tokens); set B==1,
`sequence_length = 64*1024`.

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
