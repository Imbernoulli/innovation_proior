**Problem (from step 10).** The FP8 head landed 3.2770 in 1395 steps (≈3.142 min); since then the
architecture has been tuned hard (the immediately-preceding config runs ~1750 steps at ~99 ms/step), so the
next real gain has to come from a structural fix. With sliding-window attention, softmax forces every head
to spread a full unit of probability over its window even when nothing is relevant, so heads learn to dump
that leftover mass on the beginning-of-sequence token as an attention sink. But under a sliding window the
BoS token is often outside the window, and with rotary embeddings its effective position shifts with
distance — so the sink is unreliable. The model needs a *distance-independent* way to perform a context-based
no-op, without leaning on a sink token.

**Key idea (sparse per-head attention gate).** Add a learned, per-head gate on the attention output,
computed from a small *sparse* slice of the residual stream — just the first 12 of 768 dimensions of x. The
gate is `sigmoid(linear(x[..., :12], gate_weights))` per head, and it multiplies the head's output y. When
the gate is ~0 the head contributes nothing for that token; when ~1 it fires normally. Because it's computed
from the token's own representation, it's distance-independent — a clean context-based no-op that decouples
"where attention points" from "whether the head fires at all," so the model no longer needs the BoS sink.

**Why it works.** Softmax can't abstain, so a head with nothing useful to attend to must route mass
somewhere; the gate gives it an explicit downstream valve to scale its contribution to zero instead, and a
gate doesn't depend on any sink token's presence or relative position. Only 12 dimensions feed it because the
no-op decision is coarse and low-rank, so the gate matmul is a rounding error against attention cost (hence
"sparse"). Zero-initializing the gate bank starts every gate at `sigmoid(0)=0.5` — a neutral, symmetric start
that asserts no opinion and learns toward 0 or 1 from the data. The gate adds a slight per-step cost but
removes the unreliable-sink tax, saving on the order of 50 steps.

**Change / code.** A `[num_gated_layers, num_heads, 12]` gate bank, zero-initialized; inside the attention
forward, multiply y by the sigmoid of a tiny linear map of the first 12 residual dims, broadcast per head.

```python
# a per-head attention gate driven by only the first 12 residual dims (sparse):
# attn_gate_bank: one [num_heads, 12] weight per gated layer
self.attn_gate_bank = nn.Parameter(torch.zeros(num_gated_layers, num_heads, 12))

# inside the attention forward, after computing the attention output y of shape (B, T, num_heads, head_dim):
# sparse context-based gate: sigmoid of a tiny linear map from 12 residual dims, per head, no-op when ~0
y = y * torch.sigmoid(F.linear(x[..., :12], attn_gate_w)).view(B, T, self.num_heads, 1)
```
