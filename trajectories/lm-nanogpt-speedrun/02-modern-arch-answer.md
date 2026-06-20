**Problem (from step 1).** Muon made the optimizer condition-blind and nearly free per step (22.3 min,
6200 steps), so the architecture it optimizes is now the bottleneck on step count. The body is still a
faithful 2019 GPT-2 block — GELU MLP, fused QKV, head dim 64, LayerNorm-style normalization, a
depth-dependent attention-scale fudge — each a conservative default with a better-behaved 2024 replacement.

**Key idea (modernized architecture).** Apply the accumulated "modern transformer" deltas at once:
(1) GELU → **ReLU²** (`relu(x).square()`), a sharper, harder-sparse MLP nonlinearity with a stronger
gradient where it fires; (2) **QK-norm** — RMS-normalize per-head q and k after rotary so attention logits
qᵀk can't run away and the softmax stays well-scaled; (3) parameter-free **RMSNorm** everywhere
(`F.rms_norm(x,(x.size(-1),))`), dropping the unused learned affine; (4) **split QKV** into separate
`c_q/c_k/c_v` so Muon orthogonalizes three clean n×n matrices; (5) **head dim 64 → 128** (n_head 12 → 6)
for richer per-head subspaces and kernel-friendly sizes; (6) **zero-init the residual output projections**
(`c_proj.weight.zero_()` in attn and MLP) so the network starts as an exact identity stack and the
`1/√(2·n_layer)` attn-scale can be deleted; (7) **pad the vocab 50257 → 50304** for tensor-core-friendly
head/embedding matmuls.

**Why it works.** ReLU² and wider heads give a better-conditioned function to fit; QK-norm and zero-init
residual projections make the early dynamics gentle enough to push the learning rate; the padded vocab
shaves per-step matmul time at zero loss cost (padding rows are never targets). The same Muon+AdamW recipe
then reaches 3.28 in fewer steps. The schedule (iteration count, warmdown) is re-tuned alongside.

**Change / code.** Replaces the attention/MLP/block/config of the prior script.

```python
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head; self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)   # split QKV
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj.weight.data.zero_()                              # zero init
        self.rotary = Rotary(self.head_dim)
    def forward(self, x):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)
        cos, sin = self.rotary(q)
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))   # QK norm
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True)
        y = y.transpose(1, 2).contiguous().view_as(x)
        return self.c_proj(y)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)
        self.c_proj.weight.data.zero_()                              # zero init
    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()                                       # ReLU² > GELU
        return self.c_proj(x)

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = CausalSelfAttention(config); self.mlp = MLP(config)
    def forward(self, x):
        x = x + self.attn(F.rms_norm(x, (x.size(-1),)))              # no attn_scale
        x = x + self.mlp(F.rms_norm(x, (x.size(-1),)))
        return x

@dataclass
class GPTConfig:
    vocab_size : int = 50304   # padded to multiple of 128
    n_layer : int = 12
    n_head : int = 6           # head dim 128
    n_embd : int = 768
```
