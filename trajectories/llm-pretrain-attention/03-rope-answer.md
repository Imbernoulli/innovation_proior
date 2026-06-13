**Problem (from step 2).** RoPE + QK-Norm cleared the floor (2.2885 → 2.2589) and confirmed relative
position is the real lever. But the QK-Norm half is the parameter-free RMSNorm form, which pins every head
to a fixed `sqrt(d_k)·cos(angle)` logit — a hard sharpness ceiling — and removes the model's ability to
learn per-head confidence through the q/k norms. Once RoPE (itself norm-preserving) has fixed position,
that guard may be redundant or even slightly costly.

**Key idea.** Strip QK-Norm back out and run **plain RoPE**. Position is still injected by rotating q and
k by an angle proportional to index (relative-by-construction logit, frozen geometric frequencies, base
10000, built-in long-range decay, orthogonal so norm-preserving), but the q/k *magnitudes* are handed
back to the optimizer as a learnable sharpness control instead of being pinned to the cosine ceiling.

**Why it works.** RoPE removes the absolute-additive handicap and, being orthogonal, already damps the
position-direction drift QK-Norm was guarding against — so removing RMSNorm recovers per-head sharpness
freedom (`||q||·||k||·cos(angle)`) at little stability cost. This is the cleanest ablation on the ladder:
one operation deleted, everything else held.

**Scaffold edit / hyperparameters.** Identical to step 2 minus the two `F.rms_norm` calls:
`use_pos_emb = False`; `inv_freq = 1/(10000 ** (arange(0, head_dim, 2)/head_dim))`; per forward build
`cos,sin` from `outer(arange(T), inv_freq)`; split-half rotation on q and k only (never v),
`q = self._apply_rope(q, T)`. Frozen frequencies, no learned `g`, no `CONFIG_OVERRIDES`.

**What to watch.** Predict a small win on `val_loss` (a couple of thousandths below 2.2589) and on
perplexity (LAMBADA most sensitive). Possible split: plain RoPE best on LM metrics while RoPE + QK-Norm
clings to an edge on a multiple-choice downstream task (ARC-Easy / PIQA) — in which case plain RoPE is the
strongest *language model* on the primary objective, with the combined form not strictly dominated. A
clearly *worse* plain-RoPE run would falsify the redundancy hypothesis.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py (lines 33–70) — step 3: RoPE (plain)
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        self.use_pos_emb = False  # RoPE replaces learned position embeddings
        # Precompute RoPE frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def _apply_rope(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (T, head_dim/2)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, T, head_dim/2)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.cat([y1, y2], dim=-1).type_as(x)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Apply rotary position embeddings to Q and K
        q = self._apply_rope(q, T)
        k = self._apply_rope(k, T)
        if self.flash:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
```
