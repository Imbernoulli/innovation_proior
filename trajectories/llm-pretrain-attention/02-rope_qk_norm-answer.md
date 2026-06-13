**Problem (from step 1).** QK-Norm fixed the logit *scale* but left position *absolute and additive*:
the learned `wpe` makes `q_m^T k_n` depend on absolute slots `m,n`, not the offset `m - n`, so every
relative dependency in the text is reconstructed indirectly. That representational handicap — not
stability — is what the 2.2885 floor is paying for.

**Key idea.** Inject position so the logit depends only on contents and `m - n`. Demand
`<f_q(x_m,m), f_k(x_n,n)> = g(x_m, x_n, m-n)` and solve: in 2D over the complex plane the stable branch
keeps magnitude position-free and makes phase arithmetic in position, i.e. **rotate q and k by an angle
proportional to position**. Lift across `d/2` planes at the geometric frequencies `theta_i =
10000^{-2(i-1)/d}`; rotations compose, so the offset appears as one rotation `R_{n-m}` between the
content projections — relative by construction, no learned table, norm-preserving, with a built-in
long-range decay. Then **keep the step-1 QK-Norm and stack it under RoPE**: RMSNorm strips q/k magnitude,
RoPE rotates direction, they are orthogonal — `q = _apply_rope(F.rms_norm(q), T)`.

**Why it works.** RoPE removes the absolute-additive handicap at the root (the offset is the only
position signal in the logit); QK-Norm continues to hold the score robust to q/k norm drift. The two
compose without conflict because rotation preserves the norm RMSNorm imposes.

**Scaffold edit / hyperparameters.** Set `use_pos_emb = False` so `GPT.forward` skips `wpe`. Precompute
`inv_freq = 1/(10000 ** (arange(0, head_dim, 2)/head_dim))`; per forward build `cos,sin` from
`outer(arange(T), inv_freq)`; apply the **split-half** rotation (`x1=x[...,:d], x2=x[...,d:]`,
`y1=x1·cos-x2·sin`, `y2=x1·sin+x2·cos`) after RMSNorm, on q and k only (never v). Frozen frequencies,
base 10000, no learned `g`, no `CONFIG_OVERRIDES`.

**What to watch.** Should clear the 2.2885 floor on `val_loss` by a representational margin, with LAMBADA
(long-range last-word) the most sensitive perplexity. Open question for the next rung: is the QK-Norm half
redundant — even slightly costly — once position is relative? If the gain is good-but-not-clearly-better,
strip back to plain RoPE.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py (lines 33–70) — step 2: RoPE + QK-Norm
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
        # RoPE frequencies
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def _apply_rope(self, x, seq_len):
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
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
        # QK-Norm then RoPE (norm before rotation; rotation preserves the imposed norm)
        q = self._apply_rope(F.rms_norm(q, (q.size(-1),)), T)
        k = self._apply_rope(F.rms_norm(k, (k.size(-1),)), T)
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
