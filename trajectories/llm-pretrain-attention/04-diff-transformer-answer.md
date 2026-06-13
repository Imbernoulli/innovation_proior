**Problem (from step 3).** Every rung so far ends in a single softmax — a strictly-positive distribution
that cannot drive an irrelevant token's weight to zero or take mass back off it. Over a 1024-token
context the aggregate of those small irrelevant weights is most of the mass, so each output averages in a
low-level floor of junk. QK-Norm (scale) and RoPE (position) cannot reach this: the noise floor is a
property of softmax positivity, not of where the attention peak sits. Plain RoPE's 2.2570 still pays for
it.

**Key idea.** Form attention as the **difference of two softmax maps**, like a differential amplifier
cancelling common-mode noise: `(softmax(Q1 K1^T/sqrt(d)) - lambda*softmax(Q2 K2^T/sqrt(d))) V` with
`Q=[Q1;Q2]`, `K=[K1;K2]`. The two halves share the correlated noise floor over irrelevant tokens
(common-mode), so the subtraction cancels it; the signal survives. The result is a *signed* map — it can
zero out an irrelevant value's contribution, which a single positive softmax structurally cannot. Built on
the strongest baseline's position scheme (RoPE on the doubled q/k sub-heads).

**Why it works.** On irrelevant tokens the halves see the same content, so their floors correlate and
`A1 - lambda*A2 -> 0`; on relevant tokens the model makes them disagree, so the difference is large.
`lambda = 0` is not the optimum because cancelling the floor lowers loss. Parameter/FLOP-matched, so the
gain is purely from the changed attention *shape*.

**Scaffold edit / hyperparameters.** Halve the head dim (`head_dim = n_embd//n_head//2 = 32`): `2*n_head`
query/key sub-heads of dim 32, `n_head` value heads of dim 64 — total widths unchanged, fused `c_attn`
intact. Learnable `lambda = exp(λq1·λk1) - exp(λq2·λk2) + lambda_init`, four `head_dim` vectors init
`N(0,0.1)`. Per-head `RMSNorm(2*head_dim)` on the output, then `*(1 - lambda_init)`. RoPE on the q/k
sub-heads, `use_pos_emb = False`. Two harness-forced compromises: (1) `config` carries no layer index, so
the depth schedule `0.8-0.6*exp(-0.3*(l-1))` is unavailable — fix `lambda_init = 0.8` (its deep asymptote);
(2) the fused SDPA returns only the averaged output, not the two maps to subtract, so use the manual
masked-softmax path. No `CONFIG_OVERRIDES`.

**Bar to clear (no feedback — finale).** Beat plain RoPE: `val_loss` 2.2570, WikiText-2 43.17, LAMBADA
65.81, ARC-Easy 57.32, HellaSwag 34.48, PIQA 64.42, WinoGrande 51.70. Expect `val_loss` and both
perplexities below RoPE, LAMBADA (long-passage last-word retrieval-under-noise) the most. To validate:
the fixed `lambda_init` may over-cancel early layers; watch the first few hundred iterations for the
signed map's higher init variance (the per-head norm + `(1-lambda_init)` should hold it). If `val_loss`
does not drop below 2.2570, the attention-noise floor is not yet binding at 355M / 7.1B tokens / 1024
context — the differential advantage grows with scale and context length.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py (lines 33–70) — finale: Differential Transformer + RoPE
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
        self.dropout = config.dropout
        # Differential attention: halve the per-head dim so the doubled q/k stays budget-matched.
        # 2*n_head query/key sub-heads of dim head_dim; n_head value heads of dim 2*head_dim.
        self.head_dim = config.n_embd // config.n_head // 2
        self.scaling = self.head_dim ** -0.5
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
        self.use_pos_emb = False  # RoPE replaces learned position embeddings
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)
        # config carries no layer index -> use the depth schedule's deep asymptote as a fixed init.
        self.lambda_init = 0.8
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim).normal_(mean=0, std=0.1))
        # Per-head normalization over the 2*head_dim attention output (the GroupNorm/sub-LN).
        self.subln = nn.RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=True)

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
        # 2*n_head query/key sub-heads of dim head_dim; n_head value heads of dim 2*head_dim.
        q = q.view(B, T, 2 * self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, 2 * self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, 2 * self.head_dim).transpose(1, 2)
        # Relative position on the query/key sub-heads.
        q = self._apply_rope(q, T)
        k = self._apply_rope(k, T)
        # Two softmax maps (manual path: the fused kernel hides the maps the subtraction needs).
        att = (q @ k.transpose(-2, -1)) * self.scaling
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        att = att.view(B, self.n_head, 2, T, T)
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        att = att[:, :, 0] - lambda_full * att[:, :, 1]            # (B, n_head, T, T), signed
        y = att @ v                                                # (B, n_head, T, 2*head_dim)
        y = self.subln(y) * (1.0 - self.lambda_init)               # per-head norm + fixed gain comp.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
```
