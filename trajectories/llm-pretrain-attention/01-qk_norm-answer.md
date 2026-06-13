**Problem.** The fixed `1/sqrt(d_k)` logit scale is tuned for unit-variance `q,k` *at init*, but the
pre-LN block normalizes only the *input* to attention — `W_q`, `W_k` are free to grow, and the optimizer
has an incentive to grow them (scaling up logits is the cheapest way to sharpen the softmax). So per-head
`q,k` norms creep up over the run, the effective logit standard deviation inflates, and the softmax
drifts toward saturation late in training — a slow, silent loss of gradient exactly when refinement
matters. The scale is a moving target the static factor cannot track.

**Key idea.** Make the logit depend on the *directions* of `q,k`, not their magnitudes: normalize each
per-head `q` and `k` before the product so a drifting `||q||,||k||` can no longer inflate the score. The
clean derivation is L2-normalize + a learned per-head scale `g` replacing `1/sqrt(d_k)`. What this
scaffold actually exposes is **RMSNorm on `q` and `k`** (`F.rms_norm` along the head dim) with the fused
SDPA path left intact — `RMSNorm(x) = sqrt(d)·x/||x||`, so after SDPA's internal `1/sqrt(d)` the realized
logit is `sqrt(d_k)·cos(angle)`: a cosine similarity scaled by the *constant* `sqrt(d_k)=8`, no learned
`g`. Magnitude drift is stripped at the root; the deliberate-sharpening upside of a learned `g` is the
one thing given up.

**Why it works.** The logit can no longer blow up just because the weights grew, so the
saturation-creep failure mode is removed. It is a pure stability fix — no new information, no position
change, no head-structure change — so it isolates the score-stability effect with nothing confounded.

**Scaffold edit / hyperparameters.** Insert two `F.rms_norm` calls on the per-head `q,k` (after the
reshape to `(B, n_head, T, head_dim)`, before SDPA); never on `v`. No new parameters, no learnable
scale, no `CONFIG_OVERRIDES`. Position untouched: `use_pos_emb = True`, learned `wpe` stays on.

**What to watch.** A *modest* improvement — cleaner late-training gradients, slightly lower/steadier
`val_loss`, downstream tracking the LM gain — but the absolute-additive `wpe` handicap is fully in place,
so this should be the **weakest** rung, beaten by anything that fixes how *order* enters.

```python
# EDITABLE region of nanoGPT/custom_pretrain.py (lines 33–70) — step 1: QK-Norm (RMSNorm on q,k)
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
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
        self.use_pos_emb = True

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        # Apply QK normalization via RMSNorm (per head, along head_dim) — strips q/k magnitude drift
        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
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
