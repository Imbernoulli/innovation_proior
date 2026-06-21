The baseline here is textbook GPT-2 attention: project to $q, k, v$, form the logit $q_m^\top k_n / \sqrt{d_k}$, causal-mask, softmax, average the values, with order supplied by a learned absolute position table `wpe` added before the stack. The whole point of the $1/\sqrt{d_k}$ factor is a variance argument — for $q, k$ with unit-variance entries the dot product $\sum_{i=1}^{d_k} q_i k_i$ has variance about $d_k$ and standard deviation about $\sqrt{d_k}$, which at $d_k = 64$ is a spread of $\pm 8$, already near the softmax's saturated regime, so dividing by $\sqrt{d_k}$ rescales the logit back to unit standard deviation where the softmax is responsive. That argument is correct *at initialization*, and only there. The block is pre-norm, $x \leftarrow x + \mathrm{Attn}(\mathrm{LN}(x))$, so the LayerNorm normalizes the *input* to attention but says nothing about what $W_q, W_k$ do to it. Those matrices are free and weight-decayed but not norm-bounded, and the optimizer has a standing incentive to grow them — scaling up the logits is the cheapest way to sharpen the softmax, which lowers loss on many tokens — so the per-head $q$ and $k$ norms creep upward over the run. Once they have, say, doubled, the effective logit standard deviation has quadrupled, the fixed $1/\sqrt{d_k}$ can no longer track it, and the softmax drifts back toward saturation *late* in training, exactly when I want clean gradients to keep refining. The scale is not a one-time constant; it is a moving target the static factor cannot follow.

I propose **QK-Norm** — normalize $q$ and $k$ before the product so the logit depends on their *directions*, not their magnitudes. The clean way to make a dot product magnitude-invariant is to divide each vector by its length: with $\hat q = q / \lVert q \rVert$ and $\hat k = k / \lVert k \rVert$, the logit $\hat q^\top \hat k = \cos(\angle)$ lives in $[-1, 1]$ no matter how $W_q, W_k$ evolve, so the saturation-creep failure is removed at the root. But pure cosine over-corrects: a logit vector confined to $[-1,1]$ spans at most a range of $2$, and $\exp(1)/\exp(-1) \approx 7.4$ is then the most contrast any two positions can have, so the softmax can never concentrate. The fix is to reintroduce a *single* scale that is decoupled from the q/k drift — the clean derivation is L2-normalize then multiply by a learned per-head $g$ replacing $1/\sqrt{d_k}$, so the optimizer sets the sharpness deliberately, through one number per head, instead of letting it leak in through full-matrix weight growth.

What the scaffold actually exposes lands me at the parameter-free realization of the same idea, and it is worth being exact about why it works out. The loop runs the fused `scaled_dot_product_attention` (SDPA), which applies its own internal $1/\sqrt{d_k}$ and its own softmax — I cannot insert a custom $g$ inside the fused kernel without giving up the fused path. The primitive the loop does offer is RMSNorm, and RMSNorm of a vector $x \in \mathbb{R}^d$ is $x / \mathrm{rms}(x)$ with $\mathrm{rms}(x) = \lVert x \rVert / \sqrt{d}$, i.e. $\mathrm{RMSNorm}(x) = \sqrt{d}\, x / \lVert x \rVert$ — exactly L2 normalization up to the fixed factor $\sqrt{d}$. Apply it to both $q$ and $k$ and the product becomes $(\sqrt{d}\,\hat q)^\top(\sqrt{d}\,\hat k) = d\,\cos(\angle)$, and then SDPA divides by $\sqrt{d}$, so the realized logit is
$$ s_{m,n} = \sqrt{d_k}\,\cos(\angle(q_m, k_n)), $$
a cosine similarity scaled by the *constant* $\sqrt{d_k} = 8$ — the same nominal range the original $1/\sqrt{d_k}$ was designed to produce at init, with $\sqrt{d_k}$ playing the role of $g$, fixed rather than learned. So the version that fits is: RMSNorm $q$ and $k$ along the head dimension, leave SDPA, the causal mask, and `wpe` exactly as is. The honest cost is that the learned $g$ is gone — the sharpness ceiling is pinned at $\sqrt{d_k}$ rather than tuned — but the property I came for survives: the logit is now invariant to q/k norm drift, so the saturation-creep failure is fixed and only the deliberate-sharpening upside is left on the table.

A few details make the edit faithful. The RMSNorm is applied **per head**, along the `head_dim` axis, *after* the reshape to $(B, n_\text{head}, T, \text{head\_dim})$ — normalizing across the full `n_embd` before splitting would mix heads and destroy the per-head direction I want to preserve. It is applied to $q$ and $k$ only, never to $v$: $v$ carries the content that gets averaged into the output, and the saturation problem lives entirely in the logit, which $v$ is not part of. And `use_pos_emb` stays `True` — this intervention says nothing about position, so the learned `wpe` table is untouched. This is deliberately the floor of the ladder: it is the only candidate change that touches neither the position scheme nor the head structure, so whatever it buys is purely the score-stability effect with nothing else confounded. Because it removes a failure mode without adding any new information — position is still absolute-additive through `wpe` — I expect a modest improvement and expect this to be the *weakest* rung, beaten by anything that fixes how order actually enters the logit.

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
