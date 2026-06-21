RoPE + QK-Norm confirmed the position thesis and, in the same breath, flagged the redundancy I said to watch. Validation loss landed at $2.2589$ against the QK-Norm floor of $2.2885$ — a drop of $0.0296$ in cross-entropy, far past anything a stability tweak alone produces — with WikiText-2 $43.65 \to 43.44$, LAMBADA $69.99 \to 67.20$ (the largest drop, as expected, since last-word prediction over a long passage is precisely where relative-offset encoding earns its keep), and downstream lifting across ARC-Easy $55.64 \to 57.83$, HellaSwag $33.41 \to 34.24$, PIQA $63.17 \to 64.74$. So the position fix is real and general. But the $2.2589$ stacked RoPE on top of the parameter-free RMSNorm QK-Norm, and that half, in this scaffold, forces every per-head $q$ and $k$ onto a fixed-norm sphere so the realized logit is $\sqrt{d_k}\,\cos(\angle)$ — a cosine similarity scaled by the *constant* $\sqrt{d_k} = 8$. That constant is a hard ceiling on how sharp attention can get: the pre-mask logit vector spans at most $\pm 8$ around the cosine range, so the softmax contrast between the best and worst position is bounded no matter how confident the model should be. When position was absolute and additive that ceiling was a fair trade — I was buying robustness to genuine late-training magnitude drift. But now position is relative, injected by rotation, and RoPE itself is norm-preserving (the rotation $R_m$ is orthogonal), so a chunk of the instability QK-Norm guarded against is already damped by the geometry of the position scheme. Meanwhile RMSNorm throws away a usable degree of freedom: plain RoPE keeps the logit as $\lVert q\rVert\,\lVert k\rVert\,\cos(\angle)$ modulated by the relative rotation, and the model can grow $\lVert q\rVert, \lVert k\rVert$ on heads that *should* attend sharply and keep them small on heads that should stay diffuse. The hypothesis is concrete: once RoPE has fixed position, the RMSNorm half no longer buys enough stability to justify the sharpness it sacrifices, and removing it should *recover* a little loss by handing the q/k magnitudes back to the optimizer as a learnable sharpness control.

I propose **plain RoPE**: the same relative-position injection, with QK-Norm stripped back out. Let me re-establish the construction so it stands on its own. Attention is order-blind — with $q, k, v$ linear in the token embeddings the computation is permutation-equivariant, so position must be injected by hand, and the only quantity deciding which token attends to which is the logit $q_m^\top k_n$. The default fed order additively and absolutely through `wpe`, and expanding $q_m^\top k_n$ with $q = W_q(x_m + p_m)$, $k = W_k(x_n + p_n)$ produced four terms, three carrying *absolute* $p_m / p_n$ — so the logit depended on the buffer slot, not the offset $m - n$ that language relations turn on. RoPE removes this by construction, and it is a *solve*, not a patch. Demand that the encoded inner product depend on position only through the difference, $\langle f_q(x_m, m), f_k(x_n, n)\rangle = g(x_m, x_n, m - n)$, with boundary $f(x, 0) = Wx$. In two dimensions, identify $\mathbb{R}^2$ with the complex plane and use $\langle a, b\rangle = \mathrm{Re}[a\,b^*]$; write $f$ in polar form and match magnitude and phase. The magnitude equation, pinned by the boundary at offset $0$, forces the magnitude position-independent — the stable, norm-preserving branch, because I do not want position to amplify one side and shrink the other — and the phase equation forces the phase arithmetic in position, the same extra angle $m\theta$ on both query and key. The solution is rotation:
$$ f_q(x_m, m) = (W_q x_m)\, e^{i m\theta}, \qquad f_k(x_n, n) = (W_k x_n)\, e^{i n\theta}, $$
so $\langle f_q, f_k\rangle = \mathrm{Re}\big[(W_q x_m)(W_k x_n)^* e^{i(m-n)\theta}\big]$ — absolute $m, n$ appear only through $e^{i(m-n)\theta}$.

Lift to the real head dimension by splitting it into $d/2$ independent 2-planes and rotating each at its own frequency. The inner product is a sum over planes, each relative-only by the 2D argument, and the sum of relative-only-per-plane is relative-only — linearity glues it. The block-diagonal $R_m$ has $i$-th $2\times 2$ block a rotation by $m\theta_i$; rotations compose, so $R_m^\top R_n = R_{n-m}$ and $q_m^\top k_n = x_m^\top W_q^\top R_{n-m} W_k x_n$ — the offset in a single rotation between the content projections, no learned table, no clip, no distance bias. The frequencies reuse the sinusoidal geometric spectrum $\theta_i = 10000^{-2(i-1)/d}$: fast planes resolve local offsets, slow planes carry coarse position, which gives the long-range decay envelope as a free prior — as $\lvert m - n\rvert$ grows the phases spread across frequencies, the partial sums lose coherence, and the positional contribution decays, so far-apart tokens interact less. And because $R$ is orthogonal it preserves norm, so it can never blow up or collapse the representation through 24 layers — which is exactly the property that makes the QK-Norm half partly redundant: RoPE already keeps q/k from drifting in the position-dependent direction.

The scaffold edit is *simpler* than the previous rung — I remove the two `F.rms_norm` calls and keep everything else. Position is no longer additive, so `self.use_pos_emb = False` and `GPT.forward` skips the `wpe` add (it gates on exactly that flag). I precompute $\text{inv\_freq} = 1/\big(10000^{\,\text{arange}(0,\text{head\_dim},2)/\text{head\_dim}}\big)$ as a buffer; per forward I form $\text{freqs} = \text{outer}(\text{arange}(T), \text{inv\_freq})$, take $\cos$ and $\sin$, and apply the *split-half* rotation — $x_1 = x[..., :d]$, $x_2 = x[..., d:]$ with $d = \text{head\_dim}/2$, $y_1 = x_1\cos - x_2\sin$, $y_2 = x_1\sin + x_2\cos$, concatenate $[y_1, y_2]$ — to $q$ and $k$ only, never $v$. The split-half layout pairs coordinate $i$ with $i + d$ as the two legs of one plane (the LLaMA/NeoX convention), equivalent up to a fixed permutation to the interleaved $(2i, 2i+1)$ pairing but the code must be consistent. The fused SDPA path, the causal mask, and the output projection are untouched, so the only difference from the combined rung is the QK-Norm removal: $q = \text{\_apply\_rope}(q, T)$ instead of $q = \text{\_apply\_rope}(\mathrm{RMSNorm}(q), T)$. That makes it the cleanest ablation on the ladder — one operation deleted, everything else held — so whatever the loss does between $2.2589$ and this run is attributable to the RMSNorm removal alone. RoPE here remains the *frozen* sinusoidal schedule at base $10000$, frequencies unlearned and base untuned, unchanged from before. I predict a small win on `val_loss` — a couple of thousandths below $2.2589$ — because handing the q/k magnitudes back to the optimizer recovers a little of what the $\sqrt{d_k}$ ceiling pinned, with LAMBADA the most sensitive perplexity. I also expect a possible *split* downstream: plain RoPE best on the language-modeling metrics while RoPE + QK-Norm clings to an edge on a multiple-choice task or two (ARC-Easy, PIQA) where extra logit-scale robustness helps calibration — in which case plain RoPE is the strongest *language model* on the primary objective, with the combined form not strictly dominated. A clearly *worse* plain-RoPE run would falsify the redundancy hypothesis and say RMSNorm was doing real stabilization work even under relative position.

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
