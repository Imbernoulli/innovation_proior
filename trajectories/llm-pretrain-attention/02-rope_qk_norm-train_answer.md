QK-Norm came back where I argued it would: validation loss $2.2885$, WikiText-2 perplexity $43.65$, LAMBADA $69.99$, and downstream ARC-Easy $55.64$, HellaSwag $33.41$, PIQA $63.17$, WinoGrande $51.30$. A clean, stable run — the score-stability fix did its job and nothing diverged — but it is the floor, for the precise reason I predicted: RMSNorm stabilizes the logit *scale* and touches nothing about *position*. The model is still fed order through the learned absolute `wpe` table, so every dependency in the text — a verb agreeing with a subject three tokens back, a closing quote matching one forty tokens back — must be reconstructed out of *absolute* slot signals. Self-attention is order-blind by construction: with $q, k, v$ linear in the token embeddings the computation is permutation-equivariant, so position must be injected by hand, and the only quantity that decides which token attends to which is the logit $q_m^\top k_n$. The default injects position additively and absolutely, $x_i \leftarrow x_i + p_i$, so $q_m = W_q(x_m + p_m)$, $k_n = W_k(x_n + p_n)$; expand the logit and it is four terms, $x_m^\top W_q^\top W_k x_n$ (pure content, fine) plus three terms each carrying *absolute* $p_m$ or $p_n$. The logit depends on where $m$ and $n$ sit in the buffer, not on the offset $m - n$ that language relations actually turn on. That representational handicap — not stability — is what the $2.2885$ is paying for.

I propose **RoPE stacked under QK-Norm**: rotary position encoding to inject order through the *relative* offset, with the validated parameter-free QK-Norm kept underneath. RoPE is not a patch on the additive expansion — it is a *solve*. Demand that the encoded inner product depend on position only through the difference: write $q_m = f_q(x_m, m)$, $k_n = f_k(x_n, n)$ and require $\langle f_q(x_m,m), f_k(x_n,n)\rangle = g(x_m, x_n, m-n)$ for some $g$ that sees position only through $m - n$, with the boundary $f(x, 0) = Wx$ so it reduces to ordinary attention at position $0$. Solve in the simplest nontrivial dimension, $d = 2$, by identifying $\mathbb{R}^2$ with the complex plane and using $\langle a, b\rangle = \mathrm{Re}[a\,b^*]$. Write $f$ in polar form, magnitude times phase, and match the two sides: the magnitude equation, evaluated at offset $0$ against the boundary, forces the magnitude to be position-independent (the stable, norm-preserving branch — I do not want position to amplify one side and shrink the other), and the phase equation forces the phase to be *arithmetic* in position, the same extra angle $m\theta$ added to query and key on top of each vector's own angle. The solution is a rotation:
$$ f_q(x_m, m) = (W_q x_m)\, e^{i m\theta}, \qquad f_k(x_n, n) = (W_k x_n)\, e^{i n\theta}, $$
so $\langle f_q, f_k\rangle = \mathrm{Re}\big[(W_q x_m)(W_k x_n)^* e^{i(m-n)\theta}\big]$ — the absolute positions appear *only* through $e^{i(m-n)\theta}$. I did not bolt anything on; the relative property fell out of the demand.

Lift to the real head dimension by chopping it into $d/2$ independent 2-planes and rotating each at its own frequency. The inner product is a sum over planes, each plane is relative-only by the 2D argument, and a sum of relative-only-per-plane is relative-only — linearity does the gluing. Stack the rotations into a block-diagonal $R_m$ whose $i$-th $2\times 2$ block is a rotation by $m\theta_i$; rotations compose by adding angles, so $R_m^\top R_n = R_{n-m}$ and $q_m^\top k_n = x_m^\top W_q^\top R_{n-m} W_k x_n$ — the offset sits in a single rotation between the content projections, no learned table, no clip, no distance bias. The frequencies reuse the sinusoidal geometric spectrum $\theta_i = 10000^{-2(i-1)/d}$: fast planes that spin quickly and resolve local offsets, slow planes that barely move over the whole sequence and carry coarse position. That choice is not arbitrary — it makes the scheme the relative-rotation version of the sinusoidal encoding, and it gives a long-range decay as a free prior: as $\lvert m - n\rvert$ grows the phases spread across frequencies, the partial sums lose coherence, and the positional contribution decays, so far-apart tokens interact less, all else equal. And $R$ is orthogonal, so applying it can never blow up or collapse the representation as it propagates through 24 layers.

The move that makes this step 2 and not a fresh start is that I keep the QK-Norm I just validated and stack it under RoPE. The $2.2885$ run proved the score-stability fix is real and free, and it is *orthogonal* to position: RMSNorm strips q/k magnitude, RoPE rotates q/k direction, and they compose. The order is forced. RMSNorm rescales each per-head vector to a fixed norm $\sqrt{d}\,x/\lVert x\rVert$, RoPE rotates it, and rotation is norm-preserving, so the norm RMSNorm imposes survives the rotation — the clean reading is "normalize the direction, then place it in position-space," i.e. $q = \text{\_apply\_rope}(\mathrm{RMSNorm}(q), T)$, same for $k$.

In the edit surface, two things are forced. First, position is no longer additive, so I turn *off* the learned `wpe`: setting `self.use_pos_emb = False` makes `GPT.forward` (which gates the `wpe` add on exactly that flag) skip the absolute table entirely — this is the one mechanism a rung has to replace position without editing anything outside the attention class, and it is mandatory, because leaving `wpe` on while rotating q/k would double up two position schemes. Second, I realize the rotation elementwise rather than building the sparse block-diagonal matrix. I precompute $\text{inv\_freq} = 1/\big(10000^{\,\text{arange}(0,\text{head\_dim},2)/\text{head\_dim}}\big)$ as a buffer; per forward I form $\text{freqs} = \text{outer}(\text{arange}(T), \text{inv\_freq})$ and take $\cos$ and $\sin$. The layout is *split-half* (the LLaMA/NeoX convention): split each per-head vector into $x_1 = x[..., :d]$ and $x_2 = x[..., d:]$ with $d = \text{head\_dim}/2$, rotate as $y_1 = x_1\cos - x_2\sin$, $y_2 = x_1\sin + x_2\cos$, then concatenate $[y_1, y_2]$. This pairs coordinate $i$ with $i + d$ as the two legs of one 2-plane, equivalent up to a fixed permutation to the interleaved $(2i, 2i+1)$ pairing but the code must be consistent. I apply it to $q$ and $k$ only, never $v$: position belongs in the logit, not the values. The omissions are the same ones I already accepted — RoPE here is the *frozen* sinusoidal schedule at base $10000$ (the frequencies barely move from this initialization, so there is no reason to spend parameters on them), and the QK-Norm half is still the parameter-free RMSNorm form. The new content of this rung is purely the relative-position fix layered on top, so I expect it to clear $2.2885$ by a representational margin, with LAMBADA the most sensitive perplexity since its long-range last-word prediction is exactly where relative-offset encoding earns its keep. The question this leaves open for the next rung: once position is relative, is the RMSNorm half still helping, or is it quietly costing me the deliberate-sharpening the $\sqrt{d_k}$ ceiling cannot provide?

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
