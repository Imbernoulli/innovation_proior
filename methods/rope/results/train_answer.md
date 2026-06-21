Self-attention has no built-in sense of order. With $q$, $k$, $v$ computed linearly from the token embeddings $\{x_1, \dots, x_N\}$, the output at position $m$ is a softmax-weighted sum of values whose weights come from the logits $q_m^T k_n$, and the whole computation is permutation-equivariant: shuffle the tokens and the outputs shuffle identically, so "the dog bit the man" and "the man bit the dog" collapse to the same bag of representations. For language that is fatal, so position has to be put in by hand. The single place it can enter, the only quantity that decides which token attends to which, is the logit $q_m^T k_n$; everything downstream is a consequence. What I actually want from that logit is sharp: it should depend on the contents $x_m, x_n$ and on the relative offset $m-n$ alone — a verb three words after its subject is the same relation whether the sentence starts at token 5 or token 500 — it should carry a sensible decay-with-distance prior, it should impose no hard length cap, and it should enter as a per-token transformation of $q$ and $k$ so it survives the linear-attention factorization that never materializes the $N\times N$ logit matrix.

No existing scheme hits all of these at once. The default, sinusoidal absolute encoding, adds a fixed vector $p_i$ to each embedding before projecting, $f_t(x_i,i)=W_t(x_i+p_i)$. But expanding the logit with $q_m=W_q(x_m+p_m)$ and $k_n=W_k(x_n+p_n)$ gives $$q_m^T k_n = x_m^T W_q^T W_k x_n + x_m^T W_q^T W_k p_n + p_m^T W_q^T W_k x_n + p_m^T W_q^T W_k p_n,$$ where three of four terms carry absolute position $p_m$ or $p_n$, never the difference $m-n$; the relative structure has to be learned indirectly from absolute signals, and the learned-embedding variant additionally caps length at some $L$ and learns nothing past it. The relative-position line repairs this but never cleanly. Shaw et al. (2018) injected a learned relative embedding into the keys and values, $r=\text{clip}(m-n,r_{\min},r_{\max})$, but the clip discards every distinction beyond a window, it is a learned table with no closed form, it perturbs the values and not just the logit, and the relative term sits buried inside the expanded dot product rather than being a per-token transform. Transformer-XL (Dai et al., 2019) surgically edits that four-term expansion with sinusoidal relative encodings and two trainable global vectors; T5 (Raffel et al., 2020) collapses everything to $x_m^T W_q^T W_k x_n + b_{m,n}$ with a single learned scalar relative bias; DeBERTa (He et al., 2020) keeps the two content-times-position cross terms. They all work, but every one of them begins with "add a position vector, expand the product" and then hand-edits the resulting terms to look relative, every one parameterizes the relative signal with a learned table or bias, and every one leaves that signal inside the pairwise logit — which is exactly why none of them port to linear attention, where position must ride on the per-token features $\phi(q_m)$ and $\psi(k_n)$, not on an entry of the matrix the method refuses to build.

I propose Rotary Position Embedding (RoPE). Rather than patch someone else's expansion, I write down the relative property as an equation and solve for the injection function. Let $q_m=f_q(x_m,m)$ and $k_n=f_k(x_n,n)$ and demand $$\langle f_q(x_m,m), f_k(x_n,n)\rangle = g(x_m,x_n,m-n), \qquad f_q(x,0)=W_q x, \quad f_k(x,0)=W_k x,$$ so the inner product depends on position only through the difference, reducing to ordinary attention at offset zero. I solve this first in $d=2$, because a 2-vector is a complex number and the rotation hunch hiding in the sinusoid — shifting a $(\sin,\cos)$ pair at frequency $w$ from $i$ to $i+k$ is exactly a rotation by $wk$, independent of $i$ — makes rotations trivial. Identify $\mathbb{R}^2$ with $\mathbb{C}$ and use $\langle a,b\rangle=\mathrm{Re}[a b^*]$, and ask the complex product itself to be relative: $f_q(x_q,m) f_k(x_k,n)^* = g(x_q,x_k,m-n)$. Writing each factor in polar form $R e^{i\Theta}$, the product multiplies magnitudes and subtracts phases, so matching the two pieces separately gives $R_q R_k = R_g$ and $\Theta_q-\Theta_k=\Theta_g$. Setting $m=n$ (zero offset) and using the boundary $R_g(\cdot,\cdot,0)=\|q\|\|k\|$ forces $R_q(x_q,m)R_k(x_k,m)=\|q\|\|k\|$ for all $m$. Matching the magnitude still permits a degenerate reciprocal scale — one side amplified, the other shrunk — which in the general relative case becomes an exponential distance bias rather than a rotation; I reject that branch because I want a stable, norm-preserving per-token map whose only positional effect is relative phase, so I freeze $R_q(x_q,m)=\|q\|$ and $R_k(x_k,m)=\|k\|$. Position does not live in the magnitude. The phase equation at $m=n$ then reads $\Theta_q(x_q,m)-\theta_q=\Theta_k(x_k,m)-\theta_k$, whose two sides depend on disjoint variables, so each equals a function $\phi(m)$ of position alone: both phases take the form $\Theta_f(x,m)=\phi(m)+\theta_x$. Substituting back into the general phase equation with $m=n+1$ shows $\phi(n+1)-\phi(n)$ equals a constant, so $\phi$ has constant first difference — it is arithmetic, $\phi(m)=m\theta$ after folding the free offset into the boundary. The solution is therefore a rotation by an angle proportional to position: $$f_q(x_q,m)=q\,e^{i m\theta}, \qquad f_k(x_k,n)=k\,e^{i n\theta}, \qquad \langle f_q,f_k\rangle=\mathrm{Re}[q k^* e^{i(m-n)\theta}],$$ in which the absolute indices appear only through $e^{i(m-n)\theta}$. The relative property was not added; it fell out of the demand.

Lifting to real dimension $d$ (even) is the honest generalization of one plane at one frequency: chop the space into $d/2$ independent 2-planes and rotate each at its own frequency. This is legitimate because the inner product is a sum over planes and each plane is independently relative-only by the 2D argument, so linearity glues them into a relative-only whole. The injection is $f_{q,k}(x_m,m)=R^d_{\Theta,m}W_{q,k}x_m$ with $R^d_{\Theta,m}$ block-diagonal, its $i$-th $2\times2$ block a rotation by $m\theta_i$. Rotations compose by adding angles, $(R^d_{\Theta,m})^T R^d_{\Theta,n}=R^d_{\Theta,n-m}$, so the logit becomes $$q_m^T k_n = (R^d_{\Theta,m}W_q x_m)^T (R^d_{\Theta,n}W_k x_n) = x_m^T W_q^T R^d_{\Theta,n-m} W_k x_n,$$ the offset sitting in a single rotation matrix sandwiched between the content projections — no learned table, no clip, no bias bucket — and because $R$ is orthogonal it preserves norms, so propagating it through layers cannot blow up or collapse the representation, which is the stability I demanded when the magnitude dropped out in 2D. For the frequencies I reuse the sinusoidal geometric spectrum $\theta_i=10000^{-2(i-1)/d}$, $i=1,\dots,d/2$: fast-spinning planes resolve local offsets, slow-spinning planes stay nearly fixed across the sequence and carry coarse position. This is not an arbitrary borrow; it makes RoPE literally the relative-rotation version of sinusoidal encoding, and in practice letting the $\theta_i$ be learned moves them almost nothing from this initialization, so I freeze them and spend no parameters there.

The decay prior is not just asserted; it is provable. Grouping into $d/2$ complex pairs with $h_i=q_{[2i:2i+1]}k_{[2i:2i+1]}^*$ (pure content) and partial sums $S_j=\sum_{i<j}e^{i(m-n)\theta_i}$ (pure position, $S_0=0$, $h_{d/2}=0$), summation by parts — the discrete Abel transformation — rewrites $\sum_i h_i e^{i(m-n)\theta_i}=-\sum_i S_{i+1}(h_{i+1}-h_i)$, the boundary terms vanishing, hence $$\left|\sum_i h_i e^{i(m-n)\theta_i}\right| \le \Big(\max_i |h_{i+1}-h_i|\Big)\sum_i |S_{i+1}|.$$ This factors into a content piece that knows nothing of $m-n$ times a purely positional envelope $\tfrac{1}{d/2}\sum_i|S_i|$, and because the $\theta_i$ are a geometric sweep, as $|m-n|$ grows the phases spread across frequencies and the partial sums lose coherence, so the envelope decays — far-apart tokens contribute a less coherent positional signal, the inductive bias I wanted. Two practicalities close the design. First, I never build the block-diagonal matrix (mostly zeros, $O(d^2)$ for an $O(d)$ job): the rotation is elementwise, $R^d_{\Theta,m}x = x\odot\text{cos\_vec} + \text{rotate}(x)\odot\text{sin\_vec}$ with $\text{rotate}(x)=[-x_2,x_1,-x_4,x_3,\dots]$ the per-pair 90-degree swap, two multiplies and an add. Second, the linear-attention compatibility no additive scheme could give: because the rotation preserves norm it can sit after the non-negative feature maps, $\text{Attention}_m=\sum_n (R_m\phi(q_m))^T(R_n\psi(k_n))v_n / \sum_n \phi(q_m)^T\psi(k_n)$, so position rides on the per-token features and the $O(N)$ associativity trick survives, with the denominator deliberately left un-rotated so the normalizer stays non-negative and away from zero. An additive bias $b_{m,n}$ cannot sit on per-token features by construction, which is precisely why RoPE ports and the others do not.

```python
import torch
import torch.nn as nn

def inverse_frequencies(head_dim, base=10000, device=None):
    # one-based theta_i = base^{-2(i-1)/d}; arange(0, d, 2) is the code form.
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for pairwise rotations")
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))

def roformer_sinusoidal_pos(positions, head_dim, base=10000):
    inv_freq = inverse_frequencies(head_dim, base, positions.device)
    angles = positions[:, None].float() * inv_freq[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)

def apply_roformer_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
    sinusoidal_pos = sinusoidal_pos.to(device=query_layer.device, dtype=query_layer.dtype)
    sin, cos = sinusoidal_pos.chunk(2, dim=-1)
    sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
    cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)

    rotate_half_query = torch.stack(
        [-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1
    ).reshape_as(query_layer)
    query_layer = query_layer * cos_pos + rotate_half_query * sin_pos

    rotate_half_key = torch.stack(
        [-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1
    ).reshape_as(key_layer)
    key_layer = key_layer * cos_pos + rotate_half_key * sin_pos

    if value_layer is not None:
        rotate_half_value = torch.stack(
            [-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1
        ).reshape_as(value_layer)
        value_layer = value_layer * cos_pos + rotate_half_value * sin_pos
        return query_layer, key_layer, value_layer
    return query_layer, key_layer

def llama_rotary_tables(position_ids, head_dim, base=10000, dtype=None):
    inv_freq = inverse_frequencies(head_dim, base, position_ids.device)
    inv_freq = inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids = position_ids[:, None, :].float()
    freqs = (inv_freq @ position_ids).transpose(1, 2)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos(), emb.sin()
    if dtype is not None:
        cos, sin = cos.to(dtype=dtype), sin.to(dtype=dtype)
    return cos, sin

def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)

def apply_llama_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class PositionStrategy:
    def __init__(self, head_dim, base=10000, layout="llama"):
        if layout not in {"llama", "roformer"}:
            raise ValueError("layout must be 'llama' or 'roformer'")
        self.head_dim = head_dim
        self.base = base
        self.layout = layout

    def apply(self, q, k, positions):
        if self.layout == "roformer":
            sinusoidal_pos = roformer_sinusoidal_pos(positions, self.head_dim, self.base)
            return apply_roformer_rotary_position_embeddings(sinusoidal_pos, q, k)

        position_ids = positions[None, :].expand(q.shape[0], -1)
        cos, sin = llama_rotary_tables(position_ids, self.head_dim, self.base, dtype=q.dtype)
        return apply_llama_rotary_pos_emb(q, k, cos, sin)

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, position):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.position = position

    def forward(self, x, positions=None, mask=None):
        B, T, _ = x.shape
        if positions is None:
            positions = torch.arange(T, device=x.device)
        split = lambda t: t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q, k, v = split(self.Wq(x)), split(self.Wk(x)), split(self.Wv(x))
        q, k = self.position.apply(q, k, positions)
        logits = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            logits = logits.masked_fill(mask, float("-inf"))
        o = logits.softmax(dim=-1) @ v
        return self.Wo(o.transpose(1, 2).reshape(B, T, -1))
position = PositionStrategy(head_dim=64, layout="llama")
attention = SelfAttention(d_model=768, n_heads=12, position=position)
```
