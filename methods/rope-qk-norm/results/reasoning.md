Let me start from where this attention block actually hurts when I pretrain a GPT on FineWeb, because there are two separate pains and I keep conflating them. The block is `softmax(q_m^T k_n / sqrt(d_k))` over a causal window, and position comes in by adding a learned vector `wpe[i]` to the token embedding before I ever project to `q, k, v`. Pain one: that position scheme is wrong in shape. Pain two: the score itself drifts. I want to fix both without touching anything outside this module — the optimizer, the schedule, the data, the loss are all frozen — so whatever I do has to live inside how `q` and `k` are formed and scored. Let me take the position pain first, because it is the more structural of the two.

Self-attention with no position signal is permutation-equivariant: the output at position `m` is a softmax-weighted sum of values, the weights come from `q_m^T k_n`, and if `q, k, v` are linear in the token embeddings then permuting the tokens just permutes the outputs. "The dog bit the man" and "the man bit the dog" give the same bag of representations. So position has to be injected, and the only quantity that decides which token attends to which is the logit `q_m^T k_n` — everything downstream is a consequence of that. So whatever I do, position has to end up changing `q_m^T k_n`. That is the target. Now what does the learned absolute scheme actually do to that target? With `q_m = W_q(x_m + p_m)`, `k_n = W_k(x_n + p_n)`,

  q_m^T k_n = x_m^T W_q^T W_k x_n + x_m^T W_q^T W_k p_n + p_m^T W_q^T W_k x_n + p_m^T W_q^T W_k p_n.

The first term is pure content. The other three carry `p_m` here, `p_n` there, both together — absolute position, *where each token sits in the buffer*. But what language cares about is the gap `m − n`: a verb three words after its subject is the same relation whether the clause starts at token 5 or token 500. The model can in principle disentangle relative offset from these absolute signals, but I am making it work for something I could hand it directly. And the learned table caps the context at the trained length `L` and learns nothing past it. So the absolute-additive route pushes the relative structure into a place where it has to be learned indirectly, and it doesn't extrapolate. I want the logit to depend on `x_m`, `x_n`, and `m − n` only.

There is a clue I keep coming back to from the sinusoidal encoding. Take a single `(sin, cos)` pair at frequency `w`. Shifting `i -> i + k` sends `(sin(wi), cos(wi))` to `(sin(w(i+k)), cos(w(i+k)))`, and that is exactly a rotation of the pair by a fixed angle `wk`, independent of `i`. So sinusoids already secretly encode a relative shift as a rotation — it is right there — but the additive scheme never uses rotation as the mechanism; it adds the vector and hopes. Rotation, relative shift. File that.

The relative-position line of work tries to fix the shape, but watch *how* it does it. Shaw and colleagues (2018) leave `q = W_q x_m` and inject a learned relative embedding into the key, `f_k(x_n, n) = W_k(x_n + p~_r)` with `r = clip(m − n, r_min, r_max)`. Real progress — the logit finally sees `m − n` — but it clips away every distinction beyond the window, it is a learned table (extra parameters, no closed form), and it pokes position into the values too. Transformer-XL (Dai et al. 2019), T5 (Raffel et al. 2020), DeBERTa (He et al. 2020) all sharpen this but keep the same starting move: take that four-term additive expansion and surgically edit the terms until they look relative — sinusoidal relative embeddings, trainable global vectors, a single learned scalar bias `b_{m,n}` bucketed by distance. Every one of them parameterizes the relative signal with a learned table or bias, and every one leaves that signal sitting *inside* the `N×N` pairwise logit as an additive term. None of them is a clean per-token transform of `q` and `k`.

So instead of patching someone else's expansion, let me write down what I want as a constraint and solve for it. Let `q_m = f_q(x_m, m)` and `k_n = f_k(x_n, n)` be whatever injects position. The demand is

  < f_q(x_m, m), f_k(x_n, n) > = g(x_m, x_n, m − n)

for some `g` that depends on positions only through the difference, with the boundary condition `f_q(x, 0) = W_q x`, `f_k(x, 0) = W_k x` so it reduces to ordinary attention when there is no position. That is the whole specification. Solve for `f_q, f_k`.

Start in `d = 2`, because a 2-vector is a complex number and complex numbers make rotations trivial, and the rotation hunch is still nagging. Identify `R^2` with `C`, and use that for complex `a, b` the real inner product of the corresponding 2-vectors is `Re[a b*]`. So `q_m^T k_n = Re[f_q f_k*]`, and the cleanest way to force the relative structure is to ask the *complex* product itself to depend on `m − n`:

  f_q(x_q, m) f_k(x_k, n)* = g(x_q, x_k, m − n).

Write everything in polar form, magnitude times phase: `f_q = R_q e^{i Theta_q}`, `f_k = R_k e^{i Theta_k}`, `g = R_g e^{i Theta_g}`. The product `f_q f_k*` multiplies magnitudes and subtracts phases, so matching magnitude and phase separately,

  R_q(x_q, m) R_k(x_k, n) = R_g(x_q, x_k, m − n),
  Theta_q(x_q, m) − Theta_k(x_k, n) = Theta_g(x_q, x_k, m − n),

with the boundary `R_q(x_q, 0) = ||q||`, `Theta_q(x_q, 0) = theta_q` (the angle of the unencoded query), and the same for `k`. Set `m = n`, so the offset is zero. The magnitude equation becomes `R_q(x_q, m) R_k(x_k, m) = R_g(x_q, x_k, 0)`, and the boundary fixes `R_g(·, 0) = ||q|| ||k||`, so `R_q(x_q, m) R_k(x_k, m) = ||q|| ||k||` for every `m`. Matching the magnitude still leaves a degenerate reciprocal-scale branch where one side is amplified and the other shrunk by a position-dependent factor — but I do not want position to act as a distance-dependent gain on the representation, I want a stable per-token map whose only positional effect is a relative phase. The clean branch freezes each magnitude at its position-0 value: `R_q(x_q, m) = ||q||`, `R_k(x_k, m) = ||k||`. Position does not live in the magnitude. It is a pure phase thing. (Hold that thought — *position preserves the norm* is going to matter enormously when I get to the second pain.)

Now the phase at `m = n`: `Theta_q(x_q, m) − Theta_k(x_k, m) = theta_q − theta_k`, i.e. `Theta_q(x_q, m) − theta_q = Theta_k(x_k, m) − theta_k`. The left side depends only on `(x_q, m)`, the right only on `(x_k, m)`, and they are equal, so each is a function of `m` alone — call it `phi(m)`. Both phases have the same shape `Theta_f(x, m) = phi(m) + theta_x`. Plug back into the general phase equation with `n = m + 1`: `phi(m+1) + theta_q − (phi(m) + theta_k) = Theta_g(x_q, x_k, 1)`, so `phi(m+1) − phi(m) = Theta_g(·, 1) + theta_k − theta_q`, a constant in `m`. A constant first difference means `phi` is arithmetic, `phi(m) = m·theta + gamma`, and `gamma` is a free global offset I fold into the boundary and set to zero. So

  f_q(x_q, m) = ||q|| e^{i(theta_q + m·theta)} = q · e^{i m theta},
  f_k(x_k, n) = ||k|| e^{i(theta_k + n·theta)} = k · e^{i n theta}.

Position is a rotation. Multiply the complex query by `e^{i m theta}`, the key by `e^{i n theta}`. Check the logit: `Re[q e^{i m theta} (k e^{i n theta})*] = Re[q k* e^{i(m−n)theta}]`. The absolute positions appear only through `e^{i(m−n)theta}` — the logit depends on `m − n` and nothing else absolute. Exactly the demand, and I did not add anything, I solved for it. As a real matrix this is `f(x_m, m) = [[cos m·theta, −sin m·theta],[sin m·theta, cos m·theta]] W x_m` — rotate the projected vector by an angle proportional to its position.

Lift to real `d`. The plane gave me one rotation at one frequency; the honest generalization is to chop `d`-dimensional space into `d/2` independent 2-planes and rotate each at its own frequency. Why is that allowed to give relative-ness in the full space? Because the inner product is a sum over the planes, each plane independently satisfies `<rotate_m, rotate_n> = relative-only` by the 2D argument, and a sum of relative-only-per-plane is relative-only — linearity of the inner product does the gluing. So `f_{q,k}(x_m, m) = R^d_{Theta,m} W_{q,k} x_m` with `R^d_{Theta,m}` block-diagonal, the `i`-th `2×2` block a rotation by `m·theta_i`. The payoff is the clean compositional identity: rotations compose by adding angles, `(R^d_{Theta,m})^T R^d_{Theta,n} = R^d_{Theta,n−m}`, so

  q_m^T k_n = (R^d_{Theta,m} W_q x_m)^T (R^d_{Theta,n} W_k x_n) = x_m^T W_q^T R^d_{Theta,n−m} W_k x_n.

The offset sits in a single rotation matrix sandwiched between the content projections. No learned table, no clip, no bias bucket. And `R` is orthogonal, so it preserves norms — applying it cannot blow up or collapse the representation as it propagates. What frequencies? I have `d/2` to pick, and the sinusoid keeps pulling me: its geometric wavelength spread is what gave multi-resolution coverage, fast hands for local offsets, slow hands for global, so reuse exactly that schedule `theta_i = 10000^{-2(i−1)/d}` for `i = 1..d/2`. This makes the construction literally the relative-rotation version of sinusoidal encoding. Learning the `theta_i` barely moves them from this init, so there is no reason to spend parameters; freeze them. And one practical realization: I will not build that sparse block-diagonal matrix and matrix-multiply, that is `O(d^2)` for an `O(d)` operation. Rotating the pair `(x_{2i}, x_{2i+1})` by `m·theta_i` is `x_{2i} cos − x_{2i+1} sin` and `x_{2i+1} cos + x_{2i} sin`, so elementwise it is `x ⊙ cos_vec + rotate(x) ⊙ sin_vec`, where `rotate` does the per-pair 90-degree swap and `cos_vec`/`sin_vec` are the cosines/sines at the per-position angles. Two elementwise multiplies and an add. (The split-half layout `rotate_half(x) = cat(−x_2, x_1)` over contiguous halves, with `cos`/`sin` each concatenated with themselves, is the same rotation up to a fixed permutation of the head dimension and is cleaner in tensor code; I will use that form.)

Good — that is the position pain solved: drop the learned `wpe`, set the module to carry its own position, rotate `q` and `k` per head by their position angles. Now the second pain, which is completely independent of where position lives, and I want to be careful not to let the RoPE solution make me think I am done.

The score `q_m^T k_n` is unbounded, and the softmax sees only differences. Stare at a concrete row of logits `[760, 752, 750]`: its softmax is `[0.99962, 0.00034, 0.00005]`, essentially one-hot. And `softmax([12, 4, 2])` is exactly the same distribution, because the differences are the same. So a gap of 8 between the top logit and the next is enough to all-but-silence everything else, and whether that gap rides on a baseline of 6 or of 750 is irrelevant. The trouble is that when the logits live at large magnitudes — hundreds — an 8-point gap is *small on a relative basis*, just over one percent, yet it produces a near-deterministic choice. So a head can collapse into winner-take-all not because one connection is decisively more important but because the dot products happen to be big and even a slight, possibly accidental, lead gets amplified into a near one-hot row with near-zero entropy. And here is the part that makes it a *training* problem and not just an expressivity problem: this gets worse as training proceeds. It is documented that scaling these models up makes the loss diverge after a few thousand steps, and the cause is traced to attention logits growing extremely large — the same one-hot, zero-entropy collapse, now bad enough to kill the run. At small scale you can reproduce it just by cranking the learning rate, and the diagnostic is the max attention logit (largest in the early layers) climbing without bound. So this is not hypothetical; it is the failure mode I have to keep this run away from.

Why is `1/sqrt(d_k)` not already handling this? Let me re-derive exactly what that factor does. Model the components of `q` and `k` as independent, mean zero, variance one. Then `q·k = sum_{i=1}^{d_k} q_i k_i`, each product has mean `E[q_i]E[k_i] = 0` and variance `E[q_i^2]E[k_i^2] = 1`, so the sum has mean 0 and variance `d_k`; the typical score scales like `sqrt(d_k)`, and dividing by `sqrt(d_k)` pulls the variance back to 1. But read what I just proved: it controls the scale *under the initialization assumptions* — independent, unit-variance `q` and `k`. The moment training moves the projection matrices, those assumptions rot. The query and key norms grow, their components correlate, and `q·k` is free to grow again. `1/sqrt(d_k)` is a one-time variance normalization at `t = 0`; it sets the expected scale of a fresh layer and then does nothing. It is not a bound. Nothing in the mechanism stops the logits from drifting back up into saturation over training. That is the gap.

What I actually want is a bound on the per-vector magnitude that holds for *all* of training, not just at init. Decompose the score: `q·k = ||q|| ||k|| cos(angle)`. The directional part `cos` is already in `[−1, 1]`; it is the magnitudes `||q||`, `||k||` that are unbounded and that `1/sqrt(d_k)` only momentarily tames. So if the magnitudes are the saboteur, take control of them: fix `||q||` and `||k||` to a constant, for every token, for the whole run. I already have exactly the tool — a normalization that pins a vector's magnitude. RMSNorm maps `a` to `a / RMS(a) · g` with `RMS(a) = sqrt((1/n) sum a_i^2)`, so after RMSNorm each component has unit root-mean-square (times the gain), and a `d_k`-vector therefore has Euclidean norm `sqrt(d_k)` times the gain's RMS — pinned, independent of how large the raw projection got. (LayerNorm would do the same; but its extra move is re-centering, subtracting the mean, and the relevant fact is that re-centering does not reduce the variance of the activations or gradients — it is the *re-scaling* that controls magnitude. So drop the centering and use RMSNorm: same magnitude control, cheaper, and it is a clean drop-in.) Apply it to `q` and `k` before they are dotted.

Now be careful about *which* vectors and *along which* dimension, because there are several places one could normalize and they are not equivalent. The dot product that feeds the softmax is computed *per head*, between the head-dimension slices of length `d_k = d / n_head`, *after* the multi-head split. Normalizing the full pre-split embedding does not make the per-head slices have controlled norm, and it is the per-head slice geometry the softmax sees. So the normalization has to be *after* the split, *along the head dimension* — the dimension the dot product contracts. And which of `q, k, v`? Only `q` and `k`. The values `v` never go through the softmax; they are the things being averaged by the attention weights, not scored against each other. Normalizing `v` would deform the averaged output to solve a problem that does not exist. So: RMSNorm the per-head queries and keys along the head dimension, leave the values alone.

Now the magnitude bound is real and permanent. After RMSNorm with no extra gain, each component of `q` has unit root-mean-square, so `||q||_2 = sqrt(d_k)` exactly, and likewise `||k||_2 = sqrt(d_k)`, regardless of how large the raw projection grew — I just watched a `q` with norm in the hundreds come out at exactly `sqrt(d_k)`. By Cauchy-Schwarz `|q·k| <= ||q|| ||k|| = d_k`, so `|q·k| / sqrt(d_k) <= sqrt(d_k)` — the raw logit is now ceilinged by a fixed function of `d_k`, no longer by the freely-growing projection norms. If I were to add RMSNorm's optional learned per-element gain back in, the ceiling would scale by that gain's magnitude, but the gain is a single small learned parameter under weight decay, not a freely drifting projection, so it cannot run away the way the unconstrained norms did. Here I do not even need the gain: pinning the norm to exactly `sqrt(d_k)` and dividing by `sqrt(d_k)` is already the right calibration, and the residual-stream normalization and `1/sqrt(d_k)` set the overall scale — so I apply RMSNorm to `q, k` with no learned weight. The logit-growth instability is closed off by construction, which is the kind of guarantee `1/sqrt(d_k)` could never give because it was a calibration at init rather than a property of the operation.

And — this is the decision I want to be deliberate about — I am keeping `1/sqrt(d_k)`, not replacing it. There is a sibling fix that goes further: L2-normalize `q` and `k` to *unit* length so each entry is a bare cosine in `[−1, 1]`, and then *replace* `1/sqrt(d_k)` with a learnable temperature `g_0 ≈ log2(L^2 − L)` to stretch the squeezed cosines back out, because a row of values all in `[−1, 1]` softmaxes to nearly uniform and a head could never concentrate. That works, but it throws away the magnitude entirely and re-buys the temperature with a separate learned scalar, which is a heavier rewrite of the scoring than I need. RMSNorm does something gentler: it does not crush the norm to 1, it *fixes* it to `sqrt(d_k)`. So the logits are not crammed into `[−1, 1]`; they sit at a controlled scale of order `sqrt(d_k)` before the divide, and `1/sqrt(d_k)` brings that back to the `O(1)` range the softmax wants — exactly the calibration it was designed for, except now the premise that the per-vector scale is `~sqrt(d_k)` is *enforced for all of training* rather than merely true at init. There is no need for a separate learned temperature: the magnitude is fixed, `1/sqrt(d_k)` recovers `O(1)`, and softmax can still be as sharp or as flat as the content demands because the cosine between two `sqrt(d_k)`-norm vectors, times `sqrt(d_k)`, spans a perfectly usable range. So the score becomes `softmax(RMSNorm(q)^T RMSNorm(k) / sqrt(d_k))` and I leave `1/sqrt(d_k)` exactly where it was. Minimal change, maximal reuse of what already works.

Now the question that actually justifies *combining* the two fixes rather than picking one: do RoPE and the RMSNorm-on-`q`,`k` interfere? They are aimed at orthogonal failure modes — RoPE controls *where* position lives in the logit (relative offset only), QK-norm controls *how big* the logit can get (magnitude bounded for the whole run). One is about the relative-position content of `q·k`, the other about its scale. The reason I can just stack them, and the reason it does not matter much which order, is the property that dropped out of the 2D derivation: RoPE is a rotation, and a rotation is orthogonal, so it *preserves the norm*. Trace it through. If I RMSNorm first and rotate second — fix `||q||, ||k||` to `sqrt(d_k)`, then apply `R_m`, `R_n` — the rotation does not change the norm, so the magnitude control RMSNorm installed *survives* the rotation: the logit is still ceilinged at `d_k`, and it equals `(RMSNorm(x_m))^T W_q^T R_{n−m} W_k (RMSNorm(x_n))`, still relative-only. If instead I rotate first and RMSNorm second, the RMSNorm reads the rotated vector and pins *its* norm — but the rotated vector has the same norm as the unrotated one, so I land at the same fixed magnitude, and because RMSNorm is applied along the same head dimension the rotation acts on, the relative-position structure of the dot product is undisturbed. The two orders are near-equivalent precisely because `R` is orthogonal. So there is no conflict to resolve: each fix leaves the other's guarantee intact. I will normalize before rotating — pin the magnitude on the raw projected `q, k`, then let RoPE carry position into the now-magnitude-controlled vectors — which matches the order the standard pretraining recipe uses, but the orthogonality argument is what tells me it is safe either way.

Let me make sure nothing about the surrounding stack fights this. I am dropping the learned absolute position embedding, so I flip the module's position flag off — RoPE is the only position signal now, and leaving `wpe` on would double-count and reintroduce the absolute contamination I just removed. The RMSNorm on `q, k` lives strictly inside attention, on the per-head scores; it complements, not replaces, whatever normalization sits on the residual stream. And the causal mask and the rest of the softmax path are untouched, so I can keep the fast fused `scaled_dot_product_attention` with `is_causal=True` — its math is identical to the manual masked-softmax-with-`1/sqrt(d_k)` path, which I keep as a fallback.

So let me write the scoring path I would actually ship, filling the two open slots — does the module carry position (yes, via RoPE, so `use_pos_emb = False`) and what transforms the per-head `q, k` before scoring (RMSNorm then RoPE):

```python
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


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
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1, 1, config.block_size, config.block_size),
            )
        # RoPE carries position now, so the model must NOT add learned wpe on top.
        self.use_pos_emb = False
        # Frozen RoPE frequencies theta_i = 10000^{-2i/head_dim}: the sinusoidal
        # geometric spread, fast planes (local) to slow planes (global).
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

    def _apply_rope(self, x, seq_len):
        # x: [B, n_head, T, head_dim]. Rotate each 2-plane by an angle = position * theta_i.
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)                  # [T, head_dim/2] = m * theta_i
        cos = freqs.cos().unsqueeze(0).unsqueeze(0)
        sin = freqs.sin().unsqueeze(0).unsqueeze(0)
        d = x.shape[-1] // 2
        x1, x2 = x[..., :d], x[..., d:]                        # split-half layout
        # block rotation: [x1, x2] -> [x1 cos - x2 sin, x1 sin + x2 cos]
        y1 = x1 * cos - x2 * sin
        y2 = x1 * sin + x2 * cos
        return torch.cat([y1, y2], dim=-1).type_as(x)

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        # QK-norm THEN RoPE: RMSNorm pins ||q||,||k|| along the head dim so the logit
        # magnitude can't run away over training; the rotation is norm-preserving, so
        # that magnitude control survives, and the logit ends up relative-only AND bounded.
        q = self._apply_rope(F.rms_norm(q, (q.size(-1),)), T)
        k = self._apply_rope(F.rms_norm(k, (k.size(-1),)), T)

        if self.flash:
            # identical math to the masked-softmax path below, with 1/sqrt(d_k) kept
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))   # 1/sqrt(d_k) kept
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y
```

The chain is tight. Attention is order-blind and order only matters through the logit `q_m^T k_n`, so I demanded the logit depend on contents and the relative offset `m − n` alone; solving that functional equation in 2D over the complex plane forced position to be a *pure rotation* (magnitude carries no position), which tiled across `d/2` geometric-frequency planes lifts to a single relative rotation `R_{n−m}` between the content projections — no learned position table, no length cap, and orthogonal so norm-preserving. Separately, the score is unbounded and the softmax only sees differences, so large logits saturate it into near-one-hot, zero-entropy rows, and `1/sqrt(d_k)` only calibrates the scale at init, not for all of training — which is the documented logit-growth divergence. Bounding the magnitude permanently means pinning `||q||, ||k||`, which is exactly RMSNorm applied to the per-head `q, k` along the head dimension (values left alone, since they are averaged not scored); this ceilings the logit at a fixed function of `d_k`, and I keep `1/sqrt(d_k)` because RMSNorm fixes the scale to `~sqrt(d_k)` rather than crushing it to a cosine, so the original calibration is exactly what brings it back to `O(1)` — now enforced through training instead of merely true at init. The two fixes target orthogonal failures and combine without interference precisely because RoPE is a rotation: the norm RMSNorm pins is preserved by the rotation, so the logit comes out both relative-only and magnitude-bounded, in either order. In code it is just `F.rms_norm(q, (d_k,))` then the elementwise `x*cos + rotate(x)*sin`, with `1/sqrt(d_k)` and the causal softmax left exactly where they were.
