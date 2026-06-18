OK, let me think this through from scratch, because the thing that hurts is concrete and I want to stare at it before reaching for any architecture. Attention trains beautifully in parallel — I form \(QK^\top\), an \(N\times N\) block of scores, softmax it, multiply by \(V\), and because the whole sequence is known at training time all \(N\) output rows come out of two big matrix multiplies that saturate the GPU. The trouble is generation. To produce token \(n\) I take its query \(q_n\) and compare it against the keys of every earlier token, \(O(n)\) work for one step, and I have to keep all those past keys and values around — a cache that grows linearly with how far I've gotten. So per-step decode cost grows with context, and memory grows with context. The deployment story is bad exactly where language models are headed: long context.

Now flip to the recurrent network. It carries a fixed-size state \(s_n\), folds in the new input, reads off an output: constant work per step, constant memory, no growing cache. That is the decode profile I want. But an RNN's state update \(s_n = f(s_{n-1}, x_n)\) is sequential — each step waits on the last — so I can't parallelize training over the time axis the way attention does. So I have two operators with mirror-image strengths: attention parallelizes training but its inference is \(O(n)\) per step with growing memory; the RNN infers in \(O(1)\) but won't parallelize. The question I actually want to answer: is there a *single* operator that has both — a parallel form to train and a recurrent form to infer — computing the same function, without giving up quality?

Let me think about why I can't just take attention and "run it recurrently." The blocker is the softmax. The score \(\exp(q_n\cdot k_m)\) tangles \(q_n\) and \(k_m\) inside the exponential; it does not split into something-about-the-query times something-about-the-key. Because it doesn't factor, I can't fold the past into a fixed summary — every old key has to be revisited through that nonlinearity. People have attacked this from three directions and each gives up something. One: linearize attention — replace \(\exp(q\cdot k)\) by \(\phi(q)^\top\phi(k)\), so by associativity \((\phi(Q)\phi(K)^\top)V=\phi(Q)(\phi(K)^\top V)\), the key-side sum becomes a running state, and you get an \(O(1)\) recurrence. But the output is a *normalized* average, \(\phi(q_n)^\top S_n/(\phi(q_n)^\top z_n)\) with a denominator \(z_n=\sum_{m\le n}\phi(k_m)\), and that normalizer is the part that misbehaves — it dilutes the weights, it's sensitive to \(\phi\), and the models end up below Transformers and bad at position. Two: go back to an explicit recurrence but make it cheap with element-wise channel operations — \(O(1)\) inference, decent quality, but element-wise mixing caps capacity and it doesn't parallelize over time as a matmul would. Three: swap attention for a state-space recurrence \(s_n=As_{n-1}+Bx_n,\,o_n=Cs_n\) whose unrolled form is a convolution you can FFT — parallel and good at long range, but \(A,B,C\) don't depend on the token content, so it's not doing the content-based comparison attention does.

So nobody sits in all three corners at once. Let me not try to *approximate softmax* — that's what the linearization people did, and it inherits softmax's normalizer baggage. Let me instead start from the recurrence and ask what parallel form it naturally admits, then see if that form looks like attention. Start as general as I can: a linear recurrence with a *state matrix*, not a scalar decay. Project the input to a value \(v_n\), carry a state \(s_n\), and read out with a query:
\[
s_n = A\,s_{n-1} + k_n^\top v_n, \qquad o_n = q_n s_n,
\]
with \(A\in\mathbb{R}^{d\times d}\), \(k_n,q_n\in\mathbb{R}^{1\times d}\). The state accumulates an outer-product-like term \(k_n^\top v_n\) each step, transformed by \(A\). Now unroll it — substitute \(s_{n-1}\) into \(s_n\) repeatedly. \(s_1 = k_1^\top v_1\); \(s_2 = A k_1^\top v_1 + k_2^\top v_2\); \(s_n = \sum_{m=1}^n A^{n-m} k_m^\top v_m\). Read out:
\[
o_n = q_n s_n = \sum_{m=1}^n q_n A^{n-m} k_m^\top v_m.
\]
Stare at that. This is the bridge I was looking for. A linear recurrence, *unrolled*, is already a weighted sum over the entire past — every term \(m\le n\) contributes, weighted by \(q_n A^{n-m} k_m^\top\). That is the shape of causal attention: output at \(n\) is a sum over \(m\le n\) of (interaction of position \(n\) with position \(m\)) times \(v_m\). The recurrence gives me the \(O(1)\) inference for free, and the unrolled sum is a candidate parallel form. The whole game is now in the matrix \(A^{n-m}\): I need to make this weight content-aware and cheap enough to compute in parallel.

First, content-awareness — this is what separates me from the state-space line, where the mixing doesn't see the tokens. Make the query and key projections depend on the input: \(Q = XW_Q\), \(K = XW_K\), learnable \(W_Q,W_K\in\mathbb{R}^{d\times d}\). Now \(q_n,k_m\) are computed from content, so \(q_n A^{n-m}k_m^\top\) is a genuine content-based score, modulated by \(A^{n-m}\).

Now \(A^{n-m}\). A general matrix power is the expensive, opaque part. Let me diagonalize: \(A = \Lambda\,\mathrm{diag}(\gamma e^{i\theta})\,\Lambda^{-1}\) with \(\gamma,\theta\in\mathbb{R}^d\) — I'm allowing complex eigenvalues, written in polar form as a magnitude \(\gamma\) and a phase \(\theta\) per dimension. Then \(A^{n-m} = \Lambda\,\mathrm{diag}(\gamma e^{i\theta})^{n-m}\,\Lambda^{-1}\), and the \(\Lambda,\Lambda^{-1}\) sit on the outside, multiplying \(q_n\) on the left and \(k_m^\top\) on the right. So absorb \(\Lambda\) into \(W_Q\) and \(\Lambda^{-1}\) into \(W_K\) — they're learnable matrices anyway, the change of basis is free. What's left is a *diagonal* power:
\[
o_n = \sum_{m=1}^n q_n\,(\gamma e^{i\theta})^{n-m}\,k_m^\top v_m,
\]
where \((\gamma e^{i\theta})^{n-m}\) acts coordinate-wise. Now split the relative exponent across the two positions: \((\gamma e^{i\theta})^{n-m} = (\gamma e^{i\theta})^{n}\,(\gamma e^{i\theta})^{-m}\), and attach each piece to its own factor:
\[
o_n = \sum_{m=1}^n \big(q_n(\gamma e^{i\theta})^{n}\big)\big(k_m(\gamma e^{i\theta})^{-m}\big)^\top v_m.
\]
Look at what each factor is. \(q_n(\gamma e^{i\theta})^n\) is the query, scaled in magnitude by \(\gamma^n\) and rotated by \(e^{in\theta}\); \(k_m(\gamma e^{i\theta})^{-m}\) carries the inverse magnitude and phase on the key side. That is the xPos shape: query and key receive reciprocal position-dependent factors so their interaction depends on the relative offset. So the position encoding I'd normally bolt on by hand falls out of the recurrence's state matrix. That's a good sign — it means the decay-and-rotation isn't an arbitrary add-on, it's what \(A^{n-m}\) *is* once diagonalized.

Carrying a separate magnitude \(\gamma_i\) per dimension is more bookkeeping than I want, and the \(\gamma^{-m}\) on the key grows unboundedly as \(m\) shrinks, which is numerically ugly. Let me simplify \(\gamma\) from a per-dimension vector to a single scalar (per head). Then \(\gamma^{n-m}\) pulls out of the per-coordinate structure entirely, and I can keep just the phase rotation inside the query/key factors:
\[
o_n = \sum_{m=1}^n \gamma^{n-m}\,\big(q_n e^{in\theta}\big)\big(k_m e^{im\theta}\big)^\dagger v_m,
\]
with \(\dagger\) the conjugate transpose. This is the sign-sensitive point: I rotate the key by the same positive phase before taking the conjugate inner product, so the score contains the relative phase \(e^{i(n-m)\theta}\). In a real implementation, rotating both \(q\) and \(k\) with the same RoPE map and then taking an ordinary dot product supplies the same conjugate effect through \(R(n)^\top R(m)\). The scalar \(\gamma^{n-m}\) is now a clean per-distance decay multiplying a rotary-encoded content score. This is fully parallelizable: every term in the sum is an independent product, no nonlinearity coupling positions, no softmax. I started from a recurrence and I've landed on a parallel, position-aware, content-based weighted sum. Let me name this operator — call it retention, since it's literally a state that *retains* a decaying summary of the past — and pin down its three faces.

The parallel face, for training. Pack the rotation into the projections: \(Q = (XW_Q)\odot\Theta\), \(K=(XW_K)\odot\overline{\Theta}\), \(V=XW_V\), with \(\Theta_n = e^{in\theta}\) and \(\overline\Theta\) its conjugate. The decay-and-causality go into one matrix:
\[
D_{nm} = \begin{cases}\gamma^{\,n-m}, & n\ge m\\[2pt] 0, & n<m.\end{cases}
\]
This single matrix \(D\) does two jobs at once: above the diagonal it is zero, which is the causal mask (a position can't see the future); on and below the diagonal it is \(\gamma^{n-m}\), the exponential decay by relative distance. Then
\[
\mathrm{Retention}(X) = (QK^\top\odot D)\,V.
\]
\(QK^\top\) is the content score matrix, \(\odot D\) masks-and-decays it, times \(V\) gives the outputs — all positions at once, two matmuls and an elementwise multiply, exactly the GPU-friendly shape attention had, but with softmax deleted and the decay mask in its place.

The recurrent face, for inference. I want to recover the running state. Recall \(s_n=\sum_{m\le n}\gamma^{n-m}k_m^\top v_m\) (now with the scalar \(\gamma\) and the rotation folded into \(k,q\)). Write it as a recurrence in the \(d_k\times d_v\) state \(S\):
\[
S_n = \gamma S_{n-1} + K_n^\top V_n, \qquad \mathrm{Retention}(X_n) = Q_n S_n.
\]
Each step: decay the old state by \(\gamma\), add the rank-one outer product \(K_n^\top V_n\), read out by left-multiplying with \(Q_n\). State is fixed size \(d_k\times d_v\) regardless of \(n\) — constant memory, constant work per step. That's the \(O(1)\) inference.

I should *prove* these two faces compute the same thing, not just assert it, because the whole premise is that I can train with one and infer with the other. Unroll the recurrence: \(S_1 = K_1^\top V_1\), \(S_2 = \gamma K_1^\top V_1 + K_2^\top V_2\), and in general
\[
S_n = \sum_{m=1}^n \gamma^{\,n-m} K_m^\top V_m.
\]
Then the recurrent output is
\[
Q_n S_n = Q_n\sum_{m=1}^n \gamma^{\,n-m}K_m^\top V_m = \sum_{m=1}^n \gamma^{\,n-m}\,(Q_n K_m^\top)\,V_m,
\]
where I pulled \(Q_n\) inside the sum and grouped \(Q_n K_m^\top\) as the scalar score. Now compare to row \(n\) of the parallel form. \((QK^\top\odot D)\) has entry \((n,m)\) equal to \((Q_nK_m^\top)\,D_{nm} = (Q_nK_m^\top)\,\gamma^{n-m}\) for \(m\le n\) and \(0\) for \(m>n\). Multiplying that row by \(V\) gives \(\sum_{m\le n}\gamma^{n-m}(Q_nK_m^\top)V_m\) — the upper-triangular entries vanish because \(D_{nm}=0\) there, which is why the recurrence only sums \(m\le n\). Term for term, identical. So the parallel and recurrent faces are the same function; the causal mask in \(D\) is the same statement as "the state only accumulates the past." Good — train with the matmul, infer with the recurrence, no approximation.

Now the third face, and this one I have to actually work out because it's not obvious. The parallel form is \(O(N^2)\) — fine for short training sequences, but for long sequences I'd like to keep memory bounded; the pure recurrence is \(O(N)\) but sequential, so it underuses the GPU during training. I want a hybrid: chop the sequence into chunks, run the *parallel* form *inside* each chunk (so I still get the matmul parallelism), and pass information *across* chunks with the *recurrence* (so memory doesn't blow up with the number of chunks). Let chunk length be \(B\); chunk \(i\) covers global positions \(Bi,\dots,B(i+1)-1\); let \(j=0,\dots,B-1\) index position within a chunk, so global position is \(Bi+j\).

For a query at local position \(j\) in chunk \(i\), its output splits into two pieces: contributions from keys *inside the same chunk* (local positions \(\le j\)) and contributions from keys in *all earlier chunks*. The inside part is just the parallel form restricted to the chunk: \((Q_{[i]}K_{[i]}^\top\odot D)\,V_{[i]}\), with the same \(B\times B\) decay-mask \(D\). The cross part needs a summary of everything before chunk \(i\), carried recurrently. Let \(R_{i-1}\) be that summary — a \(d_k\times d_v\) state holding the keys/values of chunks \(0..i-1\). I have to be careful about *where the decay clock is zeroed*, because that's the only place this can go wrong.

Think about a single key at local position \(j'\) inside chunk \(i\), with value \(V_{[i],j'}\). When does its decay get measured from? If I'm going to store it into the running state \(R\) and have a *later* chunk read it, I want \(R\) to hold its contribution measured *up to the end of chunk \(i\)* — i.e. to the last local position \(B-1\) of that chunk — so that a reader only has to add the gap from chunk \(i\)'s boundary onward. The decay from local position \(j'\) to the chunk's last position \(B-1\) is \(\gamma^{(B-1)-j'}\). So when I fold chunk \(i\) into the state I pre-weight each value by that within-chunk decay. Write \(\zeta_{j'} = \gamma^{\,B-1-j'}\) and accumulate:
\[
R_i = K_{[i]}^\top\,(V_{[i]}\odot\zeta) + \gamma^{B} R_{i-1}.
\]
The \(\gamma^B R_{i-1}\) is the cross-chunk decay: advancing the running summary by one whole chunk multiplies it by \(\gamma^B\), because the boundary moved \(B\) positions forward. So \(R_i\) is the state, decayed to the boundary of chunk \(i\), of all keys in chunks \(0..i\).

Now the reader. A query at local position \(j\) in chunk \(i\) reads \(R_{i-1}\), which is the summary of chunks \(0..i-1\) measured up to the boundary of chunk \(i-1\) — that boundary is the last position of chunk \(i-1\), global position \(B i - 1\). The query is at global position \(Bi + j\). The decay from that boundary to the query is \(\gamma^{(Bi+j) - (Bi-1)} = \gamma^{\,j+1}\). So the cross-chunk output is \(R_{i-1}\) read by the query, scaled by \(\gamma^{j+1}\):
\[
\text{cross}_j = (Q_{[i]} R_{i-1})_j\cdot \gamma^{\,j+1}, \qquad \xi_j = \gamma^{\,j+1}.
\]
Putting the two together,
\[
\mathrm{Retention}(X_{[i]}) = \underbrace{(Q_{[i]}K_{[i]}^\top\odot D)\,V_{[i]}}_{\text{inner-chunk, parallel}} \;+\; \underbrace{(Q_{[i]}R_{i-1})\odot\xi}_{\text{cross-chunk, recurrent}}.
\]
Let me sanity-check the decay bookkeeping is globally consistent by tracing one key all the way to one query. Take a key at local position \(j'\) in chunk \(i-1\) (global \(B(i-1)+j'\)) and a query at local \(j\) in chunk \(i\) (global \(Bi+j\)). The true relative distance is \((Bi+j)-(B(i-1)+j') = B + j - j'\), so the correct weight is \(\gamma^{B+j-j'}\). Now trace it through the chunkwise path. When chunk \(i-1\) was folded into \(R_{i-1}\), that key got pre-weighted by \(\zeta_{j'} = \gamma^{(B-1)-j'}\). Then \(R\) didn't advance another full chunk before being read (the reader is the very next chunk \(i\), reading \(R_{i-1}\) directly, no extra \(\gamma^B\) factor applied to \(R_{i-1}\) at read time). Then the reader scales by \(\xi_j = \gamma^{j+1}\). Total exponent: \((B-1-j') + (j+1) = B + j - j'\). Matches \(\gamma^{B+j-j'}\) exactly. The clock zeroing is consistent — the within-chunk decay to the boundary plus the boundary-to-query decay reconstitutes the true relative decay. So the chunkwise face equals the parallel face on the cross terms too; combined with the inner term being literally the parallel form, the chunkwise computation equals the full retention. Three faces, one function. Complexity per chunk: the inner matmuls are \(O(B^2 d)\), the state update and read are \(O(B d^2)\), so over \(N/B\) chunks the total is \(O\big(N(B + d)d\big)\) — linear in \(N\). That's the long-sequence training mode.

Now, is this single-head thing expressive enough? A scalar \(\gamma\) fixes one decay rate — one timescale of memory. But different parts of language want different horizons: some heads should keep a long tail of context, some should be sharply local. With attention I'd get diversity from multiple heads in different subspaces; here I have a second axis to vary — the decay rate itself. So use \(h\) heads, each with its own \(\gamma\), spanning a range of timescales. Multi-scale retention: pick \(\gamma\) per head as something like \(\gamma = 1 - 2^{-5-\mathrm{arange}(0,h)}\), so the decays geometrically span from "fast forgetting" to "almost no decay." Each head runs the retention above with its own \(\gamma_i\) (and shares the rotation \(\theta\)). Then concatenate the heads.

But there's a wrinkle that comes precisely *from* the multi-scale choice. Heads with different \(\gamma\) produce outputs with different magnitudes — a near-1 \(\gamma\) sums many terms and grows large, a small \(\gamma\) sums few. If I normalize all heads jointly, the high-variance heads swamp the rest. So normalize each head *separately*: group normalization over the head dimension, one group per head, which balances the per-head variances before mixing. This is the Sub-LayerNorm idea — normalize inside the sublayer per head.

There's a second thing I lost when I deleted the softmax: a nonlinearity. Softmax was doing double duty — normalizing *and* injecting a nonlinearity into the mixing. Retention is now an entirely linear map from \(V\) to output (given the scores). A stack of linear maps with linear FFNs in between would be underpowered. I want to restore gating-style nonlinearity without bringing back the \(O(n)\) softmax. Add a content-dependent gate on the output: project the input through \(W_G\), push it through a swish nonlinearity, and multiply it elementwise into the normalized retention output before the final projection. So with \(Y = \mathrm{GroupNorm}_h(\mathrm{Concat}(\mathrm{head}_1,\dots,\mathrm{head}_h))\),
\[
\mathrm{MSR}(X) = \big(\mathrm{swish}(XW_G)\odot Y\big)\,W_O.
\]
The gate gives a multiplicative, data-dependent nonlinearity — the missing expressiveness — and \(W_O\) mixes the heads back to model dimension.

Now numerics, because in practice the scores \(QK^\top\odot D\) can have a wide dynamic range — \(\gamma^{n-m}\) decays geometrically, and summing many terms can blow up or underflow. Here the per-head normalization I just added pays off in an unexpected way: it is scale-invariant, \(\mathrm{GroupNorm}(\alpha\cdot\mathrm{head}) = \mathrm{GroupNorm}(\mathrm{head})\), and I can implement the same per-head scale invariance with RMSNorm. That means I can multiply a head's pre-normalization output by convenient scalar factors without changing the normalized result. That buys three stabilizers. First, scale the content scores like attention does: use \(QK^\top/\sqrt{d}\), implemented by scaling \(K\), so dot products don't grow with head dimension. Second, row-normalize the decay mask, \(\tilde D_{nm} = D_{nm}/\sqrt{\sum_{i=1}^n D_{ni}}\), so a row's total decay weight doesn't explode for late positions. Third, normalize the full score matrix by row magnitude: with \(R = QK^\top\odot D\), divide by a clamped row absolute sum before multiplying by \(V\). In the recurrent and chunkwise paths the same idea appears as explicit scale tracking and scale alignment; it is not the bare mathematical recurrence, but it is the same function after the per-head normalization.

Stack it into the architecture. A block, pre-norm and residual like a Transformer block but with multi-scale retention where attention used to be, then a feed-forward:
\[
Y^l = \mathrm{MSR}(\mathrm{LN}(X^l)) + X^l,\qquad X^{l+1} = \mathrm{FFN}(\mathrm{LN}(Y^l)) + Y^l,
\]
with \(\mathrm{FFN}(X) = \mathrm{gelu}(XW_1)W_2\). One more bookkeeping detail to keep comparisons fair: I should match parameter counts to a Transformer. Attention uses \(W_Q,W_K,W_V,W_O\in\mathbb{R}^{d\times d}\) (\(\approx 4d^2\)) and an FFN with intermediate \(4d\) (\(\approx 8d^2\)). Retention adds the gate \(W_G\) and widens the value head; setting \(W_Q,W_K\in\mathbb{R}^{d\times d}\), \(W_G,W_V\in\mathbb{R}^{d\times 2d}\), \(W_O\in\mathbb{R}^{2d\times d}\) gives \(\approx 8d^2\) in retention, so to keep the total matched I shrink the FFN intermediate to \(2d\). Widening \(V\) to twice the \(Q/K\) dimension also makes sense from the recurrent view: the state \(S\) is \(d_k\times d_v\), so a bigger \(d_v\) is literally more memory capacity in the hidden state.

Now write it as code, three computation paths sharing the same projections and the same rotation-and-decay, so that the parallel path trains, the recurrent path decodes, and the chunkwise path handles long sequences — and they all compute the one retention function I proved equivalent above, with the stabilizing scale terms included rather than hidden.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    # the per-head norm I settled on: RMS (no mean-subtract), still scale-invariant
    def __init__(self, dim, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x):
        x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return x if self.weight is None else x * self.weight


def rotate_every_two(x):
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def theta_shift(x, sin, cos):
    return (x * cos) + (rotate_every_two(x) * sin)


class RetNetRelPos(nn.Module):
    def __init__(self, embed_dim, num_heads, chunk_size=512):
        super().__init__()
        # rotary angles theta (the e^{i theta} phase), shared across heads
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        # multi-scale decay: gamma_h = 1 - 2^{-5-h}, one per head, kept in log space
        decay = torch.log(1 - 2 ** (-5 - torch.arange(num_heads, dtype=torch.float)))
        self.register_buffer("angle", angle)
        self.register_buffer("decay", decay)
        self.recurrent_chunk_size = chunk_size

    def forward(self, slen, activate_recurrent=False, chunkwise_recurrent=False):
        if activate_recurrent:
            sin = torch.sin(self.angle * (slen - 1))
            cos = torch.cos(self.angle * (slen - 1))
            return (sin, cos), self.decay.exp()

        index = torch.arange(slen).to(self.decay)
        sin = torch.sin(index[:, None] * self.angle[None, :])
        cos = torch.cos(index[:, None] * self.angle[None, :])

        if chunkwise_recurrent:
            b = self.recurrent_chunk_size
            block_index = torch.arange(b).to(self.decay)
            tri = torch.tril(torch.ones(b, b).to(self.decay))
            raw = torch.masked_fill(
                block_index[:, None] - block_index[None, :],
                ~tri.bool(),
                float("inf"),
            )
            raw = torch.nan_to_num(torch.exp(raw * self.decay[:, None, None]))
            value_inner_decay = raw[:, -1] / raw[:, -1].sum(dim=-1, keepdim=True)
            value_inner_decay = value_inner_decay.unsqueeze(-1)
            scale = raw.sum(dim=-1, keepdim=True).sqrt()
            inner_mask = raw / scale
            cross_decay = torch.exp(self.decay * b)[:, None, None]
            query_inner_decay = torch.exp(self.decay[:, None] * (block_index + 1))
            query_inner_decay = query_inner_decay[:, :, None] / (
                scale / raw[:, -1].sum(dim=-1)[:, None, None]
            )
            return (sin, cos), (
                inner_mask,
                cross_decay,
                query_inner_decay,
                value_inner_decay,
            )

        mask = torch.tril(torch.ones(slen, slen).to(self.decay))
        mask = torch.masked_fill(index[:, None] - index[None, :], ~mask.bool(), float("inf"))
        mask = torch.nan_to_num(torch.exp(mask * self.decay[:, None, None]))
        mask = mask / mask.sum(dim=-1, keepdim=True).sqrt()
        return (sin, cos), mask


class MultiScaleRetention(nn.Module):
    def __init__(self, embed_dim, value_dim, num_heads, gate_fn="swish", layernorm_eps=1e-6):
        super().__init__()
        self.embed_dim = embed_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        self.head_dim = value_dim // num_heads
        self.key_dim = embed_dim // num_heads
        self.scaling = self.key_dim ** -0.5
        self.gate_fn = F.silu if gate_fn == "swish" else F.gelu
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, value_dim, bias=False)
        self.g_proj = nn.Linear(embed_dim, value_dim, bias=False)
        self.out_proj = nn.Linear(value_dim, embed_dim, bias=False)
        self.group_norm = RMSNorm(
            self.head_dim, eps=layernorm_eps, elementwise_affine=False
        )

    def parallel_forward(self, qr, kr, v, mask):
        bsz, tgt_len, _ = v.size()
        vr = v.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        qk = (qr @ kr.transpose(-1, -2)) * mask
        qk = qk / qk.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1, max=5e4)
        return (qk @ vr).transpose(1, 2)

    def recurrent_forward(self, qr, kr, v, decay, incremental_state):
        bsz = v.size(0)
        v = v.view(bsz, self.num_heads, self.head_dim, 1)
        kv = kr * v
        if "prev_key_value" in incremental_state:
            prev_kv = incremental_state["prev_key_value"]
            prev_scale = incremental_state["scale"]
            scale = prev_scale * decay + 1
            old = prev_kv * (prev_scale.sqrt() * decay / scale.sqrt()).view(
                self.num_heads, 1, 1
            )
            new = kv / scale.sqrt().view(self.num_heads, 1, 1)
            kv = old + new
        else:
            scale = torch.ones_like(decay)
        incremental_state["prev_key_value"] = kv
        incremental_state["scale"] = scale
        return torch.sum(qr * kv, dim=3)

    def chunk_recurrent_forward(self, qr, kr, v, inner_mask):
        mask, cross_decay, query_inner_decay, value_inner_decay = inner_mask
        bsz, tgt_len, _ = v.size()
        chunk_len = mask.size(1)
        assert tgt_len % chunk_len == 0
        num_chunks = tgt_len // chunk_len
        qr = qr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        kr = kr.view(bsz, self.num_heads, num_chunks, chunk_len, self.key_dim).transpose(1, 2)
        v = v.view(bsz, num_chunks, chunk_len, self.num_heads, self.head_dim).transpose(2, 3)

        kr_t = kr.transpose(-1, -2)
        qk = (qr @ kr_t) * mask
        inner_scale = qk.detach().abs().sum(dim=-1, keepdim=True).clamp(min=1)
        inner = (qk / inner_scale) @ v

        kv = kr_t @ (v * value_inner_decay)
        kv_state = torch.zeros(bsz, self.num_heads, self.key_dim, self.head_dim).to(v)
        kv_scale = torch.ones(bsz, self.num_heads, 1, 1).to(v)
        kv_recurrent, cross_scale = [], []
        for i in range(num_chunks):
            kv_recurrent.append(kv_state / kv_scale)
            cross_scale.append(kv_scale)
            kv_state = kv_state * cross_decay + kv[:, i]
            kv_scale = (
                kv_state.detach()
                .abs()
                .sum(dim=-2, keepdim=True)
                .max(dim=-1, keepdim=True)
                .values
                .clamp(min=1)
            )
        kv_recurrent = torch.stack(kv_recurrent, dim=1)
        cross_scale = torch.stack(cross_scale, dim=1)
        all_scale = torch.maximum(inner_scale, cross_scale)
        cross = (qr * query_inner_decay) @ kv_recurrent
        output = inner / (all_scale / inner_scale) + cross / (all_scale / cross_scale)
        return output.transpose(2, 3)

    def forward(self, x, rel_pos, chunkwise_recurrent=False, incremental_state=None):
        bsz, tgt_len, _ = x.size()
        (sin, cos), inner_mask = rel_pos
        q, k, v, g = self.q_proj(x), self.k_proj(x), self.v_proj(x), self.g_proj(x)
        k = k * self.scaling
        q = q.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        k = k.view(bsz, tgt_len, self.num_heads, self.key_dim).transpose(1, 2)
        qr = theta_shift(q, sin, cos)               # q e^{i n theta}
        kr = theta_shift(k, sin, cos)               # k e^{i m theta}

        if incremental_state is not None:
            out = self.recurrent_forward(qr, kr, v, inner_mask, incremental_state)
        elif chunkwise_recurrent:
            out = self.chunk_recurrent_forward(qr, kr, v, inner_mask)
        else:
            out = self.parallel_forward(qr, kr, v, inner_mask)

        out = self.group_norm(out).reshape(bsz, tgt_len, self.head_dim * self.num_heads)
        return self.out_proj(self.gate_fn(g) * out)
```

So the chain, end to end. Attention trains in parallel because it forms the whole \(N\times N\) score matrix at once, but that same matrix makes autoregressive decoding \(O(n)\) per step with a key–value cache that grows without bound; an RNN decodes in \(O(1)\) but won't parallelize for training, and the existing efficient variants each give up either quality or content-awareness. Starting instead from a linear recurrence with a state matrix and unrolling it shows the recurrence *already is* a causal weighted sum over the past — the bridge to attention — so making the query/key projections content-aware and diagonalizing the state matrix turns its powers into a per-distance scalar decay \(\gamma^{n-m}\) times a rotary relative phase, which is exactly an xPos-style position factor falling out for free. That yields the retention operator with three equivalent faces: a parallel form \((QK^\top\odot D)V\) with the decay-and-causal matrix \(D\) for training; a recurrent form \(S_n=\gamma S_{n-1}+K_n^\top V_n,\;o_n=Q_n S_n\) for \(O(1)\) inference (equal because \(S_n\) unrolls to \(\sum_{m\le n}\gamma^{n-m}K_m^\top V_m\)); and a chunkwise form that runs the parallel face inside chunks and carries the state recurrently across them, with the key-to-boundary exponent \(B-1-j'\) and the boundary-to-query exponent \(j+1\) adding back to the true relative distance \(B+j-j'\). Using a different \(\gamma\) per head makes the memory multi-scale, per-head normalization balances the resulting variances and makes the numeric stabilizers harmless after normalization, and a swish gate restores the nonlinearity that deleting softmax removed.
