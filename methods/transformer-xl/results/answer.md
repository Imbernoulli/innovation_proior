# Transformer-XL

## Problem

Autoregressive language modeling factorizes $P(\mathbf{x})=\prod_t P(x_t\mid x_{<t})$,
and the quality of each conditional is bounded by how far back the model's
context encoding actually reaches. Self-attention removes the vanishing-gradient
ceiling that limits recurrent models (any two positions are one hop apart), but
the standard practice of training on independent fixed-length segments throws
that advantage away: the reachable dependency is capped at the segment length,
the leading tokens of each segment are predicted with little or no context
(*context fragmentation*), and full-context evaluation requires re-encoding a
sliding window from scratch at every step. Transformer-XL keeps the attention
architecture but lets context flow across segment boundaries, lifting the
effective dependency length far past one segment while making evaluation reuse
computation.

## Key idea

**Segment-level recurrence with state reuse.** Cache the previous segment's
hidden states and let the current segment attend into them. For consecutive
segments of length $L$, at layer $n$ with cached memory $\mathbf{m}^{n-1}$:
$$
\widetilde{\mathbf{h}}^{n-1}=\big[\mathrm{SG}(\mathbf{m}^{n-1})\circ\mathbf{h}^{n-1}\big],\quad
\mathbf{q}^n=\mathbf{h}^{n-1}{\mathbf{W}_q^n}^\top,\quad
\mathbf{k}^n,\mathbf{v}^n=\widetilde{\mathbf{h}}^{n-1}{\mathbf{W}_{k,E}^n}^\top,\;\widetilde{\mathbf{h}}^{n-1}{\mathbf{W}_v^n}^\top.
$$
The query comes from the current segment only; keys/values come from the
extended context. The stop-gradient $\mathrm{SG}$ keeps backprop inside the
current segment (a sequence-valued analogue of truncated BPTT). The recurrence
shifts one layer down per segment, so the largest reachable dependency grows as
$O(N\times L)$ for $N$ layers. Memory can span $M$ cached states across several
previous segments; $M$ can be enlarged at evaluation.

**Relative positional encoding.** Under state reuse, absolute positions repeat
every segment, so cached and current tokens at the same offset become
indistinguishable. The fix encodes only the *relative* distance $i-j$ and injects
it into the per-layer attention score. Decompose the standard absolute score
$q_i^\top k_j$ (with $q$ from $\mathbf{E}_{x_i}+\mathbf{U}_i$, $k$ from
$\mathbf{E}_{x_j}+\mathbf{U}_j$) into four terms and re-parameterize:
$$
A^{\text{rel}}_{i,j}=
\underbrace{\mathbf{E}_{x_i}^\top\mathbf{W}_q^\top\mathbf{W}_{k,E}\,\mathbf{E}_{x_j}}_{(a)\ \text{content addressing}}
+\underbrace{\mathbf{E}_{x_i}^\top\mathbf{W}_q^\top\mathbf{W}_{k,R}\,\mathbf{R}_{i-j}}_{(b)\ \text{content-dependent positional bias}}
+\underbrace{u^\top\mathbf{W}_{k,E}\,\mathbf{E}_{x_j}}_{(c)\ \text{global content bias}}
+\underbrace{v^\top\mathbf{W}_{k,R}\,\mathbf{R}_{i-j}}_{(d)\ \text{global positional bias}}.
$$
Three changes: (1) the key's absolute position $\mathbf{U}_j$ becomes the
relative sinusoid $\mathbf{R}_{i-j}$ (parameter-free, so it generalizes to
distances unseen in training); (2) the query's absolute position
$\mathbf{U}_i^\top\mathbf{W}_q^\top$, identical for every query, becomes two
learnable global vectors $u$ (against content keys) and $v$ (against location
keys); (3) the key projection splits into $\mathbf{W}_{k,E}$ (content) and
$\mathbf{W}_{k,R}$ (location). No positional vector is added at the input.

**Full per-layer computation** ($n=1,\dots,N$, $\mathbf{h}^0=\mathbf{E}_\mathbf{s}$):
$$
A^n_{i,j}=(\mathbf{q}^n_i+u)^\top\mathbf{k}^n_j+(\mathbf{q}^n_i+v)^\top(\mathbf{W}_{k,R}^n\mathbf{R}_{i-j}),
$$
$$
\mathbf{a}^n=\text{Masked-Softmax}(A^n/\sqrt{d})\,\mathbf{v}^n,\quad
\mathbf{o}^n=\text{LayerNorm}(\text{Linear}(\mathbf{a}^n)+\mathbf{h}^{n-1}),\quad
\mathbf{h}^n=\text{FFN}(\mathbf{o}^n).
$$

**Linear-time position term.** Only $M+L$ distinct distances occur. Stacking the
location keys $\mathbf{Q}_k=\mathbf{W}_{k,R}\mathbf{R}_{M+L-1-k}$ and computing the
dense product $\widetilde{\mathbf{B}}=\mathbf{q}\mathbf{Q}^\top$, each row of the
required term-$(b)$ matrix is a left-shift of a row of $\widetilde{\mathbf{B}}$;
one matmul plus a shift recovers it in linear time (same for term $(d)$ via
$(\mathbf{Q}v)^\top$).

## Code

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    """Relative sinusoid R_{i-j}; parameter-free so it extrapolates to longer memory."""
    def __init__(self, demb):
        super().__init__()
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq):
        sinusoid = torch.ger(pos_seq, self.inv_freq)
        return torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)[:, None, :]


class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0):
        super().__init__()
        self.n_head, self.d_head = n_head, d_head
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)  # W_q, W_{k,E}, W_v
        self.r_net = nn.Linear(d_model, n_head * d_head, bias=False)        # W_{k,R}
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.drop, self.dropatt = nn.Dropout(dropout), nn.Dropout(dropatt)
        self.scale = 1 / (d_head ** 0.5)

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        return x_padded[1:].view_as(x)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        # r_w_bias = u, r_r_bias = v
        qlen, bsz = w.size(0), w.size(1)
        cat = torch.cat([mems, w], 0) if mems is not None else w
        w_heads = self.qkv_net(cat)
        r_head_k = self.r_net(r)
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        w_head_q = w_head_q[-qlen:]                      # query: current segment only
        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)
        r_head_k = r_head_k.view(klen, self.n_head, self.d_head)

        AC = torch.einsum('ibnd,jbnd->ijbn', (w_head_q + r_w_bias, w_head_k))   # (a)+(c)
        BD = torch.einsum('ibnd,jnd->ijbn', (w_head_q + r_r_bias, r_head_k))    # (b)+(d)
        BD = self._rel_shift(BD)

        attn_score = (AC + BD).mul_(self.scale)
        if attn_mask is not None:
            attn_score = attn_score.float().masked_fill(
                attn_mask[None, :, :, None], -float('inf')).type_as(attn_score)
        attn_prob = self.dropatt(F.softmax(attn_score, dim=1))

        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))
        attn_vec = attn_vec.contiguous().view(qlen, bsz, self.n_head * self.d_head)
        attn_out = self.drop(self.o_net(attn_vec))
        return self.layer_norm(w + attn_out)


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True), nn.Dropout(dropout),
            nn.Linear(d_inner, d_model), nn.Dropout(dropout))
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.layer_norm(x + self.net(x))


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, dropatt=0):
        super().__init__()
        self.dec_attn = RelPartialLearnableMultiHeadAttn(n_head, d_model, d_head, dropout, dropatt)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout)

    def forward(self, dec_inp, r, r_w_bias, r_r_bias, dec_attn_mask=None, mems=None):
        out = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias, attn_mask=dec_attn_mask, mems=mems)
        return self.pos_ff(out)


class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tgt_len, mem_len, clamp_len=-1):
        super().__init__()
        self.n_layer, self.d_model, self.mem_len, self.clamp_len = n_layer, d_model, mem_len, clamp_len
        self.word_emb = nn.Embedding(n_token, d_model)
        self.pos_emb = PositionalEmbedding(d_model)
        self.drop = nn.Dropout(dropout)
        self.r_w_bias = nn.Parameter(torch.Tensor(n_head, d_head))   # u
        self.r_r_bias = nn.Parameter(torch.Tensor(n_head, d_head))   # v
        self.layers = nn.ModuleList([
            RelPartialLearnableDecoderLayer(n_head, d_model, d_head, d_inner, dropout, dropatt)
            for _ in range(n_layer)])
        self.out_layer = nn.Linear(d_model, n_token)

    def init_mems(self):
        if self.mem_len <= 0:
            return None
        p = next(self.parameters())
        return [torch.empty(0, dtype=p.dtype, device=p.device) for _ in range(self.n_layer + 1)]

    def _update_mems(self, hids, mems, qlen, mlen):
        if mems is None:
            return None
        with torch.no_grad():
            end = mlen + qlen
            beg = max(0, end - self.mem_len)
            return [torch.cat([m, h], 0)[beg:end].detach() for m, h in zip(mems, hids)]

    def _forward(self, dec_inp, mems):
        qlen, bsz = dec_inp.size()
        word_emb = self.word_emb(dec_inp)
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        dec_attn_mask = torch.triu(
            word_emb.new_ones(qlen, klen), diagonal=1 + mlen).bool()[:, :, None]

        pos_seq = torch.arange(klen - 1, -1, -1.0, device=word_emb.device, dtype=word_emb.dtype)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.drop(self.pos_emb(pos_seq))
        core_out = self.drop(word_emb)

        hids = [core_out]
        for i, layer in enumerate(self.layers):
            mems_i = None if mems is None else mems[i]
            core_out = layer(core_out, pos_emb, self.r_w_bias, self.r_r_bias,
                             dec_attn_mask=dec_attn_mask, mems=mems_i)
            hids.append(core_out)
        new_mems = self._update_mems(hids, mems, qlen, mlen)
        return self.drop(core_out), new_mems

    def forward(self, data, target, *mems):
        if not mems:
            mems = self.init_mems()
        hidden, new_mems = self._forward(data, mems=mems)
        logit = self.out_layer(hidden)
        loss = F.cross_entropy(logit.view(-1, logit.size(-1)), target.view(-1), reduction='none')
        return [loss] + (new_mems if new_mems is not None else [])
```

The official implementation additionally uses adaptive input embeddings and an
adaptive softmax for large vocabularies, multi-head projections, and an optional
pre-LayerNorm variant; the structure above is the core.
