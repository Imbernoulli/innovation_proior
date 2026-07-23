# Transformer-XL

## Problem

Autoregressive language modeling needs each prediction $P(x_t\mid x_{<t})$ to
use distant context. A fixed-segment Transformer decoder has short attention
paths inside a segment, but no information crosses segment boundaries. That
caps reachable dependency length at the segment length, starves the first
tokens of each segment of real history, and makes full-context evaluation
recompute nearly identical sliding windows.

## Method

**Segment-level recurrence.** Cache previous hidden states and reuse them as
memory. For current segment states $\mathbf h^{n-1}$ and memory
$\mathbf m^{n-1}$ at layer $n$,
$$
\widetilde{\mathbf h}^{n-1}=[\,\mathrm{SG}(\mathbf m^{n-1})\circ \mathbf h^{n-1}\,],
$$
$$
\mathbf q^n=\mathbf h^{n-1}{\mathbf W_q^n}^{\top},\qquad
\mathbf k^n=\widetilde{\mathbf h}^{n-1}{\mathbf W_{k,E}^n}^{\top},\qquad
\mathbf v^n=\widetilde{\mathbf h}^{n-1}{\mathbf W_v^n}^{\top}.
$$
Queries come only from the current segment; keys and values come from memory
plus current segment. $\mathrm{SG}$ keeps gradients inside the current segment.
Because the recurrence goes one layer down per previous segment, the reachable
dependency length grows as $O(NL)$ for $N$ layers and segment-scale memory.

**Relative positional score.** Absolute position ids repeat in every segment, so
cached and current positions collide. Put a relative distance in the attention
score instead. For a current query row $i$ and an extended-context key column
$j$, the causal distance is $\delta=M+i-j$ when the memory length is $M$; in the
paper's common-time-axis notation this is written as $i-j$. Future positions are
masked, so $\delta\ge 0$ for visible keys.

The absolute score expansion
$$
\mathbf E_{x_i}^{\top}\mathbf W_q^{\top}\mathbf W_k\mathbf E_{x_j}
+\mathbf E_{x_i}^{\top}\mathbf W_q^{\top}\mathbf W_k\mathbf U_j
+\mathbf U_i^{\top}\mathbf W_q^{\top}\mathbf W_k\mathbf E_{x_j}
+\mathbf U_i^{\top}\mathbf W_q^{\top}\mathbf W_k\mathbf U_j
$$
is reparameterized as
$$
A^n_{i,j}=(\mathbf q^n_i+u)^\top\mathbf k^n_j
+(\mathbf q^n_i+v)^\top(\mathbf W_{k,R}^n\mathbf R_{\delta}),
$$
where $\mathbf R_\delta$ is a fixed sinusoid, $u$ is the global content-key
bias, and $v$ is the global location-key bias. The fixed sinusoid is kept
separate from the learned projection so distances longer than those seen during
training still have structured encodings.

The relative term is computed by projecting the $M+L$ distinct relative
sinusoids once, multiplying queries by that stack, and applying the `_rel_shift`
alignment. This avoids a per-pair relative-key projection and the quadratic
relative-key tensor; the usual $L(M+L)$ attention logits still exist.

## Code

This is the `attn_type=0` core computation. The full implementation additionally
uses adaptive input embeddings, projected adaptive softmax, sampled softmax,
`same_length`, training-script initialization, and optional pre-LayerNorm and
absolute-position variants.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEmbedding(nn.Module):
    """Fixed sinusoid for relative distances."""
    def __init__(self, demb):
        super().__init__()
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq):
        sinusoid = torch.ger(pos_seq, self.inv_freq)
        return torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)[:, None, :]


class RelPartialLearnableMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout, dropatt=0.0):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.qkv_net = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        self.r_net = nn.Linear(d_model, n_head * d_head, bias=False)
        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.dropatt = nn.Dropout(dropatt)
        self.scale = 1 / (d_head ** 0.5)

    def _rel_shift(self, x):
        zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]),
                               device=x.device, dtype=x.dtype)
        x_padded = torch.cat([zero_pad, x], dim=1)
        x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
        return x_padded[1:].view_as(x)

    def forward(self, w, r, r_w_bias, r_r_bias, attn_mask=None, mems=None):
        qlen, bsz = w.size(0), w.size(1)
        cat = w if mems is None or mems.numel() == 0 else torch.cat([mems, w], 0)

        w_heads = self.qkv_net(cat)
        r_head_k = self.r_net(r)
        w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
        w_head_q = w_head_q[-qlen:]
        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)
        r_head_k = r_head_k.view(r.size(0), self.n_head, self.d_head)

        ac = torch.einsum("ibnd,jbnd->ijbn", (w_head_q + r_w_bias, w_head_k))
        bd = torch.einsum("ibnd,jnd->ijbn", (w_head_q + r_r_bias, r_head_k))
        bd = self._rel_shift(bd)

        attn_score = (ac + bd).mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                mask = attn_mask[None, :, :, None]
            elif attn_mask.dim() == 3:
                mask = attn_mask[:, :, :, None]
            else:
                raise ValueError("attention mask must be 2D or 3D")
            attn_score = attn_score.float().masked_fill(
                mask, -float("inf")).type_as(attn_score)

        attn_prob = self.dropatt(F.softmax(attn_score, dim=1))
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, w_head_v))
        attn_vec = attn_vec.contiguous().view(
            qlen, bsz, self.n_head * self.d_head)
        attn_out = self.drop(self.o_net(attn_vec))
        return self.layer_norm(w + attn_out)


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_inner, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.layer_norm(x + self.net(x))


class RelPartialLearnableDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropout, dropatt=0.0):
        super().__init__()
        self.dec_attn = RelPartialLearnableMultiHeadAttn(
            n_head, d_model, d_head, dropout, dropatt)
        self.pos_ff = PositionwiseFF(d_model, d_inner, dropout)

    def forward(self, dec_inp, r, r_w_bias, r_r_bias,
                dec_attn_mask=None, mems=None):
        out = self.dec_attn(dec_inp, r, r_w_bias, r_r_bias,
                            attn_mask=dec_attn_mask, mems=mems)
        return self.pos_ff(out)


class MemTransformerLM(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropout, dropatt, tgt_len, mem_len, ext_len=0, clamp_len=-1):
        super().__init__()
        self.n_layer = n_layer
        self.d_model = d_model
        self.mem_len = mem_len
        self.ext_len = ext_len
        self.clamp_len = clamp_len
        self.emb_scale = d_model ** 0.5

        self.word_emb = nn.Embedding(n_token, d_model)
        self.pos_emb = PositionalEmbedding(d_model)
        self.drop = nn.Dropout(dropout)
        self.r_w_bias = nn.Parameter(torch.Tensor(n_head, d_head))
        self.r_r_bias = nn.Parameter(torch.Tensor(n_head, d_head))
        self.layers = nn.ModuleList([
            RelPartialLearnableDecoderLayer(
                n_head, d_model, d_head, d_inner, dropout, dropatt)
            for _ in range(n_layer)
        ])
        self.out_layer = nn.Linear(d_model, n_token)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.r_w_bias, mean=0.0, std=0.02)
        nn.init.normal_(self.r_r_bias, mean=0.0, std=0.02)

    def init_mems(self):
        if self.mem_len <= 0:
            return None
        p = next(self.parameters())
        return [torch.empty(0, dtype=p.dtype, device=p.device)
                for _ in range(self.n_layer + 1)]

    def _update_mems(self, hids, mems, qlen, mlen):
        if mems is None:
            return None
        with torch.no_grad():
            end = mlen + max(0, qlen - self.ext_len)
            beg = max(0, end - self.mem_len)
            new_mems = []
            for mem, hid in zip(mems, hids):
                cat = hid if mem.numel() == 0 else torch.cat([mem, hid], dim=0)
                new_mems.append(cat[beg:end].detach())
            return new_mems

    def _forward(self, dec_inp, mems):
        qlen, bsz = dec_inp.size()
        word_emb = self.word_emb(dec_inp) * self.emb_scale
        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen

        dec_attn_mask = torch.triu(
            word_emb.new_ones(qlen, klen), diagonal=1 + mlen
        ).bool()[:, :, None]

        pos_seq = torch.arange(klen - 1, -1, -1.0,
                               device=word_emb.device, dtype=word_emb.dtype)
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

        core_out = self.drop(core_out)
        new_mems = self._update_mems(hids, mems, qlen, mlen)
        return core_out, new_mems

    def forward(self, data, target, *mems):
        if not mems:
            mems = self.init_mems()
        hidden, new_mems = self._forward(data, mems=mems)
        tgt_len = target.size(0)
        pred_hid = hidden[-tgt_len:]
        logit = self.out_layer(pred_hid)
        loss = F.cross_entropy(
            logit.reshape(-1, logit.size(-1)),
            target.reshape(-1),
            reduction="none",
        ).view_as(target)
        return [loss] + (new_mems if new_mems is not None else [])
```
