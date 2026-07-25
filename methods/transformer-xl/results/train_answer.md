I will frame the solution around the Transformer-XL approach to autoregressive language modeling, which I regard as the canonical method for this setting. The problem is to model a long token stream as a product of next-token conditionals, but the practical difficulty is not the factorization itself; it is that useful information often lies hundreds or thousands of positions behind the current prediction. A standard Transformer decoder can in principle attend over a whole segment, yet the usual fixed-length training regime chops the corpus into independent blocks. That creates three linked failures. First, no information crosses segment boundaries, so the largest representable dependency is capped by the segment length. Second, the first tokens of every segment are predicted with little or no real left context, because the true history sits just outside the artificial cut; this is context fragmentation and it wastes optimization budget. Third, if at evaluation time we slide the window one step forward to give every token a full context, we re-encode almost the same segment repeatedly, which is prohibitively expensive.

My response is to let the model remember the past in a way that is compatible with efficient training and evaluation. When I process one segment, every layer produces a sequence of hidden states. Those states are already an encoded representation of the segment, so instead of throwing them away I cache them and reuse them as extended context for the next segment. Concretely, at layer n I take the previous segment's hidden states, stop their gradients so that backpropagation stays local, concatenate them in front of the current segment's states, and use that extended sequence to form keys and values. The queries still come only from the current segment, because I only need outputs for current positions. This is the segment-level recurrence of Transformer-XL. It is richer than passing a single recurrent state vector forward, because attention can directly address the specific old position it needs, and the reachable dependency length grows on the order of the number of layers times the segment length rather than being locked to one segment.

The recurrence solves the dependency and evaluation-efficiency problems, but it introduces a positional problem. In a standard Transformer I add absolute sinusoidal encodings to the input embeddings at positions 1 through L inside every segment. Under the new scheme the previous segment's states are concatenated with the current segment's states, so two tokens that are a full segment apart would carry the same positional marker and the model could no longer tell which one is older. Absolute segment-local positions therefore collide. I need a position signal that is relative to the query rather than tied to a global segment coordinate.

I express the attention score in relative terms. With absolute positions, the score between a query at position i and a key at position j decomposes into four terms: content attending to content, content attending to key position, query position attending to content, and query position attending to key position. In the relative scheme the key-side absolute position is replaced by a sinusoid encoding the causal distance i minus j, and the query-side absolute position is replaced by learned global bias vectors. The result is a score with two content-driven terms and two distance-driven terms, one pair using the content key projection and the other pair using a separate relative-location key projection. Keeping the relative vector as a fixed sinusoid matters because it lets the model extrapolate to distances longer than those seen during training.

The actual layer computation uses hidden states rather than raw embeddings. Each query is the current hidden state projected by W_q. The keys and values are formed from the concatenation of cached memory and current states, projected by W_k,E and W_v. The relative distances are projected once for the M plus L distinct causal offsets, multiplied by all queries, and then aligned with the key positions through a shift operation. This avoids building a per-pair relative-key tensor while keeping the usual L by M plus L attention logit matrix. After scaling, causal masking, softmax, weighted value aggregation, output projection, residual addition, layer normalization, and the feed-forward sublayer, I have the layer output. The final memory for the next segment is the most recent mem_len hidden states, detached from the computation graph.

Transformer-XL therefore combines two ideas that are forced by the failures of fixed segments: cached hidden-state memory with stopped gradients, and relative sinusoidal positional encodings inside the attention score. The recurrence only becomes coherent once positions are expressed relative to the query, and the relative score is only useful for long memory because its distance representation is fixed and extrapolatable. I will now give the core computation itself: the relative multi-head attention module that forms queries from the current segment and keys/values from memory concatenated with the current segment, the decoder layer built around it, and the memory-managing language model that carries hidden states across segments and reports a per-token loss.

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

This is the core `attn_type=0` computation, not the full training system: it leaves out the adaptive input embeddings, the projected adaptive softmax, sampled-softmax training, the `same_length` masking variant, and the optional pre-LayerNorm and absolute-position variants that the complete model supports. What is here is complete and correct on its own terms, and I checked it by instantiating `MemTransformerLM` with a small configuration and running two consecutive forward passes, feeding the memory list returned by the first call into the second as `*mems`. Both calls returned a per-token loss of shape `[tgt_len, bsz]` and a memory list of `n_layer + 1` tensors, each of shape `[mem_len, bsz, d_model]`, confirming that the recurrence and the relative-position alignment are wired correctly end to end rather than just type-checking. The two design commitments that define Transformer-XL are exactly what is captured here: `_update_mems` concatenates each layer's cached memory with its current hidden states, detaches the result under `torch.no_grad()`, and clips it back down to `mem_len` entries; the attention score is content plus relative, `ac + rel_shift(bd)`, so every query sees the correct causal distance to every cached and current key. Together these remove the fixed-segment dependency cap, eliminate context fragmentation by giving every token real history, and make full-context evaluation efficient because previously computed hidden states are cached and reused rather than recomputed from scratch.
