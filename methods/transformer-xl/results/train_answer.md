I will frame the solution around the Transformer-XL approach to autoregressive language modeling, which I regard as the canonical method for this setting. The problem is to model a long token stream as a product of next-token conditionals, but the practical difficulty is not the factorization itself; it is that useful information often lies hundreds or thousands of positions behind the current prediction. A standard Transformer decoder can in principle attend over a whole segment, yet the usual fixed-length training regime chops the corpus into independent blocks. That creates three linked failures. First, no information crosses segment boundaries, so the largest representable dependency is capped by the segment length. Second, the first tokens of every segment are predicted with little or no real left context, because the true history sits just outside the artificial cut; this is context fragmentation and it wastes optimization budget. Third, if at evaluation time we slide the window one step forward to give every token a full context, we re-encode almost the same segment repeatedly, which is prohibitively expensive.

My response is to let the model remember the past in a way that is compatible with efficient training and evaluation. When I process one segment, every layer produces a sequence of hidden states. Those states are already an encoded representation of the segment, so instead of throwing them away I cache them and reuse them as extended context for the next segment. Concretely, at layer n I take the previous segment's hidden states, stop their gradients so that backpropagation stays local, concatenate them in front of the current segment's states, and use that extended sequence to form keys and values. The queries still come only from the current segment, because I only need outputs for current positions. This is the segment-level recurrence of Transformer-XL. It is richer than passing a single recurrent state vector forward, because attention can directly address the specific old position it needs, and the reachable dependency length grows on the order of the number of layers times the segment length rather than being locked to one segment.

The recurrence solves the dependency and evaluation-efficiency problems, but it introduces a positional problem. In a standard Transformer I add absolute sinusoidal encodings to the input embeddings at positions 1 through L inside every segment. Under the new scheme the previous segment's states are concatenated with the current segment's states, so two tokens that are a full segment apart would carry the same positional marker and the model could no longer tell which one is older. Absolute segment-local positions therefore collide. I need a position signal that is relative to the query rather than tied to a global segment coordinate.

I express the attention score in relative terms. With absolute positions, the score between a query at position i and a key at position j decomposes into four terms: content attending to content, content attending to key position, query position attending to content, and query position attending to key position. In the relative scheme the key-side absolute position is replaced by a sinusoid encoding the causal distance i minus j, and the query-side absolute position is replaced by learned global bias vectors. The result is a score with two content-driven terms and two distance-driven terms, one pair using the content key projection and the other pair using a separate relative-location key projection. Keeping the relative vector as a fixed sinusoid matters because it lets the model extrapolate to distances longer than those seen during training.

The actual layer computation uses hidden states rather than raw embeddings. Each query is the current hidden state projected by W_q. The keys and values are formed from the concatenation of cached memory and current states, projected by W_k,E and W_v. The relative distances are projected once for the M plus L distinct causal offsets, multiplied by all queries, and then aligned with the key positions through a shift operation. This avoids building a per-pair relative-key tensor while keeping the usual L by M plus L attention logit matrix. After scaling, causal masking, softmax, weighted value aggregation, output projection, residual addition, layer normalization, and the feed-forward sublayer, I have the layer output. The final memory for the next segment is the most recent mem_len hidden states, detached from the computation graph.

Transformer-XL therefore combines two ideas that are forced by the failures of fixed segments: cached hidden-state memory with stopped gradients, and relative sinusoidal positional encodings inside the attention score. The recurrence only becomes coherent once positions are expressed relative to the query, and the relative score is only useful for long memory because its distance representation is fixed and extrapolatable. I will now give a compact, runnable illustration that captures the core mechanism: segment-level memory reuse and the relative positional score with the alignment shift.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativePositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0.0, d_model, 2.0) / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, length):
        pos_seq = torch.arange(length - 1, -1, -1.0, device=self.inv_freq.device)
        sinusoid = torch.ger(pos_seq, self.inv_freq)
        return torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)[:, None, :]


def rel_shift(x):
    zero_pad = torch.zeros((x.size(0), 1, *x.size()[2:]), device=x.device, dtype=x.dtype)
    x_padded = torch.cat([zero_pad, x], dim=1)
    x_padded = x_padded.view(x.size(1) + 1, x.size(0), *x.size()[2:])
    return x_padded[1:].view_as(x)


class TransformerXLBlock(nn.Module):
    def __init__(self, d_model, n_head, d_head, dropout=0.0):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_head
        self.qkv = nn.Linear(d_model, 3 * n_head * d_head, bias=False)
        self.r_proj = nn.Linear(d_model, n_head * d_head, bias=False)
        self.out = nn.Linear(n_head * d_head, d_model, bias=False)
        self.scale = 1.0 / (d_head ** 0.5)
        self.r_w_bias = nn.Parameter(torch.zeros(n_head, d_head))
        self.r_r_bias = nn.Parameter(torch.zeros(n_head, d_head))
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.ReLU(inplace=True),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, pos_emb, mask, mem=None):
        qlen, bsz, _ = x.shape
        cat = x if mem is None or mem.numel() == 0 else torch.cat([mem, x], dim=0)
        klen = cat.size(0)

        w_heads = self.qkv(cat)
        w_q, w_k, w_v = torch.chunk(w_heads, 3, dim=-1)
        w_q = w_q[-qlen:].view(qlen, bsz, self.n_head, self.d_head)
        w_k = w_k.view(klen, bsz, self.n_head, self.d_head)
        w_v = w_v.view(klen, bsz, self.n_head, self.d_head)

        r_head_k = self.r_proj(pos_emb).view(pos_emb.size(0), self.n_head, self.d_head)

        ac = torch.einsum("ibnd,jbnd->ijbn", (w_q + self.r_w_bias, w_k))
        bd = torch.einsum("ibnd,jnd->ijbn", (w_q + self.r_r_bias, r_head_k))
        bd = rel_shift(bd)

        attn_score = (ac + bd) * self.scale
        if mask is not None:
            attn_score = attn_score.masked_fill(mask, float("-inf"))
        attn_prob = F.softmax(attn_score, dim=1)
        attn_vec = torch.einsum("ijbn,jbnd->ibnd", (attn_prob, w_v))
        attn_vec = attn_vec.contiguous().view(qlen, bsz, -1)
        attn_out = self.dropout(self.out(attn_vec))
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.ff(x))
        return x


def demo():
    d_model, n_head, d_head = 64, 4, 16
    seg_len, mem_len, batch = 8, 8, 2
    block = TransformerXLBlock(d_model, n_head, d_head)
    pos_enc = RelativePositionalEncoding(d_model)

    x = torch.randn(seg_len, batch, d_model)
    mem = torch.randn(mem_len, batch, d_model)
    klen = mem_len + seg_len
    pos_emb = pos_enc(klen)
    mask = torch.triu(torch.ones(seg_len, klen), diagonal=1 + mem_len).bool()
    mask = mask[:, :, None, None].expand(seg_len, klen, batch, n_head)

    out = block(x, pos_emb, mask, mem=mem)
    print("output shape:", out.shape)
    new_mem = torch.cat([mem, out], dim=0).detach()[-mem_len:]
    print("updated memory shape:", new_mem.shape)


if __name__ == "__main__":
    demo()
```

The snippet above is not a full training system, but it does run and it isolates exactly the two design commitments that define Transformer-XL: memory is concatenated with the current segment and then detached for the next step, and the attention score is built from content keys plus relative positional keys shifted so that each query sees the correct causal distance. Those two pieces together remove the fixed-segment dependency cap, reduce context fragmentation by giving every token a real history, and make full-context evaluation efficient because previously computed hidden states are reused rather than recomputed from scratch.
