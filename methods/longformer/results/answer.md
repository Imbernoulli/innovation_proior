# Longformer: linear-time attention for long documents

## Problem

Transformer self-attention is O(n²) in time and memory because it scores every token against every other token. That caps practical input length (≈512 tokens for BERT/RoBERTa-style encoders) and forces long documents to be truncated, chunked, or fed through retrieve-then-read pipelines that lose cross-context information. Longformer replaces full attention with a pattern that scales **linearly** in sequence length while preserving long-range and task-level reach, and is a drop-in replacement for full attention in a pretrained model.

## Key idea

Sparsify the n×n attention matrix into a small, structured set of (token, token) pairs:

1. **Sliding window (local).** Each token attends to w/2 tokens on each side. Cost O(n·w), linear in n. Stacking ℓ layers grows the receptive field to ℓ·w (the CNN trick: depth recovers global reach), so top layers see the whole document even though no single attention op leaves its window.

2. **Dilated sliding window.** Insert gaps of size d into the window (like dilated CNNs). Same number of attended positions ⇒ same compute, but per-layer reach becomes (w/2)·d each side, so the receptive field grows to **ℓ·d·w** — tens of thousands of tokens. Multi-head: some heads dilate (long reach), some keep d=1 (sharp local context). Used for autoregressive character-LM; not used in the pretrain-finetune setting (it conflicts with pretrained weights).

3. **Global attention.** Local-only attention cannot aggregate the whole sequence into a decision token or let every document token see the question. So a few **pre-selected, task-specific** tokens get global attention — [CLS] for classification, all question tokens for QA. It is **symmetric**: a global token attends to all tokens and all tokens attend to it. With g (constant) global tokens the added cost is O(g·n), so the combined local+global attention is still **O(n)**.

4. **Separate projections for global attention.** Global attention serves a different role than local, so it uses its own Q_g, K_g, V_g (initialized as copies of the local Q, K, V) — critical for downstream quality.

5. **Banded matmul implementation.** The windowed/dilated pattern needs a matmul that produces only a band of diagonals — not a framework primitive. Three implementations: a `loop` reference (memory-tight, slow, supports dilation, test-only); `sliding_chunks` (overlapping 2w blocks, one dense matmul, fast, ~2× memory, no dilation — used for pretrain/finetune); a TVM-compiled CUDA kernel (memory-optimal, supports dilation + autoregressive — used for character-LM).

6. **Continue from a pretrained checkpoint.** Swap RoBERTa's dense self-attention for the sliding window (window=512 to match RoBERTa's compute), extend the learned position embeddings from 512 to 4096 by **copying** the 512-block repeatedly (preserves the learned local-position bias; random init breaks the model), and continue MLM for a few thousand updates. Character-LM uses **staged training**: start short, double window+sequence and halve LR each phase, with windows that increase with depth and dilation only on 2 upper-layer heads.

## Complexity / receptive field

- Window, one layer: O(n·w); ℓ layers: receptive field ≈ ℓ·w.
- Dilation d: receptive field ≈ ℓ·d·w, compute unchanged.
- Global on g constant tokens: O(g·n). Total: O(n·(w + g)) = O(n).

## Code

Grounded in the canonical implementation (overlapping-chunk banded attention + symmetric global attention with separate projections).

```python
import math
import torch
from torch import nn
import torch.nn.functional as F


def _chunk(x, w):
    """Split into overlapping blocks of size 2w (overlap w) via as_strided."""
    x = x.view(x.size(0), x.size(1) // (w * 2), w * 2, x.size(2))
    chunk_size = list(x.size());   chunk_size[1] = chunk_size[1] * 2 - 1
    chunk_stride = list(x.stride()); chunk_stride[1] = chunk_stride[1] // 2
    return x.as_strided(size=chunk_size, stride=chunk_stride)


def _skew(x, direction, padding_value):
    x = F.pad(x, direction, value=padding_value)
    return x.view(*x.size()[:-2], x.size(-1), x.size(-2))


def sliding_chunks_matmul_qk(q, k, w, padding_value):
    """Banded q @ k^T: returns (bsz, seqlen, num_heads, 2w+1), only the +/- w band."""
    bsz, seqlen, num_heads, head_dim = q.size()
    assert seqlen % (w * 2) == 0
    chunks_count = seqlen // w - 1
    q = q.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    k = k.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    chunk_q, chunk_k = _chunk(q, w), _chunk(k, w)
    chunk_attn = torch.einsum('bcxd,bcyd->bcxy', (chunk_q, chunk_k))      # one matmul per block
    diag = _skew(chunk_attn, (0, 0, 0, 1), padding_value)                 # diagonals -> columns
    out = diag.new_empty((bsz * num_heads, chunks_count + 1, w, w * 2 + 1))
    out[:, :-1, :, w:] = diag[:, :, :w, :w + 1]
    out[:, -1, :, w:]  = diag[:, -1, w:, :w + 1]
    out[:, 1:, :, :w]  = diag[:, :, -(w + 1):-1, w + 1:]
    out[:, 0, 1:w, 1:w] = diag[:, 0, :w - 1, 1 - w:]
    return out.view(bsz, num_heads, seqlen, 2 * w + 1).transpose(2, 1)


class LongformerSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.embed_dim = config.hidden_size
        # local windowed attention
        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key   = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)
        # separate projections for global tokens (init as copies of the local ones)
        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global   = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.dropout = config.attention_probs_dropout_prob
        self.attention_window = config.attention_window[layer_id]

    def forward(self, hidden_states, attention_mask=None):
        # attention_mask: -ve no attention | 0 local | +ve global token
        attention_mask = attention_mask.squeeze(2).squeeze(1)
        is_global = attention_mask > 0
        key_padding_mask = attention_mask < 0
        max_g = is_global.long().sum(dim=1).max()

        hidden_states = hidden_states.transpose(0, 1)
        seq_len, bsz, _ = hidden_states.size()
        q = self.query(hidden_states); k = self.key(hidden_states); v = self.value(hidden_states)
        q = q / math.sqrt(self.head_dim)                                  # 1/sqrt(d_k)
        q = q.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)

        # local banded scores
        attn_weights = sliding_chunks_matmul_qk(q, k, self.attention_window, padding_value=0)

        # global columns: all queries also score against the global tokens' keys
        if max_g > 0:
            sel_k = k.new_zeros(bsz, max_g, self.num_heads, self.head_dim)
            # ... gather global tokens' keys into sel_k ...
            global_cols = torch.einsum('blhd,bshd->blhs', (q, sel_k))
            attn_weights = torch.cat((global_cols, attn_weights), dim=-1)

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # fp32 for stability
        attn_probs = F.dropout(attn_weights.type_as(q), p=self.dropout, training=self.training)

        # aggregate: global columns (dense) + local band
        attn = sliding_chunks_matmul_pv(
            attn_probs.narrow(-1, max_g, attn_probs.size(-1) - max_g) if max_g > 0 else attn_probs,
            v, self.attention_window)
        # ... add the global-column value contribution to attn ...
        attn = attn.transpose(0, 1).reshape(seq_len, bsz, self.embed_dim)

        # global rows: recompute global tokens' output with the SEPARATE projections,
        # attending to the entire sequence (symmetric)
        if max_g > 0:
            sel_hidden = ...  # gather hidden states at global positions
            qg = self.query_global(sel_hidden) / math.sqrt(self.head_dim)
            kg = self.key_global(hidden_states)
            vg = self.value_global(hidden_states)
            gw = torch.bmm(qg, kg.transpose(1, 2))                        # O(g*n)
            gw = gw.masked_fill(key_padding_mask[:, None, :], -1e4)
            gp = F.dropout(F.softmax(gw, dim=-1, dtype=torch.float32).type_as(qg),
                           p=self.dropout, training=self.training)
            g_out = torch.bmm(gp, vg)
            # ... scatter g_out back into attn at the global positions ...
        return (attn.transpose(0, 1),)


# sliding_chunks_matmul_pv: matching value aggregation for the banded scores (omitted for brevity;
# pads V by w on both ends, strides into 3w chunks, skews probs, einsum 'bcwd,bcdh->bcwh').
```

```python
# Drop into a pretrained encoder; extend position embeddings by COPYING the 512-block.
class Longformer(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.encoder.layer):
            layer.attention.self = LongformerSelfAttention(config, layer_id=i)


def extend_position_embeddings(model, max_pos):                 # 512 -> 4096
    old = model.embeddings.position_embeddings.weight          # (512, d)
    new = old.new_empty(max_pos, old.size(1)); step = old.size(0)
    for s in range(0, max_pos, step):
        new[s:s + step] = old[:min(step, max_pos - s)]         # tile the learned block
    model.embeddings.position_embeddings = nn.Embedding.from_pretrained(new, freeze=False)
```

For seq2seq (LED), the same local+global attention replaces full self-attention in the **encoder** of an encoder-decoder model; the decoder keeps full attention over the encoded sequence and previously decoded tokens.
