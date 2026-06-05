# Longformer: linear-time attention for long documents

## Problem

Transformer self-attention is O(n²) in time and memory because it scores every token against every other token. That caps practical input length (about 512 tokens for BERT/RoBERTa-style encoders) and forces long documents to be truncated, chunked, or fed through retrieve-then-read pipelines that lose cross-context information. Longformer replaces full attention with a sparse pattern that scales linearly in sequence length while preserving long-range and task-level reach.

## Key idea

Sparsify the n×n attention matrix into a small, structured set of token-token pairs. Let W be the total logical local window; the implementation stores `one_sided_window = W/2`.

1. **Sliding window.** Each token attends to W/2 tokens on each side. Cost O(n·W), linear in n. Stacking ℓ layers grows the receptive field to ℓ·W, so top layers can integrate a whole document even though a single attention operation stays local.

2. **Dilated sliding window.** Insert gaps of size d into the window. The number of attended positions is unchanged, but per-layer reach becomes (W/2)·d each side, so the receptive field grows to ℓ·d·W. Some heads can dilate for long reach while undilated heads preserve sharp local context.

3. **Global attention.** A few pre-selected, task-specific tokens get global attention: [CLS] for classification, question tokens for QA. It is symmetric: a global token attends to all tokens, and all tokens attend to it. With g global tokens the added cost is O(g·n), so local plus global attention remains O(n).

4. **Separate global projections.** Global attention uses its own Q_g, K_g, V_g, initialized from the local Q, K, V, so the global role can specialize without forcing local scores to serve two jobs.

5. **Banded implementation.** The local pattern needs a matmul that returns only a band of diagonals. The practical implementation uses overlapping 2r chunks for the non-dilated pretrain/finetune case (`r = one_sided_window`), plus a TVM CUDA kernel for the dilated/autoregressive character-LM case.

6. **Pretrained initialization.** Swap RoBERTa's dense self-attention for an effective W=512 local window (`one_sided_window=256`), extend learned position embeddings by copying the learned block repeatedly, then continue MLM. Character-LM uses staged training: start short, double window and sequence length each phase, and halve the learning rate.

## Complexity

- Local window: O(n·W), receptive field ≈ ℓ·W after ℓ layers.
- Dilated local window: same compute, receptive field ≈ ℓ·d·W.
- Global tokens: O(g·n), with g independent of n.
- Total: O(n·(W + g)) = O(n).

## Code

```python
import math
import torch
from torch import nn
import torch.nn.functional as F
from transformers.modeling_roberta import RobertaModel
from longformer.diagonaled_mm_tvm import mask_invalid_locations


def _skew(x, direction, padding_value):
    x = F.pad(x, direction, value=padding_value)
    return x.view(*x.size()[:-2], x.size(-1), x.size(-2))


def _skew2(x, padding_value):
    bsz_heads, chunks, one_sided_window, width = x.size()
    x = F.pad(x, (0, one_sided_window + 1), value=padding_value)
    x = x.view(bsz_heads, chunks, -1)
    x = x[:, :, :-one_sided_window]
    x = x.view(bsz_heads, chunks, one_sided_window, one_sided_window + width)
    return x[:, :, :, :-1]


def _chunk(x, one_sided_window):
    x = x.view(x.size(0), x.size(1) // (one_sided_window * 2), one_sided_window * 2, x.size(2))
    chunk_size = list(x.size())
    chunk_size[1] = chunk_size[1] * 2 - 1
    chunk_stride = list(x.stride())
    chunk_stride[1] = chunk_stride[1] // 2
    return x.as_strided(size=chunk_size, stride=chunk_stride)


def banded_query_key(q, k, one_sided_window, padding_value):
    bsz, seqlen, num_heads, head_dim = q.size()
    assert seqlen % (one_sided_window * 2) == 0
    assert q.size() == k.size()
    chunks_count = seqlen // one_sided_window - 1

    q = q.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    k = k.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    chunk_q = _chunk(q, one_sided_window)
    chunk_k = _chunk(k, one_sided_window)
    chunk_attn = torch.einsum("bcxd,bcyd->bcxy", (chunk_q, chunk_k))
    diagonal_chunk_attn = _skew(chunk_attn, (0, 0, 0, 1), padding_value)

    band = diagonal_chunk_attn.new_empty(
        (bsz * num_heads, chunks_count + 1, one_sided_window, one_sided_window * 2 + 1)
    )
    band[:, :-1, :, one_sided_window:] = diagonal_chunk_attn[:, :, :one_sided_window, :one_sided_window + 1]
    band[:, -1, :, one_sided_window:] = diagonal_chunk_attn[:, -1, one_sided_window:, :one_sided_window + 1]
    band[:, 1:, :, :one_sided_window] = diagonal_chunk_attn[:, :, -(one_sided_window + 1):-1, one_sided_window + 1:]
    band[:, 0, 1:one_sided_window, 1:one_sided_window] = diagonal_chunk_attn[
        :, 0, :one_sided_window - 1, 1 - one_sided_window:
    ]
    band = band.view(bsz, num_heads, seqlen, 2 * one_sided_window + 1).transpose(2, 1)
    mask_invalid_locations(band, one_sided_window, 1, False)
    return band


def banded_probability_value(prob, v, one_sided_window):
    bsz, seqlen, num_heads, head_dim = v.size()
    assert seqlen % (one_sided_window * 2) == 0
    assert prob.size()[:3] == v.size()[:3]
    assert prob.size(3) == 2 * one_sided_window + 1
    chunks_count = seqlen // one_sided_window - 1

    chunk_prob = prob.transpose(1, 2).reshape(
        bsz * num_heads, seqlen // one_sided_window, one_sided_window, 2 * one_sided_window + 1
    )
    v = v.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    padded_v = F.pad(v, (0, 0, one_sided_window, one_sided_window), value=-1)
    chunk_v = padded_v.as_strided(
        size=(bsz * num_heads, chunks_count + 1, 3 * one_sided_window, head_dim),
        stride=(
            padded_v.stride()[0],
            one_sided_window * padded_v.stride()[1],
            padded_v.stride()[1],
            padded_v.stride()[2],
        ),
    )
    skewed_prob = _skew2(chunk_prob, padding_value=0)
    context = torch.einsum("bcwd,bcdh->bcwh", (skewed_prob, chunk_v))
    return context.view(bsz, num_heads, seqlen, head_dim).transpose(1, 2)


class LongSequenceSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be a multiple of num_attention_heads")
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.embed_dim = config.hidden_size
        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key   = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)
        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global   = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.dropout = config.attention_probs_dropout_prob
        self.layer_id = layer_id
        self.one_sided_window = config.attention_window[layer_id]
        dilation = getattr(config, "attention_dilation", None)
        self.attention_dilation = dilation[layer_id] if dilation is not None else 1
        self.attention_mode = getattr(config, "attention_mode", "sliding_chunks")
        assert self.one_sided_window > 0
        assert self.attention_mode == "sliding_chunks"
        assert self.attention_dilation == 1

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=False,
    ):
        assert encoder_hidden_states is None
        assert encoder_attention_mask is None
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
            key_padding_mask = attention_mask < 0
            extra_attention_mask = attention_mask > 0
            remove_from_windowed_attention_mask = attention_mask != 0

            num_extra_indices_per_batch = extra_attention_mask.long().sum(dim=1)
            max_num_extra_indices_per_batch = int(num_extra_indices_per_batch.max().item())
            if max_num_extra_indices_per_batch <= 0:
                extra_attention_mask = None
            else:
                extra_attention_mask_nonzeros = extra_attention_mask.nonzero(as_tuple=True)
                zero_to_max = torch.arange(
                    0, max_num_extra_indices_per_batch, device=num_extra_indices_per_batch.device
                )
                selection_padding_mask = zero_to_max < num_extra_indices_per_batch.unsqueeze(dim=-1)
                selection_padding_mask_nonzeros = selection_padding_mask.nonzero(as_tuple=True)
                selection_padding_mask_zeros = (selection_padding_mask == 0).nonzero(as_tuple=True)
        else:
            extra_attention_mask = None
            key_padding_mask = None
            remove_from_windowed_attention_mask = None

        hidden_states = hidden_states.transpose(0, 1)
        seq_len, bsz, embed_dim = hidden_states.size()
        q = self.query(hidden_states); k = self.key(hidden_states); v = self.value(hidden_states)
        q /= math.sqrt(self.head_dim)
        q = q.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = banded_query_key(q, k, self.one_sided_window, padding_value=0)
        mask_invalid_locations(attn_weights, self.one_sided_window, self.attention_dilation, False)
        if remove_from_windowed_attention_mask is not None:
            remove = remove_from_windowed_attention_mask.unsqueeze(dim=-1).unsqueeze(dim=-1)
            float_mask = remove.type_as(q).masked_fill(remove, -10000.0)
            ones = float_mask.new_ones(size=float_mask.size())
            diagonal_mask = banded_query_key(ones, float_mask, self.one_sided_window, padding_value=0)
            attn_weights += diagonal_mask

        if extra_attention_mask is not None:
            selected_k = k.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_k[selection_padding_mask_nonzeros] = k[extra_attention_mask_nonzeros]
            selected_attn_weights = torch.einsum('blhd,bshd->blhs', (q, selected_k))
            if selection_padding_mask_zeros[0].numel() > 0:
                selected_attn_weights[
                    selection_padding_mask_zeros[0], :, :, selection_padding_mask_zeros[1]
                ] = -10000
            attn_weights = torch.cat((selected_attn_weights, attn_weights), dim=-1)

        attn_weights_float = F.softmax(attn_weights, dim=-1, dtype=torch.float32)
        if key_padding_mask is not None:
            attn_weights_float = torch.masked_fill(
                attn_weights_float, key_padding_mask.unsqueeze(-1).unsqueeze(-1), 0.0
            )
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training)
        v = v.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)

        attn = 0
        if extra_attention_mask is not None:
            selected_probs = attn_probs.narrow(-1, 0, max_num_extra_indices_per_batch)
            selected_v = v.new_zeros(bsz, max_num_extra_indices_per_batch, self.num_heads, self.head_dim)
            selected_v[selection_padding_mask_nonzeros] = v[extra_attention_mask_nonzeros]
            attn = torch.matmul(selected_probs.transpose(1, 2),
                                selected_v.transpose(1, 2).type_as(selected_probs)).transpose(1, 2)
            attn_probs = attn_probs.narrow(
                -1, max_num_extra_indices_per_batch, attn_probs.size(-1) - max_num_extra_indices_per_batch
            ).contiguous()
        attn += banded_probability_value(attn_probs, v, self.one_sided_window)
        attn = attn.type_as(hidden_states)
        attn = attn.transpose(0, 1).reshape(seq_len, bsz, embed_dim)

        if extra_attention_mask is not None:
            selected_hidden_states = hidden_states.new_zeros(max_num_extra_indices_per_batch, bsz, embed_dim)
            selected_hidden_states[selection_padding_mask_nonzeros[::-1]] = hidden_states[
                extra_attention_mask_nonzeros[::-1]
            ]

            q_global = self.query_global(selected_hidden_states) / math.sqrt(self.head_dim)
            k_global = self.key_global(hidden_states)
            v_global = self.value_global(hidden_states)
            q_global = q_global.contiguous().view(
                max_num_extra_indices_per_batch, bsz * self.num_heads, self.head_dim
            ).transpose(0, 1)
            k_global = k_global.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            v_global = v_global.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

            global_weights = torch.bmm(q_global, k_global.transpose(1, 2))
            global_weights = global_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            if selection_padding_mask_zeros[0].numel() > 0:
                global_weights[selection_padding_mask_zeros[0], :, selection_padding_mask_zeros[1], :] = -10000.0
            if key_padding_mask is not None:
                global_weights = global_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2), -10000.0
                )
            global_weights = global_weights.view(bsz * self.num_heads, max_num_extra_indices_per_batch, seq_len)
            global_probs = F.softmax(global_weights, dim=-1, dtype=torch.float32)
            global_probs = F.dropout(global_probs.type_as(global_weights), p=self.dropout, training=self.training)
            selected_attn = torch.bmm(global_probs, v_global)
            selected_attn = selected_attn.view(bsz, self.num_heads, max_num_extra_indices_per_batch, self.head_dim)
            selected_attn = selected_attn[selection_padding_mask_nonzeros[0], :, selection_padding_mask_nonzeros[1]]
            attn[extra_attention_mask_nonzeros[::-1]] = selected_attn.view(
                len(selection_padding_mask_nonzeros[0]), -1
            ).type_as(hidden_states)
            attn_weights = global_weights

        context_layer = attn.transpose(0, 1)
        if output_attentions:
            if extra_attention_mask is not None:
                attn_weights = attn_weights.view(bsz, self.num_heads, max_num_extra_indices_per_batch, seq_len)
            else:
                attn_weights = attn_weights.permute(0, 2, 1, 3)
            return (context_layer, attn_weights)
        return (context_layer,)


class LongSequenceEncoder(RobertaModel):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.encoder.layer):
            layer.attention.self = LongSequenceSelfAttention(config, layer_id=i)


def extend_position_embeddings(model, max_positions):
    old = model.embeddings.position_embeddings.weight
    new_weight = old.new_empty(max_positions, old.size(1))
    step = old.size(0)
    for start in range(0, max_positions, step):
        length = min(step, max_positions - start)
        new_weight[start:start + length] = old[:length]
    padding_idx = model.embeddings.position_embeddings.padding_idx
    new_embedding = nn.Embedding(max_positions, old.size(1), padding_idx=padding_idx)
    new_embedding.weight.data.copy_(new_weight)
    model.embeddings.position_embeddings = new_embedding
    model.config.max_position_embeddings = max_positions
```

For seq2seq, the same local+global attention replaces full self-attention in the encoder of an encoder-decoder model; the decoder keeps full attention over the encoded sequence and previously decoded tokens.
