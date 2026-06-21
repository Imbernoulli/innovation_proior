The thing that actually hurts is that a Transformer encoder which works beautifully on sentences cannot be run over a whole document — a multi-paragraph QA context, a full article, a coreference passage that spans pages, tens of thousands of characters for character-level language modeling. The model tops out around 512 tokens, and the reason is structural: self-attention computes, for every token, a score against every other token. With queries and keys $Q, K \in \mathbb{R}^{n\times d}$ the product $QK^\top$ is an $n\times n$ matrix, and the layer computes $\mathrm{softmax}(QK^\top/\sqrt{d_k})\,V$. Materializing and normalizing that score matrix costs $O(n^2 d_k)$ time and, the part that bites first, $O(n^2)$ memory. Double the sequence and the attention cost quadruples; somewhere in the low thousands of tokens the $n^2$ matrix simply will not fit on the GPU. The reflex in the field is to not process the long thing at all — truncate the document, or chop it into 512-token chunks, encode each, and glue the pieces back with task-specific machinery; for multi-hop QA, retrieve first and only read what comes back. But every one of these leaks information exactly where it is needed: truncation discards the tail, chunking severs any dependency that crosses a boundary and then demands bespoke per-task code to recover what attention should have handled for free, and retrieve-then-read cascades the retriever's mistakes into the reader. These are workarounds for a model that cannot see far, not a model that can.

What makes the quadratic acceptable to give up is that most of the $n^2$ entries are nearly inert. Probes of trained encoders find that an enormous fraction of attention heads are local — keying on the previous token, the next token, or a tight neighborhood — and local context is repeatedly the most informative signal for building a token's representation. Language structure is mostly local with a thinner thread of genuinely long-range dependence laid on top, so dense attention is the wrong default; the local interaction should be the default and long range should be paid for only where it is actually needed. I propose Longformer, an attention mechanism that scales linearly in sequence length yet keeps both long-range and task-level reach, built so it can be dropped into an already-pretrained encoder.

The core is a sliding window. Instead of attending to all $n$ tokens, each token attends to a fixed total window $W$ — $W/2$ tokens to its left and $W/2$ to its right — the same band for every token, like a 1-D convolution kernel running over the sequence. The score "matrix" is no longer $n\times n$ but a band of about $W$ entries per row around the diagonal, so time and memory are $O(n\cdot W)$, linear in $n$ when $W$ is held fixed. This obviously breaks long-range reach, and the fix is the CNN's: depth. At layer 1 a token's output depends on inputs within $W/2$ on each side; at layer 2 it depends on layer-1 outputs within $W/2$, each of which already summarized $W/2$, so the reach is $W/2 + W/2 = W$ per side. By induction, after $\ell$ layers the one-sided reach is $\ell\cdot(W/2)$ and the total receptive field is about $\ell\cdot W$ tokens. The low layers stay local and cheap, the high layers integrate the whole document, and no single attention operation ever looks beyond its window. To stretch the field further without paying for it, I borrow dilation from à-trous convolution and WaveNet: a dilated sliding window where a token attends to $W/2$ positions spaced $d$ apart rather than $W/2$ adjacent ones. The number of attended positions is unchanged, so compute is unchanged, but per-layer reach becomes $(W/2)\cdot d$ per side and the field grows to $\ell\cdot d\cdot W$, into the tens of thousands. Dilation does skip the local positions that matter most, but attention is multi-headed and the heads need not be identical: some heads run undilated ($d=1$) to keep sharp local detail while others dilate to reach far, and lower layers — whose job is to build clean local features — stay undilated, with a touch of dilation introduced only on a couple of upper-layer heads where distant reach pays off. The same logic says the window should grow from bottom to top: narrow and cheap where features are local, wide where the layer must integrate the whole sequence.

For autoregressive character-level language modeling this dilated, causal, depth-growing window is essentially the whole story, but for the pretrain-finetune paradigm it is not enough, and it fails in a specific way. The shape of the needed interaction there is task-dependent. Masked-token prediction wants local context, so windowing is fine. But classification must squeeze the entire sequence into one decision vector — conventionally the representation of a special $\texttt{[CLS]}$ token — and a windowed $\texttt{[CLS]}$ only ever sees its own window; it would take many layers for information even to reach it. QA concatenates a question with a long document and needs every document token compared against the question, yet a token deep in the passage is many windows away and may never effectively see it. The right fix is to let exactly those few positions talk to everyone. I add global attention on a small, pre-selected, task-chosen set of tokens — $\texttt{[CLS]}$ for classification, the question tokens for QA — that attend to every token and to which every token attends. The symmetry is essential: if $\texttt{[CLS]}$ could read the sequence but no one could read it, it could summarize but not broadcast; a genuine information hub both gathers from everyone and is visible to everyone. With $g$ global tokens this adds $g\cdot n$ (they attend to all) plus $n\cdot g$ (all attend to them), i.e. $O(g\cdot n)$, and since $g$ is a small task-defined constant the combined cost is

$$O(n\cdot W + n\cdot g) = O\big(n\,(W+g)\big) = O(n).$$

Because the global role is categorically different from local scoring — $\texttt{[CLS]}$ comparing itself against a heterogeneous document, a question token scanning a whole passage — forcing both through one set of projections asks a single linear map to serve two similarity functions. So global attention gets its own $Q_g, K_g, V_g$, initialized as copies of the local $Q, K, V$ so they begin equivalent and specialize during training.

Two things make it run in practice. First, the banded (and dilated) matmul is not a framework primitive — PyTorch and TensorFlow give only dense matmul, which would re-materialize the full $n\times n$ and defeat the purpose — so the band is built by hand three ways for three regimes: a slow diagonal loop as a correctness reference; an overlapping-chunk version for the non-dilated pretrain-finetune case that splits the sequence into blocks of $2r$ (with $r = W/2$) overlapping by $r$, runs one fused dense matmul per block so every query's $\pm r$ window lands inside some block, then skews and masks the products into the banded layout (this computes a few extra entries it discards, costing about twice the tight memory, which is harmless at these lengths and very fast); and a compiled CUDA kernel, generated from a high-level description of the banded multiplication, for the dilated autoregressive character-LM regime where it stores only the non-zero band and reaches the longest sequences. Second, to avoid pretraining from scratch I make the attention a true drop-in: take a released RoBERTa checkpoint, replace its dense self-attention with an effective window $W=512$ ($r=256$ per side, matching the original 512-token length), and continue MLM briefly so the model adapts. Dilation is dropped here because the pretrained heads were never trained to skip positions; that is also why the chunked, dilation-free implementation is exactly the right tool for this setting. The learned absolute position embeddings only cover positions up to 512, so to feed 4096 tokens I extend them by copying the learned 512-block and tiling it along the sequence rather than initializing randomly — this preserves the local positional relationships the pretrained heads depend on, leaving only seams at tile boundaries that continued training smooths out. On the character-LM side, training is staged: start with a short sequence length and small window, and on each phase double both while halving the learning rate, so the cheap short phases take most of the gradient steps and the expensive longest-window phase is kept short. The whole construction composes back to the goal — a band plus a constant number of full rows and columns, depth and dilation supplying the long-range reach the band gives up, global tokens supplying the task-level all-to-all it cannot, and a pretrained checkpoint inherited rather than retrained — for total cost $O(n)$.

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
