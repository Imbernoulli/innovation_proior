Let me start from the thing that actually hurts. I have a Transformer encoder that works beautifully on sentences, and I want to run it over a whole document — a multi-paragraph QA context, a full article, a coreference passage that spans pages, tens of thousands of characters for character-level language modeling. And I can't. The model tops out around 512 tokens. Why exactly? Because self-attention computes, for every token, a score against every other token. Q is n×d, K is n×d, and Q Kᵀ is n×n. Softmax that, multiply by V. The whole mechanism is built on the all-pairs interaction, and that interaction is what makes it both powerful and quadratic. Time is O(n²·d) and — the part that bites first — memory is O(n²), because I have to materialize the score matrix to softmax it. Double the sequence and the attention cost quadruples; somewhere in the low thousands of tokens the n² matrix simply doesn't fit on the GPU. So the question is sharp: can I get an attention whose cost grows like n, not n², without throwing away the long-range reach that made attention worth using?

The reflex answer in the field is to not process the long thing at all — truncate the document, or chop it into 512-token chunks, encode each chunk, and glue the pieces back with some task-specific model on top; for multi-hop QA, run a retriever first and only read what it returns. But every one of these leaks information exactly where I need it. Truncation throws away the tail. Chunking severs any dependency that crosses a chunk boundary, and then I'm building bespoke machinery per task to recover what the attention should have handled for free. The retrieve-then-read pipeline cascades the retriever's mistakes into the reader. These are workarounds for a model that can't see far, not a model that can. I want to remove the bottleneck itself.

So let me look at what the n² is actually buying me. Every token attends to every token — but how much of that is real? When people probe the attention heads of trained encoders, an enormous fraction of them are local: heads that key almost entirely on the previous token, or the next token, or a tight neighborhood. Local context keeps coming up as the thing that matters most for building a token's representation. Which means most of the n² entries in that score matrix are nearly inert — I'm paying quadratic cost to compute attention weights that are essentially zero. Language structure is mostly local, with a thinner thread of genuinely long-range dependence laid on top. If that's true, then dense attention is the wrong default; I should make the *local* interaction the default and pay for long range only where I actually need it.

That immediately suggests: don't let each token attend to all n. Let it attend to a fixed total window W — W/2 tokens to its left and W/2 to its right. A sliding window, the same shape for every token, like a 1-D convolution kernel running over the sequence. Now what does this cost? Each of the n tokens computes scores against only about W others, so the score "matrix" isn't n×n anymore — it's a band with about W entries per row around the diagonal, n·W entries. Time and memory are O(n·W), and if I hold W fixed and small relative to n, that's linear in n. The quadratic is gone.

But I've obviously broken something. A token can now only see W/2 tokens away. How does information from the far end of the document ever reach the front? This is exactly the problem a CNN has with a small kernel, and the CNN's answer is depth. One convolution sees a local patch; stack two and the second layer's patch is built from first-layer outputs that each already summarized a patch, so the effective receptive field grows. Let me actually count it for the windowed attention. At layer 1, a token's output depends on inputs within W/2 on each side. At layer 2, it depends on layer-1 outputs within W/2 on each side, and each of *those* depended on inputs within W/2 — so the layer-2 output reaches W/2 + W/2 = W on each side. By induction, after ℓ layers the one-sided reach is ℓ·(W/2), so the total receptive field is about ℓ·W tokens. With a normal stack of layers and a modest window, the top layers can already see across the whole document, even though no single attention operation ever looked beyond its window. The high layers form a representation of the entire input; the low layers stay local and cheap. So I keep linear cost and recover global reach through depth — the conv trick, transplanted to attention.

Now is ℓ·W actually enough? For a 12-layer model with, say, W = 512, that's around 6K tokens of reach — fine for documents, but I want character-level LM over tens of thousands of characters, and I don't want to blow up W (that costs compute) or stack absurdly many layers (that costs compute and parameters). I want a bigger receptive field for free. The CNN people hit this too and solved it with dilation: put gaps of size d inside the kernel, so the kernel covers a span of d·(kernel size) while still touching the same number of positions. WaveNet stacks dilated convolutions to get an exponentially growing temporal context at constant per-layer cost. Transplant that: a *dilated* sliding window, where instead of attending to the W/2 immediately adjacent positions on each side, a token attends to W/2 positions spaced d apart. The number of attended positions is unchanged, so the compute is unchanged — still O(n·W) — but each layer now reaches (W/2)·d on each side. Redo the receptive-field count: one-sided reach per layer becomes (W/2)·d, so after ℓ layers the field is ℓ·d·W. Even a small dilation pushes this to tens of thousands of tokens. So dilation enlarges reach without spending anything.

There's a tension, though: dilation skips positions, and the skipped local positions are exactly the ones I argued matter most. If I dilate every head I lose the fine local detail. But attention is multi-headed — each head computes its own scores, and I don't have to make them identical. So let some heads run with no dilation (d = 1) to keep dense local context, and let other heads dilate to reach far. The heads divide the labor: a few "see far," the rest "see sharp." And for the lower layers, where the job is to build clean local features in the first place, I shouldn't dilate at all — give them their full local capacity — and only introduce a little dilation on a couple of heads in the upper layers, where reaching distant tokens pays off. When I think about how to set the window across depth, the same logic says lower layers can be cheap and narrow (local features) while upper layers should be wide (they're the ones integrating the whole sequence), so increasing the window from bottom to top balances cost against representational power better than a fixed window — and far better than the reverse.

So the windowed-plus-dilated pattern gives me cheap local attention with a receptive field I can dial up to the whole document. For autoregressive character LM I think this is actually the whole story — the dilated sliding window, causal (each token attends only to a window of *previous* tokens), with the window growing across layers and a touch of dilation up top. Let me hold that and turn to the other setting I care about, because there it's not enough, and I want to feel exactly where it fails.

The other setting is the pretrain-finetune paradigm — an MLM-pretrained bidirectional encoder, finetuned per task. Here the *shape* of the needed interaction is task-dependent in a way pure windowing can't express. For masked-token prediction, local context is mostly what you need, so windowing is fine. But for classification, the model is supposed to squeeze the entire sequence into one decision vector — conventionally the representation of a special [CLS] token — and a windowed token can't aggregate the whole sequence; it would take many layers for information to even reach [CLS], and [CLS] only ever sees its own window. For QA, the input is the question concatenated with a long document, and *every* document token needs to be compared against the question — but a document token deep in the passage is many windows away from the question and may never effectively see it. The windowed inductive bias is right for building local token representations and wrong for these task-level information flows. I could try to fix it with more layers or wider windows, but that's fighting the architecture; the real issue is that a handful of specific positions need to talk to the whole sequence, and locality forbids exactly that.

So let those few positions be special. Pick a small set of pre-selected tokens — task-chosen: the [CLS] token for classification, all the question tokens for QA — and give them *global* attention: they attend to every token in the sequence, and every token attends to them. It has to be symmetric. If [CLS] could read the whole sequence but no one could read [CLS], it could summarize but not broadcast; if document tokens could read the question but the question couldn't read them, the comparison is one-directional. A genuine information hub both gathers from everyone and is visible to everyone. Now [CLS] aggregates the full sequence in a single layer, and every document token sees every question token directly. What does this cost? If there are g such tokens, the global part adds g·n (they attend to all n) plus n·g (all n attend to them) — that's O(g·n) — and g is a small task-defined constant, independent of n. So the combined local-plus-global attention is O(n·W + n·g) = O(n). Still linear. I've bought exactly the task-level reach I was missing, for the price of a constant number of dense rows and columns, and I've replaced the elaborate chunk-and-combine architectures with a single inductive-bias knob: just mark which tokens are global.

Now, do the global tokens reuse the same Q, K, V projections as the windowed attention? Let me think about what they're doing. The local projections are tuned to score a token against its near neighbors — a fairly homogeneous, local comparison. The global tokens are doing something categorically different: [CLS] comparing itself against an entire heterogeneous document, a question token scanning a whole passage for relevance. Forcing both through one set of projections asks a single linear map to serve two quite different similarity functions, and the global role is the rarer, more specialized one. That's too rigid. Give global attention its own projections — a separate Q_g, K_g, V_g used only when computing the global scores. To not start them from nowhere, initialize them as copies of the local Q, K, V; they begin equivalent and then can specialize during training.

That settles the mechanism. Now I have to actually compute it, and here's a wall: the windowed (and dilated) attention needs a *banded* matrix multiplication — the output is zero except on a few diagonals around the main one — and that operation simply does not exist as a primitive in PyTorch or TensorFlow. The frameworks give me dense matmul, which would materialize the full n×n and defeat the entire point. So I have to build the banded matmul myself, and there's a real engineering tradeoff in how.

The most obvious version: loop over the diagonals. For each offset in the band, multiply the appropriately shifted slices of Q and K to get that diagonal of scores. This only ever computes and stores the non-zero band, so it's perfectly memory-efficient and it naturally handles dilation (the offsets just step by d). But a Python loop over diagonals is brutally slow — fine as a correctness reference, useless for real training.

A faster version for the non-dilated case needs a second symbol, because the implementation naturally stores the one-sided radius. Let r = W/2. I chunk the sequence into overlapping blocks of size 2r, with each block overlapping its neighbor by r. Within a block, do one dense matmul of the block's queries against the block's keys — that's a single fused operation the GPU loves — and because consecutive blocks overlap by r, every query's full ±r window lands inside some block. Then I just have to reshape and mask the block products into the banded layout (skew the diagonals into columns, drop the entries outside the window). This uses one big batched matmul instead of a loop, so it's fast; the cost is that it computes some entries it then throws away, so it uses about twice the memory of a perfectly tight implementation. For pretraining and finetuning, where I'm not at the extreme sequence lengths and dilation isn't needed, that memory overhead is a non-issue and the speed is exactly what I want. So this chunked implementation is the one for the pretrain-finetune setting; it doesn't support dilation, but I don't need dilation there (more on that in a moment).

And for the regime that needs it all — character LM, longest sequences, dilation, autoregressive masking — neither of the above is ideal: the loop is too slow, the chunked version can't dilate. So I write a custom CUDA kernel for the banded (dilated) matmul. Rather than hand-write CUDA, I describe the banded multiplication in a high-level form and let a compiler stack generate and optimize the device code; the kernel computes, for each head, exactly the band whose offsets step by that head's dilation d, supports the causal/autoregressive case, and stores only the non-zero values, so it's the most memory-efficient and lets me reach the longest sequences. Three implementations for three needs: loop for testing, chunks for pretrain-finetune, compiled CUDA kernel for character LM.

Now the last piece, and it's about not wasting compute on training. I have a perfectly good pretrained bidirectional encoder already — it cost an enormous amount to train its MLM weights. The whole appeal of making my attention a *drop-in replacement* for full attention is that I shouldn't have to pretrain from scratch. I can take the released checkpoint, swap its dense self-attention for my sliding-window attention, and *continue* MLM pretraining for a relatively short while so the model adapts to the new pattern. For that swap to be gentle, I should match the existing model's compute budget: use an effective window W = 512, implemented as r = 256 tokens on each side, because that equals the original 512-token sequence length. And here I find that dilation, which helped so much for character LM, fights the pretrained weights — the pretrained heads were never trained to skip positions. So no dilation in the pretrain-finetune setting; that's also why the chunked implementation, which doesn't support dilation, is exactly the right tool there.

There's a subtler compatibility problem: the pretrained model has learned *absolute* position embeddings, but only up to position 512, and I want to feed it 4096 tokens. I need embeddings for positions 512 through 4095. If I initialize them randomly, I wreck the model's first signal beyond the original range — the encoder has learned, through its attention heads, a strong sense of relative local position (attend to previous token, to next token), and that machinery is keyed to the learned embedding structure; random new embeddings make positions past 512 look like noise. Instead, initialize the new positions by *copying* the existing 512-block, tiled repeatedly along the sequence. Now the local position structure is preserved everywhere — a token at position 1000 has the same positional relationship to its neighbors as a token at position 488 did — with a seam only at the tile boundaries, which continued training can smooth out.

One more training detail, back on the character-LM side, about *how* to ramp up to long sequences. If I just start training at the maximum window and sequence length, I'm spending the most expensive compute from step one, while the model is still flailing to learn basic local context. The model needs to nail local structure first and only then learn to exploit long range. So train in stages: start with a short sequence length and a small window, and on each subsequent phase double both the window and the sequence length while halving the learning rate. The cheap short-sequence phases do the bulk of the gradient steps; the expensive longest-window, longest-sequence phase is saved for the end and kept short. This makes the whole training tractable while still ending at the long context I need.

Let me make sure the whole thing actually composes into linear cost and global reach, because that was the entire goal. Local windowed attention: O(n·W), and with ℓ layers a receptive field of ℓ·W; with dilation d on some heads, ℓ·d·W, into the tens of thousands. Global attention on g task-tokens: O(g·n) with g a constant, giving any-to-any reach for exactly the positions that need it. Sum: O(n·(W + g)) = O(n). The quadratic score matrix is replaced by a band plus a constant number of full rows and columns; depth and dilation supply the long-range reach the band gives up; the global tokens supply the task-level all-to-all the band can't; separate projections let the global role specialize; a hand-built banded matmul (loop / chunked / compiled kernel, per regime) makes it run; and copy-initialized position embeddings plus continued MLM let me inherit a pretrained model instead of paying to train one. I can now write the code that occupies the empty self-attention slot.

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
            extra_attention_mask = attention_mask > 0            # which tokens are global
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

So the chain I end up with is: full attention is O(n²) because it scores all pairs, which is infeasible for long documents and mostly wasted since language structure is dominated by local context. Restricting each token to a sliding window of W neighbors makes attention O(n·W), and stacking ℓ layers grows the receptive field to ℓ·W, recovering global reach through depth the way a CNN does; dilating the window on some heads pushes the field to ℓ·d·W at no extra cost while undilated heads keep local detail. Because a few task positions ([CLS], question tokens) genuinely need to see the whole sequence and locality forbids that, those tokens get symmetric global attention — O(g·n) with g constant, so the total stays O(n) — and because their role differs from local attention they get their own Q/K/V projections, initialized from the local ones. The banded matmul this needs isn't a framework primitive, so it's built three ways (loop for tests, overlapping-chunk single-matmul for pretrain-finetune, compiled CUDA kernel with dilation for character LM). Finally, to avoid pretraining from scratch, the windowed attention is dropped into a pretrained checkpoint, its position embeddings extended by copying the learned position block, and MLM continued briefly — yielding a long-document model that scales linearly and reads the whole input in one pass.
