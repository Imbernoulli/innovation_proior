Let me start from the thing that actually hurts. I have a Transformer encoder that works beautifully on sentences, and I want to run it over a whole document — a multi-paragraph QA context, a full article, a coreference passage that spans pages, tens of thousands of characters for character-level language modeling. And I can't. The model tops out around 512 tokens. Why exactly? Because self-attention computes, for every token, a score against every other token. Q is n×d, K is n×d, and Q Kᵀ is n×n. Softmax that, multiply by V. The whole mechanism is built on the all-pairs interaction, and that interaction is what makes it both powerful and quadratic. Time is O(n²·d) and — the part that bites first — memory is O(n²), because I have to materialize the score matrix to softmax it. Double the sequence and the attention cost quadruples; somewhere in the low thousands of tokens the n² matrix simply doesn't fit on the GPU. So the question is sharp: can I get an attention whose cost grows like n, not n², without throwing away the long-range reach that made attention worth using?

The reflex answer in the field is to not process the long thing at all — truncate the document, or chop it into 512-token chunks, encode each chunk, and glue the pieces back with some task-specific model on top; for multi-hop QA, run a retriever first and only read what it returns. But every one of these leaks information exactly where I need it. Truncation throws away the tail. Chunking severs any dependency that crosses a chunk boundary, and then I'm building bespoke machinery per task to recover what the attention should have handled for free. The retrieve-then-read pipeline cascades the retriever's mistakes into the reader. These are workarounds for a model that can't see far, not a model that can. I want to remove the bottleneck itself.

So let me look at what the n² is actually buying me. Every token attends to every token — but how much of that is real? When people probe the attention heads of trained encoders, an enormous fraction of them are local: heads that key almost entirely on the previous token, or the next token, or a tight neighborhood. Local context keeps coming up as the thing that matters most for building a token's representation. Which means most of the n² entries in that score matrix are nearly inert — I'm paying quadratic cost to compute attention weights that are essentially zero. Language structure is mostly local, with a thinner thread of genuinely long-range dependence laid on top. If that's true, then dense attention is the wrong default; I should make the *local* interaction the default and pay for long range only where I actually need it.

That immediately suggests: don't let each token attend to all n. Let it attend to a fixed window — w/2 tokens to its left and w/2 to its right. A sliding window, the same shape for every token, like a 1-D convolution kernel running over the sequence. Now what does this cost? Each of the n tokens computes scores against only w others, so the score "matrix" isn't n×n anymore — it's a band of width w around the diagonal, n·w entries. Time and memory are O(n·w), and if I hold w fixed and small relative to n, that's linear in n. The quadratic is gone.

But I've obviously broken something. A token can now only see w/2 tokens away. How does information from the far end of the document ever reach the front? This is exactly the problem a CNN has with a small kernel, and the CNN's answer is depth. One convolution sees a local patch; stack two and the second layer's patch is built from first-layer outputs that each already summarized a patch, so the effective receptive field grows. Let me actually count it for the windowed attention. At layer 1, a token's output depends on inputs within w/2 on each side. At layer 2, it depends on layer-1 outputs within w/2 on each side, and each of *those* depended on inputs within w/2 — so the layer-2 output reaches w/2 + w/2 = w on each side. By induction, after ℓ layers the one-sided reach is ℓ·(w/2), so the total receptive field is about ℓ·w tokens. With a normal stack of layers and a modest window, the top layers can already see across the whole document, even though no single attention operation ever looked beyond its window. The high layers form a representation of the entire input; the low layers stay local and cheap. So I keep linear cost and recover global reach through depth — the conv trick, transplanted to attention.

Now is ℓ·w actually enough? For a 12-layer model with, say, w = 512, that's around 6K tokens of reach — fine for documents, but I want character-level LM over tens of thousands of characters, and I don't want to blow up w (that costs compute) or stack absurdly many layers (that costs compute and parameters). I want a bigger receptive field for free. The CNN people hit this too and solved it with dilation: put gaps of size d inside the kernel, so the kernel covers a span of d·(kernel size) while still touching the same number of positions. WaveNet stacks dilated convolutions to get an exponentially growing temporal context at constant per-layer cost. Transplant that: a *dilated* sliding window, where instead of attending to the w/2 immediately adjacent positions on each side, a token attends to w/2 positions spaced d apart. The number of attended positions is unchanged, so the compute is unchanged — still O(n·w) — but each layer now reaches (w/2)·d on each side. Redo the receptive-field count: one-sided reach per layer becomes (w/2)·d, so after ℓ layers the field is ℓ·d·w. Even a small dilation pushes this to tens of thousands of tokens. So dilation enlarges reach without spending anything.

There's a tension, though: dilation skips positions, and the skipped local positions are exactly the ones I argued matter most. If I dilate every head I lose the fine local detail. But attention is multi-headed — each head computes its own scores, and I don't have to make them identical. So let some heads run with no dilation (d = 1) to keep dense local context, and let other heads dilate to reach far. The heads divide the labor: a few "see far," the rest "see sharp." And for the lower layers, where the job is to build clean local features in the first place, I shouldn't dilate at all — give them their full local capacity — and only introduce a little dilation on a couple of heads in the upper layers, where reaching distant tokens pays off. When I think about how to set the window across depth, the same logic says lower layers can be cheap and narrow (local features) while upper layers should be wide (they're the ones integrating the whole sequence), so increasing the window from bottom to top balances cost against representational power better than a fixed window — and far better than the reverse.

So the windowed-plus-dilated pattern gives me cheap local attention with a receptive field I can dial up to the whole document. For autoregressive character LM I think this is actually the whole story — the dilated sliding window, causal (each token attends only to a window of *previous* tokens), with the window growing across layers and a touch of dilation up top. Let me hold that and turn to the other setting I care about, because there it's not enough, and I want to feel exactly where it fails.

The other setting is the pretrain-finetune paradigm — an MLM-pretrained bidirectional encoder, finetuned per task. Here the *shape* of the needed interaction is task-dependent in a way pure windowing can't express. For masked-token prediction, local context is mostly what you need, so windowing is fine. But for classification, the model is supposed to squeeze the entire sequence into one decision vector — conventionally the representation of a special [CLS] token — and a windowed token can't aggregate the whole sequence; it would take many layers for information to even reach [CLS], and [CLS] only ever sees its own window. For QA, the input is the question concatenated with a long document, and *every* document token needs to be compared against the question — but a document token deep in the passage is many windows away from the question and may never effectively see it. The windowed inductive bias is right for building local token representations and wrong for these task-level information flows. I could try to fix it with more layers or wider windows, but that's fighting the architecture; the real issue is that a handful of specific positions need to talk to the whole sequence, and locality forbids exactly that.

So let those few positions be special. Pick a small set of pre-selected tokens — task-chosen: the [CLS] token for classification, all the question tokens for QA — and give them *global* attention: they attend to every token in the sequence, and every token attends to them. It has to be symmetric. If [CLS] could read the whole sequence but no one could read [CLS], it could summarize but not broadcast; if document tokens could read the question but the question couldn't read them, the comparison is one-directional. A genuine information hub both gathers from everyone and is visible to everyone. Now [CLS] aggregates the full sequence in a single layer, and every document token sees every question token directly. What does this cost? If there are g such tokens, the global part adds g·n (they attend to all n) plus n·g (all n attend to them) — that's O(g·n) — and g is a small task-defined constant, independent of n. So the combined local-plus-global attention is O(n·w + n·g) = O(n). Still linear. I've bought exactly the task-level reach I was missing, for the price of a constant number of dense rows and columns, and I've replaced the elaborate chunk-and-combine architectures with a single inductive-bias knob: just mark which tokens are global.

Now, do the global tokens reuse the same Q, K, V projections as the windowed attention? Let me think about what they're doing. The local projections are tuned to score a token against its near neighbors — a fairly homogeneous, local comparison. The global tokens are doing something categorically different: [CLS] comparing itself against an entire heterogeneous document, a question token scanning a whole passage for relevance. Forcing both through one set of projections asks a single linear map to serve two quite different similarity functions, and the global role is the rarer, more specialized one. That's too rigid. Give global attention its own projections — a separate Q_g, K_g, V_g used only when computing the global scores. To not start them from nowhere, initialize them as copies of the local Q, K, V; they begin equivalent and then specialize during training. This separation turns out to matter a lot for downstream quality.

That settles the mechanism. Now I have to actually compute it, and here's a wall: the windowed (and dilated) attention needs a *banded* matrix multiplication — the output is zero except on a few diagonals around the main one — and that operation simply does not exist as a primitive in PyTorch or TensorFlow. The frameworks give me dense matmul, which would materialize the full n×n and defeat the entire point. So I have to build the banded matmul myself, and there's a real engineering tradeoff in how.

The most obvious version: loop over the diagonals. For each offset in the band, multiply the appropriately shifted slices of Q and K to get that diagonal of scores. This only ever computes and stores the non-zero band, so it's perfectly memory-efficient and it naturally handles dilation (the offsets just step by d). But a Python loop over diagonals is brutally slow — fine as a correctness reference, useless for real training.

A faster version for the non-dilated case: chunk the sequence into overlapping blocks of size 2w, with each block overlapping its neighbor by w. Within a block, do one dense matmul of the block's queries against the block's keys — that's a single fused operation the GPU loves — and because consecutive blocks overlap by w, every query's full ±w/2 window lands inside some block. Then I just have to reshape and mask the block products into the banded layout (skew the diagonals into columns, drop the entries outside the window). This uses one big batched matmul instead of a loop, so it's fast; the cost is that it computes some entries it then throws away, so it uses about twice the memory of a perfectly tight implementation. For pretraining and finetuning, where I'm not at the extreme sequence lengths and dilation isn't needed, that memory overhead is a non-issue and the speed is exactly what I want. So this chunked implementation is the one for the pretrain-finetune setting; it doesn't support dilation, but I don't need dilation there (more on that in a moment).

And for the regime that needs it all — character LM, longest sequences, dilation, autoregressive masking — neither of the above is ideal: the loop is too slow, the chunked version can't dilate. So I write a custom CUDA kernel for the banded (dilated) matmul. Rather than hand-write CUDA, I describe the banded multiplication in a high-level form and let a compiler stack generate and optimize the device code; the kernel computes, for each head, exactly the band whose offsets step by that head's dilation d, supports the causal/autoregressive case, and stores only the non-zero values, so it's the most memory-efficient and lets me reach the longest sequences. Three implementations for three needs: loop for testing, chunks for pretrain-finetune, compiled CUDA kernel for character LM.

Now the last piece, and it's about not wasting compute on training. I have a perfectly good pretrained bidirectional encoder already — it cost an enormous amount to train its MLM weights. The whole appeal of making my attention a *drop-in replacement* for full attention is that I shouldn't have to pretrain from scratch. I can take the released checkpoint, swap its dense self-attention for my sliding-window attention, and *continue* MLM pretraining for a relatively short while so the model adapts to the new pattern. For that swap to be gentle, I should match the existing model's compute budget: use window size 512, because that equals the original 512-token sequence length, so per-layer the attention does the same amount of work it was trained to do. And here I find that dilation, which helped so much for character LM, actually hurts when continuing from pretrained weights — the pretrained heads were never trained to skip positions, and forcing gaps into the window fights the learned weights. So no dilation in the pretrain-finetune setting; that's also why the chunked implementation, which doesn't support dilation, is exactly the right tool there.

There's a subtler compatibility problem: the pretrained model has learned *absolute* position embeddings, but only up to position 512, and I want to feed it 4096 tokens. I need embeddings for positions 512 through 4095. If I initialize them randomly, I wreck the model — the encoder has learned, through its attention heads, a strong sense of relative local position (attend to previous token, to next token), and that machinery is keyed to the learned embedding structure; random new embeddings make positions past 512 look like noise. Instead, initialize the new positions by *copying* the existing 512-block, tiled repeatedly along the sequence. Now the local position structure is preserved everywhere — a token at position 1000 has the same positional relationship to its neighbors as a token at position 488 did — with a seam only at the tile boundaries, which continued training smooths out. This copy initialization is almost embarrassingly simple but it's the difference between the swapped model being immediately usable and being broken: with random init the model is hopeless, with copy init it starts close to the original and converges in just a few thousand updates.

One more training detail, back on the character-LM side, about *how* to ramp up to long sequences. If I just start training at the maximum window and sequence length, I'm spending the most expensive compute from step one, while the model is still flailing to learn basic local context. The model needs to nail local structure first and only then learn to exploit long range. So train in stages: start with a short sequence length and a small window, and on each subsequent phase double both the window and the sequence length while halving the learning rate. The cheap short-sequence phases do the bulk of the gradient steps; the expensive longest-window, longest-sequence phase is saved for the end and kept short. This makes the whole training tractable while still ending at the long context I need.

Let me make sure the whole thing actually composes into linear cost and global reach, because that was the entire goal. Local windowed attention: O(n·w), and with ℓ layers a receptive field of ℓ·w; with dilation d on some heads, ℓ·d·w, into the tens of thousands. Global attention on g task-tokens: O(g·n) with g a constant, giving any-to-any reach for exactly the positions that need it. Sum: O(n·(w + g)) = O(n). The quadratic score matrix is replaced by a band plus a constant number of full rows and columns; depth and dilation supply the long-range reach the band gives up; the global tokens supply the task-level all-to-all the band can't; separate projections let the global role specialize; a hand-built banded matmul (loop / chunked / compiled kernel, per regime) makes it run; and copy-initialized position embeddings plus continued MLM let me inherit a pretrained model instead of paying to train one. Now let me write it.

```python
import math
import torch
from torch import nn
import torch.nn.functional as F

# Banded q @ k^T: only the diagonals within +/- w of the main diagonal are computed.
# Chunked implementation (no dilation): split the sequence into overlapping blocks of
# size 2w (overlap w) so each query's full window lands inside one block; do ONE dense
# matmul per block, then skew/mask the block products into the (n, 2w+1) banded layout.
def sliding_chunks_matmul_qk(q, k, w, padding_value):          # q,k: (bsz, n, heads, d)
    bsz, seqlen, num_heads, head_dim = q.size()
    assert seqlen % (w * 2) == 0
    chunks_count = seqlen // w - 1
    q = q.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    k = k.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    chunk_q = _chunk(q, w)                                     # overlapping 2w blocks via as_strided
    chunk_k = _chunk(k, w)
    chunk_attn = torch.einsum('bcxd,bcyd->bcxy', (chunk_q, chunk_k))   # one matmul per block
    diagonal_chunk_attn = _skew(chunk_attn, (0, 0, 0, 1), padding_value)  # diagonals -> columns
    # reassemble into the band: w lower diagonals | self | w upper diagonals
    diagonal_attn = diagonal_chunk_attn.new_empty((bsz * num_heads, chunks_count + 1, w, w * 2 + 1))
    diagonal_attn[:, :-1, :, w:] = diagonal_chunk_attn[:, :, :w, :w + 1]
    diagonal_attn[:, -1, :, w:] = diagonal_chunk_attn[:, -1, w:, :w + 1]
    diagonal_attn[:, 1:, :, :w] = diagonal_chunk_attn[:, :, -(w + 1):-1, w + 1:]
    diagonal_attn[:, 0, 1:w, 1:w] = diagonal_chunk_attn[:, 0, :w - 1, 1 - w:]
    diagonal_attn = diagonal_attn.view(bsz, num_heads, seqlen, 2 * w + 1).transpose(2, 1)
    mask_invalid_locations(diagonal_attn, w, 1, False)        # kill the band entries that ran off the ends
    return diagonal_attn                                      # (bsz, n, heads, 2w+1)


def sliding_chunks_matmul_pv(prob, v, w):                     # the matching value aggregation
    bsz, seqlen, num_heads, head_dim = v.size()
    chunks_count = seqlen // w - 1
    chunk_prob = prob.transpose(1, 2).reshape(bsz * num_heads, seqlen // w, w, 2 * w + 1)
    v = v.transpose(1, 2).reshape(bsz * num_heads, seqlen, head_dim)
    padded_v = F.pad(v, (0, 0, w, w), value=-1)               # pad so each chunk has its 3w context
    chunk_v = padded_v.as_strided(size=(bsz * num_heads, chunks_count + 1, 3 * w, head_dim),
                                  stride=(padded_v.stride()[0], w * padded_v.stride()[1],
                                          padded_v.stride()[1], padded_v.stride()[2]))
    skewed_prob = _skew2(chunk_prob, padding_value=0)
    context = torch.einsum('bcwd,bcdh->bcwh', (skewed_prob, chunk_v))
    return context.view(bsz, num_heads, seqlen, head_dim).transpose(1, 2)


class LongformerSelfAttention(nn.Module):
    def __init__(self, config, layer_id):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.embed_dim = config.hidden_size
        # local windowed attention: the usual q/k/v
        self.query = nn.Linear(config.hidden_size, self.embed_dim)
        self.key   = nn.Linear(config.hidden_size, self.embed_dim)
        self.value = nn.Linear(config.hidden_size, self.embed_dim)
        # SEPARATE projections for the few global tokens (init as copies of the local ones)
        self.query_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.key_global   = nn.Linear(config.hidden_size, self.embed_dim)
        self.value_global = nn.Linear(config.hidden_size, self.embed_dim)
        self.dropout = config.attention_probs_dropout_prob
        self.layer_id = layer_id
        self.attention_window = config.attention_window[layer_id]   # per-layer window (grows with depth)
        self.attention_dilation = config.attention_dilation[layer_id]  # per-layer/head dilation (1 = none)

    def forward(self, hidden_states, attention_mask=None):
        # attention_mask convention: -ve = no attention, 0 = local only, +ve = global token
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(dim=2).squeeze(dim=1)
            key_padding_mask = attention_mask < 0
            extra_attention_mask = attention_mask > 0            # which tokens are global
            remove_from_windowed = attention_mask != 0
            num_extra = extra_attention_mask.long().sum(dim=1)
            max_num_extra = num_extra.max()
        else:
            extra_attention_mask = None; key_padding_mask = None; remove_from_windowed = None

        hidden_states = hidden_states.transpose(0, 1)
        seq_len, bsz, embed_dim = hidden_states.size()
        q = self.query(hidden_states); k = self.key(hidden_states); v = self.value(hidden_states)
        q /= math.sqrt(self.head_dim)                            # 1/sqrt(d_k) scaling
        q = q.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)

        # --- LOCAL: banded scores instead of the full n x n ---
        attn_weights = sliding_chunks_matmul_qk(q, k, self.attention_window, padding_value=0)
        mask_invalid_locations(attn_weights, self.attention_window, self.attention_dilation, False)
        # global tokens are excluded from the windowed scores (they get the dense path below)
        # ... (mask `remove_from_windowed` positions in the band) ...

        # --- GLOBAL (columns): every token also scores against the global tokens ---
        if extra_attention_mask is not None:
            selected_k = k.new_zeros(bsz, max_num_extra, self.num_heads, self.head_dim)
            # gather the global tokens' keys, then score ALL queries against them
            selected_attn_weights = torch.einsum('blhd,bshd->blhs', (q, selected_k))
            attn_weights = torch.cat((selected_attn_weights, attn_weights), dim=-1)  # band + global cols

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)  # fp32 for stability
        attn_probs = F.dropout(attn_weights.type_as(q), p=self.dropout, training=self.training)
        v = v.view(seq_len, bsz, self.num_heads, self.head_dim).transpose(0, 1)

        # aggregate values: global columns (dense) + local band
        attn = 0
        if extra_attention_mask is not None:
            selected_v = v.new_zeros(bsz, max_num_extra, self.num_heads, self.head_dim)
            selected_probs = attn_probs.narrow(-1, 0, max_num_extra)
            attn = torch.matmul(selected_probs.transpose(1, 2),
                                selected_v.transpose(1, 2)).transpose(1, 2)
            attn_probs = attn_probs.narrow(-1, max_num_extra, attn_probs.size(-1) - max_num_extra)
        attn += sliding_chunks_matmul_pv(attn_probs, v, self.attention_window)
        attn = attn.transpose(0, 1).reshape(seq_len, bsz, embed_dim)

        # --- GLOBAL (rows): recompute the global tokens' OWN output with the SEPARATE projections,
        #     attending to the ENTIRE sequence (symmetric: they read everyone, everyone read them) ---
        if extra_attention_mask is not None:
            selected_hidden = ...  # gather hidden states at the global positions
            qg = self.query_global(selected_hidden) / math.sqrt(self.head_dim)
            kg = self.key_global(hidden_states)
            vg = self.value_global(hidden_states)
            g_weights = torch.bmm(qg, kg.transpose(1, 2))        # global queries vs ALL keys -> dense, O(g*n)
            g_probs = F.softmax(g_weights, dim=-1, dtype=torch.float32)
            g_out = torch.bmm(F.dropout(g_probs, p=self.dropout, training=self.training), vg)
            attn[extra positions] = g_out                        # overwrite the global tokens' outputs
        return (attn.transpose(0, 1),)
```

```python
# Inheriting a pretrained encoder: swap dense self-attention for the windowed one,
# extend position embeddings by COPYING the 512-block, then continue MLM briefly.
class Longformer(PretrainedRobertaModel):
    def __init__(self, config):
        super().__init__(config)
        for i, layer in enumerate(self.encoder.layer):
            layer.attention.self = LongformerSelfAttention(config, layer_id=i)

def extend_position_embeddings(model, max_pos):                 # 512 -> 4096
    old = model.embeddings.position_embeddings.weight           # (512, d)
    new = old.new_empty(max_pos, old.size(1))
    step = old.size(0)
    for start in range(0, max_pos, step):                       # tile the learned 512-block
        new[start:start + step] = old[:min(step, max_pos - start)]
    model.embeddings.position_embeddings = nn.Embedding.from_pretrained(new, freeze=False)
```

The causal chain, end to end: full attention is O(n²) because it scores all pairs, which is infeasible for long documents and mostly wasted since language structure is dominated by local context. Restricting each token to a sliding window of w neighbors makes attention O(n·w) — linear — and stacking ℓ layers grows the receptive field to ℓ·w, recovering global reach through depth the way a CNN does; dilating the window on some heads pushes the field to ℓ·d·w at no extra cost while undilated heads keep local detail. Because a few task positions ([CLS], question tokens) genuinely need to see the whole sequence and locality forbids that, those tokens get symmetric global attention — O(g·n) with g constant, so the total stays O(n) — and because their role differs from local attention they get their own Q/K/V projections, initialized from the local ones. The banded matmul this needs isn't a framework primitive, so it's built three ways (loop for tests, overlapping-chunk single-matmul for pretrain-finetune, compiled CUDA kernel with dilation for character LM). Finally, to avoid pretraining from scratch, the windowed attention is dropped into a pretrained checkpoint, its position embeddings extended by copying the learned 512-block (preserving the local position bias), and MLM continued for a short staged schedule — yielding a long-document model that scales linearly and reads the whole input in one pass.
