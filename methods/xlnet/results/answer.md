# XLNet: Generalized Autoregressive Pretraining

## Problem

Pretrain a language representation that (1) captures deep bidirectional context for downstream understanding tasks, (2) uses no artificial input symbols (no pretrain–finetune discrepancy), (3) makes no conditional-independence assumption among predicted tokens, and (4) stays inside the autoregressive density-estimation framework so language-modeling advances transfer. Autoregressive LMs satisfy (2)–(4) but are unidirectional; the masked/denoising autoencoder (BERT) is bidirectional but violates (2) ([MASK]) and (3) (independence assumption).

## Key idea — permutation language modeling

Keep the autoregressive product form, but maximize the expected log-likelihood over **all factorization orders** rather than a fixed forward order. With Z_T the set of permutations of (1,…,T), z_t the t-th element and z_{<t} the prefix:

  max_θ  E_{z∼Z_T} [ Σ_{t=1}^T log p_θ(x_{z_t} | x_{z_{<t}}) ].

Because one shared θ is trained across all orders, in expectation every position conditions on tokens from both sides — bidirectional context — while every term is a genuine AR conditional (no independence assumption, no [MASK]). Only the *factorization order* is permuted, realized through attention masks; the sequence order and positional encodings stay natural, so finetuning sees ordinary text.

## Target-aware two-stream self-attention

A naive softmax p(x_{z_t}=x | x_{z_{<t}}) ∝ exp(e(x)ᵀ h_θ(x_{z_{<t}})) is **independent of the target position z_t**: two permutations with the same prefix but different next-positions i≠j produce identical distributions, which is wrong. So the representation must take z_t as input, g_θ(x_{z_{<t}}, z_t):

  p_θ(X_{z_t}=x | x_{z_{<t}}) = exp(e(x)ᵀ g_θ(x_{z_{<t}}, z_t)) / Σ_{x′} exp(e(x′)ᵀ g_θ(x_{z_{<t}}, z_t)).

Building g hits a contradiction: to *predict* x_{z_t} the representation must exclude its own content, but to serve as *context* for later predictions it must include it. Resolved with **two streams** sharing parameters:
- **Content stream** h_{z_t}: attends to x_{z_{≤t}} (includes its own token); a standard self-attention hidden state. Init h_i^(0)=e(x_i).
- **Query stream** g_{z_t}: attends to x_{z_{<t}} and the position z_t but **not** the content x_{z_t}. Init g_i^(0)=w (a shared trainable vector). The top layer g_{z_t}^(M) feeds the softmax.

Per layer m, both streams take keys/values from the content representations:

  g_{z_t}^(m) ← Attention(Q=g_{z_t}^(m−1), KV=h_{z_{<t}}^(m−1))   (query: position, not content)
  h_{z_t}^(m) ← Attention(Q=h_{z_t}^(m−1), KV=h_{z_{≤t}}^(m−1))   (content: position and content)

The content stream is exactly a normal Transformer, so at finetuning the query stream is dropped.

## Partial prediction

The full objective converges slowly. Split z at cutpoint c and predict only the long-context tail z_{>c}:

  max_θ  E_{z∼Z_T} [ Σ_{t=c+1}^{|z|} log p_θ(x_{z_t} | x_{z_{<t}}) ],

with about 1/K tokens predicted (K≈6). Non-target tokens need no query stream, saving compute. Versus BERT on the same target set T (non-targets N): BERT optimizes Σ_{x∈T} log p(x|N); XLNet optimizes Σ_{x∈T} log p(x | N ∪ T_{<x}), covering every dependency BERT covers plus those into earlier targets — a strict superset.

## Transformer-XL integration

- **Relative positional encoding** (required by the keep-natural-order/mask-only design): the attention logit is decomposed into a content term, a relative-position term, and a relative-segment term, with global learnable biases; distances use actual indices z_i−z_j. Applied to both streams.
- **Segment recurrence**: cache previous segment's content states h̃^(m) as read-only memory; current segment attends to [h̃^(m−1), h^(m−1)]. Because positions are relative, the memory is reusable independent of the previous segment's factorization order.
- **Relative segment encoding** (multi-segment inputs [CLS,A,SEP,B,SEP]): encode only same-segment-vs-not with learnable s_+/s_−, term (q_i+b)ᵀ s_{ij}; generalizes and supports >2 segments. Next-sentence-prediction is dropped (no consistent gain).

## Full layer computation (two-stream, TXL backbone)

Init h_t=e(x_t), g_t=w; memory h̃^(m). For m=1..M:

  ĥ_{z_t}^(m) = LayerNorm( h_{z_t}^(m−1) + RelAttn(h_{z_t}^(m−1), [h̃^(m−1), h_{z_{≤t}}^(m−1)]) )
  h_{z_t}^(m) = LayerNorm( ĥ_{z_t}^(m) + PosFF(ĥ_{z_t}^(m)) )
  ĝ_{z_t}^(m) = LayerNorm( g_{z_t}^(m−1) + RelAttn(g_{z_t}^(m−1), [h̃^(m−1), h_{z_{<t}}^(m−1)]) )
  g_{z_t}^(m) = LayerNorm( ĝ_{z_t}^(m) + PosFF(ĝ_{z_t}^(m)) )
  p_θ(X_{z_t}=x | x_{z_{<t}}) = exp(e(x)ᵀ g_{z_t}^(M)) / Σ_{x′} exp(e(x′)ᵀ g_{z_t}^(M))

## Code

```python
import numpy as np
import tensorflow as tf

def head_projection(h, d_model, n_head, d_head, init, name):
    w = tf.get_variable('{}/kernel'.format(name), [d_model, n_head, d_head],
                        dtype=h.dtype, initializer=init)
    return tf.einsum('ibh,hnd->ibnd', h, w)

def post_attention(h, attn_vec, d_model, n_head, d_head, dropout, is_training, init, residual=True):
    proj_o = tf.get_variable('o/kernel', [d_model, n_head, d_head], dtype=h.dtype, initializer=init)
    attn_out = tf.einsum('ibnd,hnd->ibh', attn_vec, proj_o)
    attn_out = tf.layers.dropout(attn_out, dropout, training=is_training)
    base = attn_out + h if residual else attn_out
    return tf.contrib.layers.layer_norm(base, begin_norm_axis=-1, scope='LayerNorm')

def rel_shift(x, klen=-1):
    s = tf.shape(x)
    x = tf.reshape(x, [s[1], s[0], s[2], s[3]])
    x = tf.slice(x, [1, 0, 0, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, [s[0], s[1] - 1, s[2], s[3]])
    return tf.slice(x, [0, 0, 0, 0], [-1, klen, -1, -1])

def rel_attn_core(q_head, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                  r_w_bias, r_r_bias, r_s_bias, attn_mask, dropatt, is_training, scale):
    ac = tf.einsum('ibnd,jbnd->ijbn', q_head + r_w_bias, k_head_h)        # content-to-content
    bd = tf.einsum('ibnd,jbnd->ijbn', q_head + r_r_bias, k_head_r)        # content-to-rel-position
    bd = rel_shift(bd, klen=tf.shape(ac)[1])
    if seg_mat is None:
        ef = 0
    else:
        ef = tf.einsum('ibnd,snd->ibns', q_head + r_s_bias, seg_embed)    # relative-segment
        ef = tf.einsum('ijbs,ibns->ijbn', seg_mat, ef)
    attn_score = (ac + bd + ef) * scale
    if attn_mask is not None:
        attn_score = attn_score - 1e30 * attn_mask
    attn_prob = tf.layers.dropout(tf.nn.softmax(attn_score, 1), dropatt, training=is_training)
    return tf.einsum('ijbn,jbnd->ibnd', attn_prob, v_head_h)

def two_stream_rel_attn(h, g, r, mems, r_w_bias, r_r_bias, seg_mat, r_s_bias, seg_embed,
                        attn_mask_h, attn_mask_g, target_mapping,
                        d_model, n_head, d_head, dropout, dropatt, is_training, init, scope='rel_attn'):
    scale = 1 / (d_head ** 0.5)
    with tf.variable_scope(scope, reuse=False):
        cat = tf.concat([mems, h], 0) if mems is not None and mems.shape.ndims > 1 else h
        k_head_h = head_projection(cat, d_model, n_head, d_head, init, 'k')   # shared K from content
        v_head_h = head_projection(cat, d_model, n_head, d_head, init, 'v')   # shared V from content
        k_head_r = head_projection(r,   d_model, n_head, d_head, init, 'r')   # relative-position keys

        # content stream: query from h, mask allows self (z_<=t)
        q_head_h = head_projection(h, d_model, n_head, d_head, init, 'q')
        attn_vec_h = rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                                   r_w_bias, r_r_bias, r_s_bias, attn_mask_h, dropatt, is_training, scale)
        output_h = post_attention(h, attn_vec_h, d_model, n_head, d_head, dropout, is_training, init)

    with tf.variable_scope(scope, reuse=True):
        # query stream: query from g, mask forbids self (z_<t)
        q_head_g = head_projection(g, d_model, n_head, d_head, init, 'q')
        if target_mapping is not None:
            q_head_g = tf.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping)  # only predicted slots
            attn_vec_g = rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                                       r_w_bias, r_r_bias, r_s_bias, attn_mask_g, dropatt, is_training, scale)
            attn_vec_g = tf.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping)
        else:
            attn_vec_g = rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                                       r_w_bias, r_r_bias, r_s_bias, attn_mask_g, dropatt, is_training, scale)
        output_g = post_attention(g, attn_vec_g, d_model, n_head, d_head, dropout, is_training, init)
    return output_h, output_g

def local_perm(inputs, targets, is_masked, perm_size, seq_len, SEP_ID, CLS_ID):
    """Sample a factorization order and compile it into attention masks."""
    index = tf.range(seq_len, dtype=tf.int64)
    index = tf.transpose(tf.reshape(index, [-1, perm_size]))
    index = tf.random_shuffle(index)
    index = tf.reshape(tf.transpose(index), [-1])                      # ranks define the order

    non_func = tf.logical_not(tf.logical_or(tf.equal(inputs, SEP_ID), tf.equal(inputs, CLS_ID)))
    non_mask = tf.logical_and(tf.logical_not(is_masked), non_func)     # context (not predicted)
    masked_or_func = tf.logical_not(non_mask)

    rev_index = tf.where(non_mask, -tf.ones([seq_len], tf.int64), index)  # context rank = -1 (seen by all)
    target_tokens = tf.logical_and(masked_or_func, non_func)
    target_mask = tf.cast(target_tokens, tf.float32)                  # predicted -> in loss

    self_rev_index = tf.where(target_tokens, rev_index, rev_index + 1)   # target cannot see itself
    perm_mask = tf.logical_and(self_rev_index[:, None] <= rev_index[None, :], masked_or_func)
    perm_mask = tf.cast(perm_mask, tf.float32)                         # 1 => i cannot attend j (query mask)

    new_targets = tf.concat([inputs[0:1], targets[:-1]], axis=0)
    return perm_mask, new_targets, target_mask, inputs, target_mask   # perm_mask, target, tgt_mask, inp_k, inp_q

def two_stream_loss(hidden, target, target_mask, n_token, d_model, init):
    softmax_w = tf.get_variable('softmax_w', [n_token, d_model], initializer=init)
    softmax_b = tf.get_variable('softmax_b', [n_token], initializer=tf.zeros_initializer())
    logits = tf.einsum('ibd,nd->ibn', hidden, softmax_w) + softmax_b
    per_tok = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits)
    return tf.reduce_sum(per_tok * target_mask) / tf.reduce_sum(target_mask)
```

In the full stack: the query-stream mask is `perm_mask` (excludes self); the content-stream mask `non_tgt_mask` is `perm_mask` with the diagonal re-allowed (a position sees itself). The query stream is initialized from a shared mask embedding `w`, the content stream from word embeddings; only target positions carry a query stream and contribute to the loss.

XLNet-Large config: 24 layers, d_model 1024, 16 heads, d_head 64, FFN inner 4096, GeLU, dropout 0.1, partial-prediction K=6, max length 512, batch 8192, lr 4e-4 (linear decay, 40K warmup), 500K steps, Adam ε=1e-6, weight decay 0.01.
