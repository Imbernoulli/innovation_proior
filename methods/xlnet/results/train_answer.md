We have effectively unlimited unlabeled text and a small amount of labeled data per downstream understanding task, and the recipe is fixed: pretrain a deep network self-supervised on the raw text, then finetune. The only thing left to choose is the pretraining objective, and I want one that hits four things at once that are currently in tension. It must expose every token to deep bidirectional context, because deciding whether a span answers a question or whether one sentence entails another needs a token's representation to depend on what lies on both sides of it. It must introduce no artificial input symbol that appears in pretraining but never in real downstream text, or finetuning has to spend itself correcting for a mismatch. It must model the joint distribution of the tokens it predicts honestly, without assuming those predicted tokens are mutually independent given the context, because language is full of high-order, long-range dependencies that live precisely between two targets. And it should stay inside the autoregressive density-estimation framework so that ongoing language-modeling progress — deeper nets, longer context — transfers straight into pretraining.

The two leading families each get some of this and miss the rest, and the exact shape of what each misses is what a better objective must repair. Autoregressive (AR) language modeling factorizes the joint by the chain rule in a fixed forward order, $\max_\theta \sum_{t=1}^T \log p_\theta(x_t \mid x_{<t})$ with $p_\theta(x_t\mid x_{<t}) = \exp(h_\theta(x_{1:t-1})^\top e(x_t)) / \sum_{x'} \exp(h_\theta(x_{1:t-1})^\top e(x'))$. This is the product rule, exact and assumption-free, with no fake symbols and no pretrain–finetune mismatch, and being a genuine language model it inherits every architectural advance. But $h_\theta(x_{1:t-1})$ only ever sees the left context, so it is structurally one-eyed; gluing a separately trained backward model on top, as ELMo does, gives two independent one-eyed models taped together at the very top, not deep joint reasoning over both sides. The denoising autoencoder (BERT) corrupts the input by replacing about 15% of tokens with a special $[\text{MASK}]$ symbol, calls the corrupted sequence $\hat x$, and reconstructs the masked set $\bar x$ via $\max_\theta \log p_\theta(\bar x \mid \hat x) \approx \sum_t m_t \log p_\theta(x_t \mid \hat x)$. Because reconstruction is not density estimation, each hidden state may attend to both sides, and that bidirectionality is exactly why it wins on understanding. But the $\approx$ is load-bearing: the true joint over masked tokens is replaced by a product, so all masked tokens are reconstructed independently given the unmasked ones. In $[\text{New}, \text{York}, \text{is}, \text{a}, \text{city}]$ with both "New" and "York" masked, it trains $p(\text{New}\mid \text{is},\text{a},\text{city})$ and $p(\text{York}\mid\text{is},\text{a},\text{city})$ but never $p(\text{York}\mid\text{New})$ — the dependency between the two things it is predicting is invisible to the loss. On top of that, $[\text{MASK}]$ is everywhere in pretraining and nowhere in real data, and softening this by occasionally substituting the true token must stay rare or the objective goes trivial. And it sits outside the density-estimation framework entirely.

So I want the left column of virtues — honest product rule, no independence assumption, no fake symbols, density estimation — together with bidirectionality. I propose XLNet, generalized autoregressive pretraining. The unidirectionality of AR is tied to one arbitrary choice: the forward order. The chain rule does not care about order; any permutation of the $T$ positions factorizes the same joint $p(x)$ exactly. So rather than commit to one order, I keep AR's exact product form and maximize the expected log-likelihood over a random factorization order. Letting $\mathcal{Z}_T$ be all $T!$ permutations of $(1,\dots,T)$, with $z_t$ the $t$-th element and $z_{<t}$ the prefix,
$$\max_\theta\ \mathbb{E}_{z\sim\mathcal{Z}_T}\!\left[\sum_{t=1}^T \log p_\theta(x_{z_t}\mid x_{z_{<t}})\right].$$
Every term is still a single AR conditional — product rule, no independence assumption, no $[\text{MASK}]$. With one shared $\theta$ across all orders, in expectation the token at any position conditions on every other token, sometimes those before it in the order and sometimes those after, so it gathers context from both sides. The between-target dependency BERT drops is recovered: whenever "New" precedes "York" in the sampled order, the term $\log p(\text{York}\mid\dots,\text{New})$ is literally in the sum. Crucially, "permutation" means permuting the factorization order, not shuffling the tokens. If I physically scrambled the input I would destroy word order and reintroduce a mismatch, since finetuning only ever sees natural text. Instead the sequence stays in its natural order, positional encodings stay tied to the original positions, and the permutation is realized purely as an attention mask: under order $z$, the slot predicted at step $t$ may attend only to positions $z_{<t}$. This is exactly why a masked self-attention Transformer is the natural backbone and an RNN is not — an RNN welds context to the left-to-right order, whereas an attention mask decouples which positions a query may see from the physical layout. Mechanically this resembles the orderless density estimators MADE and orderless NADE, which also share one model across sampled orders by masking, but their goal is the opposite: they want an order-agnostic, position-unaware estimator that for an MLP slides toward bag-of-words. I want the reverse — permute the factorization order while staying fully order-aware about positions — which is why the positional encodings stay on the original positions and only the mask changes.

Trying to implement the conditional with a standard Transformer exposes the first wall. The ordinary move is to softmax against a single hidden state, $p_\theta(X_{z_t}=x\mid x_{z_{<t}}) \propto \exp(e(x)^\top h_\theta(x_{z_{<t}}))$, but $h_\theta$ depends only on the context and not on which position $z_t$ is about to be predicted. Take two permutations sharing a prefix $z_{<t}$ but with $z_t^{(1)}=i\neq j=z_t^{(2)}$: the context is identical, so $h_\theta(x_{z_{<t}})$ is the same vector, and the model emits the identical distribution for positions $i$ and $j$ even though the true word distributions there differ. The standard softmax is position-blind about its target, and under permuted orders that blindness is fatal. The fix is to make the representation aware of which position it predicts,
$$p_\theta(X_{z_t}=x\mid x_{z_{<t}}) = \frac{\exp\!\big(e(x)^\top g_\theta(x_{z_{<t}}, z_t)\big)}{\sum_{x'}\exp\!\big(e(x')^\top g_\theta(x_{z_{<t}}, z_t)\big)},$$
where $g_\theta(x_{z_{<t}}, z_t)$ takes the target position $z_t$ as an input. Now positions $i$ and $j$ attend from different positions and get different $g$'s.

Building $g$ exposes the deeper contradiction. To predict $x_{z_t}$, the representation $g_\theta(x_{z_{<t}}, z_t)$ must use the position $z_t$ but absolutely must not use the content $x_{z_t}$ — if it could see the token sitting there, prediction would be trivial copying. Yet that same position $z_t$ will later serve as context for predicting some $x_{z_j}$ with $j>t$, and to be useful context it had better encode its own content. One vector per position cannot both exclude and include its own content. So I keep two streams, sharing all parameters. The content stream $h_{z_t}$ attends to $x_{z_{\le t}}$, including its own token — an ordinary self-attention hidden state, initialized $h_i^{(0)}=e(x_i)$. The query stream $g_{z_t}$ attends to the context $x_{z_{<t}}$ and to the position $z_t$ but not the content $x_{z_t}$, initialized $g_i^{(0)}=w$, a single shared trainable "slot to be predicted" vector identical at every position; its target-position identity comes from attending from position $z_t$ through the relative encodings, so the seed need not carry it. Per layer, both streams draw keys and values from the content representations $h$ — that is what carries token information — and differ only in which query asks and in the attention range:
$$g_{z_t}^{(m)} \leftarrow \mathrm{Attention}\big(Q=g_{z_t}^{(m-1)},\, KV=h_{z_{<t}}^{(m-1)}\big),\qquad h_{z_t}^{(m)} \leftarrow \mathrm{Attention}\big(Q=h_{z_t}^{(m-1)},\, KV=h_{z_{\le t}}^{(m-1)}\big),$$
with $z_{<t}$ strictly-before for the query stream so it never sees its own content, and $z_{\le t}$ up-to-and-including for the content stream so it does. Prediction reads off the top query state, $p_\theta(X_{z_t}=x\mid x_{z_{<t}}) \propto \exp(e(x)^\top g_{z_t}^{(M)})$. Because the content stream's update is exactly standard self-attention, at finetuning I simply drop the query stream and use the content stream as an ordinary Transformer — the two-stream apparatus leaves no residue and introduces no new mismatch.

The full objective converges slowly: it predicts even the first position from nothing and the second from one token, a punishing high-variance problem for one shared $\theta$ over all orders. The informative predictions are the late ones, where the conditioning prefix is long. So I split $z$ at a cutpoint $c$ into a non-target prefix $z_{\le c}$ and a target tail $z_{>c}$ and predict only the tail,
$$\max_\theta\ \mathbb{E}_{z\sim\mathcal{Z}_T}\!\left[\sum_{t=c+1}^{|z|} \log p_\theta(x_{z_t}\mid x_{z_{<t}})\right],$$
choosing $c$ so that roughly $1/K$ of the tokens are predicted ($K\approx 6$). The non-targets need only their content stream as context, so a compact target mapping materializes query slots only for the predicted positions, saving compute. Superficially this looks like masking, but the difference survives: writing the targets as $T$ and non-targets as $N=x\setminus T$, BERT optimizes $\sum_{x\in T}\log p(x\mid N)$, while XLNet optimizes $\sum_{x\in T}\log p(x\mid N\cup T_{<x})$, where $T_{<x}$ is the set of targets preceding $x$ in the order. For any dependency $(x,U)$ with $U\subseteq N$ both capture it, but whenever $U$ reaches into the earlier targets, only XLNet captures it — a strictly denser signal from the same target set. Against a plain forward AR model the same lens shows its limit: it can only cover $U$ among tokens before $x$ in the original order, never $(\text{New}, \{\text{York}\})$, which XLNet covers in expectation over orders.

Finally I fold in the Transformer-XL machinery, since the objective is already in the AR framework. Relative positional encoding is the reusable way to keep the natural-order geometry under mask-only permutation: the attention logit between a query at actual position $i$ and a key at $j$ depends on position only through the relative distance, $A_{ij} = (q_i+u)^\top k_j + (q_i+v)^\top W_R\, r_{i-j}$, a content-to-content term plus a content-to-relative-position term with global learnable biases $u, v$. Distances use the actual sequence indices $i-j$ (equivalently $z_t-z_s$ in factorization-order notation), never the sampled-order ranks, so the true geometry stays visible regardless of order; this is applied in both streams and is exactly what keeps me order-aware where the orderless estimators are not. Segment recurrence then extends context: process the previous segment, cache its per-layer content states $\tilde h^{(m)}$ as read-only memory, and let the current segment's streams attend to $[\tilde h^{(m-1)}, h_{z_{\le t}}^{(m-1)}]$ (and $z_{<t}$ for the query stream). Because positions are relative, the cached states are just vectors at known actual positions, so the update does not depend on the previous segment's factorization order $\tilde z$ at all — the memory is reusable freely, and in expectation the model uses it averaged over all orders of the previous segment. For multi-segment finetuning inputs $[\text{CLS}, A, \text{SEP}, B, \text{SEP}]$ I avoid an absolute segment embedding (inconsistent with the relative philosophy and incapable of going past two segments) and instead encode segments relatively: for a query at $i$ attending a key at $j$ I ask only whether they share a segment, using $s_+$ if so and $s_-$ if not, adding $(q_i+b)^\top s_{ij}$ to the logit. Only the same-segment-or-not relation matters, which generalizes and supports more than two segments; since the token-level objective already couples segments through attention, no separate next-sentence-prediction loss is added. The complete per-layer computation, with multi-head relative attention, residual, layer norm, and position-wise FFN for both streams, is
$$\hat h_{z_t}^{(m)} = \mathrm{LayerNorm}\big(h_{z_t}^{(m-1)} + \mathrm{RelAttn}(h_{z_t}^{(m-1)}, [\tilde h^{(m-1)}, h_{z_{\le t}}^{(m-1)}])\big),\quad h_{z_t}^{(m)} = \mathrm{LayerNorm}\big(\hat h_{z_t}^{(m)} + \mathrm{PosFF}(\hat h_{z_t}^{(m)})\big),$$
$$\hat g_{z_t}^{(m)} = \mathrm{LayerNorm}\big(g_{z_t}^{(m-1)} + \mathrm{RelAttn}(g_{z_t}^{(m-1)}, [\tilde h^{(m-1)}, h_{z_{<t}}^{(m-1)}])\big),\quad g_{z_t}^{(m)} = \mathrm{LayerNorm}\big(\hat g_{z_t}^{(m)} + \mathrm{PosFF}(\hat g_{z_t}^{(m)})\big),$$
the only difference between the two RelAttn calls being the key/value range. In implementation I never recompute attention per position: I compile the whole permutation into a single static mask over the natural-order sequence, give the two streams two slightly different masks (the query-stream mask `perm_mask` excludes self; the content-stream mask is the same with the diagonal re-allowed so a position sees itself), and let `target_mask` gate the loss. The relative attention is the Transformer-XL `ac + bd + ef` decomposition — content, relative-position, and relative-segment terms — and the loss is the tied-embedding cross-entropy of the target tokens against the top query states, averaged over the predicted tail.

```python
import tensorflow as tf

def embedding_lookup(x, n_token, d_embed, initializer):
    table = tf.get_variable('lookup_table', [n_token, d_embed], initializer=initializer)
    return tf.nn.embedding_lookup(table, x), table

def head_projection(h, d_model, n_head, d_head, initializer, name):
    w = tf.get_variable('{}/kernel'.format(name), [d_model, n_head, d_head],
                        dtype=h.dtype, initializer=initializer)
    return tf.einsum('ibh,hnd->ibnd', h, w)

def post_attention(h, attn_vec, d_model, n_head, d_head, dropout,
                   is_training, initializer, residual=True):
    proj_o = tf.get_variable('o/kernel', [d_model, n_head, d_head],
                             dtype=h.dtype, initializer=initializer)
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

def positionwise_ffn(inp, d_model, d_inner, dropout, kernel_initializer,
                     activation, is_training, reuse=None):
    with tf.variable_scope('ff', reuse=reuse):
        output = tf.layers.dense(inp, d_inner, activation=activation,
                                 kernel_initializer=kernel_initializer, name='layer_1')
        output = tf.layers.dropout(output, dropout, training=is_training, name='drop_1')
        output = tf.layers.dense(output, d_model,
                                 kernel_initializer=kernel_initializer, name='layer_2')
        output = tf.layers.dropout(output, dropout, training=is_training, name='drop_2')
        return tf.contrib.layers.layer_norm(output + inp, begin_norm_axis=-1, scope='LayerNorm')

def cache_mem(curr_out, prev_mem, mem_len, reuse_len=None):
    if mem_len is None or mem_len == 0:
        return None
    if reuse_len is not None and reuse_len > 0:
        curr_out = curr_out[:reuse_len]
    if prev_mem is None:
        new_mem = curr_out[-mem_len:]
    else:
        new_mem = tf.concat([prev_mem, curr_out], 0)[-mem_len:]
    return tf.stop_gradient(new_mem)

def build_pretraining_inputs(inputs, targets, is_selected, perm_size, seq_len,
                             sep_id, cls_id, num_predict=None):
    index = tf.range(seq_len, dtype=tf.int64)
    index = tf.transpose(tf.reshape(index, [-1, perm_size]))
    index = tf.random_shuffle(index)
    index = tf.reshape(tf.transpose(index), [-1])

    non_func = tf.logical_not(tf.logical_or(tf.equal(inputs, sep_id), tf.equal(inputs, cls_id)))
    non_target = tf.logical_and(tf.logical_not(is_selected), non_func)
    target_or_func = tf.logical_not(non_target)

    rev_index = tf.where(non_target, -tf.ones([seq_len], tf.int64), index)
    target_tokens = tf.logical_and(target_or_func, non_func)
    target_mask = tf.cast(target_tokens, tf.float32)

    self_rev_index = tf.where(target_tokens, rev_index, rev_index + 1)
    perm_mask = tf.logical_and(self_rev_index[:, None] <= rev_index[None, :], target_or_func)
    perm_mask = tf.cast(perm_mask, tf.float32)

    new_targets = tf.concat([inputs[0:1], targets[:-1]], axis=0)
    features = {
        'perm_mask': perm_mask,
        'target': new_targets,
        'target_mask': target_mask,
        'input_k': inputs,
        'input_q': target_mask,
    }

    if num_predict is not None:
        indices = tf.boolean_mask(tf.range(seq_len, dtype=tf.int64), tf.cast(target_mask, tf.bool))
        actual = tf.shape(indices)[0]
        pad_len = num_predict - actual
        target_mapping = tf.one_hot(indices, seq_len, dtype=tf.float32)
        target_mapping = tf.concat([target_mapping, tf.zeros([pad_len, seq_len])], axis=0)
        mapped_target = tf.boolean_mask(new_targets, tf.cast(target_mask, tf.bool))
        mapped_target = tf.concat([mapped_target, tf.zeros([pad_len], dtype=mapped_target.dtype)], axis=0)
        mapped_mask = tf.concat([tf.ones([actual], tf.float32), tf.zeros([pad_len], tf.float32)], axis=0)
        features.update({'target_mapping': target_mapping, 'target': mapped_target, 'target_mask': mapped_mask})
    return features

def objective_attention_layer(h, g, r, mems, r_w_bias, r_r_bias, seg_mat, r_s_bias, seg_embed,
                              attn_mask_h, attn_mask_g, target_mapping,
                              d_model, n_head, d_head, dropout, dropatt,
                              is_training, kernel_initializer, scope='rel_attn'):
    scale = 1 / (d_head ** 0.5)
    with tf.variable_scope(scope, reuse=False):
        cat = tf.concat([mems, h], 0) if mems is not None and mems.shape.ndims > 1 else h
        k_head_h = head_projection(cat, d_model, n_head, d_head, kernel_initializer, 'k')   # shared K from content
        v_head_h = head_projection(cat, d_model, n_head, d_head, kernel_initializer, 'v')   # shared V from content
        k_head_r = head_projection(r,   d_model, n_head, d_head, kernel_initializer, 'r')   # relative-position keys

        # content stream: query from h, mask allows self (z_<=t)
        q_head_h = head_projection(h, d_model, n_head, d_head, kernel_initializer, 'q')
        attn_vec_h = rel_attn_core(q_head_h, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                                   r_w_bias, r_r_bias, r_s_bias, attn_mask_h, dropatt, is_training, scale)
        output_h = post_attention(h, attn_vec_h, d_model, n_head, d_head, dropout, is_training, kernel_initializer)

    with tf.variable_scope(scope, reuse=True):
        # query stream: query from g, mask forbids self (z_<t)
        q_head_g = head_projection(g, d_model, n_head, d_head, kernel_initializer, 'q')
        if target_mapping is not None:
            q_head_g = tf.einsum('mbnd,mlb->lbnd', q_head_g, target_mapping)
            attn_vec_g = rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                                       r_w_bias, r_r_bias, r_s_bias, attn_mask_g, dropatt, is_training, scale)
            attn_vec_g = tf.einsum('lbnd,mlb->mbnd', attn_vec_g, target_mapping)
        else:
            attn_vec_g = rel_attn_core(q_head_g, k_head_h, v_head_h, k_head_r, seg_embed, seg_mat,
                                       r_w_bias, r_r_bias, r_s_bias, attn_mask_g, dropatt, is_training, scale)
        output_g = post_attention(g, attn_vec_g, d_model, n_head, d_head, dropout, is_training, kernel_initializer)
    return output_h, output_g

def pretraining_loss(hidden, target, target_mask, lookup_table, n_token, d_model,
                     initializer, use_tpu=False):
    softmax_w = lookup_table
    softmax_b = tf.get_variable('bias', [n_token], dtype=hidden.dtype,
                                initializer=tf.zeros_initializer())
    logits = tf.einsum('ibd,nd->ibn', hidden, softmax_w) + softmax_b
    if use_tpu:
        one_hot_target = tf.one_hot(target, n_token, dtype=logits.dtype)
        loss = -tf.reduce_sum(tf.nn.log_softmax(logits) * one_hot_target, -1)
    else:
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=logits)
    return tf.reduce_sum(loss * target_mask) / tf.reduce_sum(target_mask)
```
