The problem is how to make one language system handle many different language tasks without collecting fresh labels or changing the model for each task. The standard recipe is still narrow: pick a task, gather supervised examples, and train or fine-tune a model. Transfer learning helps by pre-training on unlabeled text, but it usually ends with task-specific fine-tuning, task-specific heads, and task-specific input formatting. That means the system is not really one model doing many things; it is one initialization followed by many small specialized systems. The alternative is to treat the task description itself as part of the input text, so that p(output | input, task) becomes an ordinary next-token prediction problem. The obstacle is that this only works if the training corpus actually contains a wide variety of naturally occurring demonstrations, the tokenizer can represent any benchmark string without lossy preprocessing, and the model is large and stable enough to learn those patterns.

Existing approaches fall short in three ways. First, single-domain corpora such as books or Wikipedia produce strong language models for one style of text but contain too few natural examples of translation, summarization, question answering, or dialogue. Raw web scrapes have the variety but are noisy and duplicated. Second, standard subword tokenizers rely on a fixed vocabulary and an unknown-token escape, which makes them fragile on rare strings and benchmarks. Byte-level modeling covers everything but wastes capacity on spelling and encoding details. Third, the original Transformer block places layer normalization after the residual addition, which makes very deep stacks unstable because the residual stream is normalized at every step instead of staying an additive path.

The method is GPT-2. It trains a single left-to-right Transformer language model on a broad web corpus and evaluates it zero-shot by prompting or scoring continuations, with no parameter updates and no task-specific heads.

The data pipeline starts from outbound links posted to Reddit with at least three karma, using that as a cheap human quality filter. Article text is extracted, deduplicated, and heuristically cleaned. Documents linked after December 2017 are excluded to create a temporal cutoff, and Wikipedia documents are removed to reduce benchmark contamination. The resulting corpus contains slightly over eight million documents and about forty gigabytes of text. This is not a task dataset; it is a broad field of naturally occurring demonstrations that embed many induced tasks.

The tokenizer is byte-level byte-pair encoding. UTF-8 bytes are mapped to reversible Unicode characters, which keeps the base alphabet at 256 symbols instead of the full Unicode code-point set. BPE merges are then applied after splitting the text by character category, so letters merge with letters, numbers with numbers, punctuation with punctuation, and whitespace follows its own pattern. A leading space is allowed to merge with a word because English word boundaries are informative. This yields a vocabulary of 50,257 tokens with no unknown token and no lossy normalization, so the model can score any benchmark string directly.

The model is a decoder-only Transformer with causal masked self-attention. The residual block is changed so that layer normalization sits before the attention and feed-forward sub-blocks, and the residual addition happens after each sub-block. A final layer normalization is added at the top of the stack before the output projection. The residual branch weights are scaled at initialization by one over the square root of the number of residual layers, so that the accumulated residual signal stays controlled as depth grows. Learned token and position embeddings are used, and the output projection is tied to the token embedding matrix. The context length is 1024 tokens and the batch size is 512. A family of models is trained at log-spaced sizes: 117M, 345M, 762M, and 1542M parameters. Only the learning rate is tuned per size on held-out corpus text, not on downstream tasks, so the zero-shot claim remains valid.

Evaluation is done by conditioning on natural text. For language-modeling benchmarks the model simply computes the likelihood of the benchmark text. For cloze or multiple-choice tasks it scores candidate continuations. For generation tasks it decodes from a prompt such as a question followed by an answer marker. Because the tokenizer is reversible, no dataset-specific preprocessing is needed. N-gram overlap checks are used to monitor memorization and interpret the metrics honestly.

The tokenizer is a direct instantiation of the byte-to-unicode map together with the category-splitting regex used to segment text before merging:

```python
def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(ord("¡"), ord("¬") + 1))
    bs += list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

pat = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
```

The residual stack itself is the block and model functions, built with the same norm, attn, and mlp helpers used throughout the stack: each block normalizes before attention and before the MLP, adds both branches back onto the residual stream, and the final call to norm together with the tied wte matmul turns the last hidden state into logits over the vocabulary:

```python
def attention_mask(nd, ns, *, dtype):
    i = tf.range(nd)[:, None]
    j = tf.range(ns)
    m = i >= j - ns + nd
    return tf.cast(m, dtype)

def mask_attn_weights(w):
    _, _, nd, ns = shape_list(w)
    b = attention_mask(nd, ns, dtype=w.dtype)
    b = tf.reshape(b, [1, 1, nd, ns])
    return w * b - tf.cast(1e10, w.dtype) * (1 - b)

def multihead_attn(q, k, v):
    w = tf.matmul(q, k, transpose_b=True)
    w = w * tf.rsqrt(tf.cast(v.shape[-1].value, w.dtype))
    w = mask_attn_weights(w)
    w = softmax(w)
    return tf.matmul(w, v)

def block(x, scope, *, past, hparams):
    with tf.variable_scope(scope):
        nx = x.shape[-1].value
        a, present = attn(norm(x, "ln_1"), "attn", nx, past=past, hparams=hparams)
        x = x + a
        m = mlp(norm(x, "ln_2"), "mlp", nx * 4, hparams=hparams)
        x = x + m
        return x, present

def model(hparams, X, past=None, scope="model", reuse=False):
    with tf.variable_scope(scope, reuse=reuse):
        results = {}
        batch, sequence = shape_list(X)
        wpe = tf.get_variable(
            "wpe", [hparams.n_ctx, hparams.n_embd],
            initializer=tf.random_normal_initializer(stddev=0.01))
        wte = tf.get_variable(
            "wte", [hparams.n_vocab, hparams.n_embd],
            initializer=tf.random_normal_initializer(stddev=0.02))
        past_length = 0 if past is None else tf.shape(past)[-2]
        h = tf.gather(wte, X) + tf.gather(wpe, positions_for(X, past_length))

        presents = []
        pasts = tf.unstack(past, axis=1) if past is not None else [None] * hparams.n_layer
        for layer, past in enumerate(pasts):
            h, present = block(h, "h%d" % layer, past=past, hparams=hparams)
            presents.append(present)

        results["present"] = tf.stack(presents, axis=1)
        h = norm(h, "ln_f")
        h_flat = tf.reshape(h, [batch * sequence, hparams.n_embd])
        logits = tf.matmul(h_flat, wte, transpose_b=True)
        results["logits"] = tf.reshape(logits, [batch, sequence, hparams.n_vocab])
        return results
```
