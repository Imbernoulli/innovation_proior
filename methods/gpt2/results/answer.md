# GPT-2: language models as unsupervised multitask learners

## Core Claim

A task conditional can be represented as ordinary language-model conditioning when the task, input, and output are all serialized as tokens:

```text
p(x) = product_{i=1}^n p(s_i | s_1, ..., s_{i-1})
p(output | input, task) = p(next output tokens | preceding task and input tokens)
```

The supervised task loss is the next-token loss restricted to the output span. At the ideal optimum, a language model over text that contains task demonstrations also solves those induced task conditionals. The practical recipe is to provide enough diverse demonstrations in natural text and enough model capacity to learn them.

## Training Recipe

Use a large, high-quality, diverse web corpus:

- collect outbound Reddit links with at least 3 karma as a cheap human quality filter;
- extract article text with Dragnet and Newspaper;
- deduplicate and heuristically clean;
- use the preliminary cutoff with no links after December 2017;
- remove Wikipedia documents to reduce benchmark contamination;
- train on the resulting slightly-over-8M-document, about-40GB text corpus.

Use byte-level BPE:

- map UTF-8 bytes to reversible Unicode characters, so every string is representable;
- start from 256 byte symbols rather than a full Unicode-code-point base vocabulary;
- apply BPE after regex splitting by character category, with a leading-space exception;
- use a 50,257-token vocabulary and no unknown token.

Use a decoder-only Transformer language model:

- causal masked self-attention with score scale `1/sqrt(d_k)`;
- pre-layer-normalized residual blocks;
- a final layer normalization after the last block;
- learned token and position embeddings, with logits tied to the token embedding matrix;
- context length 1024 and batch size 512;
- residual-layer weights scaled at initialization by `1/sqrt(N)`, where `N` is the number of residual layers in the stack.

The model family is log-spaced:

```text
parameters  layers  d_model
117M        12      768
345M        24      1024
762M        36      1280
1542M       48      1600
```

## OpenAI Source Core

The released TensorFlow source uses the following block order and output path. The residual-depth initialization scale is a training-design note; the released graph code exposes ordinary `conv1d(..., w_init_stdev=0.02)` initializers and checkpoint-loaded variables, so the graph below should not pretend to apply that scaling.

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

The small released hparams file sets:

```json
{
  "n_vocab": 50257,
  "n_ctx": 1024,
  "n_embd": 768,
  "n_head": 12,
  "n_layer": 12
}
```

## Tokenizer Core

The byte mapping is reversible and avoids whitespace/control bytes that would break the BPE string machinery:

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
```

The category split that keeps BPE merges from crossing most category boundaries is:

```python
pat = re.compile(
    r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)
```

Encoding applies the regex first, converts each matched token to UTF-8 bytes, maps each byte through `bytes_to_unicode()`, and then applies BPE merges inside that matched category span. Decoding reverses BPE tokens to byte-mapped characters, maps those characters back to bytes, and decodes UTF-8.
