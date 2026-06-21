# Context: fast autoregressive decoding with attention-based sequence models (circa 2019)

## Research question

Attention-based sequence models train fast because every position in a sequence can be processed
in parallel, but they *generate* slowly. In autoregressive generation each new token depends on the
token before it, so the positions must be produced one at a time, and at every step the attention
layers must re-read the entire memory of what came before. The precise question is: on modern
accelerators, where arithmetic throughput is roughly two orders of magnitude higher than memory
bandwidth, what is actually limiting the speed of incremental (one-token-at-a-time) attention
decoding, and can the attention layer be restructured so that the same amount of arithmetic is done
against far less memory traffic?

## Background

The dominant sequence model is the attention-based encoder-decoder, which has largely displaced
recurrent models. Its core primitive is **neural attention** (Bahdanau, Cho & Bengio, 2014): given a
query vector `q` and a set of `m` (key, value) pairs collected into matrices `K` (shape `[m, k]`) and
`V` (shape `[m, v]`), attention returns a weighted sum of the values, with weights obtained by
comparing the query to each key. The standard instantiation is **scaled dot-product attention**:
`logits = q · Kᵀ`, `weights = softmax(logits)`, `y = weights · V`. Attention is a content-based
soft lookup: the query asks a question, the keys advertise what each memory position offers, and the
values carry the payload that gets mixed.

On top of this, the prevailing architecture runs **`h` attention layers ("heads") in parallel**
(Vaswani et al., 2017). The query, keys, and values for each head come from `h` different learned
linear projections of the inputs (`P_q`, `P_k`, `P_v`, each of shape `[h, d, ·]`), the `h` head
outputs are individually projected by `P_o` and summed. For a model of width `d` the standard
choice is `k = v = d/h` (e.g. `d = 512`, `h = 8`, `d_k = d_v = 64`), so that the total computational
cost is similar to a single full-width attention. Running heads in parallel lets the model attend to
information from different representation subspaces at once; this is widely understood as the source
of multi-head attention's representational power over a single head — many independent read patterns
applied to the same memory.

A second piece of background is the **hardware reality**: on GPUs/TPUs the peak rate of arithmetic
operations can be ~100× the rate at which operands can be streamed from memory. An operation is only
fast if the ratio of memory bytes accessed to arithmetic operations performed (its arithmetic
intensity's inverse) is small — i.e. each loaded byte is reused in many flops. When that ratio
approaches 1 the computation is memory-bandwidth bound: the arithmetic units sit idle waiting for
operands, and raw flops no longer predict speed.

The relevant diagnostic facts about the existing design space, all knowable from analyzing the
standard architecture:

- **Training-time (batched) attention is bandwidth-friendly.** Process all `n` query positions at
  once and the arithmetic is `Θ(b n d²)` while memory access is `O(b n d + b h n² + d²)`, giving a
  memory-to-arithmetic ratio of `O(1/k + 1/(b n))` — comfortably small. Each loaded key/value is
  reused across all `n` query positions, so bandwidth is amortized.
- **Incremental (one-position-at-a-time) decoding is not.** Generating token by token, across `n`
  calls the arithmetic is still `Θ(b n d²)`, but the memory access becomes `Θ(b n² d + n d²)`: the
  first term comes from reloading the cached `K` and `V` at every step, the second from reloading the
  projection matrices. The ratio is now `Θ(n/d + 1/b)`. When `n ≈ d` or the batch is small `b ≈ 1`,
  the ratio is close to 1 — incremental decoding is memory-bandwidth bound.
- **The two terms behave very differently.** The `1/b` term is easy to push down: just use a larger
  batch (memory permitting). The `n/d` term traces directly to the cached keys/values that must be
  reloaded each step. At full length, one of `K` or `V` has size
  `b · h · m · k = b · n · d` under `m = n`, `k = d/h`; summed over the `n` incremental calls, that
  becomes the cumulative reload term `Θ(b n² d)`.
- **KV-cache footprint scales with the number of KV heads.** The bytes the decoder must store and
  reload per generated token are `2 · (#layers) · n_kv_heads · d_head · bytes_per_element` — two for
  K and V, times one `[n_kv_heads, d_head]` matrix per layer. Standard multi-head attention has
  `n_kv_heads = h`.

## Baselines

These are the structures a faster attention layer would be measured against.

**Single-head dot-product attention (Bahdanau et al., 2014; the scaled form, Vaswani et al., 2017).**
One query projection, one key projection, one value projection. `y = softmax(q · Kᵀ / √d_k) · V`.
Simple and the cheapest K/V state.

**Multi-head attention (Vaswani et al., 2017).** Run `h` heads in parallel, each with its own
query/key/value projection of width `d/h`, then project and sum the head outputs. Concretely, with
`P_q, P_k, P_v` of shape `[h, d, ·]` and `P_o` of shape `[h, d, v]`:
`Q = X·P_q`, `K = M·P_k`, `V = M·P_v`, `logits = Q·Kᵀ`, `weights = softmax(logits)`,
`O = weights·V`, `Y = O·P_o`. This is the quality baseline. Every head materializes and caches its
own `K` and `V`; in incremental decoding the cached state and hence the per-step memory traffic scale
with `h`.

**Reducing the number of heads or the per-head dimension.** A direct way to shrink the cached K/V is
to use fewer heads `h`, or smaller key/value dimensions `d_k, d_v` (widening the feed-forward layers
to keep the total parameter count fixed).

**Limiting or compressing the attended positions.** An orthogonal family reduces `n` itself: restrict
each position to attend only to a local neighborhood (e.g. the previous 31 positions), or otherwise
compress the number of memory positions attended to (Liu et al., 2018; Zhang, Xiong & Su, 2018;
Povey et al., 2018). These change which and how many positions are attended to, and they address the
`n` factor in the cached-state cost.

## Evaluation settings

The natural yardsticks already in use for sequence models at this scale:

- **WMT 2014 English-German machine translation.** Encoder-decoder model, 6 layers,
  `d_model = 1024`, `d_ff = 4096`, `h = 8`, `d_k = d_v = 128`, learned positional embeddings,
  embedding/output weight sharing; ~211M parameters. Trained ~100k steps (~20 epochs), batch of 128
  examples of 256-token source and 256-token target sequences, on a TPUv3 cluster. Quality measured
  by BLEU (sacrebleu, `wmt13`/`wmt14`, `en-de`, intl tokenization), under both greedy and beam-search
  (beam 4, `α = 0.6`) decoding, plus per-subword-token dev perplexity.
- **Billion-Word Language Modeling Benchmark (Chelba et al., 2013).** Transformer-decoder LM, 6
  layers, `d_model = 1024`, `d_ff = 8192`, `h = 8`, `d_k = d_v = 128`; ~192M parameters. Trained
  ~136k steps (10 epochs) at 64k tokens/batch. Quality measured by per-word dev perplexity.
- **Speed protocol.** Training and inference timed on a single TPUv2 (8 cores). Training time
  reported per (input+target) token. Inference timed as incremental greedy generation on a batch of
  1024 sequences (128 per core), source length 128, target length 128, reporting amortized
  microseconds per token for encoder and per decoder step — to expose the memory-bandwidth bottleneck
  the analysis predicts.
- **Parameter-count control.** When an attention variant removes parameters, the feed-forward hidden
  width is widened to restore the baseline's total parameter count, so quality comparisons are at
  equal capacity.

## Code framework

The substrate is the standard attention layer as it already exists, written in einsum (named-index
tensor contraction) over the projection tensors. The pieces that exist before any redesign: the
dot-product attention kernel, the multi-head batched forward pass, and — the part that actually
matters for decoding speed — the *incremental* self-attention step that maintains a running cache of
keys and values and appends one position per call. The slot to be redesigned is the shape and
indexing of the key/value path inside these functions; the query path and output projection are
given.

```python
import tensorflow as tf  # einsum: named-index tensor contraction


def DotProductAttention(q, K, V):
    """One query against m (key, value) pairs.
    q: [k]   K: [m, k]   V: [m, v]  ->  y: [v]"""
    logits = tf.einsum("k,mk->m", q, K)
    weights = tf.softmax(logits)
    return tf.einsum("m,mv->v", weights, V)


def MultiheadAttentionBatched(X, M, mask, P_q, P_k, P_v, P_o):
    """Batched parallel attention.
    X: [b, n, d]   M: [b, m, d]   mask: [b, h, n, m]
    P_q: [h, d, k]   P_k: [h, d, k]   P_v: [h, d, v]   P_o: [h, d, v]  ->  Y: [b, n, d]"""
    Q = tf.einsum("bnd,hdk->bhnk", X, P_q)
    K = tf.einsum("bmd,hdk->bhmk", M, P_k)
    V = tf.einsum("bmd,hdv->bhmv", M, P_v)
    logits = tf.einsum("bhnk,bhmk->bhnm", Q, K)
    weights = tf.softmax(logits + mask)
    O = tf.einsum("bhnm,bhmv->bhnv", weights, V)
    Y = tf.einsum("bhnv,hdv->bnd", O, P_o)
    return Y


def SelfAttentionIncremental(x, prev_K, prev_V, P_q, P_k, P_v, P_o):
    """One decode step: form q for the new position, append its (k, v) to the cache, attend.
    x: [b, d]   prev_K / prev_V: the cached keys/values so far.
    P_q: [h, d, k]   P_o: [h, d, v]
    Returns y: [b, d] and the grown caches new_K, new_V.

    The query side and output projection are fixed below. The key/value projection
    and the SHAPE of the (K, V) cache that gets reloaded every step is the slot to design.
    """
    q = tf.einsum("bd,hdk->bhk", x, P_q)          # query keeps its h heads

    # TODO: project the new position's key/value and append to the running cache.
    #       The cache shape decided here is exactly what gets reloaded each decode
    #       step, so it sets the per-step memory traffic.
    new_K = None  # TODO
    new_V = None  # TODO

    # attend with the (possibly newly shaped) cache, then mix heads and project out
    # logits = <q against new_K>
    # weights = tf.softmax(logits)
    # o = <weights against new_V>
    y = tf.einsum("bhv,hdv->bd", None, P_o)        # output projection over h head outputs
    return y, new_K, new_V
```

The query projection (`h` heads) and the output projection `P_o` are settled; what remains open is
the key/value projection and, crucially, the shape of the `(K, V)` cache that the incremental step
reloads on every call.
