# Context: autoregressive generation with attention-based decoders (circa 2017-2019)

## Research question

An attention-based sequence decoder generates one token at a time:
`p(y) = prod_{t=1}^{n} p(y_t | y_{<t}, x)`. To produce token `y_t` it must have already
produced `y_{t-1}`, so generation is intrinsically *sequential* in `t` — there is no way to
compute all output positions at once the way training does. The precise problem is the cost
of that sequential loop. A self-attention layer computes, for a query at the current
position, a softmax-weighted read over the key/value pairs of every position so far; the most
direct way to advance one step is to run the whole decoder on the entire prefix
`[y_1, ..., y_t]` and take the last position's output. Done naively, this re-runs the layer
over the full prefix at *every* step, so the work to emit one token grows with how many
tokens have already been emitted, and the total work to emit `n` tokens grows much faster
than `n` itself. The goal is to make per-step generation cost reflect only the *new* work a
step actually introduces, without changing what the model computes, i.e. with the decoder's
output identical token-for-token to the naive re-run. What a solution must achieve: each step
does a bounded amount of fresh computation rather than redoing the whole prefix, the result
is numerically exactly the parallel-pass result, and the bookkeeping avoids rebuilding state
that is already determined. Doing this well also forces a clear-eyed account of *what* a step
then spends its time on, because the
hardware these decoders run on is much better at arithmetic than at moving data.

## Background

The field state is set by attention-based encoder-decoder and decoder-only sequence models.
The load-bearing concepts:

- **Neural attention (Bahdanau et al. 2014).** An attention function takes a query vector `q`
  and a set of `m` key/value pairs (matrices `K`, `V`) and returns a weighted sum of the
  values, the weights coming from a compatibility score between `q` and each key. Bahdanau's
  original score was additive (a small feed-forward net); the dominant alternative is the
  dot-product score, which is a single matrix multiply and so far faster and more
  memory-efficient on GPU/TPU. Attention connects any two positions in `O(1)` sequential
  operations, which is why it displaced sequence-aligned recurrence for cross-positional
  communication.

- **Scaled dot-product attention.** Packing queries, keys, values into matrices,
  `Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V`. The `1/sqrt(d_k)` is a variance
  correction: if the components of `q` and `k` are independent with mean 0 and variance 1,
  then `q . k = sum_{i=1}^{d_k} q_i k_i` has mean 0 and variance `d_k`, so unscaled logits
  grow like `sqrt(d_k)`; large logits push the softmax into a near-one-hot regime where its
  gradient is minuscule. Dividing by `sqrt(d_k)` keeps logits unit-scale.

- **Multi-head attention.** Instead of one attention over `d_model`-dimensional vectors, run
  `h` attentions in parallel over learned projections to `d_k = d_v = d_model/h` dimensions,
  concatenate, and project: `MultiHead(Q,K,V) = Concat(head_1,...,head_h) W^O`,
  `head_i = Attention(Q W^Q_i, K W^K_i, V W^V_i)`. With `d_k = d_v = d_model/h` the total cost
  matches single-head attention at full width (`h=8`, `d_k=d_v=64` for `d_model=512`). The
  query, key, and value of every position thus come from cheap linear projections of that
  position's layer input.

- **Causal (masked) self-attention and the autoregressive property.** In a decoder, position
  `i` may attend only to positions `j <= i`. This is enforced inside scaled dot-product
  attention by setting the logits of illegal connections (the strict upper triangle) to
  `-inf` before the softmax, so `exp(-inf) = 0` zeroes their weight. Combined with the
  one-position output offset, this guarantees the prediction at `i` depends only on outputs
  before `i` — the property that makes the stack a valid autoregressive model. The mask is
  therefore both a correctness condition for generation and the source of the triangular
  data-dependence pattern that any exact one-step implementation has to preserve.

- **The train/inference asymmetry of these decoders.** During training the entire target
  sequence is known (teacher forcing), so the masked self-attention layer is one big batched
  `softmax(QK^T/sqrt(d_k))V` over all `n` positions at once — fully parallel across the
  sequence, which is exactly why these models train fast on modern accelerators. At generation
  time that parallelism is gone: the layer's output at one position determines the token fed in
  at the next, so the steps cannot be run simultaneously. The diagnostic observation about the
  hardware is the other half of the background: on modern GPU/TPU, arithmetic throughput can be
  roughly two orders of magnitude higher than memory bandwidth, so whether a kernel is fast is
  governed less by its FLOP count than by how many bytes it must move per FLOP — the ratio of
  memory access to arithmetic. A useful rule of thumb in the parallel (training) regime: with
  `m = n`, `k = v = d/h`, and `n <= d`, the batched multi-head attention layer does
  `Theta(b n d^2)` arithmetic and touches `O(bnd + b h n^2 + d^2)` memory, a ratio of
  `O(1/k + 1/(bn))` that is comfortably small — the layer is compute-bound and the hardware is
  used well.

## Baselines

The prior approaches to running such a decoder's generation loop, and where each stalls.

**Re-run the full prefix every step (the naive decode loop).** The most literal way to
generate: keep the growing token list `[y_1, ..., y_t]`, and at each step feed the whole list
through the decoder, read the last position's logits, sample/`argmax` the next token, append,
repeat. It is obviously correct — it is the same forward pass training uses, just stopped at
the last position. Its cost is the problem. At step `t` the layer recomputes the projections
`Q, K, V` for all `t` positions (`Theta(b t d^2)` arithmetic per layer for batch size `b`)
and the all-pairs attention read (`Theta(b t^2 d)`). Summing over `t = 1..n` gives
`Theta(b n^2 d^2) + Theta(b n^3 d)` per layer. Under the standard `n <= d` regime, the
projection term dominates, so the naive decode loop's arithmetic is `Theta(b n^2 d^2)`.
**Gap:** the same prefix positions are processed again and again as the prefix grows. The
loop is correct, but its arithmetic grows because it repeatedly rebuilds the whole
lower-triangular computation just to read the newest row.

**Bound the sequence to a fixed window.** One way to stop the per-step cost from growing is to
cap how many past positions a step may look at — restrict attention to the current position
and a fixed number of recent ones, so each step's read is over a constant-size set rather than
the whole prefix. **Gap:** this changes *what the model computes* — positions outside the
window are simply invisible to the query, so any dependency longer than the window is dropped.
It trades the model's full-context behavior for a bounded loop; it is not a faithful execution
of the same decoder, so it cannot serve as the exact result the model would have produced.

**Compress or summarize the set of attended positions.** A related line reduces the number of
memory positions a query reads over by compressing the past into fewer representative
key/value slots (local neighborhoods, learned summaries, pooled memory). **Gap:** like
windowing, this is an approximation of the attention read — the query no longer attends over
the true set of past key/value pairs, so the output diverges from the model's exact
full-context output. It can be a good speed/quality trade, but it is not an exact execution of
the same decoder.

## Evaluation settings

The natural yardsticks for a decoder generation loop, all pre-existing:

- **Autoregressive sequence-to-sequence generation** on a translation benchmark such as WMT
  English-German, with an encoder-decoder Transformer (e.g. 6 layers, `d_model` order
  1024, `h = 8`); decoded with greedy maximum-likelihood and with beam search (beam 4),
  scored by BLEU (e.g. via sacrebleu) and per-subword-token perplexity on the dev set.
- **Decoder-only language modeling** on a large corpus (e.g. the Billion-Word benchmark),
  scored by per-word / per-token perplexity. A pure decoder-self-attention stack with no
  encoder, exercising exactly the masked-causal generation loop.
- **Hardware timing of the decode loop** as the operational metric the problem is really
  about: amortized time per generated token for the generation loop (and separately for
  the parallel encoder pass), measured under fixed batch size and sequence length, on a single
  accelerator; tracked alongside the memory footprint of whatever state the loop carries
  between steps. The settings fix the model size, the batch size, the source/target lengths,
  and the decoding strategy (greedy vs. beam); the loop's per-step latency and the size of its
  carried state are read off under those fixed settings.

## Code framework

The generation loop runs on top of an attention decoder that already exists: a stack of
layers, each with masked multi-head self-attention and the per-position projections
`W^Q, W^K, W^V, W^O`, and a final projection to vocabulary logits. The scaled-dot-product
attention primitive, the multi-head projections, the causal mask, and the embedding/output
layers are all in place. What is not settled is how a single-step forward should carry state
between iterations while remaining exactly equivalent to the full-prefix pass. The code below
exposes a generic per-layer state object and a container for those objects; its TODOs mark the
unsolved part without specifying what the remembered data must be.

```python
import torch


class LayerState:
    """State carried for one decoder layer. Its contents and update rule are
    the object to be designed."""

    def __init__(self):
        # TODO: choose the remembered tensors for this layer.
        pass

    def update(self, *new_parts):
        # TODO: define how this layer's remembered state advances.
        pass


class StepState:
    """Container for per-layer state carried by the generation loop."""

    def __init__(self):
        self.layers = []

    def layer(self, layer_idx):
        # TODO: choose how per-layer state objects are created and reused.
        pass

    def advance_attention_layer(self, layer, x_t, layer_idx):
        # TODO: compute this layer's one-step output and update the layer state.
        pass


def self_attention_step(layer, x_t, state, layer_idx):
    """One masked self-attention layer, advanced by one position.
    x_t: [b, 1, d] hidden state of the single new position.
    The attention primitive and projections exist; the state-dependent single-step
    implementation is the open slot."""
    return state.advance_attention_layer(layer, x_t, layer_idx)

def decode(model, prompt_ids, max_new_tokens):
    """Existing autoregressive loop: prefill on the prompt, then emit one token at a time."""
    state = StepState()
    # prefill: run the prompt through the stack, populating `state`
    hidden = model.embed(prompt_ids)
    for layer_idx, layer in enumerate(model.layers):
        hidden = layer.forward_prefill(hidden, state, layer_idx)   # fills state for the prompt
    logits = model.to_logits(hidden[:, -1:])
    next_id = logits.argmax(-1)
    out = [next_id]
    for _ in range(max_new_tokens - 1):
        x = model.embed(next_id)                         # [b, 1, d]
        for layer_idx, layer in enumerate(model.layers):
            x = x + self_attention_step(layer, x, state, layer_idx)   # uses carried state
            x = x + layer.ffn(x)
        next_id = model.to_logits(x).argmax(-1)
        out.append(next_id)
    return torch.cat(out, dim=-1)
```

The empty slot is the pair `LayerState` / `StepState`: what a layer remembers, how that memory
is updated, and how the loop finds the right layer state while still producing exactly the
full-prefix result.
